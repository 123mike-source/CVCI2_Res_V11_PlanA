"""Observed motion conditioning for the existing point reliability gate."""
import hashlib

import torch
from torch import nn


MOTION_GATE_VERSION = 7
MOTION_GATE_FORMAT = 'cvci2res_gate12_x0_motion_gate_v7'
DEFAULT_MOTION_GATE_CONFIG = {
    'schema': 'can_history20_v1', 'feature_dim': 20, 'projection_dim': 128,
    'velocity_scale_mps': 15., 'acceleration_scale_mps2': 5.,
    'yaw_rate_scale_radps': 1., 'feature_clip': 5.,
    'initialization': 'zero_projection',
}
INTEGRATED_MOTION_VERSION = 8
INTEGRATED_MOTION_FORMAT = 'cvci2res_gate12_x0_integrated_motion_world_v8'
INTEGRATED_MOTION_GATE_CONFIG = dict(DEFAULT_MOTION_GATE_CONFIG, injection='concat_existing_gate')

MOTION_MONITOR_KEYS = (
    'motion/input_valid_fraction', 'motion/gate_low_speed',
    'motion/gate_increasing_speed', 'motion/gate_moving',
    'motion/projection_norm',
)


def canonical_motion_gate_config(config):
    if config is None:
        return None
    if dict(config) not in (DEFAULT_MOTION_GATE_CONFIG, INTEGRATED_MOTION_GATE_CONFIG):
        raise ValueError('Unsupported observed motion gate feature contract')
    return dict(config)


def zero_motion_projection():
    # Construction must not advance the original initialization/training RNG.
    with torch.random.fork_rng(devices=[]):
        projection = nn.Linear(20, 128, bias=False)
        nn.init.zeros_(projection.weight)
    return projection


def observed_motion_features(statuses, metas, prior_ego_deltas, active):
    """Return [B,6,20], using only supplied current/past CAN and old predictions.

    CAN10: ax,ay,az,wx,wy,wz,vx,vy,vz,steering; history is oldest to newest.
    Feature order: vxy(6), axy(6), wz(3), d|v|/dt(2), prior-vxy gap(2), horizon(1).
    The sixth prior is the existing extrapolated prior, not a new future map.
    """
    if statuses.ndim != 3 or statuses.shape[1:] != (3, 10):
        raise ValueError('Motion gate requires original [B,3,10] CAN history')
    b = statuses.shape[0]
    if prior_ego_deltas.shape != (b, 6, 2) or active.shape != (b,):
        raise ValueError('Motion gate prior/active shapes differ')
    times = statuses.new_tensor([[float(m['timestamp']) for m in history]
                                for history in metas], dtype=torch.float64)
    if times.shape != (b, 3):
        raise ValueError('Motion history must have three timestamps')
    dt = (times[:, 1:] - times[:, :-1]).float()
    finite = torch.isfinite(statuses).all(dim=-1).all(dim=-1)
    finite = finite & torch.isfinite(prior_ego_deltas).all(dim=-1).all(dim=-1)
    valid_time = torch.isfinite(dt).all(-1) & (dt >= .399999).all(-1) & (dt <= .600001).all(-1)
    valid = active.bool() & finite & valid_time
    clean = torch.nan_to_num(statuses.float(), nan=0., posinf=0., neginf=0.)
    prior = torch.nan_to_num(prior_ego_deltas.float(), nan=0., posinf=0., neginf=0.)
    velocity = clean[..., 6:8]
    acceleration = clean[..., :2]
    speed = velocity.norm(dim=-1)
    safe_dt = torch.nan_to_num(dt, nan=.5, posinf=.5, neginf=.5).clamp(.399999, .600001)
    trend = (speed[:, 1:] - speed[:, :-1]) / safe_dt
    shared = torch.cat((velocity.reshape(b, 6) / 15., acceleration.reshape(b, 6) / 5.,
                        clean[..., 5].reshape(b, 3), trend / 5.), dim=-1)
    prior_gap = (prior / .5 - velocity[:, -1:, :]) / 15.
    horizon = torch.arange(1, 7, device=statuses.device, dtype=torch.float32)[None, :, None] / 6.
    features = torch.cat((shared[:, None, :].expand(-1, 6, -1), prior_gap,
                          horizon.expand(b, -1, -1)), dim=-1).clamp(-5., 5.)
    features = features.masked_fill(~valid[:, None, None], 0.).detach()
    return features, valid


def warm_start_v6(model, path):
    """Migrate all A9 tensors; leave only explicit new projections at zero."""
    from temporal_matched_deform_model import (MATCHED_DEFORM_FORMAT, MATCHED_DEFORM_SCOPE,
                                               is_trainable_state)
    saved = torch.load(str(path), map_location='cpu')
    meta = saved.get('metadata', {})
    if saved.get('format') != MATCHED_DEFORM_FORMAT or meta.get('architecture_version') != 6:
        raise ValueError('Warm start requires a complete v6 joint checkpoint')
    if meta.get('training_scope') != MATCHED_DEFORM_SCOPE:
        raise ValueError('Warm start training scope differs')
    for key in ('fusion_config', 'sampler_config'):
        if meta.get(key) != getattr(model, key):
            raise ValueError('Warm start configuration differs: ' + key)
    state = saved['trainable']
    current = model.state_dict()
    expected = {n for n in current if is_trainable_state(n)}
    fresh = {n for n in expected if n.endswith('.future_motion_projection.weight')}
    count = 6 if model.motion_gate_config is not None else 0
    if len(fresh) != count or set(state) != expected - fresh:
        raise RuntimeError('Warm start permits only the six explicit new projection tensors')
    model.load_state_dict(state, strict=False)
    for name, original in state.items():
        if not torch.equal(current[name].detach().cpu(), original):
            raise RuntimeError('Warm start tensor not restored: ' + name)
    if any(torch.count_nonzero(current[n]).item() for n in fresh):
        raise RuntimeError('Motion projections must be exactly zero at initialization')
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'status': 'PASS', 'source': str(path), 'source_sha256': digest.hexdigest(),
            'source_epoch': saved.get('epoch'), 'restored_tensor_count': len(state),
            'new_zero_tensors': sorted(fresh), 'new_parameter_count': sum(current[n].numel() for n in fresh),
            'optimizer_restored': False, 'scheduler_restored': False,
            'rolling_history': 'empty', 'architecture_version': MOTION_GATE_VERSION if fresh else 6}
