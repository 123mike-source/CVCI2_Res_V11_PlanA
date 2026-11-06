"""V10: use supervised intermediate trajectories to select existing BEV evidence."""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from matched_diffusion import MatchedDiTLayer, _stat

CANDIDATE_VERSION = 10
CANDIDATE_FORMAT = 'cvci2res_geobev_candidate_v10'
CANDIDATE_SCOPE = 'geobev_candidate_readout_partial_irwm'
GEOMETRY_FIELDS = {
    'token_positions_ego': (None, 2),
    'token_axes_ego': (None, 2, 2),
    'token_radii_m': (None, 2),
    'window_centers_ego': (5, 2),
    'window_axes_ego': (5, 2, 2),
    'window_radii_m': (5, 2),
}
GATE_EXTRA_DIM = 6


def config_for_mode(mode):
    if mode == 'legacy':
        return None
    if mode not in ('dynamic', 'current', 'full'):
        raise ValueError('Unknown candidate interaction mode')
    return dict(schema='supervised_candidate_readout_v1', mode=mode,
                dynamic_positions=True, current_points=5 if mode != 'dynamic' else 1,
                current_radius_m=1., current_center_mass=.8,
                future_geometry=mode == 'full', future_gate_context=mode == 'full',
                distance_initial=.01, distance_max=1., distance_squared_cap=9.,
                position_gradient='detached', world_condition='retained_cross_frame_prior',
                world_rollouts_per_forward=1)


def canonical_config(config):
    if config is None:
        return None
    expected = config_for_mode(config.get('mode'))
    if config != expected:
        raise ValueError('Unsupported candidate interaction contract')
    return copy.deepcopy(expected)


def monitor_keys():
    return ['candidate/current_center_weight', 'candidate/current_valid_fraction',
            'candidate/position_change_m', 'candidate/window_supported_fraction',
            'candidate/window_distance', 'candidate/future_distance_strength',
            'candidate/distance_clipped_fraction', 'candidate/distance_bias_span',
            'candidate/current_content_score_abs', 'candidate/current_bias_span']


def empty_geometry(batch, tokens, device):
    return {key: torch.zeros((batch, *(tokens if n is None else n for n in shape)),
                             device=device, dtype=torch.float32)
            for key, shape in GEOMETRY_FIELDS.items()}


def sampled_geometry(points, prior_lidar, prior_ego, forward_lidar, rotation, map_ids):
    """IRWM maps translate by prior P_h and retain current LiDAR axes."""
    from world_models.matched_prior_sampler import retained_prior_axes
    if rotation is None or rotation.shape != (len(prior_lidar), 2, 2):
        raise ValueError('Candidate geometry requires the actual lidar-to-ego rotation')
    with torch.autocast(device_type=prior_lidar.device.type, enabled=False):
        rotation = rotation.float()
        lidar = prior_lidar.float()
        tangent, normal = retained_prior_axes(lidar, forward_lidar.float())
        axes_lidar = torch.stack((tangent, normal), 2)
        axes_ego = torch.einsum('bij,bhaj->bhai', rotation, axes_lidar)
        centers = prior_ego.float().cumsum(1)[:, :5]
        offsets_ego = torch.einsum('bij,bnj->bni', rotation, points['coordinates'].float())
        # offsets are already in translated future-map coordinates; add P_h once.
        token_positions = centers[:, map_ids] + offsets_ego
        radii = points['window_radii'].float()
        return dict(token_positions_ego=token_positions,
                    token_axes_ego=axes_ego[:, map_ids], token_radii_m=radii[:, map_ids],
                    window_centers_ego=centers, window_axes_ego=axes_ego,
                    window_radii_m=radii)


class CandidateDiTLayer(MatchedDiTLayer):
    def __init__(self, original, config):
        super().__init__(original.motion_gate_config)
        self.candidate_config = canonical_config(config)
        # Explicitly migrate the entire learned layer before adding parameters.
        self.load_state_dict(original.state_dict(), strict=True)
        if self.candidate_config['current_points'] == 5:
            self.current_query = nn.Linear(256, 256)
            nn.init.zeros_(self.current_query.weight)
            nn.init.zeros_(self.current_query.bias)
            self.current_point_logits = nn.Parameter(torch.tensor([math.log(16.), 0., 0., 0., 0.]))
        if self.candidate_config['future_gate_context']:
            previous = self.future_reliability[0]
            if previous.in_features != 282 or not self.integrated_motion:
                raise ValueError('Candidate gate requires the existing concatenated motion gate')
            expanded = nn.Linear(282 + GATE_EXTRA_DIM, previous.out_features)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :282].copy_(previous.weight)
                expanded.bias.copy_(previous.bias)
            self.future_reliability[0] = expanded
            fraction = self.candidate_config['distance_initial'] / self.candidate_config['distance_max']
            self.future_distance_logit = nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))))

    def read_current(self, hidden, positions, dense_bev, monitor=False):
        if self.candidate_config['current_points'] == 1:
            return super().read_current(hidden, positions, dense_bev, monitor)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            positions = positions.float()
            deltas = torch.cat((positions[:, :1], positions[:, 1:] - positions[:, :-1]), 1)
            lengths = deltas.norm(dim=-1, keepdim=True)
            forward = torch.zeros_like(deltas)
            forward[..., 0] = 1.
            tangent = torch.where(lengths > .05, deltas / lengths.clamp_min(.05), forward)
            normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
            offsets = torch.stack((torch.zeros_like(tangent), tangent, -tangent, normal, -normal), 2)
            points = positions[:, :, None] + self.candidate_config['current_radius_m'] * offsets
            grid = torch.stack((points[..., 1] / 32., points[..., 0] / 32.), -1)
            bound = grid.new_tensor([1 - 1 / dense_bev.shape[-1], 1 - 1 / dense_bev.shape[-2]])
            valid = (grid.abs() <= bound).all(-1) & torch.isfinite(grid).all(-1)
            safe_grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
        values = F.grid_sample(self.bev_value(dense_bev).float(), safe_grid,
                               align_corners=False, mode='bilinear', padding_mode='zeros')
        values = values.permute(0, 2, 3, 1).to(hidden.dtype).masked_fill(~valid[..., None], 0.)
        query = self.current_query(F.layer_norm(hidden, (256,)))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            scores = (query.float()[:, :, None] * F.layer_norm(values.float(), (256,))).sum(-1) / 16.
            content_scores = scores
            scores = scores + self.current_point_logits.float()
            any_valid = valid.any(-1)
            safe_valid = valid.clone()
            safe_valid[..., 0] |= ~any_valid
            weights = scores.masked_fill(~safe_valid, float('-inf')).softmax(-1)
            weights = weights * any_valid[..., None]
            result = (weights[..., None] * values.float()).sum(2).to(hidden.dtype)
        if monitor:
            self.last_stats['candidate/current_center_weight'] = _stat(weights[..., 0].sum(), any_valid.sum())
            self.last_stats['candidate/current_valid_fraction'] = _stat(valid.sum(), valid.numel())
            self.last_stats['candidate/current_content_score_abs'] = _stat((content_scores.abs() * valid).sum(), valid.sum())
            self.last_stats['candidate/current_bias_span'] = _stat(self.current_point_logits.max() - self.current_point_logits.min(), 1)
        return result

    def future_context(self, future, positions, prior_positions, monitor=False):
        if not self.candidate_config['future_geometry']:
            return future
        if any(key not in future for key in GEOMETRY_FIELDS):
            raise ValueError('Candidate future token geometry is missing')
        result = dict(future)
        with torch.autocast(device_type=positions.device.type, enabled=False):
            positions = positions.float()
            delta = positions[:, :, None] - future['token_positions_ego'].float()[:, None]
            local = torch.einsum('bqni,bnai->bqna', delta, future['token_axes_ego'].float())
            local = local / future['token_radii_m'].float()[:, None].clamp_min(.1)
            raw_distance = local.square().sum(-1)
            distance = raw_distance.clamp(max=self.candidate_config['distance_squared_cap'])
            strength = self.future_distance_logit.sigmoid() * self.candidate_config['distance_max']
            bias = -strength * distance
            original_bias = future['bias']
            b, q, n = bias.shape
            heads = original_bias.shape[0] // b
            result['bias'] = original_bias + bias[:, None].expand(b, heads, q, n).reshape(b * heads, q, n).to(original_bias)
            horizon = torch.arange(q, device=positions.device).clamp_max(4)
            center = future['window_centers_ego'].float()[:, horizon]
            axes = future['window_axes_ego'].float()[:, horizon]
            radius = future['window_radii_m'].float()[:, horizon].clamp_min(.1)
            window_local = torch.einsum('bqi,bqai->bqa', positions - center, axes) / radius
            window_distance = window_local.abs().amax(-1)
            counts = torch.stack([future['valid'][:, future['map_ids'] == h].float().mean(-1) for h in range(5)], -1)
            fraction = counts[:, horizon]
            supported = (window_distance <= 1) & (fraction > 0)
            previous = future.get('candidate_previous_positions', positions).float()
            changes = ((positions - previous) / 32.).clamp(-4, 4)
            has_map = (torch.arange(q, device=positions.device) < 5).float()[None, :, None].expand(b, -1, -1)
            result['candidate_gate_extra'] = torch.cat((changes, window_distance.clamp(max=4)[..., None],
                fraction[..., None], supported.float()[..., None], has_map), -1)
        if monitor:
            active = future['active'].float()[:, None]
            count = active.sum() * q
            self.last_stats['candidate/position_change_m'] = _stat(((positions - previous).norm(dim=-1) * active).sum(), count)
            self.last_stats['candidate/window_supported_fraction'] = _stat((supported * active).sum(), count)
            self.last_stats['candidate/window_distance'] = _stat((window_distance * active).sum(), count)
            self.last_stats['candidate/future_distance_strength'] = _stat(strength, 1)
            token_mask = future['valid'][:, None] & future['active'][:, None, None]
            token_mask = token_mask.expand_as(distance)
            self.last_stats['candidate/distance_clipped_fraction'] = _stat(
                ((raw_distance >= self.candidate_config['distance_squared_cap']) * token_mask).sum(), token_mask.sum())
            span = bias.masked_fill(~token_mask, float('-inf')).amax(-1) - bias.masked_fill(~token_mask, float('inf')).amin(-1)
            span = torch.where(token_mask.any(-1), span, torch.zeros_like(span))
            self.last_stats['candidate/distance_bias_span'] = _stat(span.sum(), token_mask.any(-1).sum())
        return result

    def reliability_context(self, inputs, future):
        if self.candidate_config['future_gate_context']:
            return torch.cat((inputs, future['candidate_gate_extra'].to(inputs)), -1)
        return inputs


def activate(model, config):
    config = canonical_config(config)
    if config is None:
        return model
    if model.architecture_version != 9 or not hasattr(model, 'geobev_config'):
        raise ValueError('Candidate interaction currently requires the GeoBEV V9 base')
    planner = model.base.planner
    # Do not perturb other newly initialized modules or subsequent noise draws.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(
            int(getattr(model, 'seed', 2026)) + 1010).get_state())
        for index, original in enumerate(planner.layers):
            if isinstance(original, MatchedDiTLayer):
                replacement = CandidateDiTLayer(original, config)
                planner.layers[index] = replacement.to(next(original.parameters()).device)
    planner.candidate_config = config
    model.candidate_config = config
    model.future_encoder.return_candidate_geometry = config['future_geometry']
    model.architecture_version = CANDIDATE_VERSION
    from world_models.world_partial import full_world_tuning
    model.training_scope = 'geobev_candidate_readout_full_irwm' if full_world_tuning(model.world_tuning_config) else CANDIDATE_SCOPE
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(model.is_trainable(name))
    model.train(model.training)
    return model


def warm_start_v9(model, path):
    from geobev_encoder.checkpoint import load
    from planning_runtime import file_sha256
    saved = torch.load(str(path), map_location='cpu')
    if saved.get('metadata', {}).get('architecture_version') != 9:
        raise ValueError('Warm-start source must be an audited GeoBEV V9 checkpoint')
    load(model, saved)
    return dict(status='PASS', source=str(path), source_sha256=file_sha256(path),
                source_epoch=saved.get('epoch'), source_architecture_version=9,
                consumed_source_tensors=len(saved['trainable']),
                new_parameter_slices=['candidate current readout and geometry/gate extensions'],
                optimizer_restored=False, scheduler_restored=False, rolling_history='empty')


def build(checkpoint, *args, **kwargs):
    from geobev_encoder.checkpoint import build as build_v9
    from geobev_encoder import GEOBEV_FORMAT
    meta = checkpoint.get('metadata', {})
    if checkpoint.get('format') != CANDIDATE_FORMAT or meta.get('architecture_version') != CANDIDATE_VERSION:
        raise ValueError('Candidate checkpoint format/version differs')
    config = canonical_config(meta.get('candidate_config'))
    if config is None:
        raise ValueError('Candidate checkpoint configuration is missing')
    proxy = dict(checkpoint, format=GEOBEV_FORMAT,
                 metadata=dict(meta, architecture_version=9))
    return activate(build_v9(proxy, *args, **kwargs), config)


def load(model, checkpoint):
    from temporal_matched_deform_model import load_matched_deform_state
    if checkpoint.get('format') != CANDIDATE_FORMAT or model.architecture_version != CANDIDATE_VERSION:
        raise ValueError('Candidate checkpoint/model required')
    meta = checkpoint['metadata']
    from training_optimization import check_training_conditioning
    check_training_conditioning(model, meta)
    for key in ('training_scope', 'fusion_config', 'sampler_config', 'motion_gate_config',
                'world_tuning_config', 'geobev_config', 'candidate_config'):
        if meta.get(key) != getattr(model, key):
            raise ValueError('Candidate checkpoint contract differs: ' + key)
    if meta.get('architecture_version') != CANDIDATE_VERSION:
        raise ValueError('Candidate checkpoint metadata version differs')
    load_matched_deform_state(model, checkpoint['trainable'], CANDIDATE_VERSION)


def metadata(model):
    from geobev_encoder.checkpoint import metadata as metadata_v9
    result = metadata_v9(model)
    result.update(candidate_config=model.candidate_config,
                  candidate_sampling_position='previous_supervised_x0_detached',
                  current_bev_points=model.candidate_config['current_points'],
                  future_attention_geometry='ego_metric_soft_bias' if model.candidate_config['future_geometry'] else 'legacy',
                  world_condition='retained_cross_frame_prior', world_rollouts_per_forward=1,
                  fresh_trainable_modules=result['fresh_trainable_modules'] +
                      ['base.planner.layers.*.current_query.*', 'base.planner.layers.*.current_point_logits'])
    return result
