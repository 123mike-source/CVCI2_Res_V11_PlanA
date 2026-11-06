"""V11 Plan A: original S1..S11, token gates and metric prefix supervision.

This module is opt-in. Legacy checkpoints retain their twelve-layer codec,
state keys and loss. Original pretraining is loaded/audited before activation.
"""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from candidate_interaction import CandidateDiTLayer
from matched_diffusion import MatchedX0Planner, _stat

PLAN_A_VERSION = 11
PLAN_A_FORMAT = 'cvci2res_geobev_plan_a_v11'
PLAN_A_SCOPE = 'geobev_plan_a_11_layers_partial_irwm'
DEFAULT_CONFIG = dict(schema='plan_a_v1', depth=11,
    roles=['self', 'interaction'] * 5 + ['self'],
    interpolation='(1-g)*h+g*postnorm_original_update', gate_range=[0., 2.],
    gate_inputs='block_input_layernorm256_geometry6_observed_motion20',
    ego='folded_single_key_value_output_linear',
    candidate_readouts='all_11_detached_head_outputs',
    codec='original_affine_unclipped_gt_metric_bounds_v1',
    delta_bounds_m=[[-1., 12.], [-5., 5.]],
    position_loss='final_self_t99_fixed_prefix_weights_continuous_valid_v2',
    position_loss_ramp_updates=500, position_loss_weight=0.)


def canonical_config(config=None):
    result = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        if set(config) != set(result):
            raise ValueError('Plan A configuration keys differ')
        result.update(copy.deepcopy(config))
    for key in result:
        if key in ('position_loss_weight', 'codec', 'delta_bounds_m', 'position_loss'):
            continue
        if result[key] != DEFAULT_CONFIG[key]:
            raise ValueError('Unsupported Plan A contract: ' + key)
    if result['codec'] not in (DEFAULT_CONFIG['codec'], 'legacy_clipped_affine_test_only'):
        raise ValueError('Unknown trajectory codec')
    if result['delta_bounds_m'] != DEFAULT_CONFIG['delta_bounds_m']:
        raise ValueError('Unknown metric output bounds')
    if result['position_loss'] not in (DEFAULT_CONFIG['position_loss'],
            'final_self_t99_continuous_valid_prefix_mean_1_2_3s'):
        raise ValueError('Unknown position-loss formulation')
    if not math.isfinite(result['position_loss_weight']) or result['position_loss_weight'] < 0:
        raise ValueError('Invalid calibrated position loss coefficient')
    return result


def adopt_module(destination, source):
    """Transfer an already loaded module without drawing RNG or renaming keys."""
    nn.Module.__init__(destination)
    for key, value in source.__dict__.items():
        destination.__dict__[key] = value.copy() if isinstance(value, dict) else value


def fold_single_key_attention(attention):
    if attention.dropout or attention.add_zero_attn or attention.bias_k is not None:
        raise ValueError('Ego attention must have one unmodified key, no dropout')
    if not attention._qkv_same_embed_dim:
        raise ValueError('Ego attention embedding dimensions differ')
    # Fold on CPU FP32, independent of autocast/TF32. There is exactly one key,
    # so softmax=1 and Q/K have no effect; ego still varies with each frame.
    weight = attention.in_proj_weight.detach().float().cpu().chunk(3)[2]
    bias = attention.in_proj_bias.detach().float().cpu().chunk(3)[2]
    out_weight = attention.out_proj.weight.detach().float().cpu()
    out_bias = attention.out_proj.bias.detach().float().cpu()
    linear = nn.Linear(attention.embed_dim, attention.embed_dim)
    with torch.no_grad():
        linear.weight.copy_(out_weight @ weight)
        linear.bias.copy_(out_weight @ bias + out_bias)
    return linear.to(device=attention.in_proj_weight.device)


class PlanAInteraction(CandidateDiTLayer):
    def __init__(self, original):
        adopt_module(self, original)
        self.ego_projection = fold_single_key_attention(self.ego_attn)
        del self.ego_attn
        self.interaction_gate = nn.Sequential(nn.Linear(282, 128), nn.SiLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.interaction_gate[-1].weight)
        nn.init.zeros_(self.interaction_gate[-1].bias)
        self.last_interaction_gates = None

    @staticmethod
    def interpolate(hidden, transformed, gate):
        # Compute interpolation in FP32; preserve the original branch's dtype.
        result = (1. - gate.float()) * hidden.float() + gate.float() * transformed.float()
        return result.to(transformed.dtype)

    def gate_values(self, hidden, positions, prior, timestep_fraction, motion):
        if motion is None or motion.shape != (*hidden.shape[:2], 20):
            raise ValueError('Plan A gates require observed [B,6,20] motion, including cold start')
        if prior is None:
            prior = torch.zeros_like(positions)
        horizon = torch.arange(1, 7, device=hidden.device, dtype=hidden.dtype)[None, :, None] / 6.
        geometry = torch.cat(((prior / 32.).clamp(-4, 4),
            ((positions - prior) / 32.).clamp(-4, 4),
            horizon.expand(hidden.shape[0], -1, -1),
            timestep_fraction[:, None, None].expand(-1, 6, -1)), -1)
        inputs = torch.cat((F.layer_norm(hidden.float(), (256,)),
                            geometry.float(), motion.float()), -1)
        # New gates are intentionally FP32 to retain sub-BF16 scalar updates.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return 2. * self.interaction_gate(inputs).sigmoid()

    def forward(self, hidden, positions, dense_bev, agents, ego, time, status,
                future=None, prior_positions=None, diffusion_fraction=None,
                monitor=False, motion_features=None):
        if monitor:
            self.last_stats = {}
        gates = self.gate_values(hidden, positions, prior_positions, diffusion_fraction, motion_features)
        self.last_interaction_gates = gates.detach()
        updates = (
            lambda h: self.norms[1](h + self.agent_attn(h, agents, agents, need_weights=False)[0]),
            lambda h: self.norms[2](h + self.ego_projection(ego)),
            lambda h: self.norms[0](h + self.read_current(h, positions, dense_bev, monitor)),
        )
        for index, (name, operation) in enumerate(zip(('agent', 'ego', 'bev'), updates)):
            transformed = operation(hidden)
            gate = gates[..., index:index+1]
            if monitor:
                ratio = (transformed.float()-hidden.float()).norm(dim=-1) / hidden.float().norm(dim=-1).clamp_min(1e-6)
                self.last_stats['interaction/'+name+'/gate'] = _stat(gate.sum(), gate.numel())
                self.last_stats['interaction/'+name+'/amplify_fraction'] = _stat((gate > 1.).sum().float(), gate.numel())
                self.last_stats['interaction/'+name+'/raw_residual_ratio'] = _stat(ratio.sum(), ratio.numel())
                self.last_stats['interaction/'+name+'/gated_residual_ratio'] = _stat((ratio*gate[..., 0]).sum(), ratio.numel())
            hidden = self.interpolate(hidden, transformed, gate)
        return self._finish_interaction(hidden, positions, time, status, future,
            prior_positions, diffusion_fraction, monitor, motion_features)


def prefix_position_loss(predicted_delta, true_delta, mask, timestep,
                         formulation='final_self_t99_fixed_prefix_weights_continuous_valid_v2'):
    """Isotropic metres, exact MeanPrefix on complete trajectories.

    Point k needs all first k delta labels. The six metric weights remain fixed
    even for partial labels; normalize by their valid weight sum per sample.
    The previous partial-window rule is retained only for old V11 checkpoints.
    """
    valid = mask.bool().long().cumprod(1).bool() & (timestep == 99)[:, None]
    # Sanitize invalid suffixes before cumsum: NaN/garbage labels at an invalid
    # delta must not contaminate values or gradients of an earlier valid point.
    difference = torch.where(valid[..., None], predicted_delta.float()-true_delta.float(),
                             torch.zeros_like(predicted_delta, dtype=torch.float32))
    distance = torch.linalg.vector_norm(difference.cumsum(1), dim=-1)
    if formulation == 'final_self_t99_fixed_prefix_weights_continuous_valid_v2':
        weight = distance.new_tensor([11.,11.,5.,5.,2.,2.])[None]*valid
        samples = weight.sum(-1) > 0
        values = (distance*weight).sum(-1)/weight.sum(-1).clamp_min(1)
        return (values*samples).sum()/samples.sum().clamp_min(1), samples.sum()
    if formulation != 'final_self_t99_continuous_valid_prefix_mean_1_2_3s':
        raise ValueError('Unknown position loss')
    sums, counts = [], []
    for horizon in (2, 4, 6):
        count = valid[:, :horizon].sum(-1)
        mean = (distance[:, :horizon]*valid[:, :horizon]).sum(-1)/count.clamp_min(1)
        sums.append(mean)
        counts.append(count > 0)
    available = torch.stack(counts, -1)
    values = (torch.stack(sums, -1)*available).sum(-1)/available.sum(-1).clamp_min(1)
    samples = available.any(-1)
    return (values*samples).sum()/samples.sum().clamp_min(1), samples.sum()


class PlanAPlanner(MatchedX0Planner):
    def __init__(self, original, config):
        adopt_module(self, original)
        self.plan_a_config = canonical_config(config)
        self.layers = nn.ModuleList(list(original.layers)[:11])
        for i, layer in enumerate(self.layers):
            if layer.interaction:
                self.layers[i] = PlanAInteraction(layer).to(next(layer.parameters()).device)
        self.position_loss_update = 0
        self.gradient_probe_requested = False
        self.gradient_probe_context = None

    def predict(self, *args, **kwargs):
        if self.gradient_probe_requested and torch.is_grad_enabled():
            from position_gradient_monitor import detached
            self.gradient_probe_context = dict(args=detached(args), kwargs=detached(kwargs),
                autocast=torch.is_autocast_enabled(), autocast_dtype=torch.get_autocast_gpu_dtype())
        return super().predict(*args, **kwargs)

    def normalize(self, delta):
        if self.plan_a_config['codec'] == 'legacy_clipped_affine_test_only':
            return super().normalize(delta)
        return torch.stack(((delta[..., 0]-3.55)/4.05, delta[..., 1]/3.), -1)

    def clip_normalized(self, prediction):
        if self.plan_a_config['codec'] == 'legacy_clipped_affine_test_only':
            return prediction.clamp(-1, 1)
        lower = prediction.new_tensor([(-1.-3.55)/4.05, -5./3.])
        upper = prediction.new_tensor([(12.-3.55)/4.05, 5./3.])
        return torch.maximum(torch.minimum(prediction, upper), lower)

    def ddim_update(self, value, prediction, alpha, next_alpha):
        if self.plan_a_config['codec'] == 'legacy_clipped_affine_test_only':
            return super().ddim_update(value, prediction, alpha, next_alpha)
        value, prediction = value.float(), prediction.float()
        alpha, next_alpha = alpha.float(), next_alpha.float()
        epsilon = (value-alpha.sqrt()*prediction)/(1-alpha).sqrt()
        return next_alpha.sqrt()*self.clip_normalized(prediction)+(1-next_alpha).sqrt()*epsilon

    def extend_loss_terms(self, terms, predictions, delta, mask, timestep):
        position, count = prefix_position_loss(self.denormalize(predictions[-1].float()), delta, mask, timestep,
                                               formulation=self.plan_a_config['position_loss'])
        if self.gradient_probe_requested and self.gradient_probe_context is not None:
            self.gradient_probe_context.update(delta=delta.detach(), mask=mask.detach(), timestep=timestep.detach())
        config = self.plan_a_config
        ramp = min(1., max(0., self.position_loss_update/config['position_loss_ramp_updates']))
        coefficient = float(config['position_loss_weight'])*ramp
        terms.update(position=position, position_samples=count,
                     position_coefficient=position.new_tensor(coefficient))
        terms['total'] = terms['x0']+coefficient*position
        return terms


def activate(model, config=None):
    if model.architecture_version != 10 or model.candidate_config['schema'] != 'supervised_candidate_readout_v1':
        raise ValueError('Plan A activation requires an audited full V10 model')
    if any(model.candidate_config.get(k) != v for k, v in
           [('dynamic_positions', True), ('current_points', 5), ('future_geometry', True)]):
        raise ValueError('Plan A requires full candidate geometry and five-point current readout')
    from world_models.world_partial import full_world_tuning
    if full_world_tuning(model.world_tuning_config):
        raise ValueError('Plan A uses the partial IRWM tuning strategy')
    config = canonical_config(config)
    original = model.base.planner
    if len(original.layers) != 12:
        raise ValueError('Load original twelve-layer pretraining before Plan A activation')
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(int(model.seed)+1111).get_state())
        model.base.planner = PlanAPlanner(original, config)
    retained = model.base.planner.state_dict()
    unchanged = [name for name in retained if name in original.state_dict()]
    assert all(torch.equal(retained[name], original.state_dict()[name]) for name in unchanged)
    model.plan_a_initialization_audit = dict(status='PASS', retained_tensors_exact=len(unchanged),
        strict_original_load_before_activation=True, original_depth=12, final_depth=11,
        discarded_layer='interaction_12', folded_ego_layers=[2,4,6,8,10],
        new_gate_output_zero=True, initial_interaction_gate=1.,
        source_trained_res_checkpoint_loaded=False)
    model.plan_a_config = config
    model.architecture_version = PLAN_A_VERSION
    model.training_scope = PLAN_A_SCOPE
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(model.is_trainable(name))
    model.train(model.training)
    return model


def build(checkpoint, *args, **kwargs):
    from candidate_interaction import build as build_v10, CANDIDATE_FORMAT
    meta = checkpoint.get('metadata', {})
    if checkpoint.get('format') != PLAN_A_FORMAT or meta.get('architecture_version') != PLAN_A_VERSION:
        raise ValueError('Plan A checkpoint format/version differs')
    config = canonical_config(meta.get('plan_a_config'))
    if config['codec'] == 'legacy_clipped_affine_test_only':
        raise ValueError('Test-only codec cannot be loaded as a production checkpoint')
    # The parent constructor has 12 layers; restore V11 conditioning only after
    # cropping and gates have been created. No V10 weights are consumed here.
    parent_meta = dict(meta, architecture_version=10)
    parent_meta.pop('training_conditioning', None)
    proxy = dict(checkpoint, format=CANDIDATE_FORMAT, metadata=parent_meta)
    model = activate(build_v10(proxy, *args, **kwargs), config)
    from training_optimization import restore_training_conditioning
    restore_training_conditioning(model, meta)
    return model


def load(model, checkpoint):
    from temporal_matched_deform_model import load_matched_deform_state
    from training_optimization import check_training_conditioning
    if checkpoint.get('format') != PLAN_A_FORMAT or model.architecture_version != PLAN_A_VERSION:
        raise ValueError('Plan A checkpoint/model required')
    meta = checkpoint['metadata']
    check_training_conditioning(model, meta)
    for key in ('training_scope', 'fusion_config', 'sampler_config', 'motion_gate_config',
                'world_tuning_config', 'geobev_config', 'candidate_config', 'plan_a_config'):
        if meta.get(key) != getattr(model, key):
            raise ValueError('Plan A checkpoint contract differs: '+key)
    load_matched_deform_state(model, checkpoint['trainable'], PLAN_A_VERSION)


def metadata(model):
    from candidate_interaction import metadata as parent_metadata
    result = parent_metadata(model)
    result.update(plan_a_config=model.plan_a_config, diffusion_depth=11,
        self_attention_layers=6, interaction_layers=5,
        planner_layout='stem_self_1_plus_5_interaction_self_pairs',
        final_output='self_11', trajectory_codec=model.plan_a_config['codec'],
        trajectory_loss='weighted_11_layer_x0_plus_ramped_final_t99_metric_mean_prefix',
        loss_definition='weighted_11_layer_x0_plus_position_and_configured_perception_depth',
        current_interaction_order='token_gated_postnorm_agent_ego_bev',
        original_initialization='strict_original_12_layer_load_then_crop_interaction_12',
        fresh_trainable_modules=result['fresh_trainable_modules']+
            ['base.planner.layers.*.interaction_gate.*'])
    return result
