"""Explicit optimizer groups and opt-in training diagnostics for v6 weights."""
import math
import numpy as np
import torch

TIME_BUCKETS = (('t00_24', 0, 24), ('t25_49', 25, 49), ('t50_74', 50, 74),
                ('t75_99', 75, 99), ('t99', 99, 99))
LOSS_DIAGNOSTIC_COUNT_KEYS = tuple('valid_samples_' + label for label, _, _ in TIME_BUCKETS)
def loss_diagnostic_keys(depth=12):
    return tuple('layer_%02d_x0' % i for i in range(1, depth+1)) + (
        'all_layers_t00_24', 'all_layers_t25_49', 'all_layers_t50_74',
        'all_layers_t75_99', 'all_layers_t99', 'last_layer_t99') + tuple(
        'layer_%02d_%s_x0' % (i, label) for i in range(1, depth+1) for label, _, _ in TIME_BUCKETS
    ) + LOSS_DIAGNOSTIC_COUNT_KEYS + tuple(
        'first_candidate_' + metric + '_' + label
        for metric in ('error_m', 'window_coverage') for label, _, _ in TIME_BUCKETS
    ) + tuple('layer_%02d_t99_prefix_l2_m' % i for i in range(1, depth+1))

LOSS_DIAGNOSTIC_KEYS = loss_diagnostic_keys()


def canonical_layer_weights(weights=None, depth=12):
    if weights is None:
        return [1. / depth] * depth
    values = [float(value) for value in weights]
    if len(values) != depth or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('%d positive finite x0 layer weights are required' % depth)
    total = sum(values)
    return [value / total for value in values]


def optimizer_groups(model, profile='legacy', lr=1e-4, image_lr=None,
                     pretrained_lr=None, planner_lr=None, future_lr=None, offset_lr=None, world_lr=None,
                     gate_lr=None, distance_lr=None, current_point_lr=None,
                     weight_decay=1e-4, new_weight_decay=None):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or profile not in ('legacy', 'differential'):
        raise ValueError('invalid optimizer profile/trainable scope')
    if profile == 'legacy':
        if any(x is not None for x in (image_lr, pretrained_lr, planner_lr, future_lr, offset_lr,
                                      gate_lr, distance_lr, current_point_lr, new_weight_decay)):
            raise ValueError('explicit group LRs require differential profile')
        rates = {'image_backbone': lr*.5, 'legacy_other': lr}
    else:
        new_rate = lr if future_lr is None else future_lr
        rates = {'image_backbone': 5e-6 if image_lr is None else image_lr,
                 'pretrained_other': 1e-5 if pretrained_lr is None else pretrained_lr,
                 'pretrained_planner': 2e-5 if planner_lr is None else planner_lr,
                 'new_future': new_rate, 'new_offset': new_rate if offset_lr is None else offset_lr}
    if hasattr(model, 'geobev_config'):
        if profile != 'differential': raise ValueError('GeoBEV requires differential learning rates')
        rates['new_bev_bridge'] = lr if future_lr is None else future_lr
        rates['geobev_depth_heads'] = rates['pretrained_other']
    if any(n.startswith('world.') for n, p in named):
        if profile != 'differential' or getattr(model, 'world_tuning_config', None) is None:
            raise AssertionError('World gradients need an explicit differential tuning contract')
        rates['world_tuned'] = 1e-6 if world_lr is None else world_lr
    if getattr(model, 'candidate_config', None):
        rates['new_current_readout'] = lr if future_lr is None else future_lr
    if getattr(model, 'plan_a_config', None):
        rates['interaction_gate_body'] = lr if future_lr is None else future_lr
        rates['interaction_gate_output'] = 1e-3 if gate_lr is None else gate_lr
    controls = {'control_future_gate': ('.future_gate', gate_lr),
                'control_future_distance': ('.future_distance_logit', distance_lr),
                'control_current_points': ('.current_point_logits', current_point_lr)}
    for key, (_, rate) in controls.items():
        if rate is not None:
            rates[key] = rate
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError('learning rates must be positive finite values')
    if any(not math.isfinite(v) or v < 0 for v in (weight_decay,) + (() if new_weight_decay is None else (new_weight_decay,))):
        raise ValueError('Weight decay must be finite and nonnegative')
    modules = dict(model.named_modules())
    groups, names, reasons = {}, {}, {}
    split = new_weight_decay is not None
    new_groups = {'new_future', 'new_offset', 'new_bev_bridge', 'new_current_readout'}
    norm_types = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm,
                  torch.nn.modules.instancenorm._InstanceNorm)
    for name, parameter in named:
        control = next((key for key, (suffix, rate) in controls.items()
                        if rate is not None and name.endswith(suffix)), None)
        is_interaction_gate = '.interaction_gate.' in name
        if is_interaction_gate:
            group = 'interaction_gate_output' if '.interaction_gate.2.' in name else 'interaction_gate_body'
        elif control is not None:
            group = control
        elif name.startswith('world.'):
            if not model.is_trainable(name): raise AssertionError('Unapproved world parameter: '+name)
            group = 'world_tuned'
        elif profile == 'legacy':
            group = 'image_backbone' if name.startswith('base.backbone.image.') else 'legacy_other'
        elif name.startswith('future_encoder.offset.'):
            group = 'new_offset'
        elif name.startswith('future_encoder.') or (name.startswith('base.planner.') and '.future_' in name):
            group = 'new_future'
        elif name.startswith(('base.geobev.img_view_transformer.depth_net.depth_head.', 'base.geobev.img_view_transformer.depth_convs.')):
            group = 'geobev_depth_heads'
        elif name.startswith(('base.bridge.', 'base.occupancy_projection.')):
            group = 'new_bev_bridge'
        elif name.startswith(('base.backbone.image.', 'base.geobev.img_backbone.')):
            group = 'image_backbone'
        elif name.startswith('base.planner.') and any(part in name for part in ('.current_query.', '.current_point_logits')):
            group = 'new_current_readout'
        elif name.startswith('base.planner.'):
            group = 'pretrained_planner'
        elif name.startswith('base.'):
            group = 'pretrained_other'
        else:
            raise AssertionError('unclassified trainable tensor: '+name)
        decay, reason = weight_decay, 'legacy_pretrained_matrix'
        if is_interaction_gate:
            decay, reason = 0., 'token_interpolation_gate'
        elif group == 'geobev_depth_heads':
            decay, reason = 0., 'protect_hard_foreground_rows'
        elif control is not None:
            decay, reason = 0., 'independent_control'
        elif split:
            owner_name, _, leaf = name.rpartition('.')
            owner = modules[owner_name]
            if leaf == 'bias' or leaf.endswith('_bias'):
                decay, reason = 0., 'bias'
            elif isinstance(owner, norm_types):
                decay, reason = 0., 'normalization'
            elif isinstance(owner, torch.nn.Embedding):
                decay, reason = 0., 'embedding'
            elif any(part in name.split('.') for part in
                     ('future_reliability', 'future_motion_projection', 'spatial_gate')):
                decay, reason = 0., 'gate_module'
            elif name.endswith(('.future_gate', '.future_distance_logit', '.current_point_logits')) or parameter.ndim < 2:
                decay, reason = 0., 'control_or_vector'
            elif group in new_groups:
                decay, reason = new_weight_decay, 'new_matrix'
        key = group + ('_decay' if decay else '_no_decay') if split and control is None else group
        if key not in groups:
            groups[key] = dict(name=key, source_group=group, params=[], lr=rates[group], weight_decay=decay)
            names[key], reasons[key] = [], {}
        assert groups[key]['weight_decay'] == decay
        groups[key]['params'].append(parameter)
        names[key].append(name)
        reasons[key][name] = reason
    for key, (_, rate) in controls.items():
        if rate is not None and not any(g['source_group'] == key for g in groups.values()):
            raise ValueError('Requested control group has no parameters: ' + key)
    order = {key: i for i, key in enumerate(rates)}
    result = sorted(groups.values(), key=lambda g: (order[g['source_group']], g['weight_decay'] == 0))
    seen = [id(p) for group in result for p in group['params']]
    assert len(seen) == len(set(seen)) == len(named)
    assert set(seen) == {id(p) for _, p in named}
    manifest = [{'name': group['name'], 'peak_lr': group['lr'],
                 'source_group': group['source_group'], 'weight_decay': group['weight_decay'],
                 'parameter_count': sum(p.numel() for p in group['params']),
                 'tensor_count': len(group['params']), 'parameter_names': names[group['name']],
                 'decay_reasons': reasons[group['name']]} for group in result]
    return result, manifest


def configure_training(model, args):
    fraction = float(args.t99_fraction)
    weights = tuple(float(x) for x in args.auxiliary_weights)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('t99 fraction must be in [0,1]')
    if len(weights) != 3 or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError('invalid auxiliary weights')
    if args.world_readout_steps not in (5, 6):
        raise ValueError('world readout steps must be 5 or 6')
    model.base.planner.training_t99_fraction = fraction
    model.base.planner.loss_diagnostics_enabled = bool(args.loss_diagnostics)
    model.training_cold_start_planning = bool(getattr(args, 'train_cold_start_planning', False))
    layer_weights = canonical_layer_weights(getattr(args, 'x0_layer_weights', None), len(model.base.planner.layers))
    model.base.planner.layer_loss_weights = (None if getattr(args, 'x0_layer_weights', None) is None else layer_weights)
    model.training_auxiliary_weights = weights
    color = None
    if hasattr(model, 'configure_color_augmentation'):
        color = model.configure_color_augmentation(getattr(args, 'geobev_color_probability', 0.),
            getattr(args, 'geobev_color_strength', .1), args.seed,
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
    elif getattr(args, 'geobev_color_probability', 0.):
        raise ValueError('Color augmentation requires GeoBEV')
    if model.world is not None:
        model.world.readout_steps = args.world_readout_steps
        if getattr(model, 'world_tuning_config', None) is not None:
            model.world.checkpoint_future = not getattr(args, 'no_world_checkpoint', False)
    return {'world_activation_checkpointing': bool(getattr(model.world, 'checkpoint_future', False)) if model.world_tuning_config is not None else False,
            'optimizer_profile': args.optimizer_profile, 't99_fraction': fraction,
            'train_cold_start_planning': model.training_cold_start_planning,
            'auxiliary_weights': list(weights), 'loss_diagnostics': bool(args.loss_diagnostics),
            'x0_layer_weights': layer_weights, 'color_augmentation': color,
            'decay_policy': 'split_new_v1' if getattr(args, 'new_weight_decay', None) is not None else 'legacy',
            'new_weight_decay': getattr(args, 'new_weight_decay', None),
            'world_readout_steps': args.world_readout_steps,
            'loss_diagnostic_keys': list(loss_diagnostic_keys(len(model.base.planner.layers)))}


def training_conditioning_metadata(model):
    planner = model.base.planner
    weights = getattr(planner, 'layer_loss_weights', None)
    contract = dict(schema='v10_training_conditioning_v1',
                x0_weighting='equal' if weights is None else 'weighted',
                x0_layer_weights=canonical_layer_weights(depth=len(planner.layers)) if weights is None else list(weights),
                t99_fraction=float(getattr(planner, 'training_t99_fraction', 0.)),
                geobev_color_augmentation=getattr(model, 'color_augmentation_config', None),
                image_preprocessing='shared_exact_pil_resize_crop_v1')
    if getattr(model, 'training_cold_start_planning', False):
        contract.update(schema='v10_training_conditioning_v2',
                        planning_loss_mask='future_label_validity_only_including_cold_start')
    if getattr(model, 'plan_a_config', None):
        contract.update(schema='v11_plan_a_training_conditioning_v1', plan_a_config=model.plan_a_config)
    return contract


def restore_training_conditioning(model, metadata):
    contract = metadata.get('training_conditioning')
    if contract is None:
        return  # Legacy checkpoints keep equal loss and deterministic evaluation inputs.
    if contract.get('schema') not in ('v10_training_conditioning_v1', 'v10_training_conditioning_v2', 'v11_plan_a_training_conditioning_v1'):
        raise ValueError('Unknown training conditioning schema')
    cold = contract['schema'] in ('v10_training_conditioning_v2', 'v11_plan_a_training_conditioning_v1')
    if cold and contract.get('planning_loss_mask') != 'future_label_validity_only_including_cold_start':
        raise ValueError('Unknown cold-start supervision contract')
    model.training_cold_start_planning = cold
    weights = contract['x0_layer_weights']
    canonical_layer_weights(weights, len(model.base.planner.layers))
    if abs(sum(weights) - 1.) > 1e-6 or contract['x0_weighting'] not in ('equal', 'weighted'):
        raise ValueError('Invalid saved x0 weighting')
    fraction = float(contract['t99_fraction'])
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid saved time distribution')
    model.base.planner.layer_loss_weights = list(weights) if contract['x0_weighting'] == 'weighted' else None
    model.base.planner.training_t99_fraction = fraction
    from geobev_encoder.augmentation import canonical_color_config
    color = canonical_color_config(contract['geobev_color_augmentation'])
    model.configure_color_augmentation(0. if color is None else color['probability'],
        .1 if color is None else color['strength'], model.seed)
    if training_conditioning_metadata(model) != contract:
        raise ValueError('Saved training/input contract differs')


def check_training_conditioning(model, metadata):
    contract = metadata.get('training_conditioning')
    if contract is not None and contract != training_conditioning_metadata(model):
        raise ValueError('Checkpoint training/input configuration differs')


class ControlUpdateMonitor:
    """Exact scalar updates, without random draws or model state changes."""
    def __init__(self, model):
        self.parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad and
            name.endswith(('.future_gate', '.future_distance_logit', '.current_point_logits'))]
        self.cumulative = {name: torch.zeros_like(p) for name, p in self.parameters}
        self.last = {}
        self.updates = 0

    @torch.no_grad()
    def before_step(self):
        self.before = {name: p.detach().clone() for name, p in self.parameters}

    @torch.no_grad()
    def after_step(self, monitor=False):
        self.updates += 1
        for name, p in self.parameters:
            update = p.detach() - self.before[name]
            self.cumulative[name].add_(update.abs())
            if monitor:
                for key, value in (('value', p.float().mean()), ('update_abs', update.abs().mean()),
                                   ('cumulative_abs', self.cumulative[name].mean()),
                                   ('gradient_abs', p.grad.abs().mean() if p.grad is not None else p.new_zeros(()))):
                    self.last['control/' + name + '/' + key] = torch.stack((value.detach(), value.new_ones(())))
        self.before = {}

    def state_dict(self):
        return dict(updates=self.updates, cumulative={n: v.detach().cpu().clone() for n, v in self.cumulative.items()})

    def load_state_dict(self, state):
        if set(state['cumulative']) != set(self.cumulative):
            raise ValueError('Control monitor parameter names differ')
        self.updates = int(state['updates'])
        for name, parameter in self.parameters:
            self.cumulative[name] = state['cumulative'][name].to(parameter)


@torch.no_grad()
def regularization_snapshot(optimizer):
    return [[p.detach().clone() for p in group['params']] for group in optimizer.param_groups]


@torch.no_grad()
def regularization_monitor(optimizer, before):
    result = {}
    for group, old_values in zip(optimizer.param_groups, before):
        norm = torch.stack([p.float().square().sum() for p in old_values]).sum().sqrt()
        decay = norm * group['lr'] * group['weight_decay']
        updates = [p.detach().float() - old.float() for p, old in zip(group['params'], old_values)]
        update_norm = torch.stack([u.square().sum() for u in updates]).sum().sqrt()
        gradient_norm = torch.stack([(u + old.float() * group['lr'] * group['weight_decay']).square().sum()
                                    for u, old in zip(updates, old_values)]).sum().sqrt()
        for name, value in (('weight_norm', norm), ('decay_update_norm', decay),
                            ('actual_update_norm', update_norm), ('adam_component_norm', gradient_norm),
                            ('decay_to_adam_ratio', decay / gradient_norm.clamp_min(1e-12))):
            result['regularization/' + group['name'] + '/' + name] = torch.stack((value, value.new_ones(())))
    return result


def restore_fused_optimizer_devices(optimizer):
    """Torch 2.0 restores AdamW step tensors to CPU unless explicitly repaired."""
    moved = 0
    for group in optimizer.param_groups:
        if not group.get('fused', False):
            continue
        for parameter in group['params']:
            for key, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.device != parameter.device:
                    optimizer.state[parameter][key] = value.to(device=parameter.device)
                    moved += 1
    return moved


def exact_state_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(exact_state_equal(x,y) for x,y in zip(a,b))
    return a == b


def verify_restored_state(model, optimizer, scheduler, sampler, current_rng, saved, rank, control_monitor=None):
    from temporal_matched_deform_model import is_trainable_state
    actual = {n:v for n,v in model.state_dict().items() if is_trainable_state(n, model)}
    checks = dict(model=exact_state_equal(actual, saved['trainable']),
        optimizer=exact_state_equal(optimizer.state_dict(), saved['optimizer']),
        scheduler=exact_state_equal(scheduler.state_dict(), saved['scheduler']),
        sampler=exact_state_equal(sampler.state_dict(), saved['rank_states'][rank]['sampler']),
        rolling=exact_state_equal(model.stream.state_dict(), saved['rank_states'][rank]['rolling']),
        rng=exact_state_equal(current_rng, saved['rank_states'][rank]['rng']))
    if control_monitor is not None:
        checks['control_monitor'] = exact_state_equal(control_monitor.state_dict(),
            saved['rank_states'][rank]['control_monitor'])
    if not all(checks.values()):
        raise AssertionError('Checkpoint restoration lost state: '+str(checks))
    return dict(rank=rank,status='PASS',checks=checks)


@torch.no_grad()
def loss_diagnostics(predictions, clean, valid, timestep, denormalize=None,
                     target_positions=None, candidate_positions=None, future=None, future_mask=None, clip_normalized=None):
    # Detached reductions add no random draws or gradients to the training loss.
    squared = torch.stack([((p.detach().float()-clean).square()*valid).sum((1, 2)) for p in predictions])
    denominator = 2*valid.sum((1, 2))
    pairs = [torch.stack((value.sum(), denominator.sum())) for value in squared]
    average = squared.mean(0)
    for low, high in ((0,24), (25,49), (50,74), (75,99), (99,99)):
        select = ((timestep >= low) & (timestep <= high)).float()
        pairs.append(torch.stack(((average*select).sum(), (denominator*select).sum())))
    select = (timestep == 99).float()
    pairs.append(torch.stack(((squared[-1]*select).sum(), (denominator*select).sum())))
    selections = [((timestep >= low) & (timestep <= high)).float() for _, low, high in TIME_BUCKETS]
    for layer in squared:
        for select in selections:
            pairs.append(torch.stack(((layer*select).sum(), (denominator*select).sum())))
    for select in selections:
        count = (select * (denominator > 0)).sum()
        pairs.append(torch.stack((count, count.new_ones(()))))
    zero = clean.new_zeros(2)
    error = None if candidate_positions is None else (candidate_positions.float() - target_positions).norm(dim=-1)
    for select in selections:
        weight = valid[..., 0] * select[:, None]
        pairs.append(zero if error is None else torch.stack(((error * weight).sum(), weight.sum())))
    coverage = coverage_mask = None
    if candidate_positions is not None and future is not None and 'window_centers_ego' in future:
        local = torch.einsum('bhi,bhai->bha',
            candidate_positions[:, :5].float() - future['window_centers_ego'].float(),
            future['window_axes_ego'].float())
        coverage = (local.abs() <= future['window_radii_m'].float()).all(-1).float()
        coverage_mask = valid[:, :5, 0] * future_mask.float()[:, None]
    for select in selections:
        weight = None if coverage_mask is None else coverage_mask * select[:, None]
        pairs.append(zero if coverage is None else torch.stack(((coverage * weight).sum(), weight.sum())))
    complete = valid[..., 0].bool().all(-1) & (timestep == 99)
    for prediction in predictions:
        if denormalize is None:
            pairs.append(zero)
            continue
        value = prediction.detach().float()
        positions = denormalize(value.clamp(-1, 1) if clip_normalized is None else clip_normalized(value)).cumsum(1)
        distances = (positions - target_positions).norm(dim=-1)
        prefix = torch.stack([distances[:, :h].mean(-1) for h in (2, 4, 6)]).mean(0)
        pairs.append(torch.stack(((prefix * complete).sum(), complete.sum().to(prefix))))
    assert len(pairs) == len(loss_diagnostic_keys(len(predictions)))
    return torch.stack(pairs)


def diagnostic_values(pairs, depth=12):
    return {key: float(value[0]) if key in LOSS_DIAGNOSTIC_COUNT_KEYS else float(value[0] / value[1])
            for key, value in zip(loss_diagnostic_keys(depth), pairs) if float(value[1]) > 0}
