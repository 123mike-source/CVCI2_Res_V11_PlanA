"""Observe loss balance on independent planner parameters, without DDP hooks.

Replay only the deterministic planner with the exact training noise/context.
Perception/IRWM, stream state and training RNG are not advanced. The detached
parameter replicas ensure autograd.grad never touches the real DDP reducer.
"""
import torch
from torch import nn
import torch.distributed as dist
from torch.func import functional_call


def detached(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detached(v) for v in value)
    if isinstance(value, list):
        return [detached(v) for v in value]
    return value


def selected(name):
    return name.startswith(('planner.head.', 'planner.layers.9.', 'planner.layers.10.'))


class _Replay(nn.Module):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner

    def forward(self, context):
        predictions = self.planner.predict(*context['args'], **context['kwargs'])
        delta, mask, timestep = (context[k] for k in ('delta', 'mask', 'timestep'))
        with torch.autocast(device_type=delta.device.type, enabled=False):
            clean = self.planner.normalize(delta.float())
            valid = mask[..., None].float()
            denominator = (2*valid.sum()).clamp_min(1)
            layers = torch.stack([((p.float()-clean).square()*valid).sum()/denominator for p in predictions])
            weights = self.planner.layer_loss_weights
            x0 = layers.mean() if weights is None else (layers*layers.new_tensor(weights)).sum()
            from plan_a import prefix_position_loss
            position, count = prefix_position_loss(self.planner.denormalize(predictions[-1].float()),
                delta, mask, timestep, formulation=self.planner.plan_a_config['position_loss'])
        return x0, position, count


def measure(planner, target_ratio=.2):
    context = planner.gradient_probe_context
    if context is None or 'delta' not in context:
        raise RuntimeError('No captured training loss/context for gradient monitoring')
    if not 0 < target_ratio < 1:
        raise ValueError('Loss-gradient target must be between zero and one')
    wrapper = _Replay(planner)
    parameters = dict(wrapper.named_parameters())
    identities = {n: id(p) for n, p in parameters.items()}
    replicas = {n: p.detach().clone().requires_grad_(selected(n)) if selected(n) else p.detach()
                for n, p in parameters.items()}
    replicas.update({n: v.detach() for n, v in wrapper.named_buffers()})
    trainable = [replicas[n] for n in parameters if selected(n)]
    device = context['delta'].device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    # Attributes are diagnostic caches, not state_dict entries. Restore them so
    # the regular monitor still describes the original train/deployment passes.
    names = ('last_stats', 'last_interaction_gates', 'last_point_gate',
             'last_residual_ratio', 'last_training_candidate_positions')
    attributes = [(module, name, getattr(module, name)) for module in planner.modules()
                  for name in names if hasattr(module, name)]
    planner.gradient_probe_requested = False
    try:
        with torch.enable_grad(), torch.autocast(device_type=device.type,
                enabled=context['autocast'], dtype=context['autocast_dtype']):
            x0, position, count = functional_call(wrapper, replicas, (context,), strict=True)
        vectors = []
        for i, loss in enumerate((x0, position)):
            gradients = torch.autograd.grad(loss, trainable, retain_graph=i == 0, allow_unused=True)
            vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten()
                                for p, g in zip(trainable, gradients)])
            if dist.is_initialized():
                dist.all_reduce(vector)
                vector.div_(dist.get_world_size())
            if not torch.isfinite(vector).all():
                raise FloatingPointError('Nonfinite position-gradient diagnostic')
            vectors.append(vector)
        if dist.is_initialized():
            dist.all_reduce(count)
        norm_x0, norm_pos = (float(v.norm()) for v in vectors)
        valid = norm_x0 > 0 and int(count) > 0
        ramp = min(1., max(0., planner.position_loss_update/planner.plan_a_config['position_loss_ramp_updates']))
        base = planner.plan_a_config['position_loss_weight']
        full_ratio = base*norm_pos/max(norm_x0, 1e-30)
        cosine = float(torch.dot(vectors[0], vectors[1])/(vectors[0].norm()*vectors[1].norm()).clamp_min(1e-30))
        result = dict(schema='global_loss_parameter_gradient_ratio_v1',
            scope='shared_head_and_layers_10_11', optimizer_update=planner.position_loss_update,
            valid=valid, t99_valid_samples=int(count), gradient_x0_norm=norm_x0,
            gradient_position_unweighted_norm=norm_pos, base_coefficient=base,
            effective_coefficient=base*ramp, ramp=ramp,
            weighted_ratio=full_ratio*ramp, ratio_at_full_coefficient=full_ratio,
            target_ratio=target_ratio, effective_target_ratio=target_ratio*ramp,
            reference_low=target_ratio*.5, reference_high=target_ratio*1.5,
            outside_reference_band=bool(valid and not target_ratio*.5 <= full_ratio <= target_ratio*1.5),
            gradient_cosine=cosine,
            x0_replay=float(x0), position_replay_m=float(position),
            interpretation='parameter-gradient norms before joint clipping/Adam; not a fraction of final parameter updates')
    finally:
        for module, name, value in attributes:
            setattr(module, name, value)
        planner.gradient_probe_context = None
        planner.gradient_probe_requested = False
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise AssertionError('Gradient replay changed CPU RNG')
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise AssertionError('Gradient replay changed CUDA RNG')
    if {n: id(p) for n, p in wrapper.named_parameters()} != identities:
        raise AssertionError('Functional gradient replay did not restore real parameters')
    return result
