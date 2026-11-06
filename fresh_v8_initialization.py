"""Explicit, audited original-pretraining plus fresh V8 Res initialization."""
import hashlib

import torch

FRESH_V8_PROFILE = 'original_cvci2_irwm_pretraining_fresh_v8_res_v1'
EXPECTED_BASE_SHA256 = 'b70f1668f2c6c83d01e559e45e652394703d72cb34fd2e1182cb9fd2c42e9a5d'
EXPECTED_WORLD_SHA256 = 'aa63288216282fa45fac48bbdbedd2fa67624b31058ea650ed5f8cc43c7b6d86'


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def initialize_stable_fresh_columns(model):
    """Keep fresh motion/radius additions neutral without drawing more RNG."""
    layers = [x for x in model.base.planner.layers if hasattr(x, 'future_reliability')]
    if len(layers) != 6:
        raise ValueError('V8 requires six future interaction layers')
    for layer in layers:
        if layer.future_reliability[0].in_features != 282:
            raise ValueError('V8 motion gate input dimension differs')
        layer.future_reliability[0].weight[:, 262:].zero_()
    if model.sampler_config.get('schema') == 'center_multiscale_motion_v1':
        geometry = model.future_encoder.geometry[0]
        if geometry.in_features != 27:
            raise ValueError('V8 sampler motion input dimension differs')
        geometry.weight[:, 7:].zero_()
        model.future_encoder.offset.weight[-2:].zero_()
        model.future_encoder.offset.bias[-2:].zero_()


@torch.no_grad()
def audit_fresh_v8(model, base_path, world_path, seed):
    if model.architecture_version != 8 or model.sampler_config.get('schema') != 'center_multiscale_motion_v1':
        raise ValueError('Fresh audit requires the approved V8 architecture')
    if model.world is None or model.world_tuning_config is None:
        raise ValueError('This arm requires the pretrained partially trainable IRWM')
    base_sha, world_sha = file_digest(base_path), file_digest(world_path)
    if (base_sha, world_sha) != (EXPECTED_BASE_SHA256, EXPECTED_WORLD_SHA256):
        raise ValueError('Fresh run must use the original approved pretrained checkpoints')
    original = torch.load(str(base_path), map_location='cpu')
    source = original.get('model', original)
    shared = model.base.state_dict()
    if not all(n in shared and torch.equal(shared[n].detach().cpu(), v) for n, v in source.items()):
        raise RuntimeError('CVCI2 parameters are not the original pretrained tensors')
    shared_count = len(source); base_epoch = original.get('epoch')
    del original, source
    original = torch.load(str(world_path), map_location='cpu')
    source = original.get('state_dict', original.get('model', original))
    source = {n[7:] if n.startswith('module.') else n: v for n, v in source.items()}
    target = model.world.model.state_dict()
    allowed = {'pts_bbox_head.transformer.reference_points.weight', 'pts_bbox_head.transformer.reference_points.bias'}
    missing = [n for n in target if n not in source and n not in allowed and not n.endswith('num_batches_tracked')]
    if missing:
        raise RuntimeError('Missing pretrained IRWM tensors: ' + str(missing))
    loaded = [n for n in target if n in source]
    if not all(torch.equal(target[n].detach().cpu(), source[n]) for n in loaded):
        raise RuntimeError('IRWM is not initialized from the original pretrained tensors')
    del original, source, target
    current = model.state_dict()
    motion = {n: v[:, 262:] for n, v in current.items() if n.endswith('.future_reliability.0.weight')}
    assert len(motion) == 6 and all(torch.count_nonzero(v) == 0 for v in motion.values())
    layers = [x for x in model.base.planner.layers if hasattr(x, 'future_gate')]
    assert all(float(x.future_gate) == 0 and x.future_reliability[-1].weight.count_nonzero() == 0
               and x.future_reliability[-1].bias.count_nonzero() == 0 for x in layers)
    assert not any('future_motion_projection' in n for n, p in model.named_parameters())
    sampler = model.future_encoder
    assert sampler.geometry[0].weight[:, 7:].count_nonzero() == 0
    assert sampler.offset.weight.count_nonzero() == 0 and sampler.offset.bias.count_nonzero() == 0
    assert not any(m.training for m in model.world.modules())
    tuned_count = sum(p.numel() for p in model.world.parameters() if p.requires_grad)
    assert tuned_count == 907712
    res = {n: v for n, v in current.items() if n.startswith('future_encoder.') or
           (n.startswith('base.planner.') and '.future_' in n)}
    assert any(v.float().std() > .01 for n, v in res.items() if n.endswith('future_attn.in_proj_weight'))
    return {'status': 'PASS', 'profile': FRESH_V8_PROFILE, 'seed': seed,
            'base_checkpoint': str(base_path), 'base_sha256': base_sha, 'base_source_epoch': base_epoch,
            'world_checkpoint': str(world_path), 'world_sha256': world_sha,
            'original_cvci2_tensors_verified': shared_count, 'original_irwm_tensors_verified': len(loaded),
            'known_unused_generated_reference_tensors': sorted(allowed),
            'res_tensor_count': len(res), 'fresh_res_state_sha256': state_digest(res),
            'motion_input_columns_zero': True, 'sampler_motion_columns_zero': True,
            'offset_and_radius_outputs_zero': True, 'global_future_gates_zero': True,
            'world_trainable_parameters': tuned_count, 'world_all_modules_eval': True,
            'warm_start_checkpoint': None, 'optimizer_restored': False, 'scheduler_restored': False,
            'rolling_state': 'empty', 'prior_res_checkpoint_loaded': False}
