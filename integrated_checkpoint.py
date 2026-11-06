"""Audited migration into the integrated motion and adaptive readout model."""
import hashlib

import torch


def warm_start_integrated(model, path):
    from motion_gate import INTEGRATED_MOTION_GATE_CONFIG
    if model.motion_gate_config != INTEGRATED_MOTION_GATE_CONFIG:
        raise ValueError('Integrated migration requires the concatenated motion gate')
    saved = torch.load(str(path), map_location='cpu')
    meta = saved.get('metadata', {})
    version = meta.get('architecture_version')
    expected_format = {6: 'cvci2res_gate12_x0_rollout_v6', 7: 'cvci2res_gate12_x0_motion_gate_v7'}
    if version not in expected_format or saved.get('format') != expected_format[version]:
        raise ValueError('Warm start requires a complete v6/v7 joint checkpoint')
    if meta.get('training_scope') != 'all_cvci2_gate12_and_future_encoder_frozen_world':
        raise ValueError('Warm-start source scope differs')
    if meta.get('fusion_config') != model.fusion_config:
        raise ValueError('Warm-start fusion settings differ')
    if max(meta['sampler_config']['points_per_map']) != model.future_encoder.max_points:
        raise ValueError('Offset controller size must be explicitly migrated')
    source = saved['trainable']
    current = model.state_dict()
    inherited = {}
    consumed = set()
    slices = []
    for name, target in current.items():
        if not name.startswith(('base.', 'future_encoder.')):
            continue
        if name not in source:
            raise RuntimeError('Unexpected new tensor: ' + name)
        value = source[name]
        consumed.add(name)
        if name.endswith('.future_reliability.0.weight'):
            if value.shape != (128, 262) or target.shape != (128, 282):
                raise ValueError('Gate dimensions changed')
            result = torch.zeros_like(target, device='cpu')
            result[:, :262] = value
            old_projection = name.replace('future_reliability.0.weight', 'future_motion_projection.weight')
            if version == 7:
                result[:, 262:] = source[old_projection]
                consumed.add(old_projection)
            slices.append(name + '[:,262:] ' + ('folded_v7_projection' if version == 7 else 'zero_initialized'))
        elif name == 'future_encoder.geometry.0.weight' and target.shape != value.shape:
            if value.shape != (256, 7) or target.shape != (256, 27):
                raise ValueError('Sampler geometry dimensions changed')
            result = torch.zeros_like(target, device='cpu'); result[:, :7] = value
            slices.append(name + '[:,7:] zero_initialized')
        elif name in ('future_encoder.offset.weight', 'future_encoder.offset.bias') and target.shape != value.shape:
            if target.shape[0] != value.shape[0] + 2 or target.shape[1:] != value.shape[1:]:
                raise ValueError('Sampler output dimensions changed')
            result = torch.zeros_like(target, device='cpu'); result[:-2] = value
            slices.append(name + '[-2:] zero_initialized_window_radii')
        else:
            if value.shape != target.shape:
                raise ValueError('Unapproved shape migration: ' + name)
            result = value
        inherited[name] = result
    if consumed != set(source):
        raise RuntimeError('Unconsumed checkpoint tensors: ' + str(set(source) - consumed))
    model.load_state_dict(inherited, strict=False)
    actual = model.state_dict()
    for name, value in inherited.items():
        if not torch.equal(actual[name].detach().cpu(), value):
            raise RuntimeError('Migration verification failed: ' + name)
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return {'status': 'PASS', 'source': str(path), 'source_sha256': digest.hexdigest(),
            'source_epoch': saved.get('epoch'), 'source_architecture_version': version,
            'architecture_version': 8, 'source_tensor_count': len(source),
            'consumed_source_tensors': len(consumed), 'restored_or_expanded_tensors': len(inherited),
            'new_parameter_slices': slices, 'world_initialization': 'original_pretrained_IRWM',
            'world_trainable_parameter_count': sum(p.numel() for n, p in model.named_parameters()
                                                   if n.startswith('world.') and p.requires_grad),
            'sampler_configuration_change': {'from': meta['sampler_config'], 'to': model.sampler_config},
            'optimizer_restored': False, 'scheduler_restored': False, 'rolling_history': 'empty'}
