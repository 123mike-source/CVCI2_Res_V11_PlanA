"""Explicit V9 checkpoint dispatch and original-pretraining audit."""
import hashlib
from pathlib import Path
import torch

from . import GEOBEV_FORMAT, GEOBEV_VERSION, GEOBEV_PROFILE


def audit_fresh_geobev(model, base_path, world_path, seed):
    from fresh_v8_initialization import (file_digest, state_digest, EXPECTED_BASE_SHA256, EXPECTED_WORLD_SHA256)
    if model.architecture_version != GEOBEV_VERSION:
        raise ValueError('Fresh GeoBEV audit requires V9')
    assert file_digest(base_path) == EXPECTED_BASE_SHA256
    assert file_digest(world_path) == EXPECTED_WORLD_SHA256
    source = torch.load(str(base_path), map_location='cpu')['model']
    current = model.base.state_dict()
    skipped = ('backbone.', 'bev_down.', 'dense_proj.')
    retained = {n: v for n, v in source.items() if not n.startswith(skipped)}
    assert all(n in current and torch.equal(current[n].cpu(), v) for n, v in retained.items())
    assert not any(n.startswith(skipped) for n in current)
    original = torch.load(str(world_path), map_location='cpu')
    world = original.get('state_dict', original.get('model', original))
    world = {n[7:] if n.startswith('module.') else n: v for n, v in world.items()}
    actual = model.world.model.state_dict()
    allowed = {'pts_bbox_head.transformer.reference_points.weight', 'pts_bbox_head.transformer.reference_points.bias'}
    assert all(n in world or n in allowed or n.endswith('num_batches_tracked') for n in actual)
    assert all(torch.equal(v.cpu(), world[n]) for n, v in actual.items() if n in world)
    from world_models.world_partial import full_world_tuning
    world_parameters = sum(p.numel() for p in model.world.parameters() if p.requires_grad)
    expected = sum(p.numel() for p in model.world.parameters()) if full_world_tuning(model.world_tuning_config) else 907712
    assert world_parameters == expected
    assert not any(m.training for m in model.world.modules())
    assert model.base.geobev.initialization_audit['status'] == 'PASS'
    state = model.state_dict()
    additions = {n: v for n, v in state.items() if n.startswith('future_encoder.') or
                 (n.startswith('base.planner.') and '.future_' in n) or
                 n.startswith(('base.bridge.', 'base.occupancy_projection.'))}
    assert all(float(layer.future_gate) == 0 for layer in model.base.planner.layers if hasattr(layer, 'future_gate'))
    return dict(status='PASS', profile=GEOBEV_PROFILE, seed=seed,
                retained_cvci2_tensors_verified=len(retained),
                discarded_cvci2_prefixes=list(skipped), geobev=model.base.geobev.initialization_audit,
                original_irwm_tensors_verified=sum(n in world for n in actual),
                world_trainable_parameters=world_parameters, fresh_res_state_sha256=state_digest(additions),
                optimizer_restored=False, scheduler_restored=False, prior_res_checkpoint_loaded=False,
                lidar_feature_input=False)


def build(checkpoint, *args, **kwargs):
    from .model import GeoBEVTemporalModel
    meta = checkpoint.get('metadata', {})
    if checkpoint.get('format') != GEOBEV_FORMAT or meta.get('architecture_version') != GEOBEV_VERSION:
        raise ValueError('GeoBEV checkpoint format/version differs')
    required = ('geobev_config', 'fusion_config', 'sampler_config', 'initialization_seed',
                'world_tuning_config', 'motion_gate_config')
    if any(k not in meta for k in required):
        raise ValueError('Incomplete GeoBEV checkpoint metadata')
    args = list(args)
    if args:
        args[0] = None
    else:
        kwargs['base_checkpoint'] = None
    requested_world = bool(kwargs.get('enable_world', True))
    kwargs['enable_world'] = True  # Preserve tuned tensors even with its forward disabled.
    model = GeoBEVTemporalModel(*args, geobev_checkpoint=None, geobev_config=meta['geobev_config'],
        depth_loss_weight=meta['geobev_config']['depth_loss_weight'],
        fusion_config=meta['fusion_config'], sampler_config=meta['sampler_config'],
        initialization_seed=meta['initialization_seed'], motion_gate_config=meta['motion_gate_config'],
        world_tuning_config=meta['world_tuning_config'], **kwargs)
    model.enable_world = requested_world
    from training_optimization import restore_training_conditioning
    restore_training_conditioning(model, meta)
    return model


def load(model, checkpoint):
    from temporal_matched_deform_model import load_matched_deform_state
    if checkpoint.get('format') != GEOBEV_FORMAT:
        raise ValueError('GeoBEV checkpoint required')
    meta = checkpoint['metadata']
    from training_optimization import check_training_conditioning
    check_training_conditioning(model, meta)
    for key in ('training_scope', 'fusion_config', 'sampler_config', 'motion_gate_config',
                'world_tuning_config', 'geobev_config'):
        if meta.get(key) != getattr(model, key):
            raise ValueError('GeoBEV checkpoint contract differs: ' + key)
    load_matched_deform_state(model, checkpoint['trainable'], GEOBEV_VERSION)


def metadata(model):
    weights = model.training_auxiliary_weights
    from training_optimization import training_conditioning_metadata
    conditioning = training_conditioning_metadata(model)
    return dict(bev_encoder='geobev', geobev_config=model.geobev_config,
                input_modalities=['six_cameras', 'observed_can_bus', 'navigation'],
                lidar_role='training_supervision_only', world_bev_encoder='pretrained_irwm_bevformer',
                motion_injection='concatenated_into_existing_reliability_linear',
                sampler_pattern=model.sampler_config['schema'], training_scope=model.training_scope,
                world_tuning_config=model.world_tuning_config,
                world_trainable_parameters=sum(p.numel() for p in model.world.parameters() if p.requires_grad),
                world_rollout_steps=5,
                auxiliary_loss_weights=dict(agent_class=weights[0], agent_box=weights[1], occupancy=weights[2],
                                            depth=model.geobev_config['depth_loss_weight']),
                joint_pretrained_modules=['GeoBEV six core encoder modules', 'retained CVCI2 query heads and DiT',
                                          ('all registered IRWM parameters' if model.world_tuning_config.get('parameter_scope') == 'all_registered_irwm_parameters' else 'IRWM last future decoder and action fusion only')],
                fresh_trainable_modules=['base.bridge.*', 'base.occupancy_projection.*',
                                         'base.planner.layers.*.future_*', 'future_encoder.*'],
                occupancy_supervision='current_ego_xy_64m_64x64_visible_masked_bce_dice',
                depth_supervision='current_lidar_zbuffer_valid_bins_only_equal_two_scale_ce',
                training_conditioning=conditioning,
                trajectory_loss=conditioning['x0_weighting'] + '_12_layer_masked_x0_mse',
                loss_definition=conditioning['x0_weighting'] + '_12_layer_x0_mse_plus_configured_class_box_masked_occupancy_depth')
