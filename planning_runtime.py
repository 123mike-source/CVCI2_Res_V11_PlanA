"""Current checkpoint factory, shared paths and frozen original baseline."""
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

from cvci2_gate12 import Gated12Model
from matched_diffusion import MatchedX0Planner
from world_models.rolling_state import RollingState, ego_positions_to_lidar_deltas, sample_noise

ROOT=Path(__file__).resolve().parent
DEFAULT_BASE=ROOT.parent/'Models/best_endpoint_1step.pth'
DEFAULT_WORLD_ROOT=ROOT/'world_configs'
DEFAULT_WORLD=ROOT.parent/'Models/MMO_MSO_with_plan_tightly_coupled.pth'


def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def is_base_future_state(name):
    return name.startswith('planner.layers.') and any('.'+part in name for part in
        ('future_attn.','future_norm.','future_gate'))


def source_fingerprints(root):
    excluded = {'ENV', '.python', '.env-tools', '.env-cache', '.git', 'vendor',
                'backups', 'cache', 'experiments', 'results', '__pycache__', 'logs'}
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in excluded]
        files.extend(Path(directory)/name for name in names if name.endswith('.py'))
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(files)}


def synchronize_base_buffers(model):
    """Use rank-zero perception BN buffers for checkpoints and exact resumption."""
    import torch.distributed as dist
    if dist.is_initialized():
        for value in model.base.buffers():
            dist.broadcast(value, src=0)


class BaselineTemporalModel(nn.Module):
    """Latest CVCI2 gate12 x0 planner with the retained three-frame evaluation protocol."""
    def __init__(self,base_checkpoint=DEFAULT_BASE,world_checkpoint=DEFAULT_WORLD,
                 world_root=DEFAULT_WORLD_ROOT,enable_world=False,seed=2026,world=None):
        super().__init__()
        if enable_world or world is not None:raise ValueError('baseline has no world-model branch')
        self.base=Gated12Model()
        source=torch.load(str(base_checkpoint),map_location='cpu')
        self.base.load_state_dict(source.get('model',source),strict=True)
        self.base.planner=MatchedX0Planner(self.base.planner)
        del source
        self.stream=RollingState();self.seed=seed
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.eval()

    def reset(self):self.stream.reset()

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self,batch,compute_loss=True,disable_future=True,profile=False,
                initial_noise=None,ddim_steps=1,timestep_indices=None,skip_warmup=False):
        device=batch['camera'].device
        def mark():
            if profile and device.type=='cuda':torch.cuda.synchronize(device)
            return time.perf_counter()
        started=mark();history=self.stream.update(batch)
        if skip_warmup and not compute_loss and not bool(history['ready'].any()):
            return dict(loss=None,prediction=None,ready=history['ready'],prior_valid=history['valid'],
                        prior=history['prior'],timings={})
        ego,agents,dense,status,_,canbus=self.base.conditions(batch['camera'],batch['lidar'],batch['status'],batch['command'])
        timings={'prepare_conditions':mark()-started}
        started=mark();loss=None
        if compute_loss:
            loss=self.base.planner.loss(batch['trajectory_delta'],batch['mask']*history['ready'][:,None],
                                        ego,agents,dense,status,canbus=canbus)
        timings['diffusion_loss']=mark()-started;started=mark()
        prediction=self.base.planner.sample(len(ego),ego,agents,dense,status,
            initial_noise=sample_noise(batch['meta'],self.seed,device) if initial_noise is None else initial_noise,
            inference_steps=ddim_steps,timestep_indices=timestep_indices,canbus=canbus)
        self.stream.set_prediction(ego_positions_to_lidar_deltas(prediction,batch['lidar_to_ego']),history['ready'])
        timings['ddim_sample']=mark()-started
        return dict(loss=loss,prediction=prediction,ready=history['ready'],prior_valid=history['valid'],
                    prior=history['prior'],timings=timings)


def build_temporal_model(checkpoint=None,*args,**kwargs):
    if checkpoint is None:
        kwargs.setdefault('enable_world',False)
        return BaselineTemporalModel(*args,**kwargs)
    from temporal_matched_deform_model import (MatchedDeformTemporalModel,
        MATCHED_DEFORM_FORMAT, MATCHED_DEFORM_VERSION)
    from motion_gate import MOTION_GATE_FORMAT, MOTION_GATE_VERSION, canonical_motion_gate_config
    meta=checkpoint.get('metadata',{})
    fmt=checkpoint.get('format')
    expected={MATCHED_DEFORM_FORMAT:MATCHED_DEFORM_VERSION, MOTION_GATE_FORMAT:MOTION_GATE_VERSION}
    if fmt not in expected or meta.get('architecture_version')!=expected[fmt]:
        raise ValueError('Unsupported checkpoint architecture/format pair')
    if any(key not in meta for key in ('fusion_config','sampler_config','initialization_seed')):
        raise ValueError('Incomplete current checkpoint metadata')
    motion=canonical_motion_gate_config(meta.get('motion_gate_config'))
    if (fmt==MOTION_GATE_FORMAT)!=(motion is not None):
        raise ValueError('Checkpoint motion gate contract differs from its version')
    args=list(args)
    if args:args[0]=None
    else:kwargs['base_checkpoint']=None
    return MatchedDeformTemporalModel(*args,fusion_config=meta['fusion_config'],
        sampler_config=meta['sampler_config'],initialization_seed=meta['initialization_seed'],
        motion_gate_config=motion,**kwargs)


def load_planning_checkpoint(model,checkpoint):
    from temporal_matched_deform_model import (load_matched_deform_state,MATCHED_DEFORM_FORMAT,MATCHED_DEFORM_SCOPE)
    from motion_gate import MOTION_GATE_FORMAT, canonical_motion_gate_config
    meta=checkpoint.get('metadata',{})
    expected=MOTION_GATE_FORMAT if model.motion_gate_config is not None else MATCHED_DEFORM_FORMAT
    if checkpoint.get('format')!=expected:raise ValueError('Checkpoint format differs from model')
    if meta.get('training_scope')!=MATCHED_DEFORM_SCOPE:raise RuntimeError('training scope mismatch')
    if meta.get('fusion_config')!=model.fusion_config:raise RuntimeError('fusion configuration mismatch')
    if meta.get('sampler_config')!=model.sampler_config:raise RuntimeError('sampler configuration mismatch')
    if canonical_motion_gate_config(meta.get('motion_gate_config'))!=model.motion_gate_config:
        raise RuntimeError('motion gate configuration mismatch')
    load_matched_deform_state(model,checkpoint['trainable'],meta.get('architecture_version'))


def planning_model_metadata(model):
    architecture = dict(attention_backend=getattr(model.base.planner,'attention_backend','automatic'), diffusion_depth=12, self_attention_layers=6, interaction_layers=6,
                        prediction_type='x0', trajectory_loss='equal_12_layer_masked_x0_mse',
                        current_interaction_order='agent_ego_gated_bev', direct_canbus_concat=True,
                        current_bev_gate='independent_canbus_navigation_se',
                        auxiliary_loss_weights=dict(agent_class=.2,agent_box=.2,occupancy=.1))
    if hasattr(model,'future_encoder'):
        return dict(**architecture,spatial_encoding=True,motion_gate_config=model.motion_gate_config,sampler_config=model.sampler_config,
            future_condition_tokens=model.future_encoder.num_tokens,global_future_tokens=0,qformer_compression=False,
            sampling_alignment='same_time_diagonal',sampled_future_horizons_s=[.5,1.,1.5,2.,2.5],
            sampled_extrapolated_waypoint=False,sampled_future_map_count=5,world_rollout_steps=6,
            planner_output_points=6,singleton_sampling='fixed_matched_center' if 1 in model.sampler_config['points_per_map'] else 'not_applicable')
    return dict(**architecture,spatial_encoding=False,sampler_config=None,future_condition_tokens=0,
                global_future_tokens=0,qformer_compression=False)

# Syntax checked only; appended to planning_runtime by installer.

# Integrated v8 dispatch; legacy v6/v7 checkpoints retain their own factory.
_legacy_build_temporal_model_v7 = build_temporal_model
_legacy_load_planning_checkpoint_v7 = load_planning_checkpoint
_legacy_planning_model_metadata_v7 = planning_model_metadata


def build_temporal_model(checkpoint=None, *args, **kwargs):
    from motion_gate import INTEGRATED_MOTION_GATE_CONFIG, INTEGRATED_MOTION_FORMAT
    if checkpoint is None or checkpoint.get('format') != INTEGRATED_MOTION_FORMAT:
        return _legacy_build_temporal_model_v7(checkpoint, *args, **kwargs)
    from temporal_matched_deform_model import MatchedDeformTemporalModel
    meta = checkpoint.get('metadata', {})
    if meta.get('architecture_version') != 8 or meta.get('motion_gate_config') != INTEGRATED_MOTION_GATE_CONFIG:
        raise ValueError('Integrated checkpoint architecture differs')
    for key in ('fusion_config', 'sampler_config', 'initialization_seed', 'world_tuning_config'):
        if key not in meta:
            raise ValueError('Missing integrated metadata: ' + key)
    args = list(args)
    if args:
        args[0] = None
    else:
        kwargs['base_checkpoint'] = None
    requested_world = bool(kwargs.get('enable_world', True))
    # Tuned world tensors remain part of the checkpoint even when its forward is disabled.
    if meta['world_tuning_config'] is not None:
        kwargs['enable_world'] = True
    model = MatchedDeformTemporalModel(*args, fusion_config=meta['fusion_config'],
        sampler_config=meta['sampler_config'], initialization_seed=meta['initialization_seed'],
        motion_gate_config=meta['motion_gate_config'], world_tuning_config=meta['world_tuning_config'], **kwargs)
    model.enable_world = requested_world
    return model


def load_planning_checkpoint(model, checkpoint):
    from motion_gate import INTEGRATED_MOTION_GATE_CONFIG, INTEGRATED_MOTION_FORMAT
    if checkpoint.get('format') != INTEGRATED_MOTION_FORMAT:
        return _legacy_load_planning_checkpoint_v7(model, checkpoint)
    from temporal_matched_deform_model import load_matched_deform_state
    meta = checkpoint.get('metadata', {})
    for key in ('training_scope', 'fusion_config', 'sampler_config', 'motion_gate_config', 'world_tuning_config'):
        if meta.get(key) != getattr(model, key):
            raise ValueError('Integrated checkpoint differs: ' + key)
    if model.motion_gate_config != INTEGRATED_MOTION_GATE_CONFIG:
        raise ValueError('Integrated checkpoint requires concatenated motion inputs')
    load_matched_deform_state(model, checkpoint['trainable'], meta['architecture_version'])


def planning_model_metadata(model):
    result = _legacy_planning_model_metadata_v7(model)
    if getattr(model, 'architecture_version', None) == 8:
        result.update(motion_injection='concatenated_into_existing_reliability_linear',
            world_tuning_config=model.world_tuning_config, training_scope=model.training_scope,
            sampler_pattern=model.sampler_config.get('schema', 'legacy_grid'),
            world_trainable_parameters=sum(p.numel() for n, p in model.named_parameters()
                                           if n.startswith('world.') and p.requires_grad))
    return result


# V9 GeoBEV has its own geometry and checkpoint contract; old factories are retained.
_before_geobev_build = build_temporal_model
_before_geobev_load = load_planning_checkpoint
_before_geobev_metadata = planning_model_metadata


def build_temporal_model(checkpoint=None, *args, **kwargs):
    from geobev_encoder import GEOBEV_FORMAT
    from plan_a import PLAN_A_FORMAT
    from candidate_interaction import CANDIDATE_FORMAT
    if checkpoint is not None and checkpoint.get('format') == PLAN_A_FORMAT:
        from plan_a import build
        return build(checkpoint, *args, **kwargs)
    if checkpoint is not None and checkpoint.get('format') == CANDIDATE_FORMAT:
        from candidate_interaction import build
        return build(checkpoint, *args, **kwargs)
    if checkpoint is not None and checkpoint.get('format') == GEOBEV_FORMAT:
        from geobev_encoder.checkpoint import build
        return build(checkpoint, *args, **kwargs)
    return _before_geobev_build(checkpoint, *args, **kwargs)


def load_planning_checkpoint(model, checkpoint):
    from geobev_encoder import GEOBEV_FORMAT
    from plan_a import PLAN_A_FORMAT
    from candidate_interaction import CANDIDATE_FORMAT
    if checkpoint.get('format') == PLAN_A_FORMAT:
        from plan_a import load
        return load(model, checkpoint)
    if checkpoint.get('format') == CANDIDATE_FORMAT:
        from candidate_interaction import load
        return load(model, checkpoint)
    if checkpoint.get('format') == GEOBEV_FORMAT:
        from geobev_encoder.checkpoint import load
        return load(model, checkpoint)
    return _before_geobev_load(model, checkpoint)


def planning_model_metadata(model):
    result = _before_geobev_metadata(model)
    if getattr(model, 'architecture_version', None) == 11:
        from plan_a import metadata
        result.update(metadata(model))
    elif getattr(model, 'architecture_version', None) == 10:
        from candidate_interaction import metadata
        result.update(metadata(model))
    elif getattr(model, 'architecture_version', None) == 9:
        from geobev_encoder.checkpoint import metadata
        result.update(metadata(model))
    return result
