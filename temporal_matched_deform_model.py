"""Five same-time future BEV neighbourhoods condition a six-point diffusion planner."""
import time
import torch
from torch import nn

from planning_runtime import DEFAULT_BASE,DEFAULT_WORLD,DEFAULT_WORLD_ROOT
from matched_diffusion import MatchedX0Planner,MatchedDiTLayer,DEFAULT_FUSION_CONFIG,base_monitoring_keys,_stat
from cvci2_gate12 import Gated12Model
from cvci2_gate12.baseline_transfuser import agent_detection_loss, occupancy_loss
from world_models.matched_prior_sampler import MatchedPriorSampler,canonical_sampler_config,retained_prior_axes
from world_models.rolling_state import RollingState,ego_positions_to_lidar_deltas,sample_noise

from motion_gate import (canonical_motion_gate_config, observed_motion_features,
                         MOTION_GATE_VERSION, MOTION_MONITOR_KEYS, INTEGRATED_MOTION_GATE_CONFIG)

MATCHED_DEFORM_VERSION=6
MATCHED_DEFORM_FORMAT='cvci2res_gate12_x0_rollout_v6'
MATCHED_DEFORM_SCOPE='all_cvci2_gate12_and_future_encoder_frozen_world'
MATCHED_DEFORM_INITIALIZATION='pretrained_cvci2_gate12_joint_finetune_frozen_world'

def is_trainable_state(name, model=None):
    if name.startswith(('base.','future_encoder.')): return True
    config = getattr(model, 'world_tuning_config', None)
    return config is not None and any(name.startswith('world.'+p) for p in config['parameter_prefixes'])

def matched_deform_state_dict(model):
    return {n:x.detach().cpu().clone() for n,x in model.state_dict().items() if is_trainable_state(n, model)}

def load_matched_deform_state(model, state, architecture_version=MATCHED_DEFORM_VERSION):
    expected_version = model.architecture_version
    if architecture_version != expected_version or not isinstance(model, MatchedDeformTemporalModel):
        raise ValueError('gate12 x0 checkpoint requires the v6 model factory')
    expected = {n for n in model.state_dict() if is_trainable_state(n, model)}
    if set(state) != expected:
        raise RuntimeError(f'gate12 x0 state keys differ: missing={expected-set(state)}, extra={set(state)-expected}')
    model.load_state_dict(state, strict=False)

def monitoring_keys():
    keys = set(base_monitoring_keys())
    keys.update(f'planning/prior_l2_m/h{h+1}' for h in range(6))
    keys.update(f'planning/outside_window_fraction/h{h+1}' for h in range(5))
    keys.update(('planning/retained_prior_l2_m', 'planning/extrapolated_prior_l2_m'))
    keys.update('interaction/'+branch+'/'+metric for branch in ('agent','ego','bev')
                for metric in ('gate','amplify_fraction','raw_residual_ratio','gated_residual_ratio'))
    keys.update('plan_a/'+metric for metric in ('position_loss_m','position_loss_coefficient','position_t99_samples','output_saturation_fraction'))
    return sorted(keys)

def initialize_future_extensions(model, seed):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
        model.base.planner = MatchedX0Planner(model.base.planner, model.motion_gate_config)
        if model.sampler_config.get('schema') == 'center_multiscale_motion_v1':
            from world_models.adaptive_sampler import AdaptivePriorSampler
            model.future_encoder = AdaptivePriorSampler(model.sampler_config)
        else:
            model.future_encoder = MatchedPriorSampler(model.sampler_config)
        if model.architecture_version == 8:
            from fresh_v8_initialization import initialize_stable_fresh_columns
            initialize_stable_fresh_columns(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(model.is_trainable(name))


class MatchedDeformTemporalModel(nn.Module):
    def __init__(self, base_checkpoint=DEFAULT_BASE, world_checkpoint=DEFAULT_WORLD,
                 world_root=DEFAULT_WORLD_ROOT, enable_world=True, seed=2026, world=None,
                 fusion_config=None, sampler_config=None, initialization_seed=2026, motion_gate_config=None, world_tuning_config=None):
        super().__init__()
        self.fusion_config = dict(DEFAULT_FUSION_CONFIG)
        if fusion_config is not None:
            if set(fusion_config) != set(self.fusion_config):
                raise ValueError('gate12 fusion configuration keys differ')
            self.fusion_config.update(fusion_config)
        if not 0 <= self.fusion_config['future_dropout'] <= 1:
            raise ValueError('invalid future_dropout')
        self.motion_gate_config = canonical_motion_gate_config(motion_gate_config)
        self.last_motion_features = None
        self.architecture_version = (8 if self.motion_gate_config == INTEGRATED_MOTION_GATE_CONFIG else
                                     7 if self.motion_gate_config is not None else 6)
        self.world_tuning_config = None
        self.training_scope = MATCHED_DEFORM_SCOPE
        if world_tuning_config is not None:
            from world_models.world_partial import canonical_world_tuning, PARTIAL_WORLD_SCOPE
            self.world_tuning_config = canonical_world_tuning(world_tuning_config)
            from world_models.world_partial import full_world_tuning, FULL_WORLD_SCOPE
            self.training_scope = FULL_WORLD_SCOPE if full_world_tuning(self.world_tuning_config) else PARTIAL_WORLD_SCOPE
        if sampler_config is not None and sampler_config.get('schema') == 'center_multiscale_motion_v1':
            from world_models.adaptive_sampler import canonical_adaptive_sampler
            self.sampler_config = canonical_adaptive_sampler(sampler_config)
            if self.architecture_version != 8: raise ValueError('Adaptive sampler requires integrated motion')
        else:
            self.sampler_config = canonical_sampler_config(sampler_config)
        self.base = Gated12Model()
        self.base_epoch = None
        if base_checkpoint is not None:
            source = torch.load(str(base_checkpoint), map_location='cpu')
            self.base.load_state_dict(source.get('model', source), strict=True)
            self.base_epoch = source.get('epoch')
            del source
        self.world = world
        if enable_world and world is None:
            from world_models.irwm_wrapper import FrozenIRWMFutureBEV
            if self.world_tuning_config is None:
                self.world = FrozenIRWMFutureBEV(str(world_root), str(world_checkpoint))
            else:
                from world_models.world_partial import PartialIRWMFutureBEV
                from world_models.world_partial import full_world_tuning
                world_type = PartialIRWMFutureBEV
                if full_world_tuning(self.world_tuning_config):
                    from world_models.world_full import FullIRWMFutureBEV
                    world_type = FullIRWMFutureBEV
                self.world = world_type(str(world_root), str(world_checkpoint), tuning_config=self.world_tuning_config)
        self.enable_world, self.seed = enable_world, seed
        self.world_checkpoint = str(world_checkpoint)
        self.stream = RollingState()
        self.last_monitor = {}
        initialize_future_extensions(self, initialization_seed)
        self.train(False)

    def is_trainable(self, name):
        if not is_trainable_state(name, self):
            return False
        if self.sampler_config['offset_mode'] == 'fixed' and name.startswith(tuple(
                'future_encoder.'+x for x in ('geometry.', 'ego_projection.', 'controller_norm.', 'offset.'))):
            return False
        return True

    def train(self, mode=True):
        super().train(mode)
        self.base.train(mode)
        if self.world is not None:
            self.world.eval()
        if hasattr(self, 'future_encoder'):
            self.future_encoder.train(mode)
            self.base.planner.train(mode)
        return self

    def _build_conditions(self, batch, history, compute_loss):
        return self.base.conditions(batch['camera'], batch['lidar'], batch['status'], batch['command'])

    def _auxiliary_losses(self, agents, occupancy_feature, batch):
        states, logits = self.base.agent_head(agents)
        agent_class, agent_box = agent_detection_loss(states, logits, batch['agent_states'], batch['agent_labels'])
        occupancy = occupancy_loss(self.base.occupancy_head(occupancy_feature), batch['lidar'])
        weights = getattr(self, 'training_auxiliary_weights', (.2, .2, .1))
        return dict(agent_class=agent_class, agent_box=agent_box, occupancy=occupancy,
                    depth=occupancy * 0,
                    auxiliary=weights[0]*agent_class + weights[1]*agent_box + weights[2]*occupancy)

    def reset(self):
        self.stream.reset()

    def forward(self, batch, compute_loss=True, disable_future=False, profile=False,
                initial_noise=None, ddim_steps=1, timestep_indices=None, skip_warmup=False,
                monitor=False):
        timings = {}
        self.last_monitor = {}
        self.future_encoder.last_stats = {}
        device = batch['camera'].device
        def mark():
            if (profile or monitor) and device.type == 'cuda':
                torch.cuda.synchronize(device)
            return time.perf_counter()
        start = mark()
        with torch.autocast(device_type=device.type, enabled=False):
            history = self.stream.update(batch)
        if skip_warmup and not compute_loss and not bool(history['ready'].any()):
            return dict(loss=None, prediction=None, ready=history['ready'], prior_valid=history['valid'], timings={})
        ego, agents, dense, status, occupancy_feature, canbus = self._build_conditions(batch, history, compute_loss)
        timings['conditions'] = mark()-start
        condition_mask = history['valid'] & (not disable_future) & self.enable_world
        if self.training and self.fusion_config['future_dropout']:
            condition_mask = condition_mask & (torch.rand(condition_mask.shape, device=device) >= self.fusion_config['future_dropout'])
        indices = condition_mask.nonzero(as_tuple=False).flatten()
        with torch.autocast(device_type=device.type, enabled=False):
            rotation = batch['lidar_to_ego'][:, :2, :2]
            prior_ego = torch.einsum('bij,bkj->bki', rotation.float(), history['prior'].float())
            prior_positions = prior_ego.cumsum(1)
            forward_ego = torch.tensor([1.,0.],device=device,dtype=torch.float32)[None].expand(len(ego),-1)
            forward_lidar = torch.linalg.solve(rotation.double(), forward_ego.double()[...,None]).squeeze(-1).float()
        motion_features = None
        self.last_motion_features = None
        if self.motion_gate_config is not None:
            with torch.autocast(device_type=device.type, enabled=False):
                motion_features, motion_valid = observed_motion_features(
                    history['statuses'], history['metas'], prior_ego, condition_mask)
            self.last_motion_features = motion_features
            if monitor:
                self.last_monitor['motion/input_valid_fraction'] = _stat(
                    motion_valid.sum().float(), condition_mask.sum().float())
        n = self.future_encoder.num_tokens
        future = dict(tokens=torch.zeros(len(ego), n, 256, device=device, dtype=torch.float32),
                      valid=torch.zeros(len(ego), n, dtype=torch.bool, device=device),
                      log_prior=torch.zeros(len(ego), n, device=device, dtype=torch.float32),
                      map_ids=self.future_encoder.map_ids, waypoint_ids=self.future_encoder.waypoint_ids)
        if getattr(self, 'candidate_config', None) and self.candidate_config['future_geometry']:
            from candidate_interaction import empty_geometry
            future.update(empty_geometry(len(ego), n, device))
        start = mark()
        if len(indices):
            # Keep frozen IRWM/custom CUDA operators on their original FP32 contract.
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.world_tuning_config is not None), torch.autocast(device_type=device.type, enabled=False):
                features = self.world(history['images'][indices], history['projections'][indices],
                                      [history['metas'][int(i)] for i in indices], history['prior'][indices], history['statuses'][indices])
            timings['world'] = mark()-start
            start = mark()
            extra = {'motion_features': motion_features[indices]} if self.sampler_config.get('schema') else {}
            if getattr(self, 'candidate_config', None) and self.candidate_config['future_geometry']:
                extra['lidar_to_ego_rotation'] = rotation[indices]
            selected = self.future_encoder(features, history['prior'][indices], prior_ego[indices],
                                            ego[indices], forward_lidar[indices], monitor=monitor, **extra)
            for key in selected:
                future[key] = future[key].index_copy(0, indices, selected[key].to(future[key].dtype))
            del features
        else:
            timings['world'] = 0.
        timings['future_encoder'] = mark()-start
        effective = condition_mask & future['valid'].any(1)
        terms = None
        start = mark()
        self.base.planner.monitor = monitor
        if compute_loss:
            planning_mask = batch['mask']
            if not getattr(self, 'training_cold_start_planning', False):
                planning_mask = planning_mask * history['ready'][:, None]
            terms = self.base.planner.loss(batch['trajectory_delta'], planning_mask,
                                          ego, agents, dense, status, future, effective, prior_positions, return_details=True, canbus=canbus, motion_features=motion_features)
            if monitor:
                supervised = planning_mask.bool().any(-1)
                self.last_monitor['planning/cold_start_supervised_fraction'] = _stat(
                    (supervised & ~history['ready']).sum().float(), len(supervised))
                self.last_monitor['planning/ready_supervised_fraction'] = _stat(
                    (supervised & history['ready']).sum().float(), len(supervised))
            if monitor and 'position' in terms:
                self.last_monitor['plan_a/position_loss_m'] = _stat(terms['position'], 1)
                self.last_monitor['plan_a/position_loss_coefficient'] = _stat(terms['position_coefficient'], 1)
                self.last_monitor['plan_a/position_t99_samples'] = _stat(terms['position_samples'].float(), 1)
            terms.update(self._auxiliary_losses(agents, occupancy_feature, batch))
            auxiliary = terms['auxiliary']
            terms['total'] = terms['total'] + auxiliary
            # Future branches can be inactive during cold start or dropout on any rank.
            terms['total'] = terms['total'] + sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        self.base.planner.monitor = False
        timings['diffusion_loss'] = mark()-start
        if monitor:
            self.last_monitor.update(self.future_encoder.last_stats)
            for i, layer in enumerate(self.base.planner.layers):
                if isinstance(layer, MatchedDiTLayer):
                    for key, pair in layer.last_stats.items():
                        self.last_monitor[key] = self.last_monitor.get(key, torch.zeros_like(pair))+pair
                    self.last_monitor[f'fusion/global_gate/layer_{i}'] = _stat(layer.future_gate.tanh(), 1)
        start = mark()
        with torch.no_grad():
            detached = {k:v.detach() for k,v in future.items()}
            prediction = self.base.planner.sample(len(ego), ego, agents, dense, status,
                                                  future_tokens=detached, future_mask=effective, canbus=canbus,
                                                  future_prior_positions=prior_positions, motion_features=motion_features,
                                                  initial_noise=sample_noise(batch['meta'], self.seed, device) if initial_noise is None else initial_noise,
                                                  inference_steps=ddim_steps, timestep_indices=timestep_indices)
            with torch.autocast(device_type=device.type, enabled=False):
                lidar_delta = ego_positions_to_lidar_deltas(prediction.float(), batch['lidar_to_ego'])
                self.stream.set_prediction(lidar_delta, history['ready'])
            if monitor:
                tangent, normal = retained_prior_axes(history['prior'], forward_lidar)
                gap = lidar_delta.cumsum(1)-history['prior'].cumsum(1)
                sampled_gap = gap[:, :5]
                widths = sampled_gap.new_tensor(self.sampler_config['window_m'])[None,None].expand(len(ego),5,2)
                coverage_valid = history['valid'].float()
                if self.sampler_config.get('schema'):
                    widths = widths.clone()
                    if len(indices): widths[indices] = self.future_encoder.last_window_radii
                    coverage_valid = effective.float()
                outside = ((sampled_gap*tangent).sum(-1).abs() > widths[...,0]) | ((sampled_gap*normal).sum(-1).abs() > widths[...,1])
                valid = history['valid'].float()
                self.last_monitor['planning/outside_window_fraction'] = _stat((outside*coverage_valid[:, None]).sum(), coverage_valid.sum()*5)
                for h in range(5):
                    self.last_monitor[f'planning/outside_window_fraction/h{h+1}'] = _stat((outside[:, h]*coverage_valid).sum(), coverage_valid.sum())
                for h in range(6):
                    self.last_monitor[f'planning/prior_gap_m/h{h+1}'] = _stat((gap[:, h].norm(dim=-1)*valid).sum(), valid.sum())
                target = batch['trajectory_delta'].cumsum(1)
                prior_error = (prior_positions-target).norm(dim=-1)
                prior_mask = batch['mask']*valid[:, None]
                for h in range(6):
                    self.last_monitor[f'planning/prior_l2_m/h{h+1}'] = _stat((prior_error[:, h]*prior_mask[:, h]).sum(), prior_mask[:, h].sum())
                self.last_monitor['planning/retained_prior_l2_m'] = _stat((prior_error[:, :5]*prior_mask[:, :5]).sum(), prior_mask[:, :5].sum())
                self.last_monitor['planning/extrapolated_prior_l2_m'] = _stat((prior_error[:, 5]*prior_mask[:, 5]).sum(), prior_mask[:, 5].sum())
                for sec, h in enumerate((1, 3, 5), 1):
                    m = batch['mask'][:, h]*history['ready']
                    self.last_monitor[f'planning/train_batch_ddim_l2_m/{sec}s'] = _stat(((prediction[:, h]-target[:, h]).norm(dim=-1)*m).sum(), m.sum())
                    m = m*history['valid']
                    self.last_monitor[f'planning/train_batch_prior_l2_m/{sec}s'] = _stat(((prior_positions[:, h]-target[:, h]).norm(dim=-1)*m).sum(), m.sum())
        timings['ddim_sample'] = mark()-start
        if monitor:
            for key, value in timings.items():
                self.last_monitor[f'timing/{key}_seconds'] = dense.new_tensor([value, 1.])
        return dict(loss_position=None if terms is None else terms.get('position', terms['x0']*0),
                    position_loss_coefficient=None if terms is None else terms.get('position_coefficient', terms['x0'].new_zeros(())),
                    loss=None if terms is None else terms['total'],
                    loss_diagnostics=None if terms is None else terms.get('diagnostics'),
                    loss_diffusion=None if terms is None else terms['diffusion'],
                    loss_x0=None if terms is None else terms['x0'],
                    loss_x0_unweighted=None if terms is None else terms['x0_unweighted'],
                    loss_auxiliary=None if terms is None else terms['auxiliary'],
                    loss_agent_class=None if terms is None else terms['agent_class'],
                    loss_agent_box=None if terms is None else terms['agent_box'],
                    loss_depth=None if terms is None else terms['depth'],
                    loss_occupancy=None if terms is None else terms['occupancy'], prediction=prediction,
                    ready=history['ready'], prior_valid=history['valid'], future_used=effective,
                    prior=history['prior'], timings=timings)

