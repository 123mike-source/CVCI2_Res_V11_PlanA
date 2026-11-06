"""V9: GeoBEV current planning conditions with the retained V8 rolling IRWM."""
from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F

from temporal_matched_deform_model import MatchedDeformTemporalModel
from world_models.rolling_state import RollingState
from cvci2_gate12.baseline_transfuser import agent_detection_loss
from . import GEOBEV_VERSION, GEOBEV_PROFILE, architecture_config
from .encoder import GeoBEVEncoder
from .losses import sparse_depth_loss, masked_occupancy_loss
from .augmentation import SceneColorAugmentation, apply_color, color_config

GEOBEV_SCOPE = 'geobev_camera_planner_v8_future_partial_irwm'
GEO_FIELDS = ('geobev_image', 'geobev_sensor2ego', 'geobev_intrinsics',
              'geobev_post_rots', 'geobev_post_trans', 'geobev_ego2global')


class GeoBEVRollingState(RollingState):
    def __init__(self, continuity_bounds=None):
        super().__init__(continuity_bounds=continuity_bounds)
        self.color = None
        self.augmentation_enabled = False

    def configure_color(self, config, seed):
        if self.slots:
            raise RuntimeError('Configure color augmentation before loading/consuming a stream')
        self.color = None if config is None else SceneColorAugmentation(config, seed)

    def reset(self):
        super().reset()
        if self.color is not None:
            self.color.reset()

    @staticmethod
    def _frame(batch, i):
        frame = RollingState._frame(batch, i)
        frame.update({k: batch[k][i].detach() for k in GEO_FIELDS})
        return frame

    def update(self, batch):
        if self.augmentation_enabled and self.color is not None:
            factors = self.color.prepare(batch['meta'])
            batch = dict(batch, geobev_image=torch.stack([
                apply_color(image, factor) for image, factor in zip(batch['geobev_image'], factors)]))
        history = super().update(batch)
        pairs = [slot['frames'][-2:] if len(slot['frames']) > 1 else slot['frames'] * 2 for slot in self.slots]
        for key in GEO_FIELDS:
            # Use the current differentiable image input and detached past observations.
            history[key] = torch.stack([torch.stack([f[key] for f in pair]) for pair in pairs])
        history['geobev_history_valid'] = torch.tensor(
            [len(s['frames']) > 1 for s in self.slots], device=batch['camera'].device)
        return history

    def state_dict(self):
        rolling = super().state_dict()
        if self.color is None:
            return rolling
        return dict(schema='geobev_scene_color_rolling_v1', rolling=rolling,
                    color=self.color.state_dict())

    def load_state_dict(self, saved, dataset, device):
        if isinstance(saved, list):
            if self.color is not None:
                raise ValueError('An unaugmented rolling state cannot resume augmented training')
            return super().load_state_dict(saved, dataset, device)
        if saved.get('schema') != 'geobev_scene_color_rolling_v1' or self.color is None:
            raise ValueError('Unknown augmented rolling checkpoint contract')
        self.color.load_state_dict(saved['color'])
        super().load_state_dict(saved['rolling'], dataset, device)
        if self.augmentation_enabled:
            if len(self.slots) != len(self.color.slots):
                raise ValueError('Rolling/color stream slot count differs')
            for slot, color in zip(self.slots, self.color.slots):
                for frame in slot['frames']:
                    if frame['meta']['scene_token'] != color['scene_token']:
                        raise ValueError('Color parameters belong to a different scene')
                    frame['geobev_image'] = apply_color(frame['geobev_image'], color['factors'])


class GeoBEVPlanningBase(nn.Module):
    def __init__(self, original, geobev_checkpoint, seed):
        super().__init__()
        retained = ('status', 'route', 'memory_pos', 'queries', 'decoder',
                    'agent_head', 'occupancy_head', 'planner', 'spatial_gate')
        for name in retained:
            setattr(self, name, getattr(original, name))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1909)
            self.geobev = GeoBEVEncoder(geobev_checkpoint)
            self.bridge = nn.Sequential(nn.Conv2d(256, 256, 1), nn.GroupNorm(32, 256), nn.SiLU())
            self.occupancy_projection = nn.Conv2d(256, 64, 1)

    def conditions_from_history(self, batch, history):
        feature, depth_logits, visible = self.geobev(history)
        feature = self.bridge(feature.to(batch['status'].dtype))
        low_tokens = F.adaptive_avg_pool2d(feature, (8, 8)).flatten(2).transpose(1, 2)
        canbus = self.status(batch['status'])[:, None]
        global_token = canbus + self.route(batch['command'])[:, None]
        memory = torch.cat((low_tokens, global_token), 1) + self.memory_pos.weight[None]
        queries = self.decoder(self.queries.weight[None].expand(len(feature), -1, -1), memory)
        occupancy = self.occupancy_projection(feature)
        spatial = self.spatial_gate(feature, batch['status'], batch['command'])
        return (queries[:, :1], queries[:, 1:], spatial, global_token, occupancy, canbus), depth_logits, visible


class GeoBEVTemporalModel(MatchedDeformTemporalModel):
    def __init__(self, *args, geobev_checkpoint=None, geobev_config=None, depth_loss_weight=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if self.architecture_version != 8 or self.world_tuning_config is None:
            raise ValueError('GeoBEV V9 requires integrated V8 motion, adaptive sampling and partial IRWM tuning')
        if not math.isfinite(depth_loss_weight) or depth_loss_weight < 0:
            raise ValueError('Depth weight must be finite and nonnegative')
        self.geobev_config = architecture_config(depth_loss_weight)
        if geobev_config is not None and geobev_config != self.geobev_config:
            raise ValueError('Unsupported GeoBEV geometry/checkpoint configuration')
        if geobev_checkpoint is None and kwargs.get('base_checkpoint', args[0] if args else None) is not None:
            geobev_checkpoint = Path(__file__).resolve().parents[2] / 'Models/geobev-r50-nuimage-cbgs.pth'
        self.base = GeoBEVPlanningBase(self.base, geobev_checkpoint, kwargs.get('initialization_seed', 2026))
        self.architecture_version = GEOBEV_VERSION
        from world_models.world_partial import full_world_tuning
        self.training_scope = 'geobev_camera_planner_full_irwm' if full_world_tuning(self.world_tuning_config) else GEOBEV_SCOPE
        self.stream = GeoBEVRollingState()
        self.color_augmentation_config = None
        self.initialization_profile = GEOBEV_PROFILE
        self.training_auxiliary_weights = (.2, .2, .1)
        self._geobev_losses = None
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(self.is_trainable(name))
        self.train(False)

    def configure_color_augmentation(self, probability, strength, seed, rank=0):
        config = color_config(probability, strength)
        self.stream.configure_color(config, int(seed) + 1000003 + 1009 * int(rank))
        self.color_augmentation_config = config
        return config

    def train(self, mode=True):
        super().train(mode)
        if isinstance(getattr(self, 'stream', None), GeoBEVRollingState):
            if self.stream.color is not None and self.stream.augmentation_enabled != bool(mode):
                self.stream.reset()
            self.stream.augmentation_enabled = bool(mode)
        return self

    def _build_conditions(self, batch, history, compute_loss):
        conditions, logits, visible = self.base.conditions_from_history(batch, history)
        self._geobev_losses = (logits, visible) if compute_loss else None
        return conditions

    def _auxiliary_losses(self, agents, occupancy_feature, batch):
        logits, visible = self._geobev_losses
        self._geobev_losses = None
        states, classes = self.base.agent_head(agents)
        agent_class, agent_box = agent_detection_loss(states, classes,
                                                      batch['agent_states'], batch['agent_labels'])
        occupancy, cells = masked_occupancy_loss(self.base.occupancy_head(occupancy_feature), batch['lidar'], visible)
        depth, points = sparse_depth_loss(logits, batch)
        weights = self.training_auxiliary_weights
        auxiliary = weights[0] * agent_class + weights[1] * agent_box + weights[2] * occupancy
        auxiliary = auxiliary + self.geobev_config['depth_loss_weight'] * depth
        self.last_monitor['geobev/occupancy_valid_cells'] = torch.stack((cells, cells.new_tensor(len(agents))))
        self.last_monitor['geobev/depth_valid_points'] = torch.stack((points, points.new_tensor(len(agents))))
        return dict(agent_class=agent_class, agent_box=agent_box, occupancy=occupancy,
                    depth=depth, auxiliary=auxiliary)
