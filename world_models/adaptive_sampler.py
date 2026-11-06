"""Motion-aware, centre-anchored two-scale readout at constant token cost."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from world_models.matched_prior_sampler import MatchedPriorSampler, retained_prior_axes

ADAPTIVE_SAMPLER_CONFIG = {
    'schema': 'center_multiscale_motion_v1',
    'points_per_map': [9, 9, 16, 25, 36],
    'window_m': [6., 3.],
    'offset_mode': 'learned',
    'horizon_window_scale': [.5, .625, .75, .875, 1.],
    'radius_multiplier_bounds': [.5, 1.5],
    'motion_feature_dim': 20,
    'centre': 'exact_zero_in_translated_future_map',
    'template': 'anchor_inner_quarter_outer_three_quarters',
}


def canonical_adaptive_sampler(config):
    if config != ADAPTIVE_SAMPLER_CONFIG:
        raise ValueError('Unsupported adaptive sampler contract')
    return {k: list(v) if isinstance(v, list) else v
            for k, v in ADAPTIVE_SAMPLER_CONFIG.items()}


class AdaptivePriorSampler(MatchedPriorSampler):
    def __init__(self, config):
        advanced = canonical_adaptive_sampler(config)
        super().__init__({k: advanced[k] for k in ('points_per_map', 'window_m', 'offset_mode')})
        self.config = advanced
        self.geometry[0] = nn.Linear(27, 256)
        # Motion enters the original controller. The last two outputs of its
        # existing offset layer learn the two window radii, without a new branch.
        self.offset = nn.Linear(256, self.max_points * 2 + 2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        template = torch.zeros_like(self.initial_unit_offsets)
        movable = torch.zeros(5, self.max_points, 1)
        for h, count in enumerate(self.config['points_per_map']):
            inner = max(4, (count - 1) // 3)
            outer = count - 1 - inner
            cursor = 1
            for n, radius, phase in ((inner, .25, 0.), (outer, .75, math.pi / max(outer, 1))):
                angles = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n) + phase
                template[h, cursor:cursor + n] = radius * torch.stack((angles.cos(), angles.sin()), -1)
                cursor += n
            movable[h, 1:count] = 1.
        self.initial_unit_offsets = template
        self.movable = movable

    def sample_features(self, future_bev, prior_lidar_deltas, prior_ego_deltas,
                        ego_context, forward_lidar, monitor=False, motion_features=None):
        if future_bev.ndim == 4 and tuple(future_bev.shape[2:]) == (40000, 256):
            maps = future_bev.view(-1, future_bev.shape[1], 200, 200, 256).permute(0, 1, 4, 2, 3)
        elif future_bev.ndim == 5 and tuple(future_bev.shape[2:]) == (256, 200, 200):
            maps = future_bev
        else:
            raise ValueError('Adaptive sampler requires full-resolution IRWM maps')
        batch = len(maps)
        if maps.shape[1] not in (5, 6) or motion_features is None or motion_features.shape != (batch, 6, 20):
            raise ValueError('Adaptive sampler requires five maps and observed [B,6,20] motion')
        if prior_lidar_deltas.shape != (batch, 6, 2) or prior_ego_deltas.shape != (batch, 6, 2):
            raise ValueError('Adaptive prior shape differs')
        ego = prior_ego_deltas[:, :5]
        positions = ego.cumsum(1)
        time = torch.arange(1, 6, device=maps.device, dtype=ego.dtype)[None, :, None] / 6.
        geometry = torch.cat((positions / 32., ego / ego.new_tensor([8.1, 3.]),
                              F.normalize(ego, dim=-1, eps=1e-3), time.expand(batch, -1, -1)), -1)
        inputs = torch.cat((geometry, motion_features[:, :5].to(geometry)), -1)
        control = self.controller_norm(self.geometry(inputs) + self.ego_projection(ego_context))
        raw = self.offset(control)
        residual = raw[..., :-2].reshape(batch, 5, self.max_points, 2).tanh()
        multiplier = 1. + .5 * raw[..., -2:].tanh()
        radii = multiplier * maps.new_tensor(self.config['window_m'])[None, None]
        radii = radii * maps.new_tensor(self.config['horizon_window_scale'])[None, :, None]
        self.last_window_radii = radii.detach()
        units = self.initial_unit_offsets + .25 * residual * self.movable
        tangent, normal = retained_prior_axes(prior_lidar_deltas, forward_lidar)
        self.last_stats = {}
        features, valids, coordinates, offsets, priors = [], [], [], [], []
        for h, count in enumerate(self.config['points_per_map']):
            local = units[:, h, :count] * radii[:, h:h + 1]
            points = local[..., :1] * tangent[:, h:h + 1] + local[..., 1:] * normal[:, h:h + 1]
            # The map is already translated by the predicted P_h. Its matched
            # centre is zero; adding P_h here would misalign it a second time.
            grid = points / 51.2
            valid = (grid.abs() <= 1 - 1 / 200.).all(-1) & torch.isfinite(grid).all(-1)
            safe = torch.where(valid[..., None], grid, torch.zeros_like(grid))
            values = F.grid_sample(maps[:, h].float(), safe[:, :, None].float(),
                                   align_corners=False, mode='bilinear', padding_mode='zeros')
            values = values.squeeze(-1).transpose(1, 2).masked_fill(~valid[..., None], 0.)
            features.append(values); valids.append(valid); coordinates.append(points); offsets.append(local)
            priors.append(-valid.sum(-1, keepdim=True).clamp_min(1).to(values).log().expand(-1, count))
            if monitor:
                n = valid.numel()
                self.last_stats['sampler/valid_fraction/h%d' % (h + 1)] = torch.stack((valid.sum().float(), maps.new_tensor(n)))
                for axis, name in enumerate(('abs_tangent_m', 'abs_normal_m')):
                    self.last_stats['sampler/%s/h%d' % (name, h + 1)] = torch.stack((local[..., axis].abs().sum().detach(), maps.new_tensor(n)))
                    self.last_stats['sampler/window_%s_m/h%d' % ('tangent' if axis == 0 else 'normal', h + 1)] = torch.stack((radii[:, h, axis].sum().detach(), maps.new_tensor(batch)))
                saturation = ((residual[:, h, :count].abs() > .95).float() * self.movable[h, :count]).sum()
                self.last_stats['sampler/offset_saturation/h%d' % (h + 1)] = torch.stack((saturation.detach(), maps.new_tensor(2 * n)))
                drift = ((units[:, h, :count] - self.initial_unit_offsets[h, :count]) * radii[:, h:h + 1]).norm(dim=-1)
                self.last_stats['sampler/template_drift_m/h%d' % (h + 1)] = torch.stack((drift.sum().detach(), maps.new_tensor(n)))
        return dict(features=torch.cat(features, 1), valid=torch.cat(valids, 1),
                    coordinates=torch.cat(coordinates, 1), offsets=torch.cat(offsets, 1),
                    log_prior=torch.cat(priors, 1), geometry=geometry, window_radii=radii)

    def forward(self, future_bev, prior_lidar_deltas, prior_ego_deltas, ego_context,
                forward_lidar, monitor=False, motion_features=None, lidar_to_ego_rotation=None):
        points = self.sample_features(future_bev, prior_lidar_deltas, prior_ego_deltas,
                                     ego_context, forward_lidar, monitor, motion_features)
        geometry = points['geometry'][:, self.waypoint_ids]
        time = (self.map_ids.to(geometry) + 1)[None, :, None].expand(len(future_bev), -1, -1) / 6.
        position = torch.cat((geometry[..., :2],
                              points['offsets'] / points['offsets'].new_tensor(self.config['window_m']),
                              time, time), -1)
        tokens = self.output_norm(self.value_projection(points['features']) + self.position(position)
                                  + self.future_time(self.map_ids)[None] + self.waypoint_time(self.waypoint_ids)[None])
        tokens = tokens.masked_fill(~points['valid'][..., None], 0.)
        if monitor:
            valid = points['valid']
            self.last_stats['sampler/valid_tokens_per_future_sample'] = torch.stack((valid.sum().float(), tokens.new_tensor(len(tokens))))
            self.last_stats['sampler/token_norm'] = torch.stack(((tokens.norm(dim=-1) * valid).sum().detach(), valid.sum().float()))
        result = dict(tokens=tokens, valid=points['valid'], log_prior=points['log_prior'])
        if getattr(self, 'return_candidate_geometry', False):
            from candidate_interaction import sampled_geometry
            result.update(sampled_geometry(points, prior_lidar_deltas, prior_ego_deltas,
                forward_lidar, lidar_to_ego_rotation, self.map_ids))
        return result
