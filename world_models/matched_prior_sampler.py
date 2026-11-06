"""Same-time neighbourhood tokens from five retained (not extrapolated) points."""
import math

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_SAMPLER_CONFIG = dict(points_per_map=[36, 25, 16, 9, 1],
                              window_m=[6., 3.], offset_mode='learned')
RESIDUAL_WINDOW_FRACTION = .25


def canonical_sampler_config(config=None):
    result = dict(points_per_map=list(DEFAULT_SAMPLER_CONFIG['points_per_map']),
                  window_m=list(DEFAULT_SAMPLER_CONFIG['window_m']), offset_mode='learned')
    if config is not None:
        if set(config) != set(result):
            raise ValueError('v5 sampler configuration keys differ')
        result.update(config)
    points, window = result['points_per_map'], result['window_m']
    if len(points) != 5 or any(type(k) is not int or not 1 <= k <= 64 or
                              math.isqrt(k)**2 != k for k in points):
        raise ValueError('points_per_map must contain five square budgets in [1,64]')
    if len(window) != 2 or any(not math.isfinite(x) or x <= 0 for x in window):
        raise ValueError('window_m must contain positive finite tangent/normal half widths')
    if result['offset_mode'] not in ('learned', 'fixed'):
        raise ValueError('offset_mode must be learned or fixed')
    return dict(points_per_map=list(points), window_m=[float(x) for x in window],
                offset_mode=result['offset_mode'])


def retained_prior_axes(prior_lidar_deltas, forward_lidar):
    """Use only the five retained deltas, including when filling stopped points."""
    deltas = prior_lidar_deltas[:, :5]
    lengths = deltas.norm(dim=-1, keepdim=True)
    moving = lengths[..., 0] > 1e-3
    directions = deltas / lengths.clamp_min(1e-3)
    distance = (torch.arange(5, device=deltas.device)[:, None] -
                torch.arange(5, device=deltas.device)[None]).abs()
    indices = (distance[None] + (~moving[:, None]).long()*100).argmin(-1)
    neighbours = directions.gather(1, indices[..., None].expand(-1, -1, 2))
    fallback = F.normalize(forward_lidar, dim=-1, eps=1e-6)[:, None].expand(-1, 5, -1)
    tangent = torch.where(moving.any(1)[:, None, None], neighbours, fallback)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), -1)
    return tangent, normal


class MatchedPriorSampler(nn.Module):
    """One waypoint per same-time map; each bilinear sample remains a token.

    IRWM future map h is already translated by P_h, with current LiDAR axes.
    Consequently its matched waypoint centre is P_h - P_h = (0,0).
    Geometry and axes still use P_h/delta_h; the sixth prior/map are unread.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = canonical_sampler_config(config)
        self.num_tokens = sum(self.config['points_per_map'])
        self.geometry = nn.Sequential(nn.Linear(7, 256), nn.SiLU(), nn.Linear(256, 256))
        self.ego_projection = nn.Linear(256, 256)
        self.controller_norm = nn.LayerNorm(256)
        self.max_points = max(self.config['points_per_map'])
        self.offset = nn.Linear(256, self.max_points*2)
        self.value_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256))
        self.position = nn.Sequential(nn.Linear(6, 256), nn.SiLU(), nn.Linear(256, 256))
        # Keep actual 0.5..3.0 s IDs: retain IDs 0..4, never renumber /5.
        self.future_time = nn.Embedding(6, 256)
        self.waypoint_time = nn.Embedding(6, 256)
        self.output_norm = nn.LayerNorm(256)
        template = torch.zeros(5, self.max_points, 2)
        movable = torch.zeros(5, 1, 1)
        ids = []
        self.slices = []
        cursor = 0
        for h, k in enumerate(self.config['points_per_map']):
            side = math.isqrt(k)
            if k > 1:
                axis = torch.linspace(-.75, .75, side)
                longitudinal, lateral = torch.meshgrid(axis, axis, indexing='ij')
                template[h, :k] = torch.stack((longitudinal, lateral), -1).reshape(k, 2)
                movable[h] = 1
            ids.extend([h]*k)
            self.slices.append(slice(cursor, cursor+k))
            cursor += k
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.offset.requires_grad_(self.config['offset_mode'] == 'learned')
        self.register_buffer('initial_unit_offsets', template, persistent=False)
        self.register_buffer('movable', movable, persistent=False)
        self.register_buffer('map_ids', torch.tensor(ids), persistent=False)
        self.register_buffer('waypoint_ids', torch.tensor(ids), persistent=False)
        self.last_stats = {}

    def sample_features(self, future_bev, prior_lidar_deltas, prior_ego_deltas,
                        ego_context, forward_lidar, monitor=False):
        if future_bev.ndim == 4 and future_bev.shape[1] in (5, 6) and future_bev.shape[2:] == (40000, 256):
            maps = future_bev.view(-1, future_bev.shape[1], 200, 200, 256).permute(0, 1, 4, 2, 3)
        elif future_bev.ndim == 5 and future_bev.shape[1] in (5, 6) and future_bev.shape[2:] == (256, 200, 200):
            maps = future_bev
        else:
            raise ValueError('expected five or six 200x200 future BEVs with 256 channels')
        b = maps.shape[0]
        if prior_lidar_deltas.shape != (b, 6, 2) or prior_ego_deltas.shape != (b, 6, 2):
            raise ValueError('both priors must be [B,6,2]')
        # Slice before every geometry/direction calculation: no sixth-point fallback.
        lidar = prior_lidar_deltas[:, :5]
        ego = prior_ego_deltas[:, :5]
        positions, ego_positions = lidar.cumsum(1), ego.cumsum(1)
        ego_direction = F.normalize(ego, dim=-1, eps=1e-3)
        time = torch.arange(1, 6, device=maps.device, dtype=maps.dtype)[None, :, None]/6
        geometry = torch.cat((ego_positions/32, ego/ego.new_tensor([8.1, 3.]),
                              ego_direction, time.expand(b, -1, -1)), -1)
        control = self.controller_norm(self.geometry(geometry) + self.ego_projection(ego_context))
        residual = self.offset(control).view(b, 5, self.max_points, 2).tanh()
        if self.config['offset_mode'] == 'fixed':
            residual = residual*0
        # Grid spans 75% of half width; learnable displacement <= remaining 25%.
        # A singleton always stays at the exact matched centre, even after training.
        unit_offsets = self.initial_unit_offsets + RESIDUAL_WINDOW_FRACTION*residual*self.movable
        tangent, normal = retained_prior_axes(prior_lidar_deltas, forward_lidar)
        radii = maps.new_tensor(self.config['window_m'])
        self.last_stats = {}
        sampled, valid_all, coordinates, offsets_all, log_priors = [], [], [], [], []
        for h, k in enumerate(self.config['points_per_map']):
            local_offset = unit_offsets[:, h, :k]*radii
            metric_offset = (local_offset[..., :1]*tangent[:, h:h+1] +
                             local_offset[..., 1:]*normal[:, h:h+1])
            reference = positions[:, h:h+1]-positions[:, h:h+1]
            points = reference+metric_offset
            grid = points/51.2
            valid = (grid.abs() <= (1-1/200.)).all(-1) & torch.isfinite(grid).all(-1)
            safe_grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
            value = F.grid_sample(maps[:, h].float(), safe_grid[:, :, None].float(),
                                  align_corners=False, mode='bilinear', padding_mode='zeros')
            value = value.squeeze(-1).transpose(1, 2).masked_fill(~valid[..., None], 0)
            count = valid.sum(-1, keepdim=True).clamp_min(1)
            sampled.append(value)
            valid_all.append(valid)
            coordinates.append(points)
            offsets_all.append(local_offset)
            # Equal initial group mass despite different sampling densities.
            log_priors.append(-count.to(maps).log().expand(-1, k))
            if monitor:
                n = valid.numel()
                self.last_stats[f'sampler/valid_fraction/h{h+1}'] = torch.stack((valid.sum().float(), maps.new_tensor(n)))
                for axis, name in enumerate(('abs_tangent_m', 'abs_normal_m')):
                    self.last_stats[f'sampler/{name}/h{h+1}'] = torch.stack((local_offset[..., axis].abs().sum().detach(), maps.new_tensor(n)))
                saturation = (residual[:, h, :k].abs() > .95).float().sum()*self.movable[h, 0, 0]
                self.last_stats[f'sampler/offset_saturation/h{h+1}'] = torch.stack((saturation.detach(), maps.new_tensor(2*n)))
                drift = ((unit_offsets[:, h, :k]-self.initial_unit_offsets[h, :k])*radii).norm(dim=-1)
                self.last_stats[f'sampler/template_drift_m/h{h+1}'] = torch.stack((drift.sum().detach(), maps.new_tensor(n)))
        return dict(features=torch.cat(sampled, 1), valid=torch.cat(valid_all, 1),
                    coordinates=torch.cat(coordinates, 1), offsets=torch.cat(offsets_all, 1),
                    log_prior=torch.cat(log_priors, 1), geometry=geometry)

    def forward(self, future_bev, prior_lidar_deltas, prior_ego_deltas, ego_context,
                forward_lidar, monitor=False):
        points = self.sample_features(future_bev, prior_lidar_deltas, prior_ego_deltas,
                                      ego_context, forward_lidar, monitor)
        b = future_bev.shape[0]
        geometry = points['geometry'][:, self.waypoint_ids]
        time = (self.map_ids.to(geometry)+1)[None, :, None].expand(b, -1, -1)/6
        position = torch.cat((geometry[..., :2],
                              points['offsets']/points['offsets'].new_tensor(self.config['window_m']),
                              time, time), -1)
        tokens = self.output_norm(self.value_projection(points['features']) + self.position(position) +
                                  self.future_time(self.map_ids)[None] + self.waypoint_time(self.waypoint_ids)[None])
        tokens = tokens.masked_fill(~points['valid'][..., None], 0)
        if monitor:
            valid = points['valid']
            self.last_stats['sampler/valid_tokens_per_future_sample'] = torch.stack((valid.sum().float(), tokens.new_tensor(b)))
            self.last_stats['sampler/token_norm'] = torch.stack(((tokens.norm(dim=-1)*valid).sum().detach(), valid.sum().float()))
        return dict(tokens=tokens, valid=points['valid'], log_prior=points['log_prior'])
