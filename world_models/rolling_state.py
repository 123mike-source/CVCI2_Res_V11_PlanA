"""Detached rolling predictions, strict 0.5-second continuity and resume state."""
import copy
import hashlib

import numpy as np
import torch

from .trajectory_state import shift_plan_to_current_frame

TRAJECTORY_TRANSFORM_VERSION = 'inverse_projected_xy_v2'


def positions_to_deltas(positions):
    return torch.cat((positions[:, :1], positions[:, 1:] - positions[:, :-1]), dim=1)


def ego_positions_to_lidar_deltas(positions, lidar_to_ego):
    deltas = positions_to_deltas(positions)
    # Invert exactly the projected XY map used by the dataset targets.
    # With roll/pitch, its transpose is not its inverse.
    rotation_xy = lidar_to_ego[:, :2, :2].double()
    return torch.linalg.solve(rotation_xy, deltas.double().transpose(1, 2)).transpose(1, 2).float()


def prior_points_in_future_frame(prior_deltas):
    # IRWM predicts on future grids translated by cumulative prior (no yaw).
    # The ego point at horizon k is therefore the origin of future map k.
    positions = prior_deltas.cumsum(1)
    return positions - positions


def sample_noise(metas, seed, device):
    result = []
    for meta in metas:
        digest = hashlib.sha256(f'{seed}:{meta["token"]}:planner'.encode()).digest()
        generator = torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8], 'little') % (2**63 - 1))
        result.append(torch.randn(6, 2, generator=generator))
    return torch.stack(result).to(device)


class RollingState:
    def __init__(self, continuity_bounds=None):
        self.slots = []
        self.continuity_bounds = continuity_bounds

    def reset(self):
        self.slots = []

    @staticmethod
    def _frame(batch, i):
        return {'image': batch['world_image'][i].detach(),
                'projection': batch['projection'][i].detach(),
                'status': batch['world_status'][i].detach(),
                'meta': copy.deepcopy(batch['meta'][i]),
                'index': int(batch['data_index'][i])}

    def update(self, batch):
        size = len(batch['meta'])
        if len(self.slots) != size:
            self.slots = [{'frames': [], 'prediction': None} for _ in range(size)]
        prior = batch['world_image'].new_zeros(size, 6, 2)
        ready, valid, images, projections, statuses, metas = [], [], [], [], [], []
        for i, slot in enumerate(self.slots):
            current = self._frame(batch, i)
            history = slot['frames']
            previous = history[-1] if history else None
            continuous = False
            if previous is not None and current['meta']['scene_token'] == previous['meta']['scene_token']:
                dt = current['meta']['timestamp'] - previous['meta']['timestamp']
                continuous = (abs(dt - 0.5) <= 0.1 if self.continuity_bounds is None else
                              self.continuity_bounds[0] <= dt <= self.continuity_bounds[1])
            if not continuous:
                history = []
                slot['prediction'] = None
            history = (history + [current])[-3:]
            slot['frames'] = history
            is_ready = len(history) == 3
            has_prior = is_ready and slot['prediction'] is not None
            if has_prior:
                previous_t = torch.as_tensor(previous['meta']['T_global'], device=prior.device, dtype=torch.float64)
                current_t = torch.as_tensor(current['meta']['T_global'], device=prior.device, dtype=torch.float64)
                prior[i] = shift_plan_to_current_frame(slot['prediction'][None].double(),
                                                      previous_t[None], current_t[None])[0].float()
            padded = [history[0]] * (3 - len(history)) + history
            ready.append(is_ready)
            valid.append(has_prior)
            images.append(torch.stack([f['image'] for f in padded]))
            projections.append(torch.stack([f['projection'] for f in padded]))
            statuses.append(torch.stack([f['status'] for f in padded]))
            metas.append([f['meta'] for f in padded])
        return {'images': torch.stack(images), 'projections': torch.stack(projections),
                'statuses': torch.stack(statuses), 'metas': metas, 'prior': prior,
                'ready': torch.tensor(ready, device=prior.device),
                'valid': torch.tensor(valid, device=prior.device)}

    def set_prediction(self, lidar_deltas, ready):
        for i, slot in enumerate(self.slots):
            if bool(ready[i]):
                slot['prediction'] = lidar_deltas[i].detach().clone()

    def state_dict(self):
        return [{'indices': [f['index'] for f in s['frames']],
                 'prediction': None if s['prediction'] is None else s['prediction'].cpu()}
                for s in self.slots]

    def load_state_dict(self, saved, dataset, device):
        from temporal_dataset import temporal_collate, to_device
        self.slots = []
        for slot in saved:
            frames = []
            for index in slot['indices']:
                batch = to_device(temporal_collate([dataset[index]]), device)
                frames.append(self._frame(batch, 0))
            self.slots.append({'frames': frames, 'prediction': None if slot['prediction'] is None
                               else slot['prediction'].to(device)})
