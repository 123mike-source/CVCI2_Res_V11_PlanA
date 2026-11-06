"""Pretrained IRWM with a narrowly scoped, FP32 planning gradient path."""
import copy

import torch
from torch.utils.checkpoint import checkpoint

from world_models.irwm_wrapper import FrozenIRWMFutureBEV

WORLD_TUNING_CONFIG = {
    'schema': 'last_decoder_and_action_fusion_v1',
    'parameter_prefixes': [
        'model.future_pred_head.transformer.decoder.layers.2.',
        'model.future_pred_head.fusion_mlp.',
    ],
    'parameter_count': 907712,
    'gradient_rollout': 'full_five_step_bptt',
    'checkpoint': 'nonreentrant_future_head',
    'module_mode': 'eval_with_gradients',
    'precision': 'fp32',
}
PARTIAL_WORLD_SCOPE = 'all_cvci2_future_and_last_irwm_decoder_action_fusion'
FULL_WORLD_TUNING_CONFIG = {
    'schema': 'all_irwm_observed_history_and_rollout_v1',
    'parameter_prefixes': ['model.'],
    'parameter_scope': 'all_registered_irwm_parameters',
    'gradient_history': 'all_three_observed_frames',
    'gradient_rollout': 'full_five_step_bptt',
    'checkpoint': 'nonreentrant_image_bev_and_future_head',
    'module_mode': 'eval_with_gradients',
    'precision': 'fp32',
}
FULL_WORLD_SCOPE = 'all_cvci2_future_and_full_irwm'


def full_world_tuning(config):
    return config is not None and config.get('schema') == FULL_WORLD_TUNING_CONFIG['schema']


def canonical_world_tuning(config):
    if config is None:
        return None
    if config not in (WORLD_TUNING_CONFIG, FULL_WORLD_TUNING_CONFIG):
        raise ValueError('Unsupported IRWM tuning contract')
    return copy.deepcopy(config)


def world_state_selected(name, config):
    return config is not None and any(name.startswith('world.' + p)
                                     for p in config['parameter_prefixes'])


class PartialIRWMFutureBEV(FrozenIRWMFutureBEV):
    def __init__(self, *args, tuning_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tuning_config = canonical_world_tuning(tuning_config)
        if self.tuning_config is None:
            raise ValueError('Partial IRWM needs an explicit tuning scope')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(any(name.startswith(p) for p in
                                         self.tuning_config['parameter_prefixes']))
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        expected = sum(p.numel() for p in self.parameters()) if full_world_tuning(self.tuning_config) else self.tuning_config['parameter_count']
        if count != expected:
            raise RuntimeError('Pretrained IRWM scope changed: ' + str(count))
        self.model.eval()
        self.readout_steps = 5

    def _encode_observations(self, images, history_metas, current_metas):
        with torch.no_grad():
            previous, _ = self.model.obtain_history_bev(images[:, :-1], history_metas)
            self.model.eval()
            return self.model.obtain_ref_bev(images[:, -1], current_metas, previous)

    def forward(self, image_history, projection, meta_history, prior_deltas,
                ego_status_history=None):
        if image_history.ndim != 6 or tuple(image_history.shape[1:3]) != (3, 6):
            raise ValueError('IRWM expects three observed six-camera frames')
        if prior_deltas.shape[1:] != (6, 2):
            raise ValueError('IRWM requires six predicted prior deltas')
        if ego_status_history is not None and ego_status_history.shape != (len(image_history), 3, 10):
            raise ValueError('IRWM motion history shape differs')
        self.model.eval()
        images = self._convert_images(image_history)
        history_metas, current_metas = self._prepare_metas(
            images, projection, meta_history, prior_deltas, ego_status_history)
        reference = self._encode_observations(images, history_metas, current_metas)
        del images, history_metas
        # Partial mode's reference is detached by its encoder. Full mode keeps
        # the gradient into the backbone and all three observed BEV frames.
        prev_input = reference.unsqueeze(1)
        ref_history = torch.eye(4, device=reference.device, dtype=reference.dtype).view(1, 1, 4, 4).repeat(len(reference), 1, 1, 1)
        action = {
            'command': reference.new_zeros(len(reference), self.rollout_steps + 1, 1),
            'vel_steering': reference.new_zeros(len(reference), self.rollout_steps + 1, 4),
        }
        steps = getattr(self, 'readout_steps', 5)
        if steps != 5:
            raise ValueError('This tuning contract uses the five consumed future maps')
        futures = []
        for frame_index in range(1, steps + 1):
            plan = prior_deltas[:, :frame_index, :2].detach()
            target, aligned, ref_future, future_history = self.model._align_bev_coordnates(
                frame_index, ref_history, current_metas, plan)
            # Bind all step-dependent values: checkpoint recomputation happens
            # after the loop and must never read a later step's dictionaries.
            step_action = dict(action, plan_traj=plan)
            step_condition = {'occ_gts': None, 'future2history': future_history}

            def decode(memory, target_points, aligned_points, frame=frame_index,
                       action_values=step_action, conditions=step_condition):
                result, _ = self.model.future_pred_head(
                    memory, current_metas, frame, action_values, conditions,
                    tgt_points=target_points, bev_h=self.bev_h, bev_w=self.bev_w,
                    ref_points=aligned_points)
                return result[-1]

            if torch.is_grad_enabled() and getattr(self, 'checkpoint_future', True):
                update = checkpoint(decode, prev_input, target, aligned,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                update = decode(prev_input, target, aligned)
            next_bev = prev_input[:, -1] + update
            futures.append(next_bev)
            # Keep the gradient through all rollout steps; no temporal detach.
            prev_input = next_bev.unsqueeze(1)
            ref_history = ref_future.unsqueeze(1)
        return torch.stack(futures, 1)
