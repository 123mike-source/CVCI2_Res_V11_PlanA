"""Full IRWM gradients through three observed frames and five future steps."""
import copy

import torch
from torch.utils.checkpoint import checkpoint

from world_models.world_partial import PartialIRWMFutureBEV


class FullIRWMFutureBEV(PartialIRWMFutureBEV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tuning_config['schema'] != 'all_irwm_observed_history_and_rollout_v1':
            raise ValueError('Full IRWM requires the full tuning contract')
        # MMCV ResNet.train(False) still invokes _freeze_stages(). Disable its
        # original stage-freezing policy before any parent train/eval calls.
        self.model.img_backbone.frozen_stages = -1
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        assert all(p.requires_grad for p in self.parameters())

    def _encode_observations(self, images, history_metas, current_metas):
        # Inference uses the original implementation, including its batching.
        if not torch.is_grad_enabled():
            return super()._encode_observations(images, history_metas, current_metas)
        self.model.eval()
        batch, length, cameras, channels, height, width = images[:, :-1].shape
        history_metas = copy.deepcopy(history_metas)
        extraction_metas = [meta[length - 1] for meta in history_metas]

        def extract_history(queue):
            # The vendor extractor squeezes its argument in-place at batch=1.
            # Clone to avoid mutating checkpoint inputs or the observed queue.
            queue = queue.reshape(batch * length, cameras, channels, height, width).clone()
            return tuple(self.model.extract_feat(img=queue, len_queue=length,
                                                img_metas=copy.deepcopy(extraction_metas)))

        enabled = getattr(self, 'checkpoint_future', True)

        def execute(function, *args):
            if enabled:
                return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)
            return function(*args)

        features = execute(extract_history, images[:, :-1])
        previous = None
        for frame_index in range(length):
            metas = [each[frame_index] for each in history_metas]
            if not metas[0]['prev_bev_exists']:
                previous = None
            frame_features = [each[:, frame_index] for each in features]
            count = len(frame_features)
            has_previous = previous is not None

            def encode_history(*values, metadata=copy.deepcopy(metas),
                               feature_count=count, use_previous=has_previous):
                prior = values[feature_count].clone() if use_previous else None
                return self.model.pts_bbox_head(list(values[:feature_count]),
                    copy.deepcopy(metadata), prior, only_bev=True)

            inputs = tuple(frame_features) + ((previous,) if has_previous else ())
            previous = execute(encode_history, *inputs)

        has_previous = previous is not None

        def encode_current(image, *prior, metadata=copy.deepcopy(current_metas),
                           use_previous=has_previous):
            return self.model.obtain_ref_bev(image.clone(), copy.deepcopy(metadata),
                                             prior[0].clone() if use_previous else None)

        inputs = (images[:, -1],) + ((previous,) if has_previous else ())
        return execute(encode_current, *inputs)
