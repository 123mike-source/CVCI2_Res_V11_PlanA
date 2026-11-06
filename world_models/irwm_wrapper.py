"""Frozen IR-WM adapter used by CVCI3.

The wrapper deliberately calls the same WorldHeadV1 decoder weights six times
and returns the final decoder-layer BEV feature at each rollout step.  It does
not use future images, future occupancy, or future ground-truth pose.
"""

import copy
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
# Versioned runtime copied from CVCI3; importing it never imports its planner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "irwm_runtime"))
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


class FrozenIRWMFutureBEV(nn.Module):
    def __init__(
        self,
        drive_occworld_root: str,
        checkpoint: str,
        embed_dims: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        rollout_steps: int = 6,
    ):
        super().__init__()
        if rollout_steps != 6:
            raise ValueError("CVCI3 requires a six-step shared-decoder rollout")
        # CVCI3 vendors the plugin under an isolated namespace because both
        # upstream projects otherwise use the top-level package name
        # ``projects.mmdet3d_plugin`` in the same Python process.
        import irwm_projects.mmdet3d_plugin  # noqa: F401

        cfg_path = os.path.join(
            drive_occworld_root,
            "projects/configs/e2e/MMO_MSO_with_plan_tightly_coupled.py",
        )
        cfg = Config.fromfile(cfg_path)
        cfg.model.pretrained = None
        cfg.model.img_backbone.pretrained = None
        cfg.model.turn_on_plan = False
        cfg.model.future_pred_frame_num = rollout_steps
        cfg.model.test_future_frame_num = rollout_steps
        cfg.model.bev_h = bev_h
        cfg.model.bev_w = bev_w
        cfg.model.future_pred_head.history_queue_length = 2
        cfg.model.future_pred_head.memory_queue_len = 1
        self.model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        loaded = load_checkpoint(self.model, checkpoint, map_location="cpu", strict=False)
        source = loaded.get("state_dict", loaded.get("model", loaded))
        source = {key.removeprefix("module.") if hasattr(key, "removeprefix")
                  else (key[7:] if key.startswith("module.") else key): value
                  for key, value in source.items()}
        target = self.model.state_dict()
        allowed_missing = {"pts_bbox_head.transformer.reference_points.weight",
                           "pts_bbox_head.transformer.reference_points.bias"}
        missing = {key for key in target if key not in source
                   and key not in allowed_missing and not key.endswith("num_batches_tracked")}
        unexpected = {key for key in source if key not in target
                      and not key.startswith(("plan_head.", "planning_metric_v2."))}
        mismatched = {key for key in source.keys() & target.keys()
                      if source[key].shape != target[key].shape}
        if missing or unexpected or mismatched:
            raise RuntimeError(f"IRWM checkpoint mismatch: missing={missing}, unexpected={unexpected}, shape={mismatched}")
        del loaded, source, target
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.rollout_steps = rollout_steps
        self.register_buffer(
            "cvci_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "cvci_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "irwm_mean",
            torch.tensor([103.53, 116.28, 123.675]).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        # Parent CVCI3.train() must never switch the frozen world model back
        # to training mode (grid-mask/dropout/BN would otherwise change it).
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _convert_images(self, image_history: Tensor) -> Tensor:
        """CVCI1 RGB normalization -> IR-WM BGR/std=1 normalization."""
        raw_rgb = image_history * self.cvci_std.to(image_history) + self.cvci_mean.to(image_history)
        raw_bgr = raw_rgb[:, :, :, [2, 1, 0], :, :]
        return raw_bgr - self.irwm_mean.to(image_history)

    def _future_transforms(self, prior_deltas: Tensor) -> tuple:
        batch = prior_deltas.shape[0]
        cumulative = prior_deltas[:, :, :2].cumsum(dim=1)
        identity = torch.eye(4, device=prior_deltas.device, dtype=prior_deltas.dtype).view(1, 4, 4).repeat(batch, 1, 1)
        future2ref = [identity.clone()]
        ref2future = [identity.clone()]
        for index in range(self.rollout_steps):
            forward = identity.clone()
            # Drive-OccWorld stores coordinate transforms for row vectors.
            forward[:, 3, :2] = cumulative[:, index]
            inverse = torch.linalg.inv(forward)
            future2ref.append(forward)
            ref2future.append(inverse)
        return future2ref, ref2future

    @staticmethod
    def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a scalar-first quaternion."""
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = np.trace(matrix)
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                quaternion = np.array([
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ])
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                quaternion = np.array([
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ])
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                quaternion = np.array([
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ])
        return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)

    @staticmethod
    def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
        """Convert a scalar-first quaternion to a 3x3 rotation matrix."""
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0.0:
            raise ValueError("ego2global_rotation quaternion has zero norm")
        w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @classmethod
    def _motion_metas(
        cls, sample_history: List[dict], ego_status_history: np.ndarray = None
    ) -> List[dict]:
        """Build Drive-OccWorld history metadata using its pretrained contract."""
        transforms = [np.asarray(meta["T_global"], dtype=np.float64) for meta in sample_history]
        timestamps = [float(meta["timestamp"]) for meta in sample_history]
        if ego_status_history is not None and len(ego_status_history) != len(sample_history):
            raise ValueError("ego_status_history must have one row per history frame")
        output = []
        previous_ego_translation = None
        previous_timestamp = None
        previous_yaw = None
        previous_scene = None
        for frame_index, (meta, transform, timestamp) in enumerate(
            zip(sample_history, transforms, timestamps)
        ):
            base = copy.deepcopy(meta)
            lidar_rotation = transform[:3, :3]
            ego_translation = np.asarray(
                meta.get("ego2global_translation", transform[:3, 3]), dtype=np.float64
            )
            if "ego2global_rotation" in meta:
                ego_quaternion = np.asarray(meta["ego2global_rotation"], dtype=np.float32)
                ego_rotation = cls._quaternion_to_rotation(ego_quaternion)
            else:
                ego_rotation = lidar_rotation
                ego_quaternion = cls._rotation_to_quaternion(ego_rotation)
            yaw_degrees = np.degrees(
                np.arctan2(ego_rotation[1, 0], ego_rotation[0, 0])
            )
            can_bus = np.zeros(18, dtype=np.float32)
            can_bus[3:7] = ego_quaternion
            if ego_status_history is not None:
                can_bus[7:16] = np.asarray(
                    ego_status_history[frame_index][:9], dtype=np.float32
                )
            can_bus[-2] = np.radians(yaw_degrees)

            scene_token = meta.get("scene_token")
            continuous = (
                previous_timestamp is not None
                and (previous_scene is None or scene_token is None or scene_token == previous_scene)
                and timestamp > previous_timestamp
                and timestamp - previous_timestamp <= 1.0
            )
            if continuous:
                can_bus[:3] = ego_translation - previous_ego_translation
                can_bus[-1] = (yaw_degrees - previous_yaw + 180.0) % 360.0 - 180.0
            base["can_bus"] = can_bus
            base["lidar2global_rotation"] = lidar_rotation.astype(np.float32)
            base["prev_bev_exists"] = continuous
            output.append(base)
            previous_ego_translation = ego_translation
            previous_timestamp = timestamp
            previous_yaw = yaw_degrees
            previous_scene = scene_token
        return output

    def _prepare_metas(
        self,
        image_history: Tensor,
        projection: Tensor,
        meta_history: List[List[dict]],
        prior_deltas: Tensor,
        ego_status_history: Tensor = None,
    ) -> tuple:
        batch, frames, cameras, _, height, width = image_history.shape
        future2ref, ref2future = self._future_transforms(prior_deltas)
        current_metas = []
        history_metas = []
        for sample_index in range(batch):
            sample_ego_status = None
            if ego_status_history is not None:
                sample_ego_status = self._to_numpy(ego_status_history[sample_index])
            motion_metas = self._motion_metas(
                meta_history[sample_index], sample_ego_status
            )
            sample_history = []
            for frame_index in range(frames):
                base = motion_metas[frame_index]
                base["ref_lidar_to_cur_lidar"] = np.eye(4, dtype=np.float32)
                frame_projection = (projection[sample_index, frame_index]
                                    if projection.dim() == 5 else projection[sample_index])
                base["lidar2img"] = self._to_numpy(frame_projection)
                base["img_shape"] = [(height, width, 3) for _ in range(cameras)]
                base["ori_shape"] = [(height, width, 3) for _ in range(cameras)]
                base["pad_shape"] = [(height, width, 3) for _ in range(cameras)]
                base["scale_factor"] = np.ones(4, dtype=np.float32)
                base["future2ref_lidar_transform"] = [x[sample_index].detach().cpu().numpy() for x in future2ref]
                base["ref2future_lidar_transform"] = [x[sample_index].detach().cpu().numpy() for x in ref2future]
                sample_history.append(base)
            history_metas.append(sample_history)
            current_metas.append(sample_history[-1])
        return history_metas, current_metas

    @torch.no_grad()
    def forward(
        self,
        image_history: Tensor,
        projection: Tensor,
        meta_history: List[List[dict]],
        prior_deltas: Tensor,
        ego_status_history: Tensor = None,
    ) -> Tensor:
        if image_history.dim() != 6 or tuple(image_history.shape[1:3]) != (3, 6):
            raise ValueError(f"IR-WM expects image history [B,3,6,C,H,W], got {tuple(image_history.shape)}")
        if prior_deltas.shape[1:] != (6, 2):
            raise ValueError(f"prior_deltas must be [B,6,2], got {tuple(prior_deltas.shape)}")
        if ego_status_history is not None and ego_status_history.shape != (image_history.shape[0], 3, 10):
            raise ValueError(
                f"ego_status_history must be [B,3,10], got {tuple(ego_status_history.shape)}"
            )
        image_history = self._convert_images(image_history)
        history_metas, current_metas = self._prepare_metas(
            image_history, projection, meta_history, prior_deltas, ego_status_history
        )

        # Use the latest two history images to initialize the current BEV.  No
        # future image or future ground-truth pose is read here.
        previous_bev, _ = self.model.obtain_history_bev(image_history[:, :-1], history_metas)
        reference_bev = self.model.obtain_ref_bev(image_history[:, -1], current_metas, previous_bev)
        prev_bev_input = reference_bev.unsqueeze(1)
        ref_to_history = torch.eye(4, device=reference_bev.device, dtype=reference_bev.dtype).view(1, 1, 4, 4).repeat(reference_bev.shape[0], 1, 1, 1)
        condition_frames = self.rollout_steps + 1  # reference frame + six futures
        action_condition = {
            "command": torch.zeros(reference_bev.shape[0], condition_frames, 1, device=reference_bev.device, dtype=reference_bev.dtype),
            "vel_steering": torch.zeros(reference_bev.shape[0], condition_frames, 4, device=reference_bev.device, dtype=reference_bev.dtype),
        }
        cond_norm = {"occ_gts": None}
        future_bev = []
        readout_steps = getattr(self, 'readout_steps', self.rollout_steps)
        if readout_steps not in (5, 6):
            raise ValueError('readout_steps must retain five or six future maps')
        # Keep six-step conditioning metadata identical; only omit an unread
        # final decoder call when the caller explicitly requests five maps.
        for frame_index in range(1, readout_steps + 1):
            plan_traj = prior_deltas[:, :frame_index, :2]
            action_condition["plan_traj"] = plan_traj
            tgt_grids, aligned_prev_grids, ref2future, future_to_history = self.model._align_bev_coordnates(
                frame_index, ref_to_history, current_metas, plan_traj
            )
            cond_norm["future2history"] = future_to_history
            pred_feat, _ = self.model.future_pred_head(
                prev_bev_input,
                current_metas,
                frame_index,
                action_condition,
                cond_norm,
                tgt_points=tgt_grids,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                ref_points=aligned_prev_grids,
            )
            next_bev = prev_bev_input[:, -1] + pred_feat[-1]
            future_bev.append(next_bev)
            prev_bev_input = next_bev.unsqueeze(1)
            ref_to_history = ref2future.unsqueeze(1)
        return torch.stack(future_bev, dim=1)

    def _semantic_labels(self, decoder_layers: Tensor) -> Tensor:
        """Decode one BEV state with the checkpoint's occupancy head.

        ``decoder_layers`` retains all intermediate WorldHead decoder states;
        using only the last latent layer here would select the wrong auxiliary
        occupancy head.  The returned labels are arranged in the detector's
        conventional ``[B, H, W, Z]`` order.
        """
        logits = self.model.future_pred_head.forward_head(decoder_layers.unsqueeze(0))
        # [Lout=1, decoder_layer, B, HW, Z, class] -> [B, HW, Z]
        labels = logits[0, -1].argmax(dim=-1)
        batch, _, height = labels.shape
        labels = labels.view(batch, self.bev_w, self.bev_h, height).transpose(1, 2)
        return labels.contiguous()

    @torch.no_grad()
    def forward_with_semantics(
        self,
        image_history: Tensor,
        projection: Tensor,
        meta_history: List[List[dict]],
        prior_deltas: Tensor,
        ego_status_history: Tensor = None,
    ) -> tuple:
        """Return CVCI's six future features and semantic occupancy labels.

        This uses exactly the same autoregressive, action-conditioned rollout
        as :meth:`forward`, then applies the *trained* WorldHeadV1 occupancy
        decoder.  It is intended for qualitative rendering only and never
        reads future images, occupancies, poses, or annotations.
        """
        if image_history.dim() != 6 or tuple(image_history.shape[1:3]) != (3, 6):
            raise ValueError(f"IR-WM expects image history [B,3,6,C,H,W], got {tuple(image_history.shape)}")
        if prior_deltas.shape[1:] != (6, 2):
            raise ValueError(f"prior_deltas must be [B,6,2], got {tuple(prior_deltas.shape)}")
        if ego_status_history is not None and ego_status_history.shape != (image_history.shape[0], 3, 10):
            raise ValueError(
                f"ego_status_history must be [B,3,10], got {tuple(ego_status_history.shape)}"
            )

        image_history = self._convert_images(image_history)
        history_metas, current_metas = self._prepare_metas(
            image_history, projection, meta_history, prior_deltas, ego_status_history
        )
        previous_bev, _ = self.model.obtain_history_bev(image_history[:, :-1], history_metas)
        reference_bev = self.model.obtain_ref_bev(image_history[:, -1], current_metas, previous_bev)

        num_decoder_layers = len(self.model.future_pred_head.bev_pred_head)
        decoder_state = reference_bev.unsqueeze(0).repeat(num_decoder_layers, 1, 1, 1)
        semantic_states = [self._semantic_labels(decoder_state)]
        prev_bev_input = reference_bev.unsqueeze(1)
        ref_to_history = torch.eye(
            4, device=reference_bev.device, dtype=reference_bev.dtype
        ).view(1, 1, 4, 4).repeat(reference_bev.shape[0], 1, 1, 1)
        condition_frames = self.rollout_steps + 1
        action_condition = {
            "command": torch.zeros(reference_bev.shape[0], condition_frames, 1, device=reference_bev.device, dtype=reference_bev.dtype),
            "vel_steering": torch.zeros(reference_bev.shape[0], condition_frames, 4, device=reference_bev.device, dtype=reference_bev.dtype),
        }
        cond_norm = {"occ_gts": None}
        future_bev = []
        for frame_index in range(1, self.rollout_steps + 1):
            plan_traj = prior_deltas[:, :frame_index, :2]
            action_condition["plan_traj"] = plan_traj
            tgt_grids, aligned_prev_grids, ref2future, future_to_history = self.model._align_bev_coordnates(
                frame_index, ref_to_history, current_metas, plan_traj
            )
            cond_norm["future2history"] = future_to_history
            pred_feat, _ = self.model.future_pred_head(
                prev_bev_input,
                current_metas,
                frame_index,
                action_condition,
                cond_norm,
                tgt_points=tgt_grids,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                ref_points=aligned_prev_grids,
            )
            decoder_state = decoder_state + pred_feat
            future_bev.append(decoder_state[-1])
            semantic_states.append(self._semantic_labels(decoder_state))
            prev_bev_input = decoder_state[-1].unsqueeze(1)
            ref_to_history = ref2future.unsqueeze(1)
        return torch.stack(future_bev, dim=1), torch.stack(semantic_states, dim=1)
