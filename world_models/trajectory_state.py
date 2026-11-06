"""Receding-horizon trajectory state used by CVCI3 train and evaluation."""

from typing import Optional

import torch
from torch import Tensor


def shift_plan_to_current_frame(
    previous_deltas: Tensor,
    previous_t_global: Tensor,
    current_t_global: Tensor,
) -> Tensor:
    """Shift a previous six-delta plan into the current ego frame.

    ``previous_deltas`` are frame-to-frame deltas expressed in the previous
    ego/lidar frame. The first predicted point has already been executed by
    the current frame, so it is removed before extrapolating one replacement
    point. ``T_global`` maps lidar coordinates to the global frame.
    """
    if previous_deltas.ndim != 3 or previous_deltas.shape[1:] != (6, 2):
        raise ValueError(f"previous_deltas must be [B,6,2], got {tuple(previous_deltas.shape)}")
    if previous_t_global.shape != current_t_global.shape or previous_t_global.shape[-2:] != (4, 4):
        raise ValueError("previous_t_global and current_t_global must both be [B,4,4]")

    previous_positions = previous_deltas.cumsum(dim=1)
    retained = previous_positions[:, 1:]
    extrapolated = retained[:, -1:] + (retained[:, -1:] - retained[:, -2:-1])
    positions = torch.cat((retained, extrapolated), dim=1)

    homogeneous = torch.cat(
        (
            positions,
            torch.zeros_like(positions[..., :1]),
            torch.ones_like(positions[..., :1]),
        ),
        dim=-1,
    )
    current_from_previous = torch.linalg.inv(current_t_global).bmm(previous_t_global)
    current_positions = current_from_previous.bmm(homogeneous.transpose(1, 2)).transpose(1, 2)[..., :2]
    return torch.cat((current_positions[:, :1], current_positions[:, 1:] - current_positions[:, :-1]), dim=1)


def meta_transform(meta: dict, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Read a 4x4 lidar-to-global transform from an image metadata dict."""
    value = meta.get("T_global")
    if value is None:
        raise KeyError("img_metas must contain T_global for recurrent plan transforms")
    return torch.as_tensor(value, device=device, dtype=dtype)


def meta_scene(meta: dict) -> Optional[str]:
    """Return the explicit scene id when the dataset wrapper provides one."""
    value = meta.get("scene_token")
    return None if value is None else str(value)
