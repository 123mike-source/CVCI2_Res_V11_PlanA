"""Interaction-only ablation: agent -> ego -> spatial sampling, preserving state."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .baseline_transfuser import DiTLayer
from .xpred_canbus import NuScenesXPredCanbus


class AgentEgoSpatialLayer(DiTLayer):
    def forward(self, hidden, positions, dense_bev, agents, ego, time, status):
        if not self.interaction:
            return super().forward(hidden, positions, dense_bev, agents, ego, time, status)
        # Norm indices remain attached to their original branch: spatial=0,
        # agent=1, ego=2, FFN=3. Only execution order changes.
        hidden = self.norms[1](hidden + self.agent_attn(hidden, agents, agents, need_weights=False)[0])
        hidden = self.norms[2](hidden + self.ego_attn(hidden, ego, ego, need_weights=False)[0])
        grid = torch.stack((positions[..., 1]/32.0, positions[..., 0]/32.0), dim=-1)
        value = self.bev_value(dense_bev)
        sampled = F.grid_sample(value.float(), grid[:, :, None].float(), mode='bilinear',
                                padding_mode='zeros', align_corners=False).to(hidden.dtype)
        hidden = self.norms[0](hidden + sampled.squeeze(-1).transpose(1, 2))
        hidden = self.norms[3](hidden + self.ffn(hidden))
        scale, shift = self.modulation(torch.cat((time, status), dim=-1)).chunk(2, dim=-1)
        return hidden*(1.0+scale)+shift


class AgentEgoSpatialModel(NuScenesXPredCanbus):
    def __init__(self, image_checkpoint=None):
        super().__init__(image_checkpoint)
        # Preserve parameter values, names and post-initialization RNG state.
        # Self-attention blocks retain the original class and object.
        with torch.random.fork_rng(devices=[]):
            for index, layer in enumerate(self.planner.layers):
                if layer.interaction:
                    replacement = AgentEgoSpatialLayer(True)
                    replacement.load_state_dict(layer.state_dict(), strict=True)
                    self.planner.layers[index] = replacement
