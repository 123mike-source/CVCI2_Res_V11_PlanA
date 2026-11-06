"""Twelve-block x-pred planner; conditional gates affect spatial sampling only."""
import torch
from torch import nn

from .order_model import AgentEgoSpatialModel


class SELayerMLP(nn.Module):
    """ResWorld resworld_head.py:64-77, with channels broadcast over H,W."""
    def __init__(self, channels=256):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = nn.ReLU()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = nn.Sigmoid()

    def weights(self, condition):
        return self.gate(self.mlp_expand(self.act1(self.mlp_reduce(condition))))

    def forward(self, feature, condition):
        return feature * self.weights(condition)[:, :, None, None]


class ConditionalSpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate from the original status/route encoders used by query memory,
        # direct noisy-token concat, and modulation. No history/future GT input.
        self.canbus_mlp = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256))
        self.canbus_se = SELayerMLP()
        self.navi_embedding = nn.Embedding(3, 256)
        self.navi_se = SELayerMLP()

    def forward(self, feature, status, command):
        canbus = self.canbus_mlp(status.float())
        navigation = self.navi_embedding(command.argmax(-1))
        return self.navi_se(self.canbus_se(feature, canbus), navigation)


class Gated12Model(AgentEgoSpatialModel):
    def __init__(self, image_checkpoint=None):
        super().__init__(image_checkpoint)
        # Retain the first 12 initialized blocks exactly; unregister blocks 12-15.
        # The original head, concat, and query encoders keep their initialization.
        self.planner.layers = nn.ModuleList(list(self.planner.layers)[:12])
        self.spatial_gate = ConditionalSpatialGate()

    def conditions(self, camera, lidar, status, command):
        ego, agents, spatial, global_token, occupancy, canbus = super().conditions(
            camera, lidar, status, command)
        # super().conditions has already decoded queries from ungated memory.
        # Out-of-place multiplication preserves the perception/occupancy path.
        gated_spatial = self.spatial_gate(spatial, status, command)
        return ego, agents, gated_spatial, global_token, occupancy, canbus

    @torch.no_grad()
    def sample(self, camera, lidar, status, command, initial_noise=None, inference_steps=1):
        if inference_steps not in (1, 2):
            raise ValueError("This experiment evaluates only one or two denoising steps")
        ego, agents, spatial, global_token, _, canbus = self.conditions(camera, lidar, status, command)
        return self.planner.sample(camera.shape[0], ego, agents, spatial, global_token,
                                   inference_steps=inference_steps, canbus=canbus,
                                   initial_noise=initial_noise)
