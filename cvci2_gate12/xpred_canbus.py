"""Original sixteen-layer CVCI2: x0 prediction and direct CAN bus concat.

Preserve delta trajectories, six waypoint tokens, fixed sampling coordinates
within a network call, the cosine schedule, auxiliary losses and five DDIM steps.
Only ImageNet image weights are loaded; all model parameters remain trainable.
"""
import torch
from torch import nn

from .baseline_transfuser import (
    AgentHead, PureNoisePlanner, SinusoidalTime, TransFuserBackbone,
    agent_detection_loss, interpolate, occupancy_loss,
)


def ddim_xpred_update(value, prediction, alpha, next_alpha):
    """Deterministic DDIM with an x0 prediction; all arithmetic is FP32.

Estimate epsilon from x0 by dividing by sqrt(1-alpha), avoiding the old
epsilon-to-x0 division by sqrt(alpha) at the near-pure-noise starting step.
As in standard DDIM, derive epsilon before clipping predicted x0.
"""
    value, prediction = value.float(), prediction.float()
    alpha, next_alpha = alpha.float(), next_alpha.float()
    epsilon = (value - alpha.sqrt() * prediction) / (1-alpha).sqrt()
    clean = prediction.clamp(-1, 1)
    return next_alpha.sqrt() * clean + (1-next_alpha).sqrt() * epsilon


class XPredCanbusPlanner(PureNoisePlanner):
    def __init__(self, steps=6, train_timesteps=100):
        super().__init__(steps, train_timesteps)
        # Add after the original modules so common initialization stays identical.
        self.canbus_concat = nn.Linear(512, 256)

    def predict(self, noisy, timestep, ego, agents, dense, status, canbus=None):
        if canbus is None:
            raise ValueError("Direct concat requires the separate CAN bus embedding")
        time = self.time(timestep)[:, None]
        time_action = SinusoidalTime(256).to(noisy.device)(timestep)[:, None].expand(-1, self.steps, -1)
        hidden = self.action_fuse(torch.cat((self.action(noisy.float()), time_action), -1))
        state_tokens = canbus.expand(-1, self.steps, -1)
        hidden = self.canbus_concat(torch.cat((hidden, state_tokens), -1))
        hidden = hidden + self.position.weight[None] + self.ego_norm(ego).expand(-1, self.steps, -1)
        positions = self.denormalize(noisy.float().clamp(-1, 1)).cumsum(1)
        outputs = []
        for layer in self.layers:
            # Preserve original time + global(CAN bus + navigation) modulation.
            hidden = layer(hidden, positions, dense, agents, ego, time, status)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                outputs.append(self.head(hidden.float()))
        return outputs

    def loss(self, delta, mask, ego, agents, dense, status, canbus=None):
        clean = self.normalize(delta.float())
        timestep = torch.randint(0, self.train_timesteps, (delta.shape[0],), device=delta.device)
        noise = torch.randn_like(clean)
        alpha = self.alpha[timestep].float().view(-1, 1, 1)
        noisy = alpha.sqrt() * clean + (1-alpha).sqrt() * noise
        valid = mask[..., None].float()
        losses = [((prediction-clean).square()*valid).sum() / (2*valid.sum()).clamp_min(1)
                  for prediction in self.predict(noisy, timestep, ego, agents, dense, status, canbus)]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, batch, ego, agents, dense, status, inference_steps=5,
               canbus=None, initial_noise=None):
        value = (torch.randn(batch, self.steps, 2, device=dense.device)
                 if initial_noise is None else initial_noise.float())
        indices = torch.linspace(self.train_timesteps-1, 0, inference_steps,
                                 device=dense.device).round().long()
        for index, step in enumerate(indices):
            prediction = self.predict(value, step.expand(batch), ego, agents, dense, status, canbus)[-1]
            if index+1 == len(indices):
                value = prediction.float().clamp(-1, 1)
            else:
                value = ddim_xpred_update(value, prediction, self.alpha[step], self.alpha[indices[index+1]])
        return self.denormalize(value).cumsum(1)


class NuScenesXPredCanbus(nn.Module):
    def __init__(self, image_checkpoint=None):
        super().__init__()
        # Same construction order as the original model for paired initialization.
        self.backbone = TransFuserBackbone(image_checkpoint)
        self.bev_down = nn.Conv2d(512, 256, 1)
        self.status = nn.Linear(8, 256)
        self.route = nn.Linear(3, 256)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)
        self.memory_pos = nn.Embedding(65, 256)
        self.queries = nn.Embedding(31, 256)
        layer = nn.TransformerDecoderLayer(256, 8, 1024, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 3)
        self.dense_proj = nn.Conv2d(320, 256, 1)
        self.agent_head = AgentHead()
        self.occupancy_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        self.planner = XPredCanbusPlanner()

    def conditions(self, camera, lidar, status, command):
        low, dense = self.backbone(camera, lidar)
        low_tokens = self.bev_down(low).flatten(2).transpose(1, 2)
        canbus = self.status(status)[:, None]
        global_token = canbus + self.route(command)[:, None]
        memory = torch.cat((low_tokens, global_token), 1) + self.memory_pos.weight[None]
        query = self.decoder(self.queries.weight[None].expand(camera.shape[0], -1, -1), memory)
        low_dense = interpolate(low_tokens.transpose(1, 2).reshape(-1, 256, 8, 8),
                                (64, 64), mode="bilinear", align_corners=False)
        planner_dense = self.dense_proj(torch.cat((low_dense, dense), 1))
        return query[:, :1], query[:, 1:], planner_dense, global_token, dense, canbus

    def forward(self, camera, lidar, status, command, delta, mask, agent_states, agent_labels):
        ego, agents, dense, global_token, occupancy_feature, canbus = self.conditions(camera, lidar, status, command)
        trajectory = self.planner.loss(delta, mask, ego, agents, dense, global_token, canbus)
        states, logits = self.agent_head(agents)
        agent_class, agent_box = agent_detection_loss(states, logits, agent_states, agent_labels)
        occupancy = occupancy_loss(self.occupancy_head(occupancy_feature), lidar)
        loss = trajectory + 0.2*agent_class + 0.2*agent_box + 0.1*occupancy
        components = torch.stack((trajectory, agent_class, agent_box, occupancy)).detach()
        return loss, components

    @torch.no_grad()
    def sample(self, camera, lidar, status, command, initial_noise=None):
        ego, agents, dense, global_token, _, canbus = self.conditions(camera, lidar, status, command)
        return self.planner.sample(camera.shape[0], ego, agents, dense, global_token,
                                   canbus=canbus, initial_noise=initial_noise)
