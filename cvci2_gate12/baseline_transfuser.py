"""Language-free My-DiffusionDrive TransFuser + pure-noise DiT for nuScenes."""

import math

import timm
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def interpolate(value, *args, **kwargs):
    dtype = value.dtype
    source = value.float() if dtype == torch.bfloat16 else value
    return F.interpolate(source, *args, **kwargs).to(dtype)


class FusionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(True),
                                 nn.Linear(dim * 4, dim), nn.Dropout(0.1))

    def forward(self, value):
        value = value + self.attn(self.norm1(value), self.norm1(value), self.norm1(value),
                                  need_weights=False)[0]
        return value + self.mlp(self.norm2(value))


class ScaleFusion(nn.Module):
    def __init__(self, image_dim, lidar_dim):
        super().__init__()
        self.lidar_to_image = nn.Conv2d(lidar_dim, image_dim, 1)
        self.image_to_lidar = nn.Conv2d(image_dim, lidar_dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, 320, image_dim))
        self.blocks = nn.ModuleList([FusionBlock(image_dim), FusionBlock(image_dim)])

    def forward(self, image, lidar):
        image_small = F.adaptive_avg_pool2d(image, (8, 32))
        lidar_small = F.adaptive_avg_pool2d(self.lidar_to_image(lidar), (8, 8))
        tokens = torch.cat((image_small.flatten(2), lidar_small.flatten(2)), dim=2).transpose(1, 2)
        tokens = tokens + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        image_update = tokens[:, :256].transpose(1, 2).reshape(-1, image.shape[1], 8, 32)
        lidar_update = tokens[:, 256:].transpose(1, 2).reshape(-1, image.shape[1], 8, 8)
        image = image + interpolate(image_update, image.shape[-2:], mode="bilinear", align_corners=False)
        lidar_update = self.image_to_lidar(lidar_update)
        lidar = lidar + interpolate(lidar_update, lidar.shape[-2:], mode="bilinear", align_corners=False)
        return image, lidar


class TransFuserBackbone(nn.Module):
    def __init__(self, image_checkpoint=None):
        super().__init__()
        overlay = dict(file=image_checkpoint) if image_checkpoint else None
        self.image = timm.create_model("resnet34", pretrained=bool(image_checkpoint),
                                       pretrained_cfg_overlay=overlay, features_only=True)
        self.lidar = timm.create_model("resnet34", pretrained=False, in_chans=1, features_only=True)
        self.fusions = nn.ModuleList([ScaleFusion(c, c) for c in (64, 128, 256, 512)])
        self.c5 = nn.Conv2d(512, 64, 1)
        self.up1 = nn.Conv2d(64, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, 64, 3, padding=1)

    @staticmethod
    def _advance(layers, return_layers, value):
        for name, module in layers:
            value = module(value)
            if name in return_layers:
                break
        return value

    def forward(self, image, lidar):
        image_layers, lidar_layers = iter(self.image.items()), iter(self.lidar.items())
        if len(self.image.return_layers) > 4:
            image = self._advance(image_layers, self.image.return_layers, image)
            lidar = self._advance(lidar_layers, self.lidar.return_layers, lidar)
        for fusion in self.fusions:
            image = self._advance(image_layers, self.image.return_layers, image)
            lidar = self._advance(lidar_layers, self.lidar.return_layers, lidar)
            image, lidar = fusion(image, lidar)
        dense = F.relu(self.c5(lidar))
        dense = F.relu(self.up1(interpolate(dense, scale_factor=2, mode="bilinear", align_corners=False)))
        dense = F.relu(self.up2(interpolate(dense, size=(64, 64), mode="bilinear", align_corners=False)))
        return lidar, dense


class SinusoidalTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=time.device) / max(half - 1, 1))
        phase = time.float()[:, None] * freq[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class AgentHead(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(dim, 1024), nn.ReLU(True), nn.Linear(1024, 5))
        self.label = nn.Linear(dim, 1)
        nn.init.constant_(self.label.bias, -2.0)

    def forward(self, agents):
        state = self.state(agents)
        state = torch.cat((state[..., :3].tanh(), state[..., 3:].sigmoid()), dim=-1)
        return state, self.label(agents).squeeze(-1)


def normalize_agent_states(states):
    scale = states.new_tensor((32.0, 32.0, math.pi, 12.0, 5.0))
    states = states / scale
    return torch.cat((states[..., :3].clamp(-1, 1), states[..., 3:].clamp(0, 1)), dim=-1)


def agent_detection_loss(pred_states, pred_logits, target_states, target_labels):
    targets = normalize_agent_states(target_states)
    with torch.no_grad():
        box_cost = torch.cdist(pred_states.float(), targets.float(), p=1)
        box_cost = box_cost * target_labels[:, None].float()
        positive_cost = -F.logsigmoid(pred_logits.float())[:, :, None]
        negative_cost = -F.logsigmoid(-pred_logits.float())[:, :, None]
        class_cost = torch.where(target_labels[:, None].bool(), positive_cost, negative_cost)
        costs = class_cost + box_cost
        matching = [linear_sum_assignment(cost.cpu().numpy()) for cost in costs]

    batch_indices = torch.cat([
        torch.full((len(source),), index, device=pred_states.device, dtype=torch.long)
        for index, (source, _) in enumerate(matching)
    ])
    source_indices = torch.cat([
        torch.as_tensor(source, device=pred_states.device, dtype=torch.long)
        for source, _ in matching
    ])
    target_indices = torch.cat([
        torch.as_tensor(target, device=pred_states.device, dtype=torch.long)
        for _, target in matching
    ])
    matched_labels = target_labels[batch_indices, target_indices].float()
    class_loss = F.binary_cross_entropy_with_logits(
        pred_logits[batch_indices, source_indices], matched_labels
    )
    box_error = F.smooth_l1_loss(
        pred_states[batch_indices, source_indices], targets[batch_indices, target_indices],
        reduction="none",
    ).sum(dim=-1)
    box_loss = (box_error * matched_labels).sum() / matched_labels.sum().clamp_min(1)
    return class_loss, box_loss


def occupancy_loss(logits, lidar):
    target = (F.max_pool2d(lidar, kernel_size=4) > 0).to(logits.dtype)
    positive = target.sum().clamp_min(1)
    pos_weight = ((target.numel() - positive) / positive).clamp(1, 20)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return 0.5 * (bce + dice)


class DiTLayer(nn.Module):
    def __init__(self, interaction, dim=256, heads=8):
        super().__init__()
        self.interaction = interaction
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True) if not interaction else None
        self.agent_attn = nn.MultiheadAttention(dim, heads, batch_first=True) if interaction else None
        self.ego_attn = nn.MultiheadAttention(dim, heads, batch_first=True) if interaction else None
        self.bev_value = nn.Conv2d(256, 256, 3, padding=1) if interaction else None
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(4 if interaction else 2)]
        )
        self.ffn = nn.Sequential(nn.Linear(dim, 1024), nn.SiLU(), nn.Linear(1024, dim))
        self.modulation = nn.Sequential(nn.Mish(), nn.Linear(512, 512))

    def forward(self, hidden, positions, dense_bev, agents, ego, time, status):
        if not self.interaction:
            hidden = self.norms[0](hidden + self.self_attn(hidden, hidden, hidden, need_weights=False)[0])
        else:
            grid = torch.stack((positions[..., 1] / 32.0, positions[..., 0] / 32.0), dim=-1)
            value = self.bev_value(dense_bev)
            sampled = F.grid_sample(value.float(), grid[:, :, None].float(),
                                    mode="bilinear", padding_mode="zeros", align_corners=False).to(hidden.dtype)
            hidden = self.norms[0](hidden + sampled.squeeze(-1).transpose(1, 2))
            hidden = self.norms[1](hidden + self.agent_attn(hidden, agents, agents, need_weights=False)[0])
            hidden = self.norms[2](hidden + self.ego_attn(hidden, ego, ego, need_weights=False)[0])
        ffn_norm = self.norms[3] if self.interaction else self.norms[1]
        hidden = ffn_norm(hidden + self.ffn(hidden))
        scale, shift = self.modulation(torch.cat((time, status), dim=-1)).chunk(2, dim=-1)
        return hidden * (1.0 + scale) + shift


class PureNoisePlanner(nn.Module):
    def __init__(self, steps=6, train_timesteps=100):
        super().__init__()
        self.steps, self.train_timesteps = steps, train_timesteps
        self.action = nn.Linear(2, 256)
        self.action_fuse = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 256))
        self.position = nn.Embedding(steps, 256)
        self.ego_norm = nn.LayerNorm(256)
        self.time = nn.Sequential(SinusoidalTime(256), nn.Linear(256, 1024), nn.Mish(), nn.Linear(1024, 256))
        self.layers = nn.ModuleList([DiTLayer(index % 2 == 1) for index in range(16)])
        self.head = nn.Sequential(nn.Linear(256, 1024), nn.SiLU(), nn.LayerNorm(1024), nn.Linear(1024, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        x = torch.linspace(0, train_timesteps + 1, train_timesteps + 1)
        curve = torch.cos(((x / (train_timesteps + 1)) + 0.008) / 1.008 * math.pi / 2).square()
        curve = curve / curve[0]
        betas = (1.0 - curve[1:] / curve[:-1]).clamp(0, 0.999)
        self.register_buffer("alpha", torch.cumprod(1.0 - betas, dim=0))

    @staticmethod
    def normalize(delta):
        forward = ((delta[..., 0:1] + 0.5) / 8.1).clamp(0, 1) * 2 - 1
        left = (delta[..., 1:2] / 3.0).clamp(-1, 1)
        return torch.cat((forward, left), dim=-1)

    @staticmethod
    def denormalize(value):
        forward = (value[..., 0:1] + 1) * 0.5 * 8.1 - 0.5
        left = value[..., 1:2] * 3.0
        return torch.cat((forward, left), dim=-1)

    def predict(self, noisy, timestep, ego, agents, dense, status):
        time = self.time(timestep)[:, None]
        time_action = SinusoidalTime(256).to(noisy.device)(timestep)[:, None].expand(-1, self.steps, -1)
        hidden = self.action_fuse(torch.cat((self.action(noisy), time_action), dim=-1))
        hidden = hidden + self.position.weight[None] + self.ego_norm(ego).expand(-1, self.steps, -1)
        positions = self.denormalize(noisy.clamp(-1, 1)).cumsum(dim=1)
        outputs = []
        for layer in self.layers:
            hidden = layer(hidden, positions, dense, agents, ego, time, status)
            outputs.append(self.head(hidden))
        return outputs

    def loss(self, delta, mask, ego, agents, dense, status):
        clean = self.normalize(delta)
        timestep = torch.randint(0, self.train_timesteps, (delta.shape[0],), device=delta.device)
        noise = torch.randn_like(clean)
        alpha = self.alpha[timestep].view(-1, 1, 1).to(clean.dtype)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        valid = mask[:, :, None].to(clean.dtype)
        losses = [((pred - noise).square() * valid).sum() / (valid.sum() * 2).clamp_min(1)
                  for pred in self.predict(noisy, timestep, ego, agents, dense, status)]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample(self, batch, ego, agents, dense, status, inference_steps=5):
        value = torch.randn(batch, self.steps, 2, device=dense.device)
        indices = torch.linspace(
            self.train_timesteps - 1, 0, inference_steps, device=dense.device
        ).round().long()
        for index, step in enumerate(indices):
            timestep = step.expand(batch)
            epsilon = self.predict(value, timestep, ego, agents, dense, status)[-1]
            alpha = self.alpha[step]
            clean = ((value - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
            if index + 1 == len(indices):
                value = clean
            else:
                next_alpha = self.alpha[indices[index + 1]]
                value = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
        return self.denormalize(value).cumsum(dim=1)


class NuScenesTransFuserDiT(nn.Module):
    def __init__(self, image_checkpoint=None, agent_class_weight=0.2,
                 agent_box_weight=0.2, occupancy_weight=0.1):
        super().__init__()
        self.agent_class_weight = agent_class_weight
        self.agent_box_weight = agent_box_weight
        self.occupancy_weight = occupancy_weight
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
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1)
        )
        self.planner = PureNoisePlanner()

    def conditions(self, camera, lidar, status, command):
        low, dense = self.backbone(camera, lidar)
        low_tokens = self.bev_down(low).flatten(2).transpose(1, 2)
        global_token = (self.status(status) + self.route(command))[:, None]
        memory = torch.cat((low_tokens, global_token), dim=1) + self.memory_pos.weight[None]
        query = self.decoder(self.queries.weight[None].expand(camera.shape[0], -1, -1), memory)
        ego, agents = query[:, :1], query[:, 1:]
        low_dense = interpolate(low_tokens.transpose(1, 2).reshape(-1, 256, 8, 8),
                                (64, 64), mode="bilinear", align_corners=False)
        planner_dense = self.dense_proj(torch.cat((low_dense, dense), dim=1))
        return ego, agents, planner_dense, global_token, dense

    def forward(self, camera, lidar, status, command, delta, mask, agent_states, agent_labels):
        ego, agents, dense, global_token, occupancy_feature = self.conditions(
            camera, lidar, status, command
        )
        diffusion = self.planner.loss(delta, mask, ego, agents, dense, global_token)
        pred_states, pred_logits = self.agent_head(agents)
        agent_class, agent_box = agent_detection_loss(
            pred_states, pred_logits, agent_states, agent_labels
        )
        occupancy = occupancy_loss(self.occupancy_head(occupancy_feature), lidar)
        total = (diffusion + self.agent_class_weight * agent_class
                 + self.agent_box_weight * agent_box + self.occupancy_weight * occupancy)
        components = torch.stack((diffusion.detach(), agent_class.detach(),
                                  agent_box.detach(), occupancy.detach()))
        return total, components

    @torch.no_grad()
    def sample(self, camera, lidar, status, command):
        ego, agents, dense, global_token, _ = self.conditions(camera, lidar, status, command)
        return self.planner.sample(camera.shape[0], ego, agents, dense, global_token)
