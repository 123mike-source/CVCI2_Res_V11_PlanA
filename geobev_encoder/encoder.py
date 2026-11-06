"""Checkpoint-compatible GeoBEV feature path and metric BEV geometry."""
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models.backbones import ResNet

from .blocks import CustomFPN, CustomResNet, FPN_LSS, DepthNet
from . import GEOBEV_SHA256


def camera_mlp(sensor, intrinsics, post_rot, post_trans):
    b, n = sensor.shape[:2]
    one = sensor.new_ones(b, n)
    zero = sensor.new_zeros(b, n)
    values = [intrinsics[..., 0, 0], intrinsics[..., 1, 1],
              intrinsics[..., 0, 2], intrinsics[..., 1, 2],
              post_rot[..., 0, 0], post_rot[..., 0, 1], post_trans[..., 0],
              post_rot[..., 1, 0], post_rot[..., 1, 1], post_trans[..., 1],
              one, zero, zero, one, one]
    return torch.cat((torch.stack(values, -1), sensor[..., :3, :].reshape(b, n, 12)), -1)


class RadialCartesianSampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = 118
        self.depth_net = DepthNet(512, 512, 80, self.D + 1,
                                 use_dcn=False, aspp_mid_channels=96)
        self.context_convs = nn.ModuleList([nn.Conv2d(512, 80, 1)])
        self.depth_upconvs = nn.ModuleList([nn.Conv2d(512, 512, 1)])
        self.depth_convs = nn.ModuleList([nn.Conv2d(512, self.D + 1, 3, padding=1)])
        # GeoBEV's grid has row=y, col=x. The planner has row=x, col=y.
        axis = torch.arange(256, dtype=torch.float32) * .4 - 51.2 + .2
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('bev_points', torch.stack((x, y, torch.zeros_like(x)), -1), persistent=False)

    def forward(self, features, sensor_to_keyego, intrinsics, post_rot, post_trans,
                reference_calibration):
        b, n = features[0].shape[:2]
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            mlp = camera_mlp(reference_calibration.float(), intrinsics.float(),
                             post_rot.float(), post_trans.float())
            coarse = features[0].float().flatten(0, 1)
            logits0, context, depth_feature = self.depth_net(coarse, mlp)
            fine = features[1].float().flatten(0, 1)
            context = F.interpolate(context, scale_factor=2, mode='bilinear', align_corners=False)
            context = context + self.context_convs[0](fine)
            depth_feature = F.interpolate(depth_feature, scale_factor=2, mode='bilinear', align_corners=False)
            depth_feature = self.depth_upconvs[0](depth_feature) + fine
            logits1 = self.depth_convs[0](depth_feature)
            depth = logits1[:, :self.D]
            weights = torch.where(depth.softmax(1) >= 1 / self.D, depth.sigmoid(), torch.zeros_like(depth))
            # Keep the pretrained detection model's hard foreground filter.
            weights = weights * (logits1[:, self.D:].sigmoid() >= .1)
            radial = torch.matmul(weights.permute(0, 3, 1, 2),
                                  context.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).contiguous()
            points = self.bev_points[None, None].expand(b, n, -1, -1, -1)
            rot = sensor_to_keyego[..., :3, :3].float()
            trans = sensor_to_keyego[..., :3, 3].float()
            cam = torch.einsum('bnij,bnhwj->bnhwi', torch.linalg.inv(rot),
                               points - trans[:, :, None, None])
            projected = torch.einsum('bnij,bnhwj->bnhwi', intrinsics.float(), cam)
            distance = projected[..., 2]
            xy = projected[..., :2] / torch.where(distance.abs() > 1e-6, distance, torch.ones_like(distance))[..., None]
            pixel = torch.cat((xy, distance[..., None]), -1)
            pixel = torch.einsum('bnij,bnhwj->bnhwi', post_rot.float(), pixel)
            pixel = pixel + post_trans.float()[:, :, None, None]
            column = pixel[..., 0] / 8
            depth_index = (pixel[..., 2] - 1) / .5
            # Preserve the official grid_sample normalization (align_corners=False).
            grid = torch.stack((column / radial.shape[-1] * 2 - 1,
                                depth_index / self.D * 2 - 1), -1)
            sampled = F.grid_sample(radial, grid.flatten(0, 1), align_corners=False)
            bev = sampled.view(b, n, 80, 256, 256).sum(1)
            visible = ((distance >= 1) & (distance < 60) &
                       (pixel[..., 0] >= 0) & (pixel[..., 0] < 704) &
                       (pixel[..., 1] >= 0) & (pixel[..., 1] < 256)).any(1, keepdim=True)
            return bev, (logits0, logits1), visible


def planner_grid(size=64, device=None):
    # Each destination cell is centred in a physical metre interval.
    axis = (torch.arange(size, device=device, dtype=torch.float32) + .5) * (64 / size) - 32
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x / 51.2, y / 51.2), -1)


def crop_planner_bev(native):
    grid = planner_grid(device=native.device)[None].expand(len(native), -1, -1, -1)
    return F.grid_sample(native.float(), grid, align_corners=False)


class GeoBEVEncoder(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.img_backbone = ResNet(depth=50, num_stages=4, out_indices=(1, 2, 3),
                                   frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=True),
                                   norm_eval=True, style='pytorch')
        self.img_neck = CustomFPN(in_channels=[1024, 2048], out_channels=512,
                                  num_outs=1, start_level=0, out_ids=[0])
        self.img_view_transformer = RadialCartesianSampling()
        self.pre_process_net = CustomResNet(numC_input=80, num_layer=[2],
                                             num_channels=[80], stride=[1], backbone_output_ids=[0])
        self.img_bev_encoder_backbone = CustomResNet(numC_input=160, num_channels=[160, 320, 640])
        self.img_bev_encoder_neck = FPN_LSS(in_channels=800, out_channels=256)
        self.initialization_audit = None
        if checkpoint is not None:
            self.load_pretrained(checkpoint)
        self.train(False)

    def load_pretrained(self, checkpoint):
        digest = hashlib.sha256()
        with Path(checkpoint).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != GEOBEV_SHA256:
            raise ValueError('GeoBEV source checkpoint fingerprint differs')
        source = torch.load(str(checkpoint), map_location='cpu', weights_only=True)['state_dict']
        expected = self.state_dict()
        selected = {n: v for n, v in source.items() if n in expected}
        missing = set(expected) - set(selected)
        extra = [n for n in source if n not in expected and not n.startswith('pts_bbox_head.')]
        mismatch = [n for n in selected if selected[n].shape != expected[n].shape]
        if missing or extra or mismatch:
            raise RuntimeError('GeoBEV source differs: ' + str(dict(missing=sorted(missing), extra=extra, shape=mismatch)))
        self.load_state_dict(selected, strict=True)
        self.initialization_audit = dict(status='PASS', source=str(checkpoint), sha256=digest.hexdigest(),
                                         verified_tensors=len(selected), discarded='CenterHead only')

    def train(self, mode=True):
        super().train(mode)
        # Current/history encodings share parameters but never update BN statistics.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def encode_frame(self, images, sensor, intrinsics, post_rot, post_trans, reference_sensor):
        b, n = images.shape[:2]
        c3, c4, c5 = self.img_backbone(images.flatten(0, 1))
        coarse = self.img_neck((c4, c5))
        features = [coarse.reshape(b, n, *coarse.shape[1:]), c3.reshape(b, n, *c3.shape[1:])]
        bev, logits, visible = self.img_view_transformer(features, sensor, intrinsics,
                                                         post_rot, post_trans, reference_sensor)
        return self.pre_process_net(bev)[0], logits, visible

    def forward(self, history):
        images = history['geobev_image']
        poses = history['geobev_ego2global'].double()
        sensors = history['geobev_sensor2ego'].double()
        key_inverse = torch.linalg.inv(poses[:, -1])
        aligned = (key_inverse[:, None, None] @ poses[:, :, None] @ sensors).float()
        fields = [history[k] for k in ('geobev_intrinsics', 'geobev_post_rots', 'geobev_post_trans')]
        current_sensor = aligned[:, -1]
        current, logits, visible = self.encode_frame(images[:, -1], current_sensor,
            *(v[:, -1] for v in fields), current_sensor)
        with torch.no_grad():
            previous, _, _ = self.encode_frame(images[:, -2], aligned[:, -2],
                *(v[:, -2] for v in fields), current_sensor)
            previous = previous * history['geobev_history_valid'][:, None, None, None]
        encoded = self.img_bev_encoder_neck(self.img_bev_encoder_backbone(torch.cat((current, previous), 1)))
        return crop_planner_bev(encoded), logits, crop_planner_bev(visible.float()) > .5
