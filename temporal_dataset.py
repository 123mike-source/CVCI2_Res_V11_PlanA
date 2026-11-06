"""Single-frame dual image views; histories belong to the consuming stream."""
import json
import hashlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data._utils.collate import default_collate

from nuscenes_dataset import NuScenesTransFuserDataset, resolve_path


def pose_matrix(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Quaternion(rotation).rotation_matrix
    result[:3, 3] = translation
    return result


class TemporalNuScenesDataset(NuScenesTransFuserDataset):
    def __init__(self, data_root, ann_file):
        super().__init__(data_root, ann_file)
        if not all('scene_token' in info for info in self.infos):
            sample_file = Path(data_root) / 'v1.0-trainval' / 'sample.json'
            with open(sample_file) as stream:
                mapping = {s['token']: s['scene_token'] for s in json.load(stream)}
            for info in self.infos:
                info['scene_token'] = mapping[info['token']]
        self.scenes = OrderedDict()
        for index, info in enumerate(self.infos):
            self.scenes.setdefault(str(info['scene_token']), []).append(index)
        self.scenes = OrderedDict((s, sorted(v, key=lambda i: self.infos[i]['timestamp']))
                                  for s, v in sorted(self.scenes.items()))
        # Resume indices are meaningful only with the same ordered annotations.
        digest = hashlib.sha256()
        for info in self.infos:
            digest.update(json.dumps([str(info['token']), str(info['scene_token']),
                                      int(info['timestamp'])], separators=(',', ':')).encode())
        self.stream_fingerprint = digest.hexdigest()
        self.camera_names = tuple(self.infos[0]['cams'])
        if len(self.camera_names) != 6:
            raise ValueError('IRWM requires six cameras')
        self._decoded = None
        self._resized = None

    def _resized_view(self, name, image, dimensions, crop):
        """Reuse identical PIL transforms, without changing pixels or calibration."""
        from PIL import Image
        key = (name, tuple(dimensions), tuple(crop))
        if self._resized is not None and key in self._resized:
            return self._resized[key]
        value = np.asarray(Image.fromarray(image).resize(tuple(dimensions)).crop(tuple(crop)))
        if self._resized is not None:
            self._resized[key] = value
        return value

    def _camera(self, info):
        images = self._decoded
        stitched = np.concatenate((images['CAM_FRONT_LEFT'][:, 600:],
                                   images['CAM_FRONT'],
                                   images['CAM_FRONT_RIGHT'][:, :1000]), axis=1)
        stitched = cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_AREA)
        value = stitched.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy((value - self.mean) / self.std)

    def _world_views(self, info):
        images, projections = [], []
        for name in self.camera_names:
            image = self._decoded[name]
            height, width = image.shape[:2]
            # SparseDrive test-time ResizeCropFlip: bottom crop after resize.
            resize = max(256 / height, 704 / width)
            new_w, new_h = int(width * resize), int(height * resize)
            crop_x, crop_y = int((new_w - 704) / 2), new_h - 256
            # PIL matches the upstream deterministic image transform.
            transformed = self._resized_view(name, image, (new_w, new_h),
                (crop_x, crop_y, crop_x + 704, crop_y + 256)).astype(np.float32)
            normalized = ((transformed - np.array([123.675, 116.28, 103.53], np.float32))
                          / np.array([58.395, 57.12, 57.375], np.float32))
            images.append(normalized.transpose(2, 0, 1))
            camera = info['cams'][name]
            camera_to_lidar = np.eye(4, dtype=np.float64)
            camera_to_lidar[:3, :3] = camera['sensor2lidar_rotation']
            camera_to_lidar[:3, 3] = camera['sensor2lidar_translation']
            intrinsic = np.eye(4, dtype=np.float64)
            intrinsic[:3, :3] = camera['cam_intrinsic']
            augmentation = np.eye(4, dtype=np.float64)
            augmentation[0, 0] = augmentation[1, 1] = resize
            augmentation[0, 2], augmentation[1, 2] = -crop_x, -crop_y
            projections.append(augmentation @ intrinsic @ np.linalg.inv(camera_to_lidar))
        return (torch.from_numpy(np.stack(images)),
                torch.from_numpy(np.stack(projections).astype(np.float32)))

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        info = self.infos[index]
        self._decoded = {}
        self._resized = {}
        try:
            for name in self.camera_names:
                image = cv2.imread(resolve_path(self.data_root, info['cams'][name]['data_path']))
                if image is None:
                    raise FileNotFoundError(info['cams'][name]['data_path'])
                self._decoded[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = super().__getitem__(index)
            world_image, projection = self._world_views(info)
        finally:
            self._decoded = None
            self._resized = None
        status = np.asarray(info['ego_status'], np.float32)
        if status.shape != (10,):
            raise ValueError('world branch requires the original 10D ego_status')
        lidar_to_ego = pose_matrix(info['lidar2ego_rotation'], info['lidar2ego_translation'])
        ego_to_global = pose_matrix(info['ego2global_rotation'], info['ego2global_translation'])
        output.update(world_image=world_image, projection=projection,
                      world_status=torch.from_numpy(status),
                      lidar_to_ego=torch.from_numpy(lidar_to_ego),
                      meta={'token': str(info['token']), 'scene_token': str(info['scene_token']),
                            'timestamp': float(info['timestamp']) / 1e6,
                            'T_global': ego_to_global @ lidar_to_ego,
                            'ego2global_translation': np.asarray(info['ego2global_translation']),
                            'ego2global_rotation': np.asarray(info['ego2global_rotation']),
                            'camera_names': self.camera_names},
                      data_index=index)
        return output


def temporal_collate(samples):
    return {key: [s[key] for s in samples] if key == 'meta' else
            default_collate([s[key] for s in samples]) for key in samples[0]}


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}
