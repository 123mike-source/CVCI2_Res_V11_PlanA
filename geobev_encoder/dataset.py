"""Six-camera GeoBEV observations; LiDAR is training supervision only."""
import cv2
import numpy as np
import torch

from temporal_dataset import TemporalNuScenesDataset, pose_matrix
from nuscenes_dataset import resolve_path
from . import CAMERAS


class GeoBEVNuScenesDataset(TemporalNuScenesDataset):
    def __init__(self, data_root, ann_file, supervision=True):
        super().__init__(data_root, ann_file)
        self.supervision = bool(supervision)
        if set(self.camera_names) != set(CAMERAS):
            raise ValueError('GeoBEV needs all six named nuScenes cameras')

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        info = self.infos[index]
        self._decoded = {}
        self._resized = {}
        lidar_to_ego = pose_matrix(info['lidar2ego_rotation'], info['lidar2ego_translation'])
        ego_to_global = pose_matrix(info['ego2global_rotation'], info['ego2global_translation'])
        images, sensors, intrinsics, post_rots, post_trans = [], [], [], [], []
        try:
            for name in self.camera_names:
                raw = cv2.imread(resolve_path(self.data_root, info['cams'][name]['data_path']))
                if raw is None:
                    raise FileNotFoundError(info['cams'][name]['data_path'])
                self._decoded[name] = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            # Preserve the pretrained IRWM preprocessing and its original camera order.
            world_image, projection = self._world_views(info)
            for name in CAMERAS:
                raw = self._decoded[name]
                h, w = raw.shape[:2]
                scale = 704 / w
                nw, nh = int(w * scale), int(h * scale)
                left, top = max(0, nw - 704) // 2, nh - 256
                resized = self._resized_view(name, raw, (nw, nh),
                    (left, top, left + 704, top + 256))
                # Official GeoBEV mmlabNormalize applies to_rgb=True to a PIL RGB array.
                # Match that channel reversal rather than silently changing pretrained input.
                normalized = (resized[..., ::-1].astype(np.float32) -
                              np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
                images.append(normalized.transpose(2, 0, 1))
                cam = info['cams'][name]
                camera_to_lidar = np.eye(4, dtype=np.float64)
                camera_to_lidar[:3, :3] = cam['sensor2lidar_rotation']
                camera_to_lidar[:3, 3] = cam['sensor2lidar_translation']
                sensors.append(lidar_to_ego @ camera_to_lidar)
                intrinsics.append(np.asarray(cam['cam_intrinsic'], np.float32))
                post_rots.append(np.diag([scale, scale, 1]).astype(np.float32))
                post_trans.append(np.array([-left, -top, 0], np.float32))
        finally:
            self._decoded = None
            self._resized = None
        delta = np.asarray(info['gt_ego_fut_trajs'], np.float32)
        delta_ego = np.pad(delta, ((0, 0), (0, 1))) @ lidar_to_ego[:3, :3].T
        output = dict(
            # Compatibility sentinel for device/batch accounting; never used as a feature.
            camera=torch.zeros(3, 1, 1),
            status=self._status(info), command=torch.from_numpy(np.asarray(info['gt_ego_fut_cmd'], np.float32)),
            trajectory_delta=torch.from_numpy(delta_ego[:, :2].astype(np.float32)),
            mask=torch.from_numpy(np.asarray(info['gt_ego_fut_masks'], np.float32)),
            world_image=world_image, projection=projection,
            world_status=torch.from_numpy(np.asarray(info['ego_status'], np.float32)),
            lidar_to_ego=torch.from_numpy(lidar_to_ego),
            geobev_image=torch.from_numpy(np.ascontiguousarray(np.stack(images))),
            geobev_sensor2ego=torch.from_numpy(np.stack(sensors)),
            geobev_intrinsics=torch.from_numpy(np.stack(intrinsics)),
            geobev_post_rots=torch.from_numpy(np.stack(post_rots)),
            geobev_post_trans=torch.from_numpy(np.stack(post_trans)),
            geobev_ego2global=torch.from_numpy(ego_to_global),
            data_index=index,
            meta=dict(token=str(info['token']), scene_token=str(info['scene_token']),
                      timestamp=float(info['timestamp']) / 1e6,
                      T_global=ego_to_global @ lidar_to_ego,
                      ego2global_translation=np.asarray(info['ego2global_translation']),
                      ego2global_rotation=np.asarray(info['ego2global_rotation']), camera_names=self.camera_names),
        )
        if self.supervision:
            occupancy, rotation, translation = self._lidar_and_rotation(info)
            states, labels = self._agent_targets(info, rotation, translation)
            output.update(lidar=occupancy, agent_states=states, agent_labels=labels)
            points = np.fromfile(resolve_path(self.data_root, info['lidar_path']), dtype=np.float32).reshape(-1, 5)[:, :3]
            depth_targets = {16: [], 8: []}
            for name, intrinsic, pr, pt in zip(CAMERAS, intrinsics, post_rots, post_trans):
                cam = info['cams'][name]
                rot = np.asarray(cam['sensor2lidar_rotation'], np.float64)
                trans = np.asarray(cam['sensor2lidar_translation'], np.float64)
                camera_points = np.linalg.solve(rot, (points - trans).T).T
                xyz = camera_points @ intrinsic.T
                z = xyz[:, 2]
                xy = xyz[:, :2] / np.maximum(z[:, None], 1e-6)
                pixels = xy * np.diag(pr)[:2] + pt[:2]
                valid = (np.isfinite(z) & np.isfinite(pixels).all(-1) & (z >= 1) & (z < 60) &
                         (pixels[:, 0] >= 0) & (pixels[:, 0] < 704) &
                         (pixels[:, 1] >= 0) & (pixels[:, 1] < 256))
                for stride in depth_targets:
                    h, w = 256 // stride, 704 // stride
                    target = np.full(h * w, np.inf, np.float32)
                    pixel = np.floor(pixels[valid] / stride).astype(np.int64)
                    np.minimum.at(target, pixel[:, 1] * w + pixel[:, 0], z[valid])
                    target[~np.isfinite(target)] = 0
                    depth_targets[stride].append(target.reshape(h, w))
            for stride, targets in depth_targets.items():
                output['geobev_depth_s%d' % stride] = torch.from_numpy(np.stack(targets))
        return output


class GeoBEVEvaluationDataset(GeoBEVNuScenesDataset):
    def __init__(self, data_root, ann_file):
        # No LiDAR file is opened when evaluating the camera-only model.
        super().__init__(data_root, ann_file, supervision=False)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result['target_lidar'] = torch.from_numpy(np.asarray(self.infos[index]['gt_ego_fut_trajs'], np.float32))
        return result
