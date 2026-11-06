"""Minimal nuScenes adapter for the language-free TransFuser baseline."""

import pickle

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from data_paths import resolve_path


class NuScenesTransFuserDataset(Dataset):
    dynamic_classes = {
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "motorcycle", "bicycle", "pedestrian",
    }
    max_agents = 30

    def __init__(self, data_root, ann_file):
        self.data_root = data_root
        with open(ann_file, "rb") as stream:
            payload = pickle.load(stream)
        self.infos = payload["infos"] if isinstance(payload, dict) else payload
        self.mean = np.asarray([0.485, 0.456, 0.406], np.float32)[:, None, None]
        self.std = np.asarray([0.229, 0.224, 0.225], np.float32)[:, None, None]

    def __len__(self):
        return len(self.infos)

    def _camera(self, info):
        images = {}
        for name in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"):
            path = resolve_path(self.data_root, info["cams"][name]["data_path"])
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            images[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 1000 px from each side camera + 1600 px front = 3600:900 = 4:1.
        stitched = np.concatenate(
            (images["CAM_FRONT_LEFT"][:, 600:], images["CAM_FRONT"],
             images["CAM_FRONT_RIGHT"][:, :1000]), axis=1,
        )
        stitched = cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_AREA)
        tensor = stitched.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy((tensor - self.mean) / self.std)

    def _lidar_and_rotation(self, info):
        path = resolve_path(self.data_root, info["lidar_path"])
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
        rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix.astype(np.float32)
        translation = np.asarray(info["lidar2ego_translation"], dtype=np.float32)
        points = points @ rotation.T + translation
        points = points[(points[:, 2] < 100.0)]
        points = points[points[:, 2] > 0.2]
        bins = np.linspace(-32.0, 32.0, 257)
        histogram = np.histogramdd(points[:, :2], bins=(bins, bins))[0]
        histogram = np.minimum(histogram, 5.0) / 5.0
        return torch.from_numpy(histogram[None].astype(np.float32)), rotation, translation

    def _agent_targets(self, info, rotation, translation):
        boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7))), dtype=np.float32)
        names = np.asarray(info.get("gt_names", []))
        valid = np.asarray(info.get("valid_flag", np.ones(len(boxes))), dtype=bool)
        states = np.zeros((self.max_agents, 5), dtype=np.float32)
        labels = np.zeros(self.max_agents, dtype=np.float32)
        if len(boxes) == 0:
            return torch.from_numpy(states), torch.from_numpy(labels)

        centers = boxes[:, :3] @ rotation.T + translation
        yaw_offset = Quaternion(info["lidar2ego_rotation"]).yaw_pitch_roll[0]
        headings = (boxes[:, 6] + yaw_offset + np.pi) % (2 * np.pi) - np.pi
        candidates = np.stack(
            (centers[:, 0], centers[:, 1], headings, boxes[:, 3], boxes[:, 4]), axis=-1
        )
        in_range = (np.abs(centers[:, 0]) <= 32.0) & (np.abs(centers[:, 1]) <= 32.0)
        is_dynamic = np.asarray([name in self.dynamic_classes for name in names])
        candidates = candidates[valid & in_range & is_dynamic]
        if len(candidates):
            candidates = candidates[np.argsort(np.linalg.norm(candidates[:, :2], axis=-1))]
            candidates = candidates[:self.max_agents]
            states[:len(candidates)] = candidates
            labels[:len(candidates)] = 1.0
        return torch.from_numpy(states), torch.from_numpy(labels)

    @staticmethod
    def _status(info):
        raw = np.asarray(info.get("ego_status", np.zeros(10)), dtype=np.float32)
        # All entries are current-frame CAN bus values; no future command leaks in.
        return torch.from_numpy(np.concatenate((raw[0:3], raw[5:6], raw[6:9], raw[9:10])))

    def __getitem__(self, index):
        info = self.infos[index]
        lidar, rotation, translation = self._lidar_and_rotation(info)
        delta_lidar = np.asarray(info["gt_ego_fut_trajs"], dtype=np.float32)
        delta_xyz = np.concatenate((delta_lidar, np.zeros((6, 1), np.float32)), axis=-1)
        delta_ego = (delta_xyz @ rotation.T)[:, :2]
        agent_states, agent_labels = self._agent_targets(info, rotation, translation)
        return {
            "camera": self._camera(info),
            "lidar": lidar,
            "status": self._status(info),
            "command": torch.from_numpy(np.asarray(info["gt_ego_fut_cmd"], np.float32)),
            "trajectory_delta": torch.from_numpy(delta_ego),
            "mask": torch.from_numpy(np.asarray(info["gt_ego_fut_masks"], np.float32)),
            "agent_states": agent_states,
            "agent_labels": agent_labels,
        }
