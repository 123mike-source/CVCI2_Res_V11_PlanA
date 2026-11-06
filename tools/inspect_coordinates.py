"""Audit nuScenes LiDAR, camera, and ego-trajectory coordinate conventions."""

import argparse
from pathlib import Path
import sys
import pickle

import cv2
import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_paths import DEFAULT_DATA_ROOT, resolve_path


def load_infos(path):
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, dict) and "infos" in payload:
        return payload["infos"], payload.get("metadata", {})
    return payload, {}


def percentile_text(values):
    quantiles = np.percentile(values, [0, 1, 5, 50, 95, 99, 100])
    return " ".join(f"{value:.3f}" for value in quantiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="nuScenes data root (default: %(default)s)")
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()

    infos, metadata = load_infos(args.ann_file)
    subset = infos[: min(args.samples, len(infos))]
    trajectories = np.stack(
        [np.asarray(info["gt_ego_fut_trajs"], dtype=np.float32).cumsum(0) for info in subset]
    )
    masks = np.stack(
        [np.asarray(info["gt_ego_fut_masks"], dtype=np.float32) for info in subset]
    ).astype(bool)
    valid_x = trajectories[..., 0][masks]
    valid_y = trajectories[..., 1][masks]

    info = infos[0]
    lidar_path = resolve_path(args.data_root, info["lidar_path"])
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    lidar2ego_rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix.astype(np.float32)
    lidar2ego_translation = np.asarray(info["lidar2ego_translation"], dtype=np.float32)
    lidar_ego = lidar @ lidar2ego_rotation.T + lidar2ego_translation

    first_absolute_lidar = trajectories[0]
    first_absolute_ego_axes = first_absolute_lidar @ lidar2ego_rotation.T[:2, :2]
    camera_shapes = {}
    for name, camera in info["cams"].items():
        image = cv2.imread(resolve_path(args.data_root, camera["data_path"]), cv2.IMREAD_COLOR)
        camera_shapes[name] = None if image is None else tuple(image.shape)

    print("metadata", metadata)
    print("num_infos", len(infos), "audited", len(subset))
    print("info_keys", sorted(info.keys()))
    print("camera_names", list(info["cams"].keys()))
    print("camera_shapes", camera_shapes)
    print("lidar_path", lidar_path)
    print("lidar2ego_rotation", np.array2string(lidar2ego_rotation, precision=6))
    print("lidar2ego_translation", lidar2ego_translation.tolist())
    print("raw_lidar_xyz_percentiles")
    for axis, index in zip("xyz", range(3)):
        print(axis, percentile_text(lidar[:, index]))
    print("ego_lidar_xyz_percentiles")
    for axis, index in zip("xyz", range(3)):
        print(axis, percentile_text(lidar_ego[:, index]))
    print(
        "height_counts_raw",
        "z>0.2", int((lidar[:, 2] > 0.2).sum()),
        "z<=0.2", int((lidar[:, 2] <= 0.2).sum()),
    )
    print(
        "height_counts_ego",
        "z>0.2", int((lidar_ego[:, 2] > 0.2).sum()),
        "z<=0.2", int((lidar_ego[:, 2] <= 0.2).sum()),
    )
    print("absolute_trajectory_x_percentiles", percentile_text(valid_x))
    print("absolute_trajectory_y_percentiles", percentile_text(valid_y))
    print("first_delta_trajectory", np.array2string(info["gt_ego_fut_trajs"], precision=4))
    print("first_absolute_lidar", np.array2string(first_absolute_lidar, precision=4))
    print("first_absolute_ego_axes", np.array2string(first_absolute_ego_axes, precision=4))


if __name__ == "__main__":
    main()
