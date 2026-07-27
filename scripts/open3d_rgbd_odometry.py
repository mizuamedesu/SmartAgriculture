#!/usr/bin/env python3
"""Estimate a camera trajectory from an aligned RGB-D sequence with Open3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def load_frames(cache_root: Path) -> list[dict]:
    frames = []
    for path in sorted((cache_root / "metadata").glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            frames.append(json.load(handle))
    if not frames:
        raise ValueError("RGB-D odometry cache contains no frames")
    return frames


def intrinsic_for(frame: dict) -> o3d.camera.PinholeCameraIntrinsic:
    intr = frame["intrinsics"]
    return o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]),
        int(intr["height"]),
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["ppx"]),
        float(intr["ppy"]),
    )


def load_rgbd(frame: dict) -> o3d.geometry.RGBDImage:
    color = o3d.io.read_image(frame["files"]["rgb"])
    depth = o3d.io.read_image(frame["files"]["depth"])
    depth_units_m = float(frame["depthUnitsM"])
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1.0 / depth_units_m,
        depth_trunc=float(frame["maxDepthM"]),
        convert_rgb_to_intensity=True,
    )


def camera_to_world_for_app(pose: np.ndarray) -> dict:
    # Open3D camera coordinates are +X right, +Y down, +Z forward.
    # AgriScan uses +X right, +Y up, -Z forward.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    converted = flip @ pose @ flip
    return {
        "rotation": converted[:3, :3].astype(float).tolist(),
        "translation": converted[:3, 3].astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    frames = load_frames(args.cache_root)
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_min = min(float(frame["minDepthM"]) for frame in frames)
    option.depth_max = max(float(frame["maxDepthM"]) for frame in frames)
    option.depth_diff_max = 0.075
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([12, 7, 4])
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    camera_to_world = np.eye(4, dtype=np.float64)
    poses = [camera_to_world.copy()]
    previous = load_rgbd(frames[0])
    succeeded = 0
    failed = 0

    for index in range(1, len(frames)):
        current = load_rgbd(frames[index])
        intrinsic = intrinsic_for(frames[index])
        success, previous_to_current, _ = (
            o3d.pipelines.odometry.compute_rgbd_odometry(
                previous,
                current,
                intrinsic,
                np.eye(4, dtype=np.float64),
                jacobian,
                option,
            )
        )
        if success and np.isfinite(previous_to_current).all():
            delta = np.linalg.inv(previous_to_current)
            translation = float(np.linalg.norm(delta[:3, 3]))
            rotation_cosine = float(
                np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
            )
            rotation_degrees = float(np.degrees(np.arccos(rotation_cosine)))
            if translation <= 0.30 and rotation_degrees <= 40.0:
                camera_to_world = camera_to_world @ delta
                succeeded += 1
            else:
                failed += 1
        else:
            failed += 1
        poses.append(camera_to_world.copy())
        previous = current

    output = {
        "schemaVersion": "agriscan-open3d-rgbd-odometry-v1",
        "open3dVersion": o3d.__version__,
        "frames": [
            {
                "frameIndex": int(frame["frameIndex"]),
                "cameraToWorld": camera_to_world_for_app(pose),
            }
            for frame, pose in zip(frames, poses, strict=True)
        ],
        "succeeded": succeeded,
        "failed": failed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
