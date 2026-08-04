#!/usr/bin/env python3
"""Run pretrained SHARP inference with MLX and merge RGB-D-aligned views.

This is inference only: the checkpoint is never trained or modified. Each RGB
keyframe is passed once through the pretrained SHARP network, metric scale is
anchored to its matching RealSense depth image, and the resulting Gaussians are
transformed into the recorded RGB-D camera trajectory.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np
from PIL import Image

from sharp_mlx.gaussian_utils import Gaussians3D, unproject_gaussians
from sharp_mlx.predictor import create_predictor
from sharp_mlx.weights import load_weights


INTERNAL_SIZE = 1536
OUTPUT_SIZE = INTERNAL_SIZE // 2
SH_C0 = 0.2820948


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrained SHARP feed-forward inference on Apple MLX"
    )
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=600_000)
    parser.add_argument("--max-views", type=int, default=100)
    return parser.parse_args()


def load_keyframes(session_root: Path, max_views: int) -> list[dict]:
    frames: list[dict] = []
    for metadata_path in sorted((session_root / "metadata").glob("*.json")):
        frame = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = frame.get("files", {})
        if files.get("rgb") and files.get("depth"):
            frames.append(frame)
    frames.sort(key=lambda item: (item.get("frameIndex", 0), item.get("frameNumber", 0)))
    if not frames:
        raise ValueError("no RGB-D keyframes were provided for SHARP inference")
    view_count = max(1, min(max_views, len(frames)))
    if view_count == len(frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, view_count).round().astype(np.int64)
    return [frames[int(index)] for index in indices]


def resize_align_corners(image: np.ndarray, size: int) -> np.ndarray:
    """Match torch bilinear interpolate(..., align_corners=True) without PyTorch."""
    source_h, source_w, _ = image.shape
    ys = np.linspace(0.0, source_h - 1.0, size, dtype=np.float32)
    xs = np.linspace(0.0, source_w - 1.0, size, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x1 = np.minimum(x0 + 1, source_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    top = image[y0][:, x0] * (1.0 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1.0 - wx) + image[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def infer_camera_gaussians(
    model,
    image: np.ndarray,
    focal_px: float,
) -> Gaussians3D:
    orig_h, orig_w = image.shape[:2]
    resized = resize_align_corners(image, INTERNAL_SIZE)
    network_input = mx.array(resized[None])
    disparity_factor = mx.array([focal_px / orig_w], dtype=mx.float32)
    predicted = model(network_input, disparity_factor)
    mx.eval(predicted)

    opacities = predicted.opacities
    if opacities.ndim == 3:
        opacities = mx.squeeze(opacities, axis=-1)
    predicted_ndc = Gaussians3D(
        mean_vectors=predicted.means,
        singular_values=predicted.scales,
        quaternions=predicted.quaternions,
        colors=predicted.colors,
        opacities=opacities,
    )
    scale_x = INTERNAL_SIZE / orig_w
    scale_y = INTERNAL_SIZE / orig_h
    intrinsics = mx.array(
        [
            [focal_px * scale_x, 0.0, orig_w * 0.5 * scale_x, 0.0],
            [0.0, focal_px * scale_y, orig_h * 0.5 * scale_y, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=mx.float32,
    )
    return unproject_gaussians(
        predicted_ndc,
        mx.eye(4, dtype=mx.float32),
        intrinsics,
        (INTERNAL_SIZE, INTERNAL_SIZE),
    )


def depth_scale_for_prediction(
    frame: dict, means: np.ndarray
) -> tuple[float, int, np.ndarray]:
    depth_raw = np.asarray(Image.open(frame["files"]["depth"]), dtype=np.float32)
    depth_m = depth_raw * float(frame.get("depthUnitsM", 0.001))
    depth_resized = np.asarray(
        Image.fromarray(depth_m, mode="F").resize(
            (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    )
    first_layer_z = means.reshape(OUTPUT_SIZE, OUTPUT_SIZE, 2, 3)[:, :, 0, 2]
    min_depth = float(frame.get("minDepthM", 0.05))
    max_depth = float(frame.get("maxDepthM", 8.0))
    valid = (
        np.isfinite(first_layer_z)
        & (first_layer_z > 0.02)
        & np.isfinite(depth_resized)
        & (depth_resized >= min_depth)
        & (depth_resized <= max_depth)
    )
    ratio_map = np.ones_like(first_layer_z, dtype=np.float32)
    ratio_map[valid] = depth_resized[valid] / first_layer_z[valid]
    ratios = ratio_map[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0.2) & (ratios <= 5.0)]
    if len(ratios) < 1_000:
        return 1.0, int(len(ratios)), np.ones(means.shape[0], dtype=np.float32)
    median = float(np.median(ratios))
    deviation = np.abs(ratios - median)
    mad = float(np.median(deviation))
    if mad > 1.0e-6:
        ratios = ratios[deviation <= 3.5 * mad]
    global_scale = float(np.clip(np.median(ratios), 0.35, 3.0))

    # Anchor the visible SHARP surface to the measured RealSense depth at
    # each corresponding pixel. The same local correction is applied to the
    # second (inpainted) Gaussian layer so its learned separation is retained.
    local_valid = (
        valid
        & np.isfinite(ratio_map)
        & (ratio_map >= max(0.2, global_scale * 0.55))
        & (ratio_map <= min(5.0, global_scale * 1.8))
    )
    local_scale = np.full_like(first_layer_z, global_scale, dtype=np.float32)
    local_scale[local_valid] = np.clip(
        ratio_map[local_valid], global_scale * 0.65, global_scale * 1.55
    )
    gaussian_scales = np.repeat(local_scale[:, :, None], 2, axis=2).reshape(-1)
    return global_scale, int(np.count_nonzero(local_valid)), gaussian_scales


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float32,
        )
    axis = int(np.argmax(np.diag(rotation)))
    if axis == 0:
        s = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-8)) * 2.0
        values = [
            (rotation[2, 1] - rotation[1, 2]) / s,
            0.25 * s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
        ]
    elif axis == 1:
        s = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-8)) * 2.0
        values = [
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            0.25 * s,
            (rotation[1, 2] + rotation[2, 1]) / s,
        ]
    else:
        s = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-8)) * 2.0
        values = [
            (rotation[1, 0] - rotation[0, 1]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
            (rotation[1, 2] + rotation[2, 1]) / s,
            0.25 * s,
        ]
    return np.asarray(values, dtype=np.float32)


def quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def transform_and_select(
    frame: dict,
    gaussians: Gaussians3D,
    point_budget: int,
) -> tuple[dict[str, np.ndarray], float, int]:
    means = np.asarray(gaussians.mean_vectors, dtype=np.float32).reshape(-1, 3)
    scales = np.asarray(gaussians.singular_values, dtype=np.float32).reshape(-1, 3)
    quaternions = np.asarray(gaussians.quaternions, dtype=np.float32).reshape(-1, 4)
    colors = np.asarray(gaussians.colors, dtype=np.float32).reshape(-1, 3)
    opacities = np.asarray(gaussians.opacities, dtype=np.float32).reshape(-1)

    metric_scale, depth_samples, local_scales = depth_scale_for_prediction(frame, means)
    means *= local_scales[:, None]
    scales *= local_scales[:, None]

    pose = frame.get("cameraToWorld")
    if pose:
        rotation = np.asarray(pose["rotation"], dtype=np.float32)
        translation = np.asarray(pose["translation"], dtype=np.float32)
    else:
        rotation = np.eye(3, dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
    # SHARP emits OpenCV camera coordinates (+x right, +y down, +z forward).
    # The app's RGB-D world uses +x right, +y up, +z backward before applying
    # camera_to_world, so rotate by pi around x before the recorded pose.
    camera_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    world_rotation = rotation @ camera_flip
    means = means @ world_rotation.T + translation
    pose_quaternion = matrix_to_quaternion(world_rotation)
    quaternions = quaternion_product(
        np.broadcast_to(pose_quaternion, quaternions.shape), quaternions
    )
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-8
    )

    valid = (
        np.all(np.isfinite(means), axis=1)
        & np.all(np.isfinite(scales), axis=1)
        & np.all(np.isfinite(colors), axis=1)
        & np.all(np.isfinite(quaternions), axis=1)
        & np.isfinite(opacities)
        & (opacities >= 0.015)
        & (np.max(scales, axis=1) <= 0.5)
    )
    indices = np.flatnonzero(valid)
    scale_boosts = np.ones(len(indices), dtype=np.float32)
    if len(indices) > point_budget:
        # SHARP predicts two Gaussians for each image ray. Layer zero is the
        # directly visible surface used for RGB-D metric anchoring; layer one
        # is the learned behind-surface completion. Uniformly sampling the
        # flattened array at a constrained budget discards half of the visible
        # image and produces a checkerboard-like point cloud. Keep most of the
        # budget for the visible layer while retaining a smaller completion
        # layer, then enlarge each selected footprint to conserve area.
        visible = indices[(indices & 1) == 0]
        completion = indices[(indices & 1) == 1]
        visible_budget = min(len(visible), max(1, int(point_budget * 0.84)))
        completion_budget = min(len(completion), point_budget - visible_budget)
        remaining = point_budget - visible_budget - completion_budget
        if remaining > 0:
            visible_budget += min(remaining, len(visible) - visible_budget)
            remaining = point_budget - visible_budget - completion_budget
        if remaining > 0:
            completion_budget += min(
                remaining, len(completion) - completion_budget
            )

        selected_groups: list[np.ndarray] = []
        boost_groups: list[np.ndarray] = []
        for group, budget in (
            (visible, visible_budget),
            (completion, completion_budget),
        ):
            if budget <= 0:
                continue
            positions = np.linspace(0, len(group) - 1, budget).astype(np.int64)
            selected_groups.append(group[positions])
            footprint_boost = float(
                np.clip(math.sqrt(len(group) / max(1, budget)), 1.0, 1.8)
            )
            boost_groups.append(
                np.full(budget, footprint_boost, dtype=np.float32)
            )

        indices = np.concatenate(selected_groups)
        scale_boosts = np.concatenate(boost_groups)
        ordering = np.argsort(indices)
        indices = indices[ordering]
        scale_boosts = scale_boosts[ordering]
    return (
        {
            "means": means[indices],
            "scales": np.clip(
                scales[indices] * scale_boosts[:, None], 1.0e-5, 0.5
            ),
            "quaternions": quaternions[indices],
            "colors": np.clip(colors[indices], 0.0, 1.0),
            "opacities": np.clip(opacities[indices], 1.0e-5, 1.0 - 1.0e-5),
        },
        metric_scale,
        depth_samples,
    )


def concatenate(parts: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    values = list(parts)
    return {
        key: np.concatenate([part[key] for part in values], axis=0)
        for key in ("means", "scales", "quaternions", "colors", "opacities")
    }


def linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    return np.where(
        colors <= 0.0031308,
        colors * 12.92,
        1.055 * np.power(np.maximum(colors, 0.0031308), 1.0 / 2.4) - 0.055,
    )


def write_ascii_ply(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(values["means"])
    colors_sh = (linear_to_srgb(values["colors"]) - 0.5) / SH_C0
    opacity_logits = np.log(values["opacities"] / (1.0 - values["opacities"]))
    log_scales = np.log(values["scales"])

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(
            "comment AgriScan Studio pretrained SHARP feed-forward inference on MLX\n"
        )
        handle.write(f"element vertex {count}\n")
        for prop in (
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ):
            handle.write(f"property float {prop}\n")
        handle.write("end_header\n")
        zeros = np.zeros((min(50_000, count), 3), dtype=np.float32)
        for start in range(0, count, 50_000):
            end = min(count, start + 50_000)
            chunk_count = end - start
            rows = np.concatenate(
                [
                    values["means"][start:end],
                    zeros[:chunk_count],
                    colors_sh[start:end],
                    opacity_logits[start:end, None],
                    log_scales[start:end],
                    values["quaternions"][start:end],
                ],
                axis=1,
            )
            np.savetxt(handle, rows, fmt="%.7g")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SHARP checkpoint not found: {args.checkpoint}")
    frames = load_keyframes(args.session_root, args.max_views)
    point_budget = max(10_000, args.max_points) // len(frames)

    model = create_predictor()
    weight_stats = load_weights(model, args.checkpoint, verbose=False)
    if weight_stats["loaded"] != 878 or weight_stats["missing"] != 0:
        raise RuntimeError(f"incomplete SHARP MLX weights: {weight_stats}")

    all_parts: list[dict[str, np.ndarray]] = []
    frame_summaries: list[dict] = []
    start_all = time.perf_counter()
    for view_index, frame in enumerate(frames, start=1):
        image_path = Path(frame["files"]["rgb"])
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        intrinsics = frame["intrinsics"]
        focal_px = (float(intrinsics["fx"]) + float(intrinsics["fy"])) * 0.5
        start = time.perf_counter()
        gaussians = infer_camera_gaussians(model, image, focal_px)
        selected, metric_scale, depth_samples = transform_and_select(
            frame, gaussians, point_budget
        )
        elapsed = time.perf_counter() - start
        all_parts.append(selected)
        frame_summaries.append(
            {
                "frameIndex": frame.get("frameIndex"),
                "inferenceSeconds": elapsed,
                "metricScale": metric_scale,
                "depthAlignmentSamples": depth_samples,
                "outputPointCount": len(selected["means"]),
            }
        )
        print(
            f"SHARP MLX view {view_index}/{len(frames)}: "
            f"{len(selected['means']):,} gaussians, "
            f"depth scale {metric_scale:.4f}, {elapsed:.2f}s",
            flush=True,
        )
        del gaussians
        mx.clear_cache()
        gc.collect()

    combined = concatenate(all_parts)
    write_ascii_ply(args.output_ply, combined)
    elapsed_all = time.perf_counter() - start_all
    summary = {
        "schemaVersion": "agriscan-sharp-mlx-inference-v1",
        "backend": "SHARP MLX",
        "mode": "pretrained-feed-forward-inference",
        "device": str(mx.default_device()),
        "checkpoint": str(args.checkpoint),
        "loadedParameterTensors": weight_stats["loaded"],
        "unusedCheckpointTensors": weight_stats["unused"],
        "inferenceViews": len(frames),
        "rawGaussiansPerView": OUTPUT_SIZE * OUTPUT_SIZE * 2,
        "outputPointCount": len(combined["means"]),
        "depthAnchored": True,
        "totalSeconds": elapsed_all,
        "frames": frame_summaries,
        "outputPly": str(args.output_ply),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"SHARP MLX completed: {len(combined['means']):,} gaussians "
        f"from {len(frames)} pretrained forward passes in {elapsed_all:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
