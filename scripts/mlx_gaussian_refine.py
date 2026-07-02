#!/usr/bin/env python3
"""High-quality 3DGS training on Apple Silicon with gsplat-mlx.

This is intentionally not a hand-rolled splat fitter. The differentiable
rendering path comes from ``gsplat_mlx.rasterization``; MLX supplies autograd
and Adam updates on Apple Silicon.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import mlx.core as mx
    from gsplat_mlx import __version__ as GSPLAT_MLX_VERSION
    from gsplat_mlx import rasterization
except Exception as exc:  # pragma: no cover - surfaced to the Tauri app.
    raise SystemExit(
        "gsplat-mlx is required for MLX 3DGS training. Install it with:\n"
        "  python3 -m pip install mlx numpy pillow scipy\n"
        "  python3 -m pip install 'git+https://github.com/RobotFlow-Labs/gsplat-mlx.git'\n"
        f"Import error: {exc}"
    )


SH_C0 = 0.2820948


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3DGS with gsplat-mlx on Apple Silicon")
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=350_000)
    parser.add_argument("--radius", type=float, default=0.0035)
    parser.add_argument("--voxel-size", type=float, default=0.0025)
    parser.add_argument("--iterations", type=int, default=1_600)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--turntable-degrees", type=float, default=360.0)
    parser.add_argument("--train-size", type=int, default=320)
    parser.add_argument("--max-train-views", type=int, default=12)
    return parser.parse_args()


def read_ply(path: Path) -> Tuple[List[str], List[List[float]]]:
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"{path} is not an ASCII PLY file")

        vertex_count: Optional[int] = None
        properties: List[str] = []
        in_vertex = False
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "end_header":
                break
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[2])

        if vertex_count is None:
            raise ValueError("PLY file has no vertex element")

        rows: List[List[float]] = []
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) < len(properties):
                raise ValueError("PLY vertex row has fewer values than declared properties")
            rows.append(values)

    return properties, rows


def property_index(properties: Sequence[str], name: str) -> Optional[int]:
    try:
        return properties.index(name)
    except ValueError:
        return None


def color_value(row: Sequence[float], fdc_idx: Optional[int], rgb_idx: Optional[int]) -> float:
    if fdc_idx is not None:
        return min(1.0, max(0.0, row[fdc_idx] * SH_C0 + 0.5))
    if rgb_idx is not None:
        value = row[rgb_idx]
        return min(1.0, max(0.0, value if value <= 1.0 else value / 255.0))
    return 0.7


def scale_value(row: Sequence[float], idx: Optional[int], fallback: float) -> float:
    if idx is None:
        return fallback
    return min(0.2, max(0.0001, math.exp(row[idx])))


def logit(value: float) -> float:
    value = min(0.999, max(0.001, value))
    return math.log(value / (1.0 - value))


class Cell:
    __slots__ = ("count", "sum_xyz", "sum_xyz2", "sum_rgb", "sum_scale", "key")

    def __init__(self, key: Tuple[int, int, int], xyz: Sequence[float], rgb: Sequence[float], scale: Sequence[float]):
        self.count = 0
        self.sum_xyz = [0.0, 0.0, 0.0]
        self.sum_xyz2 = [0.0, 0.0, 0.0]
        self.sum_rgb = [0.0, 0.0, 0.0]
        self.sum_scale = [0.0, 0.0, 0.0]
        self.key = key
        self.add(xyz, rgb, scale)

    def add(self, xyz: Sequence[float], rgb: Sequence[float], scale: Sequence[float]) -> None:
        self.count += 1
        for axis in range(3):
            value = float(xyz[axis])
            self.sum_xyz[axis] += value
            self.sum_xyz2[axis] += value * value
            self.sum_rgb[axis] += float(rgb[axis])
            self.sum_scale[axis] += float(scale[axis])

    def mean_xyz(self) -> List[float]:
        inv = 1.0 / self.count
        return [value * inv for value in self.sum_xyz]

    def mean_rgb(self) -> List[float]:
        inv = 1.0 / self.count
        return [min(1.0, max(0.0, value * inv)) for value in self.sum_rgb]

    def scale(self, radius: float) -> List[float]:
        inv = 1.0 / self.count
        out: List[float] = []
        for axis in range(3):
            mean = self.sum_xyz[axis] * inv
            variance = max(0.0, self.sum_xyz2[axis] * inv - mean * mean)
            observed = math.sqrt(variance) + self.sum_scale[axis] * inv * 0.5
            out.append(min(radius * 5.0, max(radius * 0.25, observed)))
        return out


def seed_cells(
    properties: Sequence[str],
    rows: Iterable[Sequence[float]],
    voxel_size: float,
    radius: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x_idx = property_index(properties, "x")
    y_idx = property_index(properties, "y")
    z_idx = property_index(properties, "z")
    if x_idx is None or y_idx is None or z_idx is None:
        raise ValueError("PLY is missing x/y/z properties")

    fdc = [property_index(properties, f"f_dc_{axis}") for axis in range(3)]
    rgb = [property_index(properties, name) for name in ("red", "green", "blue")]
    scales = [property_index(properties, f"scale_{axis}") for axis in range(3)]

    cells: Dict[Tuple[int, int, int], Cell] = {}
    input_count = 0
    voxel = max(0.0005, voxel_size)
    for row in rows:
        input_count += 1
        xyz = [row[x_idx], row[y_idx], row[z_idx]]
        color = [color_value(row, fdc[axis], rgb[axis]) for axis in range(3)]
        scale = [scale_value(row, scales[axis], radius) for axis in range(3)]
        key = tuple(math.floor(value / voxel) for value in xyz)
        cell = cells.get(key)
        if cell is None:
            cells[key] = Cell(key, xyz, color, scale)
        else:
            cell.add(xyz, color, scale)

    ranked = sorted(cells.values(), key=lambda item: (-item.count, item.key))
    if len(ranked) > max_points:
        ranked = ranked[:max_points]

    means = np.array([cell.mean_xyz() for cell in ranked], dtype=np.float32)
    colors = np.array([cell.mean_rgb() for cell in ranked], dtype=np.float32)
    scale_array = np.array([cell.scale(radius) for cell in ranked], dtype=np.float32)
    return means, colors, scale_array, input_count


def load_frame_metadata(session_root: Path, frame_stride: int, max_views: int) -> List[dict]:
    metadata_dir = session_root / "metadata"
    frames: List[dict] = []
    for path in sorted(metadata_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            frame = json.load(handle)
        if frame.get("files", {}).get("rgb"):
            frames.append(frame)

    frames.sort(key=lambda item: (item.get("frameIndex", 0), item.get("frameNumber", 0)))
    stride = max(1, frame_stride)
    selected = [frame for idx, frame in enumerate(frames) if idx % stride == 0]
    if max_views > 0 and len(selected) > max_views:
        indices = np.linspace(0, len(selected) - 1, max_views).round().astype(int)
        selected = [selected[int(index)] for index in indices]
    if not selected:
        raise ValueError("no RGB frames found for gsplat-mlx training")
    return selected


def resized_target_intrinsics_and_depth(frame: dict, train_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    rgb_path = Path(frame["files"]["rgb"])
    image = Image.open(rgb_path).convert("RGB")
    width, height = image.size
    max_side = max(64, train_size)
    scale = min(1.0, max_side / max(width, height))
    out_w = max(16, int(round(width * scale)))
    out_h = max(16, int(round(height * scale)))
    if (out_w, out_h) != (width, height):
        image = image.resize((out_w, out_h), Image.Resampling.LANCZOS)

    target = np.asarray(image, dtype=np.float32) / 255.0
    depth_path = Path(frame["files"]["depth"])
    depth_image = Image.open(depth_path)
    depth_raw = np.asarray(depth_image, dtype=np.float32)
    depth_units = float(frame.get("depthUnitsM", 0.001))
    depth_m = depth_raw * depth_units
    if depth_m.shape[1] != width or depth_m.shape[0] != height:
        depth_src = Image.fromarray(depth_m.astype(np.float32), mode="F")
        depth_src = depth_src.resize((width, height), Image.Resampling.NEAREST)
        depth_m = np.asarray(depth_src, dtype=np.float32)
    if (out_w, out_h) != (width, height):
        depth_src = Image.fromarray(depth_m.astype(np.float32), mode="F")
        depth_src = depth_src.resize((out_w, out_h), Image.Resampling.NEAREST)
        depth_m = np.asarray(depth_src, dtype=np.float32)
    depth_m = np.where((depth_m >= 0.02) & (depth_m <= 8.0), depth_m, 0.0)[..., None]

    intr = frame["intrinsics"]
    K = np.array(
        [
            [float(intr["fx"]) * (out_w / width), 0.0, float(intr["ppx"]) * (out_w / width)],
            [0.0, float(intr["fy"]) * (out_h / height), float(intr["ppy"]) * (out_h / height)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return target, K, depth_m.astype(np.float32), out_w, out_h


def view_matrix_for_turntable(view_index: int, view_count: int, turntable_degrees: float) -> np.ndarray:
    if abs(turntable_degrees) < 1.0e-6:
        angle = 0.0
    else:
        denom = max(1, view_count - 1)
        angle = (view_index / denom) * math.radians(turntable_degrees)
    c = math.cos(-angle)
    s = math.sin(-angle)
    rot_y_inv = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    camera_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = camera_flip @ rot_y_inv
    return view


def load_training_views(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = load_frame_metadata(args.session_root, args.frame_stride, args.max_train_views)
    targets: List[np.ndarray] = []
    depth_targets: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    viewmats: List[np.ndarray] = []
    output_size: Optional[Tuple[int, int]] = None

    for view_index, frame in enumerate(frames):
        target, K, depth_target, width, height = resized_target_intrinsics_and_depth(frame, args.train_size)
        if output_size is None:
            output_size = (width, height)
        elif output_size != (width, height):
            old_w, old_h = width, height
            image = Image.fromarray(np.clip(target * 255.0, 0, 255).astype(np.uint8))
            width, height = output_size
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            target = np.asarray(image, dtype=np.float32) / 255.0
            depth_image = Image.fromarray(depth_target[..., 0].astype(np.float32), mode="F")
            depth_image = depth_image.resize((width, height), Image.Resampling.NEAREST)
            depth_target = np.asarray(depth_image, dtype=np.float32)[..., None]
            sx = width / old_w
            sy = height / old_h
            K[0, :] *= sx
            K[1, :] *= sy

        targets.append(target)
        depth_targets.append(depth_target)
        intrinsics.append(K)
        viewmats.append(view_matrix_for_turntable(view_index, len(frames), args.turntable_degrees))

    return (
        np.stack(targets).astype(np.float32),
        np.stack(depth_targets).astype(np.float32),
        np.stack(intrinsics).astype(np.float32),
        np.stack(viewmats).astype(np.float32),
    )


def normalize_quats(quats: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(norm, mx.array(1.0e-6, dtype=mx.float32))


def adam_update(
    params: Dict[str, mx.array],
    grads: Dict[str, mx.array],
    state: Dict[str, Dict[str, mx.array]],
    lrs: Dict[str, float],
    step: int,
) -> None:
    beta1, beta2, eps = 0.9, 0.999, 1.0e-8
    for name, param in list(params.items()):
        grad = grads[name]
        if name not in state:
            state[name] = {"m": mx.zeros_like(param), "v": mx.zeros_like(param)}
        state[name]["m"] = beta1 * state[name]["m"] + (1.0 - beta1) * grad
        state[name]["v"] = beta2 * state[name]["v"] + (1.0 - beta2) * grad * grad
        m_hat = state[name]["m"] / (1.0 - beta1**step)
        v_hat = state[name]["v"] / (1.0 - beta2**step)
        params[name] = param - lrs[name] * m_hat / (mx.sqrt(v_hat) + eps)


def train_with_gsplat(
    means_np: np.ndarray,
    colors_np: np.ndarray,
    scales_np: np.ndarray,
    targets_np: np.ndarray,
    depth_targets_np: np.ndarray,
    Ks_np: np.ndarray,
    viewmats_np: np.ndarray,
    radius: float,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    view_count, height, width, _ = targets_np.shape
    n = means_np.shape[0]
    params = {
        "means": mx.array(means_np, dtype=mx.float32),
        "log_scales": mx.array(np.log(np.maximum(scales_np, 1.0e-5)), dtype=mx.float32),
        "quats": mx.concatenate([mx.ones((n, 1)), mx.zeros((n, 3))], axis=1),
        "opacity_logits": mx.full((n,), logit(0.72), dtype=mx.float32),
        "color_logits": mx.array(np.vectorize(logit)(np.clip(colors_np, 0.001, 0.999)), dtype=mx.float32),
    }
    targets = mx.array(targets_np, dtype=mx.float32)
    depth_targets = mx.array(depth_targets_np, dtype=mx.float32)
    Ks = mx.array(Ks_np, dtype=mx.float32)
    viewmats = mx.array(viewmats_np, dtype=mx.float32)
    backgrounds = mx.zeros((view_count, 3), dtype=mx.float32)
    depth_scale = max(0.02, radius * 24.0)

    names = ["means", "log_scales", "quats", "opacity_logits", "color_logits"]
    base_lr = max(5.0e-5, min(1.0e-3, radius * 0.18))
    lrs = {
        "means": base_lr,
        "log_scales": base_lr * 0.45,
        "quats": base_lr * 0.08,
        "opacity_logits": base_lr * 0.35,
        "color_logits": base_lr * 0.6,
    }
    state: Dict[str, Dict[str, mx.array]] = {}
    final_loss: Optional[float] = None

    def loss_fn(means, log_scales, quats, opacity_logits, color_logits):
        render, alpha, _ = rasterization(
            means=means,
            quats=normalize_quats(quats),
            scales=mx.exp(log_scales),
            opacities=mx.sigmoid(opacity_logits),
            colors=mx.sigmoid(color_logits),
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            backgrounds=backgrounds,
            render_mode="RGB+D",
            sh_degree=None,
            rasterize_mode="antialiased",
            differentiable=True,
        )
        rendered_rgb = render[..., :3]
        rendered_depth = render[..., 3:4]
        valid_depth = (depth_targets > 0.02).astype(mx.float32)
        rgb_loss = mx.mean(mx.abs(rendered_rgb - targets))
        depth_loss = mx.sum(mx.abs(rendered_depth - depth_targets) * valid_depth / depth_scale) / mx.maximum(
            mx.sum(valid_depth), mx.array(1.0, dtype=mx.float32)
        )
        coverage_loss = mx.mean((1.0 - alpha) * mx.maximum(mx.mean(targets, axis=-1, keepdims=True), 0.05))
        scale_reg = mx.mean(mx.square(mx.maximum(mx.exp(log_scales) - radius * 6.0, 0.0) / radius))
        opacity_reg = mx.mean(mx.square(mx.sigmoid(opacity_logits) - 0.72))
        return rgb_loss + 0.18 * depth_loss + 0.08 * coverage_loss + 0.01 * scale_reg + 0.002 * opacity_reg

    t0 = time.time()
    steps = max(0, iterations)
    for step in range(1, steps + 1):
        loss, grads_tuple = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
            params["means"],
            params["log_scales"],
            params["quats"],
            params["opacity_logits"],
            params["color_logits"],
        )
        grads = dict(zip(names, grads_tuple))
        adam_update(params, grads, state, lrs, step)
        if step == steps or step % 25 == 0:
            mx.eval(loss, *params.values())
            final_loss = float(loss)
            elapsed = time.time() - t0
            print(f"gsplat-mlx step {step}/{steps}: loss={final_loss:.6f}, elapsed={elapsed:.1f}s", flush=True)

    if steps == 0:
        final_loss = None

    quats = normalize_quats(params["quats"])
    scales = mx.exp(params["log_scales"])
    opacities = mx.sigmoid(params["opacity_logits"])
    colors = mx.sigmoid(params["color_logits"])
    mx.eval(params["means"], scales, quats, opacities, colors)
    return (
        np.array(params["means"]),
        np.array(scales),
        np.array(quats),
        np.array(opacities),
        np.array(colors),
        final_loss,
    )


def write_ply(path: Path, means: np.ndarray, scales: np.ndarray, quats: np.ndarray, opacities: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment Tomato Twin Capture trained with gsplat-mlx differentiable rasterization\n")
        handle.write(f"element vertex {len(means)}\n")
        for prop in (
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ):
            handle.write(f"property float {prop}\n")
        handle.write("end_header\n")
        for mean, scale, quat, opacity, color in zip(means, scales, quats, opacities, colors):
            fdc = [(float(channel) - 0.5) / SH_C0 for channel in color]
            log_scale = [math.log(max(0.0001, float(axis))) for axis in scale]
            opacity_logit = logit(float(opacity))
            handle.write(
                "{:.6f} {:.6f} {:.6f} 0 0 0 {:.6f} {:.6f} {:.6f} {:.6f} "
                "{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    float(mean[0]), float(mean[1]), float(mean[2]),
                    fdc[0], fdc[1], fdc[2], opacity_logit,
                    log_scale[0], log_scale[1], log_scale[2],
                    float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                )
            )


def main() -> None:
    args = parse_args()
    properties, rows = read_ply(args.input_ply)
    means, colors, scales, input_count = seed_cells(
        properties, rows, args.voxel_size, args.radius, max(1, args.max_points)
    )
    if len(means) == 0:
        raise ValueError("input PLY produced no valid Gaussian seeds")

    targets, depth_targets, Ks, viewmats = load_training_views(args)
    means, scales, quats, opacities, colors, final_loss = train_with_gsplat(
        means, colors, scales, targets, depth_targets, Ks, viewmats, args.radius, args.iterations
    )
    write_ply(args.output_ply, means, scales, quats, opacities, colors)

    summary = {
        "schemaVersion": "tomato-gsplat-mlx-train-v1",
        "backend": "gsplat-mlx",
        "gsplatMlxVersion": GSPLAT_MLX_VERSION,
        "device": str(mx.default_device()),
        "inputPointCount": input_count,
        "outputPointCount": int(len(means)),
        "trainViews": int(targets.shape[0]),
        "trainWidth": int(targets.shape[2]),
        "trainHeight": int(targets.shape[1]),
        "depthSupervision": True,
        "voxelSizeM": args.voxel_size,
        "radiusM": args.radius,
        "iterations": max(0, args.iterations),
        "finalLoss": final_loss,
        "outputPly": str(args.output_ply),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
