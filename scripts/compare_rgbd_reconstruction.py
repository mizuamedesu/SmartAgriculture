#!/usr/bin/env python3
"""Render reconstructed splats from recorded camera poses for visual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SH_C0 = 0.2820948


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_splats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header_lines = []
        for line in handle:
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        vertex_line = next(
            line for line in header_lines if line.startswith("element vertex ")
        )
        vertex_count = int(vertex_line.split()[-1])
        values = np.loadtxt(handle, dtype=np.float32, max_rows=vertex_count)
    positions = values[:, :3]
    colors = np.clip((0.5 + SH_C0 * values[:, 6:9]) * 255.0, 0, 255).astype(
        np.uint8
    )
    return positions, colors


def render_view(
    positions: np.ndarray,
    colors: np.ndarray,
    pose: dict,
    intrinsics: dict,
) -> Image.Image:
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    rotation = np.asarray(pose["rotation"], dtype=np.float32)
    translation = np.asarray(pose["translation"], dtype=np.float32)
    camera_points = (positions - translation) @ rotation
    depth = -camera_points[:, 2]
    visible = np.isfinite(depth) & (depth > 0.08) & (depth < 8.0)
    camera_points = camera_points[visible]
    visible_colors = colors[visible]
    depth = depth[visible]

    u = (
        float(intrinsics["fx"]) * camera_points[:, 0] / depth
        + float(intrinsics["ppx"])
    )
    v = (
        float(intrinsics["ppy"])
        - float(intrinsics["fy"]) * camera_points[:, 1] / depth
    )
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = np.rint(u[inside]).astype(np.int32)
    v = np.rint(v[inside]).astype(np.int32)
    depth = depth[inside]
    visible_colors = visible_colors[inside]

    canvas = Image.new("RGB", (width, height), (9, 9, 11))
    draw = ImageDraw.Draw(canvas)
    # Draw far-to-near so the nearest splat wins where points overlap.
    for index in np.argsort(depth)[::-1]:
        x = int(u[index])
        y = int(v[index])
        radius = max(1, min(5, int(round(3.2 / max(0.7, depth[index])))))
        color = tuple(int(value) for value in visible_colors[index])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def main() -> None:
    args = parse_args()
    positions, colors = load_splats(args.ply)
    with args.poses.open("r", encoding="utf-8") as handle:
        trajectory = json.load(handle)
    poses = {
        int(frame["frameIndex"]): frame["cameraToWorld"]
        for frame in trajectory["frames"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metadata_path in sorted(args.samples_dir.glob("frame_*_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        frame_index = int(metadata["frameIndex"])
        pose = poses[frame_index]
        rendered = render_view(positions, colors, pose, metadata["intrinsics"])
        rgb_path = Path(metadata["files"]["rgb"])
        recorded = Image.open(rgb_path).convert("RGB")
        if recorded.size != rendered.size:
            recorded = recorded.resize(rendered.size, Image.Resampling.BILINEAR)
        comparison = Image.new(
            "RGB", (rendered.width * 2, rendered.height), (255, 255, 255)
        )
        comparison.paste(recorded, (0, 0))
        comparison.paste(rendered, (rendered.width, 0))
        output_path = args.output_dir / f"frame_{frame_index:06}_comparison.png"
        comparison.save(output_path)
        print(output_path)


if __name__ == "__main__":
    main()
