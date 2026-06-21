use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use base64::{Engine, engine::general_purpose};
use chrono::{DateTime, Local};
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
use png::{BitDepth, ColorType, Encoder as PngEncoder};
use serde::Serialize;

use crate::capture::{
    ColorFrame, DepthFrame, DepthStats, FramePaths, FrameSummary, Intrinsics,
    ResolvedCaptureConfig, SensorFrame, default_output_root,
};

const PREVIEW_MAX_WIDTH: u32 = 320;
const PREVIEW_MAX_HEIGHT: u32 = 180;
const PREVIEW_JPEG_QUALITY: u8 = 58;

#[derive(Debug, Clone)]
pub struct SessionPaths {
    pub session_id: String,
    pub root: PathBuf,
    rgb_dir: PathBuf,
    depth_dir: PathBuf,
    pointcloud_dir: PathBuf,
    meta_dir: PathBuf,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionManifest<'a> {
    schema_version: &'static str,
    session_id: &'a str,
    created_at: String,
    target_label: &'a str,
    cultivar: &'a str,
    backend: &'a str,
    config: &'a ResolvedCaptureConfig,
    folders: SessionFolders,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionFolders {
    rgb: String,
    depth: String,
    point_cloud: String,
    metadata: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionSummary<'a> {
    schema_version: &'static str,
    session_id: &'a str,
    finished_at: String,
    status: &'a str,
    backend: &'a str,
    frames_written: u32,
    config: &'a ResolvedCaptureConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FrameMetadata<'a> {
    schema_version: &'static str,
    session_id: &'a str,
    frame_index: u32,
    frame_number: u64,
    timestamp_ms: f64,
    intrinsics: Intrinsics,
    depth_units_m: f32,
    depth: DepthStats,
    files: &'a FramePaths,
}

#[derive(Debug)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
    r: u8,
    g: u8,
    b: u8,
}

pub fn create_session(
    config: &ResolvedCaptureConfig,
    backend: &str,
) -> Result<SessionPaths, String> {
    let timestamp = Local::now();
    let session_id = format!(
        "{}_{}",
        timestamp.format("%Y%m%d_%H%M%S"),
        sanitize_id(&config.target_label)
    );

    let root = default_output_root()?.join(&session_id);
    let rgb_dir = root.join("rgb");
    let depth_dir = root.join("depth_z16");
    let pointcloud_dir = root.join("pointcloud_ply");
    let meta_dir = root.join("metadata");

    for dir in [&root, &rgb_dir, &depth_dir, &pointcloud_dir, &meta_dir] {
        fs::create_dir_all(dir).map_err(|error| format!("failed to create {dir:?}: {error}"))?;
    }

    let paths = SessionPaths {
        session_id,
        root,
        rgb_dir,
        depth_dir,
        pointcloud_dir,
        meta_dir,
    };
    write_initial_manifest(&paths, config, backend, timestamp)?;
    write_frames_csv_header(&paths)?;
    Ok(paths)
}

pub fn write_frame(
    session: &SessionPaths,
    config: &ResolvedCaptureConfig,
    frame_index: u32,
    frame: &SensorFrame,
) -> Result<FrameSummary, String> {
    let stem = format!("frame_{frame_index:06}");
    let rgb_path = frame
        .color
        .as_ref()
        .map(|_| session.rgb_dir.join(format!("{stem}_rgb.png")));
    let depth_path = session.depth_dir.join(format!("{stem}_depth_z16.png"));
    let ply_path = session.pointcloud_dir.join(format!("{stem}_cloud.ply"));
    let meta_path = session.meta_dir.join(format!("{stem}.json"));

    let color_preview = if let (Some(color), Some(path)) = (&frame.color, &rgb_path) {
        let png = encode_rgb_png(color)?;
        fs::write(path, &png).map_err(|error| format!("failed to write RGB PNG: {error}"))?;
        let preview_jpeg = encode_rgb_preview_jpeg(color)?;
        Some(data_url("image/jpeg", &preview_jpeg))
    } else {
        None
    };

    let depth_preview_jpeg = encode_depth_preview_jpeg(&frame.depth, config)?;
    let depth_preview = data_url("image/jpeg", &depth_preview_jpeg);
    let depth_z16_png = encode_depth_z16_png(&frame.depth)?;
    fs::write(&depth_path, depth_z16_png)
        .map_err(|error| format!("failed to write depth PNG: {error}"))?;

    let stats = preview_depth_stats(&frame.depth, config);
    let point_count = write_ply(&ply_path, frame, config)?;
    let summary_paths = FramePaths {
        rgb: rgb_path.as_ref().map(path_string),
        depth: path_string(&depth_path),
        point_cloud: path_string(&ply_path),
        metadata: path_string(&meta_path),
    };

    let metadata = FrameMetadata {
        schema_version: "tomato-rgbd-frame-v1",
        session_id: &session.session_id,
        frame_index,
        frame_number: frame.frame_number,
        timestamp_ms: frame.timestamp_ms,
        intrinsics: frame.intrinsics,
        depth_units_m: frame.depth.units_m,
        depth: DepthStats {
            valid_points: point_count,
            ..stats.clone()
        },
        files: &summary_paths,
    };
    write_json(&meta_path, &metadata)?;
    append_frames_csv(session, frame_index, frame, &stats, point_count, &summary_paths)?;

    Ok(FrameSummary {
        session_id: session.session_id.clone(),
        frame_index,
        timestamp_ms: frame.timestamp_ms,
        frame_number: frame.frame_number,
        color_preview_data_url: color_preview,
        depth_preview_data_url: depth_preview,
        depth: DepthStats {
            valid_points: point_count,
            ..stats
        },
        paths: summary_paths,
    })
}

pub fn preview_frame_summary(
    session_id: &str,
    config: &ResolvedCaptureConfig,
    frame_index: u32,
    frame: &SensorFrame,
) -> Result<FrameSummary, String> {
    let color_preview = frame
        .color
        .as_ref()
        .map(encode_rgb_preview_jpeg)
        .transpose()?
        .map(|jpeg| data_url("image/jpeg", &jpeg));

    let depth_preview_jpeg = encode_depth_preview_jpeg(&frame.depth, config)?;
    let depth_preview = data_url("image/jpeg", &depth_preview_jpeg);
    let stats = depth_stats(&frame.depth, config);

    Ok(FrameSummary {
        session_id: session_id.to_string(),
        frame_index,
        timestamp_ms: frame.timestamp_ms,
        frame_number: frame.frame_number,
        color_preview_data_url: color_preview,
        depth_preview_data_url: depth_preview,
        depth: stats,
        paths: FramePaths {
            rgb: None,
            depth: "-".to_string(),
            point_cloud: "-".to_string(),
            metadata: "-".to_string(),
        },
    })
}

pub fn finish_session(
    session: &SessionPaths,
    config: &ResolvedCaptureConfig,
    backend: &str,
    status: &str,
    frames_written: u32,
) -> Result<(), String> {
    let summary = SessionSummary {
        schema_version: "tomato-rgbd-session-summary-v1",
        session_id: &session.session_id,
        finished_at: Local::now().to_rfc3339(),
        status,
        backend,
        frames_written,
        config,
    };
    write_json(&session.root.join("session_summary.json"), &summary)
}

fn write_initial_manifest(
    session: &SessionPaths,
    config: &ResolvedCaptureConfig,
    backend: &str,
    timestamp: DateTime<Local>,
) -> Result<(), String> {
    let manifest = SessionManifest {
        schema_version: "tomato-rgbd-session-v1",
        session_id: &session.session_id,
        created_at: timestamp.to_rfc3339(),
        target_label: &config.target_label,
        cultivar: &config.cultivar,
        backend,
        config,
        folders: SessionFolders {
            rgb: path_string(&session.rgb_dir),
            depth: path_string(&session.depth_dir),
            point_cloud: path_string(&session.pointcloud_dir),
            metadata: path_string(&session.meta_dir),
        },
    };
    write_json(&session.root.join("dataset_manifest.json"), &manifest)
}

fn write_frames_csv_header(session: &SessionPaths) -> Result<(), String> {
    let mut file = File::create(session.root.join("frames.csv"))
        .map_err(|error| format!("failed to create frames.csv: {error}"))?;
    writeln!(
        file,
        "frame_index,frame_number,timestamp_ms,valid_points,min_depth_m,max_depth_m,mean_depth_m,rgb_path,depth_path,point_cloud_path,metadata_path"
    )
    .map_err(|error| format!("failed to write frames.csv header: {error}"))
}

fn append_frames_csv(
    session: &SessionPaths,
    frame_index: u32,
    frame: &SensorFrame,
    stats: &DepthStats,
    point_count: usize,
    paths: &FramePaths,
) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(session.root.join("frames.csv"))
        .map_err(|error| format!("failed to open frames.csv: {error}"))?;

    writeln!(
        file,
        "{},{},{:.3},{},{:.5},{:.5},{:.5},{},{},{},{}",
        frame_index,
        frame.frame_number,
        frame.timestamp_ms,
        point_count,
        stats.min_m,
        stats.max_m,
        stats.mean_m,
        csv_field(paths.rgb.as_deref().unwrap_or("")),
        csv_field(&paths.depth),
        csv_field(&paths.point_cloud),
        csv_field(&paths.metadata),
    )
    .map_err(|error| format!("failed to append frames.csv: {error}"))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize JSON: {error}"))?;
    fs::write(path, json).map_err(|error| format!("failed to write {path:?}: {error}"))
}

fn encode_rgb_png(color: &ColorFrame) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = PngEncoder::new(&mut bytes, color.width, color.height);
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|error| format!("failed to create RGB PNG: {error}"))?;
        writer
            .write_image_data(&color.rgb)
            .map_err(|error| format!("failed to encode RGB PNG: {error}"))?;
    }
    Ok(bytes)
}

fn encode_rgb_preview_jpeg(color: &ColorFrame) -> Result<Vec<u8>, String> {
    let (width, height) = preview_dimensions(color.width, color.height);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        let sy = (y * color.height as usize / height as usize).min(color.height as usize - 1);
        for x in 0..width as usize {
            let sx = (x * color.width as usize / width as usize).min(color.width as usize - 1);
            let src = (sy * color.width as usize + sx) * 3;
            let dst = (y * width as usize + x) * 3;
            rgb[dst] = color.rgb[src];
            rgb[dst + 1] = color.rgb[src + 1];
            rgb[dst + 2] = color.rgb[src + 2];
        }
    }

    encode_rgb_jpeg(width, height, &rgb)
}

fn encode_depth_z16_png(depth: &DepthFrame) -> Result<Vec<u8>, String> {
    let mut be = Vec::with_capacity(depth.z16.len() * 2);
    for value in &depth.z16 {
        be.extend_from_slice(&value.to_be_bytes());
    }

    let mut bytes = Vec::new();
    {
        let mut encoder = PngEncoder::new(&mut bytes, depth.width, depth.height);
        encoder.set_color(ColorType::Grayscale);
        encoder.set_depth(BitDepth::Sixteen);
        let mut writer = encoder
            .write_header()
            .map_err(|error| format!("failed to create depth PNG: {error}"))?;
        writer
            .write_image_data(&be)
            .map_err(|error| format!("failed to encode depth PNG: {error}"))?;
    }
    Ok(bytes)
}

fn encode_depth_preview_jpeg(
    depth: &DepthFrame,
    config: &ResolvedCaptureConfig,
) -> Result<Vec<u8>, String> {
    let (width, height) = preview_dimensions(depth.width, depth.height);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    let range = (config.max_depth_m - config.min_depth_m).max(0.01);

    for y in 0..height as usize {
        let sy = (y * depth.height as usize / height as usize).min(depth.height as usize - 1);
        for x in 0..width as usize {
            let sx = (x * depth.width as usize / width as usize).min(depth.width as usize - 1);
            let value = depth.z16[sy * depth.width as usize + sx];
            let meters = value as f32 * depth.units_m;
            let rgb_idx = (y * width as usize + x) * 3;
            if value == 0 || meters < config.min_depth_m || meters > config.max_depth_m {
                rgb[rgb_idx] = 18;
                rgb[rgb_idx + 1] = 22;
                rgb[rgb_idx + 2] = 24;
                continue;
            }

            let t = ((meters - config.min_depth_m) / range).clamp(0.0, 1.0);
            let near = 1.0 - t;
            rgb[rgb_idx] = (42.0 + 210.0 * near) as u8;
            rgb[rgb_idx + 1] = (84.0 + 120.0 * (1.0 - (t - 0.45).abs() * 1.7).max(0.0)) as u8;
            rgb[rgb_idx + 2] = (114.0 + 112.0 * t) as u8;
        }
    }

    encode_rgb_jpeg(width, height, &rgb)
}

fn encode_rgb_jpeg(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let encoder = JpegEncoder::new(&mut bytes, PREVIEW_JPEG_QUALITY);
    encoder
        .encode(rgb, width as u16, height as u16, JpegColorType::Rgb)
        .map_err(|error| format!("failed to encode preview JPEG: {error}"))?;
    Ok(bytes)
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    let scale = (PREVIEW_MAX_WIDTH as f32 / width.max(1) as f32)
        .min(PREVIEW_MAX_HEIGHT as f32 / height.max(1) as f32)
        .min(1.0);
    (
        (width as f32 * scale).round().max(1.0) as u32,
        (height as f32 * scale).round().max(1.0) as u32,
    )
}

fn depth_stats(depth: &DepthFrame, config: &ResolvedCaptureConfig) -> DepthStats {
    let mut valid = 0usize;
    let mut min_m = f32::MAX;
    let mut max_m = 0.0f32;
    let mut sum = 0.0f64;

    for value in &depth.z16 {
        let meters = *value as f32 * depth.units_m;
        if *value == 0 || meters < config.min_depth_m || meters > config.max_depth_m {
            continue;
        }
        valid += 1;
        min_m = min_m.min(meters);
        max_m = max_m.max(meters);
        sum += meters as f64;
    }

    if valid == 0 {
        return DepthStats {
            valid_points: 0,
            min_m: 0.0,
            max_m: 0.0,
            mean_m: 0.0,
        };
    }

    DepthStats {
        valid_points: valid,
        min_m,
        max_m,
        mean_m: (sum / valid as f64) as f32,
    }
}

fn preview_depth_stats(depth: &DepthFrame, config: &ResolvedCaptureConfig) -> DepthStats {
    let sample_step = ((depth.width.max(depth.height) as f32 / 320.0).ceil() as usize).max(1);
    if sample_step == 1 {
        return depth_stats(depth, config);
    }

    let mut sampled = 0usize;
    let mut valid = 0usize;
    let mut min_m = f32::MAX;
    let mut max_m = 0.0f32;
    let mut sum = 0.0f64;

    for y in (0..depth.height as usize).step_by(sample_step) {
        for x in (0..depth.width as usize).step_by(sample_step) {
            sampled += 1;
            let value = depth.z16[y * depth.width as usize + x];
            let meters = value as f32 * depth.units_m;
            if value == 0 || meters < config.min_depth_m || meters > config.max_depth_m {
                continue;
            }
            valid += 1;
            min_m = min_m.min(meters);
            max_m = max_m.max(meters);
            sum += meters as f64;
        }
    }

    if valid == 0 || sampled == 0 {
        return DepthStats {
            valid_points: 0,
            min_m: 0.0,
            max_m: 0.0,
            mean_m: 0.0,
        };
    }

    let total_pixels = (depth.width as usize).saturating_mul(depth.height as usize);
    let estimated_valid = ((valid as f64 / sampled as f64) * total_pixels as f64).round() as usize;
    DepthStats {
        valid_points: estimated_valid,
        min_m,
        max_m,
        mean_m: (sum / valid as f64) as f32,
    }
}

fn write_ply(
    path: &Path,
    frame: &SensorFrame,
    config: &ResolvedCaptureConfig,
) -> Result<usize, String> {
    let points = collect_points(frame, config);
    let file = File::create(path).map_err(|error| format!("failed to create PLY: {error}"))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "ply").map_err(io_error)?;
    writeln!(writer, "format ascii 1.0").map_err(io_error)?;
    writeln!(writer, "comment Tomato Twin Capture RGB-D point cloud").map_err(io_error)?;
    writeln!(writer, "element vertex {}", points.len()).map_err(io_error)?;
    writeln!(writer, "property float x").map_err(io_error)?;
    writeln!(writer, "property float y").map_err(io_error)?;
    writeln!(writer, "property float z").map_err(io_error)?;
    writeln!(writer, "property uchar red").map_err(io_error)?;
    writeln!(writer, "property uchar green").map_err(io_error)?;
    writeln!(writer, "property uchar blue").map_err(io_error)?;
    writeln!(writer, "end_header").map_err(io_error)?;

    for point in &points {
        writeln!(
            writer,
            "{:.6} {:.6} {:.6} {} {} {}",
            point.x, point.y, point.z, point.r, point.g, point.b
        )
        .map_err(io_error)?;
    }

    writer
        .flush()
        .map_err(|error| format!("failed to flush PLY: {error}"))?;
    Ok(points.len())
}

fn collect_points(frame: &SensorFrame, config: &ResolvedCaptureConfig) -> Vec<Point> {
    let depth = &frame.depth;
    let intr = frame.intrinsics;
    let step = config.point_stride as usize;
    let mut points = Vec::with_capacity((depth.z16.len() / step.max(1)).min(100_000));
    let color = frame.color.as_ref();

    for y in (0..depth.height as usize).step_by(step) {
        for x in (0..depth.width as usize).step_by(step) {
            let idx = y * depth.width as usize + x;
            let raw = depth.z16[idx];
            let z = raw as f32 * depth.units_m;
            if raw == 0 || z < config.min_depth_m || z > config.max_depth_m {
                continue;
            }

            let px = (x as f32 - intr.ppx) / intr.fx * z;
            let py = (y as f32 - intr.ppy) / intr.fy * z;
            let (r, g, b) = sample_color(color, x, y, depth.width, depth.height, z, config);
            points.push(Point {
                x: px,
                y: py,
                z,
                r,
                g,
                b,
            });
        }
    }

    points
}

fn sample_color(
    color: Option<&ColorFrame>,
    x: usize,
    y: usize,
    depth_width: u32,
    depth_height: u32,
    z: f32,
    config: &ResolvedCaptureConfig,
) -> (u8, u8, u8) {
    if let Some(color) = color {
        let sx = ((x as f32 / depth_width as f32) * color.width as f32)
            .floor()
            .clamp(0.0, (color.width - 1) as f32) as usize;
        let sy = ((y as f32 / depth_height as f32) * color.height as f32)
            .floor()
            .clamp(0.0, (color.height - 1) as f32) as usize;
        let idx = (sy * color.width as usize + sx) * 3;
        return (color.rgb[idx], color.rgb[idx + 1], color.rgb[idx + 2]);
    }

    let range = (config.max_depth_m - config.min_depth_m).max(0.01);
    let t = ((z - config.min_depth_m) / range).clamp(0.0, 1.0);
    (
        (220.0 * (1.0 - t) + 35.0 * t) as u8,
        (82.0 + 64.0 * t) as u8,
        (54.0 + 155.0 * t) as u8,
    )
}

fn data_url(mime: &str, data: &[u8]) -> String {
    format!("data:{mime};base64,{}", general_purpose::STANDARD.encode(data))
}

fn path_string(path: &PathBuf) -> String {
    path.to_string_lossy().to_string()
}

fn sanitize_id(value: &str) -> String {
    let mut output = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            output.push(ch.to_ascii_lowercase());
        } else if ch == '-' || ch == '_' {
            output.push(ch);
        } else if !output.ends_with('_') {
            output.push('_');
        }
    }
    output.trim_matches('_').to_string()
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn io_error(error: std::io::Error) -> String {
    error.to_string()
}
