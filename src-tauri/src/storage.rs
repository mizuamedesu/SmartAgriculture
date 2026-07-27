use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use base64::{Engine, engine::general_purpose};
use chrono::Local;
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};

use crate::{
    capture::{
        ColorFrame, DepthFrame, DepthStats, FramePaths, FrameSummary, ResolvedCaptureConfig,
        SensorFrame, default_output_root,
    },
    mcap_io::{self, McapRecorder},
};

const PREVIEW_MAX_WIDTH: u32 = 320;
const PREVIEW_MAX_HEIGHT: u32 = 180;
const PREVIEW_JPEG_QUALITY: u8 = 58;

#[derive(Clone)]
pub struct SessionPaths {
    pub session_id: String,
    pub root: PathBuf,
    backend: String,
    recording_path: PathBuf,
    recorder: Arc<Mutex<Option<McapRecorder>>>,
}

pub fn create_session(
    config: &ResolvedCaptureConfig,
    backend: &str,
) -> Result<SessionPaths, String> {
    let output_root = match config.output_root.as_deref() {
        Some(path) => PathBuf::from(path),
        None => default_output_root()?,
    };
    fs::create_dir_all(&output_root)
        .map_err(|error| format!("failed to create save location {output_root:?}: {error}"))?;
    create_session_at(&output_root, config, backend)
}

pub fn create_session_at(
    output_root: &Path,
    config: &ResolvedCaptureConfig,
    backend: &str,
) -> Result<SessionPaths, String> {
    let file_stem = sanitize_file_stem(&config.target_label);
    let session_id = format!("{}_{}", Local::now().format("%Y%m%d_%H%M%S"), file_stem);
    let root = output_root.join(&session_id);
    fs::create_dir_all(&root)
        .map_err(|error| format!("failed to create recording session {root:?}: {error}"))?;
    Ok(SessionPaths {
        recording_path: root.join(format!("{file_stem}.mcap")),
        session_id,
        root,
        backend: backend.to_string(),
        recorder: Arc::new(Mutex::new(None)),
    })
}

pub fn open_session(session_id: String, root: PathBuf) -> Result<SessionPaths, String> {
    if !root.is_dir() {
        return Err(format!(
            "recording session directory is missing: {}",
            root.to_string_lossy()
        ));
    }
    Ok(SessionPaths {
        recording_path: mcap_io::recording_path(&root),
        session_id,
        root,
        backend: "realsense".to_string(),
        recorder: Arc::new(Mutex::new(None)),
    })
}

pub fn session_recording_path(session: &SessionPaths) -> PathBuf {
    session.recording_path.clone()
}

pub fn finalize_recording_file(root: &Path, final_path: &Path) -> Result<(), String> {
    let helper_path = mcap_io::recording_path(root);
    if helper_path == final_path || !helper_path.is_file() {
        return Ok(());
    }
    if final_path.exists() {
        return Err(format!(
            "cannot rename MCAP because the destination already exists: {}",
            final_path.to_string_lossy()
        ));
    }
    fs::rename(&helper_path, final_path).map_err(|error| {
        format!(
            "failed to rename MCAP to {}: {error}",
            final_path.to_string_lossy()
        )
    })
}

pub fn write_frame(
    session: &SessionPaths,
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
    let depth_preview = data_url(
        "image/jpeg",
        &encode_depth_preview_jpeg(&frame.depth, config)?,
    );

    let mut recorder_guard = session
        .recorder
        .lock()
        .map_err(|_| "MCAP recorder state is locked".to_string())?;
    if recorder_guard.is_none() {
        *recorder_guard = Some(McapRecorder::create(
            session.recording_path.clone(),
            &session.session_id,
            &session.backend,
            config,
        )?);
    }
    let (stats, _point_count) = recorder_guard
        .as_mut()
        .ok_or_else(|| "MCAP recorder did not initialize".to_string())?
        .write_frame(frame_index, frame)?;

    let recording = path_string(&session.recording_path);
    Ok(FrameSummary {
        session_id: session.session_id.clone(),
        frame_index,
        timestamp_ms: frame.timestamp_ms,
        frame_number: frame.frame_number,
        color_preview_data_url: color_preview,
        depth_preview_data_url: depth_preview,
        depth: stats,
        paths: FramePaths {
            rgb: frame
                .color
                .as_ref()
                .map(|_| format!("{recording}#{}", mcap_io::TOPIC_COLOR)),
            depth: format!("{recording}#{}", mcap_io::TOPIC_DEPTH),
            point_cloud: format!("{recording}#{}", mcap_io::TOPIC_POINTS),
            metadata: format!("{recording}#{}", mcap_io::TOPIC_FRAME_INFO),
        },
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
    let depth_preview = data_url(
        "image/jpeg",
        &encode_depth_preview_jpeg(&frame.depth, config)?,
    );
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
    _backend: &str,
    status: &str,
    frames_written: u32,
) -> Result<(), String> {
    let mut recorder_guard = session
        .recorder
        .lock()
        .map_err(|_| "MCAP recorder state is locked".to_string())?;
    if recorder_guard.is_none() {
        *recorder_guard = Some(McapRecorder::create(
            session.recording_path.clone(),
            &session.session_id,
            &session.backend,
            config,
        )?);
    }
    let recorder = recorder_guard
        .take()
        .ok_or_else(|| "MCAP recorder did not initialize".to_string())?;
    recorder.finish(status, frames_written)?;
    Ok(())
}

fn encode_rgb_preview_jpeg(color: &ColorFrame) -> Result<Vec<u8>, String> {
    let (width, height) = preview_dimensions(color.width, color.height);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        let source_y = (y * color.height as usize / height as usize).min(color.height as usize - 1);
        for x in 0..width as usize {
            let source_x =
                (x * color.width as usize / width as usize).min(color.width as usize - 1);
            let source = (source_y * color.width as usize + source_x) * 3;
            let destination = (y * width as usize + x) * 3;
            rgb[destination..destination + 3].copy_from_slice(&color.rgb[source..source + 3]);
        }
    }
    encode_rgb_jpeg(width, height, &rgb)
}

fn encode_depth_preview_jpeg(
    depth: &DepthFrame,
    config: &ResolvedCaptureConfig,
) -> Result<Vec<u8>, String> {
    let (width, height) = preview_dimensions(depth.width, depth.height);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    let range = (config.max_depth_m - config.min_depth_m).max(0.01);

    for y in 0..height as usize {
        let source_y = (y * depth.height as usize / height as usize).min(depth.height as usize - 1);
        for x in 0..width as usize {
            let source_x =
                (x * depth.width as usize / width as usize).min(depth.width as usize - 1);
            let value = depth.z16[source_y * depth.width as usize + source_x];
            let meters = value as f32 * depth.units_m;
            let index = (y * width as usize + x) * 3;
            if value == 0 || meters < config.min_depth_m || meters > config.max_depth_m {
                rgb[index..index + 3].copy_from_slice(&[18, 22, 24]);
                continue;
            }
            let t = ((meters - config.min_depth_m) / range).clamp(0.0, 1.0);
            let near = 1.0 - t;
            rgb[index] = (42.0 + 210.0 * near) as u8;
            rgb[index + 1] = (84.0 + 120.0 * (1.0 - (t - 0.45).abs() * 1.7).max(0.0)) as u8;
            rgb[index + 2] = (114.0 + 112.0 * t) as u8;
        }
    }
    encode_rgb_jpeg(width, height, &rgb)
}

fn encode_rgb_jpeg(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    JpegEncoder::new(&mut bytes, PREVIEW_JPEG_QUALITY)
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

fn data_url(mime: &str, data: &[u8]) -> String {
    format!(
        "data:{mime};base64,{}",
        general_purpose::STANDARD.encode(data)
    )
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn sanitize_file_stem(value: &str) -> String {
    let trimmed = value.trim();
    let base = if trimmed.to_ascii_lowercase().ends_with(".mcap") {
        &trimmed[..trimmed.len().saturating_sub(5)]
    } else {
        trimmed
    };
    let mut output = String::new();
    for character in base.chars() {
        if character.is_alphanumeric() {
            output.push(character);
        } else if character == '-' || character == '_' {
            output.push(character);
        } else if !output.ends_with('_') {
            output.push('_');
        }
    }
    let sanitized = output.trim_matches('_');
    if sanitized.is_empty() {
        "scan".to_string()
    } else {
        sanitized.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{finalize_recording_file, sanitize_file_stem};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn file_name_becomes_a_safe_mcap_stem() {
        assert_eq!(sanitize_file_stem("Field 01.mcap"), "Field_01");
        assert_eq!(sanitize_file_stem("トマト A"), "トマト_A");
        assert_eq!(sanitize_file_stem("  "), "scan");
    }

    #[test]
    fn privileged_helper_output_is_renamed_to_the_requested_file_name() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agriscan-rename-{stamp}"));
        fs::create_dir_all(&root).expect("create test session");
        fs::write(root.join("recording.mcap"), b"mcap").expect("write helper output");
        let final_path = root.join("Field_01.mcap");
        finalize_recording_file(&root, &final_path).expect("rename helper output");
        assert_eq!(fs::read(&final_path).expect("read renamed MCAP"), b"mcap");
        assert!(!root.join("recording.mcap").exists());
        let _ = fs::remove_dir_all(root);
    }
}
