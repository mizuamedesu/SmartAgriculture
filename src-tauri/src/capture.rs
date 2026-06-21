use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

use crate::{realsense, storage};

#[derive(Default)]
pub struct AppState {
    recording: Mutex<Option<RecordingHandle>>,
}

struct RecordingHandle {
    session_id: String,
    root: PathBuf,
    backend: String,
    stop: Arc<AtomicBool>,
    frames_written: Arc<AtomicU32>,
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureConfig {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
    pub backend: Option<String>,
    pub target_label: Option<String>,
    pub cultivar: Option<String>,
    pub notes: Option<String>,
    pub max_frames: Option<u32>,
    pub point_stride: Option<u32>,
    pub min_depth_m: Option<f32>,
    pub max_depth_m: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedCaptureConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub backend: String,
    pub target_label: String,
    pub cultivar: String,
    pub notes: String,
    pub max_frames: Option<u32>,
    pub point_stride: u32,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
}

impl CaptureConfig {
    fn resolve(self) -> ResolvedCaptureConfig {
        let width = self.width.unwrap_or(640).clamp(320, 1280);
        let height = self.height.unwrap_or(480).clamp(240, 720);
        let fps = self.fps.unwrap_or(6).clamp(1, 30);
        let backend = self.backend.unwrap_or_else(|| "auto".to_string());
        let point_stride = self.point_stride.unwrap_or(4).clamp(1, 12);
        let min_depth_m = self.min_depth_m.unwrap_or(0.12).clamp(0.02, 4.0);
        let max_depth_m = self.max_depth_m.unwrap_or(1.4).clamp(min_depth_m + 0.01, 8.0);
        let max_frames = self.max_frames.and_then(|value| (value > 0).then_some(value));

        ResolvedCaptureConfig {
            width,
            height,
            fps,
            backend,
            target_label: non_empty(self.target_label, "mini_tomato"),
            cultivar: non_empty(self.cultivar, "unknown"),
            notes: self.notes.unwrap_or_default(),
            max_frames,
            point_stride,
            min_depth_m,
            max_depth_m,
        }
    }
}

fn non_empty(value: Option<String>, fallback: &str) -> String {
    value
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

pub trait CameraBackend: Send {
    fn capture_frame(&mut self) -> Result<SensorFrame, String>;
}

#[derive(Debug, Clone)]
pub struct SensorFrame {
    pub color: Option<ColorFrame>,
    pub depth: DepthFrame,
    pub intrinsics: Intrinsics,
    pub timestamp_ms: f64,
    pub frame_number: u64,
}

#[derive(Debug, Clone)]
pub struct ColorFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct DepthFrame {
    pub width: u32,
    pub height: u32,
    pub z16: Vec<u16>,
    pub units_m: f32,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Intrinsics {
    pub width: u32,
    pub height: u32,
    pub ppx: f32,
    pub ppy: f32,
    pub fx: f32,
    pub fy: f32,
    pub coeffs: [f32; 5],
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeProbe {
    pub sdk_loaded: bool,
    pub api_version: Option<String>,
    pub devices: Vec<CameraDevice>,
    pub status: String,
    pub install_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraDevice {
    pub name: String,
    pub serial: String,
    pub firmware: String,
    pub usb: String,
    pub product_line: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthStats {
    pub valid_points: usize,
    pub min_m: f32,
    pub max_m: f32,
    pub mean_m: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FramePaths {
    pub rgb: Option<String>,
    pub depth: String,
    pub point_cloud: String,
    pub metadata: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameSummary {
    pub session_id: String,
    pub frame_index: u32,
    pub timestamp_ms: f64,
    pub frame_number: u64,
    pub color_preview_data_url: Option<String>,
    pub depth_preview_data_url: String,
    pub depth: DepthStats,
    pub paths: FramePaths,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStarted {
    pub session_id: String,
    pub root: String,
    pub backend: String,
    pub notice: Option<String>,
    pub config: ResolvedCaptureConfig,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStopped {
    pub session_id: String,
    pub root: String,
    pub backend: String,
    pub frames_written: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureEvent {
    kind: String,
    summary: Option<FrameSummary>,
    message: Option<String>,
}

#[tauri::command]
pub fn probe_runtime() -> RuntimeProbe {
    realsense::probe_runtime()
}

#[tauri::command]
pub fn list_devices() -> Result<Vec<CameraDevice>, String> {
    realsense::list_devices()
}

#[tauri::command]
pub fn start_recording(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    config: CaptureConfig,
) -> Result<SessionStarted, String> {
    let config = config.resolve();
    let mut guard = state
        .recording
        .lock()
        .map_err(|_| "recording state is locked".to_string())?;

    if guard.is_some() {
        return Err("recording is already active".to_string());
    }

    let (mut backend, backend_name, notice) = create_backend(&config)?;
    let session = storage::create_session(&config, &backend_name)?;
    let stop = Arc::new(AtomicBool::new(false));
    let frames_written = Arc::new(AtomicU32::new(0));

    let thread_stop = Arc::clone(&stop);
    let thread_frames_written = Arc::clone(&frames_written);
    let thread_session = session.clone();
    let thread_config = config.clone();
    let backend_for_thread = backend_name.clone();

    let handle = thread::spawn(move || {
        let interval = Duration::from_secs_f64(1.0 / thread_config.fps as f64);
        let mut frame_index = 0u32;
        let mut consecutive_errors = 0u32;

        while !thread_stop.load(Ordering::SeqCst) {
            if thread_config
                .max_frames
                .is_some_and(|max_frames| frame_index >= max_frames)
            {
                break;
            }

            let loop_started = Instant::now();
            match backend.capture_frame() {
                Ok(frame) => {
                    consecutive_errors = 0;
                    frame_index += 1;
                    match storage::write_frame(&thread_session, &thread_config, frame_index, &frame) {
                        Ok(summary) => {
                            thread_frames_written.store(frame_index, Ordering::SeqCst);
                            let _ = app.emit(
                                "capture-progress",
                                CaptureEvent {
                                    kind: "frame".to_string(),
                                    summary: Some(summary),
                                    message: None,
                                },
                            );
                        }
                        Err(error) => {
                            emit_error(&app, format!("failed to save frame {frame_index}: {error}"));
                        }
                    }
                }
                Err(error) => {
                    consecutive_errors += 1;
                    emit_error(&app, error);
                    if consecutive_errors >= 8 {
                        break;
                    }
                    thread::sleep(Duration::from_millis(250));
                }
            }

            let elapsed = loop_started.elapsed();
            if elapsed < interval {
                thread::sleep(interval - elapsed);
            }
        }

        let status = if thread_stop.load(Ordering::SeqCst) {
            "stopped"
        } else {
            "finished"
        };
        let count = thread_frames_written.load(Ordering::SeqCst);
        let _ = storage::finish_session(&thread_session, &thread_config, &backend_for_thread, status, count);
        let _ = app.emit(
            "capture-progress",
            CaptureEvent {
                kind: "finished".to_string(),
                summary: None,
                message: Some(format!("{status}: {count} frames")),
            },
        );
    });

    *guard = Some(RecordingHandle {
        session_id: session.session_id.clone(),
        root: session.root.clone(),
        backend: backend_name.clone(),
        stop,
        frames_written,
        handle: Some(handle),
    });

    Ok(SessionStarted {
        session_id: session.session_id,
        root: session.root.to_string_lossy().to_string(),
        backend: backend_name,
        notice,
        config,
    })
}

#[tauri::command]
pub fn stop_recording(state: State<'_, AppState>) -> Result<SessionStopped, String> {
    let mut recording = state
        .recording
        .lock()
        .map_err(|_| "recording state is locked".to_string())?
        .take()
        .ok_or_else(|| "recording is not active".to_string())?;

    recording.stop.store(true, Ordering::SeqCst);
    if let Some(handle) = recording.handle.take() {
        handle
            .join()
            .map_err(|_| "recording thread did not finish cleanly".to_string())?;
    }

    Ok(SessionStopped {
        session_id: recording.session_id,
        root: recording.root.to_string_lossy().to_string(),
        backend: recording.backend,
        frames_written: recording.frames_written.load(Ordering::SeqCst),
    })
}

#[tauri::command]
pub fn reveal_path(path: String) -> Result<(), String> {
    let target = PathBuf::from(path);
    if !target.exists() {
        return Err("path does not exist".to_string());
    }

    let mut command = if cfg!(target_os = "macos") {
        let mut command = Command::new("open");
        command.arg(&target);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = Command::new("explorer");
        command.arg(&target);
        command
    } else {
        let mut command = Command::new("xdg-open");
        command.arg(&target);
        command
    };

    command
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("failed to reveal path: {error}"))
}

fn create_backend(
    config: &ResolvedCaptureConfig,
) -> Result<(Box<dyn CameraBackend>, String, Option<String>), String> {
    match config.backend.as_str() {
        "synthetic" => Ok((
            Box::new(SyntheticBackend::new(config)),
            "synthetic".to_string(),
            Some("Demo depth generator is active; RealSense is not being used.".to_string()),
        )),
        "realsense" => {
            let backend = realsense::RealSenseCamera::open(config)?;
            Ok((Box::new(backend), "realsense".to_string(), None))
        }
        _ => match realsense::RealSenseCamera::open(config) {
            Ok(backend) => Ok((Box::new(backend), "realsense".to_string(), None)),
            Err(error) => Ok((
                Box::new(SyntheticBackend::new(config)),
                "synthetic".to_string(),
                Some(format!("RealSense unavailable, using demo generator: {error}")),
            )),
        },
    }
}

fn emit_error(app: &tauri::AppHandle, message: String) {
    let _ = app.emit(
        "capture-progress",
        CaptureEvent {
            kind: "error".to_string(),
            summary: None,
            message: Some(message),
        },
    );
}

struct SyntheticBackend {
    config: ResolvedCaptureConfig,
    frame_number: u64,
    started_at: Instant,
}

impl SyntheticBackend {
    fn new(config: &ResolvedCaptureConfig) -> Self {
        Self {
            config: config.clone(),
            frame_number: 0,
            started_at: Instant::now(),
        }
    }
}

impl CameraBackend for SyntheticBackend {
    fn capture_frame(&mut self) -> Result<SensorFrame, String> {
        self.frame_number += 1;

        let width = self.config.width;
        let height = self.config.height;
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        let mut z16 = vec![0u16; (width * height) as usize];
        let phase = self.frame_number as f32 * 0.08;
        let units_m = 0.001;

        let tomatoes = [
            (0.40f32, 0.45f32, 0.15f32, 0.12f32, 0.34f32),
            (0.55f32, 0.38f32, 0.12f32, 0.10f32, 0.38f32),
            (0.58f32, 0.58f32, 0.14f32, 0.11f32, 0.36f32),
            (0.46f32, 0.61f32, 0.10f32, 0.08f32, 0.40f32),
        ];

        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let idx = (y * width + x) as usize;
                let rgb_idx = idx * 3;

                let leaf_wave = ((nx * 15.0 + phase).sin() * (ny * 11.0).cos()).abs();
                let mut z = 0.74 + 0.03 * (ny * 8.0 + phase).sin();
                let mut color = [
                    (33.0 + leaf_wave * 22.0) as u8,
                    (58.0 + leaf_wave * 56.0) as u8,
                    (47.0 + leaf_wave * 18.0) as u8,
                ];

                for (cx, cy, rx, ry, tomato_depth) in tomatoes {
                    let wobble_x = cx + 0.018 * (phase + cy * 6.0).sin();
                    let wobble_y = cy + 0.010 * (phase + cx * 8.0).cos();
                    let dx = (nx - wobble_x) / rx;
                    let dy = (ny - wobble_y) / ry;
                    let d = dx * dx + dy * dy;
                    if d <= 1.0 {
                        let rim = (1.0 - d).sqrt();
                        z = tomato_depth + 0.055 * d;
                        let shine = (1.0 - ((dx + 0.35).powi(2) + (dy + 0.45).powi(2)) * 5.0).max(0.0);
                        color = [
                            (165.0 + 70.0 * rim + 35.0 * shine).clamp(0.0, 255.0) as u8,
                            (38.0 + 34.0 * rim + 38.0 * shine).clamp(0.0, 255.0) as u8,
                            (31.0 + 16.0 * rim + 22.0 * shine).clamp(0.0, 255.0) as u8,
                        ];
                    }
                }

                rgb[rgb_idx] = color[0];
                rgb[rgb_idx + 1] = color[1];
                rgb[rgb_idx + 2] = color[2];
                z16[idx] = (z / units_m).round().clamp(0.0, u16::MAX as f32) as u16;
            }
        }

        Ok(SensorFrame {
            color: Some(ColorFrame { width, height, rgb }),
            depth: DepthFrame {
                width,
                height,
                z16,
                units_m,
            },
            intrinsics: Intrinsics {
                width,
                height,
                ppx: width as f32 * 0.5,
                ppy: height as f32 * 0.5,
                fx: width as f32 * 0.92,
                fy: height as f32 * 1.02,
                coeffs: [0.0; 5],
            },
            timestamp_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
            frame_number: self.frame_number,
        })
    }
}

pub fn default_output_root() -> Result<PathBuf, String> {
    let root = dirs::document_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("SmartAgricultureScans");
    fs::create_dir_all(&root).map_err(|error| format!("failed to create output root: {error}"))?;
    Ok(root)
}
