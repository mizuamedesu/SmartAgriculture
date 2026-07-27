use std::{
    fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};

use crate::{realsense, storage};

pub(crate) const REALSENSE_HELPER_PROTOCOL: &str = "agriscan-realsense-helper-mcap-v3-aligned-rgbd";

#[derive(Default)]
pub struct AppState {
    recording: Mutex<Option<RecordingHandle>>,
    preview: Mutex<Option<PreviewHandle>>,
}

struct RecordingHandle {
    session_id: String,
    root: PathBuf,
    final_recording_path: PathBuf,
    backend: String,
    stop: Arc<AtomicBool>,
    frames_written: Arc<AtomicU32>,
    handle: Option<JoinHandle<()>>,
    helper: Option<PrivilegedRecordingControl>,
}

struct PrivilegedRecordingControl {
    config_path: PathBuf,
    progress_path: PathBuf,
    stop_path: PathBuf,
    pid_path: PathBuf,
    launch_mode: String,
}

impl Drop for PrivilegedRecordingControl {
    fn drop(&mut self) {
        for path in [
            &self.config_path,
            &self.progress_path,
            &self.stop_path,
            &self.pid_path,
        ] {
            let _ = fs::remove_file(path);
        }
    }
}

struct PreviewHandle {
    session_id: String,
    backend: String,
    stop: Arc<AtomicBool>,
    frames_sent: Arc<AtomicU32>,
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
    pub output_root: Option<String>,
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
    pub output_root: Option<String>,
    pub cultivar: String,
    pub notes: String,
    pub max_frames: Option<u32>,
    pub point_stride: u32,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
}

impl CaptureConfig {
    fn resolve(self) -> ResolvedCaptureConfig {
        let width = self.width.unwrap_or(1280).clamp(320, 1280);
        let height = self.height.unwrap_or(720).clamp(240, 720);
        let fps = self.fps.unwrap_or(30).clamp(1, 30);
        let backend = self.backend.unwrap_or_else(|| "auto".to_string());
        let point_stride = self.point_stride.unwrap_or(4).clamp(1, 12);
        let min_depth_m = self.min_depth_m.unwrap_or(0.12).clamp(0.02, 4.0);
        let max_depth_m = self
            .max_depth_m
            .unwrap_or(1.4)
            .clamp(min_depth_m + 0.01, 8.0);
        let max_frames = self
            .max_frames
            .and_then(|value| (value > 0).then_some(value));

        ResolvedCaptureConfig {
            width,
            height,
            fps,
            backend,
            target_label: non_empty(self.target_label, "scan"),
            output_root: self
                .output_root
                .map(|path| path.trim().to_string())
                .filter(|path| !path.is_empty()),
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
    pub usb_devices: Vec<UsbRealSenseDevice>,
    pub status: String,
    pub install_hint: Option<String>,
    pub action_required: Option<String>,
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
pub struct UsbRealSenseDevice {
    pub product_name: String,
    pub link_speed_mbps: Option<u32>,
    pub usb_type: Option<String>,
    pub id_product: Option<String>,
    pub location_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthStats {
    pub valid_points: usize,
    pub min_m: f32,
    pub max_m: f32,
    pub mean_m: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FramePaths {
    pub rgb: Option<String>,
    pub depth: String,
    pub point_cloud: String,
    pub metadata: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub progress_path: Option<String>,
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
pub struct PrivilegedPreviewStarted {
    pub session_id: String,
    pub frame_path: String,
    pub pid_path: String,
    pub log_path: String,
    pub launch_mode: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledHelper {
    pub path: String,
    pub status: String,
    pub ready: bool,
    pub current: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CaptureEvent {
    pub kind: String,
    pub summary: Option<FrameSummary>,
    pub message: Option<String>,
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
    stop_preview_if_running(&state)?;
    let mut guard = state
        .recording
        .lock()
        .map_err(|_| "recording state is locked".to_string())?;

    if guard.is_some() {
        return Err("recording is already active".to_string());
    }

    if cfg!(target_os = "macos") && config.backend != "synthetic" {
        let started = start_privileged_recording(app, &config)?;
        let progress_path = started
            .helper
            .as_ref()
            .map(|helper| helper.progress_path.to_string_lossy().to_string());
        let response = SessionStarted {
            session_id: started.session_id.clone(),
            root: started.root.to_string_lossy().to_string(),
            backend: started.backend.clone(),
            notice: Some(format!(
                "RealSense recording uses the privileged helper ({})",
                started
                    .helper
                    .as_ref()
                    .map(|helper| helper.launch_mode.as_str())
                    .unwrap_or("unknown")
            )),
            progress_path,
            config,
        };
        *guard = Some(started);
        return Ok(response);
    }

    let (mut backend, backend_name, notice) = create_backend(&config)?;
    let session = storage::create_session(&config, &backend_name)?;
    let final_recording_path = storage::session_recording_path(&session);
    let stop = Arc::new(AtomicBool::new(false));
    let frames_written = Arc::new(AtomicU32::new(0));

    let thread_stop = Arc::clone(&stop);
    let thread_frames_written = Arc::clone(&frames_written);
    let thread_session = session.clone();
    let thread_config = config.clone();
    let backend_for_thread = backend_name.clone();
    let session_id_for_cleanup = session.session_id.clone();

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
                    match storage::write_frame(&thread_session, &thread_config, frame_index, &frame)
                    {
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
                            emit_error(
                                &app,
                                format!("failed to save frame {frame_index}: {error}"),
                            );
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
        let _ = storage::finish_session(
            &thread_session,
            &thread_config,
            &backend_for_thread,
            status,
            count,
        );
        let _ = app.emit(
            "capture-progress",
            CaptureEvent {
                kind: "finished".to_string(),
                summary: None,
                message: Some(format!("{status}: {count} frames")),
            },
        );
        cleanup_finished_recording(&app, &session_id_for_cleanup);
    });

    *guard = Some(RecordingHandle {
        session_id: session.session_id.clone(),
        root: session.root.clone(),
        final_recording_path,
        backend: backend_name.clone(),
        stop,
        frames_written,
        handle: Some(handle),
        helper: None,
    });

    Ok(SessionStarted {
        session_id: session.session_id,
        root: session.root.to_string_lossy().to_string(),
        backend: backend_name,
        notice,
        progress_path: None,
        config,
    })
}

fn start_privileged_recording(
    app: tauri::AppHandle,
    config: &ResolvedCaptureConfig,
) -> Result<RecordingHandle, String> {
    let session = storage::create_session(config, "realsense")?;
    let final_recording_path = storage::session_recording_path(&session);
    let helper_id = format!(
        "realsense_recording_{}_{}",
        chrono::Local::now().format("%H%M%S"),
        std::process::id()
    );
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join(format!("{helper_id}_config.json"));
    let progress_path = temp_dir.join(format!("{helper_id}_progress.json"));
    let stop_path = temp_dir.join(format!("{helper_id}.stop"));
    let pid_path = temp_dir.join(format!("{helper_id}.pid"));
    let log_path = temp_dir.join(format!("{helper_id}.log"));
    for stale in [&progress_path, &stop_path, &pid_path] {
        let _ = fs::remove_file(stale);
    }
    fs::write(
        &config_path,
        serde_json::to_vec(config)
            .map_err(|error| format!("failed to encode helper recording config: {error}"))?,
    )
    .map_err(|error| format!("failed to write helper recording config: {error}"))?;

    let helper_args = [
        session.root.to_string_lossy().to_string(),
        session.session_id.clone(),
        config_path.to_string_lossy().to_string(),
        progress_path.to_string_lossy().to_string(),
        stop_path.to_string_lossy().to_string(),
        log_path.to_string_lossy().to_string(),
    ];
    let launch_mode = if let Some(helper) = installed_helper_if_current() {
        let stderr = fs::File::create(&log_path)
            .map_err(|error| format!("failed to create recording helper log: {error}"))?;
        let child = Command::new(&helper)
            .arg("--realsense-helper")
            .arg("record")
            .args(&helper_args)
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| format!("failed to start installed recording helper: {error}"))?;
        fs::write(&pid_path, child.id().to_string())
            .map_err(|error| format!("failed to write recording helper pid: {error}"))?;
        "installed-helper".to_string()
    } else {
        let exe = std::env::current_exe()
            .map_err(|error| format!("failed to locate app executable: {error}"))?;
        let kill_daemons = "killall -9 UVCAssistant VDCAssistant cameracaptured appleh16camerad AppleCameraAssistant com.apple.cmio.registerassistantservice 2>/dev/null || true";
        let kill_helpers = "pkill -9 -f '(smart-agriculture-tomato-twin|realsense-helper) --realsense-helper (live|record)' 2>/dev/null || true";
        let (user_uid, user_gid) = invoking_user_ids();
        let shell = format!(
            "{kill_helpers}; {kill_daemons}; TOMATO_TWIN_UID={user_uid} TOMATO_TWIN_GID={user_gid} {} --realsense-helper record {} {} {} {} {} {} >>{} 2>&1 & echo $! > {}",
            shell_quote(&exe.to_string_lossy()),
            shell_quote(&helper_args[0]),
            shell_quote(&helper_args[1]),
            shell_quote(&helper_args[2]),
            shell_quote(&helper_args[3]),
            shell_quote(&helper_args[4]),
            shell_quote(&helper_args[5]),
            shell_quote(&log_path.to_string_lossy()),
            shell_quote(&pid_path.to_string_lossy()),
        );
        let script = format!(
            "do shell script {} with administrator privileges",
            apple_script_string(&shell)
        );
        run_osascript_with_timeout(
            PathBuf::from("/usr/bin/osascript"),
            script,
            Duration::from_secs(30),
        )
        .map_err(|error| format!("administrator recording helper failed: {error}"))?;
        "administrator-osascript".to_string()
    };

    let ready_started = Instant::now();
    loop {
        if let Ok(bytes) = fs::read(&progress_path)
            && let Ok(event) = serde_json::from_slice::<CaptureEvent>(&bytes)
        {
            match event.kind.as_str() {
                "ready" | "frame" => break,
                "error" => {
                    let _ = stop_helper_process(&pid_path, &launch_mode);
                    let _ = storage::finish_session(&session, config, "realsense", "failed", 0);
                    return Err(event
                        .message
                        .unwrap_or_else(|| "recording helper failed to start".to_string()));
                }
                _ => {}
            }
        }
        if !helper_pid_exists(&pid_path) {
            let details = fs::read_to_string(&log_path).unwrap_or_default();
            return Err(format!(
                "recording helper exited before the stream opened: {}",
                details.lines().last().unwrap_or("no helper log")
            ));
        }
        if ready_started.elapsed() > Duration::from_secs(25) {
            let _ = stop_helper_process(&pid_path, &launch_mode);
            return Err(format!(
                "timed out waiting for RealSense recording stream; helper log: {}",
                log_path.to_string_lossy()
            ));
        }
        thread::sleep(Duration::from_millis(50));
    }

    let stop = Arc::new(AtomicBool::new(false));
    let frames_written = Arc::new(AtomicU32::new(0));
    let bridge = spawn_privileged_recording_event_bridge(
        app,
        session.session_id.clone(),
        session.root.clone(),
        final_recording_path.clone(),
        progress_path.clone(),
        pid_path.clone(),
        Arc::clone(&stop),
        Arc::clone(&frames_written),
    );

    Ok(RecordingHandle {
        session_id: session.session_id,
        root: session.root,
        final_recording_path,
        backend: "realsense".to_string(),
        stop,
        frames_written,
        handle: Some(bridge),
        helper: Some(PrivilegedRecordingControl {
            config_path,
            progress_path,
            stop_path,
            pid_path,
            launch_mode,
        }),
    })
}

fn spawn_privileged_recording_event_bridge(
    app: tauri::AppHandle,
    session_id: String,
    session_root: PathBuf,
    final_recording_path: PathBuf,
    progress_path: PathBuf,
    pid_path: PathBuf,
    stop: Arc<AtomicBool>,
    frames_written: Arc<AtomicU32>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut last_frame_index = 0u32;
        let mut last_message = String::new();
        let mut helper_exit_misses = 0u32;
        loop {
            if let Ok(bytes) = fs::read(&progress_path)
                && let Ok(event) = serde_json::from_slice::<CaptureEvent>(&bytes)
            {
                if let Some(summary) = &event.summary {
                    if summary.frame_index != last_frame_index {
                        last_frame_index = summary.frame_index;
                        frames_written.store(summary.frame_index, Ordering::SeqCst);
                        let _ = app.emit("capture-progress", event.clone());
                    }
                } else if event.kind == "finished" {
                    let _ = app.emit("capture-progress", event);
                    break;
                } else if event.kind == "error" {
                    let message = event.message.clone().unwrap_or_default();
                    if message != last_message {
                        last_message = message;
                        let _ = app.emit("capture-progress", event);
                    }
                }
            }

            if helper_pid_exists(&pid_path) {
                helper_exit_misses = 0;
            } else {
                helper_exit_misses += 1;
                if helper_exit_misses > 10 {
                    let _ = app.emit(
                        "capture-progress",
                        CaptureEvent {
                            kind: "finished".to_string(),
                            summary: None,
                            message: Some(format!(
                                "recording helper exited: {} frames",
                                frames_written.load(Ordering::SeqCst)
                            )),
                        },
                    );
                    break;
                }
            }
            if stop.load(Ordering::SeqCst) && helper_exit_misses > 2 {
                break;
            }
            thread::sleep(Duration::from_millis(33));
        }
        let _ = storage::finalize_recording_file(&session_root, &final_recording_path);
        cleanup_finished_recording(&app, &session_id);
    })
}

fn cleanup_finished_recording(app: &tauri::AppHandle, session_id: &str) {
    let state = app.state::<AppState>();
    if let Ok(mut guard) = state.recording.try_lock()
        && guard
            .as_ref()
            .is_some_and(|recording| recording.session_id == session_id)
    {
        guard.take();
    }
}

#[tauri::command]
pub fn start_preview(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    config: CaptureConfig,
) -> Result<SessionStarted, String> {
    let config = config.resolve();
    if state
        .recording
        .lock()
        .map_err(|_| "recording state is locked".to_string())?
        .is_some()
    {
        return Err("stop recording before starting preview".to_string());
    }

    let mut guard = state
        .preview
        .lock()
        .map_err(|_| "preview state is locked".to_string())?;

    if guard.is_some() {
        return Err("preview is already active".to_string());
    }

    let (mut backend, backend_name, notice) = create_backend(&config)?;
    let session_id = format!("live_preview_{}", chrono::Local::now().format("%H%M%S"));
    let stop = Arc::new(AtomicBool::new(false));
    let frames_sent = Arc::new(AtomicU32::new(0));

    let thread_stop = Arc::clone(&stop);
    let thread_frames_sent = Arc::clone(&frames_sent);
    let thread_config = config.clone();
    let thread_session_id = session_id.clone();

    let handle = thread::spawn(move || {
        let interval = Duration::from_secs_f64(1.0 / thread_config.fps as f64);
        let mut frame_index = 0u32;
        let mut consecutive_errors = 0u32;

        while !thread_stop.load(Ordering::SeqCst) {
            let loop_started = Instant::now();
            match backend.capture_frame() {
                Ok(frame) => {
                    consecutive_errors = 0;
                    frame_index += 1;
                    match storage::preview_frame_summary(
                        &thread_session_id,
                        &thread_config,
                        frame_index,
                        &frame,
                    ) {
                        Ok(summary) => {
                            thread_frames_sent.store(frame_index, Ordering::SeqCst);
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
                            emit_error(&app, format!("failed to render preview: {error}"))
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

        let count = thread_frames_sent.load(Ordering::SeqCst);
        let _ = app.emit(
            "capture-progress",
            CaptureEvent {
                kind: "finished".to_string(),
                summary: None,
                message: Some(format!("preview stopped: {count} frames")),
            },
        );
    });

    *guard = Some(PreviewHandle {
        session_id: session_id.clone(),
        backend: backend_name.clone(),
        stop,
        frames_sent,
        handle: Some(handle),
    });

    Ok(SessionStarted {
        session_id,
        root: String::new(),
        backend: backend_name,
        notice,
        progress_path: None,
        config,
    })
}

#[tauri::command]
pub fn start_privileged_preview(
    app: tauri::AppHandle,
    config: CaptureConfig,
) -> Result<PrivilegedPreviewStarted, String> {
    let config = config.resolve();
    let session_id = format!(
        "realsense_preview_{}",
        chrono::Local::now().format("%H%M%S")
    );
    let frame_path = std::env::temp_dir().join(format!("{session_id}.json"));
    let pid_path = std::env::temp_dir().join(format!("{session_id}.pid"));
    let log_path = std::env::temp_dir().join(format!("{session_id}.log"));

    if let Some(helper) = installed_helper_if_current() {
        let launch_log = std::env::temp_dir().join(format!("{session_id}_launch.log"));
        let stderr = fs::File::create(&launch_log)
            .map_err(|error| format!("failed to create helper launch log: {error}"))?;
        let child = Command::new(&helper)
            .arg("--realsense-helper")
            .arg("live")
            .arg(&frame_path)
            .arg(config.width.to_string())
            .arg(config.height.to_string())
            .arg(config.fps.to_string())
            .arg(&session_id)
            .arg(&log_path)
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| format!("failed to start installed RealSense helper: {error}"))?;
        fs::write(&pid_path, child.id().to_string())
            .map_err(|error| format!("failed to write helper pid: {error}"))?;
        spawn_privileged_preview_event_bridge(app, frame_path.clone(), pid_path.clone());

        return Ok(PrivilegedPreviewStarted {
            session_id,
            frame_path: frame_path.to_string_lossy().to_string(),
            pid_path: pid_path.to_string_lossy().to_string(),
            log_path: log_path.to_string_lossy().to_string(),
            launch_mode: "installed-helper".to_string(),
        });
    }

    let exe = std::env::current_exe()
        .map_err(|error| format!("failed to locate app executable: {error}"))?;
    let osascript = PathBuf::from("/usr/bin/osascript");

    let kill_daemons = "killall -9 UVCAssistant VDCAssistant cameracaptured appleh16camerad AppleCameraAssistant com.apple.cmio.registerassistantservice 2>/dev/null || true";
    let kill_helpers = "pkill -9 -f '(smart-agriculture-tomato-twin|realsense-helper) --realsense-helper (live|record)' 2>/dev/null || true";
    let (user_uid, user_gid) = invoking_user_ids();
    let shell = format!(
        "{kill_helpers}; {kill_daemons}; (i=0; while [ $i -lt 80 ]; do {kill_daemons}; i=$((i+1)); sleep 0.05; done) >/dev/null 2>&1 & TOMATO_TWIN_UID={user_uid} TOMATO_TWIN_GID={user_gid} {} --realsense-helper live {} {} {} {} {} {} >/tmp/tomato-twin-helper-launch.log 2>&1 & echo $! > {}",
        shell_quote(&exe.to_string_lossy()),
        shell_quote(&frame_path.to_string_lossy()),
        config.width,
        config.height,
        config.fps,
        shell_quote(&session_id),
        shell_quote(&log_path.to_string_lossy()),
        shell_quote(&pid_path.to_string_lossy()),
    );
    let script = format!(
        "do shell script {} with administrator privileges",
        apple_script_string(&shell)
    );

    run_osascript_with_timeout(osascript, script, Duration::from_secs(20))
        .map_err(|error| format!("administrator preview helper failed: {error}"))?;
    spawn_privileged_preview_event_bridge(app, frame_path.clone(), pid_path.clone());

    Ok(PrivilegedPreviewStarted {
        session_id,
        frame_path: frame_path.to_string_lossy().to_string(),
        pid_path: pid_path.to_string_lossy().to_string(),
        log_path: log_path.to_string_lossy().to_string(),
        launch_mode: "administrator-osascript".to_string(),
    })
}

#[tauri::command]
pub fn install_privileged_helper() -> Result<InstalledHelper, String> {
    let source = helper_source_path()?;
    let destination = installed_helper_path();
    let destination_dir = destination
        .parent()
        .ok_or_else(|| "installed helper destination has no parent".to_string())?;

    let shell = format!(
        "mkdir -p {}; cp -f {} {}; chown root:wheel {}; chmod 4755 {}",
        shell_quote(&destination_dir.to_string_lossy()),
        shell_quote(&source.to_string_lossy()),
        shell_quote(&destination.to_string_lossy()),
        shell_quote(&destination.to_string_lossy()),
        shell_quote(&destination.to_string_lossy()),
    );
    let script = format!(
        "do shell script {} with administrator privileges",
        apple_script_string(&shell)
    );
    run_osascript_with_timeout(
        PathBuf::from("/usr/bin/osascript"),
        script,
        Duration::from_secs(60),
    )
    .map_err(|error| format!("helper install failed: {error}"))?;

    let status = privileged_helper_status()?;
    if !status.ready || !status.current {
        return Err(
            "helper was installed, but root ownership, setuid permission, or executable integrity is not ready"
                .to_string(),
        );
    }

    Ok(status)
}

#[tauri::command]
pub fn privileged_helper_status() -> Result<InstalledHelper, String> {
    let destination = installed_helper_path();
    let ready = installed_helper_if_ready().is_some();
    let current = ready && helper_protocol_is_compatible(&destination);
    let status = if current {
        "RealSense helper ready; preview and recording require no additional password".to_string()
    } else if ready {
        "RealSense helper update required once at startup".to_string()
    } else {
        "RealSense helper installation required once at startup".to_string()
    };
    Ok(InstalledHelper {
        path: destination.to_string_lossy().to_string(),
        status,
        ready,
        current,
    })
}

#[tauri::command]
pub fn ensure_privileged_helper() -> Result<InstalledHelper, String> {
    let status = privileged_helper_status()?;
    if status.ready && status.current {
        Ok(status)
    } else {
        install_privileged_helper()
    }
}

#[tauri::command]
pub fn read_privileged_preview_frame(frame_path: String) -> Result<FrameSummary, String> {
    let bytes = fs::read(&frame_path)
        .map_err(|error| format!("waiting for RealSense preview frame: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid preview frame JSON: {error}"))
}

#[tauri::command]
pub fn read_privileged_recording_frame(progress_path: String) -> Result<FrameSummary, String> {
    let bytes = fs::read(&progress_path)
        .map_err(|error| format!("waiting for RGB-D recording frame: {error}"))?;
    let event = serde_json::from_slice::<CaptureEvent>(&bytes)
        .map_err(|error| format!("invalid RGB-D recording progress JSON: {error}"))?;
    event
        .summary
        .filter(|_| event.kind == "frame")
        .ok_or_else(|| {
            event
                .message
                .unwrap_or_else(|| "waiting for RGB-D recording frame".to_string())
        })
}

#[tauri::command]
pub fn read_latest_privileged_preview_frame() -> Result<FrameSummary, String> {
    let temp_dir = std::env::temp_dir();
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = fs::read_dir(&temp_dir)
        .map_err(|error| format!("failed to scan preview temp dir: {error}"))?;

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("realsense_preview_")
            || !name.ends_with(".json")
            || name.ends_with(".tmp")
        {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if newest
            .as_ref()
            .is_none_or(|(current_modified, _)| modified > *current_modified)
        {
            newest = Some((modified, path));
        }
    }

    let (_, path) = newest.ok_or_else(|| "no RealSense preview frame JSON found".to_string())?;
    let metadata = fs::metadata(&path)
        .map_err(|error| format!("failed to stat latest preview frame: {error}"))?;
    let modified = metadata
        .modified()
        .map_err(|error| format!("failed to read latest preview timestamp: {error}"))?;
    if modified
        .elapsed()
        .map(|elapsed| elapsed > Duration::from_secs(5))
        .unwrap_or(false)
    {
        return Err(format!(
            "latest preview frame is stale: {}",
            path.to_string_lossy()
        ));
    }
    let bytes =
        fs::read(&path).map_err(|error| format!("failed to read latest preview frame: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "invalid latest preview frame JSON at {}: {error}",
            path.to_string_lossy()
        )
    })
}

#[tauri::command]
pub fn stop_privileged_preview(
    pid_path: String,
    launch_mode: Option<String>,
) -> Result<(), String> {
    stop_helper_process(
        &PathBuf::from(pid_path),
        launch_mode.as_deref().unwrap_or("administrator-osascript"),
    )
}

fn stop_helper_process(pid_path: &PathBuf, launch_mode: &str) -> Result<(), String> {
    let pid = fs::read_to_string(pid_path)
        .map_err(|error| format!("failed to read privileged helper pid: {error}"))?;
    let pid = pid.trim();
    if pid.is_empty() {
        return Ok(());
    }

    if launch_mode == "installed-helper" {
        let status = Command::new("kill")
            .arg(pid)
            .status()
            .map_err(|error| format!("failed to stop installed helper: {error}"))?;
        if status.success() {
            return Ok(());
        }
    }

    let shell = format!("kill {} 2>/dev/null || true", shell_quote(pid));
    let script = format!(
        "do shell script {} with administrator privileges",
        apple_script_string(&shell)
    );
    let _ = Command::new("/usr/bin/osascript")
        .arg("-e")
        .arg(script)
        .output();
    Ok(())
}

#[tauri::command]
pub fn stop_preview(state: State<'_, AppState>) -> Result<SessionStopped, String> {
    let mut preview = state
        .preview
        .lock()
        .map_err(|_| "preview state is locked".to_string())?
        .take()
        .ok_or_else(|| "preview is not active".to_string())?;

    preview.stop.store(true, Ordering::SeqCst);
    if let Some(handle) = preview.handle.take() {
        handle
            .join()
            .map_err(|_| "preview thread did not finish cleanly".to_string())?;
    }

    Ok(SessionStopped {
        session_id: preview.session_id,
        root: String::new(),
        backend: preview.backend,
        frames_written: preview.frames_sent.load(Ordering::SeqCst),
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
    if let Some(helper) = &recording.helper {
        fs::write(&helper.stop_path, b"stop")
            .map_err(|error| format!("failed to signal recording helper: {error}"))?;
        let started = Instant::now();
        while helper_pid_exists(&helper.pid_path) && started.elapsed() < Duration::from_secs(4) {
            thread::sleep(Duration::from_millis(50));
        }
        if helper_pid_exists(&helper.pid_path) {
            stop_helper_process(&helper.pid_path, &helper.launch_mode)?;
        }
    }
    if let Some(handle) = recording.handle.take() {
        handle
            .join()
            .map_err(|_| "recording thread did not finish cleanly".to_string())?;
    }
    storage::finalize_recording_file(&recording.root, &recording.final_recording_path)?;
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
        if target.is_dir() {
            command.arg("-a").arg("Finder").arg(&target);
        } else {
            command.arg("-R").arg(&target);
        }
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

#[allow(clippy::type_complexity)]
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
        _ => {
            let backend = realsense::RealSenseCamera::open(config)?;
            Ok((Box::new(backend), "realsense".to_string(), None))
        }
    }
}

fn stop_preview_if_running(state: &State<'_, AppState>) -> Result<(), String> {
    let maybe_preview = state
        .preview
        .lock()
        .map_err(|_| "preview state is locked".to_string())?
        .take();

    if let Some(mut preview) = maybe_preview {
        preview.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = preview.handle.take() {
            handle
                .join()
                .map_err(|_| "preview thread did not finish cleanly".to_string())?;
        }
    }

    Ok(())
}

fn spawn_privileged_preview_event_bridge(
    app: tauri::AppHandle,
    frame_path: PathBuf,
    pid_path: PathBuf,
) {
    thread::spawn(move || {
        let started = Instant::now();
        let mut last_frame_index = 0u32;
        let mut misses = 0u32;

        loop {
            if started.elapsed() > Duration::from_secs(60 * 60) {
                break;
            }

            match fs::read(&frame_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<FrameSummary>(&bytes).ok())
            {
                Some(summary) if summary.frame_index != last_frame_index => {
                    last_frame_index = summary.frame_index;
                    misses = 0;
                    let _ = app.emit(
                        "capture-progress",
                        CaptureEvent {
                            kind: "frame".to_string(),
                            summary: Some(summary),
                            message: None,
                        },
                    );
                }
                Some(_) => {
                    misses = misses.saturating_add(1);
                }
                None => {
                    misses = misses.saturating_add(1);
                }
            }

            if misses > 300 && !helper_pid_exists(&pid_path) {
                break;
            }
            thread::sleep(Duration::from_millis(33));
        }
    });
}

fn helper_pid_exists(pid_path: &PathBuf) -> bool {
    let Ok(pid_text) = fs::read_to_string(pid_path) else {
        return true;
    };
    let Ok(pid) = pid_text.trim().parse::<i32>() else {
        return false;
    };
    #[cfg(unix)]
    unsafe {
        if libc::kill(pid as libc::pid_t, 0) == 0 {
            return true;
        }
        let errno = std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or_default();
        errno == libc::EPERM
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

fn installed_helper_path() -> PathBuf {
    PathBuf::from("/usr/local/libexec/tomato-twin/realsense-helper")
}

fn helper_source_path() -> Result<PathBuf, String> {
    std::env::current_exe().map_err(|error| format!("failed to locate app executable: {error}"))
}

fn installed_helper_if_ready() -> Option<PathBuf> {
    let path = installed_helper_path();
    let metadata = fs::metadata(&path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let mode = metadata.permissions().mode();
        if !metadata.is_file()
            || metadata.uid() != 0
            || metadata.gid() != 0
            || mode & 0o4000 == 0
            || mode & 0o022 != 0
        {
            return None;
        }
    }
    Some(path)
}

fn installed_helper_if_current() -> Option<PathBuf> {
    let path = installed_helper_if_ready()?;
    helper_protocol_is_compatible(&path).then_some(path)
}

fn helper_protocol_is_compatible(path: &PathBuf) -> bool {
    let Ok(output) = Command::new(path)
        .args(["--realsense-helper", "protocol"])
        .output()
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout).trim() == REALSENSE_HELPER_PROTOCOL
}

fn run_osascript_with_timeout(
    osascript: PathBuf,
    script: String,
    timeout: Duration,
) -> Result<(), String> {
    let mut child = Command::new(osascript)
        .arg("-e")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to run osascript: {error}"))?;
    let started = Instant::now();

    loop {
        match child
            .try_wait()
            .map_err(|error| format!("failed to wait for osascript: {error}"))?
        {
            Some(status) => {
                let mut stdout = String::new();
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stdout.take() {
                    let _ = pipe.read_to_string(&mut stdout);
                }
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                if status.success() {
                    return Ok(());
                }
                return Err(format!("{stderr}{stdout}"));
            }
            None if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("administrator prompt timed out".to_string());
            }
            None => thread::sleep(Duration::from_millis(100)),
        }
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn apple_script_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(unix)]
fn invoking_user_ids() -> (u32, u32) {
    unsafe { (libc::getuid(), libc::getgid()) }
}

#[cfg(not(unix))]
fn invoking_user_ids() -> (u32, u32) {
    (0, 0)
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
                        let shine =
                            (1.0 - ((dx + 0.35).powi(2) + (dy + 0.45).powi(2)) * 5.0).max(0.0);
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
    let root = dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("AgriScan Studio")
        .join("Scans");
    fs::create_dir_all(&root).map_err(|error| format!("failed to create output root: {error}"))?;
    Ok(root)
}

#[tauri::command]
pub fn default_save_location() -> Result<String, String> {
    default_output_root().map(|path| path.to_string_lossy().to_string())
}

pub fn legacy_output_root() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Tomato Twin Capture")
        .join("Scans")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{
        AssetBuildOptions, detect_asset_tools, generate_scan_assets, load_scan_data,
    };
    use crate::mcap_io;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn synthetic_recording_completes_the_native_asset_pipeline() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let output_root = std::env::temp_dir().join(format!(
            "tomato-twin-recording-e2e-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&output_root).expect("create synthetic output root");
        let config = ResolvedCaptureConfig {
            width: 96,
            height: 72,
            fps: 30,
            backend: "synthetic".to_string(),
            target_label: "e2e_tomato".to_string(),
            output_root: Some(output_root.to_string_lossy().to_string()),
            cultivar: "test".to_string(),
            notes: "native pipeline test".to_string(),
            max_frames: Some(4),
            point_stride: 2,
            min_depth_m: 0.12,
            max_depth_m: 1.4,
        };
        let session =
            storage::create_session_at(&output_root, &config, "synthetic").expect("create session");
        let mut camera = SyntheticBackend::new(&config);
        for frame_index in 1..=4 {
            let frame = camera.capture_frame().expect("capture synthetic RGB-D");
            let summary = storage::write_frame(&session, &config, frame_index, &frame)
                .expect("persist synthetic RGB-D frame");
            assert!(summary.paths.rgb.is_some());
            assert!(summary.paths.depth.contains("e2e_tomato.mcap#"));
            assert!(summary.paths.point_cloud.contains("e2e_tomato.mcap#"));
            assert!(summary.paths.metadata.contains("e2e_tomato.mcap#"));
        }
        storage::finish_session(&session, &config, "synthetic", "finished", 4)
            .expect("finish synthetic session");
        let recording = mcap_io::find_recording_path(&session.root).expect("find MCAP recording");
        assert!(recording.is_file());
        let topics = mcap_io::validate_recording(&recording).expect("validate MCAP topics");
        assert!(topics.iter().any(|topic| topic == mcap_io::TOPIC_COLOR));
        assert_eq!(mcap_io::frame_count(&recording).expect("count frames"), 4);

        let result = generate_scan_assets(AssetBuildOptions {
            session_root: session.root.to_string_lossy().to_string(),
            max_points: Some(30_000),
            frame_stride: Some(1),
            depth_decimation: Some(2),
            gaussian_radius_m: Some(0.006),
            turntable_degrees: Some(0.0),
            export_fbx: Some(true),
            use_mlx: Some(false),
            mlx_iterations: Some(0),
            mlx_voxel_size_m: Some(0.003),
            mlx_train_size: Some(64),
            mlx_max_train_views: Some(4),
            collider_max_faces: Some(5_000),
        })
        .expect("generate native assets from recorded frames");

        assert!(PathBuf::from(&result.gaussian_ply).is_file());
        assert!(PathBuf::from(&result.splat).is_file());
        assert!(PathBuf::from(result.mesh_fbx.expect("native FBX path")).is_file());
        assert!(PathBuf::from(&result.preview_json).is_file());
        assert_eq!(
            result.tools.fbx_exporter,
            "Built-in native FBX 7.4 exporter"
        );
        assert!(result.preview.points.len() > 1_000);
        let loaded_ply =
            load_scan_data(result.gaussian_ply.clone()).expect("load existing 3DGS PLY");
        assert_eq!(loaded_ply.point_count, result.point_count);
        assert!(!loaded_ply.preview.points.is_empty());
        let loaded_splat = load_scan_data(result.splat.clone()).expect("load existing .splat");
        assert_eq!(loaded_splat.point_count, result.point_count);
        assert!(!loaded_splat.preview.points.is_empty());

        if detect_asset_tools().mlx_available {
            let refined = generate_scan_assets(AssetBuildOptions {
                session_root: session.root.to_string_lossy().to_string(),
                max_points: Some(10_000),
                frame_stride: Some(1),
                depth_decimation: Some(4),
                gaussian_radius_m: Some(0.006),
                turntable_degrees: Some(0.0),
                export_fbx: Some(true),
                use_mlx: Some(true),
                mlx_iterations: Some(2),
                mlx_voxel_size_m: Some(0.004),
                mlx_train_size: Some(64),
                mlx_max_train_views: Some(1),
                collider_max_faces: Some(5_000),
            })
            .expect("run pretrained SHARP MLX inference on recorded frames");
            assert!(refined.gaussian_ply.ends_with("scan_gaussians_mlx.ply"));
            assert!(refined.mlx_status.contains("SHARP MLX"));
            assert!(PathBuf::from(refined.splat).is_file());
        }
        let _ = fs::remove_dir_all(&output_root);
    }
}
