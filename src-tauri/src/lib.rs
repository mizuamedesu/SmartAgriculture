mod assets;
mod capture;
mod realsense;
mod storage;

use std::{
    fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use assets::{detect_asset_tools, ensure_mlx_3dgs, generate_scan_assets};
use capture::{CameraBackend, ResolvedCaptureConfig};
use capture::{
    AppState, install_privileged_helper, list_devices, probe_runtime,
    read_latest_privileged_preview_frame, read_privileged_preview_frame, reveal_path,
    start_preview, start_privileged_preview, start_recording, stop_preview,
    stop_privileged_preview, stop_recording,
};
use realsense::ensure_realsense_sdk;

pub fn run_realsense_helper(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) != Some("live") {
        return Err("expected helper mode: live".to_string());
    }
    let frame_path = PathBuf::from(args.get(1).ok_or_else(|| "missing frame path".to_string())?);
    let width = args
        .get(2)
        .ok_or_else(|| "missing width".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("invalid width: {error}"))?;
    let height = args
        .get(3)
        .ok_or_else(|| "missing height".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("invalid height: {error}"))?;
    let fps = args
        .get(4)
        .ok_or_else(|| "missing fps".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("invalid fps: {error}"))?;
    let session_id = args.get(5).ok_or_else(|| "missing session id".to_string())?.clone();
    let log_path = PathBuf::from(args.get(6).ok_or_else(|| "missing log path".to_string())?);

    let config = ResolvedCaptureConfig {
        width,
        height,
        fps,
        backend: "realsense".to_string(),
        target_label: "mini_tomato".to_string(),
        cultivar: "unknown".to_string(),
        notes: String::new(),
        max_frames: None,
        point_stride: 4,
        min_depth_m: 0.12,
        max_depth_m: 1.4,
    };

    clear_stale_realsense_helpers();
    clear_camera_daemon_owners();
    let mut camera = realsense::RealSenseCamera::open(&config)?;
    drop_privileges_after_camera_open()?;
    let _ = helper_log(&log_path, "starting privileged RealSense helper");
    let _ = helper_log(&log_path, "RealSense stream opened");

    let interval = Duration::from_secs_f64(1.0 / fps.max(1) as f64);
    let mut frame_index = 0u32;
    loop {
        let loop_started = Instant::now();
        match camera.capture_frame() {
            Ok(frame) => {
                frame_index += 1;
                let summary = storage::preview_frame_summary(&session_id, &config, frame_index, &frame)?;
                let json = serde_json::to_vec(&summary)
                    .map_err(|error| format!("failed to encode preview JSON: {error}"))?;
                let tmp = frame_path.with_extension("json.tmp");
                fs::write(&tmp, json)
                    .map_err(|error| format!("failed to write preview frame: {error}"))?;
                fs::rename(&tmp, &frame_path)
                    .map_err(|error| format!("failed to publish preview frame: {error}"))?;
            }
            Err(error) => {
                let _ = helper_log(&log_path, &format!("capture failed: {error}"));
                thread::sleep(Duration::from_millis(250));
            }
        }
        let elapsed = loop_started.elapsed();
        if elapsed < interval {
            thread::sleep(interval - elapsed);
        }
    }
}

fn clear_stale_realsense_helpers() {
    let own_pid = std::process::id();
    let output = std::process::Command::new("pgrep")
        .args([
            "-f",
            "smart-agriculture-tomato-twin --realsense-helper live|realsense-helper live",
        ])
        .output();

    let Ok(output) = output else {
        return;
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let Ok(pid) = line.trim().parse::<u32>() else {
            continue;
        };
        if pid == own_pid {
            continue;
        }
        #[cfg(unix)]
        unsafe {
            let _ = libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
    }
}

fn clear_camera_daemon_owners() {
    for _ in 0..80 {
        let _ = std::process::Command::new("killall")
            .args([
                "-9",
                "UVCAssistant",
                "VDCAssistant",
                "cameracaptured",
                "appleh16camerad",
                "AppleCameraAssistant",
                "com.apple.cmio.registerassistantservice",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        thread::sleep(Duration::from_millis(50));
    }
}

fn drop_privileges_after_camera_open() -> Result<(), String> {
    #[cfg(unix)]
    unsafe {
        let real_uid = libc::getuid();
        let effective_uid = libc::geteuid();
        let real_gid = libc::getgid();
        if effective_uid == 0 && real_uid != 0 {
            if libc::setgid(real_gid) != 0 {
                return Err("failed to drop helper group privileges".to_string());
            }
            if libc::setuid(real_uid) != 0 {
                return Err("failed to drop helper user privileges".to_string());
            }
        }
    }
    Ok(())
}

fn helper_log(path: &PathBuf, message: &str) -> Result<(), String> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("failed to open helper log: {error}"))?;
    writeln!(file, "{} {}", chrono::Local::now().to_rfc3339(), message)
        .map_err(|error| format!("failed to write helper log: {error}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            probe_runtime,
            list_devices,
            install_privileged_helper,
            start_preview,
            start_privileged_preview,
            read_privileged_preview_frame,
            read_latest_privileged_preview_frame,
            stop_privileged_preview,
            stop_preview,
            start_recording,
            stop_recording,
            reveal_path,
            ensure_realsense_sdk,
            detect_asset_tools,
            ensure_mlx_3dgs,
            generate_scan_assets
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Tomato Twin Capture");
}
