mod assets;
mod capture;
mod fbx;
mod mcap_io;
mod realsense;
mod storage;

use std::{
    fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use assets::{
    detect_asset_tools, ensure_mlx_3dgs, generate_scan_assets, load_latest_scan_assets,
    load_scan_data,
};
use capture::{
    AppState, default_save_location, ensure_privileged_helper, install_privileged_helper,
    list_devices, privileged_helper_status, probe_runtime, read_latest_privileged_preview_frame,
    read_privileged_preview_frame, read_privileged_recording_frame, reveal_path, start_preview,
    start_privileged_preview, start_recording, stop_preview, stop_privileged_preview,
    stop_recording,
};
use capture::{CameraBackend, CaptureEvent, ResolvedCaptureConfig};
use realsense::ensure_realsense_sdk;

pub fn rebuild_scan_assets_cli(session_root: &str) -> Result<String, String> {
    let result = generate_scan_assets(assets::AssetBuildOptions {
        session_root: session_root.to_string(),
        max_points: Some(1_500_000),
        frame_stride: Some(1),
        depth_decimation: Some(2),
        gaussian_radius_m: Some(0.0035),
        turntable_degrees: Some(0.0),
        export_fbx: Some(true),
        use_mlx: Some(true),
        mlx_iterations: Some(0),
        mlx_voxel_size_m: Some(0.0025),
        mlx_train_size: Some(1536),
        mlx_max_train_views: Some(100),
        collider_max_faces: Some(35_000),
    })?;
    serde_json::to_string_pretty(&serde_json::json!({
        "root": result.root,
        "pointCount": result.point_count,
        "faceCount": result.face_count,
        "gaussianPly": result.gaussian_ply,
        "meshFbx": result.mesh_fbx,
        "bounds": result.preview.bounds,
        "mlxStatus": result.mlx_status,
        "fbxStatus": result.fbx_status
    }))
    .map_err(|error| format!("failed to encode asset result: {error}"))
}

pub fn export_mcap_samples_cli(recording_path: &str, output_root: &str) -> Result<String, String> {
    let outputs = assets::export_mcap_sample_frames(
        std::path::Path::new(recording_path),
        std::path::Path::new(output_root),
    )?;
    serde_json::to_string_pretty(&outputs)
        .map_err(|error| format!("failed to encode sample frame paths: {error}"))
}

pub fn run_realsense_helper(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("protocol") => {
            println!("{}", capture::REALSENSE_HELPER_PROTOCOL);
            Ok(())
        }
        Some("live") => run_live_realsense_helper(&args[1..]),
        Some("record") => run_record_realsense_helper(&args[1..]),
        Some(mode) => Err(format!("unknown helper mode: {mode}")),
        None => Err("missing helper mode".to_string()),
    }
}

fn run_live_realsense_helper(args: &[String]) -> Result<(), String> {
    let frame_path = PathBuf::from(
        args.first()
            .ok_or_else(|| "missing frame path".to_string())?,
    );
    let width = args
        .get(1)
        .ok_or_else(|| "missing width".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("invalid width: {error}"))?;
    let height = args
        .get(2)
        .ok_or_else(|| "missing height".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("invalid height: {error}"))?;
    let fps = args
        .get(3)
        .ok_or_else(|| "missing fps".to_string())?
        .parse::<u32>()
        .map_err(|error| format!("invalid fps: {error}"))?;
    let session_id = args
        .get(4)
        .ok_or_else(|| "missing session id".to_string())?
        .clone();
    let log_path = PathBuf::from(args.get(5).ok_or_else(|| "missing log path".to_string())?);

    let config = ResolvedCaptureConfig {
        width,
        height,
        fps,
        backend: "realsense".to_string(),
        target_label: "scan_target".to_string(),
        output_root: None,
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
                let summary =
                    storage::preview_frame_summary(&session_id, &config, frame_index, &frame)?;
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

fn run_record_realsense_helper(args: &[String]) -> Result<(), String> {
    let session_root = PathBuf::from(
        args.first()
            .ok_or_else(|| "missing recording session root".to_string())?,
    );
    let session_id = args
        .get(1)
        .ok_or_else(|| "missing recording session id".to_string())?
        .clone();
    let config_path = PathBuf::from(
        args.get(2)
            .ok_or_else(|| "missing recording config path".to_string())?,
    );
    let progress_path = PathBuf::from(
        args.get(3)
            .ok_or_else(|| "missing recording progress path".to_string())?,
    );
    let stop_path = PathBuf::from(
        args.get(4)
            .ok_or_else(|| "missing recording stop path".to_string())?,
    );
    let log_path = PathBuf::from(
        args.get(5)
            .ok_or_else(|| "missing recording log path".to_string())?,
    );
    let config_data = fs::read(&config_path)
        .map_err(|error| format!("failed to read recording config: {error}"))?;
    let config: ResolvedCaptureConfig = serde_json::from_slice(&config_data)
        .map_err(|error| format!("invalid recording config: {error}"))?;
    let session = storage::open_session(session_id.clone(), session_root)?;

    clear_stale_realsense_helpers();
    clear_camera_daemon_owners();
    let mut camera = match realsense::RealSenseCamera::open(&config) {
        Ok(camera) => camera,
        Err(error) => {
            let message = format!("failed to open RealSense for recording: {error}");
            let _ = publish_capture_event(
                &progress_path,
                &CaptureEvent {
                    kind: "error".to_string(),
                    summary: None,
                    message: Some(message.clone()),
                },
            );
            let _ = storage::finish_session(&session, &config, "realsense", "failed", 0);
            return Err(message);
        }
    };
    drop_privileges_after_camera_open()?;
    let _ = helper_log(&log_path, "privileged RealSense recording stream opened");
    publish_capture_event(
        &progress_path,
        &CaptureEvent {
            kind: "ready".to_string(),
            summary: None,
            message: Some("RealSense recording stream opened".to_string()),
        },
    )?;

    let interval = Duration::from_secs_f64(1.0 / config.fps.max(1) as f64);
    let mut frame_index = 0u32;
    let mut consecutive_errors = 0u32;
    let mut final_status = "stopped";
    while !stop_path.exists() {
        if config
            .max_frames
            .is_some_and(|max_frames| frame_index >= max_frames)
        {
            final_status = "finished";
            break;
        }
        let loop_started = Instant::now();
        match camera.capture_frame() {
            Ok(frame) => {
                consecutive_errors = 0;
                frame_index += 1;
                match storage::write_frame(&session, &config, frame_index, &frame) {
                    Ok(summary) => publish_capture_event(
                        &progress_path,
                        &CaptureEvent {
                            kind: "frame".to_string(),
                            summary: Some(summary),
                            message: None,
                        },
                    )?,
                    Err(error) => {
                        let message = format!("failed to save frame {frame_index}: {error}");
                        let _ = helper_log(&log_path, &message);
                        publish_capture_event(
                            &progress_path,
                            &CaptureEvent {
                                kind: "error".to_string(),
                                summary: None,
                                message: Some(message),
                            },
                        )?;
                        consecutive_errors += 1;
                    }
                }
            }
            Err(error) => {
                consecutive_errors += 1;
                let message = format!("RealSense recording capture failed: {error}");
                let _ = helper_log(&log_path, &message);
                publish_capture_event(
                    &progress_path,
                    &CaptureEvent {
                        kind: "error".to_string(),
                        summary: None,
                        message: Some(message),
                    },
                )?;
                thread::sleep(Duration::from_millis(250));
            }
        }
        if consecutive_errors >= 8 {
            final_status = "failed";
            break;
        }
        let elapsed = loop_started.elapsed();
        if elapsed < interval {
            thread::sleep(interval - elapsed);
        }
    }

    storage::finish_session(&session, &config, "realsense", final_status, frame_index)?;
    publish_capture_event(
        &progress_path,
        &CaptureEvent {
            kind: "finished".to_string(),
            summary: None,
            message: Some(format!("{final_status}: {frame_index} frames")),
        },
    )
}

fn publish_capture_event(path: &PathBuf, event: &CaptureEvent) -> Result<(), String> {
    let json = serde_json::to_vec(event)
        .map_err(|error| format!("failed to encode recording progress: {error}"))?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json)
        .map_err(|error| format!("failed to write recording progress: {error}"))?;
    fs::rename(&tmp, path).map_err(|error| format!("failed to publish recording progress: {error}"))
}

fn clear_stale_realsense_helpers() {
    let own_pid = std::process::id();
    let output = std::process::Command::new("pgrep")
        .args([
            "-f",
            "(smart-agriculture-tomato-twin|realsense-helper) --realsense-helper (live|record)",
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
        let target_uid = std::env::var("TOMATO_TWIN_UID")
            .ok()
            .and_then(|value| value.parse::<libc::uid_t>().ok())
            .unwrap_or(real_uid);
        let target_gid = std::env::var("TOMATO_TWIN_GID")
            .ok()
            .and_then(|value| value.parse::<libc::gid_t>().ok())
            .unwrap_or(real_gid);
        if effective_uid == 0 && target_uid != 0 {
            if libc::setgid(target_gid) != 0 {
                return Err("failed to drop helper group privileges".to_string());
            }
            if libc::setuid(target_uid) != 0 {
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
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            probe_runtime,
            list_devices,
            privileged_helper_status,
            ensure_privileged_helper,
            install_privileged_helper,
            start_preview,
            start_privileged_preview,
            read_privileged_preview_frame,
            read_latest_privileged_preview_frame,
            read_privileged_recording_frame,
            stop_privileged_preview,
            stop_preview,
            start_recording,
            stop_recording,
            reveal_path,
            default_save_location,
            ensure_realsense_sdk,
            detect_asset_tools,
            ensure_mlx_3dgs,
            generate_scan_assets,
            load_latest_scan_assets,
            load_scan_data
        ])
        .run(tauri::generate_context!())
        .expect("failed to run AgriScan Studio");
}
