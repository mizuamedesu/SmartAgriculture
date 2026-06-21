mod assets;
mod capture;
mod realsense;
mod storage;

use assets::{detect_asset_tools, generate_scan_assets};
use capture::{AppState, list_devices, probe_runtime, reveal_path, start_recording, stop_recording};
use realsense::ensure_realsense_sdk;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            probe_runtime,
            list_devices,
            start_recording,
            stop_recording,
            reveal_path,
            ensure_realsense_sdk,
            detect_asset_tools,
            generate_scan_assets
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Tomato Twin Capture");
}
