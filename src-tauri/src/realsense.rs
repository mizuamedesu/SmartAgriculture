use std::{
    ffi::CStr,
    os::raw::{c_char, c_int, c_uint, c_void},
    path::PathBuf,
    ptr,
    process::Command,
    slice,
    sync::Arc,
};

use libloading::Library;

use crate::capture::{
    CameraBackend, CameraDevice, ColorFrame, DepthFrame, Intrinsics, ResolvedCaptureConfig,
    RuntimeProbe, SensorFrame, UsbRealSenseDevice,
};

const RS2_STREAM_DEPTH: c_int = 1;
const RS2_STREAM_COLOR: c_int = 2;
const RS2_FORMAT_Z16: c_int = 1;
const RS2_FORMAT_RGB8: c_int = 5;
const RS2_FORMAT_BGR8: c_int = 6;

const RS2_CAMERA_INFO_NAME: c_int = 0;
const RS2_CAMERA_INFO_SERIAL_NUMBER: c_int = 1;
const RS2_CAMERA_INFO_FIRMWARE_VERSION: c_int = 2;
const RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR: c_int = 9;
const RS2_CAMERA_INFO_PRODUCT_LINE: c_int = 10;

type Rs2Context = c_void;
type Rs2DeviceList = c_void;
type Rs2Device = c_void;
type Rs2Error = c_void;
type Rs2Pipeline = c_void;
type Rs2Config = c_void;
type Rs2PipelineProfile = c_void;
type Rs2Frame = c_void;
type Rs2StreamProfile = c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Rs2Intrinsics {
    width: c_int,
    height: c_int,
    ppx: f32,
    ppy: f32,
    fx: f32,
    fy: f32,
    model: c_int,
    coeffs: [f32; 5],
}

impl Default for Rs2Intrinsics {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            ppx: 0.0,
            ppy: 0.0,
            fx: 1.0,
            fy: 1.0,
            model: 0,
            coeffs: [0.0; 5],
        }
    }
}

type Rs2GetApiVersion = unsafe extern "C" fn(*mut *mut Rs2Error) -> c_int;
type Rs2CreateContext = unsafe extern "C" fn(c_int, *mut *mut Rs2Error) -> *mut Rs2Context;
type Rs2DeleteContext = unsafe extern "C" fn(*mut Rs2Context);
type Rs2QueryDevices =
    unsafe extern "C" fn(*const Rs2Context, *mut *mut Rs2Error) -> *mut Rs2DeviceList;
type Rs2GetDeviceCount =
    unsafe extern "C" fn(*const Rs2DeviceList, *mut *mut Rs2Error) -> c_int;
type Rs2DeleteDeviceList = unsafe extern "C" fn(*mut Rs2DeviceList);
type Rs2CreateDevice =
    unsafe extern "C" fn(*const Rs2DeviceList, c_int, *mut *mut Rs2Error) -> *mut Rs2Device;
type Rs2DeleteDevice = unsafe extern "C" fn(*mut Rs2Device);
type Rs2SupportsDeviceInfo =
    unsafe extern "C" fn(*const Rs2Device, c_int, *mut *mut Rs2Error) -> c_int;
type Rs2GetDeviceInfo =
    unsafe extern "C" fn(*const Rs2Device, c_int, *mut *mut Rs2Error) -> *const c_char;
type Rs2CreatePipeline =
    unsafe extern "C" fn(*mut Rs2Context, *mut *mut Rs2Error) -> *mut Rs2Pipeline;
type Rs2DeletePipeline = unsafe extern "C" fn(*mut Rs2Pipeline);
type Rs2PipelineStop = unsafe extern "C" fn(*mut Rs2Pipeline, *mut *mut Rs2Error);
type Rs2CreateConfig = unsafe extern "C" fn(*mut *mut Rs2Error) -> *mut Rs2Config;
type Rs2DeleteConfig = unsafe extern "C" fn(*mut Rs2Config);
type Rs2ConfigEnableStream = unsafe extern "C" fn(
    *mut Rs2Config,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut *mut Rs2Error,
);
type Rs2PipelineStartWithConfig = unsafe extern "C" fn(
    *mut Rs2Pipeline,
    *mut Rs2Config,
    *mut *mut Rs2Error,
) -> *mut Rs2PipelineProfile;
type Rs2DeletePipelineProfile = unsafe extern "C" fn(*mut Rs2PipelineProfile);
type Rs2PipelineWaitForFrames =
    unsafe extern "C" fn(*mut Rs2Pipeline, c_uint, *mut *mut Rs2Error) -> *mut Rs2Frame;
type Rs2ReleaseFrame = unsafe extern "C" fn(*mut Rs2Frame);
type Rs2EmbeddedFramesCount =
    unsafe extern "C" fn(*mut Rs2Frame, *mut *mut Rs2Error) -> c_int;
type Rs2ExtractFrame =
    unsafe extern "C" fn(*mut Rs2Frame, c_int, *mut *mut Rs2Error) -> *mut Rs2Frame;
type Rs2GetFrameStreamProfile =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> *const Rs2StreamProfile;
type Rs2GetStreamProfileData = unsafe extern "C" fn(
    *const Rs2StreamProfile,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut *mut Rs2Error,
);
type Rs2GetVideoStreamIntrinsics =
    unsafe extern "C" fn(*const Rs2StreamProfile, *mut Rs2Intrinsics, *mut *mut Rs2Error);
type Rs2GetFrameData =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> *const c_void;
type Rs2GetFrameDataSize =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> c_int;
type Rs2GetFrameWidth = unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> c_int;
type Rs2GetFrameHeight = unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> c_int;
type Rs2GetFrameStride =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> c_int;
type Rs2DepthFrameGetUnits =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> f32;
type Rs2GetFrameNumber =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> u64;
type Rs2GetFrameTimestamp =
    unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> f64;
type Rs2GetErrorMessage = unsafe extern "C" fn(*const Rs2Error) -> *const c_char;
type Rs2GetFailedFunction = unsafe extern "C" fn(*const Rs2Error) -> *const c_char;
type Rs2GetFailedArgs = unsafe extern "C" fn(*const Rs2Error) -> *const c_char;
type Rs2FreeError = unsafe extern "C" fn(*mut Rs2Error);

struct Rs2Api {
    _library: Library,
    rs2_get_api_version: Rs2GetApiVersion,
    rs2_create_context: Rs2CreateContext,
    rs2_delete_context: Rs2DeleteContext,
    rs2_query_devices: Rs2QueryDevices,
    rs2_get_device_count: Rs2GetDeviceCount,
    rs2_delete_device_list: Rs2DeleteDeviceList,
    rs2_create_device: Rs2CreateDevice,
    rs2_delete_device: Rs2DeleteDevice,
    rs2_supports_device_info: Rs2SupportsDeviceInfo,
    rs2_get_device_info: Rs2GetDeviceInfo,
    rs2_create_pipeline: Rs2CreatePipeline,
    rs2_delete_pipeline: Rs2DeletePipeline,
    rs2_pipeline_stop: Rs2PipelineStop,
    rs2_create_config: Rs2CreateConfig,
    rs2_delete_config: Rs2DeleteConfig,
    rs2_config_enable_stream: Rs2ConfigEnableStream,
    rs2_pipeline_start_with_config: Rs2PipelineStartWithConfig,
    rs2_delete_pipeline_profile: Rs2DeletePipelineProfile,
    rs2_pipeline_wait_for_frames: Rs2PipelineWaitForFrames,
    rs2_release_frame: Rs2ReleaseFrame,
    rs2_embedded_frames_count: Rs2EmbeddedFramesCount,
    rs2_extract_frame: Rs2ExtractFrame,
    rs2_get_frame_stream_profile: Rs2GetFrameStreamProfile,
    rs2_get_stream_profile_data: Rs2GetStreamProfileData,
    rs2_get_video_stream_intrinsics: Rs2GetVideoStreamIntrinsics,
    rs2_get_frame_data: Rs2GetFrameData,
    rs2_get_frame_data_size: Rs2GetFrameDataSize,
    rs2_get_frame_width: Rs2GetFrameWidth,
    rs2_get_frame_height: Rs2GetFrameHeight,
    rs2_get_frame_stride_in_bytes: Rs2GetFrameStride,
    rs2_depth_frame_get_units: Rs2DepthFrameGetUnits,
    rs2_get_frame_number: Rs2GetFrameNumber,
    rs2_get_frame_timestamp: Rs2GetFrameTimestamp,
    rs2_get_error_message: Rs2GetErrorMessage,
    rs2_get_failed_function: Rs2GetFailedFunction,
    rs2_get_failed_args: Rs2GetFailedArgs,
    rs2_free_error: Rs2FreeError,
}

unsafe impl Send for Rs2Api {}
unsafe impl Sync for Rs2Api {}

impl Rs2Api {
    fn load() -> Result<Self, String> {
        let candidates = library_candidates();
        let mut errors = Vec::new();

        for candidate in candidates {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => match unsafe { Self::from_library(library) } {
                    Ok(api) => return Ok(api),
                    Err(error) => errors.push(format!("{candidate}: {error}")),
                },
                Err(error) => errors.push(format!("{candidate}: {error}")),
            }
        }

        Err(format!(
            "librealsense2 was not found. Tried: {}",
            errors.join(" | ")
        ))
    }

    unsafe fn from_library(library: Library) -> Result<Self, String> {
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let symbol = unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|error| format!("missing symbol {}: {error}", $name))?;
                *symbol
            }};
        }

        Ok(Self {
            rs2_get_api_version: symbol!("rs2_get_api_version", Rs2GetApiVersion),
            rs2_create_context: symbol!("rs2_create_context", Rs2CreateContext),
            rs2_delete_context: symbol!("rs2_delete_context", Rs2DeleteContext),
            rs2_query_devices: symbol!("rs2_query_devices", Rs2QueryDevices),
            rs2_get_device_count: symbol!("rs2_get_device_count", Rs2GetDeviceCount),
            rs2_delete_device_list: symbol!("rs2_delete_device_list", Rs2DeleteDeviceList),
            rs2_create_device: symbol!("rs2_create_device", Rs2CreateDevice),
            rs2_delete_device: symbol!("rs2_delete_device", Rs2DeleteDevice),
            rs2_supports_device_info: symbol!(
                "rs2_supports_device_info",
                Rs2SupportsDeviceInfo
            ),
            rs2_get_device_info: symbol!("rs2_get_device_info", Rs2GetDeviceInfo),
            rs2_create_pipeline: symbol!("rs2_create_pipeline", Rs2CreatePipeline),
            rs2_delete_pipeline: symbol!("rs2_delete_pipeline", Rs2DeletePipeline),
            rs2_pipeline_stop: symbol!("rs2_pipeline_stop", Rs2PipelineStop),
            rs2_create_config: symbol!("rs2_create_config", Rs2CreateConfig),
            rs2_delete_config: symbol!("rs2_delete_config", Rs2DeleteConfig),
            rs2_config_enable_stream: symbol!("rs2_config_enable_stream", Rs2ConfigEnableStream),
            rs2_pipeline_start_with_config: symbol!(
                "rs2_pipeline_start_with_config",
                Rs2PipelineStartWithConfig
            ),
            rs2_delete_pipeline_profile: symbol!(
                "rs2_delete_pipeline_profile",
                Rs2DeletePipelineProfile
            ),
            rs2_pipeline_wait_for_frames: symbol!(
                "rs2_pipeline_wait_for_frames",
                Rs2PipelineWaitForFrames
            ),
            rs2_release_frame: symbol!("rs2_release_frame", Rs2ReleaseFrame),
            rs2_embedded_frames_count: symbol!(
                "rs2_embedded_frames_count",
                Rs2EmbeddedFramesCount
            ),
            rs2_extract_frame: symbol!("rs2_extract_frame", Rs2ExtractFrame),
            rs2_get_frame_stream_profile: symbol!(
                "rs2_get_frame_stream_profile",
                Rs2GetFrameStreamProfile
            ),
            rs2_get_stream_profile_data: symbol!(
                "rs2_get_stream_profile_data",
                Rs2GetStreamProfileData
            ),
            rs2_get_video_stream_intrinsics: symbol!(
                "rs2_get_video_stream_intrinsics",
                Rs2GetVideoStreamIntrinsics
            ),
            rs2_get_frame_data: symbol!("rs2_get_frame_data", Rs2GetFrameData),
            rs2_get_frame_data_size: symbol!("rs2_get_frame_data_size", Rs2GetFrameDataSize),
            rs2_get_frame_width: symbol!("rs2_get_frame_width", Rs2GetFrameWidth),
            rs2_get_frame_height: symbol!("rs2_get_frame_height", Rs2GetFrameHeight),
            rs2_get_frame_stride_in_bytes: symbol!(
                "rs2_get_frame_stride_in_bytes",
                Rs2GetFrameStride
            ),
            rs2_depth_frame_get_units: symbol!(
                "rs2_depth_frame_get_units",
                Rs2DepthFrameGetUnits
            ),
            rs2_get_frame_number: symbol!("rs2_get_frame_number", Rs2GetFrameNumber),
            rs2_get_frame_timestamp: symbol!("rs2_get_frame_timestamp", Rs2GetFrameTimestamp),
            rs2_get_error_message: symbol!("rs2_get_error_message", Rs2GetErrorMessage),
            rs2_get_failed_function: symbol!("rs2_get_failed_function", Rs2GetFailedFunction),
            rs2_get_failed_args: symbol!("rs2_get_failed_args", Rs2GetFailedArgs),
            rs2_free_error: symbol!("rs2_free_error", Rs2FreeError),
            _library: library,
        })
    }

    fn call<T>(&self, f: impl FnOnce(*mut *mut Rs2Error) -> T) -> Result<T, String> {
        let mut error: *mut Rs2Error = ptr::null_mut();
        let value = f(&mut error);
        self.check_error(error)?;
        Ok(value)
    }

    fn check_error(&self, error: *mut Rs2Error) -> Result<(), String> {
        if error.is_null() {
            return Ok(());
        }

        let message = unsafe {
            let message = c_string((self.rs2_get_error_message)(error));
            let failed_function = c_string((self.rs2_get_failed_function)(error));
            let failed_args = c_string((self.rs2_get_failed_args)(error));
            (self.rs2_free_error)(error);

            if failed_function.is_empty() {
                message
            } else {
                format!("{message} ({failed_function} {failed_args})")
            }
        };
        Err(message)
    }

    fn api_version(&self) -> Result<c_int, String> {
        self.call(|error| unsafe { (self.rs2_get_api_version)(error) })
    }

    fn api_version_string(&self) -> Result<String, String> {
        let version = self.api_version()?;
        let major = version / 10_000;
        let minor = (version / 100) % 100;
        let patch = version % 100;
        Ok(format!("{major}.{minor}.{patch}"))
    }
}

pub fn probe_runtime() -> RuntimeProbe {
    let usb_devices = detect_usb_realsense_devices();
    match Rs2Api::load() {
        Ok(api) => {
            let api_version = api.api_version_string().ok();
            match list_devices_with_api(&api) {
                Ok(devices) => {
                    let (status, action_required) = if devices.is_empty() && !usb_devices.is_empty() {
                        usb_diagnostic_status(&usb_devices)
                    } else if devices.is_empty() {
                        (
                            "librealsense2 loaded; no RealSense device detected".to_string(),
                            None,
                        )
                    } else {
                        (
                            format!(
                                "librealsense2 loaded; {} RealSense device(s) detected",
                                devices.len()
                            ),
                            None,
                        )
                    };
                    RuntimeProbe {
                        sdk_loaded: true,
                        api_version,
                        devices,
                        usb_devices,
                        status,
                        install_hint: None,
                        action_required,
                    }
                }
                Err(error) => RuntimeProbe {
                    sdk_loaded: true,
                    api_version,
                    devices: Vec::new(),
                    usb_devices: usb_devices.clone(),
                    status: if usb_devices.is_empty() {
                        format!("librealsense2 loaded; device query failed: {error}")
                    } else {
                        format!(
                            "RealSense is visible on USB, but librealsense cannot open it: {error}"
                        )
                    },
                    install_hint: Some("Check USB3 cable/port, then unplug and reconnect the camera.".to_string()),
                    action_required: usb_diagnostic_status(&usb_devices).1,
                },
            }
        }
        Err(error) => RuntimeProbe {
            sdk_loaded: false,
            api_version: None,
            devices: Vec::new(),
            usb_devices: usb_devices.clone(),
            status: if usb_devices.is_empty() {
                "librealsense2 is not installed or cannot be loaded".to_string()
            } else {
                "RealSense is visible on USB, but librealsense is not loadable".to_string()
            },
            install_hint: Some(format!(
                "{error}. On macOS, install the Intel RealSense SDK with `brew install librealsense`, then confirm `rs-enumerate-devices` can see the camera."
            )),
            action_required: usb_diagnostic_status(&usb_devices).1,
        },
    }
}

pub fn list_devices() -> Result<Vec<CameraDevice>, String> {
    let api = Rs2Api::load()?;
    list_devices_with_api(&api)
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SdkSetupResult {
    pub brew_path: Option<String>,
    pub sdk_loaded_before: bool,
    pub sdk_loaded_after: bool,
    pub install_ran: bool,
    pub enumerate_ran: bool,
    pub device_check_ok: bool,
    pub devices: Vec<CameraDevice>,
    pub status: String,
    pub log: Vec<String>,
}

#[tauri::command]
pub fn ensure_realsense_sdk() -> Result<SdkSetupResult, String> {
    let mut log = Vec::new();
    let sdk_loaded_before = Rs2Api::load().is_ok();
    log.push(if sdk_loaded_before {
        "librealsense2 is already loadable".to_string()
    } else {
        "librealsense2 is not loadable yet".to_string()
    });

    let brew_path = find_executable("brew");
    let mut install_ran = false;

    if !sdk_loaded_before {
        let brew = brew_path
            .as_ref()
            .ok_or_else(|| "Homebrew was not found. Install Homebrew once, then rerun SDK setup.".to_string())?;

        match run_command(brew, &["list", "--versions", "librealsense"]) {
            Ok(output) if output.status_success => {
                log.push(format!("librealsense is already installed: {}", output.summary()));
            }
            Ok(_) | Err(_) => {
                install_ran = true;
                log.push("running `brew install librealsense`".to_string());
                let output = run_command(brew, &["install", "librealsense"])?;
                log.push(output.summary());
                if !output.status_success {
                    return Ok(SdkSetupResult {
                        brew_path: brew_path_string(brew_path.as_ref()),
                        sdk_loaded_before,
                        sdk_loaded_after: false,
                        install_ran,
                        enumerate_ran: false,
                        device_check_ok: false,
                        devices: Vec::new(),
                        status: "brew install librealsense failed".to_string(),
                        log,
                    });
                }
            }
        }
    }

    let sdk_loaded_after = Rs2Api::load().is_ok();
    if sdk_loaded_after {
        log.push("librealsense2 can be loaded by the app".to_string());
    } else {
        log.push("librealsense2 is still not loadable after setup".to_string());
    }

    let usb_devices = detect_usb_realsense_devices();
    if !usb_devices.is_empty() {
        for device in &usb_devices {
            log.push(format!(
                "USB sees {} at {} Mbps{}",
                device.product_name,
                device
                    .link_speed_mbps
                    .map(|speed| speed.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                device
                    .usb_type
                    .as_ref()
                    .map(|usb_type| format!(" / USB {usb_type}"))
                    .unwrap_or_default()
            ));
        }
        if usb_devices.iter().any(|device| device.link_speed_mbps.unwrap_or(0) < 5_000) {
            log.push("RealSense is connected below USB3 speed; RGB-D streaming is unlikely to work.".to_string());
        }
    }

    if !usb_devices.is_empty() {
        if let Some(osascript) = find_executable("osascript") {
            log.push("resetting macOS camera daemons".to_string());
            let script = "do shell script \"killall VDCAssistant 2>/dev/null || true; killall AppleCameraAssistant 2>/dev/null || true; killall cameracaptured 2>/dev/null || true; killall appleh16camerad 2>/dev/null || true\" with administrator privileges";
            match run_command(&osascript, &["-e", script]) {
                Ok(output) => log.push(format!("camera daemon reset: {}", output.summary())),
                Err(error) => log.push(format!("camera daemon reset skipped: {error}")),
            }
        }
    }

    let rs_enumerate = find_executable("rs-enumerate-devices");
    let mut enumerate_ran = false;
    let mut device_check_ok = false;

    if let Some(tool) = rs_enumerate {
        enumerate_ran = true;
        let output = run_command(&tool, &[])?;
        device_check_ok = output.status_success;
        log.push(format!("rs-enumerate-devices: {}", output.summary()));
    } else {
        log.push("rs-enumerate-devices was not found in PATH or Homebrew locations".to_string());
    }

    let devices = list_devices().unwrap_or_default();
    if !devices.is_empty() {
        device_check_ok = true;
    }

    let status = if sdk_loaded_after && device_check_ok {
        format!("RealSense SDK ready; {} device(s) detected", devices.len())
    } else if sdk_loaded_after {
        "RealSense SDK ready; no camera detected yet".to_string()
    } else {
        "RealSense SDK setup did not complete".to_string()
    };

    Ok(SdkSetupResult {
        brew_path: brew_path_string(brew_path.as_ref()),
        sdk_loaded_before,
        sdk_loaded_after,
        install_ran,
        enumerate_ran,
        device_check_ok,
        devices,
        status,
        log,
    })
}

pub struct RealSenseCamera {
    api: Arc<Rs2Api>,
    context: *mut Rs2Context,
    pipeline: *mut Rs2Pipeline,
    config: *mut Rs2Config,
    profile: *mut Rs2PipelineProfile,
    started: bool,
}

unsafe impl Send for RealSenseCamera {}

impl RealSenseCamera {
    pub fn open(config: &ResolvedCaptureConfig) -> Result<Self, String> {
        Self::open_with_color(config, true).or_else(|color_error| {
            Self::open_with_color(config, false).map_err(|depth_error| {
                format!("RGB-D open failed: {color_error}; depth-only open failed: {depth_error}")
            })
        })
    }

    fn open_with_color(config: &ResolvedCaptureConfig, enable_color: bool) -> Result<Self, String> {
        let api = Arc::new(Rs2Api::load()?);
        let api_version = api.api_version()?;
        let context = api.call(|error| unsafe { (api.rs2_create_context)(api_version, error) })?;
        if context.is_null() {
            return Err("rs2_create_context returned null".to_string());
        }

        let pipeline = match api.call(|error| unsafe { (api.rs2_create_pipeline)(context, error) }) {
            Ok(pipeline) if !pipeline.is_null() => pipeline,
            Ok(_) => {
                unsafe { (api.rs2_delete_context)(context) };
                return Err("rs2_create_pipeline returned null".to_string());
            }
            Err(error) => {
                unsafe { (api.rs2_delete_context)(context) };
                return Err(error);
            }
        };

        let rs_config = match api.call(|error| unsafe { (api.rs2_create_config)(error) }) {
            Ok(rs_config) if !rs_config.is_null() => rs_config,
            Ok(_) => {
                cleanup_pipeline_context(&api, pipeline, context);
                return Err("rs2_create_config returned null".to_string());
            }
            Err(error) => {
                cleanup_pipeline_context(&api, pipeline, context);
                return Err(error);
            }
        };

        let setup = || -> Result<(), String> {
            api.call(|error| unsafe {
                (api.rs2_config_enable_stream)(
                    rs_config,
                    RS2_STREAM_DEPTH,
                    -1,
                    config.width as c_int,
                    config.height as c_int,
                    RS2_FORMAT_Z16,
                    config.fps as c_int,
                    error,
                )
            })?;
            if enable_color {
                api.call(|error| unsafe {
                    (api.rs2_config_enable_stream)(
                        rs_config,
                        RS2_STREAM_COLOR,
                        -1,
                        config.width as c_int,
                        config.height as c_int,
                        RS2_FORMAT_RGB8,
                        config.fps as c_int,
                        error,
                    )
                })?;
            }
            Ok(())
        };

        if let Err(error) = setup() {
            cleanup_config_pipeline_context(&api, rs_config, pipeline, context);
            return Err(error);
        }

        let profile = match api.call(|error| unsafe {
            (api.rs2_pipeline_start_with_config)(pipeline, rs_config, error)
        }) {
            Ok(profile) if !profile.is_null() => profile,
            Ok(_) => {
                cleanup_config_pipeline_context(&api, rs_config, pipeline, context);
                return Err("rs2_pipeline_start_with_config returned null".to_string());
            }
            Err(error) => {
                cleanup_config_pipeline_context(&api, rs_config, pipeline, context);
                return Err(error);
            }
        };

        let camera = Self {
            api,
            context,
            pipeline,
            config: rs_config,
            profile,
            started: true,
        };

        for _ in 0..4 {
            if let Ok(frameset) = camera.wait_frameset(2_000) {
                unsafe { (camera.api.rs2_release_frame)(frameset) };
            }
        }

        Ok(camera)
    }

    fn wait_frameset(&self, timeout_ms: u32) -> Result<*mut Rs2Frame, String> {
        let frameset = self.api.call(|error| unsafe {
            (self.api.rs2_pipeline_wait_for_frames)(self.pipeline, timeout_ms, error)
        })?;
        if frameset.is_null() {
            Err("rs2_pipeline_wait_for_frames returned null".to_string())
        } else {
            Ok(frameset)
        }
    }

    fn read_frameset(&self, frameset: *mut Rs2Frame) -> Result<SensorFrame, String> {
        let count = self.api.call(|error| unsafe {
            (self.api.rs2_embedded_frames_count)(frameset, error)
        })?;

        let mut depth: Option<(DepthFrame, Intrinsics, f64, u64)> = None;
        let mut color: Option<ColorFrame> = None;

        for index in 0..count {
            let frame = self
                .api
                .call(|error| unsafe { (self.api.rs2_extract_frame)(frameset, index, error) })?;
            if frame.is_null() {
                continue;
            }

            let result = self.read_stream_frame(frame);
            unsafe { (self.api.rs2_release_frame)(frame) };
            match result? {
                StreamPacket::Depth(depth_frame, intrinsics, timestamp, number) => {
                    depth = Some((depth_frame, intrinsics, timestamp, number));
                }
                StreamPacket::Color(color_frame) => {
                    color = Some(color_frame);
                }
                StreamPacket::Other => {}
            }
        }

        let (depth, intrinsics, timestamp_ms, frame_number) =
            depth.ok_or_else(|| "frameset did not include a Z16 depth frame".to_string())?;

        Ok(SensorFrame {
            color,
            depth,
            intrinsics,
            timestamp_ms,
            frame_number,
        })
    }

    fn read_stream_frame(&self, frame: *mut Rs2Frame) -> Result<StreamPacket, String> {
        let profile = self.api.call(|error| unsafe {
            (self.api.rs2_get_frame_stream_profile)(frame, error)
        })?;
        if profile.is_null() {
            return Ok(StreamPacket::Other);
        }

        let mut stream = 0;
        let mut format = 0;
        let mut index = 0;
        let mut unique_id = 0;
        let mut framerate = 0;
        self.api.call(|error| unsafe {
            (self.api.rs2_get_stream_profile_data)(
                profile,
                &mut stream,
                &mut format,
                &mut index,
                &mut unique_id,
                &mut framerate,
                error,
            )
        })?;

        if stream == RS2_STREAM_DEPTH && format == RS2_FORMAT_Z16 {
            self.read_depth_frame(frame, profile)
        } else if stream == RS2_STREAM_COLOR && (format == RS2_FORMAT_RGB8 || format == RS2_FORMAT_BGR8) {
            self.read_color_frame(frame, format)
        } else {
            Ok(StreamPacket::Other)
        }
    }

    fn read_depth_frame(
        &self,
        frame: *mut Rs2Frame,
        profile: *const Rs2StreamProfile,
    ) -> Result<StreamPacket, String> {
        let width = self.frame_int(frame, self.api.rs2_get_frame_width)? as u32;
        let height = self.frame_int(frame, self.api.rs2_get_frame_height)? as u32;
        let stride = self
            .frame_int(frame, self.api.rs2_get_frame_stride_in_bytes)?
            .max((width * 2) as c_int) as usize;
        let data_size = self.frame_int(frame, self.api.rs2_get_frame_data_size)? as usize;
        let data_ptr = self.api.call(|error| unsafe {
            (self.api.rs2_get_frame_data)(frame, error)
        })?;
        if data_ptr.is_null() {
            return Err("depth frame data pointer is null".to_string());
        }

        let bytes = unsafe { slice::from_raw_parts(data_ptr as *const u8, data_size) };
        let mut z16 = vec![0u16; (width * height) as usize];
        for y in 0..height as usize {
            let row_start = y * stride;
            for x in 0..width as usize {
                let src = row_start + x * 2;
                if src + 1 < bytes.len() {
                    z16[y * width as usize + x] = u16::from_le_bytes([bytes[src], bytes[src + 1]]);
                }
            }
        }

        let units_m = self.api.call(|error| unsafe {
            (self.api.rs2_depth_frame_get_units)(frame, error)
        })?;
        let frame_number = self.api.call(|error| unsafe {
            (self.api.rs2_get_frame_number)(frame, error)
        })?;
        let timestamp_ms = self.api.call(|error| unsafe {
            (self.api.rs2_get_frame_timestamp)(frame, error)
        })?;

        let mut raw_intrinsics = Rs2Intrinsics::default();
        self.api.call(|error| unsafe {
            (self.api.rs2_get_video_stream_intrinsics)(profile, &mut raw_intrinsics, error)
        })?;

        Ok(StreamPacket::Depth(
            DepthFrame {
                width,
                height,
                z16,
                units_m,
            },
            Intrinsics {
                width: raw_intrinsics.width as u32,
                height: raw_intrinsics.height as u32,
                ppx: raw_intrinsics.ppx,
                ppy: raw_intrinsics.ppy,
                fx: raw_intrinsics.fx,
                fy: raw_intrinsics.fy,
                coeffs: raw_intrinsics.coeffs,
            },
            timestamp_ms,
            frame_number,
        ))
    }

    fn read_color_frame(&self, frame: *mut Rs2Frame, format: c_int) -> Result<StreamPacket, String> {
        let width = self.frame_int(frame, self.api.rs2_get_frame_width)? as u32;
        let height = self.frame_int(frame, self.api.rs2_get_frame_height)? as u32;
        let stride = self
            .frame_int(frame, self.api.rs2_get_frame_stride_in_bytes)?
            .max((width * 3) as c_int) as usize;
        let data_size = self.frame_int(frame, self.api.rs2_get_frame_data_size)? as usize;
        let data_ptr = self.api.call(|error| unsafe {
            (self.api.rs2_get_frame_data)(frame, error)
        })?;
        if data_ptr.is_null() {
            return Err("color frame data pointer is null".to_string());
        }

        let bytes = unsafe { slice::from_raw_parts(data_ptr as *const u8, data_size) };
        let mut rgb = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            let row_start = y * stride;
            for x in 0..width as usize {
                let src = row_start + x * 3;
                let dst = (y * width as usize + x) * 3;
                if src + 2 < bytes.len() {
                    if format == RS2_FORMAT_BGR8 {
                        rgb[dst] = bytes[src + 2];
                        rgb[dst + 1] = bytes[src + 1];
                        rgb[dst + 2] = bytes[src];
                    } else {
                        rgb[dst] = bytes[src];
                        rgb[dst + 1] = bytes[src + 1];
                        rgb[dst + 2] = bytes[src + 2];
                    }
                }
            }
        }

        Ok(StreamPacket::Color(ColorFrame { width, height, rgb }))
    }

    fn frame_int(
        &self,
        frame: *mut Rs2Frame,
        function: unsafe extern "C" fn(*const Rs2Frame, *mut *mut Rs2Error) -> c_int,
    ) -> Result<c_int, String> {
        self.api.call(|error| unsafe { function(frame, error) })
    }
}

impl CameraBackend for RealSenseCamera {
    fn capture_frame(&mut self) -> Result<SensorFrame, String> {
        let frameset = self.wait_frameset(5_000)?;
        let result = self.read_frameset(frameset);
        unsafe { (self.api.rs2_release_frame)(frameset) };
        result
    }
}

impl Drop for RealSenseCamera {
    fn drop(&mut self) {
        unsafe {
            if self.started && !self.pipeline.is_null() {
                let mut error: *mut Rs2Error = ptr::null_mut();
                (self.api.rs2_pipeline_stop)(self.pipeline, &mut error);
                if !error.is_null() {
                    (self.api.rs2_free_error)(error);
                }
            }
            if !self.profile.is_null() {
                (self.api.rs2_delete_pipeline_profile)(self.profile);
            }
            if !self.config.is_null() {
                (self.api.rs2_delete_config)(self.config);
            }
            if !self.pipeline.is_null() {
                (self.api.rs2_delete_pipeline)(self.pipeline);
            }
            if !self.context.is_null() {
                (self.api.rs2_delete_context)(self.context);
            }
        }
    }
}

enum StreamPacket {
    Depth(DepthFrame, Intrinsics, f64, u64),
    Color(ColorFrame),
    Other,
}

fn list_devices_with_api(api: &Rs2Api) -> Result<Vec<CameraDevice>, String> {
    let api_version = api.api_version()?;
    let context = api.call(|error| unsafe { (api.rs2_create_context)(api_version, error) })?;
    if context.is_null() {
        return Err("rs2_create_context returned null".to_string());
    }

    let result = (|| -> Result<Vec<CameraDevice>, String> {
        let devices = api.call(|error| unsafe { (api.rs2_query_devices)(context, error) })?;
        if devices.is_null() {
            return Ok(Vec::new());
        }

        let count = match api.call(|error| unsafe { (api.rs2_get_device_count)(devices, error) }) {
            Ok(count) => count,
            Err(error) => {
                unsafe { (api.rs2_delete_device_list)(devices) };
                return Err(error);
            }
        };

        let mut output = Vec::new();
        for index in 0..count {
            let device = api.call(|error| unsafe { (api.rs2_create_device)(devices, index, error) })?;
            if device.is_null() {
                continue;
            }
            output.push(CameraDevice {
                name: device_info(api, device, RS2_CAMERA_INFO_NAME),
                serial: device_info(api, device, RS2_CAMERA_INFO_SERIAL_NUMBER),
                firmware: device_info(api, device, RS2_CAMERA_INFO_FIRMWARE_VERSION),
                usb: device_info(api, device, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR),
                product_line: device_info(api, device, RS2_CAMERA_INFO_PRODUCT_LINE),
            });
            unsafe { (api.rs2_delete_device)(device) };
        }

        unsafe { (api.rs2_delete_device_list)(devices) };
        Ok(output)
    })();

    unsafe { (api.rs2_delete_context)(context) };
    result
}

fn device_info(api: &Rs2Api, device: *mut Rs2Device, info: c_int) -> String {
    let supported = api
        .call(|error| unsafe { (api.rs2_supports_device_info)(device, info, error) })
        .unwrap_or(0);
    if supported == 0 {
        return String::new();
    }

    api.call(|error| unsafe { (api.rs2_get_device_info)(device, info, error) })
        .map(|ptr| unsafe { c_string(ptr) })
        .unwrap_or_default()
}

fn cleanup_pipeline_context(api: &Rs2Api, pipeline: *mut Rs2Pipeline, context: *mut Rs2Context) {
    unsafe {
        if !pipeline.is_null() {
            (api.rs2_delete_pipeline)(pipeline);
        }
        if !context.is_null() {
            (api.rs2_delete_context)(context);
        }
    }
}

fn cleanup_config_pipeline_context(
    api: &Rs2Api,
    config: *mut Rs2Config,
    pipeline: *mut Rs2Pipeline,
    context: *mut Rs2Context,
) {
    unsafe {
        if !config.is_null() {
            (api.rs2_delete_config)(config);
        }
    }
    cleanup_pipeline_context(api, pipeline, context);
}

fn library_candidates() -> Vec<String> {
    if cfg!(target_os = "macos") {
        vec![
            "/opt/homebrew/lib/librealsense2.dylib".to_string(),
            "/usr/local/lib/librealsense2.dylib".to_string(),
            "librealsense2.dylib".to_string(),
            "librealsense2.2.dylib".to_string(),
        ]
    } else if cfg!(target_os = "windows") {
        vec!["realsense2.dll".to_string(), "librealsense2.dll".to_string()]
    } else {
        vec!["librealsense2.so".to_string(), "librealsense2.so.2".to_string()]
    }
}

#[derive(Debug)]
struct CommandResult {
    status_success: bool,
    stdout: String,
    stderr: String,
}

impl CommandResult {
    fn summary(&self) -> String {
        let stdout = self.stdout.trim();
        let stderr = self.stderr.trim();
        match (stdout.is_empty(), stderr.is_empty()) {
            (false, false) => format!("{stdout}\n{stderr}"),
            (false, true) => stdout.to_string(),
            (true, false) => stderr.to_string(),
            (true, true) => {
                if self.status_success {
                    "ok".to_string()
                } else {
                    "command failed without output".to_string()
                }
            }
        }
    }
}

fn run_command(path: &PathBuf, args: &[&str]) -> Result<CommandResult, String> {
    let output = Command::new(path)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", path.to_string_lossy()))?;

    Ok(CommandResult {
        status_success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if cfg!(target_os = "macos") {
        candidates.push(PathBuf::from(format!("/opt/homebrew/bin/{name}")));
        candidates.push(PathBuf::from(format!("/usr/local/bin/{name}")));
    }

    if let Some(path_var) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path_var).map(|dir| dir.join(name)));
    }

    candidates.into_iter().find(|path| path.exists())
}

fn brew_path_string(path: Option<&PathBuf>) -> Option<String> {
    path.map(|path| path.to_string_lossy().to_string())
}

fn detect_usb_realsense_devices() -> Vec<UsbRealSenseDevice> {
    let ioreg = PathBuf::from("/usr/sbin/ioreg");
    let output = match run_command(&ioreg, &["-p", "IOUSB", "-l", "-w", "0"]) {
        Ok(output) if output.status_success => output.stdout,
        _ => return Vec::new(),
    };

    let mut devices = Vec::new();
    let mut current: Option<UsbRealSenseDevice> = None;

    for line in output.lines() {
        if line.contains("+-o ") && line.to_ascii_lowercase().contains("realsense") {
            if let Some(device) = current.take() {
                devices.push(device);
            }
            let product_name = line
                .split("+-o ")
                .nth(1)
                .and_then(|value| value.split('@').next())
                .unwrap_or("Intel RealSense")
                .trim()
                .to_string();
            current = Some(UsbRealSenseDevice {
                product_name,
                link_speed_mbps: None,
                usb_type: None,
                id_product: None,
                location_id: None,
            });
            continue;
        }

        if let Some(device) = current.as_mut() {
            if line.contains("\"UsbLinkSpeed\"") {
                if let Some(value) = parse_ioreg_u64(line) {
                    device.link_speed_mbps = Some((value / 1_000_000) as u32);
                }
            } else if line.contains("\"USB Product Name\"") || line.contains("\"kUSBProductString\"") {
                if let Some(value) = parse_ioreg_string(line) {
                    device.product_name = value;
                }
            } else if line.contains("\"idProduct\"") {
                device.id_product = parse_ioreg_u64(line).map(|value| format!("0x{value:04X}"));
            } else if line.contains("\"locationID\"") {
                device.location_id = parse_ioreg_u64(line).map(|value| format!("0x{value:X}"));
            } else if line.contains("\"bcdUSB\"") {
                if let Some(value) = parse_ioreg_u64(line) {
                    let major = (value >> 8) & 0xff;
                    let minor = (value >> 4) & 0x0f;
                    device.usb_type = Some(format!("{major}.{minor}"));
                }
            } else if line.trim() == "}" {
                if let Some(device) = current.take() {
                    devices.push(device);
                }
            }
        }
    }

    if let Some(device) = current {
        devices.push(device);
    }

    devices
}

fn usb_diagnostic_status(devices: &[UsbRealSenseDevice]) -> (String, Option<String>) {
    if devices.is_empty() {
        return ("No RealSense USB device detected".to_string(), None);
    }

    let has_slow_link = devices
        .iter()
        .any(|device| device.link_speed_mbps.unwrap_or(0) < 5_000);

    if has_slow_link {
        let summaries = devices
            .iter()
            .map(|device| {
                format!(
                    "{} at {} Mbps{}",
                    device.product_name,
                    device
                        .link_speed_mbps
                        .map(|speed| speed.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    device
                        .usb_type
                        .as_ref()
                        .map(|usb_type| format!(" / USB {usb_type}"))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        (
            format!("RealSense USB device detected, but not at USB3 speed: {summaries}"),
            Some("The D435i is connected, but the current USB link is below 5 Gbps. Use a short USB3/USB-C data cable, flip/reseat both ends, or try another Mac port; RGB-D streaming cannot open reliably at USB2 speed.".to_string()),
        )
    } else {
        (
            format!("{} RealSense USB device(s) detected, but SDK could not open them", devices.len()),
            Some("Close camera apps, run Setup SDK, then reconnect the camera if the SDK still cannot claim the interface.".to_string()),
        )
    }
}

fn parse_ioreg_u64(line: &str) -> Option<u64> {
    let value = line.split('=').nth(1)?.trim();
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value
            .split(|ch: char| !ch.is_ascii_digit())
            .find(|part| !part.is_empty())
            .and_then(|part| part.parse::<u64>().ok())
    }
}

fn parse_ioreg_string(line: &str) -> Option<String> {
    let value = line.split('=').nth(1)?.trim();
    Some(value.trim_matches('"').to_string())
}

unsafe fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .trim()
            .to_string()
    }
}
