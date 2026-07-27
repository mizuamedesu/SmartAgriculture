use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use cdr::{CdrLe, Infinite};
use jpeg_decoder::{Decoder as JpegDecoder, PixelFormat};
use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
use mcap::{Compression, MessageStream, Summary, WriteOptions, Writer, records::MessageHeader};
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};

use crate::capture::{
    ColorFrame, DepthFrame, DepthStats, Intrinsics, ResolvedCaptureConfig, SensorFrame,
};

pub const RECORDING_FILE_NAME: &str = "recording.mcap";
pub const TOPIC_COLOR: &str = "/camera/color/image/compressed";
pub const TOPIC_DEPTH: &str = "/camera/depth/image_raw";
pub const TOPIC_CAMERA_INFO: &str = "/camera/depth/camera_info";
pub const TOPIC_POINTS: &str = "/camera/depth/color/points";
pub const TOPIC_FRAME_INFO: &str = "/agriscan/frame_info";
pub const TOPIC_SESSION: &str = "/agriscan/session";

const RECORD_JPEG_QUALITY: u8 = 90;
const MCAP_CHUNK_SIZE: u64 = 4 * 1024 * 1024;

const COMPRESSED_IMAGE_SCHEMA: &str = r#"std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"#;

const IMAGE_SCHEMA: &str = r#"std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"#;

const CAMERA_INFO_SCHEMA: &str = r#"std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
"#;

const POINT_CLOUD_SCHEMA: &str = r#"std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/PointField
string name
uint32 offset
uint8 datatype
uint32 count
uint8 INT8=1
uint8 UINT8=2
uint8 INT16=3
uint8 UINT16=4
uint8 INT32=5
uint8 UINT32=6
uint8 FLOAT32=7
uint8 FLOAT64=8
"#;

const SESSION_JSON_SCHEMA: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AgriScanSession",
  "type": "object",
  "additionalProperties": true,
  "required": ["schemaVersion", "sessionId", "event"]
}"#;

const FRAME_JSON_SCHEMA: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AgriScanFrameInfo",
  "type": "object",
  "additionalProperties": false,
  "required": ["schemaVersion", "sessionId", "frameIndex", "frameNumber", "timestampMs", "depthUnitsM", "intrinsics"]
}"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameRecord {
    pub schema_version: String,
    pub session_id: String,
    pub frame_index: u32,
    pub frame_number: u64,
    pub timestamp_ms: f64,
    pub depth_units_m: f32,
    pub intrinsics: Intrinsics,
    pub valid_depth_points: usize,
    pub point_cloud_points: usize,
}

#[derive(Debug)]
pub struct DecodedRgbdFrame {
    pub info: FrameRecord,
    pub color: Option<ColorFrame>,
    pub depth: DepthFrame,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionRecord<'a> {
    schema_version: &'static str,
    session_id: &'a str,
    event: &'a str,
    timestamp: String,
    backend: &'a str,
    frames_written: u32,
    status: &'a str,
    config: &'a ResolvedCaptureConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RosTime {
    sec: i32,
    nanosec: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Header {
    stamp: RosTime,
    frame_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CompressedImage {
    header: Header,
    format: String,
    data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Image {
    header: Header,
    height: u32,
    width: u32,
    encoding: String,
    is_bigendian: u8,
    step: u32,
    data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegionOfInterest {
    x_offset: u32,
    y_offset: u32,
    height: u32,
    width: u32,
    do_rectify: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CameraInfo {
    header: Header,
    height: u32,
    width: u32,
    distortion_model: String,
    d: Vec<f64>,
    k: [f64; 9],
    r: [f64; 9],
    p: [f64; 12],
    binning_x: u32,
    binning_y: u32,
    roi: RegionOfInterest,
}

#[derive(Debug, Serialize, Deserialize)]
struct PointField {
    name: String,
    offset: u32,
    datatype: u8,
    count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct PointCloud2 {
    header: Header,
    height: u32,
    width: u32,
    fields: Vec<PointField>,
    is_bigendian: bool,
    point_step: u32,
    row_step: u32,
    data: Vec<u8>,
    is_dense: bool,
}

#[derive(Debug, Clone)]
struct ChannelIds {
    color: u16,
    depth: u16,
    camera_info: u16,
    points: u16,
    frame_info: u16,
    session: u16,
}

pub struct McapRecorder {
    path: PathBuf,
    session_id: String,
    backend: String,
    config: ResolvedCaptureConfig,
    writer: Option<Writer<File>>,
    channels: ChannelIds,
}

impl McapRecorder {
    pub fn create(
        path: PathBuf,
        session_id: &str,
        backend: &str,
        config: &ResolvedCaptureConfig,
    ) -> Result<Self, String> {
        let file =
            File::create(&path).map_err(|error| format!("failed to create MCAP: {error}"))?;
        let options = WriteOptions::new()
            .profile("ros2")
            .library("AgriScan Studio")
            .chunk_size(Some(MCAP_CHUNK_SIZE))
            .compression(Some(Compression::Zstd))
            .compression_level(3);
        let mut writer = Writer::with_options(file, options)
            .map_err(|error| format!("failed to initialize MCAP: {error}"))?;
        let channels = register_channels(&mut writer)?;
        let mut recorder = Self {
            path,
            session_id: session_id.to_string(),
            backend: backend.to_string(),
            config: config.clone(),
            writer: Some(writer),
            channels,
        };
        recorder.write_session_event("started", "recording", 0)?;
        Ok(recorder)
    }

    pub fn write_frame(
        &mut self,
        frame_index: u32,
        frame: &SensorFrame,
    ) -> Result<(DepthStats, usize), String> {
        let log_time = unix_time_ns();
        let header = Header {
            stamp: ros_time(log_time),
            frame_id: "camera_depth_optical_frame".to_string(),
        };
        let sequence = frame_index;

        if let Some(color) = &frame.color {
            let rgb_jpeg = encode_rgb_jpeg(color)?;
            let message = CompressedImage {
                header: Header {
                    stamp: header.stamp.clone(),
                    frame_id: "camera_color_optical_frame".to_string(),
                },
                format: "jpeg; rgb8".to_string(),
                data: rgb_jpeg,
            };
            self.write_cdr(self.channels.color, sequence, log_time, &message)?;
        }

        let depth_bytes = frame
            .depth
            .z16
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let depth_message = Image {
            header: header.clone(),
            height: frame.depth.height,
            width: frame.depth.width,
            encoding: "16UC1".to_string(),
            is_bigendian: 0,
            step: frame.depth.width * 2,
            data: depth_bytes,
        };
        self.write_cdr(self.channels.depth, sequence, log_time, &depth_message)?;

        let (cloud, stats, point_count) = point_cloud_message(frame, &self.config, &header);
        let frame_info = FrameRecord {
            schema_version: "agriscan-frame-info-v1".to_string(),
            session_id: self.session_id.clone(),
            frame_index,
            frame_number: frame.frame_number,
            timestamp_ms: frame.timestamp_ms,
            depth_units_m: frame.depth.units_m,
            intrinsics: frame.intrinsics,
            valid_depth_points: stats.valid_points,
            point_cloud_points: point_count,
        };
        self.write_json(self.channels.frame_info, sequence, log_time, &frame_info)?;

        let camera_info = camera_info_message(frame.intrinsics, header.clone());
        self.write_cdr(self.channels.camera_info, sequence, log_time, &camera_info)?;
        self.write_cdr(self.channels.points, sequence, log_time, &cloud)?;

        Ok((stats, point_count))
    }

    pub fn finish(mut self, status: &str, frames_written: u32) -> Result<PathBuf, String> {
        self.write_session_event("finished", status, frames_written)?;
        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| "MCAP writer is already closed".to_string())?;
        writer
            .finish()
            .map_err(|error| format!("failed to finalize MCAP: {error}"))?;
        let file = writer.into_inner();
        file.sync_all()
            .map_err(|error| format!("failed to sync MCAP: {error}"))?;
        Ok(self.path)
    }

    fn write_session_event(
        &mut self,
        event: &str,
        status: &str,
        frames_written: u32,
    ) -> Result<(), String> {
        let now = unix_time_ns();
        let record = SessionRecord {
            schema_version: "agriscan-session-v1",
            session_id: &self.session_id,
            event,
            timestamp: chrono::Local::now().to_rfc3339(),
            backend: &self.backend,
            frames_written,
            status,
            config: &self.config,
        };
        let data = serde_json::to_vec(&record)
            .map_err(|error| format!("failed to encode session metadata: {error}"))?;
        self.write_bytes(self.channels.session, frames_written, now, &data)
    }

    fn write_json<T: Serialize>(
        &mut self,
        channel_id: u16,
        sequence: u32,
        log_time: u64,
        value: &T,
    ) -> Result<(), String> {
        let data = serde_json::to_vec(value)
            .map_err(|error| format!("failed to encode MCAP JSON message: {error}"))?;
        self.write_bytes(channel_id, sequence, log_time, &data)
    }

    fn write_cdr<T: Serialize>(
        &mut self,
        channel_id: u16,
        sequence: u32,
        log_time: u64,
        value: &T,
    ) -> Result<(), String> {
        let data = cdr::serialize::<_, _, CdrLe>(value, Infinite)
            .map_err(|error| format!("failed to encode ROS 2 CDR message: {error}"))?;
        self.write_bytes(channel_id, sequence, log_time, &data)
    }

    fn write_bytes(
        &mut self,
        channel_id: u16,
        sequence: u32,
        log_time: u64,
        data: &[u8],
    ) -> Result<(), String> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| "MCAP writer is closed".to_string())?;
        writer
            .write_to_known_channel(
                &MessageHeader {
                    channel_id,
                    sequence,
                    log_time,
                    publish_time: log_time,
                },
                data,
            )
            .map_err(|error| format!("failed to write MCAP message: {error}"))
    }
}

fn register_channels(writer: &mut Writer<File>) -> Result<ChannelIds, String> {
    let compressed_image_schema = add_schema(
        writer,
        "sensor_msgs/msg/CompressedImage",
        "ros2msg",
        COMPRESSED_IMAGE_SCHEMA,
    )?;
    let image_schema = add_schema(writer, "sensor_msgs/msg/Image", "ros2msg", IMAGE_SCHEMA)?;
    let camera_info_schema = add_schema(
        writer,
        "sensor_msgs/msg/CameraInfo",
        "ros2msg",
        CAMERA_INFO_SCHEMA,
    )?;
    let point_cloud_schema = add_schema(
        writer,
        "sensor_msgs/msg/PointCloud2",
        "ros2msg",
        POINT_CLOUD_SCHEMA,
    )?;
    let frame_schema = add_schema(
        writer,
        "agriscan/FrameInfo",
        "jsonschema",
        FRAME_JSON_SCHEMA,
    )?;
    let session_schema = add_schema(
        writer,
        "agriscan/Session",
        "jsonschema",
        SESSION_JSON_SCHEMA,
    )?;
    let metadata = BTreeMap::new();
    Ok(ChannelIds {
        color: add_channel(
            writer,
            compressed_image_schema,
            TOPIC_COLOR,
            "cdr",
            &metadata,
        )?,
        depth: add_channel(writer, image_schema, TOPIC_DEPTH, "cdr", &metadata)?,
        camera_info: add_channel(
            writer,
            camera_info_schema,
            TOPIC_CAMERA_INFO,
            "cdr",
            &metadata,
        )?,
        points: add_channel(writer, point_cloud_schema, TOPIC_POINTS, "cdr", &metadata)?,
        frame_info: add_channel(writer, frame_schema, TOPIC_FRAME_INFO, "json", &metadata)?,
        session: add_channel(writer, session_schema, TOPIC_SESSION, "json", &metadata)?,
    })
}

fn add_schema(
    writer: &mut Writer<File>,
    name: &str,
    encoding: &str,
    data: &str,
) -> Result<u16, String> {
    writer
        .add_schema(name, encoding, data.as_bytes())
        .map_err(|error| format!("failed to add MCAP schema {name}: {error}"))
}

fn add_channel(
    writer: &mut Writer<File>,
    schema_id: u16,
    topic: &str,
    encoding: &str,
    metadata: &BTreeMap<String, String>,
) -> Result<u16, String> {
    writer
        .add_channel(schema_id, topic, encoding, metadata)
        .map_err(|error| format!("failed to add MCAP channel {topic}: {error}"))
}

fn encode_rgb_jpeg(color: &ColorFrame) -> Result<Vec<u8>, String> {
    let mut data = Vec::new();
    JpegEncoder::new(&mut data, RECORD_JPEG_QUALITY)
        .encode(
            &color.rgb,
            color.width as u16,
            color.height as u16,
            JpegColorType::Rgb,
        )
        .map_err(|error| format!("failed to encode recording JPEG: {error}"))?;
    Ok(data)
}

fn camera_info_message(intrinsics: Intrinsics, header: Header) -> CameraInfo {
    let fx = intrinsics.fx as f64;
    let fy = intrinsics.fy as f64;
    let cx = intrinsics.ppx as f64;
    let cy = intrinsics.ppy as f64;
    CameraInfo {
        header,
        height: intrinsics.height,
        width: intrinsics.width,
        distortion_model: "plumb_bob".to_string(),
        d: intrinsics.coeffs.into_iter().map(f64::from).collect(),
        k: [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p: [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x: 0,
        binning_y: 0,
        roi: RegionOfInterest {
            x_offset: 0,
            y_offset: 0,
            height: 0,
            width: 0,
            do_rectify: false,
        },
    }
}

fn point_cloud_message(
    frame: &SensorFrame,
    config: &ResolvedCaptureConfig,
    header: &Header,
) -> (PointCloud2, DepthStats, usize) {
    let depth = &frame.depth;
    let intrinsics = frame.intrinsics;
    let stride = config.point_stride.max(1) as usize;
    let mut data = Vec::with_capacity(depth.z16.len() / stride * 4);
    let mut valid = 0usize;
    let mut min_m = f32::MAX;
    let mut max_m = 0.0f32;
    let mut sum = 0.0f64;

    for y in (0..depth.height as usize).step_by(stride) {
        for x in (0..depth.width as usize).step_by(stride) {
            let raw = depth.z16[y * depth.width as usize + x];
            let z = raw as f32 * depth.units_m;
            if raw == 0 || z < config.min_depth_m || z > config.max_depth_m {
                continue;
            }
            valid += 1;
            min_m = min_m.min(z);
            max_m = max_m.max(z);
            sum += z as f64;
            let px = (x as f32 - intrinsics.ppx) / intrinsics.fx * z;
            let py = (y as f32 - intrinsics.ppy) / intrinsics.fy * z;
            let (r, g, b) = sample_color(
                frame.color.as_ref(),
                x,
                y,
                depth.width as usize,
                depth.height as usize,
            );
            let rgb = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
            data.extend_from_slice(&px.to_le_bytes());
            data.extend_from_slice(&py.to_le_bytes());
            data.extend_from_slice(&z.to_le_bytes());
            data.extend_from_slice(&rgb.to_le_bytes());
        }
    }

    let stats = if valid == 0 {
        DepthStats {
            valid_points: 0,
            min_m: 0.0,
            max_m: 0.0,
            mean_m: 0.0,
        }
    } else {
        DepthStats {
            valid_points: valid,
            min_m,
            max_m,
            mean_m: (sum / valid as f64) as f32,
        }
    };
    let point_count = valid;
    (
        PointCloud2 {
            header: header.clone(),
            height: 1,
            width: point_count as u32,
            fields: vec![
                PointField {
                    name: "x".to_string(),
                    offset: 0,
                    datatype: 7,
                    count: 1,
                },
                PointField {
                    name: "y".to_string(),
                    offset: 4,
                    datatype: 7,
                    count: 1,
                },
                PointField {
                    name: "z".to_string(),
                    offset: 8,
                    datatype: 7,
                    count: 1,
                },
                PointField {
                    name: "rgb".to_string(),
                    offset: 12,
                    datatype: 6,
                    count: 1,
                },
            ],
            is_bigendian: false,
            point_step: 16,
            row_step: point_count as u32 * 16,
            data,
            is_dense: true,
        },
        stats,
        point_count,
    )
}

fn sample_color(
    color: Option<&ColorFrame>,
    x: usize,
    y: usize,
    depth_width: usize,
    depth_height: usize,
) -> (u8, u8, u8) {
    let Some(color) = color else {
        return (200, 200, 200);
    };
    let sx = (x * color.width as usize / depth_width.max(1)).min(color.width as usize - 1);
    let sy = (y * color.height as usize / depth_height.max(1)).min(color.height as usize - 1);
    let index = (sy * color.width as usize + sx) * 3;
    (color.rgb[index], color.rgb[index + 1], color.rgb[index + 2])
}

fn unix_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

fn ros_time(time_ns: u64) -> RosTime {
    RosTime {
        sec: (time_ns / 1_000_000_000).min(i32::MAX as u64) as i32,
        nanosec: (time_ns % 1_000_000_000) as u32,
    }
}

pub fn frame_count(path: &Path) -> Result<usize, String> {
    let mapped = map_recording(path)?;
    let summary = Summary::read(&mapped)
        .map_err(|error| format!("failed to read MCAP summary: {error}"))?
        .ok_or_else(|| "MCAP has no finalized summary".to_string())?;
    let depth_channel = summary
        .channels
        .values()
        .find(|channel| channel.topic == TOPIC_DEPTH)
        .ok_or_else(|| format!("MCAP is missing {TOPIC_DEPTH}"))?;
    Ok(summary
        .stats
        .as_ref()
        .and_then(|stats| stats.channel_message_counts.get(&depth_channel.id))
        .copied()
        .unwrap_or(0) as usize)
}

pub fn visit_frame_indices<F>(
    path: &Path,
    frame_indices: &BTreeSet<u32>,
    mut visitor: F,
) -> Result<(), String>
where
    F: FnMut(DecodedRgbdFrame) -> Result<bool, String>,
{
    let mapped = map_recording(path)?;
    let mut pending: HashMap<u32, PendingFrame> = HashMap::new();
    let messages =
        MessageStream::new(&mapped).map_err(|error| format!("failed to open MCAP: {error}"))?;
    for message in messages {
        let message = message.map_err(|error| format!("failed to read MCAP message: {error}"))?;
        let sequence = message.sequence;
        if !frame_indices.contains(&sequence) {
            continue;
        }
        match message.channel.topic.as_str() {
            TOPIC_COLOR => {
                let image: CompressedImage = cdr::deserialize(&message.data)
                    .map_err(|error| format!("failed to decode MCAP RGB message: {error}"))?;
                pending.entry(sequence).or_default().color = Some(decode_jpeg(&image.data)?);
            }
            TOPIC_DEPTH => {
                let image: Image = cdr::deserialize(&message.data)
                    .map_err(|error| format!("failed to decode MCAP depth message: {error}"))?;
                if image.encoding != "16UC1" || image.is_bigendian != 0 {
                    return Err(format!(
                        "unsupported MCAP depth encoding: {} big-endian={}",
                        image.encoding, image.is_bigendian
                    ));
                }
                let z16 = image
                    .data
                    .chunks_exact(2)
                    .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
                    .collect();
                pending.entry(sequence).or_default().depth = Some((image.width, image.height, z16));
            }
            TOPIC_FRAME_INFO => {
                let info: FrameRecord = serde_json::from_slice(&message.data)
                    .map_err(|error| format!("failed to decode MCAP frame info: {error}"))?;
                let entry = pending.remove(&sequence).unwrap_or_default();
                let (width, height, z16) = entry
                    .depth
                    .ok_or_else(|| format!("MCAP frame {sequence} has no depth image"))?;
                let frame = DecodedRgbdFrame {
                    depth: DepthFrame {
                        width,
                        height,
                        z16,
                        units_m: info.depth_units_m,
                    },
                    color: entry.color,
                    info,
                };
                if !visitor(frame)? {
                    break;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Default)]
struct PendingFrame {
    color: Option<ColorFrame>,
    depth: Option<(u32, u32, Vec<u16>)>,
}

fn map_recording(path: &Path) -> Result<memmap2::Mmap, String> {
    let file = File::open(path).map_err(|error| format!("failed to open MCAP: {error}"))?;
    // SAFETY: capture is stopped and the file is finalized before asset generation opens it.
    unsafe { MmapOptions::new().map(&file) }
        .map_err(|error| format!("failed to memory-map MCAP: {error}"))
}

fn decode_jpeg(data: &[u8]) -> Result<ColorFrame, String> {
    let mut decoder = JpegDecoder::new(Cursor::new(data));
    let decoded = decoder
        .decode()
        .map_err(|error| format!("failed to decode MCAP RGB JPEG: {error}"))?;
    let info = decoder
        .info()
        .ok_or_else(|| "MCAP RGB JPEG has no image info".to_string())?;
    let rgb = match info.pixel_format {
        PixelFormat::RGB24 => decoded,
        PixelFormat::L8 => decoded
            .into_iter()
            .flat_map(|value| [value, value, value])
            .collect(),
        other => return Err(format!("unsupported MCAP JPEG pixel format: {other:?}")),
    };
    Ok(ColorFrame {
        width: info.width as u32,
        height: info.height as u32,
        rgb,
    })
}

#[allow(dead_code)]
pub fn validate_recording(path: &Path) -> Result<Vec<String>, String> {
    let mapped = map_recording(path)?;
    let summary = Summary::read(&mapped)
        .map_err(|error| format!("failed to validate MCAP summary: {error}"))?
        .ok_or_else(|| "MCAP is not finalized".to_string())?;
    let mut topics: Vec<_> = summary
        .channels
        .values()
        .map(|channel| channel.topic.clone())
        .collect();
    topics.sort();
    for required in [
        TOPIC_COLOR,
        TOPIC_DEPTH,
        TOPIC_CAMERA_INFO,
        TOPIC_POINTS,
        TOPIC_FRAME_INFO,
        TOPIC_SESSION,
    ] {
        if !topics.iter().any(|topic| topic == required) {
            return Err(format!("MCAP is missing required topic {required}"));
        }
    }
    Ok(topics)
}

pub fn recording_path(session_root: &Path) -> PathBuf {
    session_root.join(RECORDING_FILE_NAME)
}

pub fn find_recording_path(session_root: &Path) -> Option<PathBuf> {
    let legacy_name = recording_path(session_root);
    if legacy_name.is_file() {
        return Some(legacy_name);
    }
    let mut recordings: Vec<_> = std::fs::read_dir(session_root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        extension.eq_ignore_ascii_case("mcap")
                            || extension.eq_ignore_ascii_case("mcp")
                    })
        })
        .collect();
    recordings.sort();
    recordings.into_iter().next()
}
