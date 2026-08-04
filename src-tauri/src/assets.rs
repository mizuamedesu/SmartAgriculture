use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    ffi::OsString,
    fs::{self, File},
    io::{BufReader, BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder};
use png::{BitDepth, ColorType, Decoder, Encoder as PngEncoder};
use serde::{Deserialize, Serialize};

use crate::{
    capture::{ColorFrame, Intrinsics, default_output_root, legacy_output_root},
    fbx::{FbxMesh, FbxVertex, write_fbx},
    mcap_io::{self, DecodedRgbdFrame},
};

const SH_C0: f32 = 0.282_094_8;
const SHARP_MLX_INFERENCE_SCRIPT: &str = include_str!("../../scripts/sharp_mlx_inference.py");
const SHARP_MLX_RUNTIME: &[(&str, &str)] = &[
    (
        "__init__.py",
        include_str!("../../scripts/sharp_mlx/__init__.py"),
    ),
    (
        "blocks.py",
        include_str!("../../scripts/sharp_mlx/blocks.py"),
    ),
    (
        "decoder.py",
        include_str!("../../scripts/sharp_mlx/decoder.py"),
    ),
    (
        "gaussian.py",
        include_str!("../../scripts/sharp_mlx/gaussian.py"),
    ),
    (
        "gaussian_utils.py",
        include_str!("../../scripts/sharp_mlx/gaussian_utils.py"),
    ),
    (
        "monodepth.py",
        include_str!("../../scripts/sharp_mlx/monodepth.py"),
    ),
    (
        "predictor.py",
        include_str!("../../scripts/sharp_mlx/predictor.py"),
    ),
    (
        "spn_encoder.py",
        include_str!("../../scripts/sharp_mlx/spn_encoder.py"),
    ),
    ("vit.py", include_str!("../../scripts/sharp_mlx/vit.py")),
    (
        "weights.py",
        include_str!("../../scripts/sharp_mlx/weights.py"),
    ),
    ("LICENSE", include_str!("../../scripts/sharp_mlx/LICENSE")),
];
const OPEN3D_ODOMETRY_SCRIPT: &str = include_str!("../../scripts/open3d_rgbd_odometry.py");

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetBuildOptions {
    pub session_root: String,
    pub max_points: Option<usize>,
    pub frame_stride: Option<u32>,
    pub depth_decimation: Option<u32>,
    pub gaussian_radius_m: Option<f32>,
    pub turntable_degrees: Option<f32>,
    pub export_fbx: Option<bool>,
    pub use_mlx: Option<bool>,
    pub mlx_iterations: Option<u32>,
    pub mlx_voxel_size_m: Option<f32>,
    pub mlx_train_size: Option<u32>,
    pub mlx_max_train_views: Option<u32>,
    pub collider_max_faces: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetTools {
    pub fbx_available: bool,
    pub fbx_exporter: String,
    pub python: Option<String>,
    pub mlx_available: bool,
    pub mlx_status: String,
    pub brush_hint: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MlxSetupResult {
    pub status: String,
    pub log: Vec<String>,
    pub tools: AssetTools,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetBuildResult {
    pub root: String,
    pub seed_gaussian_ply: String,
    pub gaussian_ply: String,
    pub splat: String,
    pub mesh_obj: String,
    pub mesh_fbx: Option<String>,
    pub collider_obj: String,
    pub collision_json: String,
    pub collision_fbx: Option<String>,
    pub preview_json: String,
    pub manifest: String,
    pub point_count: usize,
    pub face_count: usize,
    pub fbx_status: String,
    pub mlx_status: String,
    pub collision_status: String,
    pub tools: AssetTools,
    pub preview: PreviewPayload,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameMetadata {
    session_id: String,
    frame_index: u32,
    frame_number: u64,
    timestamp_ms: f64,
    intrinsics: Intrinsics,
    depth_units_m: f32,
    files: FrameFiles,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FrameFiles {
    rgb: Option<String>,
    depth: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AssetManifest<'a> {
    schema_version: &'static str,
    source_session: &'a str,
    point_count: usize,
    face_count: usize,
    seed_gaussian_ply: &'a str,
    gaussian_ply: &'a str,
    splat: &'a str,
    mesh_obj: &'a str,
    mesh_fbx: Option<&'a str>,
    collider_obj: &'a str,
    collision_json: &'a str,
    collision_fbx: Option<&'a str>,
    preview_json: &'a str,
    fbx_status: &'a str,
    mlx_status: &'a str,
    collision_status: &'a str,
    options: AssetOptionsSummary,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AssetOptionsSummary {
    max_points: usize,
    frame_stride: u32,
    depth_decimation: u32,
    gaussian_radius_m: f32,
    turntable_degrees: f32,
    use_mlx: bool,
    mlx_iterations: u32,
    mlx_voxel_size_m: f32,
    mlx_train_size: u32,
    mlx_max_train_views: u32,
    collider_max_faces: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewPayload {
    pub points: Vec<PreviewPoint>,
    pub bounds: Bounds,
    pub initial_camera: Option<PreviewCamera>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub radius: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub opacity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewCamera {
    pub rotation: [[f32; 3]; 3],
    pub translation: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub center: [f32; 3],
}

#[derive(Debug, Clone)]
struct SplatPoint {
    x: f32,
    y: f32,
    z: f32,
    r: u8,
    g: u8,
    b: u8,
    radius: f32,
    scale: [f32; 3],
    rotation: [f32; 4],
    opacity_logit: f32,
}

#[derive(Debug, Clone)]
struct MeshBuild {
    vertices: Vec<SplatPoint>,
    faces: Vec<[u32; 3]>,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct RigidTransform {
    rotation: [[f32; 3]; 3],
    translation: [f32; 3],
}

impl RigidTransform {
    fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    fn rotation_y(angle: f32) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            rotation: [[cos_a, 0.0, -sin_a], [0.0, 1.0, 0.0], [sin_a, 0.0, cos_a]],
            translation: [0.0; 3],
        }
    }

    fn apply(self, point: [f32; 3]) -> [f32; 3] {
        [
            dot3(self.rotation[0], point) + self.translation[0],
            dot3(self.rotation[1], point) + self.translation[1],
            dot3(self.rotation[2], point) + self.translation[2],
        ]
    }

    fn then(self, next: Self) -> Self {
        let mut rotation = [[0.0; 3]; 3];
        for (row, values) in rotation.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = (0..3)
                    .map(|index| next.rotation[row][index] * self.rotation[index][column])
                    .sum();
            }
        }
        Self {
            rotation,
            translation: add3(mat3_vec(next.rotation, self.translation), next.translation),
        }
    }
}

#[derive(Debug)]
struct PositionedRgbdFrame {
    frame: DecodedRgbdFrame,
    camera_to_world: RigidTransform,
    min_depth_m: f32,
    max_depth_m: f32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CollisionManifest {
    schema_version: &'static str,
    collider_type: &'static str,
    collider_obj: String,
    source_mesh: String,
    point_count: usize,
    face_count: usize,
    bounds: Bounds,
    bounding_sphere: BoundingSphere,
    notes: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BoundingSphere {
    center: [f32; 3],
    radius: f32,
}

struct MlxRefinement {
    points: Vec<SplatPoint>,
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredAssetManifest {
    point_count: usize,
    face_count: usize,
    seed_gaussian_ply: String,
    gaussian_ply: String,
    splat: String,
    mesh_obj: String,
    mesh_fbx: Option<String>,
    collider_obj: String,
    collision_json: String,
    collision_fbx: Option<String>,
    preview_json: String,
    fbx_status: String,
    mlx_status: String,
    collision_status: String,
}

#[tauri::command]
pub fn detect_asset_tools() -> AssetTools {
    let python = find_python();
    let (mlx_available, mlx_status) = match python.as_deref() {
        Some(path) => probe_mlx(path),
        None => (
            false,
            "python3 not found; SHARP MLX inference unavailable".to_string(),
        ),
    };
    AssetTools {
        fbx_available: true,
        fbx_exporter: "Built-in native FBX 7.4 exporter".to_string(),
        python,
        mlx_available,
        mlx_status,
        brush_hint: "Pretrained SHARP runs as feed-forward inference on MLX/Metal. FBX export is built in and does not use Blender.".to_string(),
    }
}

#[tauri::command]
pub fn load_latest_scan_assets() -> Result<Option<AssetBuildResult>, String> {
    let mut latest: Option<(SystemTime, PathBuf)> = None;
    for scans_root in [default_output_root()?, legacy_output_root()] {
        if !scans_root.is_dir() {
            continue;
        }
        let entries = fs::read_dir(&scans_root)
            .map_err(|error| format!("failed to scan previous captures: {error}"))?;

        for entry in entries.flatten() {
            let manifest_path = entry.path().join("assets").join("asset_manifest.json");
            if !manifest_path.is_file() {
                continue;
            }
            let modified = fs::metadata(&manifest_path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if latest
                .as_ref()
                .is_none_or(|(latest_modified, _)| modified > *latest_modified)
            {
                latest = Some((modified, manifest_path));
            }
        }
    }

    let Some((_, manifest_path)) = latest else {
        return Ok(None);
    };
    load_asset_manifest(&manifest_path).map(Some)
}

#[tauri::command]
pub fn load_scan_data(path: String) -> Result<AssetBuildResult, String> {
    let source = PathBuf::from(path);
    if !source.exists() {
        return Err("selected scan data does not exist".to_string());
    }
    if source.is_dir() {
        for manifest_path in [
            source.join("asset_manifest.json"),
            source.join("assets").join("asset_manifest.json"),
        ] {
            if manifest_path.is_file() {
                return load_asset_manifest(&manifest_path);
            }
        }
        if let Some(recording) = mcap_io::find_recording_path(&source) {
            return load_mcap_preview(&recording);
        }
        return Err("selected folder has no MCAP recording or asset manifest".to_string());
    }

    match source
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mcap") | Some("mcp") => load_mcap_preview(&source),
        Some("ply") => load_point_preview(&source, read_gaussian_ply(&source)?, false),
        Some("splat") => load_point_preview(&source, read_splat(&source)?, true),
        Some("json")
            if source.file_name().and_then(|name| name.to_str()) == Some("asset_manifest.json") =>
        {
            load_asset_manifest(&source)
        }
        _ => Err("select an .mcap, .mcp, .ply, .splat, or asset_manifest.json file".to_string()),
    }
}

pub fn export_mcap_sample_frames(
    recording_path: &Path,
    output_root: &Path,
) -> Result<Vec<String>, String> {
    let total_frames = mcap_io::frame_count(recording_path)?;
    if total_frames == 0 {
        return Err("MCAP contains no RGB-D frames".to_string());
    }
    fs::create_dir_all(output_root)
        .map_err(|error| format!("failed to create diagnostic frame folder: {error}"))?;
    let indices: BTreeSet<u32> = [1, total_frames.div_ceil(2) as u32, total_frames as u32]
        .into_iter()
        .collect();
    let mut outputs = Vec::new();
    mcap_io::visit_frame_indices(recording_path, &indices, |frame| {
        let stem = format!("frame_{:06}", frame.info.frame_index);
        if let Some(color) = &frame.color {
            let color_path = output_root.join(format!("{stem}_rgb.png"));
            write_rgb_png(&color_path, color)?;
            outputs.push(path_string(&color_path));
        }
        let depth_path = output_root.join(format!("{stem}_depth_z16.png"));
        write_depth_png(
            &depth_path,
            frame.depth.width,
            frame.depth.height,
            &frame.depth.z16,
        )?;
        outputs.push(path_string(&depth_path));
        let metadata_path = output_root.join(format!("{stem}_metadata.json"));
        write_json(
            &metadata_path,
            &serde_json::json!({
                "frameIndex": frame.info.frame_index,
                "frameNumber": frame.info.frame_number,
                "timestampMs": frame.info.timestamp_ms,
                "intrinsics": frame.info.intrinsics,
                "depthUnitsM": frame.info.depth_units_m,
                "files": {
                    "rgb": frame.color.as_ref().map(|_| path_string(&output_root.join(format!("{stem}_rgb.png")))),
                    "depth": path_string(&depth_path)
                }
            }),
        )?;
        outputs.push(path_string(&metadata_path));
        Ok(true)
    })?;
    Ok(outputs)
}

fn load_mcap_preview(recording: &Path) -> Result<AssetBuildResult, String> {
    let session_root = recording
        .parent()
        .ok_or_else(|| "MCAP recording has no parent folder".to_string())?;
    generate_scan_assets(AssetBuildOptions {
        session_root: path_string(session_root),
        max_points: Some(750_000),
        frame_stride: Some(1),
        depth_decimation: Some(2),
        gaussian_radius_m: Some(0.0035),
        turntable_degrees: Some(0.0),
        export_fbx: Some(false),
        use_mlx: Some(true),
        mlx_iterations: Some(0),
        mlx_voxel_size_m: Some(0.0025),
        mlx_train_size: Some(1536),
        mlx_max_train_views: Some(100),
        collider_max_faces: Some(35_000),
    })
}

fn load_point_preview(
    source: &Path,
    points: Vec<SplatPoint>,
    source_is_splat: bool,
) -> Result<AssetBuildResult, String> {
    if points.is_empty() {
        return Err("selected 3DGS file contains no points".to_string());
    }
    let root = source
        .parent()
        .ok_or_else(|| "selected 3DGS file has no parent folder".to_string())?;
    let source_string = path_string(source);
    let preview = build_preview_payload(&points, None);
    Ok(AssetBuildResult {
        root: path_string(root),
        seed_gaussian_ply: if source_is_splat {
            "-".to_string()
        } else {
            source_string.clone()
        },
        gaussian_ply: if source_is_splat {
            "-".to_string()
        } else {
            source_string.clone()
        },
        splat: if source_is_splat {
            source_string
        } else {
            "-".to_string()
        },
        mesh_obj: "-".to_string(),
        mesh_fbx: None,
        collider_obj: "-".to_string(),
        collision_json: "-".to_string(),
        collision_fbx: None,
        preview_json: "-".to_string(),
        manifest: "-".to_string(),
        point_count: points.len(),
        face_count: 0,
        fbx_status: "No FBX loaded".to_string(),
        mlx_status: "Loaded existing 3DGS data".to_string(),
        collision_status: "No collider loaded".to_string(),
        tools: detect_asset_tools(),
        preview,
    })
}

fn load_asset_manifest(manifest_path: &Path) -> Result<AssetBuildResult, String> {
    let manifest_data = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("failed to read previous asset manifest: {error}"))?;
    let manifest: StoredAssetManifest = serde_json::from_str(&manifest_data)
        .map_err(|error| format!("failed to parse previous asset manifest: {error}"))?;
    let preview_data = fs::read_to_string(&manifest.preview_json)
        .map_err(|error| format!("failed to read previous 3D preview: {error}"))?;
    let preview: PreviewPayload = serde_json::from_str(&preview_data)
        .map_err(|error| format!("failed to parse previous 3D preview: {error}"))?;
    let asset_root = manifest_path
        .parent()
        .ok_or_else(|| "previous asset manifest has no parent directory".to_string())?;

    Ok(AssetBuildResult {
        root: path_string(asset_root),
        seed_gaussian_ply: manifest.seed_gaussian_ply,
        gaussian_ply: manifest.gaussian_ply,
        splat: manifest.splat,
        mesh_obj: manifest.mesh_obj,
        mesh_fbx: manifest.mesh_fbx,
        collider_obj: manifest.collider_obj,
        collision_json: manifest.collision_json,
        collision_fbx: manifest.collision_fbx,
        preview_json: manifest.preview_json,
        manifest: path_string(&manifest_path),
        point_count: manifest.point_count,
        face_count: manifest.face_count,
        fbx_status: manifest.fbx_status,
        mlx_status: manifest.mlx_status,
        collision_status: manifest.collision_status,
        tools: detect_asset_tools(),
        preview,
    })
}

#[tauri::command]
pub fn ensure_mlx_3dgs() -> Result<MlxSetupResult, String> {
    let system_python = find_system_python()
        .ok_or_else(|| "python3 not found; install Python 3.10+ first".to_string())?;
    let venv_dir = mlx_venv_dir();
    let python = ensure_mlx_venv(&system_python, &venv_dir)?;
    let mut log = Vec::new();
    log.push(format!("MLX 3DGS venv: {}", path_string(&venv_dir)));

    let mut commands = vec![
        vec![
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        vec![
            "-m",
            "pip",
            "install",
            "--upgrade",
            "mlx",
            "numpy",
            "pillow",
            "scipy",
            "safetensors",
        ],
        vec![
            "-m",
            "pip",
            "install",
            "--upgrade",
            "huggingface_hub[hf_xet]",
        ],
    ];

    for args in commands.drain(..) {
        let result = run_python_install(&python, &args)?;
        log.push(result);
    }

    let tools = detect_asset_tools();
    if tools.mlx_available {
        let checkpoint = ensure_sharp_checkpoint(&python)?;
        log.push(format!(
            "Pretrained SHARP checkpoint: {}",
            path_string(&checkpoint)
        ));
        Ok(MlxSetupResult {
            status: format!("{}; pretrained SHARP ready", tools.mlx_status),
            log,
            tools,
        })
    } else {
        Err(format!(
            "SHARP MLX setup finished but probe failed: {}",
            tools.mlx_status
        ))
    }
}

#[tauri::command]
pub fn generate_scan_assets(options: AssetBuildOptions) -> Result<AssetBuildResult, String> {
    let session_root = PathBuf::from(&options.session_root);
    if !session_root.exists() {
        return Err("session root does not exist".to_string());
    }

    let frame_stride = options.frame_stride.unwrap_or(1).max(1);
    let depth_decimation = options.depth_decimation.unwrap_or(4).clamp(1, 16);
    let max_points = options
        .max_points
        .unwrap_or(180_000)
        .clamp(5_000, 1_500_000);
    let gaussian_radius_m = options
        .gaussian_radius_m
        .unwrap_or(0.006)
        .clamp(0.0005, 0.05);
    let turntable_degrees = options.turntable_degrees.unwrap_or(0.0).clamp(0.0, 1080.0);
    let export_fbx = options.export_fbx.unwrap_or(true);
    let use_mlx = options.use_mlx.unwrap_or(true);
    let mlx_iterations = options.mlx_iterations.unwrap_or(0).clamp(0, 20_000);
    let mlx_voxel_size_m = options
        .mlx_voxel_size_m
        .unwrap_or(gaussian_radius_m * 0.75)
        .clamp(0.0005, 0.05);
    let mlx_train_size = options.mlx_train_size.unwrap_or(1_536).clamp(64, 1_536);
    let mlx_max_train_views = options.mlx_max_train_views.unwrap_or(100).clamp(1, 512);
    let collider_max_faces = options
        .collider_max_faces
        .unwrap_or(35_000)
        .clamp(500, 120_000);

    let recording_path = mcap_io::find_recording_path(&session_root);
    let is_mcap = recording_path.is_some();
    let selected = if is_mcap {
        Vec::new()
    } else {
        let frames = load_frame_metadata(&session_root)?;
        let selected: Vec<_> = frames
            .into_iter()
            .enumerate()
            .filter_map(|(index, frame)| {
                (index as u32).is_multiple_of(frame_stride).then_some(frame)
            })
            .collect();
        if selected.is_empty() {
            return Err("no frames selected for asset generation".to_string());
        }
        selected
    };
    let source_session = if is_mcap {
        session_root
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string()
    } else {
        selected
            .first()
            .map(|frame| frame.session_id.clone())
            .unwrap_or_else(|| "unknown".to_string())
    };

    let asset_root = session_root.join("assets");
    let gaussian_dir = asset_root.join("gaussian_splats");
    let mesh_dir = asset_root.join("mesh");
    let mlx_dir = asset_root.join("mlx");
    let odometry_dir = asset_root.join("odometry");
    let preview_dir = asset_root.join("preview");
    for dir in [
        &asset_root,
        &gaussian_dir,
        &mesh_dir,
        &mlx_dir,
        &odometry_dir,
        &preview_dir,
    ] {
        fs::create_dir_all(dir).map_err(|error| format!("failed to create {dir:?}: {error}"))?;
    }

    let (mesh, mcap_training_frames) = if is_mcap {
        build_mesh_from_mcap(
            recording_path
                .as_deref()
                .ok_or_else(|| "MCAP recording path is missing".to_string())?,
            frame_stride,
            depth_decimation,
            max_points,
            gaussian_radius_m,
            turntable_degrees,
            mlx_max_train_views,
            &odometry_dir,
        )?
    } else {
        (
            build_mesh(
                &selected,
                depth_decimation,
                max_points,
                gaussian_radius_m,
                turntable_degrees,
            )?,
            Vec::new(),
        )
    };
    let preview_camera = mcap_training_frames
        .iter()
        .max_by_key(|positioned| {
            positioned
                .frame
                .depth
                .z16
                .iter()
                .filter(|&&raw_depth| {
                    let depth_m = raw_depth as f32 * positioned.frame.info.depth_units_m;
                    depth_m >= positioned.min_depth_m && depth_m <= positioned.max_depth_m
                })
                .count()
        })
        .map(|frame| PreviewCamera {
            rotation: frame.camera_to_world.rotation,
            translation: frame.camera_to_world.translation,
        });
    if mesh.vertices.is_empty() {
        return Err("no valid depth points available for 3D reconstruction".to_string());
    }

    let seed_gaussian_ply = gaussian_dir.join("scan_gaussians_seed.ply");
    let seed_splat = gaussian_dir.join("scan_gaussians_seed.splat");
    let mlx_gaussian_ply = gaussian_dir.join("scan_gaussians_mlx.ply");
    let sharp_raw_ply = mlx_dir.join("sharp_inferred_views.ply");
    let mlx_splat = gaussian_dir.join("scan_gaussians_mlx.splat");
    let mesh_obj = mesh_dir.join("scan_surface.obj");
    let mesh_fbx = mesh_dir.join("scan_surface.fbx");
    let collider_obj = mesh_dir.join("scan_collider.obj");
    let collision_json = mesh_dir.join("scan_collision.json");
    let preview_json = preview_dir.join("preview_points.json");
    let manifest = asset_root.join("asset_manifest.json");

    let source_frame_count = if let Some(recording_path) = recording_path.as_deref() {
        mcap_io::frame_count(recording_path)?
    } else {
        selected.len()
    };
    let seed_points = if source_frame_count <= 1 {
        mesh.vertices.clone()
    } else {
        fuse_splat_points(&mesh.vertices, (gaussian_radius_m * 1.15).max(0.004))
    };
    write_gaussian_ply(&seed_gaussian_ply, &seed_points)?;
    write_splat(&seed_splat, &seed_points)?;
    write_obj(&mesh_obj, &mesh)?;
    let collider_mesh = build_collision_mesh(&mesh, collider_max_faces);
    write_obj(&collider_obj, &collider_mesh)?;
    let collision_status = write_collision_manifest(
        &collision_json,
        &collider_mesh,
        &mesh_obj,
        &collider_obj,
        collider_max_faces,
    )?;

    let mut final_points = seed_points.clone();
    let mut final_gaussian_ply = seed_gaussian_ply.clone();
    let mut final_splat = seed_splat.clone();
    let mut mlx_status = "RGB-D Gaussian seed (pretrained SHARP disabled)".to_string();

    if use_mlx {
        let (mlx_session_root, cleanup_cache) = if is_mcap {
            let cache = mlx_dir.join("mcap_training_cache");
            write_mcap_training_cache(&cache, &mcap_training_frames)?;
            (cache.clone(), Some(cache))
        } else {
            (session_root.clone(), None)
        };
        let sharp_budget = max_points.saturating_mul(3).div_ceil(4).max(10_000);
        let refinement_result = run_sharp_mlx_inference(
            &mlx_session_root,
            &sharp_raw_ply,
            &mlx_dir,
            sharp_budget,
            mlx_max_train_views,
        );
        if let Some(cache) = cleanup_cache {
            let _ = fs::remove_dir_all(cache);
        }
        let refinement = refinement_result.map_err(|error| {
            format!(
                "Pretrained SHARP MLX inference was requested but failed; no fallback was reported as success: {error}"
            )
        })?;

        let seed_budget = max_points.saturating_sub(refinement.points.len());
        let mut combined = refinement.points;
        combined.extend(sample_splat_points(&seed_points, seed_budget));
        final_points = sample_splat_points(&combined, max_points);
        write_gaussian_ply(&mlx_gaussian_ply, &final_points)?;
        final_gaussian_ply = mlx_gaussian_ply;
        final_splat = mlx_splat;
        write_splat(&final_splat, &final_points)?;
        mlx_status = format!(
            "{}; merged with {} measured RGB-D splats",
            refinement.status,
            seed_budget.min(seed_points.len())
        );
    }

    let preview = build_preview_payload(&final_points, preview_camera);
    write_preview_json(&preview_json, &preview)?;

    let fbx_status = if export_fbx {
        export_fbx_native(&mesh_fbx, &mesh, &collider_mesh)?
    } else {
        "FBX export disabled".to_string()
    };

    let tools = detect_asset_tools();
    let mesh_fbx_output = mesh_fbx.exists().then(|| path_string(&mesh_fbx));
    let collision_fbx_output = mesh_fbx.exists().then(|| path_string(&mesh_fbx));
    if export_fbx && mesh_fbx_output.is_none() {
        return Err("native FBX export completed without an output file".to_string());
    }
    let seed_gaussian_ply_string = path_string(&seed_gaussian_ply);
    let gaussian_ply_string = path_string(&final_gaussian_ply);
    let splat_string = path_string(&final_splat);
    let mesh_obj_string = path_string(&mesh_obj);
    let collider_obj_string = path_string(&collider_obj);
    let collision_json_string = path_string(&collision_json);
    let preview_json_string = path_string(&preview_json);
    let manifest_data = AssetManifest {
        schema_version: "agriscan-rgbd-assets-v1",
        source_session: &source_session,
        point_count: final_points.len(),
        face_count: mesh.faces.len(),
        seed_gaussian_ply: &seed_gaussian_ply_string,
        gaussian_ply: &gaussian_ply_string,
        splat: &splat_string,
        mesh_obj: &mesh_obj_string,
        mesh_fbx: mesh_fbx_output.as_deref(),
        collider_obj: &collider_obj_string,
        collision_json: &collision_json_string,
        collision_fbx: collision_fbx_output.as_deref(),
        preview_json: &preview_json_string,
        fbx_status: &fbx_status,
        mlx_status: &mlx_status,
        collision_status: &collision_status,
        options: AssetOptionsSummary {
            max_points,
            frame_stride,
            depth_decimation,
            gaussian_radius_m,
            turntable_degrees,
            use_mlx,
            mlx_iterations,
            mlx_voxel_size_m,
            mlx_train_size,
            mlx_max_train_views,
            collider_max_faces,
        },
    };
    write_json(&manifest, &manifest_data)?;

    Ok(AssetBuildResult {
        root: path_string(&asset_root),
        seed_gaussian_ply: seed_gaussian_ply_string,
        gaussian_ply: gaussian_ply_string,
        splat: splat_string,
        mesh_obj: mesh_obj_string,
        mesh_fbx: mesh_fbx_output,
        collider_obj: collider_obj_string,
        collision_json: collision_json_string,
        collision_fbx: collision_fbx_output,
        preview_json: preview_json_string,
        manifest: path_string(&manifest),
        point_count: final_points.len(),
        face_count: mesh.faces.len(),
        fbx_status,
        mlx_status,
        collision_status,
        tools,
        preview,
    })
}

fn load_frame_metadata(session_root: &Path) -> Result<Vec<FrameMetadata>, String> {
    let metadata_dir = session_root.join("metadata");
    let mut entries: Vec<_> = fs::read_dir(&metadata_dir)
        .map_err(|error| format!("failed to read metadata directory: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect();

    entries.sort();
    let mut frames = Vec::new();
    for path in entries {
        let data = fs::read(&path).map_err(|error| format!("failed to read {path:?}: {error}"))?;
        let frame: FrameMetadata = serde_json::from_slice(&data)
            .map_err(|error| format!("failed to parse {path:?}: {error}"))?;
        frames.push(frame);
    }

    frames.sort_by(|a, b| {
        a.frame_index
            .cmp(&b.frame_index)
            .then_with(|| a.frame_number.cmp(&b.frame_number))
            .then_with(|| {
                a.timestamp_ms
                    .partial_cmp(&b.timestamp_ms)
                    .unwrap_or(Ordering::Equal)
            })
    });
    Ok(frames)
}

fn build_mesh(
    frames: &[FrameMetadata],
    depth_decimation: u32,
    max_points: usize,
    gaussian_radius_m: f32,
    turntable_degrees: f32,
) -> Result<MeshBuild, String> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut frame_count = frames.len().max(1);
    let frame_point_limit = max_points.div_ceil(frame_count);
    if frame_count == 1 {
        frame_count = 2;
    }

    for (frame_idx, frame) in frames.iter().enumerate() {
        if vertices.len() >= max_points {
            break;
        }

        let depth = read_depth_png(&frame.files.depth)?;
        let color = match &frame.files.rgb {
            Some(path) => read_rgb_png(path).ok(),
            None => None,
        };

        let angle = if turntable_degrees.abs() < f32::EPSILON {
            0.0
        } else {
            let t = frame_idx as f32 / (frame_count - 1) as f32;
            t * turntable_degrees.to_radians()
        };

        add_frame_mesh(
            frame.intrinsics,
            frame.depth_units_m,
            depth.width,
            depth.height,
            &depth.z16,
            color.as_ref(),
            RigidTransform::rotation_y(angle),
            depth_decimation as usize,
            max_points,
            frame_point_limit,
            gaussian_radius_m,
            0.02,
            8.0,
            &mut vertices,
            &mut faces,
        );
    }

    Ok(MeshBuild { vertices, faces })
}

fn build_mesh_from_mcap(
    recording_path: &Path,
    frame_stride: u32,
    depth_decimation: u32,
    max_points: usize,
    gaussian_radius_m: f32,
    turntable_degrees: f32,
    max_train_views: u32,
    odometry_dir: &Path,
) -> Result<(MeshBuild, Vec<PositionedRgbdFrame>), String> {
    let total_frames = mcap_io::frame_count(recording_path)?;
    let (min_depth_m, max_depth_m) = mcap_io::capture_depth_range(recording_path)?;
    if total_frames == 0 {
        return Err("MCAP contains no RGB-D frames".to_string());
    }
    let stride = frame_stride.max(1) as usize;
    let selected_count = total_frames.div_ceil(stride);
    let mesh_indices = sampled_frame_indices(total_frames, stride, selected_count.min(128));
    let training_indices = sampled_frame_indices(
        total_frames,
        stride,
        selected_count.min(max_train_views.max(1) as usize),
    );
    let camera_poses = if turntable_degrees.abs() < f32::EPSILON && total_frames >= 8 {
        Some(estimate_mcap_camera_poses(
            recording_path,
            odometry_dir,
            min_depth_m,
            max_depth_m,
        )?)
    } else {
        None
    };
    // Camera tracking must run on consecutive frames. Sampling first and trying to
    // register frames that are many video frames apart makes handheld scans collapse
    // into a noisy cluster even though the original RGB-D sequence is valid.
    let requested_indices: BTreeSet<_> =
        if camera_poses.is_none() && turntable_degrees.abs() < f32::EPSILON {
            (1..=total_frames as u32).collect()
        } else {
            mesh_indices.union(&training_indices).copied().collect()
        };
    let frame_point_limit = max_points.div_ceil(mesh_indices.len().max(1));
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut training_frames = Vec::new();
    let mut previous_cloud = None::<Vec<[f32; 3]>>;
    let mut previous_pose = RigidTransform::identity();

    mcap_io::visit_frame_indices(recording_path, &requested_indices, |frame| {
        let selected_ordinal = frame.info.frame_index.saturating_sub(1) as usize / stride;
        let angle = if turntable_degrees.abs() < f32::EPSILON || selected_count <= 1 {
            0.0
        } else {
            let t = selected_ordinal as f32 / (selected_count - 1) as f32;
            t * turntable_degrees.to_radians()
        };
        let local_cloud = camera_poses
            .is_none()
            .then(|| build_alignment_cloud(&frame, min_depth_m, max_depth_m));
        let camera_to_world = if let Some(pose) = camera_poses
            .as_ref()
            .and_then(|poses| poses.get(&frame.info.frame_index))
        {
            *pose
        } else if turntable_degrees.abs() >= f32::EPSILON {
            RigidTransform::rotation_y(angle)
        } else if let (Some(source_cloud), Some(target_cloud)) =
            (local_cloud.as_deref(), previous_cloud.as_ref())
        {
            align_depth_cloud(source_cloud, target_cloud, previous_pose)
        } else {
            RigidTransform::identity()
        };
        if let Some(local_cloud) = local_cloud {
            previous_cloud = Some(
                local_cloud
                    .iter()
                    .copied()
                    .map(|point| camera_to_world.apply(point))
                    .collect(),
            );
            previous_pose = camera_to_world;
        }

        if mesh_indices.contains(&frame.info.frame_index) && vertices.len() < max_points {
            add_frame_mesh(
                frame.info.intrinsics,
                frame.info.depth_units_m,
                frame.depth.width,
                frame.depth.height,
                &frame.depth.z16,
                frame.color.as_ref(),
                camera_to_world,
                depth_decimation as usize,
                max_points,
                frame_point_limit,
                gaussian_radius_m,
                min_depth_m,
                max_depth_m,
                &mut vertices,
                &mut faces,
            );
        }
        if training_indices.contains(&frame.info.frame_index) {
            training_frames.push(PositionedRgbdFrame {
                frame,
                camera_to_world,
                min_depth_m,
                max_depth_m,
            });
        }
        Ok(true)
    })?;

    Ok((MeshBuild { vertices, faces }, training_frames))
}

fn sampled_frame_indices(total_frames: usize, stride: usize, sample_count: usize) -> BTreeSet<u32> {
    let selected_count = total_frames.div_ceil(stride);
    let samples = sample_count.max(1).min(selected_count);
    (0..samples)
        .map(|sample| {
            let ordinal = if samples <= 1 {
                (selected_count - 1) / 2
            } else {
                sample * (selected_count - 1) / (samples - 1)
            };
            (ordinal * stride + 1).min(total_frames) as u32
        })
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OdometryOutput {
    frames: Vec<OdometryPose>,
    succeeded: usize,
    failed: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OdometryPose {
    frame_index: u32,
    camera_to_world: RigidTransform,
}

fn estimate_mcap_camera_poses(
    recording_path: &Path,
    odometry_dir: &Path,
    min_depth_m: f32,
    max_depth_m: f32,
) -> Result<BTreeMap<u32, RigidTransform>, String> {
    let python = ensure_open3d_odometry_python()?;
    let output_path = odometry_dir.join("camera_poses.json");
    if let Ok(data) = fs::read(&output_path)
        && let Ok(trajectory) = serde_json::from_slice::<OdometryOutput>(&data)
        && trajectory.frames.len() == mcap_io::frame_count(recording_path)?
    {
        return Ok(trajectory
            .frames
            .into_iter()
            .map(|frame| (frame.frame_index, frame.camera_to_world))
            .collect());
    }
    let cache_root = odometry_dir.join("cache");
    let metadata_dir = cache_root.join("metadata");
    let rgb_dir = cache_root.join("rgb");
    let depth_dir = cache_root.join("depth");
    if cache_root.is_dir() {
        fs::remove_dir_all(&cache_root)
            .map_err(|error| format!("failed to clear RGB-D odometry cache: {error}"))?;
    }
    for directory in [&metadata_dir, &rgb_dir, &depth_dir] {
        fs::create_dir_all(directory)
            .map_err(|error| format!("failed to create RGB-D odometry cache: {error}"))?;
    }

    let total_frames = mcap_io::frame_count(recording_path)?;
    let indices: BTreeSet<u32> = (1..=total_frames as u32).collect();
    mcap_io::visit_frame_indices(recording_path, &indices, |frame| {
        let Some(color) = frame.color.as_ref() else {
            return Err(format!(
                "MCAP frame {} has no RGB image required for RGB-D odometry",
                frame.info.frame_index
            ));
        };
        let target_width = frame.depth.width.min(320).max(16);
        let target_height = ((frame.depth.height as f64 * target_width as f64
            / frame.depth.width.max(1) as f64)
            .round() as u32)
            .max(16);
        let rgb_path = rgb_dir.join(format!("frame_{:06}.jpg", frame.info.frame_index));
        let depth_path = depth_dir.join(format!("frame_{:06}_z16.png", frame.info.frame_index));
        let rgb = resize_rgb_nearest(color, target_width, target_height);
        let depth = resize_depth_nearest(
            &frame.depth.z16,
            frame.depth.width,
            frame.depth.height,
            target_width,
            target_height,
        );
        write_rgb_jpeg(&rgb_path, target_width, target_height, &rgb)?;
        write_depth_png(&depth_path, target_width, target_height, &depth)?;

        let scale_x = target_width as f32 / frame.depth.width.max(1) as f32;
        let scale_y = target_height as f32 / frame.depth.height.max(1) as f32;
        let intrinsics = Intrinsics {
            width: target_width,
            height: target_height,
            ppx: frame.info.intrinsics.ppx * scale_x,
            ppy: frame.info.intrinsics.ppy * scale_y,
            fx: frame.info.intrinsics.fx * scale_x,
            fy: frame.info.intrinsics.fy * scale_y,
            coeffs: frame.info.intrinsics.coeffs,
        };
        let metadata = serde_json::json!({
            "frameIndex": frame.info.frame_index,
            "depthUnitsM": frame.info.depth_units_m,
            "minDepthM": min_depth_m,
            "maxDepthM": max_depth_m,
            "intrinsics": intrinsics,
            "files": {
                "rgb": path_string(&rgb_path),
                "depth": path_string(&depth_path)
            }
        });
        write_json(
            &metadata_dir.join(format!("frame_{:06}.json", frame.info.frame_index)),
            &metadata,
        )?;
        Ok(true)
    })?;

    let script_path = odometry_dir.join("open3d_rgbd_odometry.py");
    fs::write(&script_path, OPEN3D_ODOMETRY_SCRIPT)
        .map_err(|error| format!("failed to write Open3D odometry script: {error}"))?;
    let output = Command::new(&python)
        .arg(&script_path)
        .arg("--cache-root")
        .arg(&cache_root)
        .arg("--output-json")
        .arg(&output_path)
        .output()
        .map_err(|error| format!("failed to start Open3D RGB-D odometry: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "Open3D RGB-D odometry failed: {}{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    let data = fs::read(&output_path)
        .map_err(|error| format!("failed to read RGB-D camera trajectory: {error}"))?;
    let trajectory: OdometryOutput = serde_json::from_slice(&data)
        .map_err(|error| format!("failed to parse RGB-D camera trajectory: {error}"))?;
    let attempted = trajectory.succeeded + trajectory.failed;
    if attempted > 0 && trajectory.succeeded * 2 < attempted {
        return Err(format!(
            "RGB-D camera tracking rejected too many frames: {} succeeded / {} failed",
            trajectory.succeeded, trajectory.failed
        ));
    }
    let poses: BTreeMap<_, _> = trajectory
        .frames
        .into_iter()
        .map(|frame| (frame.frame_index, frame.camera_to_world))
        .collect();
    if poses.len() != total_frames {
        return Err(format!(
            "RGB-D camera trajectory contains {} poses for {total_frames} frames",
            poses.len()
        ));
    }
    let _ = fs::remove_dir_all(cache_root);
    Ok(poses)
}

fn resize_rgb_nearest(color: &ColorFrame, width: u32, height: u32) -> Vec<u8> {
    let mut resized = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        let source_y =
            (y * color.height as usize / height.max(1) as usize).min(color.height as usize - 1);
        for x in 0..width as usize {
            let source_x =
                (x * color.width as usize / width.max(1) as usize).min(color.width as usize - 1);
            let source = (source_y * color.width as usize + source_x) * 3;
            let target = (y * width as usize + x) * 3;
            resized[target..target + 3].copy_from_slice(&color.rgb[source..source + 3]);
        }
    }
    resized
}

fn resize_depth_nearest(
    depth: &[u16],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
) -> Vec<u16> {
    let mut resized = vec![0u16; (width * height) as usize];
    for y in 0..height as usize {
        let source_y =
            (y * source_height as usize / height.max(1) as usize).min(source_height as usize - 1);
        for x in 0..width as usize {
            let source_x =
                (x * source_width as usize / width.max(1) as usize).min(source_width as usize - 1);
            resized[y * width as usize + x] = depth[source_y * source_width as usize + source_x];
        }
    }
    resized
}

fn write_rgb_jpeg(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<(), String> {
    let file =
        File::create(path).map_err(|error| format!("failed to create odometry JPEG: {error}"))?;
    JpegEncoder::new(file, 88)
        .encode(rgb, width as u16, height as u16, JpegColorType::Rgb)
        .map_err(|error| format!("failed to write odometry JPEG: {error}"))
}

fn ensure_open3d_odometry_python() -> Result<PathBuf, String> {
    let venv_dir = dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("AgriScan Studio")
        .join("rgbd-reconstruction-venv");
    let venv_python = venv_dir.join("bin").join("python");
    if python_imports_open3d(&venv_python) {
        return Ok(venv_python);
    }

    let mut system_python = [
        "/opt/homebrew/bin/python3.12",
        "/usr/local/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
        "/usr/local/bin/python3.10",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file());
    if system_python.is_none()
        && let Some(brew) = find_in_path("brew")
    {
        let status = Command::new(brew)
            .args(["install", "python@3.12"])
            .status()
            .map_err(|error| {
                format!("failed to install Python 3.12 for RGB-D odometry: {error}")
            })?;
        if status.success() {
            system_python = [
                PathBuf::from("/opt/homebrew/bin/python3.12"),
                PathBuf::from("/usr/local/bin/python3.12"),
            ]
            .into_iter()
            .find(|path| path.is_file());
        }
    }
    let system_python = system_python.ok_or_else(|| {
        "Python 3.10-3.12 is required for the Open3D RGB-D reconstruction backend".to_string()
    })?;
    if !venv_python.is_file() {
        fs::create_dir_all(
            venv_dir
                .parent()
                .ok_or_else(|| "RGB-D venv has no parent folder".to_string())?,
        )
        .map_err(|error| format!("failed to create RGB-D venv parent: {error}"))?;
        let status = Command::new(&system_python)
            .arg("-m")
            .arg("venv")
            .arg(&venv_dir)
            .status()
            .map_err(|error| format!("failed to create RGB-D reconstruction venv: {error}"))?;
        if !status.success() {
            return Err("failed to create RGB-D reconstruction venv".to_string());
        }
    }
    let output = Command::new(&venv_python)
        .args([
            "-m",
            "pip",
            "install",
            "--upgrade",
            "open3d==0.19.0",
            "numpy",
            "pillow",
        ])
        .output()
        .map_err(|error| format!("failed to install Open3D RGB-D backend: {error}"))?;
    if !output.status.success() || !python_imports_open3d(&venv_python) {
        return Err(format!(
            "Open3D RGB-D backend setup failed: {}{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    Ok(venv_python)
}

fn python_imports_open3d(python: &Path) -> bool {
    python.is_file()
        && Command::new(python)
            .args(["-c", "import open3d; print(open3d.__version__)"])
            .output()
            .is_ok_and(|output| output.status.success())
}

fn build_alignment_cloud(
    frame: &DecodedRgbdFrame,
    min_depth_m: f32,
    max_depth_m: f32,
) -> Vec<[f32; 3]> {
    let width = frame.depth.width as usize;
    let height = frame.depth.height as usize;
    let target_points = 4_000usize;
    let step = ((width * height).div_ceil(target_points) as f64)
        .sqrt()
        .ceil()
        .max(2.0) as usize;
    let intr = frame.info.intrinsics;
    let mut points = Vec::with_capacity(target_points);
    for y in (0..height).step_by(step) {
        for x in (0..width).step_by(step) {
            let raw = frame.depth.z16[y * width + x];
            if raw == 0 {
                continue;
            }
            let z = raw as f32 * frame.info.depth_units_m;
            if !(min_depth_m..=max_depth_m).contains(&z) {
                continue;
            }
            points.push([
                (x as f32 - intr.ppx) / intr.fx * z,
                -((y as f32 - intr.ppy) / intr.fy * z),
                -z,
            ]);
        }
    }
    points
}

fn align_depth_cloud(
    source: &[[f32; 3]],
    target_world: &[[f32; 3]],
    initial_pose: RigidTransform,
) -> RigidTransform {
    if source.len() < 80 || target_world.len() < 80 {
        return initial_pose;
    }

    const VOXEL_SIZE: f32 = 0.055;
    let index = build_voxel_index(target_world, VOXEL_SIZE);
    let mut pose = initial_pose;
    for iteration in 0..10 {
        let max_distance = match iteration {
            0..=2 => 0.11,
            3..=5 => 0.075,
            6..=7 => 0.05,
            _ => 0.035,
        };
        let search_radius = (max_distance / VOXEL_SIZE).ceil() as i32;
        let max_distance_squared = max_distance * max_distance;
        let mut pairs = Vec::with_capacity(source.len());
        for point in source {
            let transformed = pose.apply(*point);
            if let Some((nearest, distance_squared)) = nearest_voxel_point(
                transformed,
                &index,
                VOXEL_SIZE,
                search_radius,
                max_distance_squared,
            ) {
                pairs.push((transformed, nearest, distance_squared));
            }
        }
        if pairs.len() < 80 {
            break;
        }

        pairs.sort_unstable_by(|left, right| {
            left.2.partial_cmp(&right.2).unwrap_or(Ordering::Equal)
        });
        pairs.truncate((pairs.len() * 4 / 5).max(80));
        let Some(delta) = best_fit_transform(&pairs) else {
            break;
        };
        let translation_step = length3(delta.translation);
        let rotation_step = rotation_angle(delta.rotation);
        if translation_step > 0.25 || rotation_step > 35.0_f32.to_radians() {
            break;
        }
        pose = pose.then(delta);
        if translation_step < 0.000_15 && rotation_step < 0.000_5 {
            break;
        }
    }
    pose
}

type VoxelKey = (i32, i32, i32);

fn build_voxel_index(points: &[[f32; 3]], voxel_size: f32) -> HashMap<VoxelKey, Vec<[f32; 3]>> {
    let mut index = HashMap::<VoxelKey, Vec<[f32; 3]>>::new();
    for point in points {
        index
            .entry(voxel_key(*point, voxel_size))
            .or_default()
            .push(*point);
    }
    index
}

fn nearest_voxel_point(
    point: [f32; 3],
    index: &HashMap<VoxelKey, Vec<[f32; 3]>>,
    voxel_size: f32,
    search_radius: i32,
    max_distance_squared: f32,
) -> Option<([f32; 3], f32)> {
    let (cx, cy, cz) = voxel_key(point, voxel_size);
    let mut best = None;
    let mut best_distance_squared = max_distance_squared;
    for dz in -search_radius..=search_radius {
        for dy in -search_radius..=search_radius {
            for dx in -search_radius..=search_radius {
                let Some(candidates) = index.get(&(cx + dx, cy + dy, cz + dz)) else {
                    continue;
                };
                for candidate in candidates {
                    let distance_squared = distance_squared3(point, *candidate);
                    if distance_squared < best_distance_squared {
                        best_distance_squared = distance_squared;
                        best = Some((*candidate, distance_squared));
                    }
                }
            }
        }
    }
    best
}

fn voxel_key(point: [f32; 3], voxel_size: f32) -> VoxelKey {
    (
        (point[0] / voxel_size).floor() as i32,
        (point[1] / voxel_size).floor() as i32,
        (point[2] / voxel_size).floor() as i32,
    )
}

fn best_fit_transform(pairs: &[([f32; 3], [f32; 3], f32)]) -> Option<RigidTransform> {
    if pairs.len() < 3 {
        return None;
    }
    let count = pairs.len() as f32;
    let mut source_center = [0.0; 3];
    let mut target_center = [0.0; 3];
    for (source, target, _) in pairs {
        source_center = add3(source_center, *source);
        target_center = add3(target_center, *target);
    }
    source_center = scale3(source_center, count.recip());
    target_center = scale3(target_center, count.recip());

    let mut covariance = [[0.0_f32; 3]; 3];
    for (source, target, _) in pairs {
        let source = sub3(*source, source_center);
        let target = sub3(*target, target_center);
        for row in 0..3 {
            for column in 0..3 {
                covariance[row][column] += source[row] * target[column];
            }
        }
    }
    let s = covariance;
    let trace = s[0][0] + s[1][1] + s[2][2];
    let horn = [
        [
            trace,
            s[1][2] - s[2][1],
            s[2][0] - s[0][2],
            s[0][1] - s[1][0],
        ],
        [
            s[1][2] - s[2][1],
            s[0][0] - s[1][1] - s[2][2],
            s[0][1] + s[1][0],
            s[2][0] + s[0][2],
        ],
        [
            s[2][0] - s[0][2],
            s[0][1] + s[1][0],
            -s[0][0] + s[1][1] - s[2][2],
            s[1][2] + s[2][1],
        ],
        [
            s[0][1] - s[1][0],
            s[2][0] + s[0][2],
            s[1][2] + s[2][1],
            -s[0][0] - s[1][1] + s[2][2],
        ],
    ];
    let quaternion = largest_eigenvector_symmetric(horn);
    let rotation = quaternion_to_matrix(quaternion);
    Some(RigidTransform {
        rotation,
        translation: sub3(target_center, mat3_vec(rotation, source_center)),
    })
}

fn largest_eigenvector_symmetric(mut matrix: [[f32; 4]; 4]) -> [f32; 4] {
    let mut vectors = [[0.0_f32; 4]; 4];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..32 {
        let mut pivot = (0usize, 1usize);
        let mut largest = matrix[0][1].abs();
        for row in 0..4 {
            for column in row + 1..4 {
                if matrix[row][column].abs() > largest {
                    largest = matrix[row][column].abs();
                    pivot = (row, column);
                }
            }
        }
        if largest < 1.0e-7 {
            break;
        }
        let (p, q) = pivot;
        let theta = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[p][p] - matrix[q][q]);
        let (sin_theta, cos_theta) = theta.sin_cos();
        let mut jacobi = [[0.0_f32; 4]; 4];
        for (index, row) in jacobi.iter_mut().enumerate() {
            row[index] = 1.0;
        }
        jacobi[p][p] = cos_theta;
        jacobi[q][q] = cos_theta;
        jacobi[p][q] = -sin_theta;
        jacobi[q][p] = sin_theta;
        matrix = multiply4(transpose4(jacobi), multiply4(matrix, jacobi));
        vectors = multiply4(vectors, jacobi);
    }
    let eigen_index = (0..4)
        .max_by(|left, right| {
            matrix[*left][*left]
                .partial_cmp(&matrix[*right][*right])
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);
    let mut result = [
        vectors[0][eigen_index],
        vectors[1][eigen_index],
        vectors[2][eigen_index],
        vectors[3][eigen_index],
    ];
    let norm = result.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1.0e-8 {
        for value in &mut result {
            *value /= norm;
        }
    } else {
        result = [1.0, 0.0, 0.0, 0.0];
    }
    result
}

fn quaternion_to_matrix([w, x, y, z]: [f32; 4]) -> [[f32; 3]; 3] {
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn multiply4(left: [[f32; 4]; 4], right: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0_f32; 4]; 4];
    for (row, values) in result.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = (0..4)
                .map(|index| left[row][index] * right[index][column])
                .sum();
        }
    }
    result
}

fn transpose4(matrix: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for (row, values) in result.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = matrix[column][row];
        }
    }
    result
}

fn mat3_vec(matrix: [[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    [
        dot3(matrix[0], vector),
        dot3(matrix[1], vector),
        dot3(matrix[2], vector),
    ]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale3(vector: [f32; 3], scale: f32) -> [f32; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn length3(vector: [f32; 3]) -> f32 {
    dot3(vector, vector).sqrt()
}

fn distance_squared3(left: [f32; 3], right: [f32; 3]) -> f32 {
    dot3(sub3(left, right), sub3(left, right))
}

fn rotation_angle(rotation: [[f32; 3]; 3]) -> f32 {
    let cosine = ((rotation[0][0] + rotation[1][1] + rotation[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    cosine.acos()
}

#[allow(clippy::too_many_arguments)]
fn add_frame_mesh(
    intr: Intrinsics,
    depth_units_m: f32,
    depth_width: u32,
    depth_height: u32,
    depth_z16: &[u16],
    color: Option<&ColorFrame>,
    camera_to_world: RigidTransform,
    step: usize,
    max_points: usize,
    frame_point_limit: usize,
    gaussian_radius_m: f32,
    min_depth_m: f32,
    max_depth_m: f32,
    vertices: &mut Vec<SplatPoint>,
    faces: &mut Vec<[u32; 3]>,
) {
    let width = depth_width as usize;
    let height = depth_height as usize;
    let requested_step = step.max(1);
    let point_budget = frame_point_limit
        .max(1)
        .min(max_points.saturating_sub(vertices.len()));
    let budget_step = ((width * height).div_ceil(point_budget) as f64)
        .sqrt()
        .ceil() as usize;
    let effective_step = requested_step.max(budget_step.max(1));
    let grid_w = width.div_ceil(effective_step);
    let grid_h = height.div_ceil(effective_step);
    let mut index_grid = vec![None::<u32>; grid_w * grid_h];
    let depth_jump = gaussian_radius_m.max(0.006) * 10.0;
    let frame_vertex_start = vertices.len();

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            if vertices.len() >= max_points
                || vertices.len().saturating_sub(frame_vertex_start) >= point_budget
            {
                break;
            }
            let x = (gx * effective_step).min(width - 1);
            let y = (gy * effective_step).min(height - 1);
            let raw = depth_z16[y * width + x];
            if raw == 0 {
                continue;
            }

            let z = raw as f32 * depth_units_m;
            if !(min_depth_m..=max_depth_m).contains(&z) {
                continue;
            }

            let px = (x as f32 - intr.ppx) / intr.fx * z;
            let py = -((y as f32 - intr.ppy) / intr.fy * z);
            let pz = -z;
            let [world_x, world_y, world_z] = camera_to_world.apply([px, py, pz]);
            let (r, g, b) = sample_rgb(color, x, y, width, height);

            let vertex_index = vertices.len() as u32;
            vertices.push(SplatPoint {
                x: world_x,
                y: world_y,
                z: world_z,
                r,
                g,
                b,
                radius: gaussian_radius_m,
                scale: [gaussian_radius_m; 3],
                rotation: [1.0, 0.0, 0.0, 0.0],
                opacity_logit: 1.734_601_f32,
            });
            index_grid[gy * grid_w + gx] = Some(vertex_index);
        }
    }

    for gy in 0..grid_h.saturating_sub(1) {
        for gx in 0..grid_w.saturating_sub(1) {
            let a = index_grid[gy * grid_w + gx];
            let b = index_grid[gy * grid_w + gx + 1];
            let c = index_grid[(gy + 1) * grid_w + gx];
            let d = index_grid[(gy + 1) * grid_w + gx + 1];
            if let (Some(a), Some(b), Some(c)) = (a, b, c)
                && face_is_local(vertices, [a, b, c], depth_jump)
            {
                faces.push([a + 1, b + 1, c + 1]);
            }
            if let (Some(b), Some(d), Some(c)) = (b, d, c)
                && face_is_local(vertices, [b, d, c], depth_jump)
            {
                faces.push([b + 1, d + 1, c + 1]);
            }
        }
    }
}

fn face_is_local(vertices: &[SplatPoint], face: [u32; 3], max_distance: f32) -> bool {
    let a = &vertices[face[0] as usize];
    let b = &vertices[face[1] as usize];
    let c = &vertices[face[2] as usize];
    distance(a, b) < max_distance && distance(b, c) < max_distance && distance(c, a) < max_distance
}

fn distance(a: &SplatPoint, b: &SplatPoint) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

fn write_gaussian_ply(path: &Path, points: &[SplatPoint]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("failed to create GS PLY: {error}"))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "ply").map_err(io_error)?;
    writeln!(writer, "format ascii 1.0").map_err(io_error)?;
    writeln!(
        writer,
        "comment AgriScan Studio 3DGS seed generated from RealSense RGB-D"
    )
    .map_err(io_error)?;
    writeln!(writer, "element vertex {}", points.len()).map_err(io_error)?;
    for property in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        writeln!(writer, "property float {property}").map_err(io_error)?;
    }
    writeln!(writer, "end_header").map_err(io_error)?;

    for point in points {
        let f_dc_0 = (point.r as f32 / 255.0 - 0.5) / SH_C0;
        let f_dc_1 = (point.g as f32 / 255.0 - 0.5) / SH_C0;
        let f_dc_2 = (point.b as f32 / 255.0 - 0.5) / SH_C0;
        let scale_0 = point.scale[0].max(0.0001).ln();
        let scale_1 = point.scale[1].max(0.0001).ln();
        let scale_2 = point.scale[2].max(0.0001).ln();
        writeln!(
            writer,
            "{:.6} {:.6} {:.6} 0 0 0 {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            point.x,
            point.y,
            point.z,
            f_dc_0,
            f_dc_1,
            f_dc_2,
            point.opacity_logit,
            scale_0,
            scale_1,
            scale_2,
            point.rotation[0],
            point.rotation[1],
            point.rotation[2],
            point.rotation[3],
        )
        .map_err(io_error)?;
    }

    writer
        .flush()
        .map_err(|error| format!("failed to flush GS PLY: {error}"))
}

fn write_splat(path: &Path, points: &[SplatPoint]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("failed to create .splat: {error}"))?;
    let mut writer = BufWriter::new(file);
    for point in points {
        for value in [
            point.x,
            point.y,
            point.z,
            point.scale[0],
            point.scale[1],
            point.scale[2],
        ] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| format!("failed to write .splat: {error}"))?;
        }
        let quat = encode_splat_quaternion(point.rotation);
        writer
            .write_all(&[
                point.r, point.g, point.b, 220, quat[0], quat[1], quat[2], quat[3],
            ])
            .map_err(|error| format!("failed to write .splat: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("failed to flush .splat: {error}"))
}

fn encode_splat_quaternion(rotation: [f32; 4]) -> [u8; 4] {
    let length = (rotation[0] * rotation[0]
        + rotation[1] * rotation[1]
        + rotation[2] * rotation[2]
        + rotation[3] * rotation[3])
        .sqrt()
        .max(0.0001);
    [
        encode_quat_byte(rotation[0] / length),
        encode_quat_byte(rotation[1] / length),
        encode_quat_byte(rotation[2] / length),
        encode_quat_byte(rotation[3] / length),
    ]
}

fn encode_quat_byte(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) * 128.0) + 128.0)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn write_obj(path: &Path, mesh: &MeshBuild) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("failed to create OBJ: {error}"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# AgriScan Studio surface mesh").map_err(io_error)?;
    writeln!(writer, "# Extended vertex colors: v x y z r g b").map_err(io_error)?;
    for vertex in &mesh.vertices {
        writeln!(
            writer,
            "v {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            vertex.x,
            vertex.y,
            vertex.z,
            vertex.r as f32 / 255.0,
            vertex.g as f32 / 255.0,
            vertex.b as f32 / 255.0
        )
        .map_err(io_error)?;
    }
    for face in &mesh.faces {
        writeln!(writer, "f {} {} {}", face[0], face[1], face[2]).map_err(io_error)?;
    }
    writer
        .flush()
        .map_err(|error| format!("failed to flush OBJ: {error}"))
}

fn build_collision_mesh(mesh: &MeshBuild, max_faces: usize) -> MeshBuild {
    if mesh.faces.len() <= max_faces {
        return mesh.clone();
    }

    let stride = (mesh.faces.len() as f32 / max_faces as f32).ceil() as usize;
    let selected_faces: Vec<[u32; 3]> = mesh.faces.iter().step_by(stride).copied().collect();
    compact_mesh(mesh, &selected_faces)
}

fn compact_mesh(mesh: &MeshBuild, faces: &[[u32; 3]]) -> MeshBuild {
    let mut remap = vec![None::<u32>; mesh.vertices.len()];
    let mut vertices = Vec::new();
    let mut compact_faces = Vec::with_capacity(faces.len());

    for face in faces {
        let mut compact_face = [0u32; 3];
        let mut valid = true;
        for (slot, index) in face.iter().enumerate() {
            let Some(zero_based) = index.checked_sub(1).map(|value| value as usize) else {
                valid = false;
                break;
            };
            if zero_based >= mesh.vertices.len() {
                valid = false;
                break;
            }
            let mapped = match remap[zero_based] {
                Some(mapped) => mapped,
                None => {
                    let mapped = vertices.len() as u32 + 1;
                    vertices.push(mesh.vertices[zero_based].clone());
                    remap[zero_based] = Some(mapped);
                    mapped
                }
            };
            compact_face[slot] = mapped;
        }
        if valid {
            compact_faces.push(compact_face);
        }
    }

    MeshBuild {
        vertices,
        faces: compact_faces,
    }
}

fn write_collision_manifest(
    path: &Path,
    collider_mesh: &MeshBuild,
    source_mesh: &Path,
    collider_obj: &Path,
    max_faces: usize,
) -> Result<String, String> {
    let collider_bounds = bounds(&collider_mesh.vertices);
    let sphere = bounding_sphere(&collider_mesh.vertices, collider_bounds.center);
    let manifest = CollisionManifest {
        schema_version: "agriscan-rgbd-collision-v1",
        collider_type: "triangle_mesh",
        collider_obj: path_string(collider_obj),
        source_mesh: path_string(source_mesh),
        point_count: collider_mesh.vertices.len(),
        face_count: collider_mesh.faces.len(),
        bounds: collider_bounds,
        bounding_sphere: sphere,
        notes: format!(
            "Low-poly triangle mesh collider capped at {max_faces} faces; FBX object name uses UCX_scan_surface_00 for engine import."
        ),
    };
    write_json(path, &manifest)?;
    Ok(format!(
        "Collision collider ready: {} verts / {} faces",
        manifest.point_count, manifest.face_count
    ))
}

fn bounding_sphere(points: &[SplatPoint], center: [f32; 3]) -> BoundingSphere {
    let mut radius = 0.0_f32;
    for point in points {
        let d = ((point.x - center[0]).powi(2)
            + (point.y - center[1]).powi(2)
            + (point.z - center[2]).powi(2))
        .sqrt();
        radius = radius.max(d);
    }
    BoundingSphere { center, radius }
}

#[derive(Default)]
struct FusedSplatCell {
    xyz: [f64; 3],
    rgb: [u64; 3],
    count: u64,
}

fn sample_splat_points(points: &[SplatPoint], max_points: usize) -> Vec<SplatPoint> {
    if max_points == 0 {
        return Vec::new();
    }
    if points.len() <= max_points {
        return points.to_vec();
    }
    (0..max_points)
        .map(|index| {
            let source = index * (points.len() - 1) / (max_points - 1).max(1);
            points[source].clone()
        })
        .collect()
}

fn fuse_splat_points(points: &[SplatPoint], voxel_size: f32) -> Vec<SplatPoint> {
    let mut cells = BTreeMap::<(i32, i32, i32), FusedSplatCell>::new();
    for point in points {
        let key = (
            (point.x / voxel_size).floor() as i32,
            (point.y / voxel_size).floor() as i32,
            (point.z / voxel_size).floor() as i32,
        );
        let cell = cells.entry(key).or_default();
        cell.xyz[0] += point.x as f64;
        cell.xyz[1] += point.y as f64;
        cell.xyz[2] += point.z as f64;
        cell.rgb[0] += point.r as u64;
        cell.rgb[1] += point.g as u64;
        cell.rgb[2] += point.b as u64;
        cell.count += 1;
    }
    cells
        .into_values()
        .filter(|cell| cell.count > 0)
        .map(|cell| {
            let count = cell.count as f64;
            let radius = (voxel_size * 0.9).max(0.0025);
            SplatPoint {
                x: (cell.xyz[0] / count) as f32,
                y: (cell.xyz[1] / count) as f32,
                z: (cell.xyz[2] / count) as f32,
                r: (cell.rgb[0] / cell.count).min(255) as u8,
                g: (cell.rgb[1] / cell.count).min(255) as u8,
                b: (cell.rgb[2] / cell.count).min(255) as u8,
                radius,
                scale: [radius; 3],
                rotation: [1.0, 0.0, 0.0, 0.0],
                opacity_logit: 2.197_224_6,
            }
        })
        .collect()
}

fn build_preview_payload(
    points: &[SplatPoint],
    initial_camera: Option<PreviewCamera>,
) -> PreviewPayload {
    let preview_points = downsample_preview(points, 220_000);
    let footprint_boost =
        ((points.len() as f32 / preview_points.len().max(1) as f32).sqrt()).clamp(1.0, 1.75);
    PreviewPayload {
        bounds: bounds(points),
        initial_camera,
        points: preview_points
            .into_iter()
            .map(|point| PreviewPoint {
                x: point.x,
                y: point.y,
                z: point.z,
                r: point.r,
                g: point.g,
                b: point.b,
                radius: point.radius * footprint_boost,
                scale: point.scale.map(|scale| scale * footprint_boost),
                rotation: point.rotation,
                opacity: 1.0 / (1.0 + (-point.opacity_logit).exp()),
            })
            .collect(),
    }
}

fn write_preview_json(path: &Path, payload: &PreviewPayload) -> Result<(), String> {
    write_json(path, payload)
}

fn downsample_preview(points: &[SplatPoint], limit: usize) -> Vec<SplatPoint> {
    if points.len() <= limit {
        return points.to_vec();
    }
    (0..limit)
        .map(|index| points[index * points.len() / limit].clone())
        .collect()
}

fn bounds(points: &[SplatPoint]) -> Bounds {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for point in points {
        min[0] = min[0].min(point.x);
        min[1] = min[1].min(point.y);
        min[2] = min[2].min(point.z);
        max[0] = max[0].max(point.x);
        max[1] = max[1].max(point.y);
        max[2] = max[2].max(point.z);
    }
    Bounds {
        min,
        max,
        center: [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ],
    }
}

fn write_mcap_training_cache(
    cache_root: &Path,
    frames: &[PositionedRgbdFrame],
) -> Result<(), String> {
    if frames.is_empty() {
        return Err("MCAP contains no RGB-D frames for MLX training".to_string());
    }
    if cache_root.is_dir() {
        fs::remove_dir_all(cache_root)
            .map_err(|error| format!("failed to clear MLX training cache: {error}"))?;
    }
    let rgb_dir = cache_root.join("rgb");
    let depth_dir = cache_root.join("depth_z16");
    let metadata_dir = cache_root.join("metadata");
    for directory in [&rgb_dir, &depth_dir, &metadata_dir] {
        fs::create_dir_all(directory)
            .map_err(|error| format!("failed to create MLX training cache: {error}"))?;
    }

    for (index, positioned) in frames.iter().enumerate() {
        let frame = &positioned.frame;
        let stem = format!("frame_{:06}", index + 1);
        let depth_path = depth_dir.join(format!("{stem}_depth_z16.png"));
        write_depth_png(
            &depth_path,
            frame.depth.width,
            frame.depth.height,
            &frame.depth.z16,
        )?;
        let rgb_path = if let Some(color) = &frame.color {
            let path = rgb_dir.join(format!("{stem}_rgb.png"));
            write_rgb_png(&path, color)?;
            Some(path)
        } else {
            None
        };
        let metadata_path = metadata_dir.join(format!("{stem}.json"));
        let metadata = serde_json::json!({
            "schemaVersion": "agriscan-rgbd-frame-v1",
            "sessionId": frame.info.session_id,
            "frameIndex": index + 1,
            "frameNumber": frame.info.frame_number,
            "timestampMs": frame.info.timestamp_ms,
            "intrinsics": frame.info.intrinsics,
            "depthUnitsM": frame.info.depth_units_m,
            "minDepthM": positioned.min_depth_m,
            "maxDepthM": positioned.max_depth_m,
            "cameraToWorld": positioned.camera_to_world,
            "files": {
                "rgb": rgb_path.as_ref().map(|path| path_string(path)),
                "depth": path_string(&depth_path)
            }
        });
        write_json(&metadata_path, &metadata)?;
    }
    Ok(())
}

fn write_rgb_png(path: &Path, color: &ColorFrame) -> Result<(), String> {
    let file =
        File::create(path).map_err(|error| format!("failed to create cached RGB PNG: {error}"))?;
    let mut encoder = PngEncoder::new(BufWriter::new(file), color.width, color.height);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("failed to initialize cached RGB PNG: {error}"))?;
    writer
        .write_image_data(&color.rgb)
        .map_err(|error| format!("failed to write cached RGB PNG: {error}"))
}

fn write_depth_png(path: &Path, width: u32, height: u32, z16: &[u16]) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("failed to create cached depth PNG: {error}"))?;
    let mut encoder = PngEncoder::new(BufWriter::new(file), width, height);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Sixteen);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("failed to initialize cached depth PNG: {error}"))?;
    let bytes: Vec<_> = z16.iter().flat_map(|value| value.to_be_bytes()).collect();
    writer
        .write_image_data(&bytes)
        .map_err(|error| format!("failed to write cached depth PNG: {error}"))
}

fn run_sharp_mlx_inference(
    session_root: &Path,
    output_ply: &Path,
    mlx_dir: &Path,
    max_points: usize,
    max_views: u32,
) -> Result<MlxRefinement, String> {
    let python =
        find_python().ok_or_else(|| "python3 not found; install Python and mlx".to_string())?;
    let (mlx_available, mlx_status) = probe_mlx(&python);
    if !mlx_available {
        return Err(mlx_status);
    }

    let checkpoint = ensure_sharp_checkpoint(&python)?;
    let script_path = mlx_dir.join("sharp_mlx_inference.py");
    let runtime_dir = mlx_dir.join("sharp_mlx");
    let summary_path = mlx_dir.join("sharp_mlx_summary.json");
    fs::create_dir_all(&runtime_dir)
        .map_err(|error| format!("failed to create SHARP MLX runtime directory: {error}"))?;
    fs::write(&script_path, SHARP_MLX_INFERENCE_SCRIPT)
        .map_err(|error| format!("failed to write SHARP MLX inference script: {error}"))?;
    for (filename, source) in SHARP_MLX_RUNTIME {
        fs::write(runtime_dir.join(filename), source)
            .map_err(|error| format!("failed to write SHARP MLX runtime {filename}: {error}"))?;
    }

    let output = Command::new(&python)
        .arg(&script_path)
        .arg("--output-ply")
        .arg(output_ply)
        .arg("--summary-json")
        .arg(&summary_path)
        .arg("--session-root")
        .arg(session_root)
        .arg("--checkpoint")
        .arg(&checkpoint)
        .arg("--max-points")
        .arg(max_points.to_string())
        .arg("--max-views")
        .arg(max_views.to_string())
        .output()
        .map_err(|error| format!("failed to run pretrained SHARP on MLX: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "SHARP MLX inference failed:\n{}",
            trim_process_output(&stdout, &stderr, 24)
        ));
    }
    if !output_ply.exists() {
        return Err("SHARP MLX inference finished without output PLY".to_string());
    }

    let points = read_gaussian_ply(output_ply)?;
    if points.is_empty() {
        return Err("SHARP MLX output PLY contained no gaussians".to_string());
    }

    let status = sharp_summary_status(&summary_path).unwrap_or_else(|| {
        format!(
            "Pretrained SHARP inferred {} gaussians ({mlx_status})",
            points.len()
        )
    });

    Ok(MlxRefinement { points, status })
}

fn sharp_summary_status(path: &Path) -> Option<String> {
    let data = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&data).ok()?;
    let output = value.get("outputPointCount")?.as_u64()?;
    let views = value.get("inferenceViews")?.as_u64()?;
    let backend = value
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("SHARP MLX");
    let device = value
        .get("device")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("MLX");
    let seconds = value
        .get("totalSeconds")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    Some(format!(
        "{backend} pretrained inference generated {output} gaussians from {views} RGB keyframes on {device} in {seconds:.1}s; RealSense depth scale + RGB-D poses applied"
    ))
}

fn read_gaussian_ply(path: &Path) -> Result<Vec<SplatPoint>, String> {
    let data = fs::read_to_string(path)
        .map_err(|error| format!("failed to read MLX PLY {path:?}: {error}"))?;
    let mut lines = data.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err("MLX PLY is missing ply header".to_string());
    }

    let mut vertex_count = None::<usize>;
    let mut properties = Vec::<String>::new();
    let mut in_vertex = false;

    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed == "end_header" {
            break;
        }
        let parts: Vec<_> = trimmed.split_whitespace().collect();
        if parts.len() >= 3 && parts[0] == "element" {
            in_vertex = parts[1] == "vertex";
            if in_vertex {
                vertex_count = Some(
                    parts[2]
                        .parse::<usize>()
                        .map_err(|error| format!("invalid PLY vertex count: {error}"))?,
                );
            }
        } else if in_vertex && parts.len() >= 3 && parts[0] == "property" {
            properties.push(parts[2].to_string());
        }
    }

    let vertex_count = vertex_count.ok_or_else(|| "MLX PLY has no vertex element".to_string())?;
    let x_idx = property_index(&properties, "x")?;
    let y_idx = property_index(&properties, "y")?;
    let z_idx = property_index(&properties, "z")?;
    let fdc0_idx = property_index_opt(&properties, "f_dc_0");
    let fdc1_idx = property_index_opt(&properties, "f_dc_1");
    let fdc2_idx = property_index_opt(&properties, "f_dc_2");
    let red_idx = property_index_opt(&properties, "red");
    let green_idx = property_index_opt(&properties, "green");
    let blue_idx = property_index_opt(&properties, "blue");
    let opacity_idx = property_index_opt(&properties, "opacity");
    let scale0_idx = property_index_opt(&properties, "scale_0");
    let scale1_idx = property_index_opt(&properties, "scale_1");
    let scale2_idx = property_index_opt(&properties, "scale_2");
    let rot0_idx = property_index_opt(&properties, "rot_0");
    let rot1_idx = property_index_opt(&properties, "rot_1");
    let rot2_idx = property_index_opt(&properties, "rot_2");
    let rot3_idx = property_index_opt(&properties, "rot_3");

    let mut points = Vec::with_capacity(vertex_count);
    for line in lines.take(vertex_count) {
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<f32> = line
            .split_whitespace()
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("invalid PLY vertex value: {error}"))?;
        if values.len() < properties.len() {
            return Err("MLX PLY vertex has fewer values than header properties".to_string());
        }

        let r = read_color(&values, fdc0_idx, red_idx);
        let g = read_color(&values, fdc1_idx, green_idx);
        let b = read_color(&values, fdc2_idx, blue_idx);
        let scale = [
            read_scale(&values, scale0_idx),
            read_scale(&values, scale1_idx),
            read_scale(&values, scale2_idx),
        ];
        let radius = (scale[0] + scale[1] + scale[2]) / 3.0;
        points.push(SplatPoint {
            x: values[x_idx],
            y: values[y_idx],
            z: values[z_idx],
            r,
            g,
            b,
            radius,
            scale,
            rotation: [
                rot0_idx.map(|idx| values[idx]).unwrap_or(1.0),
                rot1_idx.map(|idx| values[idx]).unwrap_or(0.0),
                rot2_idx.map(|idx| values[idx]).unwrap_or(0.0),
                rot3_idx.map(|idx| values[idx]).unwrap_or(0.0),
            ],
            opacity_logit: opacity_idx.map(|idx| values[idx]).unwrap_or(1.734_601_f32),
        });
    }

    Ok(points)
}

fn read_splat(path: &Path) -> Result<Vec<SplatPoint>, String> {
    let data =
        fs::read(path).map_err(|error| format!("failed to read .splat file {path:?}: {error}"))?;
    if data.len() % 32 != 0 {
        return Err(".splat file size must be a multiple of 32 bytes".to_string());
    }
    let mut points = Vec::with_capacity(data.len() / 32);
    for record in data.chunks_exact(32) {
        let read_f32 = |offset: usize| {
            f32::from_le_bytes([
                record[offset],
                record[offset + 1],
                record[offset + 2],
                record[offset + 3],
            ])
        };
        let scale = [read_f32(12), read_f32(16), read_f32(20)];
        let opacity = (record[27] as f32 / 255.0).clamp(0.001, 0.999);
        points.push(SplatPoint {
            x: read_f32(0),
            y: read_f32(4),
            z: read_f32(8),
            r: record[24],
            g: record[25],
            b: record[26],
            radius: (scale[0] + scale[1] + scale[2]) / 3.0,
            scale,
            rotation: [
                (record[28] as f32 - 128.0) / 128.0,
                (record[29] as f32 - 128.0) / 128.0,
                (record[30] as f32 - 128.0) / 128.0,
                (record[31] as f32 - 128.0) / 128.0,
            ],
            opacity_logit: (opacity / (1.0 - opacity)).ln(),
        });
    }
    Ok(points)
}

fn property_index(properties: &[String], name: &str) -> Result<usize, String> {
    property_index_opt(properties, name).ok_or_else(|| format!("MLX PLY missing property {name}"))
}

fn property_index_opt(properties: &[String], name: &str) -> Option<usize> {
    properties.iter().position(|property| property == name)
}

fn read_color(values: &[f32], fdc_idx: Option<usize>, color_idx: Option<usize>) -> u8 {
    if let Some(index) = fdc_idx {
        return (((values[index] * SH_C0) + 0.5) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    if let Some(index) = color_idx {
        let value = values[index];
        return if value <= 1.0 {
            (value * 255.0).round().clamp(0.0, 255.0) as u8
        } else {
            value.round().clamp(0.0, 255.0) as u8
        };
    }
    200
}

fn read_scale(values: &[f32], scale_idx: Option<usize>) -> f32 {
    scale_idx
        .map(|idx| values[idx].exp().clamp(0.0001, 0.2))
        .unwrap_or(0.006)
}

fn export_fbx_native(
    fbx_path: &Path,
    visual_mesh: &MeshBuild,
    collider_mesh: &MeshBuild,
) -> Result<String, String> {
    let visual_vertices = fbx_vertices(visual_mesh);
    let collider_vertices = fbx_vertices(collider_mesh);
    let visual_triangles = fbx_triangles(visual_mesh)?;
    let collider_triangles = fbx_triangles(collider_mesh)?;

    write_fbx(
        fbx_path,
        FbxMesh {
            name: "scan_surface",
            vertices: &visual_vertices,
            triangles: &visual_triangles,
        },
        FbxMesh {
            name: "UCX_scan_surface_00",
            vertices: &collider_vertices,
            triangles: &collider_triangles,
        },
    )?;

    let bytes = fs::metadata(fbx_path)
        .map_err(|error| format!("failed to stat native FBX: {error}"))?
        .len();
    Ok(format!(
        "Native FBX ready: visual mesh + UCX collider, {:.1} MiB (no Blender)",
        bytes as f64 / (1024.0 * 1024.0)
    ))
}

fn fbx_vertices(mesh: &MeshBuild) -> Vec<FbxVertex> {
    mesh.vertices
        .iter()
        .map(|vertex| FbxVertex {
            position: [vertex.x as f64, vertex.y as f64, vertex.z as f64],
            color: [
                vertex.r as f64 / 255.0,
                vertex.g as f64 / 255.0,
                vertex.b as f64 / 255.0,
                1.0,
            ],
        })
        .collect()
}

fn fbx_triangles(mesh: &MeshBuild) -> Result<Vec<[u32; 3]>, String> {
    mesh.faces
        .iter()
        .enumerate()
        .map(|(face_index, face)| {
            let mut converted = [0u32; 3];
            for (slot, index) in face.iter().enumerate() {
                converted[slot] = index.checked_sub(1).ok_or_else(|| {
                    format!("mesh face {face_index} contains an invalid zero vertex index")
                })?;
            }
            Ok(converted)
        })
        .collect()
}

fn find_python() -> Option<String> {
    let venv_python = mlx_venv_python(&mlx_venv_dir());
    if venv_python.exists() {
        return Some(path_string(&venv_python));
    }
    find_system_python()
}

fn find_system_python() -> Option<String> {
    for candidate in [
        "/opt/homebrew/bin/python3.12",
        "/usr/local/bin/python3.12",
        "/opt/homebrew/bin/python3.13",
        "/usr/local/bin/python3.13",
        "/opt/homebrew/bin/python3.11",
        "/usr/local/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
        "/usr/local/bin/python3.10",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }
    find_in_path("python3")
        .or_else(|| find_in_path("python"))
        .map(|path| path.to_string_lossy().to_string())
}

fn mlx_venv_dir() -> PathBuf {
    let data_root = dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir);
    let legacy = data_root.join("Tomato Twin Capture").join("mlx-3dgs-venv");
    if legacy.exists() {
        legacy
    } else {
        data_root.join("AgriScan Studio").join("mlx-3dgs-venv")
    }
}

fn mlx_venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn sharp_checkpoint_path() -> PathBuf {
    let data_root = dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir);
    data_root
        .join("AgriScan Studio")
        .join("models")
        .join("sharp-mlx")
        .join("sharp_fp16.safetensors")
}

fn ensure_sharp_checkpoint(python: &str) -> Result<PathBuf, String> {
    const SHARP_CHECKPOINT_BYTES: u64 = 1_404_762_242;
    let checkpoint = sharp_checkpoint_path();
    if fs::metadata(&checkpoint).is_ok_and(|metadata| metadata.len() == SHARP_CHECKPOINT_BYTES) {
        return Ok(checkpoint);
    }
    let model_dir = checkpoint
        .parent()
        .ok_or_else(|| "SHARP checkpoint path has no parent directory".to_string())?;
    fs::create_dir_all(model_dir)
        .map_err(|error| format!("failed to create SHARP model directory: {error}"))?;
    let download_script = concat!(
        "from huggingface_hub import hf_hub_download\n",
        "import sys\n",
        "print(hf_hub_download(",
        "repo_id='agg23/Sharp-mlx-f16', ",
        "filename='sharp_fp16.safetensors', ",
        "local_dir=sys.argv[1]))\n"
    );
    let output = Command::new(python)
        .arg("-c")
        .arg(download_script)
        .arg(model_dir)
        .env("HF_XET_HIGH_PERFORMANCE", "1")
        .output()
        .map_err(|error| format!("failed to start pretrained SHARP download: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "failed to download pretrained SHARP checkpoint: {}",
            trim_process_output(
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
                20
            )
        ));
    }
    let bytes = fs::metadata(&checkpoint)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if bytes != SHARP_CHECKPOINT_BYTES {
        return Err(format!(
            "pretrained SHARP checkpoint is incomplete: {bytes} / {SHARP_CHECKPOINT_BYTES} bytes"
        ));
    }
    Ok(checkpoint)
}

fn ensure_mlx_venv(system_python: &str, venv_dir: &Path) -> Result<String, String> {
    let python = mlx_venv_python(venv_dir);
    if python.exists() {
        return Ok(path_string(&python));
    }

    if let Some(parent) = venv_dir.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create MLX venv parent: {error}"))?;
    }

    let output = Command::new(system_python)
        .arg("-m")
        .arg("venv")
        .arg(venv_dir)
        .output()
        .map_err(|error| format!("failed to create MLX venv: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("failed to create MLX venv: {stderr}{stdout}"));
    }

    if python.exists() {
        Ok(path_string(&python))
    } else {
        Err(format!(
            "MLX venv was created but python was not found at {}",
            path_string(&python)
        ))
    }
}

fn probe_mlx(python: &str) -> (bool, String) {
    let output = Command::new(python)
        .arg("-c")
        .arg(
            "import mlx.core as mx; import safetensors; print(f'MLX {mx.default_device()}, safetensors {safetensors.__version__}')",
        )
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let device = stdout.trim();
            let suffix = if device.is_empty() {
                "MLX and safetensors imports succeeded".to_string()
            } else {
                format!("Pretrained 3DGS inference ready: {device}")
            };
            (true, suffix)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            (
                false,
                format!("SHARP MLX runtime unavailable in {python}: {stderr}{stdout}. Use Setup MLX 3DGS to install mlx, safetensors, and the pretrained checkpoint into the app venv.")
                    .trim()
                    .to_string(),
            )
        }
        Err(error) => (
            false,
            format!("failed to run {python} for MLX probe: {error}"),
        ),
    }
}

fn run_python_install(python: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(python)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {python}: {error}"))?;
    let command = format!("{python} {}", args.join(" "));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() {
        Ok(format!(
            "{command}: ok\n{}",
            trim_install_log(&stdout, &stderr)
        ))
    } else {
        Err(format!("{command}: failed\n{stderr}{stdout}"))
    }
}

fn trim_install_log(stdout: &str, stderr: &str) -> String {
    trim_process_output(stdout, stderr, 8)
}

fn trim_process_output(stdout: &str, stderr: &str, max_lines: usize) -> String {
    let mut lines: Vec<_> = stdout
        .lines()
        .chain(stderr.lines())
        .filter(|line| !line.trim().is_empty())
        .collect();
    if lines.len() > max_lines {
        lines = lines.split_off(lines.len() - max_lines);
    }
    lines.join("\n")
}

fn find_in_path(binary: &str) -> Option<PathBuf> {
    let path_var: OsString = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

struct DepthImage {
    width: u32,
    height: u32,
    z16: Vec<u16>,
}

fn read_depth_png(path: &str) -> Result<DepthImage, String> {
    let file = File::open(path).map_err(|error| format!("failed to open depth PNG: {error}"))?;
    let decoder = Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("failed to decode depth PNG: {error}"))?;
    let mut data = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut data)
        .map_err(|error| format!("failed to read depth PNG frame: {error}"))?;
    let bytes = &data[..info.buffer_size()];
    if info.color_type != ColorType::Grayscale || info.bit_depth != BitDepth::Sixteen {
        return Err("depth PNG must be 16-bit grayscale".to_string());
    }
    let mut z16 = Vec::with_capacity((info.width * info.height) as usize);
    for chunk in bytes.chunks_exact(2) {
        z16.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    Ok(DepthImage {
        width: info.width,
        height: info.height,
        z16,
    })
}

fn read_rgb_png(path: &str) -> Result<ColorFrame, String> {
    let mut file = File::open(path).map_err(|error| format!("failed to open RGB PNG: {error}"))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read RGB PNG: {error}"))?;
    let decoder = Decoder::new(Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("failed to decode RGB PNG: {error}"))?;
    let mut data = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut data)
        .map_err(|error| format!("failed to read RGB PNG frame: {error}"))?;
    let bytes = &data[..info.buffer_size()];
    let rgb = match (info.color_type, info.bit_depth) {
        (ColorType::Rgb, BitDepth::Eight) => bytes.to_vec(),
        (ColorType::Rgba, BitDepth::Eight) => bytes
            .chunks_exact(4)
            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect(),
        _ => return Err("RGB PNG must be 8-bit RGB/RGBA".to_string()),
    };
    Ok(ColorFrame {
        width: info.width,
        height: info.height,
        rgb,
    })
}

fn sample_rgb(
    image: Option<&ColorFrame>,
    x: usize,
    y: usize,
    depth_width: usize,
    depth_height: usize,
) -> (u8, u8, u8) {
    if let Some(image) = image {
        let sx = ((x as f32 / depth_width as f32) * image.width as f32)
            .floor()
            .clamp(0.0, (image.width - 1) as f32) as usize;
        let sy = ((y as f32 / depth_height as f32) * image.height as f32)
            .floor()
            .clamp(0.0, (image.height - 1) as f32) as usize;
        let idx = (sy * image.width as usize + sx) * 3;
        return (image.rgb[idx], image.rgb[idx + 1], image.rgb[idx + 2]);
    }
    (200, 76, 54)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize JSON: {error}"))?;
    fs::write(path, json).map_err(|error| format!("failed to write {path:?}: {error}"))
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn io_error(error: std::io::Error) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn rigid_alignment_recovers_rotation_and_translation() {
        let expected = RigidTransform {
            rotation: RigidTransform::rotation_y(18.0_f32.to_radians()).rotation,
            translation: [0.045, -0.018, 0.032],
        };
        let source = [
            [-0.18, -0.12, -0.42],
            [0.16, -0.10, -0.38],
            [-0.14, 0.15, -0.51],
            [0.19, 0.11, -0.47],
            [0.02, -0.03, -0.31],
            [-0.06, 0.05, -0.62],
        ];
        let pairs: Vec<_> = source
            .into_iter()
            .map(|point| (point, expected.apply(point), 0.0))
            .collect();
        let recovered = best_fit_transform(&pairs).expect("recover rigid transform");
        for point in source {
            let actual = recovered.apply(point);
            let target = expected.apply(point);
            assert!(
                distance_squared3(actual, target) < 1.0e-8,
                "{actual:?} != {target:?}"
            );
        }
    }

    #[test]
    fn sharp_keyframe_sampling_keeps_one_hundred_frames() {
        let indices = sampled_frame_indices(685, 1, 100);
        assert_eq!(indices.len(), 100);
        assert_eq!(indices.first(), Some(&1));
        assert_eq!(indices.last(), Some(&685));
    }

    #[test]
    fn rgbd_session_builds_every_required_asset_without_external_fbx_tools() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let session_root =
            std::env::temp_dir().join(format!("tomato-twin-assets-{}-{stamp}", std::process::id()));
        let rgb_dir = session_root.join("rgb");
        let depth_dir = session_root.join("depth_z16");
        let metadata_dir = session_root.join("metadata");
        for dir in [&rgb_dir, &depth_dir, &metadata_dir] {
            fs::create_dir_all(dir).expect("create test session directory");
        }

        let rgb_path = rgb_dir.join("frame_000001_rgb.png");
        let depth_path = depth_dir.join("frame_000001_depth_z16.png");
        write_test_rgb_png(&rgb_path, 24, 24);
        write_test_depth_png(&depth_path, 24, 24);
        let metadata_path = metadata_dir.join("frame_000001.json");
        let metadata = serde_json::json!({
            "sessionId": "native_fbx_test",
            "frameIndex": 1,
            "frameNumber": 1,
            "timestampMs": 0.0,
            "intrinsics": {
                "width": 24,
                "height": 24,
                "ppx": 11.5,
                "ppy": 11.5,
                "fx": 120.0,
                "fy": 120.0,
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
            },
            "depthUnitsM": 0.001,
            "files": {
                "rgb": path_string(&rgb_path),
                "depth": path_string(&depth_path)
            }
        });
        fs::write(
            &metadata_path,
            serde_json::to_vec_pretty(&metadata).expect("serialize metadata"),
        )
        .expect("write metadata");

        let result = generate_scan_assets(AssetBuildOptions {
            session_root: path_string(&session_root),
            max_points: Some(5_000),
            frame_stride: Some(1),
            depth_decimation: Some(1),
            gaussian_radius_m: Some(0.006),
            turntable_degrees: Some(0.0),
            export_fbx: Some(true),
            use_mlx: Some(false),
            mlx_iterations: Some(0),
            mlx_voxel_size_m: Some(0.003),
            mlx_train_size: Some(64),
            mlx_max_train_views: Some(1),
            collider_max_faces: Some(500),
        })
        .expect("generate complete RGB-D asset set");

        let required = [
            &result.seed_gaussian_ply,
            &result.gaussian_ply,
            &result.splat,
            &result.mesh_obj,
            result.mesh_fbx.as_ref().expect("native FBX path"),
            &result.collider_obj,
            &result.collision_json,
            &result.preview_json,
            &result.manifest,
        ];
        for path in required {
            let metadata = fs::metadata(path).unwrap_or_else(|error| {
                panic!("required output {path} is missing: {error}");
            });
            assert!(metadata.len() > 0, "required output {path} is empty");
        }
        assert!(result.point_count >= 500);
        assert!(result.face_count > 0);
        assert!(result.fbx_status.contains("Native FBX ready"));
        assert!(result.mlx_status.contains("RGB-D Gaussian seed"));
        assert!(result.tools.fbx_available);
        assert_eq!(
            result.mesh_fbx.as_deref(),
            result.collision_fbx.as_deref(),
            "visual mesh and UCX collider share one import-ready FBX"
        );

        let manifest = fs::read_to_string(&result.manifest).expect("read asset manifest");
        assert!(manifest.contains("\"fbxStatus\""));
        assert!(manifest.contains("\"meshFbx\""));
        let _ = fs::remove_dir_all(&session_root);
    }

    fn write_test_rgb_png(path: &Path, width: u32, height: u32) {
        let file = File::create(path).expect("create RGB PNG");
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header().expect("write RGB PNG header");
        let mut rgb = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                rgb.extend_from_slice(&[180 + (x % 40) as u8, 35 + (y % 50) as u8, 28]);
            }
        }
        writer.write_image_data(&rgb).expect("write RGB PNG");
    }

    fn write_test_depth_png(path: &Path, width: u32, height: u32) {
        let file = File::create(path).expect("create depth PNG");
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(ColorType::Grayscale);
        encoder.set_depth(BitDepth::Sixteen);
        let mut writer = encoder.write_header().expect("write depth PNG header");
        let mut depth = Vec::with_capacity((width * height * 2) as usize);
        for y in 0..height {
            for x in 0..width {
                let millimeters = 420u16 + ((x + y) % 8) as u16;
                depth.extend_from_slice(&millimeters.to_be_bytes());
            }
        }
        writer.write_image_data(&depth).expect("write depth PNG");
    }
}
