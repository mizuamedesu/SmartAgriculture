use std::{
    cmp::Ordering,
    collections::BTreeSet,
    ffi::OsString,
    fs::{self, File},
    io::{BufReader, BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

use png::{BitDepth, ColorType, Decoder, Encoder as PngEncoder};
use serde::{Deserialize, Serialize};

use crate::{
    capture::{ColorFrame, Intrinsics, default_output_root, legacy_output_root},
    fbx::{FbxMesh, FbxVertex, write_fbx},
    mcap_io::{self, DecodedRgbdFrame},
};

const SH_C0: f32 = 0.282_094_8;
const MLX_REFINE_SCRIPT: &str = include_str!("../../scripts/mlx_gaussian_refine.py");

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
    ply_path: PathBuf,
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
            "python3 not found; MLX refinement unavailable".to_string(),
        ),
    };
    AssetTools {
        fbx_available: true,
        fbx_exporter: "Built-in native FBX 7.4 exporter".to_string(),
        python,
        mlx_available,
        mlx_status,
        brush_hint: "FBX is exported natively with no Blender dependency. gsplat-mlx remains the Apple Silicon 3DGS training backend.".to_string(),
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

fn load_mcap_preview(recording: &Path) -> Result<AssetBuildResult, String> {
    let session_root = recording
        .parent()
        .ok_or_else(|| "MCAP recording has no parent folder".to_string())?;
    generate_scan_assets(AssetBuildOptions {
        session_root: path_string(session_root),
        max_points: Some(350_000),
        frame_stride: Some(1),
        depth_decimation: Some(2),
        gaussian_radius_m: Some(0.0035),
        turntable_degrees: Some(360.0),
        export_fbx: Some(false),
        use_mlx: Some(false),
        mlx_iterations: Some(0),
        mlx_voxel_size_m: Some(0.0025),
        mlx_train_size: Some(320),
        mlx_max_train_views: Some(12),
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
    let preview = build_preview_payload(&points);
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
        ],
        vec![
            "-m",
            "pip",
            "install",
            "--upgrade",
            "git+https://github.com/RobotFlow-Labs/gsplat-mlx.git",
        ],
    ];

    for args in commands.drain(..) {
        let result = run_python_install(&python, &args)?;
        log.push(result);
    }

    let tools = detect_asset_tools();
    if tools.mlx_available {
        Ok(MlxSetupResult {
            status: tools.mlx_status.clone(),
            log,
            tools,
        })
    } else {
        Err(format!(
            "gsplat-mlx setup finished but probe failed: {}",
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
    let turntable_degrees = options
        .turntable_degrees
        .unwrap_or(360.0)
        .clamp(0.0, 1080.0);
    let export_fbx = options.export_fbx.unwrap_or(true);
    let use_mlx = options.use_mlx.unwrap_or(true);
    let mlx_iterations = options.mlx_iterations.unwrap_or(1_600).clamp(0, 20_000);
    let mlx_voxel_size_m = options
        .mlx_voxel_size_m
        .unwrap_or(gaussian_radius_m * 0.75)
        .clamp(0.0005, 0.05);
    let mlx_train_size = options.mlx_train_size.unwrap_or(320).clamp(64, 1024);
    let mlx_max_train_views = options.mlx_max_train_views.unwrap_or(12).clamp(1, 64);
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
    let preview_dir = asset_root.join("preview");
    for dir in [
        &asset_root,
        &gaussian_dir,
        &mesh_dir,
        &mlx_dir,
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
    if mesh.vertices.is_empty() {
        return Err("no valid depth points available for 3D reconstruction".to_string());
    }

    let seed_gaussian_ply = gaussian_dir.join("scan_gaussians_seed.ply");
    let seed_splat = gaussian_dir.join("scan_gaussians_seed.splat");
    let mlx_gaussian_ply = gaussian_dir.join("scan_gaussians_mlx.ply");
    let mlx_splat = gaussian_dir.join("scan_gaussians_mlx.splat");
    let mesh_obj = mesh_dir.join("scan_surface.obj");
    let mesh_fbx = mesh_dir.join("scan_surface.fbx");
    let collider_obj = mesh_dir.join("scan_collider.obj");
    let collision_json = mesh_dir.join("scan_collision.json");
    let preview_json = preview_dir.join("preview_points.json");
    let manifest = asset_root.join("asset_manifest.json");

    write_gaussian_ply(&seed_gaussian_ply, &mesh.vertices)?;
    write_splat(&seed_splat, &mesh.vertices)?;
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

    let mut final_points = mesh.vertices.clone();
    let mut final_gaussian_ply = seed_gaussian_ply.clone();
    let mut final_splat = seed_splat.clone();
    let mut mlx_status = "RGB-D Gaussian seed (MLX refinement disabled)".to_string();

    if use_mlx {
        let (mlx_session_root, mlx_frame_stride, cleanup_cache) = if is_mcap {
            let cache = mlx_dir.join("mcap_training_cache");
            write_mcap_training_cache(&cache, &mcap_training_frames)?;
            (cache.clone(), 1, Some(cache))
        } else {
            (session_root.clone(), frame_stride, None)
        };
        let refinement_result = run_mlx_refinement(
            &mlx_session_root,
            &seed_gaussian_ply,
            &mlx_gaussian_ply,
            &mlx_dir,
            max_points,
            gaussian_radius_m,
            mlx_voxel_size_m,
            mlx_iterations,
            mlx_frame_stride,
            turntable_degrees,
            mlx_train_size,
            mlx_max_train_views,
        );
        if let Some(cache) = cleanup_cache {
            let _ = fs::remove_dir_all(cache);
        }
        let refinement = refinement_result.map_err(|error| {
            format!(
                "MLX 3DGS refinement was requested but failed; no fallback was reported as success: {error}"
            )
        })?;
        final_points = refinement.points;
        final_gaussian_ply = refinement.ply_path;
        final_splat = mlx_splat;
        write_splat(&final_splat, &final_points)?;
        mlx_status = refinement.status;
    }

    let preview = build_preview_payload(&final_points);
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
            angle,
            depth_decimation as usize,
            max_points,
            gaussian_radius_m,
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
) -> Result<(MeshBuild, Vec<DecodedRgbdFrame>), String> {
    let total_frames = mcap_io::frame_count(recording_path)?;
    if total_frames == 0 {
        return Err("MCAP contains no RGB-D frames".to_string());
    }
    let stride = frame_stride.max(1) as usize;
    let selected_count = total_frames.div_ceil(stride);
    let mesh_indices = sampled_frame_indices(total_frames, stride, selected_count.min(64));
    let training_indices = sampled_frame_indices(
        total_frames,
        stride,
        selected_count.min(max_train_views.max(1) as usize),
    );
    let requested_indices: BTreeSet<_> = mesh_indices.union(&training_indices).copied().collect();
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut training_frames = Vec::new();

    mcap_io::visit_frame_indices(recording_path, &requested_indices, |frame| {
        let selected_ordinal = frame.info.frame_index.saturating_sub(1) as usize / stride;
        let angle = if turntable_degrees.abs() < f32::EPSILON || selected_count <= 1 {
            0.0
        } else {
            let t = selected_ordinal as f32 / (selected_count - 1) as f32;
            t * turntable_degrees.to_radians()
        };
        if mesh_indices.contains(&frame.info.frame_index) && vertices.len() < max_points {
            add_frame_mesh(
                frame.info.intrinsics,
                frame.info.depth_units_m,
                frame.depth.width,
                frame.depth.height,
                &frame.depth.z16,
                frame.color.as_ref(),
                angle,
                depth_decimation as usize,
                max_points,
                gaussian_radius_m,
                &mut vertices,
                &mut faces,
            );
        }
        if training_indices.contains(&frame.info.frame_index) {
            training_frames.push(frame);
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
                0
            } else {
                sample * (selected_count - 1) / (samples - 1)
            };
            (ordinal * stride + 1).min(total_frames) as u32
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn add_frame_mesh(
    intr: Intrinsics,
    depth_units_m: f32,
    depth_width: u32,
    depth_height: u32,
    depth_z16: &[u16],
    color: Option<&ColorFrame>,
    angle: f32,
    step: usize,
    max_points: usize,
    gaussian_radius_m: f32,
    vertices: &mut Vec<SplatPoint>,
    faces: &mut Vec<[u32; 3]>,
) {
    let width = depth_width as usize;
    let height = depth_height as usize;
    let grid_w = width.div_ceil(step);
    let grid_h = height.div_ceil(step);
    let mut index_grid = vec![None::<u32>; grid_w * grid_h];
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let depth_jump = gaussian_radius_m.max(0.006) * 10.0;

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            if vertices.len() >= max_points {
                return;
            }
            let x = (gx * step).min(width - 1);
            let y = (gy * step).min(height - 1);
            let raw = depth_z16[y * width + x];
            if raw == 0 {
                continue;
            }

            let z = raw as f32 * depth_units_m;
            if !(0.02..=8.0).contains(&z) {
                continue;
            }

            let px = (x as f32 - intr.ppx) / intr.fx * z;
            let py = -((y as f32 - intr.ppy) / intr.fy * z);
            let pz = -z;
            let rx = px * cos_a - pz * sin_a;
            let rz = px * sin_a + pz * cos_a;
            let (r, g, b) = sample_rgb(color, x, y, width, height);

            let vertex_index = vertices.len() as u32;
            vertices.push(SplatPoint {
                x: rx,
                y: py,
                z: rz,
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

fn build_preview_payload(points: &[SplatPoint]) -> PreviewPayload {
    let preview_points = downsample_preview(points, 35_000);
    PreviewPayload {
        bounds: bounds(points),
        points: preview_points
            .into_iter()
            .map(|point| PreviewPoint {
                x: point.x,
                y: point.y,
                z: point.z,
                r: point.r,
                g: point.g,
                b: point.b,
                radius: point.radius,
                scale: point.scale,
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
    let step = (points.len() as f32 / limit as f32).ceil() as usize;
    points.iter().step_by(step).cloned().collect()
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

fn write_mcap_training_cache(cache_root: &Path, frames: &[DecodedRgbdFrame]) -> Result<(), String> {
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

    for (index, frame) in frames.iter().enumerate() {
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

#[allow(clippy::too_many_arguments)]
fn run_mlx_refinement(
    session_root: &Path,
    seed_ply: &Path,
    output_ply: &Path,
    mlx_dir: &Path,
    max_points: usize,
    gaussian_radius_m: f32,
    voxel_size_m: f32,
    iterations: u32,
    frame_stride: u32,
    turntable_degrees: f32,
    train_size: u32,
    max_train_views: u32,
) -> Result<MlxRefinement, String> {
    let python =
        find_python().ok_or_else(|| "python3 not found; install Python and mlx".to_string())?;
    let (mlx_available, mlx_status) = probe_mlx(&python);
    if !mlx_available {
        return Err(mlx_status);
    }

    let script_path = mlx_dir.join("mlx_gaussian_refine.py");
    let summary_path = mlx_dir.join("mlx_refine_summary.json");
    fs::write(&script_path, MLX_REFINE_SCRIPT)
        .map_err(|error| format!("failed to write MLX script: {error}"))?;

    let output = Command::new(&python)
        .arg(&script_path)
        .arg("--input-ply")
        .arg(seed_ply)
        .arg("--output-ply")
        .arg(output_ply)
        .arg("--summary-json")
        .arg(&summary_path)
        .arg("--session-root")
        .arg(session_root)
        .arg("--max-points")
        .arg(max_points.to_string())
        .arg("--radius")
        .arg(gaussian_radius_m.to_string())
        .arg("--voxel-size")
        .arg(voxel_size_m.to_string())
        .arg("--iterations")
        .arg(iterations.to_string())
        .arg("--frame-stride")
        .arg(frame_stride.to_string())
        .arg("--turntable-degrees")
        .arg(turntable_degrees.to_string())
        .arg("--train-size")
        .arg(train_size.to_string())
        .arg("--max-train-views")
        .arg(max_train_views.to_string())
        .output()
        .map_err(|error| format!("failed to run MLX refinement: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "gsplat-mlx training failed:\n{}",
            trim_process_output(&stdout, &stderr, 24)
        ));
    }
    if !output_ply.exists() {
        return Err("MLX process finished without output PLY".to_string());
    }

    let points = read_gaussian_ply(output_ply)?;
    if points.is_empty() {
        return Err("MLX output PLY contained no gaussians".to_string());
    }

    let status = mlx_summary_status(&summary_path)
        .unwrap_or_else(|| format!("MLX refined {} gaussians ({mlx_status})", points.len()));

    Ok(MlxRefinement {
        points,
        ply_path: output_ply.to_path_buf(),
        status,
    })
}

fn mlx_summary_status(path: &Path) -> Option<String> {
    let data = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&data).ok()?;
    let input = value.get("inputPointCount")?.as_u64()?;
    let output = value.get("outputPointCount")?.as_u64()?;
    let iterations = value.get("iterations")?.as_u64()?;
    let backend = value
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("gsplat-mlx");
    let device = value
        .get("device")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("MLX");
    let train_views = value.get("trainViews").and_then(serde_json::Value::as_u64);
    let train_width = value.get("trainWidth").and_then(serde_json::Value::as_u64);
    let train_height = value.get("trainHeight").and_then(serde_json::Value::as_u64);
    let loss = value.get("finalLoss").and_then(serde_json::Value::as_f64);
    let train_shape = match (train_views, train_width, train_height) {
        (Some(views), Some(width), Some(height)) => {
            format!("{views} views at {width}x{height}")
        }
        _ => "RGB training views".to_string(),
    };
    let supervision = if value
        .get("depthSupervision")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        "RGB+D"
    } else {
        "RGB"
    };
    Some(match loss {
        Some(loss) => format!(
            "{backend} trained {input} seed points into {output} gaussians on {device}; {supervision}, {train_shape}, {iterations} iterations, final loss {loss:.5}"
        ),
        None => format!(
            "{backend} prepared {input} seed points into {output} gaussians on {device}; {supervision}, {train_shape}, {iterations} iterations"
        ),
    })
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
            "import mlx.core as mx; import gsplat_mlx; from gsplat_mlx import rasterization; print(f'gsplat-mlx {getattr(gsplat_mlx, \"__version__\", \"unknown\")} on {mx.default_device()}')",
        )
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let device = stdout.trim();
            let suffix = if device.is_empty() {
                "gsplat-mlx import succeeded".to_string()
            } else {
                format!("MLX 3DGS ready: {device}")
            };
            (true, suffix)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            (
                false,
                format!("gsplat-mlx unavailable in {python}: {stderr}{stdout}. Use Setup MLX 3DGS to install mlx and gsplat-mlx into the app venv.")
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
