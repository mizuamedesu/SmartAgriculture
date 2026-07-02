use std::{
    cmp::Ordering,
    ffi::OsString,
    fs::{self, File},
    io::{BufReader, BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

use png::{BitDepth, ColorType, Decoder};
use serde::{Deserialize, Serialize};

use crate::capture::Intrinsics;

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
    pub blender: Option<String>,
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewPayload {
    pub points: Vec<PreviewPoint>,
    pub bounds: Bounds,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub radius: f32,
}

#[derive(Debug, Clone, Serialize)]
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
        blender: find_blender().map(|path| path.to_string_lossy().to_string()),
        python,
        mlx_available,
        mlx_status,
        brush_hint: "gsplat-mlx is the active Apple Silicon 3DGS backend; it uses MLX autograd and differentiable rasterization instead of CUDA kernels.".to_string(),
    }
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

    let frames = load_frame_metadata(&session_root)?;
    if frames.is_empty() {
        return Err("no frame metadata found; capture a session first".to_string());
    }

    let selected: Vec<_> = frames
        .into_iter()
        .enumerate()
        .filter_map(|(index, frame)| (index as u32 % frame_stride == 0).then_some(frame))
        .collect();
    if selected.is_empty() {
        return Err("no frames selected for asset generation".to_string());
    }

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

    let mesh = build_mesh(
        &selected,
        depth_decimation,
        max_points,
        gaussian_radius_m,
        turntable_degrees,
    )?;
    if mesh.vertices.is_empty() {
        return Err("no valid depth points available for 3D reconstruction".to_string());
    }

    let seed_gaussian_ply = gaussian_dir.join("tomato_gaussians_seed.ply");
    let seed_splat = gaussian_dir.join("tomato_gaussians_seed.splat");
    let mlx_gaussian_ply = gaussian_dir.join("tomato_gaussians_mlx.ply");
    let mlx_splat = gaussian_dir.join("tomato_gaussians_mlx.splat");
    let mesh_obj = mesh_dir.join("tomato_surface.obj");
    let mesh_fbx = mesh_dir.join("tomato_surface.fbx");
    let collider_obj = mesh_dir.join("tomato_collider.obj");
    let collision_json = mesh_dir.join("tomato_collision.json");
    let blender_script = mesh_dir.join("obj_to_fbx.py");
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
    let mut mlx_status = if use_mlx {
        "MLX refinement requested".to_string()
    } else {
        "MLX refinement disabled".to_string()
    };

    if use_mlx {
        match run_mlx_refinement(
            &session_root,
            &seed_gaussian_ply,
            &mlx_gaussian_ply,
            &mlx_dir,
            max_points,
            gaussian_radius_m,
            mlx_voxel_size_m,
            mlx_iterations,
            frame_stride,
            turntable_degrees,
            mlx_train_size,
            mlx_max_train_views,
        ) {
            Ok(refinement) => {
                final_points = refinement.points;
                final_gaussian_ply = refinement.ply_path;
                final_splat = mlx_splat;
                write_splat(&final_splat, &final_points)?;
                mlx_status = refinement.status;
            }
            Err(error) => {
                mlx_status = format!("MLX refinement skipped: {error}");
            }
        }
    }

    let preview = build_preview_payload(&final_points);
    write_preview_json(&preview_json, &preview)?;

    let tools = detect_asset_tools();
    let fbx_status = if export_fbx {
        match export_fbx_with_blender(
            &mesh_obj,
            &collider_obj,
            &mesh_fbx,
            &blender_script,
            tools.blender.as_deref(),
        ) {
            Ok(status) => status,
            Err(error) => format!("FBX skipped: {error}"),
        }
    } else {
        "FBX export disabled".to_string()
    };

    let mesh_fbx_output = mesh_fbx.exists().then(|| path_string(&mesh_fbx));
    let collision_fbx_output = mesh_fbx.exists().then(|| path_string(&mesh_fbx));
    let seed_gaussian_ply_string = path_string(&seed_gaussian_ply);
    let gaussian_ply_string = path_string(&final_gaussian_ply);
    let splat_string = path_string(&final_splat);
    let mesh_obj_string = path_string(&mesh_obj);
    let collider_obj_string = path_string(&collider_obj);
    let collision_json_string = path_string(&collision_json);
    let preview_json_string = path_string(&preview_json);
    let manifest_data = AssetManifest {
        schema_version: "tomato-rgbd-assets-v1",
        source_session: selected
            .first()
            .map(|frame| frame.session_id.as_str())
            .unwrap_or("unknown"),
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
            frame,
            &depth,
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

#[allow(clippy::too_many_arguments)]
fn add_frame_mesh(
    frame: &FrameMetadata,
    depth: &DepthImage,
    color: Option<&RgbImage>,
    angle: f32,
    step: usize,
    max_points: usize,
    gaussian_radius_m: f32,
    vertices: &mut Vec<SplatPoint>,
    faces: &mut Vec<[u32; 3]>,
) {
    let width = depth.width as usize;
    let height = depth.height as usize;
    let grid_w = width.div_ceil(step);
    let grid_h = height.div_ceil(step);
    let mut index_grid = vec![None::<u32>; grid_w * grid_h];
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let depth_jump = gaussian_radius_m.max(0.006) * 10.0;
    let intr = frame.intrinsics;

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            if vertices.len() >= max_points {
                return;
            }
            let x = (gx * step).min(width - 1);
            let y = (gy * step).min(height - 1);
            let raw = depth.z16[y * width + x];
            if raw == 0 {
                continue;
            }

            let z = raw as f32 * frame.depth_units_m;
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
            if let (Some(a), Some(b), Some(c)) = (a, b, c) {
                if face_is_local(vertices, [a, b, c], depth_jump) {
                    faces.push([a + 1, b + 1, c + 1]);
                }
            }
            if let (Some(b), Some(d), Some(c)) = (b, d, c) {
                if face_is_local(vertices, [b, d, c], depth_jump) {
                    faces.push([b + 1, d + 1, c + 1]);
                }
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
        "comment Tomato Twin Capture 3DGS seed generated from RealSense RGB-D"
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
    writeln!(writer, "# Tomato Twin Capture surface mesh").map_err(io_error)?;
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
        schema_version: "tomato-rgbd-collision-v1",
        collider_type: "triangle_mesh",
        collider_obj: path_string(collider_obj),
        source_mesh: path_string(source_mesh),
        point_count: collider_mesh.vertices.len(),
        face_count: collider_mesh.faces.len(),
        bounds: collider_bounds,
        bounding_sphere: sphere,
        notes: format!(
            "Low-poly triangle mesh collider capped at {max_faces} faces; FBX object name uses UCX_tomato_surface_00 for engine import."
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

fn export_fbx_with_blender(
    obj_path: &Path,
    collider_obj_path: &Path,
    fbx_path: &Path,
    script_path: &Path,
    blender_path: Option<&str>,
) -> Result<String, String> {
    let blender = blender_path.ok_or_else(|| {
        "Blender was not found. Install Apple Silicon Blender, then rerun asset generation."
            .to_string()
    })?;
    fs::write(script_path, blender_script())
        .map_err(|error| format!("failed to write Blender script: {error}"))?;

    let output = Command::new(blender)
        .arg("--background")
        .arg("--python")
        .arg(script_path)
        .arg("--")
        .arg(obj_path)
        .arg(collider_obj_path)
        .arg(fbx_path)
        .output()
        .map_err(|error| format!("failed to run Blender: {error}"))?;

    if output.status.success() && fbx_path.exists() {
        Ok("FBX exported with Blender (visual mesh + UCX collider)".to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        Err(format!("Blender export failed: {stderr}{stdout}"))
    }
}

fn blender_script() -> &'static str {
    r#"
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[-3:]
obj_path = argv[0]
collider_obj_path = argv[1]
fbx_path = argv[2]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def import_obj(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)
    after = [obj for obj in bpy.context.scene.objects if obj not in before]
    return after

visual_objects = import_obj(obj_path)
for obj in visual_objects:
    obj.name = "tomato_surface"
    obj.data.name = "tomato_surface_mesh"
    obj.select_set(True)
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        if len(obj.data.polygons) == 0:
            continue
        obj.data.update()

collider_objects = import_obj(collider_obj_path)
for obj in collider_objects:
    obj.name = "UCX_tomato_surface_00"
    obj.data.name = "UCX_tomato_surface_00_mesh"
    obj["collision"] = "triangle_mesh"
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj.select_set(True)
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        obj.data.update()

bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    apply_unit_scale=True,
    bake_space_transform=False,
    axis_forward='-Z',
    axis_up='Y',
)
"#
}

fn find_blender() -> Option<PathBuf> {
    let candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/opt/homebrew/bin/blender",
        "/usr/local/bin/blender",
    ];
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    find_in_path("blender")
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
    dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("Tomato Twin Capture")
        .join("mlx-3dgs-venv")
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

struct RgbImage {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
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

fn read_rgb_png(path: &str) -> Result<RgbImage, String> {
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
    Ok(RgbImage {
        width: info.width,
        height: info.height,
        rgb,
    })
}

fn sample_rgb(
    image: Option<&RgbImage>,
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
