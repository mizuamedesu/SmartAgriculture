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
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetTools {
    pub blender: Option<String>,
    pub brush_hint: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetBuildResult {
    pub root: String,
    pub gaussian_ply: String,
    pub splat: String,
    pub mesh_obj: String,
    pub mesh_fbx: Option<String>,
    pub preview_json: String,
    pub manifest: String,
    pub point_count: usize,
    pub face_count: usize,
    pub fbx_status: String,
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
    gaussian_ply: &'a str,
    splat: &'a str,
    mesh_obj: &'a str,
    mesh_fbx: Option<&'a str>,
    preview_json: &'a str,
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
}

#[derive(Debug, Clone)]
struct MeshBuild {
    vertices: Vec<SplatPoint>,
    faces: Vec<[u32; 3]>,
}

#[tauri::command]
pub fn detect_asset_tools() -> AssetTools {
    AssetTools {
        blender: find_blender().map(|path| path.to_string_lossy().to_string()),
        brush_hint: "Brush is the best fit for Apple Silicon 3DGS training because it uses WebGPU/Burn instead of CUDA-only kernels.".to_string(),
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
    let max_points = options.max_points.unwrap_or(180_000).clamp(5_000, 1_500_000);
    let gaussian_radius_m = options.gaussian_radius_m.unwrap_or(0.006).clamp(0.0005, 0.05);
    let turntable_degrees = options.turntable_degrees.unwrap_or(360.0).clamp(0.0, 1080.0);
    let export_fbx = options.export_fbx.unwrap_or(true);

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
    let preview_dir = asset_root.join("preview");
    for dir in [&asset_root, &gaussian_dir, &mesh_dir, &preview_dir] {
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

    let gaussian_ply = gaussian_dir.join("tomato_gaussians_seed.ply");
    let splat = gaussian_dir.join("tomato_gaussians_seed.splat");
    let mesh_obj = mesh_dir.join("tomato_surface.obj");
    let mesh_fbx = mesh_dir.join("tomato_surface.fbx");
    let blender_script = mesh_dir.join("obj_to_fbx.py");
    let preview_json = preview_dir.join("preview_points.json");
    let manifest = asset_root.join("asset_manifest.json");

    write_gaussian_ply(&gaussian_ply, &mesh.vertices)?;
    write_splat(&splat, &mesh.vertices)?;
    write_obj(&mesh_obj, &mesh)?;
    let preview = build_preview_payload(&mesh.vertices);
    write_preview_json(&preview_json, &preview)?;

    let tools = detect_asset_tools();
    let fbx_status = if export_fbx {
        match export_fbx_with_blender(&mesh_obj, &mesh_fbx, &blender_script, tools.blender.as_deref()) {
            Ok(status) => status,
            Err(error) => format!("FBX skipped: {error}"),
        }
    } else {
        "FBX export disabled".to_string()
    };

    let mesh_fbx_output = mesh_fbx.exists().then(|| path_string(&mesh_fbx));
    let gaussian_ply_string = path_string(&gaussian_ply);
    let splat_string = path_string(&splat);
    let mesh_obj_string = path_string(&mesh_obj);
    let preview_json_string = path_string(&preview_json);
    let manifest_data = AssetManifest {
        schema_version: "tomato-rgbd-assets-v1",
        source_session: selected
            .first()
            .map(|frame| frame.session_id.as_str())
            .unwrap_or("unknown"),
        point_count: mesh.vertices.len(),
        face_count: mesh.faces.len(),
        gaussian_ply: &gaussian_ply_string,
        splat: &splat_string,
        mesh_obj: &mesh_obj_string,
        mesh_fbx: mesh_fbx_output.as_deref(),
        preview_json: &preview_json_string,
        options: AssetOptionsSummary {
            max_points,
            frame_stride,
            depth_decimation,
            gaussian_radius_m,
            turntable_degrees,
        },
    };
    write_json(&manifest, &manifest_data)?;

    Ok(AssetBuildResult {
        root: path_string(&asset_root),
        gaussian_ply: gaussian_ply_string,
        splat: splat_string,
        mesh_obj: mesh_obj_string,
        mesh_fbx: mesh_fbx_output,
        preview_json: preview_json_string,
        manifest: path_string(&manifest),
        point_count: mesh.vertices.len(),
        face_count: mesh.faces.len(),
        fbx_status,
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
    writeln!(writer, "comment Tomato Twin Capture 3DGS seed generated from RealSense RGB-D").map_err(io_error)?;
    writeln!(writer, "element vertex {}", points.len()).map_err(io_error)?;
    for property in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        writeln!(writer, "property float {property}").map_err(io_error)?;
    }
    writeln!(writer, "end_header").map_err(io_error)?;

    for point in points {
        let f_dc_0 = (point.r as f32 / 255.0 - 0.5) / SH_C0;
        let f_dc_1 = (point.g as f32 / 255.0 - 0.5) / SH_C0;
        let f_dc_2 = (point.b as f32 / 255.0 - 0.5) / SH_C0;
        let opacity_logit = 1.734_601_f32;
        let log_scale = point.radius.max(0.0001).ln();
        writeln!(
            writer,
            "{:.6} {:.6} {:.6} 0 0 0 {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} 1 0 0 0",
            point.x,
            point.y,
            point.z,
            f_dc_0,
            f_dc_1,
            f_dc_2,
            opacity_logit,
            log_scale,
            log_scale,
            log_scale,
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
        for value in [point.x, point.y, point.z, point.radius, point.radius, point.radius] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| format!("failed to write .splat: {error}"))?;
        }
        writer
            .write_all(&[point.r, point.g, point.b, 220, 255, 0, 0, 0])
            .map_err(|error| format!("failed to write .splat: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("failed to flush .splat: {error}"))
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

fn export_fbx_with_blender(
    obj_path: &Path,
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
        .arg(fbx_path)
        .output()
        .map_err(|error| format!("failed to run Blender: {error}"))?;

    if output.status.success() && fbx_path.exists() {
        Ok("FBX exported with Blender".to_string())
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

obj_path = sys.argv[-2]
fbx_path = sys.argv[-1]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

if hasattr(bpy.ops.wm, "obj_import"):
    bpy.ops.wm.obj_import(filepath=obj_path)
else:
    bpy.ops.import_scene.obj(filepath=obj_path)

for obj in bpy.context.scene.objects:
    obj.select_set(True)
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        if len(obj.data.polygons) == 0:
            continue
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
