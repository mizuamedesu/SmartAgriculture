import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  Activity,
  Box,
  Camera,
  CheckCircle2,
  Circle,
  Cpu,
  Download,
  FolderOpen,
  HardDrive,
  Loader2,
  Radio,
  RefreshCw,
  ScanLine,
  SlidersHorizontal,
  Square,
  WandSparkles
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Input, Label, Select, Textarea } from "./components/ui/form";
import { cn } from "./lib/utils";
import "./styles.css";

type BackendMode = "auto" | "realsense" | "synthetic";
type AppScreen = "capture" | "preview" | "settings";

interface CaptureConfig {
  width: number;
  height: number;
  fps: number;
  backend: BackendMode;
  targetLabel: string;
  outputRoot: string;
  cultivar: string;
  notes: string;
  maxFrames: number | null;
  pointStride: number;
  minDepthM: number;
  maxDepthM: number;
}

interface AssetOptions {
  maxPoints: number;
  frameStride: number;
  depthDecimation: number;
  gaussianRadiusM: number;
  turntableDegrees: number;
  exportFbx: boolean;
  useMlx: boolean;
  mlxIterations: number;
  mlxVoxelSizeM: number;
  mlxTrainSize: number;
  mlxMaxTrainViews: number;
  colliderMaxFaces: number;
}

interface CameraDevice {
  name: string;
  serial: string;
  firmware: string;
  usb: string;
  productLine: string;
}

interface RuntimeProbe {
  sdkLoaded: boolean;
  apiVersion: string | null;
  devices: CameraDevice[];
  usbDevices: UsbRealSenseDevice[];
  status: string;
  installHint: string | null;
  actionRequired: string | null;
}

interface UsbRealSenseDevice {
  productName: string;
  linkSpeedMbps: number | null;
  usbType: string | null;
  idProduct: string | null;
  locationId: string | null;
}

interface SdkSetupResult {
  status: string;
  log: string[];
}

interface MlxSetupResult {
  status: string;
  log: string[];
  tools: AssetTools;
}

interface SessionStarted {
  sessionId: string;
  root: string;
  backend: string;
  notice: string | null;
  progressPath: string | null;
}

interface SessionStopped {
  framesWritten: number;
}

interface PrivilegedPreviewStarted {
  sessionId: string;
  framePath: string;
  pidPath: string;
  logPath: string;
  launchMode: string;
}

interface InstalledHelper {
  path: string;
  status: string;
  ready: boolean;
  current: boolean;
}

interface DepthStats {
  validPoints: number;
  minM: number;
  maxM: number;
  meanM: number;
}

interface FramePaths {
  rgb: string | null;
  depth: string;
  pointCloud: string;
  metadata: string;
}

interface FrameSummary {
  sessionId: string;
  frameIndex: number;
  timestampMs: number;
  frameNumber: number;
  colorPreviewDataUrl: string | null;
  depthPreviewDataUrl: string;
  depth: DepthStats;
  paths: FramePaths;
}

interface CaptureEvent {
  kind: "frame" | "error" | "finished";
  summary: FrameSummary | null;
  message: string | null;
}

interface AssetTools {
  fbxAvailable: boolean;
  fbxExporter: string;
  python: string | null;
  mlxAvailable: boolean;
  mlxStatus: string;
  brushHint: string;
}

interface PreviewPoint {
  x: number;
  y: number;
  z: number;
  r: number;
  g: number;
  b: number;
  radius: number;
  scale: [number, number, number];
  rotation: [number, number, number, number];
  opacity: number;
}

interface PreviewPayload {
  points: PreviewPoint[];
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
    center: [number, number, number];
  };
}

interface AssetBuildResult {
  root: string;
  seedGaussianPly: string;
  gaussianPly: string;
  splat: string;
  meshObj: string;
  meshFbx: string | null;
  colliderObj: string;
  collisionJson: string;
  collisionFbx: string | null;
  previewJson: string;
  manifest: string;
  pointCount: number;
  faceCount: number;
  fbxStatus: string;
  mlxStatus: string;
  collisionStatus: string;
  tools: AssetTools;
  preview: PreviewPayload;
}

const isTauri = Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
const CAPTURE_PROFILES = [
  { label: "1280 x 720 / 30 fps", width: 1280, height: 720, fps: 30 },
  { label: "848 x 480 / 30 fps", width: 848, height: 480, fps: 30 },
  { label: "640 x 480 / 30 fps", width: 640, height: 480, fps: 30 },
  { label: "640 x 480 / 15 fps", width: 640, height: 480, fps: 15 },
  { label: "424 x 240 / 30 fps", width: 424, height: 240, fps: 30 },
  { label: "320 x 240 / 30 fps", width: 320, height: 240, fps: 30 }
] as const;

function App() {
  const [activeScreen, setActiveScreen] = useState<AppScreen>("capture");
  const [probe, setProbe] = useState<RuntimeProbe | null>(null);
  const [config, setConfig] = useState<CaptureConfig>({
    width: 1280,
    height: 720,
    fps: 30,
    backend: "realsense",
    targetLabel: "scan",
    outputRoot: window.localStorage.getItem("agriscan.outputRoot") ?? "",
    cultivar: "",
    notes: "",
    maxFrames: null,
    pointStride: 4,
    minDepthM: 0.12,
    maxDepthM: 1.4
  });
  const [assetOptions, setAssetOptions] = useState<AssetOptions>({
    maxPoints: 350000,
    frameStride: 1,
    depthDecimation: 2,
    gaussianRadiusM: 0.0035,
    turntableDegrees: 360,
    exportFbx: true,
    useMlx: true,
    mlxIterations: 1600,
    mlxVoxelSizeM: 0.0025,
    mlxTrainSize: 320,
    mlxMaxTrainViews: 12,
    colliderMaxFaces: 35000
  });
  const [recording, setRecording] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [activeSession, setActiveSession] = useState<SessionStarted | null>(null);
  const [previewSession, setPreviewSession] = useState<SessionStarted | null>(null);
  const [privilegedPreview, setPrivilegedPreview] = useState<PrivilegedPreviewStarted | null>(null);
  const [latestFrame, setLatestFrame] = useState<FrameSummary | null>(null);
  const [assetTools, setAssetTools] = useState<AssetTools | null>(null);
  const [assetResult, setAssetResult] = useState<AssetBuildResult | null>(null);
  const [probeBusy, setProbeBusy] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [captureStarting, setCaptureStarting] = useState(false);
  const [captureStopping, setCaptureStopping] = useState(false);
  const [assetBusy, setAssetBusy] = useState(false);
  const [sdkSetupBusy, setSdkSetupBusy] = useState(false);
  const [mlxSetupBusy, setMlxSetupBusy] = useState(false);
  const [helperInstallBusy, setHelperInstallBusy] = useState(false);
  const [helperStatus, setHelperStatus] = useState<InstalledHelper | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const mockTimer = useRef<number | null>(null);
  const privilegedPollTimer = useRef<number | null>(null);
  const recordingPollTimer = useRef<number | null>(null);
  const latestFrameAttachTimer = useRef<number | null>(null);
  const previewTimeoutTimer = useRef<number | null>(null);
  const previewRequestId = useRef(0);
  const privilegedReadBusy = useRef(false);
  const recordingReadBusy = useRef(false);
  const privilegedPreviewRef = useRef<PrivilegedPreviewStarted | null>(null);
  const autoSetupAttempted = useRef(false);
  const helperBootAttempted = useRef(false);

  const devices = probe?.devices ?? [];
  const backend = activeSession?.backend ?? previewSession?.backend ?? config.backend;
  const helperReady =
    !isTauri ||
    config.backend === "synthetic" ||
    Boolean(helperStatus?.ready && helperStatus.current);
  const busyMessage = captureStopping
    ? "Loading: stopping recording"
    : captureStarting
      ? "Loading: starting RGB-D recording"
      : previewLoading
        ? "Loading: opening RealSense preview"
        : sdkSetupBusy
          ? "Loading: checking SDK"
          : mlxSetupBusy
            ? "Loading: installing MLX 3DGS"
            : helperInstallBusy
              ? "Loading: preparing RealSense helper"
              : assetBusy
                ? "Loading: generating 3D assets"
                : probeBusy
                  ? "Loading: refreshing devices"
                  : null;

  useEffect(() => {
    privilegedPreviewRef.current = privilegedPreview;
  }, [privilegedPreview]);

  const pushLog = (message: string) => {
    const stamp = new Date().toLocaleTimeString("ja-JP", { hour12: false });
    setLog((current) => [`${stamp} ${message}`, ...current].slice(0, 12));
  };

  const refreshProbe = async (options?: { autoSetup?: boolean }) => {
    setProbeBusy(true);
    try {
      const runtime = await tauriCall<RuntimeProbe>("probe_runtime");
      const tools = await tauriCall<AssetTools>("detect_asset_tools");
      setProbe(runtime);
      setAssetTools(tools);
      pushLog(runtime.status);
      if (options?.autoSetup && isTauri && !runtime.sdkLoaded && !autoSetupAttempted.current) {
        autoSetupAttempted.current = true;
        pushLog("SDK missing; running automatic setup");
        await setupSdk();
      }
    } catch (error) {
      pushLog(`probe failed: ${String(error)}`);
    } finally {
      setProbeBusy(false);
    }
  };

  const setupSdk = async (): Promise<RuntimeProbe | null> => {
    setSdkSetupBusy(true);
    pushLog("checking RealSense SDK and camera connection");
    try {
      const result = await tauriCall<SdkSetupResult>("ensure_realsense_sdk");
      pushLog(result.status);
      result.log.slice(-3).reverse().forEach(pushLog);
      const runtime = await tauriCall<RuntimeProbe>("probe_runtime");
      setProbe(runtime);
      return runtime;
    } catch (error) {
      pushLog(`SDK setup failed: ${String(error)}`);
      return null;
    } finally {
      setSdkSetupBusy(false);
    }
  };

  const setupMlx3dgs = async (): Promise<AssetTools | null> => {
    setMlxSetupBusy(true);
    pushLog("installing MLX 3DGS backend: mlx + gsplat-mlx");
    try {
      const result = await tauriCall<MlxSetupResult>("ensure_mlx_3dgs");
      setAssetTools(result.tools);
      pushLog(result.status);
      result.log.slice(-3).reverse().forEach((line) => pushLog(firstLine(line)));
      return result.tools;
    } catch (error) {
      pushLog(`MLX 3DGS setup failed: ${String(error)}`);
      return null;
    } finally {
      setMlxSetupBusy(false);
    }
  };

  const installHelper = async () => {
    setHelperInstallBusy(true);
    pushLog("preparing the RealSense helper for this app version");
    try {
      const result = await tauriCall<InstalledHelper>("ensure_privileged_helper");
      setHelperStatus(result);
      pushLog(result.status);
      pushLog(result.path);
      return result;
    } catch (error) {
      pushLog(`helper install failed: ${String(error)}`);
      return null;
    } finally {
      setHelperInstallBusy(false);
    }
  };

  const stopRecordingPolling = () => {
    if (recordingPollTimer.current !== null) {
      window.clearInterval(recordingPollTimer.current);
      recordingPollTimer.current = null;
    }
    recordingReadBusy.current = false;
  };

  const startRecordingPolling = (progressPath: string) => {
    stopRecordingPolling();
    recordingPollTimer.current = window.setInterval(async () => {
      if (recordingReadBusy.current) return;
      recordingReadBusy.current = true;
      try {
        const frame = await tauriCall<FrameSummary>("read_privileged_recording_frame", {
          progressPath
        });
        setLatestFrame((current) =>
          current?.sessionId === frame.sessionId && current.frameIndex >= frame.frameIndex ? current : frame
        );
      } catch {
        // The event channel is primary; this polling path guarantees live recording preview.
      } finally {
        recordingReadBusy.current = false;
      }
    }, Math.max(33, Math.round(1000 / Math.max(1, config.fps))));
  };

  const startCapture = async () => {
    setCaptureStarting(true);
    try {
      if (previewing) {
        await stopPreview();
      }
      const wantsRealSense = config.backend === "auto" || config.backend === "realsense";
      if (wantsRealSense && !probe?.sdkLoaded) {
        await setupSdk();
      }

      stopRecordingPolling();
      setLatestFrame(null);
      const session = await tauriCall<SessionStarted>("start_recording", { config });
      setRecording(true);
      setPreviewing(false);
      setActiveSession(session);
      setPreviewSession(null);
      if (session.progressPath) {
        startRecordingPolling(session.progressPath);
      }
      pushLog(`started ${session.backend}: ${session.sessionId}`);
      pushLog("live RGB-D preview follows every recorded frame");
      if (session.notice) pushLog(session.notice);
      if (!isTauri) startMockFrames(session, config, mockTimer, setLatestFrame);
    } catch (error) {
      pushLog(`start failed: ${String(error)}`);
    } finally {
      setCaptureStarting(false);
    }
  };

  const stopPrivilegedPolling = () => {
    if (privilegedPollTimer.current !== null) {
      window.clearInterval(privilegedPollTimer.current);
      privilegedPollTimer.current = null;
    }
    privilegedReadBusy.current = false;
  };

  const stopLatestFrameAttach = () => {
    if (latestFrameAttachTimer.current !== null) {
      window.clearInterval(latestFrameAttachTimer.current);
      latestFrameAttachTimer.current = null;
    }
  };

  const clearPreviewTimeout = () => {
    if (previewTimeoutTimer.current !== null) {
      window.clearTimeout(previewTimeoutTimer.current);
      previewTimeoutTimer.current = null;
    }
  };

  const failPreviewStartup = (requestId: number, message: string) => {
    if (requestId !== previewRequestId.current) return;
    previewRequestId.current += 1;
    clearPreviewTimeout();
    stopMockFrames(mockTimer);
    stopPrivilegedPolling();
    stopLatestFrameAttach();
    setPreviewing(false);
    setPreviewLoading(false);
    setPreviewSession(null);
    setPrivilegedPreview(null);
    pushLog(message);
  };

  const readPreviewFrame = async (framePath: string) => {
    try {
      return await tauriCall<FrameSummary>("read_privileged_preview_frame", { framePath });
    } catch {
      return tauriCall<FrameSummary>("read_latest_privileged_preview_frame");
    }
  };

  const startLatestFrameAttach = () => {
    stopLatestFrameAttach();
    latestFrameAttachTimer.current = window.setInterval(async () => {
      try {
        const frame = await tauriCall<FrameSummary>("read_latest_privileged_preview_frame");
        clearPreviewTimeout();
        setLatestFrame(frame);
        setPreviewLoading(false);
        setPreviewing(true);
        setPreviewSession((current) =>
          current ?? {
            sessionId: frame.sessionId,
            root: "",
            backend: "realsense",
            notice: null,
            progressPath: null
          }
        );
      } catch {
        // Opportunistic attach loop; normal timeout and log path handle failures.
      }
    }, 150);
  };

  const startPrivilegedPolling = (framePath: string) => {
    stopPrivilegedPolling();
    startLatestFrameAttach();
    let misses = 0;
    privilegedPollTimer.current = window.setInterval(async () => {
      if (privilegedReadBusy.current) return;
      privilegedReadBusy.current = true;
      try {
        const frame = await readPreviewFrame(framePath);
        misses = 0;
        clearPreviewTimeout();
        setLatestFrame(frame);
        setPreviewLoading(false);
      } catch {
        misses += 1;
        if (misses === 30) {
          pushLog("waiting for RealSense frames from helper");
        }
      } finally {
        privilegedReadBusy.current = false;
      }
    }, Math.max(16, Math.round(1000 / Math.max(1, config.fps))));
  };

  const startPreview = async () => {
    const requestId = previewRequestId.current + 1;
    previewRequestId.current = requestId;
    clearPreviewTimeout();
    previewTimeoutTimer.current = window.setTimeout(() => {
      failPreviewStartup(
        requestId,
        "RealSense preview timed out: helper opened but no RGB-D frame arrived. Old helpers were cleaned on next start; unplug/replug the camera if this repeats."
      );
    }, 20_000);

    try {
      stopMockFrames(mockTimer);
      stopPrivilegedPolling();
      stopLatestFrameAttach();
      setPreviewing(true);
      setPreviewLoading(true);
      setLatestFrame(null);
      setPrivilegedPreview(null);

      if (!isTauri) {
        const session = {
          sessionId: `browser_preview_${Date.now()}`,
          root: "",
          backend: "synthetic",
          notice: "Browser preview mode",
          progressPath: null
        };
        setPreviewSession(session);
        startMockFrames(session, config, mockTimer, setLatestFrame);
        clearPreviewTimeout();
        setPreviewLoading(false);
        pushLog("browser demo preview started");
        return;
      }

      if (config.backend === "synthetic") {
        const session = await tauriCall<SessionStarted>("start_preview", { config });
        setPreviewSession(session);
        clearPreviewTimeout();
        setPreviewLoading(false);
        pushLog(`demo preview: ${session.sessionId}`);
        if (session.notice) pushLog(session.notice);
        return;
      }

      if (!probe?.sdkLoaded) {
        await setupSdk();
      }

      pushLog("starting RealSense preview helper");
      const started = await tauriCall<PrivilegedPreviewStarted>("start_privileged_preview", {
        config: { ...config, backend: "realsense" }
      });
      if (requestId !== previewRequestId.current) return;
      setPrivilegedPreview(started);
      setPreviewSession({
        sessionId: started.sessionId,
        root: "",
        backend: "realsense",
        notice: null,
        progressPath: null
      });
      startPrivilegedPolling(started.framePath);
      try {
        const firstFrame = await readPreviewFrame(started.framePath);
        if (requestId === previewRequestId.current) {
          clearPreviewTimeout();
          setLatestFrame(firstFrame);
          setPreviewLoading(false);
          pushLog(`RealSense frame received: ${firstFrame.frameIndex}`);
        }
      } catch {
        pushLog("RealSense helper started; waiting for first frame");
      }
      pushLog(`RealSense preview helper started: ${started.launchMode}`);
    } catch (error) {
      clearPreviewTimeout();
      setPreviewing(false);
      setPreviewLoading(false);
      setPreviewSession(null);
      setPrivilegedPreview(null);
      pushLog(`RealSense preview failed: ${String(error)}`);
    }
  };

  const stopPreview = async () => {
    previewRequestId.current += 1;
    clearPreviewTimeout();
    stopMockFrames(mockTimer);
    stopPrivilegedPolling();
    stopLatestFrameAttach();
    setPreviewLoading(false);
    try {
      if (isTauri && privilegedPreview) {
        await tauriCall<void>("stop_privileged_preview", {
          pidPath: privilegedPreview.pidPath,
          launchMode: privilegedPreview.launchMode
        });
        pushLog("RealSense preview helper stopped");
      } else if (isTauri) {
        const stopped = await tauriCall<SessionStopped>("stop_preview");
        pushLog(`preview stopped ${stopped.framesWritten} frames`);
      } else {
        pushLog("browser demo preview stopped");
      }
    } catch (error) {
      pushLog(`preview stop failed: ${String(error)}`);
    } finally {
      setPreviewing(false);
      setPreviewSession(null);
      setPrivilegedPreview(null);
    }
  };

  const stopCapture = async () => {
    setCaptureStopping(true);
    try {
      stopMockFrames(mockTimer);
      const stopped = await tauriCall<SessionStopped>("stop_recording");
      setRecording(false);
      stopRecordingPolling();
      pushLog(`stopped ${stopped.framesWritten} frames`);
    } catch (error) {
      setRecording(false);
      stopRecordingPolling();
      pushLog(`stop failed: ${String(error)}`);
    } finally {
      setCaptureStopping(false);
    }
  };

  const generateAssets = async () => {
    if (!activeSession) {
      pushLog("capture a session before asset generation");
      return;
    }
    setAssetBusy(true);
    pushLog(assetOptions.useMlx ? "building MLX-refined 3DGS, collider, OBJ, and FBX" : "building 3DGS seed, collider, OBJ, and FBX");
    try {
      if (assetOptions.useMlx && !assetTools?.mlxAvailable) {
        pushLog("MLX backend is not ready; setting it up before generation");
        const tools = await setupMlx3dgs();
        if (!tools?.mlxAvailable) {
          throw new Error("MLX 3DGS setup did not complete");
        }
      }
      const result = await tauriCall<AssetBuildResult>("generate_scan_assets", {
        options: {
          sessionRoot: activeSession.root,
          ...assetOptions
        }
      });
      setAssetResult(result);
      pushLog(`assets ready: ${result.pointCount.toLocaleString()} splats`);
      pushLog(result.mlxStatus);
      pushLog(result.collisionStatus);
      pushLog(result.fbxStatus);
    } catch (error) {
      pushLog(`asset generation failed: ${String(error)}`);
    } finally {
      setAssetBusy(false);
    }
  };

  const revealPath = async (path?: string | null) => {
    if (!path) return;
    try {
      await tauriCall("reveal_path", { path });
    } catch (error) {
      pushLog(`open folder failed: ${String(error)}`);
    }
  };

  const chooseSaveLocation = async () => {
    if (!isTauri) {
      setConfig((current) => ({ ...current, outputRoot: "/preview/3dscan" }));
      return;
    }
    try {
      const selected = await openDialog({
        directory: true,
        multiple: false,
        defaultPath: config.outputRoot || undefined,
        title: "Select save location"
      });
      if (typeof selected === "string") {
        setConfig((current) => ({ ...current, outputRoot: selected }));
        pushLog(`save location: ${selected}`);
      }
    } catch (error) {
      pushLog(`save location selection failed: ${String(error)}`);
    }
  };

  const loadPreviewData = async () => {
    if (!isTauri) {
      pushLog("file loading is available in the desktop app");
      return;
    }
    try {
      const selected = await openDialog({
        directory: false,
        multiple: false,
        defaultPath: config.outputRoot || undefined,
        title: "Load 3D scan data",
        filters: [
          { name: "3D scan data", extensions: ["mcap", "mcp", "ply", "splat", "json"] }
        ]
      });
      if (typeof selected !== "string") return;
      setAssetBusy(true);
      pushLog(`loading ${selected}`);
      const result = await tauriCall<AssetBuildResult>("load_scan_data", { path: selected });
      const sessionRoot = result.root.replace(/[\\/]assets$/, "");
      const sessionId = sessionRoot.split(/[\\/]/).filter(Boolean).pop() ?? "loaded_scan";
      setActiveSession({
        sessionId,
        root: sessionRoot,
        backend: "loaded",
        notice: "Loaded scan data",
        progressPath: null
      });
      setAssetResult(result);
      setActiveScreen("preview");
      pushLog(`loaded ${result.pointCount.toLocaleString()} splats`);
    } catch (error) {
      pushLog(`load failed: ${String(error)}`);
    } finally {
      setAssetBusy(false);
    }
  };

  useEffect(() => {
    if (config.outputRoot) {
      window.localStorage.setItem("agriscan.outputRoot", config.outputRoot);
    }
  }, [config.outputRoot]);

  useEffect(() => {
    const boot = async () => {
      if (isTauri && !config.outputRoot) {
        try {
          const saveLocation = await tauriCall<string>("default_save_location");
          setConfig((current) =>
            current.outputRoot ? current : { ...current, outputRoot: saveLocation }
          );
        } catch (error) {
          pushLog(`default save location failed: ${String(error)}`);
        }
      }
      if (isTauri && !helperBootAttempted.current) {
        helperBootAttempted.current = true;
        setHelperInstallBusy(true);
        try {
          const status = await tauriCall<InstalledHelper>("privileged_helper_status");
          setHelperStatus(status);
          if (!status.ready || !status.current) {
            pushLog("one administrator approval prepares RGB-D capture for this app version");
            const result = await tauriCall<InstalledHelper>("ensure_privileged_helper");
            setHelperStatus(result);
            pushLog(result.status);
          } else {
            pushLog(status.status);
          }
        } catch (error) {
          pushLog(`startup helper preparation failed: ${String(error)}`);
        } finally {
          setHelperInstallBusy(false);
        }
      }
      try {
        const restored = await tauriCall<AssetBuildResult | null>("load_latest_scan_assets");
        if (restored) {
          const sessionRoot = restored.root.replace(/[\\/]assets$/, "");
          const sessionId = sessionRoot.split(/[\\/]/).filter(Boolean).pop() ?? "restored_session";
          setActiveSession({
            sessionId,
            root: sessionRoot,
            backend: "restored",
            notice: "Restored generated assets",
            progressPath: null
          });
          setAssetResult(restored);
          pushLog(`restored ${restored.pointCount.toLocaleString()} splats from ${sessionId}`);
        }
      } catch (error) {
        pushLog(`previous asset restore failed: ${String(error)}`);
      }
      await refreshProbe({ autoSetup: true });
    };
    void boot();

    if (!isTauri) return undefined;
    let unlisten: (() => void) | undefined;
    listen<CaptureEvent>("capture-progress", (event) => {
      const payload = event.payload;
      if (payload.kind === "frame" && payload.summary) {
        stopMockFrames(mockTimer);
        clearPreviewTimeout();
        setPreviewLoading(false);
        setLatestFrame(payload.summary);
      } else if (payload.kind === "finished") {
        stopRecordingPolling();
        setRecording(false);
        setPreviewing(false);
        if (payload.message) pushLog(payload.message);
      } else if (payload.kind === "error" && payload.message) {
        pushLog(payload.message);
      }
    }).then((cleanup) => {
      unlisten = cleanup;
    });

    return () => {
      unlisten?.();
      clearPreviewTimeout();
      stopMockFrames(mockTimer);
      stopPrivilegedPolling();
      stopRecordingPolling();
      stopLatestFrameAttach();
      const helper = privilegedPreviewRef.current;
      if (helper && isTauri) {
        void tauriCall<void>("stop_privileged_preview", {
          pidPath: helper.pidPath,
          launchMode: helper.launchMode
        });
      }
    };
  }, []);

  const sdkBadgeVariant = probe?.sdkLoaded ? "success" : "warning";
  const deviceBadgeVariant = devices.length ? "success" : "warning";
  const screenCopy = {
    capture: {
      title: "RGB-D Capture"
    },
    preview: {
      title: "3D Preview"
    },
    settings: {
      title: "Settings"
    }
  }[activeScreen];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <aside className="fixed inset-y-0 left-0 z-30 flex w-[92px] flex-col items-center border-r border-white/10 bg-[#0d3528] px-2.5 py-5 text-white shadow-2xl shadow-emerald-950/10">
        <nav className="mt-1 flex w-full flex-col gap-3" aria-label="Primary navigation">
          <SidebarButton
            label="Capture"
            icon={<ScanLine className="h-6 w-6" />}
            active={activeScreen === "capture"}
            onClick={() => setActiveScreen("capture")}
          />
          <SidebarButton
            label="3D Preview"
            icon={<Box className="h-6 w-6" />}
            active={activeScreen === "preview"}
            onClick={() => setActiveScreen("preview")}
          />
          <SidebarButton
            label="Settings"
            icon={<SlidersHorizontal className="h-6 w-6" />}
            active={activeScreen === "settings"}
            onClick={() => setActiveScreen("settings")}
          />
        </nav>
      </aside>

      <div className="min-h-screen pl-[92px]">
        <header className="sticky top-0 z-20 border-b border-black/[0.06] bg-background/92 backdrop-blur-xl">
          <div className="flex min-h-[70px] items-center gap-5 px-6 2xl:px-8">
            <div className="min-w-0 flex-1">
              <h1 className="text-[22px] font-semibold tracking-[-0.025em]">{screenCopy.title}</h1>
            </div>

            <div className="hidden flex-wrap items-center justify-end gap-2 md:flex">
              <Badge variant={sdkBadgeVariant}>
                <Cpu className="h-3.5 w-3.5" />
                {probe?.sdkLoaded ? `SDK ${probe.apiVersion ?? ""}` : "SDK missing"}
              </Badge>
              <Badge variant={deviceBadgeVariant}>
                <Camera className="h-3.5 w-3.5" />
                {devices.length ? `${devices.length} device` : "No device"}
              </Badge>
              <Badge variant={recording || previewing ? "live" : "outline"}>
                {recording || previewing ? <Radio className="h-3.5 w-3.5" /> : <Circle className="h-3.5 w-3.5" />}
                {recording ? "Recording" : previewing ? "Live" : "Idle"}
              </Badge>
            </div>

            <Button
              size="icon"
              variant="outline"
              className="rounded-xl bg-white"
              onClick={() => refreshProbe()}
              disabled={sdkSetupBusy || mlxSetupBusy || probeBusy}
              title="Refresh devices"
            >
              <RefreshCw className={cn("h-4 w-4", (sdkSetupBusy || mlxSetupBusy || probeBusy) && "animate-spin")} />
            </Button>
          </div>
        </header>

        {busyMessage ? (
          <div className="border-b border-amber-200/70 bg-amber-50 px-6 py-2.5 text-sm font-medium text-amber-950">
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{busyMessage}</span>
            </div>
          </div>
        ) : null}

        <main className="p-5 2xl:p-7">
          {activeScreen === "capture" ? (
            <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
              <section className="min-w-0 space-y-5">
                <LiveFramePanel
                  latestFrame={latestFrame}
                  activeSession={previewing ? previewSession : activeSession}
                  previewing={previewing}
                  recording={recording}
                  loadingMessage={previewLoading ? busyMessage : null}
                />
                <CaptureStatusCards
                  activeSession={previewing ? previewSession : activeSession}
                  latestFrame={latestFrame}
                  backend={backend}
                  deviceCount={devices.length}
                  log={log}
                  revealSession={() => revealPath(activeSession?.root)}
                />
              </section>
              <CaptureCommandPanel
                config={config}
                setConfig={setConfig}
                backend={backend}
                recording={recording}
                previewing={previewing}
                previewLoading={previewLoading}
                captureStarting={captureStarting}
                captureStopping={captureStopping}
                activeSession={activeSession}
                latestFrame={latestFrame}
                helperReady={helperReady}
                startPreview={startPreview}
                stopPreview={stopPreview}
                startCapture={startCapture}
                stopCapture={stopCapture}
                chooseSaveLocation={chooseSaveLocation}
              />
            </div>
          ) : null}

          {activeScreen === "preview" ? (
            <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
              <AssetPreviewPanel
                assetResult={assetResult}
                assetBusy={assetBusy}
                loadPreviewData={loadPreviewData}
              />
              <AssetCommandPanel
                assetResult={assetResult}
                assetBusy={assetBusy}
                activeSession={activeSession}
                assetOptions={assetOptions}
                setAssetOptions={setAssetOptions}
                recording={recording}
                generateAssets={generateAssets}
                revealAssets={() => revealPath(assetResult?.root)}
              />
            </div>
          ) : null}

          {activeScreen === "settings" ? (
            <div className="grid items-start gap-5 xl:grid-cols-[minmax(380px,0.85fr)_minmax(480px,1.15fr)]">
              <ControlPanel
                config={config}
                setConfig={setConfig}
                assetOptions={assetOptions}
                setAssetOptions={setAssetOptions}
                backend={backend}
                recording={recording}
                previewing={previewing}
                previewLoading={previewLoading}
                captureStarting={captureStarting}
                captureStopping={captureStopping}
                assetBusy={assetBusy}
                activeSession={activeSession}
                previewSession={previewSession}
                startPreview={startPreview}
                stopPreview={stopPreview}
                startCapture={startCapture}
                stopCapture={stopCapture}
                generateAssets={generateAssets}
                assetTools={assetTools}
                helperReady={helperReady}
                chooseSaveLocation={chooseSaveLocation}
                settingsOnly
              />
              <OutputPanel
                activeSession={activeSession}
                latestFrame={latestFrame}
                devices={devices}
                probe={probe}
                assetResult={assetResult}
                log={log}
                setupSdk={setupSdk}
                setupMlx3dgs={setupMlx3dgs}
                installHelper={installHelper}
                sdkSetupBusy={sdkSetupBusy}
                mlxSetupBusy={mlxSetupBusy}
                helperInstallBusy={helperInstallBusy}
                helperStatus={helperStatus}
                recording={recording}
                revealSession={() => revealPath(activeSession?.root)}
              />
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

function SidebarButton({
  label,
  icon,
  active,
  onClick
}: {
  label: string;
  icon: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      aria-current={active ? "page" : undefined}
      onClick={onClick}
      className={cn(
        "group relative flex min-h-[76px] w-full flex-col items-center justify-center gap-2 rounded-2xl px-1 text-[10px] font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80",
        active
          ? "bg-white/16 text-white shadow-[inset_0_0_0_1px_rgb(255_255_255/0.18),0_12px_30px_rgb(0_0_0/0.14)]"
          : "text-white/62 hover:bg-white/8 hover:text-white"
      )}
    >
      <span
        className={cn(
          "absolute -left-2.5 h-8 w-1 rounded-r-full bg-[#a6d89a] transition-opacity",
          active ? "opacity-100" : "opacity-0"
        )}
      />
      {icon}
      <span>{label}</span>
    </button>
  );
}

function CaptureCommandPanel(props: {
  config: CaptureConfig;
  setConfig: React.Dispatch<React.SetStateAction<CaptureConfig>>;
  backend: string;
  recording: boolean;
  previewing: boolean;
  previewLoading: boolean;
  captureStarting: boolean;
  captureStopping: boolean;
  activeSession: SessionStarted | null;
  latestFrame: FrameSummary | null;
  helperReady: boolean;
  startPreview: () => void;
  stopPreview: () => void;
  startCapture: () => void;
  stopCapture: () => void;
  chooseSaveLocation: () => void;
}) {
  const disabled = props.recording || props.previewing || props.captureStarting || props.captureStopping;
  const selectedProfileIndex = CAPTURE_PROFILES.findIndex(
    (profile) =>
      profile.width === props.config.width &&
      profile.height === props.config.height &&
      profile.fps === props.config.fps
  );
  const selectedProfileValue = selectedProfileIndex >= 0 ? String(selectedProfileIndex) : "custom";
  const updateConfig = <K extends keyof CaptureConfig>(key: K, value: CaptureConfig[K]) => {
    props.setConfig((current) => ({ ...current, [key]: value }));
  };
  const applyProfile = (value: string) => {
    const profile = CAPTURE_PROFILES[Number(value)] ?? CAPTURE_PROFILES[0];
    props.setConfig((current) => ({
      ...current,
      width: profile.width,
      height: profile.height,
      fps: profile.fps
    }));
  };
  const recordDisabled =
    props.recording ||
    props.previewing ||
    props.captureStarting ||
    props.captureStopping ||
    !props.helperReady;

  return (
    <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-[0_18px_55px_rgb(24_53_40/0.08)] xl:sticky xl:top-[105px]">
      <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] px-5 py-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Capture Control</CardTitle>
            <CardDescription className="mt-1">Backend: {props.backend}</CardDescription>
          </div>
          <Badge variant={props.recording || props.previewing ? "live" : "outline"}>
            {props.recording || props.previewing ? <Radio className="h-3.5 w-3.5" /> : <Circle className="h-3.5 w-3.5" />}
            {props.recording ? "Recording" : props.previewing ? "Live" : "Ready"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 p-5">
        <div className="grid grid-cols-3 gap-3">
          <Button
            variant="outline"
            className="h-11 rounded-xl bg-white"
            onClick={props.previewing ? props.stopPreview : props.startPreview}
            disabled={props.recording || props.captureStarting || props.captureStopping || !props.helperReady}
          >
            {props.previewLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
            {props.previewLoading ? "Opening" : props.previewing ? "Stop Live" : "Live Preview"}
          </Button>
          <Button
            className="h-11 rounded-xl"
            onClick={props.startCapture}
            disabled={recordDisabled}
            aria-label="Record RGB-D"
          >
            {props.captureStarting ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
            {props.captureStarting ? "Opening" : "Record"}
          </Button>
          <Button
            variant="destructive"
            className="h-11 rounded-xl"
            onClick={props.stopCapture}
            disabled={!props.recording || props.captureStopping}
          >
            {props.captureStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
            {props.captureStopping ? "Stopping" : "Stop"}
          </Button>
        </div>

        <div className="h-px bg-border/70" />

        <Field label="Capture profile">
          <Select value={selectedProfileValue} disabled={disabled} onChange={(event) => applyProfile(event.target.value)}>
            {selectedProfileIndex < 0 ? (
              <option value="custom">
                Custom: {props.config.width} x {props.config.height} / {props.config.fps} fps
              </option>
            ) : null}
            {CAPTURE_PROFILES.map((profile, index) => (
              <option key={`${profile.width}-${profile.height}-${profile.fps}`} value={index}>
                {profile.label}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="File name">
          <Input
            value={props.config.targetLabel}
            placeholder="scan"
            disabled={disabled}
            onChange={(event) => updateConfig("targetLabel", event.target.value)}
          />
        </Field>
        <Field label="Save location">
          <div className="flex gap-2">
            <Input value={props.config.outputRoot} placeholder="Select a folder" readOnly />
            <Button
              type="button"
              size="icon"
              variant="outline"
              disabled={disabled}
              onClick={props.chooseSaveLocation}
              title="Select save location"
              aria-label="Select save location"
            >
              <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Max frames"
            value={props.config.maxFrames ?? 0}
            min={0}
            max={100000}
            disabled={disabled}
            onChange={(value) => updateConfig("maxFrames", value > 0 ? value : null)}
          />
          <Field label="Mode">
            <Select
              value={props.config.backend}
              disabled={disabled}
              onChange={(event) => updateConfig("backend", event.target.value as BackendMode)}
            >
              <option value="auto">Auto</option>
              <option value="realsense">RealSense</option>
              <option value="synthetic">Demo</option>
            </Select>
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-3 rounded-xl border bg-muted/25 p-3">
          <div>
            <div className="text-[11px] font-medium text-muted-foreground">Frames</div>
            <div className="mt-1 text-xl font-semibold tabular-nums">{props.latestFrame?.frameIndex ?? 0}</div>
          </div>
          <div className="border-l pl-3">
            <div className="text-[11px] font-medium text-muted-foreground">Session</div>
            <div className="mt-1 truncate text-sm font-semibold">{props.activeSession?.sessionId ?? "Not started"}</div>
          </div>
        </div>
        {!props.helperReady ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-950">
            Capture helper is not ready. Open Settings to prepare the device once.
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function CaptureStatusCards({
  activeSession,
  latestFrame,
  backend,
  deviceCount,
  log,
  revealSession
}: {
  activeSession: SessionStarted | null;
  latestFrame: FrameSummary | null;
  backend: string;
  deviceCount: number;
  log: string[];
  revealSession: () => void;
}) {
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="rounded-2xl border-black/[0.06] shadow-none">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            Session Status
          </div>
          <dl className="mt-4 space-y-2.5 text-xs">
            <StatusRow label="Backend" value={backend} />
            <StatusRow label="Device" value={deviceCount ? `${deviceCount} connected` : "Not connected"} />
            <StatusRow label="Frame no." value={latestFrame ? String(latestFrame.frameNumber) : "0"} />
          </dl>
        </CardContent>
      </Card>
      <Card className="rounded-2xl border-black/[0.06] shadow-none">
        <CardContent className="p-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <HardDrive className="h-4 w-4 text-primary" />
              MCAP Recording
            </div>
            <Button size="icon" variant="ghost" className="h-8 w-8" disabled={!activeSession?.root} onClick={revealSession}>
              <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
          <div className="mt-4 rounded-lg bg-muted/40 px-3 py-2">
            <div className="text-[11px] font-medium text-muted-foreground">Session path</div>
            <div className="mt-1 truncate text-xs" title={activeSession?.root ?? ""}>
              {shortPath(activeSession?.root ?? "-")}
            </div>
          </div>
          <div className="mt-3 text-xs text-muted-foreground">
            {latestFrame?.depth.validPoints.toLocaleString() ?? "0"} valid depth points in the latest frame
          </div>
        </CardContent>
      </Card>
      <Card className="rounded-2xl border-black/[0.06] shadow-none">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Activity className="h-4 w-4 text-primary" />
            Recent Activity
          </div>
          <ol className="mt-4 space-y-2.5">
            {log.slice(0, 3).map((line, index) => (
              <li key={`${line}-${index}`} className="flex gap-2 text-[11px] leading-4 text-muted-foreground">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                <span className="line-clamp-2">{line}</span>
              </li>
            ))}
            {!log.length ? <li className="text-xs text-muted-foreground">Waiting for activity</li> : null}
          </ol>
        </CardContent>
      </Card>
    </div>
  );
}

function AssetCommandPanel(props: {
  assetResult: AssetBuildResult | null;
  assetBusy: boolean;
  activeSession: SessionStarted | null;
  assetOptions: AssetOptions;
  setAssetOptions: React.Dispatch<React.SetStateAction<AssetOptions>>;
  recording: boolean;
  generateAssets: () => void;
  revealAssets: () => void;
}) {
  const updateAsset = <K extends keyof AssetOptions>(key: K, value: AssetOptions[K]) => {
    props.setAssetOptions((current) => ({ ...current, [key]: value }));
  };
  return (
    <aside className="space-y-5 xl:sticky xl:top-[105px]">
      <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-[0_18px_55px_rgb(24_53_40/0.08)]">
        <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] px-5 py-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle>Build Assets</CardTitle>
            </div>
            <WandSparkles className="h-5 w-5 text-primary" />
          </div>
        </CardHeader>
        <CardContent className="space-y-5 p-5">
          <div className="grid grid-cols-2 gap-3">
            <NumberField
              label="Max splats"
              value={props.assetOptions.maxPoints}
              min={5000}
              max={1500000}
              step={1000}
              onChange={(value) => updateAsset("maxPoints", value)}
            />
            <NumberField
              label="Frame step"
              value={props.assetOptions.frameStride}
              min={1}
              max={24}
              onChange={(value) => updateAsset("frameStride", value)}
            />
          </div>
          <label className="flex items-center justify-between rounded-xl border bg-muted/25 p-3 text-sm">
            <span className="font-medium">MLX refinement</span>
            <input
              className="h-4 w-4 rounded border-input accent-emerald-700"
              type="checkbox"
              checked={props.assetOptions.useMlx}
              onChange={(event) => updateAsset("useMlx", event.target.checked)}
            />
          </label>
          <label className="flex items-center justify-between rounded-xl border bg-muted/25 p-3 text-sm">
            <span className="font-medium">Native FBX export</span>
            <input
              className="h-4 w-4 rounded border-input accent-emerald-700"
              type="checkbox"
              checked={props.assetOptions.exportFbx}
              onChange={(event) => updateAsset("exportFbx", event.target.checked)}
            />
          </label>
          <Button
            className="h-12 w-full rounded-xl"
            onClick={props.generateAssets}
            disabled={!props.activeSession || props.recording || props.assetBusy}
          >
            {props.assetBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Box className="h-4 w-4" />}
            {props.assetBusy ? "Generating assets" : "Generate assets"}
          </Button>
        </CardContent>
      </Card>

      <Card className="rounded-2xl border-black/[0.06] shadow-none">
        <CardHeader className="border-b border-black/[0.06] pb-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle>Asset Summary</CardTitle>
            </div>
            <Button
              size="icon"
              variant="outline"
              disabled={!props.assetResult}
              onClick={props.revealAssets}
              title="Open assets folder"
              aria-label="Open assets folder"
            >
              <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3 pt-4">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Gaussians" value={props.assetResult?.pointCount.toLocaleString() ?? "0"} />
            <Stat label="Mesh faces" value={props.assetResult?.faceCount.toLocaleString() ?? "0"} />
          </div>
          <PathRow label="3DGS PLY" value={props.assetResult?.gaussianPly ?? "-"} />
          <PathRow label="FBX" value={props.assetResult?.meshFbx ?? props.assetResult?.fbxStatus ?? "-"} />
          <PathRow label="Collider" value={props.assetResult?.colliderObj ?? "-"} />
        </CardContent>
      </Card>
    </aside>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="truncate font-medium">{value}</dd>
    </div>
  );
}

function ControlPanel(props: {
  config: CaptureConfig;
  setConfig: React.Dispatch<React.SetStateAction<CaptureConfig>>;
  assetOptions: AssetOptions;
  setAssetOptions: React.Dispatch<React.SetStateAction<AssetOptions>>;
  backend: string;
  recording: boolean;
  previewing: boolean;
  previewLoading: boolean;
  captureStarting: boolean;
  captureStopping: boolean;
  assetBusy: boolean;
  activeSession: SessionStarted | null;
  previewSession: SessionStarted | null;
  startPreview: () => void;
  stopPreview: () => void;
  startCapture: () => void;
  stopCapture: () => void;
  generateAssets: () => void;
  assetTools: AssetTools | null;
  helperReady: boolean;
  chooseSaveLocation: () => void;
  settingsOnly?: boolean;
}) {
  const disabled = props.recording || props.previewing || props.captureStarting || props.captureStopping;
  const selectedProfileIndex = CAPTURE_PROFILES.findIndex(
    (profile) => profile.width === props.config.width && profile.height === props.config.height && profile.fps === props.config.fps
  );
  const selectedProfileValue = selectedProfileIndex >= 0 ? String(selectedProfileIndex) : "custom";

  const updateConfig = <K extends keyof CaptureConfig>(key: K, value: CaptureConfig[K]) => {
    props.setConfig((current) => ({ ...current, [key]: value }));
  };
  const updateAsset = <K extends keyof AssetOptions>(key: K, value: AssetOptions[K]) => {
    props.setAssetOptions((current) => ({ ...current, [key]: value }));
  };
  const applyProfile = (value: string) => {
    const profile = CAPTURE_PROFILES[Number(value)] ?? CAPTURE_PROFILES[0];
    props.setConfig((current) => ({
      ...current,
      width: profile.width,
      height: profile.height,
      fps: profile.fps
    }));
  };

  return (
    <Card className="h-fit overflow-hidden rounded-2xl border-black/[0.07] shadow-[0_18px_55px_rgb(24_53_40/0.07)]">
      <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] px-5 py-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>{props.settingsOnly ? "Capture & Reconstruction" : "Capture"}</CardTitle>
            <CardDescription className="mt-1">Backend: {props.backend}</CardDescription>
          </div>
          <Badge variant="secondary">
            {props.config.width}x{props.config.height} / {props.config.fps}fps
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 p-5">
        <Field label="Capture profile">
          <Select value={selectedProfileValue} disabled={disabled} onChange={(event) => applyProfile(event.target.value)}>
            {selectedProfileIndex < 0 ? <option value="custom">Custom: {props.config.width} x {props.config.height} / {props.config.fps} fps</option> : null}
            {CAPTURE_PROFILES.map((profile, index) => (
              <option key={`${profile.width}-${profile.height}-${profile.fps}`} value={index}>
                {profile.label}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Backend">
          <div className="grid grid-cols-3 rounded-md border bg-muted p-1">
            {(["auto", "realsense", "synthetic"] as BackendMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                disabled={disabled}
                className={cn(
                  "h-8 rounded-sm text-xs font-medium text-muted-foreground transition-colors",
                  props.config.backend === mode && "bg-background text-foreground shadow-sm"
                )}
                onClick={() => updateConfig("backend", mode)}
              >
                {mode === "synthetic" ? "Demo" : mode === "realsense" ? "RealSense" : "Auto"}
              </button>
            ))}
          </div>
        </Field>

        <div className="grid grid-cols-2 gap-3">
          <NumberField label="Point stride" value={props.config.pointStride} min={1} max={12} disabled={disabled} onChange={(v) => updateConfig("pointStride", v)} />
          <NumberField label="Min depth" value={props.config.minDepthM} min={0.02} max={4} step={0.01} disabled={disabled} onChange={(v) => updateConfig("minDepthM", v)} />
          <NumberField label="Max depth" value={props.config.maxDepthM} min={0.03} max={8} step={0.01} disabled={disabled} onChange={(v) => updateConfig("maxDepthM", v)} />
        </div>

        <Field label="File name">
          <Input
            value={props.config.targetLabel}
            placeholder="scan"
            disabled={disabled}
            onChange={(event) => updateConfig("targetLabel", event.target.value)}
          />
        </Field>
        <Field label="Save location">
          <div className="flex gap-2">
            <Input value={props.config.outputRoot} placeholder="Select a folder" readOnly />
            <Button
              type="button"
              size="icon"
              variant="outline"
              disabled={disabled}
              onClick={props.chooseSaveLocation}
              title="Select save location"
              aria-label="Select save location"
            >
              <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
        </Field>
        <Field label="Cultivar">
          <Input value={props.config.cultivar} placeholder="optional" disabled={disabled} onChange={(event) => updateConfig("cultivar", event.target.value)} />
        </Field>
        <Field label="Max frames">
          <Input
            type="number"
            min={1}
            value={props.config.maxFrames ?? ""}
            placeholder="unlimited"
            disabled={disabled}
            onChange={(event) => updateConfig("maxFrames", parseNullableNumber(event.target.value))}
          />
        </Field>
        <Field label="Notes">
          <Textarea value={props.config.notes} disabled={disabled} onChange={(event) => updateConfig("notes", event.target.value)} />
        </Field>

        {!props.settingsOnly ? (
          <div className="grid grid-cols-2 gap-3">
            <Button
              variant="secondary"
              onClick={props.previewing ? props.stopPreview : props.startPreview}
              disabled={props.recording || props.captureStarting || props.captureStopping || !props.helperReady}
            >
              {props.previewLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
              {props.previewLoading ? "Loading Preview" : props.previewing ? "Stop Live" : "Live Preview"}
            </Button>
            <Button
              onClick={props.startCapture}
              disabled={props.recording || props.previewing || props.captureStarting || !props.helperReady}
            >
              {props.captureStarting ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
              {props.captureStarting ? "Loading Record" : "Record RGB-D"}
            </Button>
            <Button className="col-span-2" variant="destructive" onClick={props.stopCapture} disabled={!props.recording || props.captureStopping}>
              {props.captureStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
              {props.captureStopping ? "Loading Stop" : "Stop Recording"}
            </Button>
          </div>
        ) : null}

        <div className="rounded-lg border bg-muted/35 p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold">3DGS / FBX</h3>
              <p className="text-xs text-muted-foreground">
                {props.assetOptions.useMlx
                  ? props.assetTools?.mlxStatus ?? "MLX setup runs automatically"
                  : props.assetTools?.fbxExporter ?? "Built-in native FBX"}
              </p>
            </div>
            <WandSparkles className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Frame step" value={props.assetOptions.frameStride} min={1} max={24} onChange={(v) => updateAsset("frameStride", v)} />
            <NumberField label="Depth step" value={props.assetOptions.depthDecimation} min={1} max={16} onChange={(v) => updateAsset("depthDecimation", v)} />
            <NumberField label="Max splats" value={props.assetOptions.maxPoints} min={5000} max={1500000} step={1000} onChange={(v) => updateAsset("maxPoints", v)} />
            <NumberField label="Radius m" value={props.assetOptions.gaussianRadiusM} min={0.0005} max={0.05} step={0.0005} onChange={(v) => updateAsset("gaussianRadiusM", v)} />
            <NumberField label="MLX iters" value={props.assetOptions.mlxIterations} min={0} max={20000} step={100} onChange={(v) => updateAsset("mlxIterations", v)} />
            <NumberField label="Voxel m" value={props.assetOptions.mlxVoxelSizeM} min={0.0005} max={0.05} step={0.0005} onChange={(v) => updateAsset("mlxVoxelSizeM", v)} />
            <NumberField label="Train px" value={props.assetOptions.mlxTrainSize} min={64} max={1024} step={32} onChange={(v) => updateAsset("mlxTrainSize", v)} />
            <NumberField label="Train views" value={props.assetOptions.mlxMaxTrainViews} min={1} max={64} onChange={(v) => updateAsset("mlxMaxTrainViews", v)} />
            <NumberField label="Turntable" value={props.assetOptions.turntableDegrees} min={0} max={1080} onChange={(v) => updateAsset("turntableDegrees", v)} />
            <NumberField label="Collider faces" value={props.assetOptions.colliderMaxFaces} min={500} max={120000} step={500} onChange={(v) => updateAsset("colliderMaxFaces", v)} />
            <label className="flex h-[58px] items-end gap-2 pb-2 text-sm">
              <input
                className="h-4 w-4 rounded border-input"
                type="checkbox"
                checked={props.assetOptions.useMlx}
                onChange={(event) => updateAsset("useMlx", event.target.checked)}
              />
              MLX refine
            </label>
            <label className="flex h-[58px] items-end gap-2 pb-2 text-sm">
              <input
                className="h-4 w-4 rounded border-input"
                type="checkbox"
                checked={props.assetOptions.exportFbx}
                onChange={(event) => updateAsset("exportFbx", event.target.checked)}
              />
              Export FBX
            </label>
          </div>
          {!props.settingsOnly ? (
            <Button className="mt-3 w-full" variant="secondary" onClick={props.generateAssets} disabled={!props.activeSession || props.recording || props.assetBusy}>
              {props.assetBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Box className="h-4 w-4" />}
              {props.assetBusy ? "Generating" : "Generate assets"}
            </Button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function LiveFramePanel({
  latestFrame,
  activeSession,
  previewing,
  recording,
  loadingMessage
}: {
  latestFrame: FrameSummary | null;
  activeSession: SessionStarted | null;
  previewing: boolean;
  recording: boolean;
  loadingMessage: string | null;
}) {
  return (
    <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-[0_18px_55px_rgb(24_53_40/0.07)]">
      <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] px-5 py-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Live RGB-D</CardTitle>
            <CardDescription className="mt-1">
              {recording ? "recording" : previewing ? "previewing" : "idle"} / {activeSession?.sessionId ?? "no session"}
            </CardDescription>
          </div>
          <div className="rounded-xl border bg-white px-3 py-2 text-right shadow-sm">
            <div className="text-lg font-semibold leading-none">{latestFrame?.frameIndex ?? 0}</div>
            <div className="text-xs text-muted-foreground">frames</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 p-5">
        <div className="relative">
          <div className="grid gap-3 lg:grid-cols-2">
            <PreviewPane
              label="RGB"
              src={latestFrame?.colorPreviewDataUrl ?? null}
              icon={<Camera className="h-7 w-7" />}
              active={previewing || recording}
            />
            <PreviewPane
              label="Depth"
              src={latestFrame?.depthPreviewDataUrl ?? null}
              icon={<ScanLine className="h-7 w-7" />}
              active={previewing || recording}
            />
          </div>
          {loadingMessage ? (
            <div className="absolute inset-0 grid place-items-center rounded-lg border bg-background/80 backdrop-blur-sm">
              <div className="flex items-center gap-3 rounded-md border bg-background px-4 py-3 shadow-sm">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                <div>
                  <div className="text-sm font-semibold">Loading</div>
                  <div className="text-xs text-muted-foreground">{loadingMessage}</div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Valid points" value={latestFrame ? latestFrame.depth.validPoints.toLocaleString() : "0"} />
          <Stat label="Mean depth" value={latestFrame ? `${latestFrame.depth.meanM.toFixed(3)} m` : "0.000 m"} />
          <Stat label="Range" value={latestFrame ? `${latestFrame.depth.minM.toFixed(3)}-${latestFrame.depth.maxM.toFixed(3)} m` : "0.000-0.000 m"} />
          <Stat label="Frame no." value={latestFrame ? String(latestFrame.frameNumber) : "0"} />
        </div>
      </CardContent>
    </Card>
  );
}

function AssetPreviewPanel({
  assetResult,
  assetBusy,
  loadPreviewData
}: {
  assetResult: AssetBuildResult | null;
  assetBusy: boolean;
  loadPreviewData: () => void;
}) {
  return (
    <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-[0_18px_55px_rgb(24_53_40/0.07)]">
      <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] px-5 py-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>3DGS Preview</CardTitle>
          </div>
          <div className="flex items-center gap-3">
            {assetResult ? (
              <span className="text-xs font-medium tabular-nums text-muted-foreground">
                {assetResult.pointCount.toLocaleString()} splats · {assetResult.faceCount.toLocaleString()} faces
              </span>
            ) : null}
            <Button
              size="icon"
              variant="outline"
              disabled={assetBusy}
              onClick={loadPreviewData}
              title="Load 3D scan data"
              aria-label="Load 3D scan data"
            >
            <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-5">
        <div className="relative">
          <SplatCanvas payload={assetResult?.preview ?? null} />
          {assetBusy ? (
            <div className="absolute inset-0 grid place-items-center rounded-lg border bg-background/80 backdrop-blur-sm">
              <div className="flex items-center gap-3 rounded-md border bg-background px-4 py-3 shadow-sm">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                <div>
                  <div className="text-sm font-semibold">Loading</div>
                  <div className="text-xs text-muted-foreground">Generating MLX 3DGS, collision, OBJ, and FBX</div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function OutputPanel(props: {
  activeSession: SessionStarted | null;
  latestFrame: FrameSummary | null;
  devices: CameraDevice[];
  probe: RuntimeProbe | null;
  assetResult: AssetBuildResult | null;
  log: string[];
  setupSdk: () => void;
  setupMlx3dgs: () => void;
  installHelper: () => void;
  sdkSetupBusy: boolean;
  mlxSetupBusy: boolean;
  helperInstallBusy: boolean;
  helperStatus: InstalledHelper | null;
  recording: boolean;
  revealSession: () => void;
}) {
  return (
    <aside className="space-y-5 xl:h-fit">
      <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-none">
        <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] pb-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle>MCAP Recording</CardTitle>
            </div>
            <Button size="icon" variant="outline" disabled={!props.activeSession} onClick={props.revealSession}>
              <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-2 pt-4">
          <PathRow label="Root" value={props.activeSession?.root ?? "-"} />
          <PathRow label="RGB topic" value={props.latestFrame?.paths.rgb ?? "-"} />
          <PathRow label="Depth topic" value={props.latestFrame?.paths.depth ?? "-"} />
          <PathRow label="Point cloud topic" value={props.latestFrame?.paths.pointCloud ?? "-"} />
          <PathRow label="Frame info topic" value={props.latestFrame?.paths.metadata ?? "-"} />
        </CardContent>
      </Card>

      <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-none">
        <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] pb-4">
          <CardTitle>Assets</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 pt-4">
          <PathRow label="Seed PLY" value={props.assetResult?.seedGaussianPly ?? "-"} />
          <PathRow label="3DGS PLY" value={props.assetResult?.gaussianPly ?? "-"} />
          <PathRow label=".splat" value={props.assetResult?.splat ?? "-"} />
          <PathRow label="OBJ" value={props.assetResult?.meshObj ?? "-"} />
          <PathRow label="FBX" value={props.assetResult?.meshFbx ?? props.assetResult?.fbxStatus ?? "-"} />
          <PathRow label="Collider OBJ" value={props.assetResult?.colliderObj ?? "-"} />
          <PathRow label="Collision JSON" value={props.assetResult?.collisionJson ?? "-"} />
          <PathRow label="Collision FBX" value={props.assetResult?.collisionFbx ?? props.assetResult?.fbxStatus ?? "-"} />
          <PathRow label="Preview" value={props.assetResult?.previewJson ?? "-"} />
        </CardContent>
      </Card>

      <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-none">
        <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] pb-4">
          <CardTitle>Device</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-4">
          <Button className="w-full" variant="secondary" onClick={props.setupSdk} disabled={props.sdkSetupBusy || props.recording}>
            <Download className={cn("h-4 w-4", props.sdkSetupBusy && "animate-bounce")} />
            {props.sdkSetupBusy ? "Setting up" : "Setup SDK"}
          </Button>
          <Button className="w-full" variant="secondary" onClick={props.setupMlx3dgs} disabled={props.mlxSetupBusy || props.recording}>
            <WandSparkles className={cn("h-4 w-4", props.mlxSetupBusy && "animate-pulse")} />
            {props.mlxSetupBusy ? "Installing 3DGS" : "Setup MLX 3DGS"}
          </Button>
          <Button
            className="w-full"
            variant="outline"
            onClick={props.installHelper}
            disabled={props.helperInstallBusy || props.recording || Boolean(props.helperStatus?.ready && props.helperStatus.current)}
          >
            <Cpu className={cn("h-4 w-4", props.helperInstallBusy && "animate-pulse")} />
            {props.helperInstallBusy
              ? "Preparing helper"
              : props.helperStatus?.ready && props.helperStatus.current
                ? "Capture helper ready"
                : "Prepare capture helper"}
          </Button>
          {props.devices.length ? (
            props.devices.map((device) => (
              <div key={`${device.serial}-${device.name}`} className="rounded-md border p-3">
                <div className="text-sm font-medium">{device.name || "RealSense"}</div>
                <div className="mt-1 truncate text-xs text-muted-foreground">
                  {[device.serial, device.usb, device.productLine].filter(Boolean).join(" / ")}
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">{props.probe?.status ?? props.probe?.installHint ?? "No device"}</p>
          )}
          {props.probe?.usbDevices?.length ? (
            <div className="space-y-2">
              {props.probe.usbDevices.map((device) => {
                const slow = (device.linkSpeedMbps ?? 0) < 5000;
                return (
                  <div
                    key={`${device.productName}-${device.locationId ?? ""}`}
                    className={cn(
                      "rounded-md border p-3",
                      slow ? "border-amber-200 bg-amber-50 text-amber-950" : "border-emerald-200 bg-emerald-50 text-emerald-950"
                    )}
                  >
                    <div className="text-sm font-medium">{device.productName}</div>
                    <div className="mt-1 text-xs">
                      USB {device.usbType ?? "unknown"} / {device.linkSpeedMbps ?? "unknown"} Mbps
                    </div>
                    {slow ? <div className="mt-2 text-xs font-medium">Current link is below USB3; RGB-D streaming will not open reliably.</div> : null}
                  </div>
                );
              })}
            </div>
          ) : null}
          {props.probe?.actionRequired ? (
            <div className="rounded-md border border-destructive/25 bg-destructive/10 p-3 text-xs leading-5 text-destructive">
              {props.probe.actionRequired}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card className="overflow-hidden rounded-2xl border-black/[0.07] shadow-none">
        <CardHeader className="border-b border-black/[0.06] bg-[#fbfcf9] pb-4">
          <CardTitle>Log</CardTitle>
        </CardHeader>
        <CardContent className="pt-4">
          <ol className="space-y-2">
            {props.log.length ? (
              props.log.map((line, index) => (
                <li key={`${line}-${index}`} className="text-xs leading-5 text-muted-foreground">
                  {line}
                </li>
              ))
            ) : (
              <li className="text-xs text-muted-foreground">Waiting for activity</li>
            )}
          </ol>
        </CardContent>
      </Card>
    </aside>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-2">
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={props.label}>
      <Input
        type="number"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        disabled={props.disabled}
        onChange={(event) => props.onChange(parseNumber(event.target.value, props.value))}
      />
    </Field>
  );
}

function PreviewPane({
  label,
  src,
  icon,
  active
}: {
  label: string;
  src: string | null;
  icon: React.ReactNode;
  active: boolean;
}) {
  return (
    <figure className="relative overflow-hidden rounded-2xl border border-black/10 bg-[#111815] shadow-inner">
      <div className="grid aspect-[4/3] place-items-center">
        {src ? (
          <img key={src.slice(0, 96)} src={src} alt={`${label} preview`} className="h-full w-full object-contain" />
        ) : (
          <div className="grid place-items-center gap-2 text-zinc-400">
            {icon}
            <span className="text-sm font-medium">{label}</span>
          </div>
        )}
      </div>
      <figcaption className="absolute left-3 top-3 flex items-center gap-2 rounded-lg bg-black/60 px-2.5 py-1.5 text-xs font-semibold text-white backdrop-blur">
        {label}
        <span className={cn("flex items-center gap-1 text-[9px] font-medium", active ? "text-emerald-300" : "text-zinc-300")}>
          <span className={cn("h-1.5 w-1.5 rounded-full", active ? "bg-emerald-400" : "bg-zinc-400")} />
          {active ? "LIVE" : src ? "FRAME" : "WAITING"}
        </span>
      </figcaption>
    </figure>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-black/[0.06] bg-muted/25 p-3">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold">{value}</div>
    </div>
  );
}

function PathRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-black/[0.06] bg-background px-3 py-2.5">
      <dt className="text-xs font-medium text-muted-foreground">{label}</dt>
      <dd title={value} className="mt-1 truncate text-sm">
        {shortPath(value)}
      </dd>
    </div>
  );
}

function SplatCanvas({ payload }: { payload: PreviewPayload | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rendererError, setRendererError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: true,
      depth: false,
      premultipliedAlpha: true
    });
    if (!gl) {
      setRendererError("WebGL 2 is unavailable");
      return undefined;
    }
    setRendererError(null);

    const program = createSplatProgram(gl);
    if (!program) {
      setRendererError("Gaussian shader compilation failed");
      return undefined;
    }

    const vao = gl.createVertexArray();
    const quadBuffer = gl.createBuffer();
    const instanceBuffer = gl.createBuffer();
    if (!vao || !quadBuffer || !instanceBuffer) {
      setRendererError("GPU buffer allocation failed");
      return undefined;
    }

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const stride = 8 * Float32Array.BYTES_PER_ELEMENT;
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
    for (const [location, size, offset] of [
      [1, 3, 0],
      [2, 4, 3],
      [3, 1, 7]
    ] as const) {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset * Float32Array.BYTES_PER_ELEMENT);
      gl.vertexAttribDivisor(location, 1);
    }
    gl.bindVertexArray(null);

    const points = payload?.points ?? [];
    const center = payload?.bounds.center ?? [0, 0, 0];
    const span = payload
      ? Math.max(
          payload.bounds.max[0] - payload.bounds.min[0],
          payload.bounds.max[1] - payload.bounds.min[1],
          payload.bounds.max[2] - payload.bounds.min[2],
          0.1
        )
      : 1;
    const packed = new Float32Array(points.length * 8);
    points.forEach((point, index) => {
      const offset = index * 8;
      packed[offset] = point.x - center[0];
      packed[offset + 1] = point.y - center[1];
      packed[offset + 2] = point.z - center[2];
      packed[offset + 3] = point.r / 255;
      packed[offset + 4] = point.g / 255;
      packed[offset + 5] = point.b / 255;
      packed[offset + 6] = Math.min(1, Math.max(0.02, point.opacity ?? 0.85));
      packed[offset + 7] = Math.max(point.radius, ...(point.scale ?? [point.radius, point.radius, point.radius]));
    });
    const order = Array.from({ length: points.length }, (_, index) => index);
    const sorted = new Float32Array(packed.length);

    const projectionLocation = gl.getUniformLocation(program, "u_projection");
    const viewportLocation = gl.getUniformLocation(program, "u_viewport");
    const rotationLocation = gl.getUniformLocation(program, "u_rotation");
    const distanceLocation = gl.getUniformLocation(program, "u_distance");
    const translationLocation = gl.getUniformLocation(program, "u_translation");
    let yaw = 0;
    let pitch = 0;
    let distance = span * 2.7;
    const translation: [number, number, number] = [0, 0, 0];
    const movementKeys = new Set<string>();
    let dragging = false;
    let previousX = 0;
    let previousY = 0;
    let animation = 0;
    let previousTime = performance.now();

    const uploadSortedPoints = () => {
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);
      const cosPitch = Math.cos(pitch);
      const sinPitch = Math.sin(pitch);
      const depth = (index: number) => {
        const offset = index * 8;
        const x = packed[offset];
        const y = packed[offset + 1];
        const z = packed[offset + 2];
        const yawZ = x * sinYaw + z * cosYaw;
        return y * sinPitch + yawZ * cosPitch;
      };
      order.sort((a, b) => depth(a) - depth(b));
      order.forEach((sourceIndex, targetIndex) => {
        sorted.set(packed.subarray(sourceIndex * 8, sourceIndex * 8 + 8), targetIndex * 8);
      });
      gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sorted, gl.DYNAMIC_DRAW);
    };

    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();
    uploadSortedPoints();

    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      previousX = event.clientX;
      previousY = event.clientY;
      canvas.focus({ preventScroll: true });
      canvas.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      yaw += (event.clientX - previousX) * 0.008;
      pitch = Math.max(-1.35, Math.min(1.35, pitch + (event.clientY - previousY) * 0.008));
      previousX = event.clientX;
      previousY = event.clientY;
      uploadSortedPoints();
    };
    const onPointerUp = (event: PointerEvent) => {
      dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      distance = Math.max(span * 0.08, Math.min(span * 80, distance * Math.exp(event.deltaY * 0.001)));
    };
    const moveCamera = (key: string, amount: number) => {
      if (key === "w") translation[2] += amount;
      if (key === "s") translation[2] -= amount;
      if (key === "a") translation[0] += amount;
      if (key === "d") translation[0] -= amount;
      if (key === "q") translation[1] += amount;
      if (key === "e") translation[1] -= amount;
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key.toLowerCase();
      if (!["w", "a", "s", "d", "q", "e", "shift"].includes(key)) return;
      event.preventDefault();
      if (key !== "shift" && !event.repeat) {
        moveCamera(key, span * (event.shiftKey ? 0.08 : 0.025));
      }
      movementKeys.add(key);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (!movementKeys.has(key)) return;
      event.preventDefault();
      movementKeys.delete(key);
    };
    const onBlur = () => {
      movementKeys.clear();
    };
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("keydown", onKeyDown);
    canvas.addEventListener("keyup", onKeyUp);
    canvas.addEventListener("blur", onBlur);

    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.CULL_FACE);

    const draw = (time: number) => {
      const delta = Math.min(0.05, (time - previousTime) / 1000);
      previousTime = time;
      if (movementKeys.size) {
        const speed = span * (movementKeys.has("shift") ? 1.8 : 0.55) * delta;
        movementKeys.forEach((key) => moveCamera(key, speed));
      }
      resize();
      gl.clearColor(0.035, 0.035, 0.043, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (points.length) {
        const aspect = canvas.width / Math.max(1, canvas.height);
        gl.uniformMatrix4fv(projectionLocation, false, perspectiveMatrix(Math.PI / 4, aspect, span * 0.002, span * 200));
        gl.uniform2f(viewportLocation, canvas.width, canvas.height);
        gl.uniform2f(rotationLocation, yaw, pitch);
        gl.uniform1f(distanceLocation, distance);
        gl.uniform3f(translationLocation, translation[0], translation[1], translation[2]);
        gl.bindVertexArray(vao);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, points.length);
        gl.bindVertexArray(null);
      }
      animation = requestAnimationFrame(draw);
    };
    animation = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animation);
      resizeObserver.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("keydown", onKeyDown);
      canvas.removeEventListener("keyup", onKeyUp);
      canvas.removeEventListener("blur", onBlur);
      gl.deleteBuffer(instanceBuffer);
      gl.deleteBuffer(quadBuffer);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
    };
  }, [payload]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={1000}
        height={620}
        tabIndex={0}
        aria-label="Interactive 3D Gaussian preview. Drag to rotate, use the mouse wheel to zoom, and use W A S D Q E to move."
        className="h-[calc(100vh-220px)] min-h-[520px] max-h-[760px] w-full touch-none cursor-grab rounded-2xl border border-black/15 bg-zinc-950 outline-none active:cursor-grabbing focus:ring-2 focus:ring-primary focus:ring-offset-2"
      />
      <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/55 px-2 py-1 text-[11px] text-zinc-300">
        {rendererError ??
          (payload?.points.length
            ? "Static preview · drag: rotate · wheel: zoom · click then WASD: move · Q/E: down/up · Shift: faster"
            : "3DGS preview appears here")}
      </div>
    </div>
  );
}

function createSplatProgram(gl: WebGL2RenderingContext) {
  const vertexSource = `#version 300 es
    precision highp float;
    layout(location = 0) in vec2 a_corner;
    layout(location = 1) in vec3 a_center;
    layout(location = 2) in vec4 a_color;
    layout(location = 3) in float a_radius;
    uniform mat4 u_projection;
    uniform vec2 u_viewport;
    uniform vec2 u_rotation;
    uniform float u_distance;
    uniform vec3 u_translation;
    out vec2 v_uv;
    out vec4 v_color;

    void main() {
      float cy = cos(u_rotation.x);
      float sy = sin(u_rotation.x);
      float cp = cos(u_rotation.y);
      float sp = sin(u_rotation.y);
      vec3 yawed = vec3(
        a_center.x * cy - a_center.z * sy,
        a_center.y,
        a_center.x * sy + a_center.z * cy
      );
      vec3 view = vec3(
        yawed.x,
        yawed.y * cp - yawed.z * sp,
        yawed.y * sp + yawed.z * cp - u_distance
      );
      view += u_translation;
      vec4 clip = u_projection * vec4(view, 1.0);
      float radius_px = clamp(
        a_radius * 3.0 * u_projection[1][1] * u_viewport.y * 0.5 / max(0.001, -view.z),
        1.2,
        96.0
      );
      vec2 clip_offset = a_corner * radius_px * 2.0 / u_viewport * clip.w;
      gl_Position = clip + vec4(clip_offset, 0.0, 0.0);
      v_uv = a_corner;
      v_color = a_color;
    }`;
  const fragmentSource = `#version 300 es
    precision highp float;
    in vec2 v_uv;
    in vec4 v_color;
    out vec4 out_color;

    void main() {
      float radius2 = dot(v_uv, v_uv);
      if (radius2 > 1.0) discard;
      float alpha = exp(-4.5 * radius2) * v_color.a;
      if (alpha < 0.004) discard;
      out_color = vec4(v_color.rgb * alpha, alpha);
    }`;
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertex || !fragment) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function perspectiveMatrix(fov: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fov / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * range,
    -1,
    0,
    0,
    2 * far * near * range,
    0
  ]);
}

async function tauriCall<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri) {
    return mockInvoke<T>(command, args);
  }
  return invoke<T>(command, args);
}

async function mockInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (command === "load_latest_scan_assets") {
    return null as T;
  }
  if (command === "default_save_location") {
    return "/preview/3dscan" as T;
  }
  if (command === "probe_runtime") {
    return {
      sdkLoaded: false,
      apiVersion: null,
      devices: [],
      usbDevices: [],
      status: "Browser preview mode",
      installHint: "Run inside Tauri to access RealSense.",
      actionRequired: null
    } as T;
  }
  if (command === "detect_asset_tools") {
    return {
      fbxAvailable: true,
      fbxExporter: "Built-in native FBX 7.4 exporter",
      python: "/preview/python3",
      mlxAvailable: false,
      mlxStatus: "Preview mode",
      brushHint: "Preview mode"
    } as T;
  }
  if (command === "ensure_realsense_sdk") {
    return {
      status: "Preview mode: SDK setup runs only inside Tauri",
      log: ["Preview mode"]
    } as T;
  }
  if (command === "ensure_mlx_3dgs") {
    return {
      status: "Preview mode: gsplat-mlx setup runs only inside Tauri",
      log: ["Preview mode"],
      tools: {
        fbxAvailable: true,
        fbxExporter: "Built-in native FBX 7.4 exporter",
        python: "/preview/python3",
        mlxAvailable: true,
        mlxStatus: "Preview MLX ready",
        brushHint: "Preview mode"
      }
    } as T;
  }
  if (
    command === "install_privileged_helper" ||
    command === "ensure_privileged_helper" ||
    command === "privileged_helper_status"
  ) {
    return {
      path: "/preview/realsense-helper",
      status: "Preview mode: helper install runs only inside Tauri",
      ready: true,
      current: true
    } as T;
  }
  if (command === "start_recording" || command === "start_preview") {
    return {
      sessionId: `preview_${Date.now()}`,
      root: "/preview/SmartAgricultureScans",
      backend: "synthetic",
      notice: "Browser preview mode",
      progressPath: null
    } as T;
  }
  if (command === "stop_recording" || command === "stop_preview") {
    return { framesWritten: 0 } as T;
  }
  if (command === "read_latest_privileged_preview_frame") {
    return {
      sessionId: `preview_${Date.now()}`,
      frameIndex: 1,
      timestampMs: 0,
      frameNumber: 1,
      colorPreviewDataUrl: drawMockFrame("rgb", 1),
      depthPreviewDataUrl: drawMockFrame("depth", 1),
      depth: { validPoints: 19200, minM: 0.31, maxM: 0.76, meanM: 0.48 },
      paths: { rgb: null, depth: "-", pointCloud: "-", metadata: "-" }
    } as T;
  }
  if (command === "generate_scan_assets" || command === "load_scan_data") {
    const points = mockPreviewPoints();
    return {
      root: "/preview/assets",
      seedGaussianPly: "/preview/assets/gaussian_splats/scan_gaussians_seed.ply",
      gaussianPly: "/preview/assets/gaussian_splats/scan_gaussians_mlx.ply",
      splat: "/preview/assets/gaussian_splats/scan_gaussians_mlx.splat",
      meshObj: "/preview/assets/mesh/scan_surface.obj",
      meshFbx: "/preview/assets/mesh/scan_surface.fbx",
      colliderObj: "/preview/assets/mesh/scan_collider.obj",
      collisionJson: "/preview/assets/mesh/scan_collision.json",
      collisionFbx: "/preview/assets/mesh/scan_surface.fbx",
      previewJson: "/preview/assets/preview/preview_points.json",
      manifest: "/preview/assets/asset_manifest.json",
      pointCount: points.length,
      faceCount: 12000,
      fbxStatus: "Native FBX ready (no Blender)",
      mlxStatus: "Preview mode",
      collisionStatus: "Preview collision collider ready",
      tools: {
        fbxAvailable: true,
        fbxExporter: "Built-in native FBX 7.4 exporter",
        python: "/preview/python3",
        mlxAvailable: false,
        mlxStatus: "Preview mode",
        brushHint: "Preview mode"
      },
      preview: {
        points,
        bounds: {
          min: [-0.28, -0.24, -0.28],
          max: [0.28, 0.24, 0.28],
          center: [0, 0, 0]
        }
      }
    } as T;
  }
  return undefined as T;
}

function startMockFrames(
  session: SessionStarted,
  config: CaptureConfig,
  timer: React.MutableRefObject<number | null>,
  setLatestFrame: React.Dispatch<React.SetStateAction<FrameSummary | null>>
) {
  stopMockFrames(timer);
  let frame = 0;
  timer.current = window.setInterval(() => {
    frame += 1;
    setLatestFrame({
      sessionId: session.sessionId,
      frameIndex: frame,
      timestampMs: frame * (1000 / config.fps),
      frameNumber: frame,
      colorPreviewDataUrl: drawMockFrame("rgb", frame),
      depthPreviewDataUrl: drawMockFrame("depth", frame),
      depth: {
        validPoints: 19200 + frame * 6,
        minM: 0.31,
        maxM: 0.76,
        meanM: 0.48
      },
      paths: {
        rgb: `${session.root}/${config.targetLabel || "scan"}.mcap#/camera/color/image/compressed`,
        depth: `${session.root}/${config.targetLabel || "scan"}.mcap#/camera/depth/image_raw`,
        pointCloud: `${session.root}/${config.targetLabel || "scan"}.mcap#/camera/depth/color/points`,
        metadata: `${session.root}/${config.targetLabel || "scan"}.mcap#/agriscan/frame_info`
      }
    });
  }, 1000 / Math.max(1, config.fps));
}

function stopMockFrames(timer: React.MutableRefObject<number | null>) {
  if (timer.current !== null) {
    window.clearInterval(timer.current);
    timer.current = null;
  }
}

function drawMockFrame(kind: "rgb" | "depth", frame: number) {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  ctx.fillStyle = kind === "rgb" ? "#263832" : "#111827";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < 18; i += 1) {
    ctx.strokeStyle = kind === "rgb" ? `rgba(79, 148, 97, ${0.2 + i / 50})` : "rgba(56, 189, 248, .16)";
    ctx.lineWidth = 10 + (i % 3);
    ctx.beginPath();
    ctx.moveTo(80 + i * 30, 0);
    ctx.bezierCurveTo(60 + i * 20, 160, 180 + i * 16, 280, 90 + i * 24, 480);
    ctx.stroke();
  }

  const tomatoes = [
    [260, 216, 72],
    [348, 184, 58],
    [372, 292, 66],
    [290, 304, 48]
  ];
  tomatoes.forEach(([x, y, radius], index) => {
    const dx = Math.sin(frame * 0.08 + index) * 8;
    const dy = Math.cos(frame * 0.06 + index) * 5;
    const gradient = ctx.createRadialGradient(x + dx - radius / 3, y + dy - radius / 3, 4, x + dx, y + dy, radius);
    if (kind === "rgb") {
      gradient.addColorStop(0, "#f6a37f");
      gradient.addColorStop(0.45, "#d63c2e");
      gradient.addColorStop(1, "#7f1f22");
    } else {
      gradient.addColorStop(0, "#f7c85f");
      gradient.addColorStop(0.55, "#cb4f48");
      gradient.addColorStop(1, "#304f83");
    }
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.ellipse(x + dx, y + dy, radius, radius * 0.82, 0.08, 0, Math.PI * 2);
    ctx.fill();
  });

  return canvas.toDataURL("image/png");
}

function mockPreviewPoints() {
  const points: PreviewPoint[] = [];
  for (let i = 0; i < 9000; i += 1) {
    const t = Math.random() * Math.PI * 2;
    const u = Math.random() * Math.PI - Math.PI / 2;
    const radius = 0.22 + Math.sin(t * 3) * 0.018;
    points.push({
      x: Math.cos(t) * Math.cos(u) * radius,
      y: Math.sin(u) * radius * 0.85,
      z: Math.sin(t) * Math.cos(u) * radius,
      r: 190 + Math.floor(Math.random() * 48),
      g: 48 + Math.floor(Math.random() * 35),
      b: 38 + Math.floor(Math.random() * 30),
      radius: 0.006,
      scale: [0.006, 0.006, 0.006],
      rotation: [1, 0, 0, 0],
      opacity: 0.85
    });
  }
  return points;
}

function parseNumber(value: string, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseNullableNumber(value: string) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function shortPath(value: string) {
  if (value.length < 44) return value;
  const parts = value.split("/");
  return parts.length > 2 ? `.../${parts.slice(-2).join("/")}` : `...${value.slice(-40)}`;
}

function firstLine(value: string) {
  return value.split("\n").find((line) => line.trim().length > 0)?.trim() ?? value;
}

createRoot(document.querySelector<HTMLDivElement>("#app")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
