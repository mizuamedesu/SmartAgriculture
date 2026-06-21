import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  Box,
  Camera,
  Circle,
  Cpu,
  Download,
  FolderOpen,
  Loader2,
  Radio,
  RefreshCw,
  ScanLine,
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

interface CaptureConfig {
  width: number;
  height: number;
  fps: number;
  backend: BackendMode;
  targetLabel: string;
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

interface SessionStarted {
  sessionId: string;
  root: string;
  backend: string;
  notice: string | null;
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
  blender: string | null;
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
  gaussianPly: string;
  splat: string;
  meshObj: string;
  meshFbx: string | null;
  previewJson: string;
  manifest: string;
  pointCount: number;
  faceCount: number;
  fbxStatus: string;
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
  const [probe, setProbe] = useState<RuntimeProbe | null>(null);
  const [config, setConfig] = useState<CaptureConfig>({
    width: 1280,
    height: 720,
    fps: 30,
    backend: "realsense",
    targetLabel: "mini_tomato",
    cultivar: "",
    notes: "",
    maxFrames: null,
    pointStride: 4,
    minDepthM: 0.12,
    maxDepthM: 1.4
  });
  const [assetOptions, setAssetOptions] = useState<AssetOptions>({
    maxPoints: 180000,
    frameStride: 1,
    depthDecimation: 4,
    gaussianRadiusM: 0.006,
    turntableDegrees: 360,
    exportFbx: true
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
  const [helperInstallBusy, setHelperInstallBusy] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const mockTimer = useRef<number | null>(null);
  const privilegedPollTimer = useRef<number | null>(null);
  const latestFrameAttachTimer = useRef<number | null>(null);
  const previewTimeoutTimer = useRef<number | null>(null);
  const previewRequestId = useRef(0);
  const privilegedReadBusy = useRef(false);
  const privilegedPreviewRef = useRef<PrivilegedPreviewStarted | null>(null);
  const autoSetupAttempted = useRef(false);

  const devices = probe?.devices ?? [];
  const backend = activeSession?.backend ?? previewSession?.backend ?? config.backend;
  const busyMessage = captureStopping
    ? "Loading: stopping recording"
    : captureStarting
      ? "Loading: starting RGB-D recording"
      : previewLoading
        ? "Loading: opening RealSense preview"
        : sdkSetupBusy
          ? "Loading: checking SDK"
          : helperInstallBusy
            ? "Loading: installing helper"
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

  const installHelper = async () => {
    setHelperInstallBusy(true);
    pushLog("installing no-sudo RealSense helper");
    try {
      const result = await tauriCall<InstalledHelper>("install_privileged_helper");
      pushLog(result.status);
      pushLog(result.path);
    } catch (error) {
      pushLog(`helper install failed: ${String(error)}`);
    } finally {
      setHelperInstallBusy(false);
    }
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

      const session = await tauriCall<SessionStarted>("start_recording", { config });
      setRecording(true);
      setPreviewing(false);
      setActiveSession(session);
      setPreviewSession(null);
      setLatestFrame(null);
      setAssetResult(null);
      pushLog(`started ${session.backend}: ${session.sessionId}`);
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
            notice: null
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
      setActiveSession(null);
      setLatestFrame(null);
      setAssetResult(null);
      setPrivilegedPreview(null);

      if (!isTauri) {
        const session = {
          sessionId: `browser_preview_${Date.now()}`,
          root: "",
          backend: "synthetic",
          notice: "Browser preview mode"
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
        notice: null
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
      pushLog(`stopped ${stopped.framesWritten} frames`);
    } catch (error) {
      setRecording(false);
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
    pushLog("building 3DGS seed, preview cloud, OBJ, and FBX");
    try {
      const result = await tauriCall<AssetBuildResult>("generate_scan_assets", {
        options: {
          sessionRoot: activeSession.root,
          ...assetOptions
        }
      });
      setAssetResult(result);
      pushLog(`assets ready: ${result.pointCount.toLocaleString()} splats`);
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

  useEffect(() => {
    refreshProbe({ autoSetup: true });

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

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="flex h-[72px] items-center gap-4 px-5">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
              TT
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-lg font-semibold tracking-normal">Tomato Twin Capture</h1>
              <p className="truncate text-sm text-muted-foreground">RealSense RGB-D scan console</p>
            </div>
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
            {busyMessage ? (
              <Badge variant="warning">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                {busyMessage}
              </Badge>
            ) : null}
          </div>

          <Button size="icon" variant="outline" onClick={() => refreshProbe()} disabled={sdkSetupBusy || probeBusy} title="Refresh devices">
            <RefreshCw className={cn("h-4 w-4", (sdkSetupBusy || probeBusy) && "animate-spin")} />
          </Button>
        </div>
      </header>
      {busyMessage ? (
        <div className="border-b bg-amber-50 px-4 py-2 text-sm font-medium text-amber-900">
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{busyMessage}</span>
          </div>
        </div>
      ) : null}

      <main className="grid gap-4 p-4 xl:grid-cols-[340px_minmax(520px,1fr)_360px]">
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
        />

        <section className="min-h-0 space-y-4">
          <LiveFramePanel
            latestFrame={latestFrame}
            activeSession={activeSession ?? previewSession}
            previewing={previewing}
            recording={recording}
            loadingMessage={previewLoading ? busyMessage : null}
          />
          <AssetPreviewPanel assetResult={assetResult} assetBusy={assetBusy} revealAssets={() => revealPath(assetResult?.root)} />
        </section>

        <OutputPanel
          activeSession={activeSession}
          latestFrame={latestFrame}
          devices={devices}
          probe={probe}
          assetResult={assetResult}
          log={log}
          setupSdk={setupSdk}
          installHelper={installHelper}
          sdkSetupBusy={sdkSetupBusy}
          helperInstallBusy={helperInstallBusy}
          recording={recording}
          revealSession={() => revealPath(activeSession?.root)}
        />
      </main>
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
    <Card className="h-fit xl:sticky xl:top-[88px]">
      <CardHeader className="border-b pb-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Capture</CardTitle>
            <CardDescription>Backend: {props.backend}</CardDescription>
          </div>
          <Badge variant="secondary">
            {props.config.width}x{props.config.height} / {props.config.fps}fps
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 pt-4">
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
          <NumberField label="PLY stride" value={props.config.pointStride} min={1} max={12} disabled={disabled} onChange={(v) => updateConfig("pointStride", v)} />
          <NumberField label="Min depth" value={props.config.minDepthM} min={0.02} max={4} step={0.01} disabled={disabled} onChange={(v) => updateConfig("minDepthM", v)} />
          <NumberField label="Max depth" value={props.config.maxDepthM} min={0.03} max={8} step={0.01} disabled={disabled} onChange={(v) => updateConfig("maxDepthM", v)} />
        </div>

        <Field label="Target">
          <Input value={props.config.targetLabel} disabled={disabled} onChange={(event) => updateConfig("targetLabel", event.target.value)} />
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

        <div className="grid grid-cols-2 gap-3">
          <Button
            variant="secondary"
            onClick={props.previewing ? props.stopPreview : props.startPreview}
            disabled={props.recording || props.captureStarting || props.captureStopping}
          >
            {props.previewLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
            {props.previewLoading ? "Loading Preview" : props.previewing ? "Stop Live" : "Live Preview"}
          </Button>
          <Button onClick={props.startCapture} disabled={props.recording || props.previewing || props.captureStarting}>
            {props.captureStarting ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanLine className="h-4 w-4" />}
            {props.captureStarting ? "Loading Record" : "Record RGB-D"}
          </Button>
          <Button className="col-span-2" variant="destructive" onClick={props.stopCapture} disabled={!props.recording || props.captureStopping}>
            {props.captureStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
            {props.captureStopping ? "Loading Stop" : "Stop Recording"}
          </Button>
        </div>

        <div className="rounded-lg border bg-muted/35 p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold">3DGS / FBX</h3>
              <p className="text-xs text-muted-foreground">{props.assetTools?.blender ? "Blender ready" : "FBX optional"}</p>
            </div>
            <WandSparkles className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Frame step" value={props.assetOptions.frameStride} min={1} max={24} onChange={(v) => updateAsset("frameStride", v)} />
            <NumberField label="Depth step" value={props.assetOptions.depthDecimation} min={1} max={16} onChange={(v) => updateAsset("depthDecimation", v)} />
            <NumberField label="Max splats" value={props.assetOptions.maxPoints} min={5000} max={1500000} step={1000} onChange={(v) => updateAsset("maxPoints", v)} />
            <NumberField label="Radius m" value={props.assetOptions.gaussianRadiusM} min={0.0005} max={0.05} step={0.0005} onChange={(v) => updateAsset("gaussianRadiusM", v)} />
            <NumberField label="Turntable" value={props.assetOptions.turntableDegrees} min={0} max={1080} onChange={(v) => updateAsset("turntableDegrees", v)} />
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
          <Button className="mt-3 w-full" variant="secondary" onClick={props.generateAssets} disabled={!props.activeSession || props.recording || props.assetBusy}>
            {props.assetBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Box className="h-4 w-4" />}
            {props.assetBusy ? "Generating" : "Generate assets"}
          </Button>
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
    <Card>
      <CardHeader className="border-b pb-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Live Frames</CardTitle>
            <CardDescription>
              {recording ? "recording" : previewing ? "previewing" : "idle"} / {activeSession?.sessionId ?? "no session"}
            </CardDescription>
          </div>
          <div className="rounded-md border bg-muted px-3 py-2 text-right">
            <div className="text-lg font-semibold leading-none">{latestFrame?.frameIndex ?? 0}</div>
            <div className="text-xs text-muted-foreground">frames</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 pt-4">
        <div className="relative">
          <div className="grid gap-3 lg:grid-cols-2">
            <PreviewPane label="RGB" src={latestFrame?.colorPreviewDataUrl ?? null} icon={<Camera className="h-7 w-7" />} />
            <PreviewPane label="Depth" src={latestFrame?.depthPreviewDataUrl ?? null} icon={<ScanLine className="h-7 w-7" />} />
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
  revealAssets
}: {
  assetResult: AssetBuildResult | null;
  assetBusy: boolean;
  revealAssets: () => void;
}) {
  return (
    <Card>
      <CardHeader className="border-b pb-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>3DGS Preview</CardTitle>
            <CardDescription>
              {assetResult
                ? `${assetResult.pointCount.toLocaleString()} gaussians / ${assetResult.faceCount.toLocaleString()} faces`
                : "generate assets after capture"}
            </CardDescription>
          </div>
          <Button size="icon" variant="outline" disabled={!assetResult} onClick={revealAssets} title="Open assets folder">
            <FolderOpen className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="relative">
          <SplatCanvas payload={assetResult?.preview ?? null} />
          {assetBusy ? (
            <div className="absolute inset-0 grid place-items-center rounded-lg border bg-background/80 backdrop-blur-sm">
              <div className="flex items-center gap-3 rounded-md border bg-background px-4 py-3 shadow-sm">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
                <div>
                  <div className="text-sm font-semibold">Loading</div>
                  <div className="text-xs text-muted-foreground">Generating 3DGS, splat, OBJ, and FBX</div>
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
  installHelper: () => void;
  sdkSetupBusy: boolean;
  helperInstallBusy: boolean;
  recording: boolean;
  revealSession: () => void;
}) {
  return (
    <aside className="space-y-4 xl:sticky xl:top-[88px] xl:h-fit">
      <Card>
        <CardHeader className="border-b pb-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <CardTitle>Dataset</CardTitle>
              <CardDescription>Capture output</CardDescription>
            </div>
            <Button size="icon" variant="outline" disabled={!props.activeSession} onClick={props.revealSession}>
              <FolderOpen className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-2 pt-4">
          <PathRow label="Root" value={props.activeSession?.root ?? "-"} />
          <PathRow label="RGB" value={props.latestFrame?.paths.rgb ?? "-"} />
          <PathRow label="Depth" value={props.latestFrame?.paths.depth ?? "-"} />
          <PathRow label="PLY" value={props.latestFrame?.paths.pointCloud ?? "-"} />
          <PathRow label="Metadata" value={props.latestFrame?.paths.metadata ?? "-"} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="border-b pb-4">
          <CardTitle>Assets</CardTitle>
          <CardDescription>3DGS, splat, mesh and FBX</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 pt-4">
          <PathRow label="3DGS PLY" value={props.assetResult?.gaussianPly ?? "-"} />
          <PathRow label=".splat" value={props.assetResult?.splat ?? "-"} />
          <PathRow label="OBJ" value={props.assetResult?.meshObj ?? "-"} />
          <PathRow label="FBX" value={props.assetResult?.meshFbx ?? props.assetResult?.fbxStatus ?? "-"} />
          <PathRow label="Preview" value={props.assetResult?.previewJson ?? "-"} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="border-b pb-4">
          <CardTitle>Device</CardTitle>
          <CardDescription>SDK and connected cameras</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 pt-4">
          <Button className="w-full" variant="secondary" onClick={props.setupSdk} disabled={props.sdkSetupBusy || props.recording}>
            <Download className={cn("h-4 w-4", props.sdkSetupBusy && "animate-bounce")} />
            {props.sdkSetupBusy ? "Setting up" : "Setup SDK"}
          </Button>
          <Button className="w-full" variant="outline" onClick={props.installHelper} disabled={props.helperInstallBusy || props.recording}>
            <Cpu className={cn("h-4 w-4", props.helperInstallBusy && "animate-pulse")} />
            {props.helperInstallBusy ? "Installing helper" : "Install no-sudo helper"}
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

      <Card>
        <CardHeader className="border-b pb-4">
          <CardTitle>Log</CardTitle>
          <CardDescription>Recent events</CardDescription>
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

function PreviewPane({ label, src, icon }: { label: string; src: string | null; icon: React.ReactNode }) {
  return (
    <figure className="overflow-hidden rounded-lg border bg-zinc-950">
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
      <figcaption className="border-t border-white/10 px-3 py-2 text-sm font-medium text-zinc-200">{label}</figcaption>
    </figure>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-muted/30 p-3">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold">{value}</div>
    </div>
  );
}

function PathRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background px-3 py-2">
      <dt className="text-xs font-medium text-muted-foreground">{label}</dt>
      <dd title={value} className="mt-1 truncate text-sm">
        {shortPath(value)}
      </dd>
    </div>
  );
}

function SplatCanvas({ payload }: { payload: PreviewPayload | null }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const angle = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return undefined;

    let animation = 0;
    const drawEmpty = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#09090b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(255,255,255,.12)";
      ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
      ctx.fillStyle = "rgba(244,244,245,.7)";
      ctx.font = "15px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("3DGS preview appears here", canvas.width / 2, canvas.height / 2);
    };

    if (!payload?.points.length) {
      drawEmpty();
      return undefined;
    }

    const points = payload.points;
    const center = payload.bounds.center;
    const span = Math.max(
      payload.bounds.max[0] - payload.bounds.min[0],
      payload.bounds.max[1] - payload.bounds.min[1],
      payload.bounds.max[2] - payload.bounds.min[2],
      0.1
    );
    const scale = (Math.min(canvas.width, canvas.height) * 0.82) / span;

    const draw = () => {
      angle.current += 0.006;
      const cos = Math.cos(angle.current);
      const sin = Math.sin(angle.current);
      const projected = points.map((point) => {
        const x = point.x - center[0];
        const y = point.y - center[1];
        const z = point.z - center[2];
        const rx = x * cos - z * sin;
        const rz = x * sin + z * cos;
        const perspective = 1.5 / (1.5 - rz);
        return {
          x: canvas.width / 2 + rx * scale * perspective,
          y: canvas.height / 2 - y * scale * perspective,
          z: rz,
          size: Math.max(1.1, point.radius * scale * perspective * 1.8),
          color: `rgb(${point.r},${point.g},${point.b})`
        };
      });
      projected.sort((a, b) => a.z - b.z);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#09090b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const point of projected) {
        ctx.globalAlpha = 0.72;
        ctx.fillStyle = point.color;
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      animation = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animation);
  }, [payload]);

  return <canvas ref={canvasRef} width={1000} height={430} className="h-[430px] w-full rounded-lg border bg-zinc-950" />;
}

async function tauriCall<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri) {
    return mockInvoke<T>(command, args);
  }
  return invoke<T>(command, args);
}

async function mockInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
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
    return { blender: null, brushHint: "Preview mode" } as T;
  }
  if (command === "ensure_realsense_sdk") {
    return {
      status: "Preview mode: SDK setup runs only inside Tauri",
      log: ["Preview mode"]
    } as T;
  }
  if (command === "install_privileged_helper") {
    return {
      path: "/preview/realsense-helper",
      status: "Preview mode: helper install runs only inside Tauri"
    } as T;
  }
  if (command === "start_recording" || command === "start_preview") {
    return {
      sessionId: `preview_${Date.now()}`,
      root: "/preview/SmartAgricultureScans",
      backend: "synthetic",
      notice: "Browser preview mode"
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
  if (command === "generate_scan_assets") {
    const points = mockPreviewPoints();
    return {
      root: "/preview/assets",
      gaussianPly: "/preview/assets/gaussian_splats/tomato_gaussians_seed.ply",
      splat: "/preview/assets/gaussian_splats/tomato_gaussians_seed.splat",
      meshObj: "/preview/assets/mesh/tomato_surface.obj",
      meshFbx: null,
      previewJson: "/preview/assets/preview/preview_points.json",
      manifest: "/preview/assets/asset_manifest.json",
      pointCount: points.length,
      faceCount: 12000,
      fbxStatus: "Preview mode",
      tools: { blender: null, brushHint: "Preview mode" },
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
        rgb: `${session.root}/rgb/frame_${String(frame).padStart(6, "0")}_rgb.png`,
        depth: `${session.root}/depth_z16/frame_${String(frame).padStart(6, "0")}_depth_z16.png`,
        pointCloud: `${session.root}/pointcloud_ply/frame_${String(frame).padStart(6, "0")}_cloud.ply`,
        metadata: `${session.root}/metadata/frame_${String(frame).padStart(6, "0")}.json`
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
      radius: 0.006
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

createRoot(document.querySelector<HTMLDivElement>("#app")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
