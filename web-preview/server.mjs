import { createServer } from "node:http";
import { execFile } from "node:child_process";
import { existsSync, promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PORT = Number(process.env.PORT || 5174);
const APP_BIN = resolveAppBin();
const TMP = os.tmpdir();

let currentSession = null;

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "127.0.0.1"}`);
    if (req.method === "GET" && url.pathname === "/") {
      return sendFile(res, path.join(__dirname, "index.html"), "text/html; charset=utf-8");
    }
    if (req.method === "POST" && url.pathname === "/api/start-preview") {
      const body = await readJson(req).catch(() => ({}));
      const config = {
        width: Number(body.width || 1280),
        height: Number(body.height || 720),
        fps: Number(body.fps || 30)
      };
      const session = await startPreview(config);
      return json(res, { ok: true, session });
    }
    if (req.method === "POST" && url.pathname === "/api/stop-preview") {
      await stopPreview();
      return json(res, { ok: true });
    }
    if (req.method === "GET" && url.pathname === "/api/frame") {
      const frame = await readNewestFrame();
      return json(res, frame);
    }
    if (req.method === "GET" && url.pathname === "/api/status") {
      return json(res, {
        ok: true,
        currentSession,
        helperPids: await helperPids(),
        newestFrame: await newestFramePath().catch(() => null),
        appBin: APP_BIN,
        tmp: TMP
      });
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: false, error: "not found" }));
  } catch (error) {
    res.writeHead(500, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: false, error: String(error?.message || error) }));
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Tomato Twin Web Preview: http://127.0.0.1:${PORT}`);
});

async function startPreview(config) {
  if (!existsSync(APP_BIN)) {
    throw new Error(`RealSense helper binary not found: ${APP_BIN}`);
  }

  const sessionId = `web_preview_${timestamp()}`;
  const framePath = path.join(TMP, `${sessionId}.json`);
  const pidPath = path.join(TMP, `${sessionId}.pid`);
  const logPath = path.join(TMP, `${sessionId}.log`);

  const killDaemons = "killall -9 UVCAssistant VDCAssistant cameracaptured appleh16camerad AppleCameraAssistant com.apple.cmio.registerassistantservice 2>/dev/null || true";
  const killHelpers = "pkill -9 -f '[s]mart-agriculture-tomato-twin --realsense-helper live' 2>/dev/null || true; pkill -9 -f '/usr/local/libexec/tomato-twin/[r]ealsense-helper live' 2>/dev/null || true";
  const daemonSweeper = `(i=0; while [ $i -lt 80 ]; do ${killDaemons}; i=$((i+1)); sleep 0.05; done) >/dev/null 2>&1 & true`;

  await osascriptAdmin([killHelpers, killDaemons, daemonSweeper].join("; "), 30_000);

  const device = await detectRealSense();
  if (!device.detected) {
    throw new Error(`RealSense device is not detected by librealsense. ${device.summary}`);
  }

  const helper = [
    shellQuote(APP_BIN),
    "--realsense-helper",
    "live",
    shellQuote(framePath),
    String(config.width),
    String(config.height),
    String(config.fps),
    shellQuote(sessionId),
    shellQuote(logPath)
  ].join(" ");
  const command = `${helper} >/tmp/tomato-twin-web-helper-launch.log 2>&1 & echo $! > ${shellQuote(pidPath)}`;

  await osascriptAdmin(command, 30_000);
  const session = { sessionId, framePath, pidPath, logPath, ...config };
  currentSession = session;

  try {
    await waitForFrame(framePath, logPath, 18_000);
  } catch (error) {
    await killPidPath(pidPath);
    currentSession = null;
    throw error;
  }

  return session;
}

async function stopPreview() {
  const pids = await helperPids();
  if (!pids.length && !currentSession?.pidPath) {
    currentSession = null;
    return;
  }

  const pidText = currentSession?.pidPath && existsSync(currentSession.pidPath)
    ? await fs.readFile(currentSession.pidPath, "utf8").catch(() => "")
    : "";
  const pidTargets = [...new Set([...pids, ...pidText.trim().split(/\s+/).filter(Boolean)])];
  if (pidTargets.length) {
    await osascriptAdmin(`kill -9 ${pidTargets.map(shellQuote).join(" ")} 2>/dev/null || true`, 20_000).catch(() => {});
  }
  currentSession = null;
}

async function readNewestFrame() {
  const framePath = await newestFramePath();
  const raw = await fs.readFile(framePath, "utf8");
  return JSON.parse(raw);
}

async function newestFramePath() {
  const candidates = [];
  for (const name of await fs.readdir(TMP)) {
    if (!/^(web_preview_|realsense_preview_).+\.json$/.test(name) || name.endsWith(".tmp")) continue;
    const full = path.join(TMP, name);
    const stat = await fs.stat(full).catch(() => null);
    if (!stat || Date.now() - stat.mtimeMs > 8_000) continue;
    candidates.push({ full, mtimeMs: stat.mtimeMs });
  }
  candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
  if (!candidates.length) throw new Error("no live RealSense frame JSON");
  return candidates[0].full;
}

async function helperPids() {
  return new Promise((resolve) => {
    execFile("pgrep", ["-f", "[s]mart-agriculture-tomato-twin --realsense-helper live|[r]ealsense-helper live"], (error, stdout) => {
      if (error) return resolve([]);
      resolve(stdout.trim().split(/\s+/).filter(Boolean));
    });
  });
}

async function detectRealSense() {
  const candidates = [
    "/opt/homebrew/bin/rs-enumerate-devices",
    "/usr/local/bin/rs-enumerate-devices",
    "rs-enumerate-devices"
  ];
  let last = "rs-enumerate-devices was not found.";
  for (const command of candidates) {
    if (command.startsWith("/") && !existsSync(command)) continue;
    try {
      const output = await execFileText(command, ["-s"], 8_000);
      const summary = output.trim() || "rs-enumerate-devices returned no text.";
      return {
        detected: librealsenseSawUsb(summary),
        summary: oneLine(summary)
      };
    } catch (error) {
      last = String(error.message || error);
      if (/No device detected/i.test(last)) {
        return { detected: librealsenseSawUsb(last), summary: oneLine(last) };
      }
    }
  }
  return { detected: true, summary: oneLine(last) };
}

function execFileText(command, args, timeout) {
  return new Promise((resolve, reject) => {
    execFile(command, args, { timeout }, (error, stdout, stderr) => {
      const output = `${stdout || ""}${stderr || ""}`;
      if (error) {
        reject(new Error(output.trim() || error.message));
        return;
      }
      resolve(output);
    });
  });
}

async function waitForFrame(framePath, logPath, timeoutMs) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const stat = await fs.stat(framePath).catch(() => null);
    if (stat && Date.now() - stat.mtimeMs < 4_000) {
      const raw = await fs.readFile(framePath, "utf8").catch(() => "");
      if (raw.includes("colorPreviewDataUrl") || raw.includes("depthPreviewDataUrl")) {
        return;
      }
    }
    await sleep(250);
  }
  const log = await fs.readFile(logPath, "utf8").catch(() => "");
  throw new Error(`RealSense helper did not publish frames within 18s. ${oneLine(log) || "No helper log was written."}`);
}

async function killPidPath(pidPath) {
  const pidText = await fs.readFile(pidPath, "utf8").catch(() => "");
  const pids = pidText.trim().split(/\s+/).filter(Boolean);
  if (pids.length) {
    await osascriptAdmin(`kill -9 ${pids.map(shellQuote).join(" ")} 2>/dev/null || true`, 20_000).catch(() => {});
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function oneLine(value) {
  return String(value || "").replace(/\s+/g, " ").trim().slice(0, 800);
}

function librealsenseSawUsb(value) {
  const text = String(value || "");
  if (/failed to claim usb interface|failed to set power state|Could not create device/i.test(text)) {
    return true;
  }
  return !/No device detected/i.test(text);
}

function osascriptAdmin(command, timeout) {
  const script = `do shell script ${appleScriptString(command)} with administrator privileges`;
  return new Promise((resolve, reject) => {
    execFile("/usr/bin/osascript", ["-e", script], { timeout }, (error, stdout, stderr) => {
      if (error) return reject(new Error(`${stderr || stdout || error.message}`.trim()));
      resolve(stdout);
    });
  });
}

async function sendFile(res, file, type) {
  const body = await fs.readFile(file);
  res.writeHead(200, { "content-type": type, "cache-control": "no-store" });
  res.end(body);
}

function json(res, value) {
  res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
  res.end(JSON.stringify(value));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => resolve(body ? JSON.parse(body) : {}));
    req.on("error", reject);
  });
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", "'\\''")}'`;
}

function appleScriptString(value) {
  return `"${String(value).replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

function timestamp() {
  const now = new Date();
  return [
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0")
  ].join("");
}

function resolveAppBin() {
  if (process.env.REALSENSE_APP_BIN) return process.env.REALSENSE_APP_BIN;
  const release = path.join(ROOT, "src-tauri", "target", "release", "smart-agriculture-tomato-twin");
  if (existsSync(release)) return release;
  return path.join(ROOT, "src-tauri", "target", "debug", "smart-agriculture-tomato-twin");
}
