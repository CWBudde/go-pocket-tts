import { ONNXBridge } from "./bridge.js";

const log = document.getElementById("log");
const textArea = document.getElementById("text");
const runBtn = document.getElementById("run");
const verifyBtn = document.getElementById("verify-models");
const modelSynthBtn = document.getElementById("synthesize-model");
const player = document.getElementById("player");
const download = document.getElementById("download");
const synthProgress = document.getElementById("synth-progress");
const synthProgressText = document.getElementById("synth-progress-text");

const onnxBridge = new ONNXBridge({ manifestPath: "./models/manifest.json" });
const requiredModelGraphs = ["text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"];

const capabilities = {
  kernelReady: false,
  modelManifestReady: false,
  modelGraphsReady: false,
  modelReady: false,
  reasons: [],
};

const progressState = {
  startedAtMs: 0,
  lastEventAtMs: 0,
  arStartedAtMs: 0,
};

function formatError(err) {
  if (!err) {
    return "unknown error";
  }
  if (typeof err === "string") {
    return err;
  }
  if (err.message) {
    return err.message;
  }
  try {
    return JSON.stringify(err);
  } catch (_) {
    return String(err);
  }
}

function decodeBase64ToBytes(base64) {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

function setAudioBlob(wavBytes) {
  const blob = new Blob([wavBytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  player.src = url;
  download.href = url;
  download.textContent = "Download hello.wav";
}

function resetProgress() {
  progressState.startedAtMs = performance.now();
  progressState.lastEventAtMs = progressState.startedAtMs;
  progressState.arStartedAtMs = 0;
  synthProgress.value = 0;
  synthProgressText.textContent = "Idle (ETA: --)";
}

function formatETA(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "--";
  }
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m > 0) {
    return `${m}m ${rem}s`;
  }
  return `${rem}s`;
}

function updateProgress(evt) {
  const now = performance.now();
  progressState.lastEventAtMs = now;

  const percent = Number(evt?.percent || 0);
  synthProgress.value = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
  const stage = evt?.stage || "working";
  const detail = evt?.detail || "";
  const current = evt?.current ?? 0;
  const total = evt?.total ?? 0;

  let etaSeconds = Number.NaN;
  const stepMatch = /step\s+(\d+)\/(\d+)/i.exec(detail);
  if (stage === "autoregressive" && stepMatch) {
    const stepNow = Number(stepMatch[1]);
    const stepTotal = Number(stepMatch[2]);
    if (progressState.arStartedAtMs === 0) {
      progressState.arStartedAtMs = now;
    }
    if (stepNow > 0 && stepTotal > 0 && stepNow <= stepTotal) {
      const elapsedAr = (now - progressState.arStartedAtMs) / 1000;
      const secPerStep = elapsedAr / stepNow;
      etaSeconds = secPerStep * Math.max(0, stepTotal - stepNow);
    }
  } else if (synthProgress.value > 1) {
    const elapsed = (now - progressState.startedAtMs) / 1000;
    const remainingPercent = Math.max(0, 100 - synthProgress.value);
    etaSeconds = elapsed * (remainingPercent / synthProgress.value);
  }

  synthProgressText.textContent = `${Math.round(synthProgress.value)}% - ${stage}${detail ? ` - ${detail}` : ""}${total ? ` (${current}/${total})` : ""} | ETA: ${formatETA(etaSeconds)}`;
}

function hasRequiredGraphs(manifest) {
  const names = new Set((manifest.graphs || []).map((g) => g.name));
  return requiredModelGraphs.every((name) => names.has(name));
}

globalThis.PocketTTSBridge = {
  runGraph: (graphName, feedsJSON) => onnxBridge.runGraph(graphName, feedsJSON),
};

async function bootKernel() {
  const go = new Go();
  const wasm = await WebAssembly.instantiateStreaming(fetch("./pockettts-kernel.wasm"), go.importObject);
  go.run(wasm.instance);

  if (!globalThis.PocketTTSKernel) {
    throw new Error("PocketTTSKernel not found after wasm startup");
  }

  log.textContent = `Kernel ready: ${globalThis.PocketTTSKernel.version}`;
}

runBtn.addEventListener("click", async () => {
  if (runBtn.disabled) {
    return;
  }
  const input = textArea.value;
  const out = globalThis.PocketTTSKernel.synthesizeWav(input);
  if (!out.ok) {
    log.textContent = `Error: ${out.error}`;
    return;
  }

  const wavBytes = decodeBase64ToBytes(out.wav_base64);
  setAudioBlob(wavBytes);
  log.textContent = `Demo synth complete\nNormalized: ${out.text}\nWAV bytes: ${wavBytes.length}`;
});

verifyBtn.addEventListener("click", async () => {
  if (verifyBtn.disabled) {
    return;
  }
  try {
    log.textContent = "Loading ONNX manifest...";
    const manifest = await onnxBridge.loadManifest();
    const graphs = manifest.graphs || [];
    if (graphs.length === 0) {
      throw new Error("manifest has no graphs");
    }

    const lines = [];
    for (const graph of graphs) {
      const t0 = performance.now();
      await onnxBridge.verifyGraph(graph);
      const dtMs = (performance.now() - t0).toFixed(1);
      lines.push(`PASS ${graph.name} (${dtMs} ms)`);
    }
    log.textContent = `ONNX smoke verify passed (${graphs.length} graphs)\n${lines.join("\n")}`;
  } catch (err) {
    log.textContent = `ONNX smoke verify failed: ${formatError(err)}`;
  }
});

modelSynthBtn.addEventListener("click", async () => {
  if (modelSynthBtn.disabled) {
    return;
  }
  try {
    modelSynthBtn.disabled = true;
    resetProgress();
    log.textContent = "Running Go wasm model synthesis...";

    const t0 = performance.now();
    const out = await globalThis.PocketTTSKernel.synthesizeModel(textArea.value, (evt) => {
      updateProgress(evt);
    });
    const dtMs = (performance.now() - t0).toFixed(1);

    if (!out.ok) {
      throw new Error(out.error || "unknown synthesis error");
    }

    const wavBytes = decodeBase64ToBytes(out.wav_base64);
    setAudioBlob(wavBytes);
    synthProgress.value = 100;
    synthProgressText.textContent = "100% - done | ETA: 0s";

    log.textContent = `Model synth (Go wasm) complete\nNormalized: ${out.text}\nTokens: ${out.token_count}\nFrames: ${out.frames}\nWAV bytes: ${wavBytes.length}\nElapsed: ${dtMs} ms`;
  } catch (err) {
    synthProgressText.textContent = `Failed: ${formatError(err)}`;
    log.textContent = `Model synth failed: ${formatError(err)}`;
  } finally {
    modelSynthBtn.disabled = false;
  }
});

function setCapabilityButtons() {
  runBtn.disabled = !capabilities.kernelReady;
  verifyBtn.disabled = !capabilities.modelReady;
  modelSynthBtn.disabled = !capabilities.modelReady;
}

async function detectCapabilities() {
  capabilities.reasons = [];

  try {
    await bootKernel();
    const selfTest = globalThis.PocketTTSKernel.synthesizeWav("hello");
    if (!selfTest?.ok) {
      throw new Error(selfTest?.error || "kernel self-test failed");
    }
    capabilities.kernelReady = true;
  } catch (err) {
    capabilities.kernelReady = false;
    capabilities.reasons.push(`kernel unavailable: ${formatError(err)}`);
  }

  try {
    const manifest = await onnxBridge.loadManifest();
    capabilities.modelManifestReady = true;
    if (!hasRequiredGraphs(manifest)) {
      capabilities.modelGraphsReady = false;
      capabilities.reasons.push(`manifest missing required graphs: ${requiredModelGraphs.join(", ")}`);
    } else {
      capabilities.modelGraphsReady = true;
    }
  } catch (err) {
    capabilities.modelManifestReady = false;
    capabilities.modelGraphsReady = false;
    capabilities.reasons.push(`model bundle unavailable: ${formatError(err)}`);
  }

  capabilities.modelReady = capabilities.kernelReady && capabilities.modelManifestReady && capabilities.modelGraphsReady;
  setCapabilityButtons();
}

function renderCapabilityStatus() {
  const lines = [];
  lines.push(`Kernel: ${capabilities.kernelReady ? "READY" : "UNAVAILABLE"}`);
  lines.push(`Model manifest: ${capabilities.modelManifestReady ? "READY" : "UNAVAILABLE"}`);
  lines.push(`Required model graphs: ${capabilities.modelGraphsReady ? "READY" : "UNAVAILABLE"}`);
  lines.push(`Model synth: ${capabilities.modelReady ? "ENABLED" : "DISABLED"}`);

  if (capabilities.reasons.length > 0) {
    lines.push("");
    lines.push("Details:");
    for (const reason of capabilities.reasons) {
      lines.push(`- ${reason}`);
    }
  }

  lines.push("");
  lines.push("Enabled actions:");
  if (!runBtn.disabled) {
    lines.push("- Generate Fallback Tone WAV");
  }
  if (!verifyBtn.disabled) {
    lines.push("- Verify ONNX Models");
  }
  if (!modelSynthBtn.disabled) {
    lines.push("- Synthesize via ONNX (Exp)");
  }
  if (runBtn.disabled && verifyBtn.disabled && modelSynthBtn.disabled) {
    lines.push("- none");
  }

  log.textContent = lines.join("\n");
}

async function initApp() {
  resetProgress();
  await detectCapabilities();
  renderCapabilityStatus();
}

initApp().catch((err) => {
  log.textContent = `Startup checks failed: ${formatError(err)}`;
});
