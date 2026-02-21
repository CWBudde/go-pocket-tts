import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.mjs";

const log = document.getElementById("log");
const textArea = document.getElementById("text");
const runBtn = document.getElementById("run");
const verifyBtn = document.getElementById("verify-models");
const modelSynthBtn = document.getElementById("synthesize-model");
const player = document.getElementById("player");
const download = document.getElementById("download");
const synthProgress = document.getElementById("synth-progress");
const synthProgressText = document.getElementById("synth-progress-text");

let manifestCache = null;
const sessionCache = new Map();
const requiredModelGraphs = ["text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"];

const capabilities = {
  kernelReady: false,
  modelManifestReady: false,
  modelGraphsReady: false,
  modelReady: false,
  reasons: [],
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
  synthProgress.value = 0;
  synthProgressText.textContent = "Idle";
}

function updateProgress(evt) {
  const percent = Number(evt?.percent || 0);
  synthProgress.value = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
  const stage = evt?.stage || "working";
  const detail = evt?.detail || "";
  const current = evt?.current ?? 0;
  const total = evt?.total ?? 0;
  synthProgressText.textContent = `${Math.round(synthProgress.value)}% - ${stage}${detail ? ` - ${detail}` : ""}${total ? ` (${current}/${total})` : ""}`;
}

async function loadManifest() {
  if (manifestCache) {
    return manifestCache;
  }
  const res = await fetch("./models/manifest.json");
  if (!res.ok) {
    throw new Error(`manifest fetch failed (${res.status}) at ./models/manifest.json`);
  }
  manifestCache = await res.json();
  return manifestCache;
}

function hasRequiredGraphs(manifest) {
  const names = new Set((manifest.graphs || []).map((g) => g.name));
  return requiredModelGraphs.every((name) => names.has(name));
}

async function getGraph(graphName) {
  const manifest = await loadManifest();
  const graph = (manifest.graphs || []).find((g) => g.name === graphName);
  if (!graph) {
    throw new Error(`graph not found in manifest: ${graphName}`);
  }
  return graph;
}

function ortTensorFromPayload(payload) {
  const shape = payload.shape || [];
  const dtype = (payload.dtype || "").toLowerCase();

  if (dtype === "float" || dtype === "float32") {
    return new ort.Tensor("float32", new Float32Array(payload.f32 || []), shape);
  }
  if (dtype === "int64") {
    const src = payload.i64 || [];
    return new ort.Tensor("int64", BigInt64Array.from(src.map((v) => BigInt(v))), shape);
  }
  if (dtype === "int32") {
    return new ort.Tensor("int32", new Int32Array(payload.i32 || []), shape);
  }

  throw new Error(`unsupported tensor dtype: ${payload.dtype}`);
}

function payloadFromTensor(tensor) {
  const out = {
    dtype: tensor.type,
    shape: tensor.dims || [],
  };

  if (tensor.data instanceof Float32Array) {
    out.f32 = Array.from(tensor.data);
    return out;
  }
  if (tensor.data instanceof BigInt64Array) {
    out.i64 = Array.from(tensor.data, (v) => Number(v));
    return out;
  }
  if (tensor.data instanceof Int32Array) {
    out.i32 = Array.from(tensor.data);
    return out;
  }

  if (ArrayBuffer.isView(tensor.data)) {
    out.f32 = Array.from(tensor.data, (v) => Number(v));
    out.dtype = "float32";
    return out;
  }

  throw new Error("unsupported tensor data view");
}

async function getSession(graphName) {
  if (sessionCache.has(graphName)) {
    return sessionCache.get(graphName);
  }

  const graph = await getGraph(graphName);
  const session = await ort.InferenceSession.create(`./models/${graph.filename}`, {
    executionProviders: ["wasm"],
  });

  const entry = { graph, session };
  sessionCache.set(graphName, entry);
  return entry;
}

async function runGraph(graphName, feedsJSON) {
  const feeds = JSON.parse(feedsJSON);
  const { session } = await getSession(graphName);

  const ortFeeds = {};
  for (const [name, payload] of Object.entries(feeds)) {
    ortFeeds[name] = ortTensorFromPayload(payload);
  }

  const outputs = await session.run(ortFeeds);
  const serialized = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    serialized[name] = payloadFromTensor(tensor);
  }

  return JSON.stringify(serialized);
}

async function verifyGraph(graph) {
  const session = await ort.InferenceSession.create(`./models/${graph.filename}`, {
    executionProviders: ["wasm"],
  });

  const feeds = {};
  for (const input of graph.inputs || []) {
    const shape = (input.shape || []).map((d) => (typeof d === "number" && d > 0 ? d : 1));
    const size = shape.reduce((a, b) => a * b, 1);
    const dtype = (input.dtype || "").toLowerCase();

    if (dtype === "int64") {
      feeds[input.name] = new ort.Tensor("int64", new BigInt64Array(size), shape);
    } else {
      feeds[input.name] = new ort.Tensor("float32", new Float32Array(size), shape);
    }
  }

  await session.run(feeds);
}

globalThis.PocketTTSBridge = {
  runGraph,
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
    const manifest = await loadManifest();
    const graphs = manifest.graphs || [];
    if (graphs.length === 0) {
      throw new Error("manifest has no graphs");
    }

    const lines = [];
    for (const graph of graphs) {
      const t0 = performance.now();
      await verifyGraph(graph);
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
    synthProgressText.textContent = "100% - done";

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
    capabilities.reasons.push(`kernel unavailable: ${err.message}`);
  }

  try {
    const manifest = await loadManifest();
    capabilities.modelManifestReady = true;
    if (!hasRequiredGraphs(manifest)) {
      capabilities.modelGraphsReady = false;
      capabilities.reasons.push(
        `manifest missing required graphs: ${requiredModelGraphs.join(", ")}`
      );
    } else {
      capabilities.modelGraphsReady = true;
    }
  } catch (err) {
    capabilities.modelManifestReady = false;
    capabilities.modelGraphsReady = false;
    capabilities.reasons.push(`model bundle unavailable: ${err.message}`);
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
