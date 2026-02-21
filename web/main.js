import { ONNXBridge } from "./bridge.js";

const startupInfo = document.getElementById("startup-info");
const log = document.getElementById("log");
const textArea = document.getElementById("text");
const loadModelBtn = document.getElementById("load-model");
const engineSelect = document.getElementById("engine");
const modelState = document.getElementById("model-state");
const synthBtn = document.getElementById("synthesize");
const player = document.getElementById("player");
const download = document.getElementById("download");
const synthProgress = document.getElementById("synth-progress");
const synthProgressText = document.getElementById("synth-progress-text");

const onnxBridge = new ONNXBridge({ manifestPath: "./models/manifest.json" });
const requiredGraphs = ["text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"];
const optionalGraphs = ["latent_to_mimi"];

const state = {
  kernelReady: false,
  kernelVersion: "",
  manifestReady: false,
  manifest: null,
  modelLoaded: false,
  activeAudioURL: "",
};

const modelConfig = {
  sampleRate: 24000,
  flowTemperature: 0.7,
  flowDecodeSteps: 1,
  flowEOSThreshold: -4.0,
  flowFramesAfterEOS: 2,
  flowMinSteps: 16,
  flowMaxSteps: 256,
  latentDim: 32,
  condDim: 1024,
  mimiDimFallback: 512,
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

function encodeWavPCM16(samples, sampleRate) {
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);

  function writeASCII(offset, text) {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  writeASCII(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  writeASCII(8, "WAVE");
  writeASCII(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeASCII(36, "data");
  view.setUint32(40, n * 2, true);

  for (let i = 0; i < n; i += 1) {
    const x = Math.max(-1, Math.min(1, Number(samples[i] || 0)));
    const q = x < 0 ? Math.round(x * 0x8000) : Math.round(x * 0x7fff);
    view.setInt16(44 + i * 2, q, true);
  }

  return new Uint8Array(buffer);
}

function setAudioBlob(wavBytes) {
  if (state.activeAudioURL) {
    URL.revokeObjectURL(state.activeAudioURL);
  }

  const blob = new Blob([wavBytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  state.activeAudioURL = url;

  player.src = url;
  download.href = url;
}

function resetProgress() {
  synthProgress.value = 0;
  synthProgressText.textContent = "Idle";
}

function updateProgress(evt) {
  const percent = Math.max(0, Math.min(100, Number(evt?.percent || 0)));
  const stage = evt?.stage || "working";
  const detail = evt?.detail ? ` - ${evt.detail}` : "";
  synthProgress.value = percent;
  synthProgressText.textContent = `${Math.round(percent)}% - ${stage}${detail}`;
}

function normalizePromptForModel(s) {
  s = String(s || "").trim();
  if (!s) {
    return "";
  }
  s = s.replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\n/g, " ");
  s = s.split(/\s+/).filter(Boolean).join(" ");
  if (!s) {
    return "";
  }
  if (!/[.!?]$/.test(s)) {
    s += ".";
  }
  return s;
}

function copyOrTile(src, n) {
  const out = new Float32Array(n);
  if (!src || src.length === 0) {
    return out;
  }
  for (let i = 0; i < n; i += 1) {
    out[i] = Number(src[i % src.length] || 0);
  }
  return out;
}

function flattenFrames(frames, frameDim) {
  const out = new Float32Array(frames.length * frameDim);
  let w = 0;
  for (const frame of frames) {
    const row = copyOrTile(frame, frameDim);
    out.set(row, w);
    w += frameDim;
  }
  return out;
}

function randomNormal() {
  let u = 0;
  let v = 0;
  while (u === 0) {
    u = Math.random();
  }
  while (v === 0) {
    v = Math.random();
  }
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function pickOutput(outputs, preferred) {
  if (preferred && outputs[preferred]) {
    return outputs[preferred];
  }
  const first = Object.values(outputs)[0];
  if (!first) {
    throw new Error("graph returned no outputs");
  }
  return first;
}

function hasRequiredGraphs(manifest) {
  const names = new Set((manifest?.graphs || []).map((g) => g.name));
  return requiredGraphs.every((name) => names.has(name));
}

function graphExists(name) {
  return (state.manifest?.graphs || []).some((g) => g.name === name);
}

async function bootKernel() {
  const go = new Go();
  const wasm = await WebAssembly.instantiateStreaming(fetch("./pockettts-kernel.wasm"), go.importObject);
  go.run(wasm.instance);

  const kernel = globalThis.PocketTTSKernel;
  if (!kernel) {
    throw new Error("PocketTTSKernel not found after wasm startup");
  }

  state.kernelReady = true;
  state.kernelVersion = String(kernel.version || "unknown");
}

async function detectManifest() {
  const manifest = await onnxBridge.loadManifest();
  state.manifest = manifest;
  state.manifestReady = true;
}

function renderStartupInfo() {
  const engine = engineSelect.value === "go" ? "Go WASM" : "JavaScript";
  const graphNames = (state.manifest?.graphs || []).map((g) => g.name);

  const lines = [
    `Kernel: ${state.kernelReady ? `ready (${state.kernelVersion})` : "unavailable"}`,
    `Manifest: ${state.manifestReady ? "found" : "not found"}`,
    `Required graphs: ${state.manifestReady && hasRequiredGraphs(state.manifest) ? "ready" : "missing"}`,
    `Loaded: ${state.modelLoaded ? "yes" : "no"}`,
    `Engine: ${engine}`,
    `Runtime: onnxruntime-web (WASM provider)`,
  ];

  if (graphNames.length > 0) {
    lines.push(`Graphs: ${graphNames.join(", ")}`);
  }

  startupInfo.textContent = lines.join("\n");
}

function renderModelState() {
  modelState.textContent = state.modelLoaded ? "Model: loaded" : "Model: not loaded";
}

function setSynthesizeEnabled() {
  const engine = engineSelect.value;
  const kernelOk = state.kernelReady;
  const modelOk = state.modelLoaded;
  const enabled = engine === "go" ? kernelOk && modelOk : kernelOk && modelOk;
  synthBtn.disabled = !enabled;
}

async function runGraph(name, feeds) {
  return onnxBridge.runGraphPayload(name, feeds);
}

async function tokenizeForModel(inputText) {
  if (!state.kernelReady) {
    throw new Error("Go WASM kernel not ready (needed for tokenizer)");
  }

  const prepared = normalizePromptForModel(inputText);
  if (!prepared) {
    throw new Error("text is empty");
  }

  const tokenized = globalThis.PocketTTSKernel.tokenize(prepared);
  if (!tokenized?.ok) {
    throw new Error(tokenized?.error || "tokenization failed");
  }

  const tokens = Array.isArray(tokenized.tokens)
    ? tokenized.tokens.map((v) => Number(v)).filter((v) => Number.isSafeInteger(v))
    : [];

  if (tokens.length === 0) {
    throw new Error("no tokens produced after preprocessing");
  }

  return {
    normalized: String(tokenized.text || prepared),
    tokens,
  };
}

async function synthesizeModelJS(inputText, onProgress) {
  const cfg = modelConfig;
  const emit = (stage, current, total, detail) => {
    const percent = total > 0 ? (current / total) * 100 : 0;
    onProgress({ stage, current, total, percent, detail });
  };

  emit("prepare", 0, 100, "normalizing and tokenizing input");
  const prep = await tokenizeForModel(inputText);
  emit("prepare", 5, 100, `tokenized ${prep.tokens.length} units`);

  emit("text_conditioner", 10, 100, "running text conditioner");
  const textCondOutputs = await runGraph("text_conditioner", {
    tokens: {
      dtype: "int64",
      shape: [1, prep.tokens.length],
      i64: prep.tokens,
    },
  });
  const textEmb = pickOutput(textCondOutputs, "text_embeddings");
  emit("text_conditioner", 15, 100, "text embeddings ready");

  const sequenceFrames = [new Float32Array(cfg.latentDim)];
  const generated = [];

  let eosStep = -1;
  let maxSteps = cfg.flowMaxSteps;
  if (maxSteps < prep.tokens.length * 5) {
    maxSteps = prep.tokens.length * 5;
  }
  if (maxSteps > cfg.flowMaxSteps) {
    maxSteps = cfg.flowMaxSteps;
  }

  emit("autoregressive", 20, 100, `starting generation loop (${maxSteps} max steps)`);

  for (let step = 0; step < maxSteps; step += 1) {
    if (step === 0 || step % 2 === 0) {
      emit("autoregressive", 20 + Math.floor((step / maxSteps) * 60), 100, `step ${step + 1}/${maxSteps}`);
    }

    const seqFlat = flattenFrames(sequenceFrames, cfg.latentDim);
    const mainOutputs = await runGraph("flow_lm_main", {
      sequence: {
        dtype: "float32",
        shape: [1, sequenceFrames.length, cfg.latentDim],
        f32: Array.from(seqFlat),
      },
      text_embeddings: textEmb,
    });

    const hidden = pickOutput(mainOutputs, "last_hidden");
    const eos = pickOutput(mainOutputs, "eos_logits");

    const condition = copyOrTile(hidden.f32 || [], cfg.condDim);
    const x = new Float32Array(cfg.latentDim);
    for (let i = 0; i < x.length; i += 1) {
      x[i] = randomNormal() * Math.sqrt(cfg.flowTemperature);
    }

    for (let k = 0; k < cfg.flowDecodeSteps; k += 1) {
      const s = k / cfg.flowDecodeSteps;
      const t = (k + 1) / cfg.flowDecodeSteps;

      const flowOut = await runGraph("flow_lm_flow", {
        condition: {
          dtype: "float32",
          shape: [1, cfg.condDim],
          f32: Array.from(condition),
        },
        s: {
          dtype: "float32",
          shape: [1, 1],
          f32: [s],
        },
        t: {
          dtype: "float32",
          shape: [1, 1],
          f32: [t],
        },
        x: {
          dtype: "float32",
          shape: [1, cfg.latentDim],
          f32: Array.from(x),
        },
      });

      const dir = pickOutput(flowOut, "flow_direction");
      const dirF32 = dir.f32 || [];
      const dt = 1.0 / cfg.flowDecodeSteps;
      for (let i = 0; i < x.length; i += 1) {
        x[i] += Number(dirF32[i % dirF32.length] || 0) * dt;
      }
    }

    sequenceFrames.push(x.slice(0));
    generated.push(x.slice(0));

    const eosVal = Array.isArray(eos.f32) && eos.f32.length > 0 ? Number(eos.f32[0]) : -10;
    if (eosVal > cfg.flowEOSThreshold && eosStep < 0) {
      eosStep = step;
    }
    if (eosStep >= 0 && step >= eosStep + cfg.flowFramesAfterEOS && step + 1 >= cfg.flowMinSteps) {
      emit("autoregressive", 80, 100, `stopped at step ${step + 1} (EOS+tail)`);
      break;
    }
  }

  if (generated.length === 0) {
    throw new Error("model loop produced no latents");
  }

  const steps = generated.length;
  const latentSeq = flattenFrames(generated, cfg.latentDim);
  emit("latent_to_mimi", 85, 100, "projecting latent for Mimi decoder");

  let mimiInput;
  if (graphExists("latent_to_mimi")) {
    const latentToMimi = await runGraph("latent_to_mimi", {
      latent: {
        dtype: "float32",
        shape: [1, steps, cfg.latentDim],
        f32: Array.from(latentSeq),
      },
    });
    mimiInput = pickOutput(latentToMimi, "mimi_latent");
  } else {
    const fallback = new Float32Array(cfg.mimiDimFallback * steps);
    for (let t = 0; t < steps; t += 1) {
      const src = generated[t];
      for (let c = 0; c < cfg.mimiDimFallback; c += 1) {
        fallback[c * steps + t] = Number(src[c % src.length] || 0);
      }
    }
    mimiInput = {
      dtype: "float32",
      shape: [1, cfg.mimiDimFallback, steps],
      f32: Array.from(fallback),
    };
  }

  const mimiOutputs = await runGraph("mimi_decoder", {
    latent: mimiInput,
  });
  const audioOut = pickOutput(mimiOutputs, "audio");

  if (!Array.isArray(audioOut.f32) || audioOut.f32.length === 0) {
    throw new Error("decoder returned empty audio");
  }

  emit("mimi_decoder", 95, 100, "encoding WAV");
  const wavBytes = encodeWavPCM16(audioOut.f32, cfg.sampleRate);
  emit("done", 100, 100, "synthesis complete");

  return {
    ok: true,
    text: prep.normalized,
    token_count: prep.tokens.length,
    frames: generated.length,
    sample_rate: cfg.sampleRate,
    wav_bytes: wavBytes,
  };
}

async function synthesizeWithGo(inputText) {
  if (!state.kernelReady) {
    throw new Error("Go WASM kernel not ready");
  }

  const out = await globalThis.PocketTTSKernel.synthesizeModel(inputText, (evt) => {
    updateProgress(evt);
  });

  if (!out?.ok) {
    throw new Error(out?.error || "synthesis failed");
  }

  return {
    ...out,
    wav_bytes: decodeBase64ToBytes(out.wav_base64 || ""),
  };
}

async function handleLoadModel() {
  try {
    loadModelBtn.disabled = true;
    modelState.textContent = "Model: loading...";

    if (!state.manifestReady) {
      await detectManifest();
    }

    if (!hasRequiredGraphs(state.manifest)) {
      throw new Error(`manifest missing required graphs: ${requiredGraphs.join(", ")}`);
    }

    const toPreload = [...requiredGraphs, ...optionalGraphs.filter((name) => graphExists(name))];
    await onnxBridge.preloadGraphs(toPreload);

    state.modelLoaded = true;
    renderModelState();
    setSynthesizeEnabled();
    renderStartupInfo();
    log.textContent = `Model loaded (${toPreload.length} graph sessions).`;
  } catch (err) {
    state.modelLoaded = false;
    renderModelState();
    setSynthesizeEnabled();
    renderStartupInfo();
    log.textContent = `Load model failed: ${formatError(err)}`;
  } finally {
    loadModelBtn.disabled = false;
  }
}

async function handleSynthesize() {
  if (synthBtn.disabled) {
    return;
  }

  const engine = engineSelect.value;
  try {
    synthBtn.disabled = true;
    resetProgress();
    log.textContent = `Synthesis started (${engine === "go" ? "Go WASM" : "JavaScript"} engine)...`;

    const t0 = performance.now();
    const result =
      engine === "go"
        ? await synthesizeWithGo(textArea.value)
        : await synthesizeModelJS(textArea.value, (evt) => updateProgress(evt));
    const elapsedMs = (performance.now() - t0).toFixed(1);

    setAudioBlob(result.wav_bytes);
    synthProgress.value = 100;
    synthProgressText.textContent = "100% - done";

    log.textContent = [
      `Synthesis complete (${engine === "go" ? "Go WASM" : "JavaScript"} engine)`,
      `Normalized: ${result.text}`,
      `Tokens: ${result.token_count}`,
      `Frames: ${result.frames}`,
      `Sample rate: ${result.sample_rate}`,
      `WAV bytes: ${result.wav_bytes.length}`,
      `Elapsed: ${elapsedMs} ms`,
    ].join("\n");
  } catch (err) {
    synthProgressText.textContent = `Failed: ${formatError(err)}`;
    log.textContent = `Synthesis failed: ${formatError(err)}`;
  } finally {
    setSynthesizeEnabled();
  }
}

function setupBridge() {
  globalThis.PocketTTSBridge = {
    runGraph: (graphName, feedsJSON) => onnxBridge.runGraph(graphName, feedsJSON),
  };
}

async function initApp() {
  setupBridge();
  resetProgress();
  renderModelState();
  setSynthesizeEnabled();
  renderStartupInfo();

  try {
    await bootKernel();
  } catch (err) {
    state.kernelReady = false;
    log.textContent = `Kernel startup failed: ${formatError(err)}`;
  }

  try {
    await detectManifest();
  } catch (err) {
    state.manifestReady = false;
    log.textContent = `Manifest check: ${formatError(err)}`;
  }

  renderStartupInfo();
  setSynthesizeEnabled();
}

loadModelBtn.addEventListener("click", handleLoadModel);
synthBtn.addEventListener("click", handleSynthesize);
engineSelect.addEventListener("change", () => {
  setSynthesizeEnabled();
  renderStartupInfo();
});

initApp().catch((err) => {
  log.textContent = `Startup failed: ${formatError(err)}`;
});
