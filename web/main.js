const startupInfo = document.getElementById("startup-info");
const log = document.getElementById("log");
const textArea = document.getElementById("text");
const loadModelBtn = document.getElementById("load-model");
const modelState = document.getElementById("model-state");
const synthBtn = document.getElementById("synthesize");
const player = document.getElementById("player");
const download = document.getElementById("download");
const synthProgress = document.getElementById("synth-progress");
const synthProgressText = document.getElementById("synth-progress-text");
const voiceSelect = document.getElementById("voice");
const temperatureSlider = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");

const state = {
  kernelReady: false,
  kernelVersion: "",
  modelLoaded: false,
  activeAudioURL: "",
  voiceManifest: null,
  voiceBytes: new Map(),
};

const modelConfig = {
  temperature: 0.7,
};

const modelAssetPath = "./models/tts_b6369a24.safetensors";

function formatError(err) {
  if (!err) return "unknown error";
  if (typeof err === "string") return err;
  if (err.message) return err.message;
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

function renderStartupInfo() {
  const lines = [
    `Kernel: ${state.kernelReady ? `ready (${state.kernelVersion})` : "unavailable"}`,
    `Model asset: ${modelAssetPath}`,
    `Loaded: ${state.modelLoaded ? "yes" : "no"}`,
    "Runtime: Go native-safetensors (pure wasm)",
    "Pipeline: Go wasm orchestration (CLI-aligned)",
  ];

  startupInfo.textContent = lines.join("\n");
}

function renderModelState() {
  modelState.textContent = state.modelLoaded ? "Model: loaded" : "Model: not loaded";
}

function setSynthesizeEnabled() {
  synthBtn.disabled = !(state.kernelReady && state.modelLoaded);
}

async function fetchBytesWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch ${url} failed (${res.status})`);
  }

  const contentLength = parseInt(res.headers.get("content-length") || "0", 10);
  if (!res.body) {
    const buf = await res.arrayBuffer();
    if (onProgress) onProgress(buf.byteLength, buf.byteLength);
    return new Uint8Array(buf);
  }

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received, contentLength);
  }

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  if (onProgress) onProgress(received, contentLength || received);
  return out;
}

async function bootKernel() {
  const go = new Go();
  const wasm = await WebAssembly.instantiateStreaming(fetch("./pockettts-kernel.wasm"), go.importObject);
  go.run(wasm.instance);

  const kernel = globalThis.PocketTTSKernel;
  if (!kernel) {
    throw new Error("PocketTTSKernel not found after wasm startup");
  }
  if (typeof kernel.synthesize !== "function") {
    throw new Error("PocketTTSKernel.synthesize is missing");
  }
  if (typeof kernel.loadModel !== "function") {
    throw new Error("PocketTTSKernel.loadModel is missing");
  }

  state.kernelReady = true;
  state.kernelVersion = String(kernel.version || "unknown");
}

async function getSelectedVoiceSafetensors() {
  const voiceId = voiceSelect.value;
  if (!voiceId) return null;

  if (state.voiceBytes.has(voiceId)) {
    return state.voiceBytes.get(voiceId);
  }

  const voice = (state.voiceManifest?.voices || []).find((v) => v.id === voiceId);
  if (!voice) return null;

  const res = await fetch(`./voices/${voice.path}`);
  if (!res.ok) {
    throw new Error(`fetch voice ${voice.path} failed (${res.status})`);
  }

  const bytes = new Uint8Array(await res.arrayBuffer());
  state.voiceBytes.set(voiceId, bytes);
  return bytes;
}

async function handleLoadModel() {
  try {
    loadModelBtn.disabled = true;
    modelState.textContent = "Model: loading...";
    synthProgress.value = 0;
    synthProgressText.textContent = "0% - downloading model...";

    const modelBytes = await fetchBytesWithProgress(modelAssetPath, (received, total) => {
      const frac = total > 0 ? Math.min(1, received / total) : 0;
      const percent = Math.round(frac * 80);
      synthProgress.value = percent;

      const mb = (received / 1048576).toFixed(1);
      if (total > 0) {
        const totalMb = (total / 1048576).toFixed(1);
        synthProgressText.textContent = `${percent}% - downloading model (${mb}/${totalMb} MB)`;
      } else {
        synthProgressText.textContent = `${percent}% - downloading model (${mb} MB)`;
      }
    });

    const result = await globalThis.PocketTTSKernel.loadModel(modelBytes, (evt) => {
      const stagePercent = Math.max(0, Math.min(100, Number(evt?.percent || 0)));
      const percent = 80 + Math.round(stagePercent * 0.2);
      synthProgress.value = percent;

      const detail = evt?.detail ? ` - ${evt.detail}` : "";
      synthProgressText.textContent = `${percent}% - initializing${detail}`;
    });

    if (!result?.ok) {
      throw new Error(result?.error || "model load failed");
    }

    synthProgress.value = 100;
    synthProgressText.textContent = "100% - model loaded";

    state.modelLoaded = true;
    renderModelState();
    setSynthesizeEnabled();
    renderStartupInfo();
    log.textContent = `Model loaded (${(modelBytes.length / 1048576).toFixed(1)} MB safetensors).`;
  } catch (err) {
    state.modelLoaded = false;
    renderModelState();
    setSynthesizeEnabled();
    renderStartupInfo();
    synthProgressText.textContent = `Load failed: ${formatError(err)}`;
    log.textContent = `Load model failed: ${formatError(err)}`;
  } finally {
    loadModelBtn.disabled = false;
  }
}

async function handleSynthesize() {
  if (synthBtn.disabled) {
    return;
  }

  try {
    synthBtn.disabled = true;
    resetProgress();
    log.textContent = "Synthesis started (Go WASM kernel)...";

    const options = { temperature: modelConfig.temperature };
    const voiceSafetensors = await getSelectedVoiceSafetensors();
    if (voiceSafetensors) {
      options.voiceSafetensors = voiceSafetensors;
    }

    const t0 = performance.now();
    const result = await globalThis.PocketTTSKernel.synthesize(textArea.value, (evt) => {
      updateProgress(evt);
    }, options);
    const elapsedMs = (performance.now() - t0).toFixed(1);

    if (!result?.ok) {
      throw new Error(result?.error || "synthesis failed");
    }

    const wavBytes = decodeBase64ToBytes(result.wav_base64 || "");
    setAudioBlob(wavBytes);

    synthProgress.value = 100;
    synthProgressText.textContent = "100% - done";

    log.textContent = [
      "Synthesis complete (Go WASM kernel)",
      `Normalized: ${result.text}`,
      `Tokens: ${result.token_count}`,
      `Chunks: ${result.chunk_count}`,
      `Sample rate: ${result.sample_rate}`,
      `Samples: ${result.sample_count}`,
      `WAV bytes: ${wavBytes.length}`,
      `Elapsed: ${elapsedMs} ms`,
    ].join("\n");
  } catch (err) {
    synthProgressText.textContent = `Failed: ${formatError(err)}`;
    log.textContent = `Synthesis failed: ${formatError(err)}`;
  } finally {
    setSynthesizeEnabled();
  }
}

function populateVoiceDropdown() {
  const voices = state.voiceManifest?.voices || [];
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.id;
    voiceSelect.appendChild(opt);
  }
}

async function initApp() {
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
    const res = await fetch("./voices/manifest.json");
    if (res.ok) {
      state.voiceManifest = await res.json();
      populateVoiceDropdown();
    }
  } catch (_) {
    // Voice manifest is optional.
  }

  renderStartupInfo();
  setSynthesizeEnabled();
}

loadModelBtn.addEventListener("click", handleLoadModel);
synthBtn.addEventListener("click", handleSynthesize);
temperatureSlider.addEventListener("input", () => {
  const val = parseFloat(temperatureSlider.value);
  modelConfig.temperature = val;
  temperatureValue.textContent = val.toFixed(2);
});

initApp().catch((err) => {
  log.textContent = `Startup failed: ${formatError(err)}`;
});
