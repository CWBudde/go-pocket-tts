import { ONNXBridge } from "./bridge.js";

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

const onnxBridge = new ONNXBridge({ manifestPath: "./models/manifest.json" });
const requiredGraphs = ["text_conditioner", "flow_lm_main", "flow_lm_flow", "latent_to_mimi", "mimi_decoder"];

const state = {
  kernelReady: false,
  kernelVersion: "",
  manifestReady: false,
  manifest: null,
  modelLoaded: false,
  activeAudioURL: "",
  voiceManifest: null,
  voiceBytes: new Map(),
};

const modelConfig = {
  temperature: 0.7,
};

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

function hasRequiredGraphs(manifest) {
  const names = new Set((manifest?.graphs || []).map((g) => g.name));
  return requiredGraphs.every((name) => names.has(name));
}

function getGraphByName(manifest, name) {
  return (manifest?.graphs || []).find((g) => g.name === name) || null;
}

function graphUncompressedSizeBytes(graph) {
  const raw = graph?.size_bytes ?? graph?.sizeBytes;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) {
    return 0;
  }
  return Math.floor(n);
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

  state.kernelReady = true;
  state.kernelVersion = String(kernel.version || "unknown");
}

async function detectManifest() {
  const manifest = await onnxBridge.loadManifest();
  state.manifest = manifest;
  state.manifestReady = true;
}

function renderStartupInfo() {
  const graphNames = (state.manifest?.graphs || []).map((g) => g.name);

  const lines = [
    `Kernel: ${state.kernelReady ? `ready (${state.kernelVersion})` : "unavailable"}`,
    `Manifest: ${state.manifestReady ? "found" : "not found"}`,
    `Required graphs: ${state.manifestReady && hasRequiredGraphs(state.manifest) ? "ready" : "missing"}`,
    `Loaded: ${state.modelLoaded ? "yes" : "no"}`,
    "Runtime: onnxruntime-web (WASM provider)",
    "Pipeline: Go wasm orchestration (CLI-aligned)",
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
  synthBtn.disabled = !(state.kernelReady && state.modelLoaded);
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
    synthProgressText.textContent = "0% - loading manifest...";

    if (!state.manifestReady) {
      await detectManifest();
    }

    if (!hasRequiredGraphs(state.manifest)) {
      throw new Error(`manifest missing required graphs: ${requiredGraphs.join(", ")}`);
    }

    const graphEntries = requiredGraphs.map((name) => {
      const graph = getGraphByName(state.manifest, name);
      return {
        name,
        sizeBytes: graphUncompressedSizeBytes(graph),
      };
    });
    const totalUncompressedBytes = graphEntries.reduce((sum, g) => sum + g.sizeBytes, 0);
    const allUncompressedSizesKnown = graphEntries.every((g) => g.sizeBytes > 0);

    const total = graphEntries.length;
    let loadedUncompressedBytes = 0;
    for (let i = 0; i < total; i += 1) {
      const { name, sizeBytes } = graphEntries[i];
      const basePercent = (i / total) * 100;
      const slicePercent = 100 / total;

      await onnxBridge.getSession(name, (received, contentLength) => {
        const rx = Number(received) > 0 ? Number(received) : 0;
        const transferTotal = Number(contentLength) > 0 ? Number(contentLength) : 0;
        const graphTotalBytes = sizeBytes > 0 ? sizeBytes : transferTotal;
        const graphReceivedBytes = graphTotalBytes > 0 ? Math.min(rx, graphTotalBytes) : rx;

        let percent;
        if (allUncompressedSizesKnown && totalUncompressedBytes > 0) {
          percent = Math.round(((loadedUncompressedBytes + graphReceivedBytes) / totalUncompressedBytes) * 100);
        } else {
          const dlFrac = graphTotalBytes > 0 ? graphReceivedBytes / graphTotalBytes : 1;
          percent = Math.round(basePercent + dlFrac * slicePercent);
        }

        const boundedPercent = Math.max(0, Math.min(100, percent));
        synthProgress.value = boundedPercent;
        if (graphTotalBytes > 0) {
          const mb = (graphReceivedBytes / 1048576).toFixed(1);
          const totalMb = (graphTotalBytes / 1048576).toFixed(1);
          synthProgressText.textContent = `${boundedPercent}% - downloading ${name} (${mb}/${totalMb} MB) [${i + 1}/${total}]`;
        } else {
          const mb = (graphReceivedBytes / 1048576).toFixed(1);
          synthProgressText.textContent = `${boundedPercent}% - downloading ${name} (${mb} MB) [${i + 1}/${total}]`;
        }
      });

      if (allUncompressedSizesKnown && totalUncompressedBytes > 0) {
        loadedUncompressedBytes += sizeBytes;
      }

      const initPercent = allUncompressedSizesKnown && totalUncompressedBytes > 0
        ? Math.round((loadedUncompressedBytes / totalUncompressedBytes) * 100)
        : Math.round(basePercent + slicePercent);
      synthProgress.value = initPercent;
      synthProgressText.textContent = `${initPercent}% - initialized ${name} [${i + 1}/${total}]`;
    }

    synthProgress.value = 100;
    synthProgressText.textContent = "100% - all graphs loaded";

    state.modelLoaded = true;
    renderModelState();
    setSynthesizeEnabled();
    renderStartupInfo();
    log.textContent = `Model loaded (${requiredGraphs.length} graph sessions).`;
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

function setupBridge() {
  globalThis.PocketTTSBridge = {
    runGraph: (graphName, feedsJSON) => onnxBridge.runGraph(graphName, feedsJSON),
  };
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
