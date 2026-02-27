const info = document.getElementById("info");
const textArea = document.getElementById("text");
const synthBtn = document.getElementById("synthesize");
const player = document.getElementById("player");
const voiceSelect = document.getElementById("voice");
const temperatureSlider = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const progressWrap = document.getElementById("progress-wrap");
const progressLabel = document.getElementById("progress-label");
const progressFill = document.getElementById("progress-fill");

const state = {
  kernelReady: false,
  kernelVersion: "",
  modelPhase: "idle",
  modelLoaded: false,
  modelError: "",
  modelBytesPromise: null,
  tokenizerBytesPromise: null,
  isSynthesizing: false,
  textDirty: false,
  activeAudioURL: "",
  voiceManifest: null,
  voiceBytes: new Map(),
  voiceDownloads: new Map(),
  voiceErrors: new Map(),
  action: "Starting...",
};

const modelConfig = {
  temperature: 0.7,
};

const modelAssetPath = "./models/tts_b6369a24.safetensors";
const tokenizerAssetPath = "./models/tokenizer.model";
const preferredVoiceID = "alba";

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
  player.style.display = "";
}

function selectedVoiceID() {
  return String(voiceSelect.value || "");
}

function findVoiceByID(voiceID) {
  return (state.voiceManifest?.voices || []).find((v) => v.id === voiceID) || null;
}

function selectedVoiceState() {
  const voiceID = selectedVoiceID();
  if (!voiceID) return "unavailable";
  if (state.voiceBytes.has(voiceID)) return "ready";
  if (state.voiceDownloads.has(voiceID)) return "downloading";
  if (state.voiceErrors.has(voiceID)) return "error";
  if (findVoiceByID(voiceID)) return "missing";
  return "unavailable";
}

function modelStatusText() {
  if (state.modelLoaded) return "ready";
  if (state.modelPhase === "downloading") return "downloading";
  if (state.modelPhase === "initializing") return "initializing";
  if (state.modelPhase === "error") return `error (${state.modelError})`;
  return "idle";
}

function voiceStatusText() {
  const voiceID = selectedVoiceID();
  const voiceState = selectedVoiceState();

  if (!voiceID) return "unavailable";
  if (voiceState === "ready") return `${voiceID} ready`;
  if (voiceState === "downloading") return `${voiceID} downloading`;
  if (voiceState === "missing") return `${voiceID} not downloaded`;
  if (voiceState === "error") return `${voiceID} error (${state.voiceErrors.get(voiceID)})`;
  return `${voiceID} unavailable`;
}

function renderInfo() {
  const lines = [
    `Kernel: ${state.kernelReady ? `ready (${state.kernelVersion})` : "not ready"}`,
    `Model: ${modelStatusText()}`,
    `Voice: ${voiceStatusText()}`,
    `State: ${state.action}`,
  ];
  info.textContent = lines.join("\n");
}

function setAction(message) {
  state.action = message;
  renderInfo();
}

function showProgress(label, pct, done) {
  progressWrap.classList.add("visible");
  progressLabel.textContent = label;
  progressFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  progressFill.classList.toggle("done", !!done);
}

function hideProgress() {
  progressWrap.classList.remove("visible");
}

const btnLabel = document.getElementById("btn-label");

function canSynthesize() {
  const baseReady = state.kernelReady && state.modelLoaded && selectedVoiceState() === "ready";
  // Allow clicking again if the user edited the text while a synthesis is running.
  return baseReady && (!state.isSynthesizing || state.textDirty);
}

function setSynthesizeEnabled() {
  const enabled = canSynthesize();
  const showSpinner = state.isSynthesizing && !state.textDirty;
  synthBtn.disabled = !enabled;
  synthBtn.classList.toggle("ready", enabled && !showSpinner);
  synthBtn.classList.toggle("synthesizing", showSpinner);
  btnLabel.textContent = showSpinner ? "Synthesizing…" : "Synthesize";
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
  renderInfo();
  setSynthesizeEnabled();
}

function populateVoiceDropdown() {
  const voices = state.voiceManifest?.voices || [];
  voiceSelect.innerHTML = "";

  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = v.id;
    voiceSelect.appendChild(opt);
  }

  if (voices.some((v) => v.id === preferredVoiceID)) {
    voiceSelect.value = preferredVoiceID;
    return;
  }

  if (voices.length > 0) {
    voiceSelect.value = voices[0].id;
  }
}

async function ensureVoiceDownloaded(voiceID) {
  if (!voiceID) return null;

  if (state.voiceBytes.has(voiceID)) {
    return state.voiceBytes.get(voiceID);
  }

  if (state.voiceDownloads.has(voiceID)) {
    return state.voiceDownloads.get(voiceID);
  }

  const voice = findVoiceByID(voiceID);
  if (!voice) {
    const msg = `voice ${voiceID} not found in manifest`;
    state.voiceErrors.set(voiceID, msg);
    throw new Error(msg);
  }

  state.voiceErrors.delete(voiceID);

  let lastPct = -1;
  const promise = (async () => {
    const bytes = await fetchBytesWithProgress(`./voices/${voice.path}`, (received, total) => {
      const pct = total > 0 ? Math.min(100, Math.round((received / total) * 100)) : 0;
      if (pct !== lastPct && pct % 10 === 0 && selectedVoiceID() === voiceID) {
        lastPct = pct;
        setAction(`Downloading voice ${voiceID} (${pct}%)...`);
      }
    });
    state.voiceBytes.set(voiceID, bytes);
    return bytes;
  })();

  state.voiceDownloads.set(voiceID, promise);
  renderInfo();
  setSynthesizeEnabled();

  try {
    const bytes = await promise;
    return bytes;
  } catch (err) {
    state.voiceErrors.set(voiceID, formatError(err));
    throw err;
  } finally {
    state.voiceDownloads.delete(voiceID);
    renderInfo();
    setSynthesizeEnabled();
  }
}

async function downloadSelectedVoiceIfNeeded() {
  const voiceID = selectedVoiceID();
  if (!voiceID) {
    renderInfo();
    setSynthesizeEnabled();
    return null;
  }

  if (state.voiceBytes.has(voiceID)) {
    renderInfo();
    setSynthesizeEnabled();
    return state.voiceBytes.get(voiceID);
  }

  setAction(`Downloading voice ${voiceID}...`);
  try {
    const bytes = await ensureVoiceDownloaded(voiceID);
    if (selectedVoiceID() === voiceID) {
      setAction(`Voice ${voiceID} ready.`);
    }
    return bytes;
  } catch (err) {
    if (selectedVoiceID() === voiceID) {
      setAction(`Voice ${voiceID} failed: ${formatError(err)}`);
    }
    return null;
  }
}

function ensureTokenizerBytesFetched() {
  if (state.tokenizerBytesPromise) {
    return state.tokenizerBytesPromise;
  }

  state.tokenizerBytesPromise = fetch(tokenizerAssetPath)
    .then((res) => {
      if (!res.ok) throw new Error(`fetch tokenizer failed (${res.status})`);
      return res.arrayBuffer();
    })
    .then((buf) => new Uint8Array(buf))
    .catch((err) => {
      state.tokenizerBytesPromise = null;
      throw err;
    });

  return state.tokenizerBytesPromise;
}

function ensureModelBytesFetched() {
  if (state.modelBytesPromise) {
    return state.modelBytesPromise;
  }

  state.modelPhase = "downloading";
  state.modelError = "";
  setAction("Downloading model...");
  setSynthesizeEnabled();

  showProgress("Downloading model...", 0, false);
  state.modelBytesPromise = fetchBytesWithProgress(modelAssetPath, (received, total) => {
    const pct = total > 0 ? Math.min(100, Math.round((received / total) * 100)) : 0;
    showProgress(`Downloading model... ${pct}%`, pct, false);
    setAction(`Downloading model (${pct}%)...`);
  }).catch((err) => {
    state.modelPhase = "error";
    state.modelError = formatError(err);
    state.modelBytesPromise = null;
    hideProgress();
    setAction(`Model download failed: ${state.modelError}`);
    renderInfo();
    setSynthesizeEnabled();
    throw err;
  });

  return state.modelBytesPromise;
}

async function autoLoadModel() {
  if (!state.kernelReady || state.modelLoaded || state.modelPhase === "initializing") {
    return;
  }

  try {
    const [modelBytes, tokenizerBytes] = await Promise.all([
      ensureModelBytesFetched(),
      ensureTokenizerBytesFetched(),
    ]);

    state.modelPhase = "initializing";
    showProgress("Initializing model...", 0, false);
    setAction("Initializing model...");
    renderInfo();

    // Go API: loadModel(modelBytes, tokenizerBytes, progressCallback)
    const result = await globalThis.PocketTTSKernel.loadModel(modelBytes, tokenizerBytes, (evt) => {
      const stage = evt?.stage || "load";
      const pct = Math.round(Math.max(0, Math.min(100, Number(evt?.percent || 0))));
      showProgress(`Initializing model (${stage})... ${pct}%`, pct, false);
      setAction(`Initializing model (${stage} ${pct}%)...`);
    });

    if (!result?.ok) {
      throw new Error(result?.error || "model load failed");
    }

    state.modelLoaded = true;
    state.modelPhase = "ready";
    showProgress("Model ready", 100, true);
    setAction("Model ready.");
    setTimeout(hideProgress, 2000);

    await downloadSelectedVoiceIfNeeded();
  } catch (err) {
    state.modelLoaded = false;
    state.modelPhase = "error";
    state.modelError = formatError(err);
    hideProgress();
    setAction(`Model load failed: ${state.modelError}`);
  } finally {
    renderInfo();
    setSynthesizeEnabled();
  }
}

function rAF() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function runSynthesis(text) {
  try {
    const options = { temperature: modelConfig.temperature };
    const voiceSafetensors = await downloadSelectedVoiceIfNeeded();
    if (voiceSafetensors) {
      options.voiceSafetensors = voiceSafetensors;
    }

    const t0 = performance.now();
    const result = await globalThis.PocketTTSKernel.synthesize(
      text,
      (evt) => {
        const pct = Math.round(Math.max(0, Math.min(100, Number(evt?.percent || 0))));
        const stage = evt?.stage || "working";
        const detail = evt?.detail || stage;
        showProgress(`Synthesizing... ${pct}%`, pct, false);
        setAction(`Synthesizing — ${detail} (${pct}%)`);
      },
      options,
    );
    const elapsedMs = (performance.now() - t0).toFixed(1);

    if (!result?.ok) {
      throw new Error(result?.error || "synthesis failed");
    }

    const wavBytes = decodeBase64ToBytes(result.wav_base64 || "");
    setAudioBlob(wavBytes);
    showProgress(`Done in ${elapsedMs} ms`, 100, true);
    setAction(`Synthesis complete in ${elapsedMs} ms.`);
    setTimeout(hideProgress, 3000);
  } catch (err) {
    hideProgress();
    setAction(`Synthesis failed: ${formatError(err)}`);
  } finally {
    state.isSynthesizing = false;
    renderInfo();
    setSynthesizeEnabled();
  }
}

function handleSynthesize() {
  if (synthBtn.disabled || state.isSynthesizing) {
    return;
  }

  state.isSynthesizing = true;
  state.textDirty = false;
  setSynthesizeEnabled();
  showProgress("Synthesizing...", 0, false);
  setAction("Synthesizing...");

  // Capture text now; user may freely edit while synthesis runs.
  const text = textArea.value;

  // Yield to the browser so the UI repaints before the heavy WASM call.
  rAF().then(() => runSynthesis(text));
}

async function loadVoiceManifestAndPrefetch() {
  try {
    const res = await fetch("./voices/manifest.json");
    if (!res.ok) {
      throw new Error(`fetch voices manifest failed (${res.status})`);
    }

    state.voiceManifest = await res.json();
    populateVoiceDropdown();
    renderInfo();
    setSynthesizeEnabled();

    await downloadSelectedVoiceIfNeeded();
  } catch (err) {
    setAction(`Voice manifest unavailable: ${formatError(err)}`);
  }
}

async function initApp() {
  renderInfo();
  setSynthesizeEnabled();

  // Start model + tokenizer downloads immediately so transfers overlap wasm startup.
  void ensureModelBytesFetched();
  void ensureTokenizerBytesFetched();

  // Voice manifest/voice download can also run in parallel with kernel boot.
  const voicesPromise = loadVoiceManifestAndPrefetch();

  try {
    await bootKernel();
  } catch (err) {
    setAction(`Kernel startup failed: ${formatError(err)}`);
    setSynthesizeEnabled();
    return;
  }

  await Promise.allSettled([autoLoadModel(), voicesPromise]);
  renderInfo();
  setSynthesizeEnabled();
}

synthBtn.addEventListener("click", () => handleSynthesize());
textArea.addEventListener("input", () => {
  if (state.isSynthesizing && !state.textDirty) {
    state.textDirty = true;
    setSynthesizeEnabled();
  }
});
voiceSelect.addEventListener("change", () => {
  renderInfo();
  setSynthesizeEnabled();
  void downloadSelectedVoiceIfNeeded();
});
temperatureSlider.addEventListener("input", () => {
  const val = parseFloat(temperatureSlider.value);
  modelConfig.temperature = val;
  temperatureValue.textContent = val.toFixed(2);
});

initApp().catch((err) => {
  setAction(`Startup failed: ${formatError(err)}`);
});
