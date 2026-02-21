import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.mjs";

const log = document.getElementById("log");
const textArea = document.getElementById("text");
const runBtn = document.getElementById("run");
const verifyBtn = document.getElementById("verify-models");
const modelSynthBtn = document.getElementById("synthesize-model");
const player = document.getElementById("player");
const download = document.getElementById("download");

const WEB_SAMPLE_RATE = 24000;
const FLOW_TEMPERATURE = 0.7;
const FLOW_LSD_DECODE_STEPS = 1;
const FLOW_EOS_THRESHOLD = -4.0;
const FLOW_FRAMES_AFTER_EOS = 2;

function decodeBase64ToBytes(base64) {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

function clampSample(v) {
  if (v > 1) {
    return 1;
  }
  if (v < -1) {
    return -1;
  }
  return v;
}

function encodeWavPCM16(samples, sampleRate = WEB_SAMPLE_RATE) {
  const channels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = channels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const riffSize = 36 + dataSize;

  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  function writeASCII(offset, text) {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  }

  writeASCII(0, "RIFF");
  view.setUint32(4, riffSize, true);
  writeASCII(8, "WAVE");
  writeASCII(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeASCII(36, "data");
  view.setUint32(40, dataSize, true);

  let p = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const s = clampSample(samples[i]);
    const q = Math.round(s * 32767);
    view.setInt16(p, q, true);
    p += 2;
  }

  return new Uint8Array(buf);
}

function shapeWithFallback(shape) {
  return shape.map((dim) => {
    if (typeof dim === "number" && Number.isFinite(dim) && dim > 0) {
      return dim;
    }
    return 1;
  });
}

function tensorSize(shape) {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

function ortType(dtype) {
  switch ((dtype || "").toLowerCase()) {
    case "float":
    case "float32":
      return "float32";
    case "int64":
      return "int64";
    case "int32":
      return "int32";
    default:
      throw new Error(`unsupported ONNX dtype in manifest: ${dtype}`);
  }
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
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalizePromptForModel(text) {
  let t = (text || "").trim();
  if (t.length === 0) {
    return t;
  }

  t = t.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  t = t.replace(/\n+/g, " ").replace(/\s+/g, " ");
  if (!/[.!?]$/.test(t)) {
    t += ".";
  }
  return t;
}

function zeroTensorFor(input) {
  const shape = shapeWithFallback(input.shape || []);
  const size = tensorSize(shape);
  const type = ortType(input.dtype);

  if (type === "float32") {
    return new ort.Tensor(type, new Float32Array(size), shape);
  }
  if (type === "int64") {
    return new ort.Tensor(type, new BigInt64Array(size), shape);
  }
  if (type === "int32") {
    return new ort.Tensor(type, new Int32Array(size), shape);
  }

  throw new Error(`cannot build feed tensor for type: ${type}`);
}

function toFloat32Array(ortTensor) {
  if (!ortTensor || !ortTensor.data) {
    throw new Error("invalid ORT tensor output");
  }
  if (ortTensor.data instanceof Float32Array) {
    return ortTensor.data;
  }
  if (ortTensor.data instanceof Float64Array) {
    return Float32Array.from(ortTensor.data);
  }
  if (ArrayBuffer.isView(ortTensor.data)) {
    return Float32Array.from(ortTensor.data);
  }
  throw new Error(`unsupported tensor data type: ${typeof ortTensor.data}`);
}

function shapeDimFromInput(input, idx, fallback) {
  if (!input || !Array.isArray(input.shape)) {
    return fallback;
  }
  const d = input.shape[idx];
  if (typeof d === "number" && Number.isFinite(d) && d > 0) {
    return d;
  }
  return fallback;
}

async function bootKernel() {
  const go = new Go();
  const wasm = await WebAssembly.instantiateStreaming(fetch("./pockettts-kernel.wasm"), go.importObject);
  go.run(wasm.instance);

  if (!globalThis.PocketTTSKernel) {
    throw new Error("PocketTTSKernel not found after wasm startup");
  }

  log.textContent = `Kernel ready: ${globalThis.PocketTTSKernel.version}`;
}

async function loadManifest(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`manifest fetch failed (${res.status}) at ${path}`);
  }
  return res.json();
}

function graphByName(manifest, name) {
  const g = (manifest.graphs || []).find((x) => x.name === name);
  if (!g) {
    throw new Error(`required graph not found in manifest: ${name}`);
  }
  return g;
}

async function createSessions(manifest, graphNames) {
  const sessions = {};
  for (const name of graphNames) {
    const graph = graphByName(manifest, name);
    sessions[name] = {
      graph,
      session: await ort.InferenceSession.create(`./models/${graph.filename}`, {
        executionProviders: ["wasm"],
      }),
    };
  }
  return sessions;
}

async function verifyGraph(graph) {
  const graphPath = `./models/${graph.filename}`;
  const session = await ort.InferenceSession.create(graphPath, {
    executionProviders: ["wasm"],
  });

  const feeds = {};
  for (const input of graph.inputs || []) {
    feeds[input.name] = zeroTensorFor(input);
  }

  await session.run(feeds);
}

async function synthesizeWithModels(inputText) {
  const prompt = normalizePromptForModel(inputText);
  const tokenized = globalThis.PocketTTSKernel.tokenize(prompt);
  if (!tokenized.ok) {
    throw new Error(tokenized.error || "tokenization failed");
  }

  const tokens = tokenized.tokens || [];
  if (tokens.length === 0) {
    throw new Error("no tokens generated after normalization");
  }

  const manifest = await loadManifest("./models/manifest.json");
  const needed = ["text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder"];
  const hasLatentToMimi = (manifest.graphs || []).some((g) => g.name === "latent_to_mimi");
  if (hasLatentToMimi) {
    needed.push("latent_to_mimi");
  }
  const sessions = await createSessions(manifest, needed);

  const textCond = sessions.text_conditioner;
  const flowMain = sessions.flow_lm_main;
  const flowFlow = sessions.flow_lm_flow;
  const latentToMimi = sessions.latent_to_mimi;
  const mimiDecoder = sessions.mimi_decoder;

  const tokenTensor = new ort.Tensor("int64", BigInt64Array.from(tokens.map((v) => BigInt(v))), [1, tokens.length]);
  const tcInputName = textCond.graph.inputs[0]?.name || "tokens";
  const tcOutputName = textCond.graph.outputs[0]?.name || "text_embeddings";
  const tcOut = await textCond.session.run({ [tcInputName]: tokenTensor });
  const textEmbTensor = tcOut[tcOutputName];
  if (!textEmbTensor) {
    throw new Error(`text_conditioner output missing: ${tcOutputName}`);
  }

  const seqDim = shapeDimFromInput(flowMain.graph.inputs[0], 2, 32);
  const condDim = shapeDimFromInput(flowFlow.graph.inputs[0], 1, 1024);
  const xDim = shapeDimFromInput(flowFlow.graph.inputs[3], 1, seqDim);
  const mimiLatDim = shapeDimFromInput(mimiDecoder.graph.inputs[0], 1, 512);

  const sequenceFrames = [];
  const generatedLatents = [];
  const bosFrame = new Float32Array(seqDim);
  bosFrame.fill(Number.NaN);
  sequenceFrames.push(bosFrame);
  const maxSteps = Math.min(256, Math.max(24, tokens.length * 5));
  const minSteps = 16;
  let eosStep = null;

  for (let step = 0; step < maxSteps; step += 1) {
    const seqLen = Math.max(1, sequenceFrames.length);
    const seqData = new Float32Array(seqLen * seqDim);
    for (let i = 0; i < sequenceFrames.length; i += 1) {
      seqData.set(sequenceFrames[i], i * seqDim);
    }

    const mainInputSequenceName = flowMain.graph.inputs[0]?.name || "sequence";
    const mainInputTextName = flowMain.graph.inputs[1]?.name || "text_embeddings";
    const mainOutputHiddenName = flowMain.graph.outputs[0]?.name || "last_hidden";
    const mainOutputEOSName = flowMain.graph.outputs[1]?.name || "eos_logits";

    const mainOut = await flowMain.session.run({
      [mainInputSequenceName]: new ort.Tensor("float32", seqData, [1, seqLen, seqDim]),
      [mainInputTextName]: textEmbTensor,
    });

    const hidden = mainOut[mainOutputHiddenName];
    const eos = mainOut[mainOutputEOSName];
    if (!hidden || !eos) {
      throw new Error("flow_lm_main outputs missing");
    }

    const hiddenData = toFloat32Array(hidden);
    const condition = new Float32Array(condDim);
    for (let i = 0; i < condDim; i += 1) {
      condition[i] = hiddenData[i % hiddenData.length];
    }

    let x = new Float32Array(xDim);
    const flowNoiseStd = Math.sqrt(FLOW_TEMPERATURE);
    for (let i = 0; i < x.length; i += 1) {
      x[i] = randomNormal() * flowNoiseStd;
    }

    const flowInputNames = (flowFlow.graph.inputs || []).map((x) => x.name);
    const flowOutName = flowFlow.graph.outputs[0]?.name || "flow_direction";

    const eulerSteps = FLOW_LSD_DECODE_STEPS;
    for (let k = 0; k < eulerSteps; k += 1) {
      const s = k / eulerSteps;
      const t = (k + 1) / eulerSteps;

      const feeds = {
        [flowInputNames[0] || "condition"]: new ort.Tensor("float32", condition, [1, condDim]),
        [flowInputNames[1] || "s"]: new ort.Tensor("float32", new Float32Array([s]), [1, 1]),
        [flowInputNames[2] || "t"]: new ort.Tensor("float32", new Float32Array([t]), [1, 1]),
        [flowInputNames[3] || "x"]: new ort.Tensor("float32", x, [1, xDim]),
      };

      const flowOut = await flowFlow.session.run(feeds);
      const dir = flowOut[flowOutName];
      if (!dir) {
        throw new Error(`flow_lm_flow output missing: ${flowOutName}`);
      }
      const dirData = toFloat32Array(dir);
      const dt = 1 / eulerSteps;
      for (let i = 0; i < x.length; i += 1) {
        x[i] += dirData[i % dirData.length] * dt;
      }
    }

    const frame = new Float32Array(seqDim);
    for (let i = 0; i < seqDim; i += 1) {
      frame[i] = x[i % x.length];
    }
    sequenceFrames.push(frame);
    generatedLatents.push(Float32Array.from(x));

    const eosData = toFloat32Array(eos);
    const eosLogit = eosData[0] || -10;
    if (eosLogit > FLOW_EOS_THRESHOLD && eosStep === null) {
      eosStep = step;
    }
    if (eosStep !== null && step >= eosStep + FLOW_FRAMES_AFTER_EOS && step + 1 >= minSteps) {
      break;
    }
  }

  if (generatedLatents.length === 0) {
    throw new Error("model loop produced no latent frames");
  }

  const steps = generatedLatents.length;
  const latentDim = shapeDimFromInput(flowFlow.graph.inputs[3], 1, 32);
  const latentSeq = new Float32Array(steps * latentDim);
  for (let t = 0; t < steps; t += 1) {
    const src = generatedLatents[t];
    for (let c = 0; c < latentDim; c += 1) {
      latentSeq[t * latentDim + c] = src[c % src.length];
    }
  }

  let mimiInputTensor = null;
  if (latentToMimi) {
    const latentToMimiInputName = latentToMimi.graph.inputs[0]?.name || "latent";
    const latentToMimiOutputName = latentToMimi.graph.outputs[0]?.name || "mimi_latent";
    const latentToMimiOut = await latentToMimi.session.run({
      [latentToMimiInputName]: new ort.Tensor("float32", latentSeq, [1, steps, latentDim]),
    });
    mimiInputTensor = latentToMimiOut[latentToMimiOutputName];
    if (!mimiInputTensor) {
      throw new Error(`latent_to_mimi output missing: ${latentToMimiOutputName}`);
    }
  } else {
    const latent = new Float32Array(mimiLatDim * steps);
    for (let t = 0; t < steps; t += 1) {
      const src = generatedLatents[t];
      for (let c = 0; c < mimiLatDim; c += 1) {
        latent[c * steps + t] = src[c % src.length];
      }
    }
    mimiInputTensor = new ort.Tensor("float32", latent, [1, mimiLatDim, steps]);
  }

  const mimiInputName = mimiDecoder.graph.inputs[0]?.name || "latent";
  const mimiOutputName = mimiDecoder.graph.outputs[0]?.name || "audio";
  const mimiOut = await mimiDecoder.session.run({
    [mimiInputName]: mimiInputTensor,
  });

  const audioTensor = mimiOut[mimiOutputName];
  if (!audioTensor) {
    throw new Error(`mimi_decoder output missing: ${mimiOutputName}`);
  }

  const audioData = toFloat32Array(audioTensor);
  if (audioData.length === 0) {
    throw new Error("decoder returned empty audio");
  }

  const wavBytes = encodeWavPCM16(audioData, WEB_SAMPLE_RATE);
  return {
    normalizedText: tokenized.text,
    tokenCount: tokens.length,
    frames: steps,
    wavBytes,
  };
}

function setAudioBlob(wavBytes) {
  const blob = new Blob([wavBytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  player.src = url;
  download.href = url;
  download.textContent = "Download hello.wav";
}

runBtn.addEventListener("click", async () => {
  const input = textArea.value;
  const out = globalThis.PocketTTSKernel.synthesizeWav(input);
  if (!out.ok) {
    log.textContent = `Error: ${out.error}`;
    return;
  }

  const wavBytes = decodeBase64ToBytes(out.wav_base64);
  setAudioBlob(wavBytes);

  log.textContent = `Normalized: ${out.text}\nSampleRate: ${out.sample_rate}\nWAV bytes: ${wavBytes.length}`;
});

verifyBtn.addEventListener("click", async () => {
  try {
    log.textContent = "Loading ONNX manifest...";
    const manifest = await loadManifest("./models/manifest.json");
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
    log.textContent = `ONNX smoke verify failed: ${err.message}`;
  }
});

modelSynthBtn.addEventListener("click", async () => {
  try {
    modelSynthBtn.disabled = true;
    log.textContent = "Running model inference loop (experimental)...";

    const t0 = performance.now();
    const out = await synthesizeWithModels(textArea.value);
    const dtMs = (performance.now() - t0).toFixed(1);

    setAudioBlob(out.wavBytes);
    log.textContent = `Model synth (experimental) complete\nNormalized: ${out.normalizedText}\nTokens: ${out.tokenCount}\nFrames: ${out.frames}\nWAV bytes: ${out.wavBytes.length}\nElapsed: ${dtMs} ms`;
  } catch (err) {
    log.textContent = `Model synth failed: ${err.message}`;
  } finally {
    modelSynthBtn.disabled = false;
  }
});

bootKernel().catch((err) => {
  log.textContent = `Kernel boot failed: ${err.message}`;
});
