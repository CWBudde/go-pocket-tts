// Microphone capture for voice prompts: raw PCM from an AudioWorklet, wrapped
// in a 16-bit mono WAV that PocketTTSKernel.cloneVoice accepts at any rate.

// The worklet forwards channel 0 in blocks of chunkSize samples; posting every
// 128-sample render quantum would flood the main thread with messages.
const workletSource = `
class PocketTTSCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Float32Array(2048);
    this.fill = 0;
    this.port.onmessage = () => {
      this.port.postMessage(this.chunk.slice(0, this.fill));
      this.fill = 0;
    };
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    let offset = 0;
    while (offset < input.length) {
      const n = Math.min(input.length - offset, this.chunk.length - this.fill);
      this.chunk.set(input.subarray(offset, offset + n), this.fill);
      this.fill += n;
      offset += n;
      if (this.fill === this.chunk.length) {
        this.port.postMessage(this.chunk.slice());
        this.fill = 0;
      }
    }
    return true;
  }
}
registerProcessor("pockettts-capture", PocketTTSCapture);
`;

// microphoneError turns getUserMedia failures into messages a user can act on.
function microphoneError(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return new Error("microphone access was denied");
    case "NotFoundError":
    case "OverconstrainedError":
      return new Error("no microphone found");
    case "NotReadableError":
      return new Error("the microphone is in use by another application");
    default:
      return err instanceof Error ? err : new Error(String(err));
  }
}

// encodeWAV16 encodes mono float samples in [-1, 1] as a 16-bit PCM WAV.
export function encodeWAV16(samples, sampleRate) {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const writeTag = (offset, tag) => {
    for (let i = 0; i < 4; i += 1) view.setUint8(offset + i, tag.charCodeAt(i));
  };

  writeTag(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeTag(8, "WAVE");
  writeTag(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeTag(36, "data");
  view.setUint32(40, dataBytes, true);

  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

// startRecording asks for the microphone and records until stop() or
// maxSeconds. Browser voice processing is off: echo cancellation, noise
// suppression and gain control change the timbre the clone would copy.
// onProgress({seconds, level}) reports the elapsed time and the RMS level of
// the latest block (0..1); onLimit fires once maxSeconds are recorded, after
// which further audio is dropped. stop() resolves to {wav, seconds, sampleRate};
// cancel() discards the recording.
export async function startRecording({ maxSeconds = 30, onProgress, onLimit } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("microphone access needs a secure context (https or localhost)");
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    throw microphoneError(err);
  }

  const ctx = new AudioContext();
  const release = () => {
    for (const track of stream.getTracks()) track.stop();
    void ctx.close().catch(() => {});
  };

  let node;
  let mute;
  let source;
  try {
    const moduleURL = URL.createObjectURL(new Blob([workletSource], { type: "text/javascript" }));
    try {
      await ctx.audioWorklet.addModule(moduleURL);
    } finally {
      URL.revokeObjectURL(moduleURL);
    }
    source = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "pockettts-capture", { numberOfInputs: 1, numberOfOutputs: 1 });
    // A node off the destination graph may never be processed; route it there
    // through a muted gain so nothing is played back.
    mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node).connect(mute).connect(ctx.destination);
    if (ctx.state === "suspended") await ctx.resume();
  } catch (err) {
    release();
    throw err;
  }

  const sampleRate = ctx.sampleRate;
  const limit = Math.round(maxSeconds * sampleRate);
  const chunks = [];
  let total = 0;
  let limitReached = false;
  let flushed = null;

  node.port.onmessage = (evt) => {
    const block = evt.data;
    if (flushed && block.length < 2048) {
      // The reply to the flush request in stop(): the partial last block.
      append(block);
      flushed();
      return;
    }
    append(block);
  };

  function append(block) {
    if (total >= limit) return;
    const keep = block.length > limit - total ? block.subarray(0, limit - total) : block;
    chunks.push(keep);
    total += keep.length;

    let sum = 0;
    for (let i = 0; i < keep.length; i += 1) sum += keep[i] * keep[i];
    const level = keep.length > 0 ? Math.sqrt(sum / keep.length) : 0;
    if (onProgress) onProgress({ seconds: total / sampleRate, level });

    if (total >= limit && !limitReached) {
      limitReached = true;
      if (onLimit) onLimit();
    }
  }

  let done = false;
  const disconnect = () => {
    done = true;
    source.disconnect();
    node.disconnect();
    mute.disconnect();
    release();
  };

  return {
    sampleRate,
    async stop() {
      if (done) throw new Error("recording already finished");
      // Ask the worklet for its partial block so the tail is not lost; give
      // up after a moment if the audio thread is gone.
      await new Promise((resolve) => {
        flushed = resolve;
        node.port.postMessage("flush");
        setTimeout(resolve, 250);
      });
      disconnect();

      const samples = new Float32Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }
      return { wav: encodeWAV16(samples, sampleRate), seconds: total / sampleRate, sampleRate };
    },
    cancel() {
      if (!done) disconnect();
    },
  };
}
