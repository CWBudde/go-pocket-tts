const log = document.getElementById("log");
const textArea = document.getElementById("text");
const runBtn = document.getElementById("run");
const player = document.getElementById("player");
const download = document.getElementById("download");

function decodeBase64ToBytes(base64) {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
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

runBtn.addEventListener("click", async () => {
  const input = textArea.value;
  const out = globalThis.PocketTTSKernel.synthesizeWav(input);
  if (!out.ok) {
    log.textContent = `Error: ${out.error}`;
    return;
  }

  const wavBytes = decodeBase64ToBytes(out.wav_base64);
  const blob = new Blob([wavBytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);

  player.src = url;
  download.href = url;
  download.textContent = "Download hello.wav";

  log.textContent = `Normalized: ${out.text}\nSampleRate: ${out.sample_rate}\nWAV bytes: ${wavBytes.length}`;
});

bootKernel().catch((err) => {
  log.textContent = `Kernel boot failed: ${err.message}`;
});
