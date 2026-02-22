/**
 * Minimal safetensors parser for loading voice embeddings in the browser.
 *
 * Format: [8-byte LE header length] [JSON header] [raw tensor data]
 * Only F32 tensors are supported (matching the Go reader).
 */

/**
 * Load the first F32 tensor from a safetensors file fetched from the given URL.
 * Returns { data: Float32Array, shape: number[] }.
 */
export async function loadVoiceEmbedding(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`fetch ${url} failed (${res.status})`);
  }

  const buf = await res.arrayBuffer();
  return parseSafetensors(buf);
}

/**
 * Parse a safetensors ArrayBuffer and return the first F32 tensor.
 * Normalizes 2D [T, D] to 3D [1, T, D].
 */
export function parseSafetensors(buf) {
  if (buf.byteLength < 8) {
    throw new Error("safetensors: file too short");
  }

  const view = new DataView(buf);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerEnd = 8 + headerLen;

  if (headerEnd > buf.byteLength) {
    throw new Error("safetensors: header exceeds file size");
  }

  const headerJSON = new TextDecoder().decode(new Uint8Array(buf, 8, headerLen));
  const header = JSON.parse(headerJSON);

  let entry = null;
  for (const [key, val] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    entry = val;
    break;
  }

  if (!entry) {
    throw new Error("safetensors: no tensors found");
  }
  if (entry.dtype !== "F32") {
    throw new Error(`safetensors: unsupported dtype ${entry.dtype}`);
  }

  const [start, end] = entry.data_offsets;
  const data = new Float32Array(buf, headerEnd + start, (end - start) / 4);

  let shape = entry.shape;
  if (shape.length === 2) {
    shape = [1, shape[0], shape[1]];
  }

  return { data, shape };
}
