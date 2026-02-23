import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.mjs";
import { parseAndValidateFeedsJSON, validateTensorPayload } from "./bridge_contract.js";

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed (${res.status})`);

  const contentLength = parseInt(res.headers.get("content-length") || "0", 10);
  if (!res.body || !contentLength) {
    const buf = await res.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength);
    return buf;
  }

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, contentLength);
  }

  const buf = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return buf.buffer;
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

export class ONNXBridge {
  constructor(options = {}) {
    this.manifestPath = options.manifestPath || "./models/manifest.json";
    this.manifestCache = null;
    this.sessionCache = new Map();
  }

  async loadManifest() {
    if (this.manifestCache) {
      return this.manifestCache;
    }
    const res = await fetch(this.manifestPath);
    if (!res.ok) {
      throw new Error(`manifest fetch failed (${res.status}) at ${this.manifestPath}`);
    }
    this.manifestCache = await res.json();
    return this.manifestCache;
  }

  async getGraph(graphName) {
    const manifest = await this.loadManifest();
    const graph = (manifest.graphs || []).find((g) => g.name === graphName);
    if (!graph) {
      throw new Error(`graph not found in manifest: ${graphName}`);
    }
    return graph;
  }

  async getSession(graphName, onProgress) {
    if (this.sessionCache.has(graphName)) {
      return this.sessionCache.get(graphName);
    }

    const graph = await this.getGraph(graphName);
    const url = `./models/${graph.filename}`;

    let session;
    if (onProgress) {
      const buf = await fetchWithProgress(url, onProgress);
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
      });
    } else {
      session = await ort.InferenceSession.create(url, {
        executionProviders: ["wasm"],
      });
    }

    const entry = { graph, session };
    this.sessionCache.set(graphName, entry);
    return entry;
  }

  async runGraph(graphName, feedsJSON) {
    const feeds = parseAndValidateFeedsJSON(feedsJSON);
    const outputs = await this.runGraphPayload(graphName, feeds);
    return JSON.stringify(outputs);
  }

  async runGraphPayload(graphName, feeds) {
    const { session } = await this.getSession(graphName);

    const ortFeeds = {};
    for (const [name, payload] of Object.entries(feeds)) {
      validateTensorPayload(payload, `feed ${name}`);
      ortFeeds[name] = ortTensorFromPayload(payload);
    }

    const outputs = await session.run(ortFeeds);
    const serialized = {};
    for (const [name, tensor] of Object.entries(outputs)) {
      serialized[name] = payloadFromTensor(tensor);
      validateTensorPayload(serialized[name], `output ${name}`);
    }

    return serialized;
  }

  async preloadGraphs(graphNames) {
    for (const name of graphNames) {
      await this.getSession(name);
    }
  }

  async verifyGraph(graph) {
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
}
