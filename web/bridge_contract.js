export const SUPPORTED_DTYPES = new Set(["float", "float32", "int32", "int64"]);

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function isSafeInt(v) {
  return Number.isSafeInteger(v);
}

function isShapeArray(shape) {
  return Array.isArray(shape) && shape.every((d) => isSafeInt(d) && d >= 0);
}

export function validateTensorPayload(payload, name = "tensor") {
  if (typeof payload !== "object" || payload === null) {
    throw new Error(`${name}: payload must be an object`);
  }

  const dtype = String(payload.dtype || "").toLowerCase();
  if (!SUPPORTED_DTYPES.has(dtype)) {
    throw new Error(`${name}: unsupported dtype ${payload.dtype}`);
  }

  if (!isShapeArray(payload.shape)) {
    throw new Error(`${name}: shape must be an array of non-negative safe integers`);
  }

  if (dtype === "float" || dtype === "float32") {
    if (!Array.isArray(payload.f32) || !payload.f32.every((v) => isFiniteNumber(v))) {
      throw new Error(`${name}: float tensor requires finite numeric f32 array`);
    }
  }

  if (dtype === "int64") {
    if (!Array.isArray(payload.i64) || !payload.i64.every((v) => isSafeInt(v))) {
      throw new Error(`${name}: int64 tensor requires safe-integer i64 array`);
    }
  }

  if (dtype === "int32") {
    if (!Array.isArray(payload.i32) || !payload.i32.every((v) => isSafeInt(v))) {
      throw new Error(`${name}: int32 tensor requires safe-integer i32 array`);
    }
  }

  return {
    dtype,
    shape: payload.shape,
    f32: payload.f32,
    i64: payload.i64,
    i32: payload.i32,
  };
}

export function parseAndValidateFeedsJSON(feedsJSON) {
  let parsed;
  try {
    parsed = JSON.parse(feedsJSON);
  } catch (err) {
    throw new Error(`invalid feeds JSON: ${err.message}`);
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("feeds JSON must decode to an object");
  }

  const validated = {};
  for (const [name, payload] of Object.entries(parsed)) {
    validated[name] = validateTensorPayload(payload, `feed ${name}`);
  }
  return validated;
}
