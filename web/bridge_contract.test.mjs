import test from "node:test";
import assert from "node:assert/strict";

import { parseAndValidateFeedsJSON, validateTensorPayload } from "./bridge_contract.js";

test("validateTensorPayload accepts finite float payload", () => {
  const out = validateTensorPayload({
    dtype: "float32",
    shape: [1, 2],
    f32: [0.1, -0.2],
  });
  assert.equal(out.dtype, "float32");
});

test("validateTensorPayload rejects NaN in float payload", () => {
  assert.throws(() => {
    validateTensorPayload({
      dtype: "float32",
      shape: [1],
      f32: [Number.NaN],
    });
  }, /finite numeric f32 array/);
});

test("validateTensorPayload rejects unsupported dtype", () => {
  assert.throws(() => {
    validateTensorPayload({ dtype: "bool", shape: [1], f32: [1] });
  }, /unsupported dtype/);
});

test("parseAndValidateFeedsJSON validates all feeds", () => {
  const feeds = parseAndValidateFeedsJSON(
    JSON.stringify({
      x: { dtype: "float32", shape: [1, 2], f32: [1, 2] },
      t: { dtype: "int64", shape: [1, 1], i64: [1] },
    })
  );
  assert.equal(Object.keys(feeds).length, 2);
});

test("parseAndValidateFeedsJSON rejects non-object JSON", () => {
  assert.throws(() => {
    parseAndValidateFeedsJSON("[]");
  }, /must decode to an object/);
});
