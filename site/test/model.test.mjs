// The model that runs in the browser computes what the Python model
// computes: scripts/dump_forward.py saved the numbers that these tests
// compare with, from the same file.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { Model, expand } from "../src/labs/model.js";
import { half, read, toHalf } from "../src/labs/safetensors.js";

const here = new URL(".", import.meta.url);
const expected = JSON.parse(readFileSync(new URL("forward.json", here)));
const file = readFileSync(new URL("../public/models/base-q4/model.safetensors", here));
const tensors = read(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const meta = JSON.parse(readFileSync(new URL("../public/models/base-q4/meta.json", here)));

test("half-precision floats", () => {
  assert.equal(half(0x3c00), 1);
  assert.equal(half(0xc000), -2);
  assert.equal(half(0x0001), 2 ** -24);
  assert.equal(half(0x7bff), 65504);
  // every half is its own nearest half, and a number between two goes to the nearer one
  for (const bits of [0x0001, 0x03ff, 0x0400, 0x3555, 0x3c01, 0x7bff, 0x8001, 0xbc00]) assert.equal(toHalf(half(bits)), half(bits));
  assert.equal(toHalf(1 + 2 ** -11), 1); // a tie goes to the even one
  assert.equal(toHalf(1 + 3 * 2 ** -11), 1 + 2 ** -9);
  assert.equal(toHalf(1 + 2 ** -11 + 2 ** -20), 1 + 2 ** -10);
  assert.equal(toHalf(0.1), half(0x2e66));
});

test("the weights are those of quantize.dequantize", () => {
  const rows = { embed: 1000, "layers.0.attn.q_proj": 0, "layers.7.mlp.down_proj": 383 };
  for (const [name, values] of Object.entries(expected.weights)) {
    const { cols, weights } = expand(tensors, name, meta.quant.bits);
    assert.deepEqual([...weights.subarray(rows[name] * cols, rows[name] * cols + values.length)], values, name);
  }
});

test("the logits and the likeliest continuation are those of the Python model", () => {
  const model = new Model(tensors, meta);
  for (const { prompt, ids, logits, greedy } of expected.prompts) {
    model.reset();
    let out;
    for (const [i, id] of ids.entries()) out = model.step(id, i === ids.length - 1);
    const worst = Math.max(...logits.map((v, i) => Math.abs(v - out[i])));
    assert.ok(worst < 5e-4, `${prompt}: a logit is off by ${worst}`);
    const written = [];
    for (let n = 0; n < greedy.length; n++) {
      written.push(out.indexOf(Math.max(...out)));
      out = model.step(written.at(-1));
    }
    assert.deepEqual(written, greedy, prompt);
  }
});
