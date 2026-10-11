// The sampling figure applies temperature, top-k and top-p as
// minllm/inference.py does.

import assert from "node:assert/strict";
import test from "node:test";

import { distribution } from "../src/labs/sampling.js";

const logits = [2, 1, 0, -1, -3];
const sum = (xs) => xs.reduce((s, v) => s + v, 0);

test("temperature, top-k and top-p", () => {
  const plain = distribution(logits, { temperature: 1, topK: 0, topP: 1 });
  assert.ok(Math.abs(sum(plain) - 1) < 1e-12);
  assert.ok(Math.abs(plain[0] / plain[1] - Math.E) < 1e-9); // a logit apart is a factor of e
  assert.deepEqual([...distribution(logits, { temperature: 0, topK: 0, topP: 1 })], [1, 0, 0, 0, 0]);
  const flat = distribution(logits, { temperature: 2, topK: 0, topP: 1 });
  assert.ok(flat[0] < plain[0] && flat[4] > plain[4]);
  const two = distribution(logits, { temperature: 1, topK: 2, topP: 1 });
  assert.deepEqual([...two].map((p) => p > 0), [true, true, false, false, false]);
  assert.ok(Math.abs(sum(two) - 1) < 1e-12);
  // top-p keeps the likeliest tokens until those before a token already sum to p
  const nucleus = distribution(logits, { temperature: 1, topK: 0, topP: 0.7 });
  assert.deepEqual([...nucleus].map((p) => p > 0), [true, true, false, false, false]);
});
