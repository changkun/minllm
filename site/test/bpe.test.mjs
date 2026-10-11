// The browser's tokenizer must give the token ids of minllm/tokenizer.py.
// test/bpe.json holds texts with the ids that Python gives them.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { Tokenizer } from "../src/labs/bpe.js";

const tok = new Tokenizer(JSON.parse(readFileSync(new URL("../public/tokenizer.json", import.meta.url))));
const cases = JSON.parse(readFileSync(new URL("./bpe.json", import.meta.url)));

test("the same token ids as the Python tokenizer", () => {
  assert.equal(tok.size, 4096);
  for (const { text, ids } of cases) {
    assert.deepEqual(tok.encode(text), ids, text);
    assert.equal(tok.decode(ids), text);
  }
});

test("a chunk is merged in the order the merges were learned", () => {
  const states = tok.steps(" kitten");
  assert.deepEqual(states[0], [...new TextEncoder().encode(" kitten")]);
  assert.ok(states.every((s, i) => i === 0 || s.length < states[i - 1].length));
  assert.equal(tok.decode(states.at(-1)), " kitten");
});
