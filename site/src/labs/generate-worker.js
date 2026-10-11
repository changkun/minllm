// The model at work, off the page's own thread so that the page stays
// responsive: it loads the 4-bit file of the story model, takes a prompt,
// and sends back one token at a time.

import { Tokenizer } from "./bpe.js";
import { Model } from "./model.js";
import { read } from "./safetensors.js";
import { distribution } from "./sampling.js";

const DIR = "/models/base-q4";
const SHOWN = 6; // the likeliest tokens sent along with every token that is drawn

let model, tok, eot;
let ids = []; // the tokens that were written after the prompt
let logits = null; // the prediction for the next token, or null when there is none to continue from
let latest = 0; // counts the requests: a loop that is no longer the latest stops

// Let the messages that have arrived be handled before going on.
const channel = new MessageChannel();
const pause = () => new Promise((resolve) => {
  channel.port1.onmessage = resolve;
  channel.port2.postMessage(0);
});

async function get(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response;
}

async function load() {
  const [meta, vocabulary, response] = await Promise.all([get(`${DIR}/meta.json`).then((r) => r.json()), get("/tokenizer.json").then((r) => r.json()), get(`${DIR}/model.safetensors`)]);
  const total = Math.round(meta.megabytes * 1e6);
  const bytes = new Uint8Array(total);
  let received = 0;
  for (const reader = response.body.getReader(); ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, received);
    received += value.length;
    postMessage({ type: "loading", received, total });
  }
  postMessage({ type: "loading", received: total, total, unpacking: true });
  await pause();
  tok = new Tokenizer(vocabulary);
  eot = tok.special.get("<|endoftext|>");
  model = new Model(read(bytes.buffer), meta);
}

// A story starts after <|endoftext|>, as in the training data, and the model ends it by writing <|endoftext|> again.
function start(prompt) {
  const room = model.cfg.max_seq_len - 1;
  const tokens = [eot, ...tok.encode(prompt)].slice(0, room);
  model.reset();
  ids = [];
  for (const [i, id] of tokens.entries()) logits = model.step(id, i === tokens.length - 1);
}

async function write({ prompt, tokens, temperature, topK, topP, more }) {
  const mine = ++latest;
  if (!model) await load();
  if (mine !== latest) return;
  if (!more || !logits) start(prompt);
  const began = performance.now();
  let reason = "length";
  for (let n = 0; n < tokens; n++) {
    if (!logits) {
      reason = "context";
      break;
    }
    const probs = distribution(Array.from(logits), { temperature, topK, topP });
    // Draw a token: walk along the probabilities until they add up to a random number.
    let r = Math.random();
    let id = 0;
    for (let i = 0; i < probs.length && r > 0; i++) {
      if (probs[i] > 0) r -= probs[(id = i)];
    }
    const order = Array.from(probs, (p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, SHOWN);
    if (!order.some(([, i]) => i === id)) order.push([probs[id], id]);
    const candidates = order.map(([p, i]) => ({ piece: tok.text(i), p, drawn: i === id }));
    if (id === eot) {
      logits = null;
      postMessage({ type: "token", end: true, candidates });
      reason = "end";
      break;
    }
    ids.push(id);
    logits = model.pos < model.cfg.max_seq_len ? model.step(id) : null;
    postMessage({ type: "token", piece: tok.text(id), p: probs[id], text: tok.decode(ids), candidates, perSecond: (1000 * (n + 1)) / (performance.now() - began) });
    await pause();
    if (mine !== latest) return; // stopped, or asked for something else
  }
  postMessage({ type: "done", reason });
}

onmessage = ({ data }) => {
  if (data.type === "stop") {
    latest++;
    postMessage({ type: "done", reason: "stopped" });
  } else {
    write(data).catch((error) => postMessage({ type: "error", message: error.message }));
  }
};
