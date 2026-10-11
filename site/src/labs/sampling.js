// The three knobs of sampling, on what the trained model really predicts
// after a prompt: temperature, top-k and top-p, as minllm/inference.py
// applies them.

import { Tokenizer } from "./bpe.js";
import { choice, html, json, slider } from "./kit.js";

// The probability of every token after the three knobs, in the order of inference.sample.
export function distribution(logits, { temperature, topK, topP }) {
  const n = logits.length;
  if (temperature === 0) {
    const best = logits.indexOf(Math.max(...logits));
    return Float64Array.from({ length: n }, (_, i) => (i === best ? 1 : 0));
  }
  let scaled = logits.map((v) => v / temperature);
  if (topK) {
    const kth = [...scaled].sort((a, b) => b - a)[Math.min(topK, n) - 1];
    scaled = scaled.map((v) => (v < kth ? -Infinity : v));
  }
  const peak = Math.max(...scaled);
  let probs = scaled.map((v) => Math.exp(v - peak));
  const sum = probs.reduce((s, v) => s + v, 0);
  probs = probs.map((v) => v / sum);
  if (topP < 1) {
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    let before = 0;
    const kept = new Float64Array(n);
    for (const [p, i] of order) {
      if (before >= topP) break;
      kept[i] = p;
      before += p;
    }
    const total = kept.reduce((s, v) => s + v, 0);
    probs = Array.from(kept, (v) => v / total);
  }
  return Float64Array.from(probs);
}

export default async function mount(el, { source = "/runs/base/next-token.json", knobs = true } = {}) {
  const [data, tok] = await Promise.all([json(source), json("/tokenizer.json").then((t) => new Tokenizer(t))]);
  const state = { prompt: 0, temperature: 1, topK: 0, topP: 1 };
  const bars = html("div", { class: "bars" });
  const readout = html("ul", { class: "readout" });
  const drawn = html("p", { class: "sample" });
  const button = html("button", { type: "button", class: "primary", text: "Sample 20 tokens" });
  const prompts = choice("After", data.prompts.map((p, i) => [`“…${p.prompt.split(" ").slice(-3).join(" ")}”`, i]), 0, (v) => ((state.prompt = v), draw()));
  const controls = html("div", { class: "controls" }, slider("Temperature", { min: 0, max: 2, step: 0.05, value: 1, format: (v) => v.toFixed(2) }, (v) => ((state.temperature = v), draw())));
  if (knobs) {
    controls.append(
      slider("Top-k", { min: 0, max: 50, value: 0, format: (v) => (v ? String(v) : "off") }, (v) => ((state.topK = v), draw())),
      slider("Top-p", { min: 0.05, max: 1, step: 0.05, value: 1, format: (v) => (v < 1 ? v.toFixed(2) : "off") }, (v) => ((state.topP = v), draw())),
    );
  }
  el.append(html("div", { class: "controls" }, prompts), controls, bars, readout);
  if (knobs) el.append(html("div", { class: "controls" }, button), drawn);
  el.append(html("p", { class: "lab-note", text: `The prediction is the ${data.model === "out/base" ? "14M model's" : "42M teacher's"} own, over all ${data.vocab_size.toLocaleString("en")} tokens, saved by scripts/dump_next_token.py in ${source.slice(1)}.` }));

  let probs;
  function draw() {
    const logits = data.prompts[state.prompt].logits;
    const plain = distribution(logits, { temperature: 1, topK: 0, topP: 1 });
    probs = distribution(logits, state);
    const order = Array.from(plain, (p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 12).map(([, i]) => i);
    const rest = 1 - order.reduce((s, i) => s + probs[i], 0);
    const row = (label, p, was) => {
      const fill = html("span", { class: `bar-fill${p === 0 ? " cut" : ""}` });
      fill.style.width = `${(100 * p).toFixed(2)}%`;
      const ghost = html("span", { class: "bar-ghost" });
      ghost.style.width = `${(100 * was).toFixed(2)}%`;
      return html("div", { class: `bar-row${p === 0 ? " cut" : ""}` }, html("span", { class: "bar-label", text: label }), html("span", { class: "bar-track" }, ghost, fill), html("span", { class: "bar-value", text: p === 0 ? "cut" : `${(100 * p).toFixed(p < 0.1 ? 1 : 0)}%` }));
    };
    bars.replaceChildren(...order.map((i) => row(tok.text(i).trim() || tok.text(i), probs[i], plain[i])), row("all the rest", Math.max(0, rest), 1 - order.reduce((s, i) => s + plain[i], 0)));
    const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
    readout.replaceChildren(
      html("li", {}, "Tokens that can still be drawn ", html("b", { text: probs.filter((p) => p > 0).length.toLocaleString("en") })),
      html("li", {}, "The likeliest gets ", html("b", { text: `${(100 * Math.max(...probs)).toFixed(0)}%` })),
      html("li", {}, "As uncertain as a choice among ", html("b", { text: `${Math.exp(entropy).toFixed(1)} tokens` })),
    );
    drawn.textContent = "";
  }
  button.addEventListener("click", () => {
    const pieces = [];
    for (let s = 0; s < 20; s++) {
      let r = Math.random();
      let i = 0;
      while (i < probs.length - 1 && (r -= probs[i]) > 0) i++;
      pieces.push(tok.text(i).trim() || "␠");
    }
    drawn.replaceChildren(html("b", { text: "Drawn: " }), pieces.join(" · "));
  });
  draw();
}
