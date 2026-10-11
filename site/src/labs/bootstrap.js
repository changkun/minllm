// How sure is a score, and is one model better than another? Resample the
// real per-problem scores of the GSM8K evaluations and watch the answer form.

import { choice, html, json, slider, svg } from "./kit.js";

const MODELS = [
  ["0.5B as published", "gsm8k-qwen"],
  ["fine-tuned, LoRA", "gsm8k-sft-gsm8k-lora"],
  ["fine-tuned, every weight", "gsm8k-sft-gsm8k"],
  ["after GRPO", "gsm8k-grpo-gsm8k"],
  ["after self-training", "gsm8k-self-gsm8k"],
  ["published, at 8 bits", "gsm8k-qwen-q8"],
  ["published, at 4 bits", "gsm8k-qwen-q4"],
];

// a small random number generator with a seed, so that a page shows the same thing twice
function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default async function mount(el) {
  const scores = Object.fromEntries(await Promise.all(MODELS.map(async ([, file]) => [file, (await json(`/runs/eval/${file}.json`)).scores])));
  const total = scores[MODELS[0][1]].length;
  const state = { a: MODELS[0][1], b: MODELS[1][1], n: total, timer: null };
  const figure = svg("svg", { viewBox: "0 0 720 190", role: "img", "aria-label": "Resampled differences between two models" });
  const readout = html("ul", { class: "readout" });
  const verdict = html("p", { class: "sample" });
  const select = (label, value, set) => {
    const node = html("select", {}, ...MODELS.map(([name, file]) => html("option", { value: file, text: name })));
    node.value = value;
    node.addEventListener("change", () => (set(node.value), run()));
    return html("label", {}, label, node);
  };
  el.append(
    html("div", { class: "controls" }, select("Model", state.a, (v) => (state.a = v)), select("compared with", state.b, (v) => (state.b = v)), slider("Problems used", { min: 50, max: total, step: 1, value: total, format: (v) => v.toLocaleString("en") }, (v) => ((state.n = v), run()))),
    figure,
    readout,
    verdict,
    html("p", { class: "lab-note", text: "Every bar is a count of resamples: the same problems drawn again with replacement, 2,000 times, and the difference between the two models on each draw. The scores are the per-problem results in runs/eval/." }),
  );

  function run() {
    clearInterval(state.timer);
    const a = scores[state.a].slice(0, state.n);
    const b = scores[state.b].slice(0, state.n);
    const n = state.n;
    const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
    const [ma, mb] = [mean(a), mean(b)];
    const rng = random(1);
    const diffs = [];
    const limit = Math.max(0.12, Math.abs(mb - ma) * 1.8);
    const bins = 60;
    const draw = () => {
      figure.replaceChildren();
      const counts = new Array(bins).fill(0);
      for (const d of diffs) counts[Math.min(bins - 1, Math.max(0, Math.floor(((d + limit) / (2 * limit)) * bins)))]++;
      const peak = Math.max(1, ...counts);
      const x = (v) => 20 + (680 * (v + limit)) / (2 * limit);
      counts.forEach((c, i) => c && svg("rect", { x: 20 + (i * 680) / bins, y: 150 - (130 * c) / peak, width: 680 / bins - 1, height: (130 * c) / peak, class: "f-s1", "fill-opacity": 0.75 }, figure));
      svg("line", { x1: 20, y1: 150.5, x2: 700, y2: 150.5, class: "s-axis" }, figure);
      svg("line", { x1: x(0), y1: 14, x2: x(0), y2: 156, class: "s-ink", "stroke-width": 1.5 }, figure);
      svg("text", { x: x(0), y: 172, "text-anchor": "middle", class: "f-ink" }, figure).textContent = "no difference";
      for (const t of [-limit, limit]) svg("text", { x: x(t), y: 172, "text-anchor": t < 0 ? "start" : "end", class: "f-muted" }, figure).textContent = `${t > 0 ? "+" : "−"}${(100 * Math.abs(t)).toFixed(0)} points`;
      if (diffs.length >= 200) {
        const sorted = [...diffs].sort((p, q) => p - q);
        const [low, high] = [sorted[Math.floor(0.025 * sorted.length)], sorted[Math.floor(0.975 * sorted.length)]];
        svg("line", { x1: x(low), y1: 160, x2: x(high), y2: 160, class: "s-s2", "stroke-width": 4, "stroke-linecap": "round" }, figure);
        const real = low > 0 || high < 0;
        const err = (xs, m) => 1.96 * Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / (n - 1) / n);
        readout.replaceChildren(
          html("li", {}, `${MODELS.find((m) => m[1] === state.a)[0]} `, html("b", { text: `${(100 * ma).toFixed(1)}% ± ${(100 * err(a, ma)).toFixed(1)}` })),
          html("li", {}, `${MODELS.find((m) => m[1] === state.b)[0]} `, html("b", { text: `${(100 * mb).toFixed(1)}% ± ${(100 * err(b, mb)).toFixed(1)}` })),
          html("li", {}, "Difference ", html("b", { text: `${mb >= ma ? "+" : "−"}${(100 * Math.abs(mb - ma)).toFixed(1)} points` })),
          html("li", {}, "95% of the resamples lie between ", html("b", { text: `${(100 * low).toFixed(1)} and ${(100 * high).toFixed(1)}` })),
        );
        verdict.replaceChildren(html("b", { text: real ? "A real difference. " : "Within noise. " }), real ? `The interval does not reach zero: on ${n.toLocaleString("en")} problems the second model is ${mb > ma ? "better" : "worse"}.` : `The interval includes zero: ${n.toLocaleString("en")} problems cannot tell these two apart.`);
      }
    };
    state.timer = setInterval(() => {
      for (let r = 0; r < 100; r++) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const j = Math.floor(rng() * n);
          sum += b[j] - a[j];
        }
        diffs.push(sum / n);
      }
      draw();
      if (diffs.length >= 2000) clearInterval(state.timer);
    }, 30);
  }
  run();
}
