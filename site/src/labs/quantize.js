// Round one real row of weights of the 0.5B model, with the rule of
// minllm/quantize.py, at any bit width and group size.

import { choice, html, json, slider, svg } from "./kit.js";

const W = 720;
const LEFT = 16;
const WIDTH = W - 2 * LEFT;

// codes = round((w - low) / scale): the same arithmetic as quantize.quantize
function round(weights, bits, size) {
  const levels = 2 ** bits - 1;
  const out = new Float64Array(weights.length);
  const groups = [];
  for (let start = 0; start < weights.length; start += size) {
    const group = weights.slice(start, start + size);
    const low = Math.min(...group);
    const scale = (Math.max(...group) - low) / levels || 1;
    group.forEach((w, i) => (out[start + i] = Math.round((w - low) / scale) * scale + low));
    groups.push({ start, low, scale });
  }
  return { out, groups };
}

export default async function mount(el) {
  const data = await json("/runs/quant/row.json");
  const sweep = await json("/runs/quant/sweep-qwen.json");
  const weights = data.weights;
  const lo = Math.min(...weights);
  const hi = Math.max(...weights);
  const x = (v) => LEFT + (WIDTH * (v - lo)) / (hi - lo);
  const state = { bits: 4, size: 64, group: 0 };

  const figure = svg("svg", { viewBox: `0 0 ${W} 250`, role: "img", "aria-label": "A row of weights and the values it is rounded to" });
  const readout = html("ul", { class: "readout" });
  const groupSlider = slider("Group shown", { min: 1, max: weights.length / 64, value: 1 }, (v) => ((state.group = v - 1), draw()));
  const controls = html(
    "div",
    { class: "controls" },
    slider("Bits per weight", { min: 2, max: 8, value: state.bits }, (v) => ((state.bits = v), draw())),
    choice("One scale per", [["row", weights.length], ["64 weights", 64], ["16 weights", 16]], state.size, (v) => ((state.size = v), draw())),
    groupSlider,
  );
  el.append(controls, figure, readout, html("p", { class: "lab-note", text: `The row is the one of the figure above: ${weights.length} weights of ${data.matrix}, row ${data.row}, from runs/quant/row.json. The loss in the last line is read from the sweep in runs/quant/sweep-qwen.json.` }));

  function draw() {
    const { bits, size } = state;
    const count = weights.length / size;
    const index = Math.min(state.group, count - 1);
    groupSlider.style.display = count > 1 ? "" : "none";
    groupSlider.querySelector("input").max = count;
    groupSlider.firstChild.textContent = `Group shown, of ${count}`;
    const { out, groups } = round(weights, bits, size);
    const g = groups[index];
    figure.replaceChildren();

    // the whole row as a histogram, with the chosen group's part of it marked
    svg("text", { x: LEFT, y: 14, class: "f-ink", "font-weight": 600 }, figure).textContent = "The row, and where its weights lie";
    const bins = 148;
    const counts = new Array(bins).fill(0);
    const mine = new Array(bins).fill(0);
    weights.forEach((w, i) => {
      const b = Math.min(bins - 1, Math.floor(((w - lo) / (hi - lo)) * bins));
      counts[b]++;
      if (i >= g.start && i < g.start + size) mine[b]++;
    });
    const peak = Math.max(...counts);
    counts.forEach((c, b) => {
      if (!c) return;
      svg("rect", { x: LEFT + (b * WIDTH) / bins, y: 84 - (56 * c) / peak, width: WIDTH / bins - 1, height: (56 * c) / peak, class: "f-s1", "fill-opacity": 0.3 }, figure);
      if (mine[b]) svg("rect", { x: LEFT + (b * WIDTH) / bins, y: 84 - (56 * mine[b]) / peak, width: WIDTH / bins - 1, height: (56 * mine[b]) / peak, class: "f-s1" }, figure);
    });
    svg("line", { x1: LEFT, y1: 84.5, x2: LEFT + WIDTH, y2: 84.5, class: "s-axis" }, figure);

    // the group: every weight as a tick, the values the codes can name as lines, and where each weight lands
    svg("text", { x: LEFT, y: 122, class: "f-ink", "font-weight": 600 }, figure).textContent = count > 1 ? `Group ${index + 1}: ${size} weights and their ${2 ** bits} values` : `The row's ${2 ** bits} values`;
    const top = 136;
    const shown = 2 ** bits <= 64;
    for (let k = 0; k < 2 ** bits && shown; k++) {
      const v = g.low + k * g.scale;
      svg("line", { x1: x(v), y1: top, x2: x(v), y2: top + 84, class: "s-s2", "stroke-width": 1.5 }, figure);
    }
    if (!shown) svg("rect", { x: x(g.low), y: top, width: x(g.low + (2 ** bits - 1) * g.scale) - x(g.low), height: 84, class: "f-s2", "fill-opacity": 0.25 }, figure);
    for (let i = g.start; i < g.start + size; i++) {
      // from the weight, above, to the value it is rounded to, below
      svg("line", { x1: x(weights[i]), y1: top + 6, x2: x(weights[i]), y2: top + 30, class: "s-s1", "stroke-width": 1.5 }, figure);
      svg("line", { x1: x(weights[i]), y1: top + 30, x2: x(out[i]), y2: top + 62, class: "s-muted", "stroke-width": 0.75 }, figure);
    }
    svg("line", { x1: LEFT, y1: top + 84.5, x2: LEFT + WIDTH, y2: top + 84.5, class: "s-axis" }, figure);
    svg("text", { x: LEFT, y: top + 102, class: "f-muted" }, figure).textContent = "Blue: a weight. Orange: a value a code can name. Each weight moves to the nearest one, by at most half a step.";

    const rms = (values) => Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length);
    const error = rms(weights.map((w, i) => out[i] - w)) / rms(weights);
    const stored = bits + 32 / size;
    const measured = sweep.settings.find((r) => r.bits === bits && (r.group || weights.length) === size);
    readout.replaceChildren(
      html("li", {}, "Step in this group ", html("b", { text: g.scale.toFixed(4) })),
      html("li", {}, "Error of the row, against the size of its weights ", html("b", { text: `${(100 * error).toFixed(1)}%` })),
      html("li", {}, "Stored per weight ", html("b", { text: `${stored.toFixed(2)} bits` })),
      html("li", {}, "The 0.5B model's file ", html("b", { text: `${Math.round((sweep.parameters * stored) / 8e6)} MB` })),
      html("li", {}, "Its held-out loss, from ", html("b", { text: sweep.loss.toFixed(3) }), " to ", html("b", { text: measured ? measured.loss.toFixed(3) : "not measured" })),
    );
  }
  draw();
}
