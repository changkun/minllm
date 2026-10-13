// The side topic 2b as a film: the model of 3b, trained on windows of 256
// tokens, read further than that. The rotary table's slow pairs reach angles
// past 256 that training never showed; three ways of stretching the table
// keep them on known ones. The loss along long pages and the worth of a
// passage seen before, without training, after 500 steps on windows of
// 1,024, and for a model trained on 1,024 from the start, are those of
// runs/context/. The turning speeds of the pairs are the method's, computed
// here as minllm/model.py's rope_table computes them.

import { svg } from "../labs/kit.js";
import { clamp, during, mix, set, show } from "./stage.js";

export const sources = {
  plain: "/runs/context/web-clean.json",
  linear4: "/runs/context/web-clean-linear4.json",
  ntk4: "/runs/context/web-clean-ntk4.json",
  yarn4: "/runs/context/web-clean-yarn4.json",
  yarn8: "/runs/context/web-clean-yarn8.json",
  none: "/runs/context/ctx-none.json",
  linear: "/runs/context/ctx-linear.json",
  ntk: "/runs/context/ctx-ntk.json",
  yarn: "/runs/context/ctx-yarn.json",
  scratch: "/runs/context/web-clean-1024.json",
  yarnLog: "/runs/context/train-ctx-yarn.jsonl",
  scratchLog: "/runs/context/train-web-clean-1024.jsonl",
};

const count = (n) => n.toLocaleString("en");
const percent = (v) => `${Math.round(100 * v)}%`;
const TURN = 2 * Math.PI;
const LONG = 1024; // the window the stretch by 4 is for
const READ = 2048; // how much of a long page is read

// The turning speed of every pair of the rotary table, in radians per token,
// as rope_table computes it: theta 10,000, a head of 64, stretched by s.
function speeds(kind = null, s = 1, original = 256, theta = 1e4, head = 64) {
  const half = head / 2;
  const base = Array.from({ length: half }, (_, i) => theta ** ((-2 * i) / head));
  if (kind === "linear") return base.map((f) => f / s);
  if (kind === "ntk") {
    const raised = theta * s ** (head / (head - 2));
    return base.map((_, i) => raised ** ((-2 * i) / head));
  }
  if (kind === "yarn") {
    // the pair at which a pair turns `turns` times within the original window
    const index = (turns) => (head * Math.log(original / (turns * TURN))) / (2 * Math.log(theta));
    const low = Math.max(Math.floor(index(32)), 0);
    const high = Math.min(Math.ceil(index(1)), head - 1);
    return base.map((f, i) => {
      const ramp = clamp((i - low) / Math.max(high - low, 1e-3)); // 0: as before, 1: interpolated
      return f * (1 - ramp) + (f / s) * ramp;
    });
  }
  return base;
}

// The frame of a chart, with its labels at 22 pixels. Returns { px, py }.
function frame(s, x, y, w, h, across, up) {
  const part = (v, { domain, log }) => (log ? Math.log(v / domain[0]) / Math.log(domain[1] / domain[0]) : (v - domain[0]) / (domain[1] - domain[0]));
  const px = (v) => x + w * part(v, across);
  const py = (v) => y + h * (1 - part(v, up));
  for (const tick of up.ticks) {
    s.line(x, py(tick), x + w, py(tick), { stroke: "grid", width: 1.5 });
    s.text(x - 12, py(tick) + 8, up.format(tick), { size: 22, anchor: "end", fill: "muted" });
  }
  s.line(x, y + h, x + w, y + h, { stroke: "axis" });
  for (const tick of across.ticks) s.text(px(tick), y + h + 30, across.format(tick), { size: 22, anchor: "middle", fill: "muted" });
  if (across.title) s.text(x + w / 2, y + h + 62, across.title, { size: 22, anchor: "middle", fill: "muted" });
  if (up.title) s.text(x, y - 16, up.title, { size: 22, fill: "muted" });
  return { px, py };
}

// A line through points, redrawn as given; hidden when there is nothing to draw.
function shape(node, points) {
  node.setAttribute("d", points.length > 1 ? "M" + points.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join(" L") : "M0 0");
  show(node, points.length > 1 ? 1 : 0);
}

// The points of a line up to x, the last one cut where x is.
function upto(points, x) {
  const kept = [];
  for (const [i, [a, b]] of points.entries()) {
    if (a <= x) {
      kept.push([a, b]);
      continue;
    }
    if (i > 0) {
      const [a0, b0] = points[i - 1];
      kept.push([x, mix(b0, b, (x - a0) / (a - a0))]);
    }
    break;
  }
  return kept;
}

// A slice of a dial, clockwise from twelve o'clock, between two angles.
function wedge(cx, cy, r, from, to) {
  if (to - from >= TURN - 1e-6) return `M${cx} ${cy - r} A${r} ${r} 0 1 1 ${cx} ${cy + r} A${r} ${r} 0 1 1 ${cx} ${cy - r} Z`;
  if (to - from <= 1e-6) return "M0 0";
  const at = (a) => `${(cx + r * Math.sin(a)).toFixed(2)} ${(cy - r * Math.cos(a)).toFixed(2)}`;
  return `M${cx} ${cy} L${at(from)} A${r} ${r} 0 ${to - from > Math.PI ? 1 : 0} 1 ${at(to)} Z`;
}

// How fast during(t, a, b) changes at t, per second.
function pace(t, a, b) {
  const u = clamp((t - a) / (b - a));
  return (6 * u * (1 - u)) / (b - a);
}

export default ({ plain, linear4, ntk4, yarn4, yarn8, none, linear, ntk, yarn, scratch, yarnLog, scratchLog }) => {
  const L = plain.trained_length; // the window the model was trained on
  const over = scratch.held_out_tokens_in_documents_over; // the same held-out pages for every run
  const pages = plain.documents;
  const worth = (run, d) => run.passage[d].unseen - run.passage[d].seen;
  const distances = Object.keys(scratch.passage).map(Number).sort((a, b) => a - b);
  const FAR = 512; // the distance the passage is compared at
  const final = (log) => log.filter((e) => "val_loss" in e).at(-1);
  // from which distance on the stretched model finds the passage better than the one trained long from the start
  const beyond = distances.find((d, i) => distances.slice(i).every((e) => worth(yarn, e) > worth(scratch, e)));

  // the loss chart of the two scenes on long pages, and the worth of a passage FAR back beside it
  function pageChart(s, runs, label) {
    const X = 150;
    const Y = 150;
    const WIDE = 600;
    const TALL = 310;
    const low = 3;
    const high = 5;
    const { px, py } = frame(s, X, Y, WIDE, TALL, { domain: [0, READ], ticks: [0, L, LONG, READ], format: count, title: "position in the page" }, { domain: [low, high], ticks: [3, 3.5, 4, 4.5, 5], format: (v) => v.toFixed(1), title: "loss" });
    for (const at of [L, LONG]) s.line(px(at), Y, px(at), Y + TALL, { stroke: "axis", width: 1.5 });
    const state = label.map((text) => s.text(X + WIDE, Y - 16, text, { size: 24, weight: 600, anchor: "end" }));
    const points = (run) => run.loss_by_32_positions.map((v, i) => [px(32 * i + 16), py(clamp(v, low, high))]);
    const head = s.group();
    head.text(810, 164, `a passage ${count(FAR)} back`, { size: 22, fill: "muted" });
    head.text(810, 192, "is worth", { size: 22, fill: "muted" });
    const rows = runs.map((r, i) => {
      const y = 246 + i * 56;
      const g = s.group();
      const swatch = g.line(810, y - 8, 846, y - 8, { stroke: r.fill, width: 4 });
      if (r.dash) swatch.setAttribute("stroke-dasharray", "9 6");
      g.text(860, y, r.name, { size: 24 });
      const curve = s.path([[0, 0], [0, 0]], { stroke: r.fill, width: 3.5 });
      if (r.dash) curve.setAttribute("stroke-dasharray", "10 7");
      return { ...r, g, curve, value: s.text(1205, y + 1, "", { size: 26, weight: 700, anchor: "end" }) };
    });
    return { px, py, points, state, head, rows };
  }

  const untrained = [
    { name: "not stretched", fill: "ink2", run: plain },
    { name: "interpolation ×4", fill: "s1", run: linear4 },
    { name: "NTK ×4", fill: "s2", run: ntk4 },
    { name: "YaRN ×4", fill: "s3", run: yarn4 },
    { name: "YaRN ×8", fill: "s3", dash: true, run: yarn8 },
  ];
  const trained = [none, linear, ntk, yarn].map((run, i) => ({ ...untrained[i], before: untrained[i].run, run }));

  return {
    subtitle: "Reading past the window the model was trained on",
    scenes: [
      {
        seconds: 6.5,
        caption: `The model of 3b was trained on web text in windows of ${L}.\nYet ${percent(over[LONG])} of the text held out is in pages over ${count(LONG)} tokens.`,
        draw(s) {
          const X = 140;
          const WIDE = 1000;
          const Y = 170;
          const TALL = 54;
          const unit = WIDE / READ;
          const page = s.group();
          page.text(X, Y - 18, "the first tokens of a long web page", { size: 24, fill: "ink2" });
          page.text(X + WIDE, Y - 18, `${count(READ)} tokens`, { size: 22, anchor: "end", fill: "muted" });
          page.rect(X, Y, WIDE, TALL, { fill: "raised", stroke: "axis", r: 8 });
          const windows = Array.from({ length: READ / L }, (_, i) => {
            const g = s.group();
            g.rect(X + i * L * unit + 5, Y + 7, L * unit - 10, TALL - 14, { fill: "tint", stroke: "ink2", r: 6 });
            g.text(X + (i + 0.5) * L * unit, Y + TALL / 2 + 8, count(L), { size: 22, anchor: "middle", fill: "ink2" });
            return g;
          });
          const alone = s.text(X, Y + TALL + 40, `In training, each window of ${L} tokens is read on its own.`, { size: 22, fill: "muted" });
          const title = s.text(X, 346, "the share of the text held out that is in pages longer than", { size: 24, fill: "ink2" });
          const rows = Object.keys(over)
            .sort((a, b) => Number(a) - Number(b))
            .map((key, i) => {
              const y = 372 + i * 62;
              const marked = Number(key) === LONG;
              const g = s.group();
              g.text(X + 170, y + 30, `${count(Number(key))} tokens`, { size: 24, anchor: "end", weight: marked ? 700 : 400 });
              const w = 760 * over[key];
              return { g, w, bar: s.rect(X + 190, y, 0, 42, { fill: marked ? "ink2" : "axis", r: 6 }), value: s.text(X + 204 + w, y + 31, percent(over[key]), { size: 26, weight: 700 }) };
            });
          return (t) => {
            show(page, during(t, 0.2, 0.6));
            windows.forEach((g, i) => show(g, during(t, 0.8 + i * 0.15, 1.1 + i * 0.15)));
            show(alone, during(t, 2.0, 2.4));
            show(title, during(t, 2.5, 2.9));
            rows.forEach((row, i) => {
              const u = during(t, 2.8 + i * 0.3, 3.8 + i * 0.3);
              show(row.g, during(t, 2.7 + i * 0.3, 3.0 + i * 0.3));
              set(row.bar, { width: row.w * u });
              show(row.bar, u > 0 ? 1 : 0);
              show(row.value, during(u, 0.8, 1));
            });
          };
        },
      },
      {
        seconds: 8,
        caption: `Rotary embeddings turn each pair of dimensions with position.\nPast ${L}, the slow pairs reach angles never seen in training.`,
        draw(s) {
          const turnsOf = (v) => (v === 0 ? "0 turns" : v >= 10 ? `${Math.round(v)} turns` : v >= 1 ? `${v.toFixed(1)} turns` : `${v.toPrecision(2)} turns`);
          const freq = speeds();
          const R = 72;
          const CY = 318;
          const counter = s.text(640, 150, "", { size: 30, weight: 600, anchor: "middle" });
          const ends = s.group();
          ends.text(200, 222, "fastest", { size: 22, anchor: "middle", fill: "muted" });
          ends.text(1080, 222, "slowest", { size: 22, anchor: "middle", fill: "muted" });
          ends.arrow(252, 215, 1028, 215, { stroke: "axis", width: 2 });
          const dials = [0, 8, 16, 24, 31].map((i, k) => {
            const cx = 200 + 220 * k;
            const g = s.group();
            g.circle(cx, CY, R, { fill: "raised" });
            const seen = svg("path", { d: "M0 0", class: "f-axis" }, g.g);
            const fresh = svg("path", { d: "M0 0", class: "f-ink2", "fill-opacity": 0.6 }, g.g);
            svg("circle", { cx, cy: CY, r: R, fill: "none", class: "s-axis", "stroke-width": 2 }, g.g);
            const hand = g.line(cx, CY, cx, CY - R, { stroke: "ink", width: 3.5 });
            g.circle(cx, CY, 5, { fill: "ink" });
            g.text(cx, 430, `pair ${i + 1}`, { size: 24, weight: 600, anchor: "middle" });
            const turns = g.text(cx, 462, "", { size: 22, anchor: "middle", fill: "ink2" });
            return { g, cx, f: freq[i], seen, fresh, hand, turns };
          });
          const known = s.group();
          known.rect(250, 512, 24, 24, { fill: "axis", r: 4 });
          known.text(286, 532, `angles reached within ${L} tokens`, { size: 22, fill: "ink2" });
          const past = s.group();
          past.rect(700, 512, 24, 24, { fill: "ink2", r: 4 }).setAttribute("fill-opacity", 0.6);
          past.text(736, 532, `new angles, first reached past ${L}`, { size: 22, fill: "ink2" });
          const A = [0.9, 3.5]; // the position runs to the end of the trained window,
          const B = [4.3, 6.0]; // and then on to 1,024
          return (t) => {
            const p = L * during(t, ...A) + (LONG - L) * during(t, ...B);
            const rate = L * pace(t, ...A) + (LONG - L) * pace(t, ...B); // tokens per second of film
            counter.textContent = `position ${count(Math.round(p))}`;
            show(counter, during(t, 0.2, 0.6));
            show(ends, during(t, 0.4, 0.8));
            show(known, during(t, 0.8, 1.2));
            show(past, during(t, B[0], B[0] + 0.4));
            dials.forEach((d, k) => {
              show(d.g, during(t, 0.2 + k * 0.1, 0.6 + k * 0.1));
              const a = p * d.f;
              set(d.hand, { x2: d.cx + R * Math.sin(a), y2: CY - R * Math.cos(a) });
              // a hand that turns too fast to follow is a blur: it fades
              const spin = rate * d.f;
              show(d.hand, spin > 0 ? clamp(8 / spin) : 1);
              const reached = Math.min(Math.min(p, L) * d.f, TURN);
              d.seen.setAttribute("d", wedge(d.cx, CY, R, 0, reached));
              d.fresh.setAttribute("d", p > L ? wedge(d.cx, CY, R, reached, Math.min(a, TURN)) : "M0 0");
              d.turns.textContent = turnsOf(a / TURN);
            });
          };
        },
      },
      {
        seconds: 8,
        caption: `Three stretches by 4 keep the slow pairs on angles they know.\nInterpolation slows all pairs alike. NTK and YaRN do not.`,
        draw(s) {
          const X = 150;
          const Y = 150;
          const WIDE = 480;
          const TALL = 320;
          const { px, py } = frame(
            s,
            X,
            Y,
            WIDE,
            TALL,
            { domain: [0, 31], ticks: [0, 7, 15, 23, 31], format: (i) => String(i + 1), title: "pair, from the fastest to the slowest" },
            { domain: [0.004, 300], log: true, ticks: [0.01, 0.1, 1, 10, 100], format: String, title: "turns within the window" },
          );
          s.line(X, py(1), X + WIDE, py(1), { stroke: "muted", width: 1.5 });
          s.text(X + 10, py(1) - 10, "one turn", { size: 22, fill: "muted" });
          const turns = (freqs, length) => freqs.map((f) => (length * f) / TURN);
          const trainedBand = turns(speeds(), L);
          const plainLine = turns(speeds(), LONG);
          const points = (values) => values.map((v, i) => [px(i), py(v)]);
          const between = (from, to, u) => from.map((v, i) => Math.exp(mix(Math.log(v), Math.log(to[i]), u)));
          const band = s.path(points(trainedBand), { stroke: "axis", width: 12 });
          const lines = [
            { fill: "ink2", from: trainedBand, to: plainLine, at: 1.0 },
            { fill: "s1", from: plainLine, to: turns(speeds("linear", 4, L), LONG), at: 2.1 },
            { fill: "s2", from: plainLine, to: turns(speeds("ntk", 4, L), LONG), at: 3.2 },
            { fill: "s3", from: plainLine, to: turns(speeds("yarn", 4, L), LONG), at: 4.3 },
          ].map((l) => ({ ...l, path: s.path(points(l.from), { stroke: l.fill, width: l.fill === "s3" ? 4 : 3.5 }) }));
          const LX = 690;
          const TX = 744;
          const legend = (y, fill, width, name, notes = []) => {
            const g = s.group();
            g.line(LX, y - 8, LX + 38, y - 8, { stroke: fill, width });
            g.text(TX, y, name, { size: 24, weight: notes.length ? 600 : 400 });
            notes.forEach((note, i) => g.text(TX, y + 30 * (i + 1), note, { size: 22, fill: "ink2" }));
            return g;
          };
          const keys = [
            legend(168, "axis", 12, `as trained, over ${L} tokens`),
            legend(210, "ink2", 4, `not stretched, over ${count(LONG)}`),
            legend(268, "s1", 4, "interpolation", ["every pair 4 times slower"]),
            legend(350, "s2", 4, "NTK", ["theta raised: the slowest pair 4 times", "slower, the fastest as before"]),
            legend(462, "s3", 4, "YaRN", ["fast pairs as before, slow ones", "interpolated, a ramp between,", "and attention a little sharper"]),
          ];
          return (t) => {
            show(band, during(t, 0.3, 0.7));
            show(keys[0], during(t, 0.3, 0.7));
            lines.forEach((l, i) => {
              const u = during(t, l.at, l.at + 0.8);
              shape(l.path, points(between(l.from, l.to, u)));
              show(l.path, during(t, l.at - 0.2, l.at));
              show(keys[i + 1], during(t, l.at - 0.2, l.at + 0.2));
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `Two things are measured: the loss along the first ${count(READ)} tokens\nof long pages, and what a passage seen before is worth.`,
        draw(s) {
          const X = 140;
          const WIDE = 1000;
          const one = s.text(X, 140, `1. The loss at every position of the first ${count(READ)} tokens of ${pages} long pages`, { size: 24 });
          const blocks = READ / 32;
          const cells = Array.from({ length: blocks }, (_, i) => s.rect(X + (i * WIDE) / blocks + 1, 164, WIDE / blocks - 2, 40, { fill: "tint", r: 3 }));
          const cursor = s.line(X, 156, X, 212, { stroke: "ink", width: 3 });
          const ticks = s.group();
          for (const at of [0, L, LONG, READ]) ticks.text(X + (WIDE * at) / READ, 234, count(at), { size: 22, anchor: "middle", fill: "muted" });
          const two = s.text(X, 300, "2. A passage of 48 tokens, other pages, then the passage again", { size: 24 });
          const R0 = 230;
          const R1 = 1150;
          const PASS = 180;
          const rowY = [384, 444];
          const box = (g, x, y, w, text, fill, stroke, ink) => {
            g.rect(x, y, w, 44, { fill, stroke, r: 6 });
            g.text(x + w / 2, y + 29, text, { size: 22, anchor: "middle", fill: ink });
          };
          const rows = rowY.map((y, r) => {
            const g = s.group();
            g.text(70, y + 30, r ? "unseen" : "seen", { size: 24, weight: 600 });
            const first = s.group();
            if (r) box(first, R0, y, PASS, "another one", "raised", "ink2", "ink2");
            else box(first, R0, y, PASS, "passage", "ink2", null, "surface");
            const filler = s.group();
            box(filler, R0 + PASS + 6, y, R1 - R0 - 2 * PASS - 12, "other pages", "tint", "axis", "muted");
            const copy = s.group();
            box(copy, R1 - PASS, y, PASS, "passage", "ink2", null, "surface");
            return { g, parts: [first, filler, copy] };
          });
          const apart = s.group();
          const mid = (R0 + R1 - PASS) / 2;
          apart.arrow(mid, 368, R0 + 2, 368, { stroke: "ink2", width: 2 });
          apart.arrow(mid, 368, R1 - PASS - 2, 368, { stroke: "ink2", width: 2 });
          apart.text(mid, 356, `${count(distances[0])} to ${count(distances.at(-1))} tokens apart`, { size: 22, anchor: "middle", fill: "ink2" });
          const rule = s.text(X, 548, "What the first copy is worth: the loss unseen minus the loss seen.", { size: 24 });
          return (t) => {
            show(one, during(t, 0.2, 0.6));
            const u = during(t, 0.8, 2.6);
            cells.forEach((c, i) => {
              show(c, during(t, 0.4, 0.7));
              c.setAttribute("class", (i + 0.5) / blocks <= u ? "f-axis" : "f-tint");
            });
            set(cursor, { x1: X + WIDE * u, x2: X + WIDE * u });
            show(cursor, t > 0.8 && u < 1 ? 1 : 0);
            show(ticks, during(t, 0.4, 0.8));
            show(two, during(t, 2.6, 3.0));
            rows.forEach((row, r) => {
              const from = 3.0 + r * 1.0;
              show(row.g, during(t, from, from + 0.3));
              row.parts.forEach((part, k) => show(part, during(t, from + k * 0.2, from + 0.3 + k * 0.2)));
            });
            show(apart, during(t, 3.6, 4.0));
            show(rule, during(t, 4.8, 5.2));
          };
        },
      },
      {
        seconds: 8,
        caption: `Not stretched, the model breaks right after ${L} tokens.\nWithout training, YaRN by 4 holds to ${count(LONG)} and by 8 to ${count(READ)}.`,
        draw(s) {
          const chart = pageChart(s, untrained, ["without training"]);
          const { px, points } = chart;
          return (t) => {
            show(chart.state[0], during(t, 0.2, 0.6));
            show(chart.head, during(t, 0.3, 0.7));
            chart.rows.forEach((r, i) => {
              const from = i ? 2.4 : 0.6;
              show(r.g, during(t, from - 0.2, from + 0.2));
              shape(r.curve, upto(points(r.run), px(READ * during(t, from, from + (i ? 2.2 : 1.6)))));
              r.value.textContent = worth(r.run, FAR).toFixed(2);
              show(r.value, during(t, 4.9 + i * 0.15, 5.3 + i * 0.15));
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `After 500 steps on windows of ${count(LONG)}, all four hold to ${count(LONG)}.\nA passage ${FAR} back is worth ${worth(yarn, FAR).toFixed(2)} to YaRN, ${worth(none, FAR).toFixed(2)} unstretched.`,
        draw(s) {
          const chart = pageChart(s, trained, ["without training", `after 500 steps on windows of ${count(LONG)}`]);
          const { points } = chart;
          return (t) => {
            show(chart.state[0], 1 - during(t, 0.6, 0.9));
            show(chart.state[1], during(t, 0.9, 1.2));
            const u = during(t, 0.9, 3.2);
            chart.rows.forEach((r) => {
              const before = points(r.before);
              shape(
                r.curve,
                points(r.run).map(([x, y], i) => [x, mix(before[i][1], y, u)]),
              );
              r.value.textContent = mix(worth(r.before, FAR), worth(r.run, FAR), u).toFixed(2);
              // the two the caption compares stay, the others step back
              const aside = r.fill === "s1" || r.fill === "s2" ? during(t, 4.0, 4.4) : 0;
              show(r.curve, 1 - 0.7 * aside);
              show(r.g, 1 - 0.6 * aside);
              show(r.value, 1 - 0.6 * aside);
            });
          };
        },
      },
      {
        seconds: 7,
        caption: "Long windows from the start copy more over short distances.\nShort windows first end lower, with 8.2M tokens more.",
        draw(s) {
          const X = 150;
          const Y = 150;
          const WIDE = 560;
          const TALL = 310;
          const { px, py } = frame(
            s,
            X,
            Y,
            WIDE,
            TALL,
            { domain: [0, READ], ticks: [0, 512, 1024, 1536, 2048], format: count, title: "tokens between the two copies" },
            { domain: [0, 2.2], ticks: [0, 0.5, 1, 1.5, 2], format: (v) => v.toFixed(1), title: "what the first copy is worth" },
          );
          const near = distances[0];
          const runs = [
            { name: "YaRN ×4 after 500 steps", fill: "s3", run: yarn, log: yarnLog, y: 180 },
            { name: `${count(LONG)} from the start`, fill: "ink", run: scratch, log: scratchLog, y: 322 },
          ].map((r) => {
            const pts = distances.map((d) => [px(d), py(worth(r.run, d))]);
            const g = s.group();
            g.line(790, r.y - 8, 826, r.y - 8, { stroke: r.fill, width: 4 });
            g.circle(808, r.y - 8, 6, { fill: r.fill });
            g.text(840, r.y, r.name, { size: 24, weight: 600 });
            g.text(840, r.y + 36, `loss on windows of ${count(LONG)}`, { size: 22, fill: "ink2" });
            g.text(840, r.y + 70, `a passage ${near} back is worth`, { size: 22, fill: "ink2" });
            return {
              ...r,
              g,
              pts,
              path: s.path(pts, { stroke: r.fill, width: 3.5 }),
              dots: pts.map(([x, y]) => s.circle(x, y, 6, { fill: r.fill })),
              loss: s.text(1205, r.y + 37, final(r.log).val_loss.toFixed(3), { size: 26, weight: 700, anchor: "end" }),
              copied: s.text(1205, r.y + 71, worth(r.run, near).toFixed(2), { size: 26, weight: 700, anchor: "end" }),
            };
          });
          const further = s.group();
          further.text(840, 470, `From ${count(beyond ?? READ)} tokens back on,`, { size: 22, fill: "ink2" });
          further.text(840, 500, "the stretched model finds more.", { size: 22, fill: "ink2" });
          return (t) => {
            const x = px(READ * during(t, 0.6, 2.6));
            runs.forEach((r, i) => {
              show(r.g, during(t, 0.4 + i * 0.3, 0.8 + i * 0.3));
              shape(r.path, upto(r.pts, x));
              r.dots.forEach((dot, k) => {
                show(dot, r.pts[k][0] <= x + 0.5 ? 1 : 0);
                set(dot, { r: k === 0 ? 6 + 4 * during(t, 3.6, 4.0) : 6 });
              });
              show(r.loss, during(t, 2.8 + i * 0.3, 3.2 + i * 0.3));
              show(r.copied, during(t, 3.6 + i * 0.3, 4.0 + i * 0.3));
            });
            show(further, beyond ? during(t, 4.4, 4.8) : 0);
          };
        },
      },
    ],
  };
};
