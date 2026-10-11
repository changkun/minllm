// Stage 4 as a film: generation is a loop, temperature and top-k reshape what
// the model predicts before a token is drawn, and a cache of keys and values
// saves the work of reading the text again. The predictions are the trained
// model's own, and the times are those of scripts/bench_inference.py.

import { Tokenizer } from "../labs/bpe.js";
import { distribution } from "../labs/sampling.js";
import { clamp, during, mix, place, set, show, spelled, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  next: "/runs/base/next-token.json",
  gpu: "/runs/bench/inference-mps.json",
  cpu: "/runs/bench/inference-cpu.json",
};

const percent = (p) => `${(100 * p).toFixed(p >= 0.0005 && p < 0.0995 ? 1 : 0)}%`;
const whole = (n) => Math.round(n).toLocaleString("en");
const plainly = { temperature: 1, topK: 0, topP: 1 };
// The ids of the n likeliest tokens of a distribution.
const likeliest = (p, n) => Array.from(p, (v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, n).map(([, i]) => i);

// The frame of a chart, as Stage.axes draws it, with labels of the size the films use.
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
  s.text(x + w / 2, y + h + 62, across.title, { size: 22, anchor: "middle", fill: "muted" });
  s.text(x, y - 16, up.title, { size: 22, fill: "muted" });
  return { px, py };
}

// A chip as the one that is talked about.
function mark(chip) {
  chip.ground.setAttribute("class", "f-s2 s-s2");
  chip.label.setAttribute("class", "film-mono f-surface");
}

// Bars standing on a line: one for each of the named tokens and, apart from them, one for all the others.
function columns(s, names, base) {
  s.line(140, base, 1140, base, { stroke: "axis" });
  return names.map((name, i) => {
    const rest = i === names.length - 1;
    const x = 160 + i * 104 + (rest ? 40 : 0);
    const bar = s.rect(x, base, 64, 0, { fill: rest ? "axis" : "s1", r: 4 });
    const value = s.text(x + 32, base, "", { size: 22, anchor: "middle", fill: "ink2" });
    return { x, bar, value, name: s.text(x + 32, base + 32, name, { size: 22, anchor: "middle", mono: !rest, fill: rest ? "muted" : "ink" }) };
  });
}

// A standing bar at a share of everything, where full is the height of a bar that holds everything.
function stand(column, share, base, full) {
  set(column.bar, { y: base - full * share, height: full * share });
  set(column.value, { y: base - full * share - 10 });
  column.value.textContent = percent(share);
}

export default ({ vocabulary, next, gpu, cpu }) => {
  const tok = new Tokenizer(vocabulary);
  const spell = (id) => spelled(tok.text(id));
  const pieces = (string) => tok.encode(string).map(spell);
  const [first, second] = next.prompts;
  const size = next.vocab_size;

  // the prediction that the knobs are turned on: its likeliest tokens, and all the others as one
  const logits = second.logits;
  const plain = distribution(logits, plainly);
  const named = likeliest(plain, 8);
  const shares = (p) => [...named.map((id) => p[id]), Math.max(0, 1 - named.reduce((sum, id) => sum + p[id], 0))];
  const names = [...named.map((id) => tok.text(id).trim()), "the other"];
  const after = second.prompt.split(" ").slice(-3).join(" ");

  // the longest text that was timed, with and without the cache
  const longest = (bench, batch) => bench.length.filter((row) => row.batch === batch).reduce((a, b) => (b.new_tokens > a.new_tokens ? b : a));
  const batches = [...new Set(gpu.length.map((row) => row.batch))].sort((a, b) => a - b);
  const timed = [
    ["GPU", gpu],
    ["CPU", cpu],
  ].flatMap(([device, bench]) => batches.map((batch) => ({ name: `${device}, ${batch} sequence${batch > 1 ? "s" : ""}`, row: longest(bench, batch) })));
  for (const run of timed) run.gain = run.row.uncached_seconds / run.row.cached_seconds;
  const gains = timed.filter((run) => run.gain > 1.5).map((run) => Math.round(run.gain));
  const one = gpu.batch.find((row) => row.batch === 1);
  const many = gpu.batch.find((row) => row.batch === 64);
  const most = gpu.batch.at(-1);

  return {
    subtitle: "From a prediction to a text",
    scenes: [
      {
        seconds: 8,
        caption: "Generation is a loop: ask the model for a prediction,\npick a token, append it, and ask again.",
        draw(s) {
          const p = distribution(first.logits, plainly);
          const top = likeliest(p, 6);
          const row = s.chips(640, 112, [...pieces(first.prompt), spell(top[0])], { center: true, size: 28, gap: 6, pad: 10 });
          const drawn = row.at(-1);
          const home = { x: drawn.x, y: drawn.y };
          mark(drawn);
          const picked = s.chip(580, 306, spell(top[0]), { size: 28, pad: 10, ink: "muted" }); // where it was drawn, once it has left
          const enter = s.arrow(250, 170, 250, 214);
          const goes = s.text(270, 200, "", { size: 22, fill: "ink2" });
          const bars = top.map((id, i) => {
            const g = s.group();
            const y = 228 + i * 38;
            g.text(220, y + 22, tok.text(id).trim(), { size: 22, anchor: "end", mono: true });
            return { g, w: 600 * p[id], share: p[id], bar: g.rect(234, y, 0, 28, { fill: "s1", r: 4 }), value: g.text(234, y + 22, "", { size: 22, fill: "ink2" }) };
          });
          const steps = [
            [100, "1  Predict", `a probability for each of ${whole(size)} tokens`],
            [580, "2  Pick", "one token is drawn"],
            [900, "3  Append", "it joins the text"],
          ].map(([x, title, sub]) => {
            const g = s.group();
            g.text(x, 500, title, { size: 26, weight: 700 });
            g.text(x, 534, sub, { size: 22, fill: "ink2" });
            return g;
          });
          const pick = s.arrow(440, 330, 564, 330);
          const up = s.arrow(580 + drawn.w + 16, 318, home.x + drawn.w / 2 - 30, home.y + drawn.h + 14);
          s.g.append(drawn.g); // above the arrow it moves along
          return (t) => {
            row.slice(0, -1).forEach((c, i) => show(c, during(t, 0.3 + i * 0.06, 0.6 + i * 0.06)));
            show(enter, during(t, 1, 1.4));
            const again = t >= 5.7;
            goes.textContent = again ? "the longer text goes in again" : "the text goes in";
            goes.setAttribute("class", again ? "f-s2" : "f-ink2");
            show(goes, again ? during(t, 5.7, 6.1) : during(t, 1, 1.4) * (1 - during(t, 5.4, 5.7)));
            show(steps[0], during(t, 1.2, 1.6));
            bars.forEach((b, i) => {
              const u = during(t, 1.5 + i * 0.1, 2.6 + i * 0.1);
              show(b.g, during(t, 1.3 + i * 0.1, 1.7 + i * 0.1));
              set(b.bar, { width: b.w * u });
              set(b.value, { x: 234 + b.w * u + 10 });
              b.value.textContent = percent(b.share * u);
            });
            bars[0].bar.setAttribute("class", t >= 3.3 ? "f-s2" : "f-s1");
            show(pick, during(t, 3.1, 3.5));
            show(steps[1], during(t, 3.1, 3.5));
            show(drawn, during(t, 3.3, 3.7));
            show(picked, during(t, 4.4, 4.8));
            show(steps[2], during(t, 4.1, 4.5));
            show(up, during(t, 4.1, 4.5));
            const move = during(t, 4.4, 5.4);
            place(drawn, mix(580, home.x, move), mix(306, home.y, move));
          };
        },
      },
      {
        seconds: 8,
        caption: "Temperature divides the scores before the softmax. Below 1\nthe likely tokens gain, above 1 the prediction flattens.",
        draw(s) {
          const BASE = 498;
          const FULL = 250;
          s.text(150, 140, `The model's prediction after “${after}”`, { size: 24, fill: "ink2" });
          s.text(150, 174, "softmax(logits / temperature)", { size: 22, fill: "muted", mono: true });
          const dial = s.text(1130, 150, "", { size: 40, weight: 700, anchor: "end", fill: "s2" });
          const among = s.text(1130, 188, "", { size: 22, anchor: "end", fill: "ink2" });
          const bars = columns(s, names, BASE);
          s.text(bars.at(-1).x + 32, BASE + 60, `${whole(size - named.length)} tokens`, { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            const T = mix(mix(mix(1, 0, during(t, 1.5, 2.7)), 1.5, during(t, 3.3, 4.6)), 1, during(t, 5.2, 6));
            const p = distribution(logits, { ...plainly, temperature: T < 0.005 ? 0 : T });
            const up = during(t, 0.3, 1.1);
            const entropy = -p.reduce((sum, v) => sum + (v > 0 ? v * Math.log(v) : 0), 0);
            dial.textContent = `temperature ${T.toFixed(2)}`;
            among.textContent = `as uncertain as a choice among ${Math.exp(entropy).toFixed(1)} tokens`;
            show(among, during(t, 1.1, 1.5));
            shares(p).forEach((share, i) => stand(bars[i], share * up, BASE, FULL));
          };
        },
      },
      {
        seconds: 7,
        caption: "Top-k keeps the k likeliest tokens and discards the rest:\nthe long tail, where one bad draw can derail a text.",
        draw(s) {
          const BASE = 498;
          const FULL = 440;
          const K = 5;
          const kept = distribution(logits, { ...plainly, topK: K });
          const before = shares(plain);
          const cut = shares(kept);
          s.text(150, 140, `The same prediction after “${after}”, at temperature 1`, { size: 24, fill: "ink2" });
          const dial = s.text(1130, 190, "", { size: 40, weight: 700, anchor: "end", fill: "s2" });
          const left = s.text(1130, 228, "", { size: 22, anchor: "end", fill: "ink2" });
          const bars = columns(s, names, BASE);
          s.text(bars.at(-1).x + 32, BASE + 60, `${whole(size - named.length)} tokens`, { size: 22, anchor: "middle", fill: "muted" });
          const edge = bars[K].x - 20;
          const fence = s.group();
          set(fence.line(edge, 262, edge, BASE + 66, { stroke: "s2", width: 2 }), { "stroke-dasharray": "3 9" });
          fence.text(edge - 16, 280, `the ${K} likeliest are kept`, { size: 22, anchor: "end", fill: "s2" });
          fence.text(edge + 16, 280, "the tail is discarded", { size: 22, fill: "s2" });
          return (t) => {
            const up = during(t, 0.3, 1.1);
            const u = during(t, 3, 4.2);
            const on = t >= 3;
            dial.textContent = on ? `top-k ${K}` : "top-k off";
            left.textContent = `${whole(on ? kept.filter((v) => v > 0).length : size)} tokens can be drawn`;
            show(fence, during(t, 1.8, 2.3));
            bars.forEach((b, i) => {
              stand(b, mix(before[i], cut[i], u) * up, BASE, FULL);
              const gone = cut[i] === 0 && u > 0.98;
              if (gone) b.value.textContent = "cut";
              b.value.setAttribute("class", gone ? "f-muted" : "f-ink2");
              if (i < bars.length - 1) b.name.setAttribute("class", `film-mono ${gone ? "f-muted" : "f-ink"}`);
            });
          };
        },
      },
      {
        seconds: 8,
        caption: "Every layer keeps the keys and values of tokens it has seen.\nA new token then costs one row of attention, not a triangle.",
        draw(s) {
          // one cell is one query reading one key, in one head of one layer
          const tokens = pieces(first.prompt).slice(0, 8);
          const N = tokens.length;
          const CELL = 40;
          const TOP = 166;
          const EACH = 0.65;
          const sides = [
            [170, "Without a cache", false],
            [750, "With a cache", true],
          ].map(([x, title, cached]) => {
            s.text(x, 146, title, { size: 26, weight: 600 });
            const rows = tokens.map((name, r) => {
              const g = s.group();
              g.text(x - 14, TOP + r * CELL + 27, name, { size: 22, anchor: "end", mono: true, fill: "ink2" });
              return { g, cells: Array.from({ length: r + 1 }, (_, c) => g.rect(x + c * CELL, TOP + r * CELL, CELL - 4, CELL - 4, { fill: "s2", r: 5 })) };
            });
            // the upper right of a triangle is free
            const count = s.text(x + 5 * CELL, 232, "", { size: 64, weight: 700, fill: "s2" });
            s.text(x + 5 * CELL, 266, "scores for this token", { size: 22, fill: "ink2" });
            s.text(x, 528, cached ? "blue: done before, keys and values kept" : "every row is computed again", { size: 22, fill: cached ? "s1" : "s2" });
            return { rows, count, cached, all: s.text(x, 560, "", { size: 22, fill: "muted" }) };
          });
          return (t) => {
            const step = clamp(Math.floor((t - 0.6) / EACH), 0, N - 1); // the row of the newest token
            const since = t - 0.6 - step * EACH;
            const n = step + 1;
            for (const side of sides) {
              side.rows.forEach(({ g, cells }, r) => {
                show(g, r < step ? 1 : r === step ? during(since, 0, 0.25) : 0);
                const now = !side.cached || r === step;
                // without a cache, the rows light up one after another at every step
                const lit = side.cached ? 1 : mix(0.4, 1, during(since, r * 0.04, r * 0.04 + 0.2));
                for (const cell of cells) {
                  cell.setAttribute("class", now ? "f-s2" : "f-s1");
                  cell.setAttribute("opacity", (now ? lit : 0.35).toFixed(2));
                }
              });
              const each = Array.from({ length: n }, (_, i) => (side.cached ? i + 1 : ((i + 1) * (i + 2)) / 2));
              side.count.textContent = String(each.at(-1));
              side.all.textContent = `${each.reduce((a, b) => a + b, 0)} scores for these ${n} tokens`;
            }
          };
        },
      },
      {
        seconds: 8,
        caption: `For one sequence on the GPU the cache changes nothing.\nOnce the arithmetic matters it is ${Math.min(...gains)} to ${Math.max(...gains)} times faster.`,
        draw(s) {
          const seconds = (v) => `${v.toFixed(v < 10 ? 2 : 1)} s`;
          const peak = Math.max(...timed.map((run) => run.gain));
          s.text(100, 144, `Seconds to generate ${timed[0].row.new_tokens} tokens with the 14M model`, { size: 26, weight: 600 });
          s.text(470, 200, "without a cache", { size: 22, anchor: "end", fill: "muted" });
          s.text(660, 200, "with a cache", { size: 22, anchor: "end", fill: "muted" });
          s.text(720, 200, "how much faster", { size: 22, fill: "muted" });
          const rows = timed.map((run, i) => {
            const g = s.group();
            const y = 228 + i * 78;
            const same = run.gain < 1.5;
            g.text(100, y + 31, run.name, { size: 24, weight: same ? 700 : 400 });
            g.text(470, y + 31, seconds(run.row.uncached_seconds), { size: 24, anchor: "end", mono: true });
            g.arrow(494, y + 22, 546, y + 22);
            g.text(660, y + 31, seconds(run.row.cached_seconds), { size: 24, anchor: "end", mono: true });
            return {
              g,
              w: (330 * run.gain) / peak,
              bar: s.rect(720, y, 0, 44, { fill: same ? "s2" : "s1", r: 5 }),
              gain: s.text(720, y + 31, same ? "no gain" : `${Math.round(run.gain)}× faster`, { size: 24, weight: 700, fill: same ? "s2" : "ink" }),
            };
          });
          return (t) => {
            rows.forEach((row, i) => {
              const from = 0.6 + i * 1.2;
              const u = during(t, from + 0.4, from + 1.2);
              show(row.g, during(t, from, from + 0.4));
              set(row.bar, { width: row.w * u });
              set(row.gain, { x: 720 + row.w * u + 14 });
              show(row.gain, during(t, from + 0.9, from + 1.3));
            });
          };
        },
      },
      {
        seconds: 7.5,
        caption: `A second sequence in the batch is almost free. From ${one.batch} to ${many.batch}\nsequences, throughput grows ${Math.floor(many.tokens_per_second / one.tokens_per_second)} times.`,
        draw(s) {
          const sizes = gpu.batch.map((row) => row.batch);
          const across = { domain: [sizes[0], sizes.at(-1)], log: true, ticks: sizes, format: whole, title: "sequences generated at once, on the GPU" };
          const up = { domain: [300, 30000], log: true, ticks: [300, 1000, 3000, 10000, 30000], format: whole, title: "tokens per second" };
          const { px, py } = frame(s, 160, 150, 600, 346, across, up);
          const points = gpu.batch.map((row) => [px(row.batch), py(row.tokens_per_second)]);
          const curve = s.path(points, { stroke: "s1", width: 4 });
          const dots = gpu.batch.map((row, i) => s.circle(points[i][0], points[i][1], row === many ? 10 : 7, { fill: row === many ? "s2" : "s1", stroke: "surface" }));
          const notes = [
            [one, 200, ""],
            [many, 330, `${Math.floor(many.tokens_per_second / one.tokens_per_second)} times as many`],
            [most, 490, ""],
          ].map(([row, y, more]) => {
            const g = s.group();
            g.text(830, y, `${whole(row.batch)} sequence${row.batch > 1 ? "s" : ""}`, { size: 22, fill: "muted" });
            const value = g.text(830, y + 38, "", { size: 28, weight: 700 });
            if (more) g.text(830, y + 74, more, { size: 24, weight: 700, fill: "s2" });
            return { g, value, row };
          });
          // how far along the line each point is
          const along = points.map((_, i) => points.slice(1, i + 1).reduce((sum, [x, y], j) => sum + Math.hypot(x - points[j][0], y - points[j][1]), 0) / curve.total);
          const reach = (i) => 0.6 + 3.4 * along[i]; // the second at which the line gets to a point
          return (t) => {
            trace(curve, clamp((t - 0.6) / 3.4));
            dots.forEach((dot, i) => show(dot, during(t, reach(i) - 0.1, reach(i) + 0.2)));
            for (const note of notes) {
              const from = reach(sizes.indexOf(note.row.batch));
              show(note.g, during(t, from, from + 0.4));
              note.value.textContent = `${whole(note.row.tokens_per_second * during(t, from, from + 0.9))} tokens per second`;
            }
          };
        },
      },
    ],
  };
};
