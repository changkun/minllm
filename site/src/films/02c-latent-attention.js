// The side topic 2c as a film: what each kind of attention keeps in the cache
// of stage 4, how latent attention keeps less and still gives every head keys
// of its own, and what that cost and bought in the runs of runs/attention/ and
// the benchmark of runs/bench/. The cache sizes are those the benchmark
// measured, in 32-bit numbers per token and layer.

import { during, set, show } from "./stage.js";

export const sources = {
  meta: "/models/base-q4/meta.json",
  bench: "/runs/bench/attention-mps.json",
  base: "/runs/base/log.jsonl",
  seed: "/runs/attention/base-seed1.jsonl",
  mha: "/runs/attention/mha.jsonl",
  mqa: "/runs/attention/mqa.jsonl",
  mla128: "/runs/attention/mla-128.jsonl",
  mla64: "/runs/attention/mla-64.jsonl",
  mla32: "/runs/attention/mla-32.jsonl",
};

const count = (n) => n.toLocaleString("en");
const loss = (v) => v.toFixed(3);
const rate = (v) => (v < 1000 ? String(Math.round(v)) : v < 10000 ? `${(v / 1000).toFixed(1)}K` : `${Math.round(v / 1000)}K`);

export default ({ meta, bench, base, seed, mha, mqa, mla128, mla64, mla32 }) => {
  const { n_layers: layers, n_heads: heads, dim } = meta.config;
  const head = dim / heads;
  const measured = Object.fromEntries(bench.models.filter((r) => r.absorb).map((r) => [r.model, r]));
  const numbers = (model) => measured[model].cache_bytes_per_token / 4 / layers; // 32-bit numbers, per token and layer
  const end = (log) => log.filter((e) => "val_loss" in e).at(-1);
  const steps = end(base).step;

  // the kinds of attention, as trained and measured, and where each one's label goes on the chart of the loss
  const kinds = [
    { model: "out/mha", log: mha, name: `${heads} key/value heads`, short: "6 heads", kv: heads, label: [16, 8, "start"] },
    { model: "out/base", log: base, name: "2, stage 2's", short: "2 heads", kv: 2, label: [16, -14, "start"] },
    { model: "out/mqa", log: mqa, name: "1 key/value head", short: "1 head", kv: 1, label: [-16, -14, "end"] },
    { model: "out/mla-128", log: mla128, name: "latent 128 + 32", short: "latent 128", latent: true, label: [0, 40, "middle"] },
    { model: "out/mla-64", log: mla64, name: "latent 64 + 32", short: "latent 64", latent: true, label: [16, -14, "start"] },
    { model: "out/mla-32", log: mla32, name: "latent 32 + 16", short: "latent 32", latent: true, label: [16, -14, "start"] },
  ].map((k) => ({ ...k, cache: numbers(k.model), loss: end(k.log).val_loss }));
  const [full, grouped, single, latent] = kinds;
  const speed = (k, batch) => measured[k.model].tokens_per_second[batch];
  const other = end(seed).val_loss;

  return {
    subtitle: "A short latent in the cache, a key per head out of it",
    scenes: [
      {
        seconds: 8,
        caption: `Stage 4's cache keeps, per token and layer, a key and a value\nof ${head} numbers for every key/value head. Fewer heads keep less.`,
        draw(s) {
          const X = 380;
          const UNIT = 700 / full.cache; // pixels per number
          s.rect(X, 112, 34, 24, { fill: "s1", r: 5 });
          s.text(X + 46, 132, "a key", { size: 22, fill: "ink2" });
          s.rect(X + 170, 112, 34, 24, { fill: "s3", r: 5 });
          s.text(X + 216, 132, "a value", { size: 22, fill: "ink2" });
          const rows = [full, grouped, single].map((k, i) => {
            const y = 190 + i * 120;
            const g = s.group();
            g.text(X - 24, y + 34, k.name, { size: 26, anchor: "end", weight: k === grouped ? 700 : 400 });
            g.text(X - 24, y + 66, ["full multi-head", "grouped-query", "multi-query"][i], { size: 22, anchor: "end", fill: "muted" });
            // a key and a value for every head, each of head numbers
            const pieces = Array.from({ length: 2 * k.kv }, (_, j) => s.rect(X + j * head * UNIT, y, head * UNIT - 3, 48, { fill: j % 2 ? "s3" : "s1", r: 6 }));
            const total = s.text(X + k.cache * UNIT + 14, y + 34, `${count(k.cache)} numbers`, { size: 26, weight: 600 });
            return { g, pieces, total, from: 0.4 + i * 1.6 };
          });
          const shared = s.text(640, 566, `With fewer key/value heads, the ${heads} query heads share their keys.`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            for (const r of rows) {
              show(r.g, during(t, r.from, r.from + 0.4));
              r.pieces.forEach((p, j) => show(p, during(t, r.from + 0.2 + j * 0.06, r.from + 0.4 + j * 0.06)));
              show(r.total, during(t, r.from + 0.6 + r.pieces.length * 0.06, r.from + 1 + r.pieces.length * 0.06));
            }
            show(shared, during(t, 5.6, 6.1));
          };
        },
      },
      {
        seconds: 9,
        caption: `Latent attention caches a latent of 128 numbers and a rotary key of 32.\nEvery head's key and value are projected up from the latent.`,
        draw(s) {
          const MID = 330;
          const token = s.box(40, MID - 44, 150, 88, "a token", "its vector x");
          const down = s.box(250, MID - 44, 150, 88, "down", "and normalize");
          const frame = s.group();
          frame.rect(460, 150, 230, 340, { fill: "surface", stroke: "s2", r: 14 });
          frame.text(575, 184, "the cache", { size: 24, weight: 600, anchor: "middle", fill: "s2" });
          const lat = s.box(484, 214, 182, 96, "latent c", "128 numbers", { stroke: "s2" });
          const rot = s.box(484, 360, 182, 96, "rotary key", "32, rotated", { stroke: "s2" });
          const up = s.box(760, 214, 170, 96, "up, per head", "W_k and W_v");
          const keys = s.box(1000, 190, 240, 144, `${heads} keys`, `${heads} values, of ${head}`);
          const shared = s.box(1000, 360, 240, 96, "1 rotary key", `for all ${heads} heads`);
          const wires = [
            s.arrow(192, MID, 246, MID),
            s.arrow(402, MID - 16, 480, 262),
            s.arrow(402, MID + 16, 480, 408),
            s.arrow(668, 262, 756, 262),
            s.arrow(932, 262, 996, 262),
            s.arrow(668, 408, 996, 408),
          ];
          const sum = s.text(575, 540, `${latent.cache} numbers, where ${grouped.short} keep ${grouped.cache}`, { size: 24, anchor: "middle", fill: "ink2" });
          const order = [token, wires[0], down, wires[1], wires[2], frame, lat, rot, wires[3], up, wires[4], keys, wires[5], shared];
          return (t) => {
            order.forEach((thing, i) => show(thing, during(t, 0.3 + i * 0.32, 0.6 + i * 0.32)));
            show(sum, during(t, 5.6, 6.1));
          };
        },
      },
      {
        seconds: 9,
        caption: "A query meets a key in a dot product, so W_k can move to the query:\ndecoding projects one query down, not every cached latent up.",
        draw(s) {
          const CACHED = 7;
          const sides = [
            { y: 150, formula: "q · (W_k c)", note: "every cached latent up, at every step" },
            { y: 390, formula: "(W_kᵀ q) · c", note: "the query down, once" },
          ];
          for (const side of sides) {
            side.g = s.group();
            side.g.text(60, side.y + 50, side.formula, { size: 34, weight: 600, mono: true });
            side.g.text(60, side.y + 92, side.note, { size: 22, fill: "muted" });
            side.latents = Array.from({ length: CACHED }, (_, j) => side.g.rect(560 + j * 90, side.y + 110, 70, 30, { fill: "s2", r: 6 }));
            side.g.text(560, side.y + 172, "the cached latents", { size: 20, fill: "muted" });
          }
          // above: each latent becomes a key; below: one query becomes a latent-sized vector
          const keys = Array.from({ length: CACHED }, (_, j) => s.rect(560 + j * 90, sides[0].y, 70, 60, { fill: "s1", r: 6 }));
          const ups = Array.from({ length: CACHED }, (_, j) => s.arrow(595 + j * 90, sides[0].y + 106, 595 + j * 90, sides[0].y + 66));
          const query = s.rect(420, sides[1].y + 10, 70, 60, { fill: "s1", r: 6 });
          const folded = s.rect(420, sides[1].y + 110, 70, 30, { fill: "s1", r: 6 });
          const fold = s.arrow(455, sides[1].y + 74, 455, sides[1].y + 104);
          const meets = Array.from({ length: CACHED }, (_, j) => s.line(492, sides[1].y + 125, 560 + j * 90 + 35, sides[1].y + 110, { stroke: "grid", width: 2 }));
          return (t) => {
            show(sides[0].g, during(t, 0.3, 0.7));
            ups.forEach((a, j) => show(a, during(t, 1 + j * 0.25, 1.3 + j * 0.25)));
            keys.forEach((k, j) => show(k, during(t, 1.1 + j * 0.25, 1.4 + j * 0.25)));
            show(sides[1].g, during(t, 3.6, 4));
            show(query, during(t, 4.2, 4.5));
            show(fold, during(t, 4.7, 5));
            show(folded, during(t, 4.9, 5.2));
            meets.forEach((m, j) => show(m, during(t, 5.6 + j * 0.08, 5.9 + j * 0.08)));
          };
        },
      },
      {
        seconds: 9,
        caption: `After ${count(steps)} steps the latent of 128 ends at ${loss(latent.loss)}, as one key/value head\n(${loss(single.loss)}) and two (${loss(grouped.loss)}, ${loss(other)} with another seed). ${heads} heads end at ${loss(full.loss)}.`,
        draw(s) {
          const values = [...kinds.map((k) => k.loss), other];
          const domain = [Math.floor(Math.min(...values) * 200) / 200 - 0.005, Math.ceil(Math.max(...values) * 200) / 200 + 0.005];
          const ticks = [];
          for (let v = Math.ceil(domain[0] * 100) / 100; v <= domain[1] + 1e-9; v += 0.01) ticks.push(v);
          const { px, py } = s.axes(170, 130, 860, 360, {
            x: { domain: [32, 1024], log: true, ticks: [32, 64, 128, 256, 512], title: "numbers cached per token and layer, log scale" },
            y: { domain, ticks, format: (v) => v.toFixed(2), title: "validation loss" },
          });
          const key = s.group();
          key.circle(800, 112, 9, { fill: "s1" });
          key.text(818, 120, "key/value heads", { size: 22, fill: "ink2" });
          key.circle(1010, 112, 9, { fill: "s2" });
          key.text(1028, 120, "latent", { size: 22, fill: "ink2" });
          const points = kinds.map((k) => {
            const g = s.group();
            g.circle(px(k.cache), py(k.loss), 10, { fill: k.latent ? "s2" : "s1" });
            const [dx, dy, anchor] = k.label;
            g.text(px(k.cache) + dx, py(k.loss) + dy, `${k.short}, ${loss(k.loss)}`, { size: 22, anchor });
            return g;
          });
          const spread = s.group();
          spread.line(px(grouped.cache), py(grouped.loss), px(grouped.cache), py(other), { stroke: "s1", width: 3 });
          spread.circle(px(grouped.cache), py(other), 7, { fill: "surface", stroke: "s1" });
          spread.text(px(grouped.cache) + 16, py(other) + 30, `another seed, ${loss(other)}`, { size: 20, fill: "muted" });
          const order = [0, 1, 2, 3, 4, 5];
          return (t) => {
            show(key, during(t, 0.2, 0.6));
            order.forEach((i, j) => show(points[i], during(t, 0.6 + j * 0.5, 1 + j * 0.5)));
            show(spread, during(t, 4.2, 4.7));
          };
        },
      },
      {
        seconds: 9,
        caption: `Alone, the latent of 128 gets ${count(Math.round(speed(latent, "1")))} tokens a second, a third of stage 2's ${count(Math.round(speed(grouped, "1")))}.\nWith 1,024 sequences at once it gets ${count(Math.round(speed(latent, "1024") / 100) * 100)}, ${(speed(latent, "1024") / speed(grouped, "1024")).toFixed(1)} times as many.`,
        draw(s) {
          const rows = bench.models;
          const sections = [
            { batch: "1", title: "one sequence", x: 60 },
            { batch: "1024", title: "1,024 sequences at once", x: 680 },
          ];
          const color = (r) => (!r.absorb ? "muted" : r.model.includes("mla") ? "s2" : "s1");
          const label = (r) => kinds.find((k) => k.model === r.model).short + (r.absorb ? "" : ", projected up");
          const made = sections.flatMap((section) => {
            const peak = Math.max(...rows.map((r) => r.tokens_per_second[section.batch]));
            s.text(section.x, 132, section.title, { size: 26, weight: 600 });
            s.text(section.x, 162, "tokens per second", { size: 20, fill: "muted" });
            return rows.map((r, i) => {
              const y = 190 + i * 52;
              const left = section.x + 230;
              const w = (300 * r.tokens_per_second[section.batch]) / peak;
              const g = s.group();
              g.text(left - 14, y + 27, label(r), { size: 22, anchor: "end", fill: "ink2" });
              const bar = g.rect(left, y + 4, 0, 30, { fill: color(r), r: 6 });
              const value = g.text(left + w + 10, y + 28, rate(r.tokens_per_second[section.batch]), { size: 22, weight: 600 });
              return { g, bar, value, w, i };
            });
          });
          return (t) => {
            made.forEach(({ g, bar, value, w, i }) => {
              show(g, during(t, 0.3 + i * 0.15, 0.6 + i * 0.15));
              set(bar, { width: w * during(t, 0.5 + i * 0.15, 1.3 + i * 0.15) });
              show(value, during(t, 1.2 + i * 0.15, 1.5 + i * 0.15));
            });
          };
        },
      },
    ],
  };
};
