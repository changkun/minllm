// Side topic 10a as a film: a weight stored as one of 16 codes, with the scale
// and the offset of its group; eight codes packed into a word; and what a bit
// width costs in loss, on the task, and in speed. The weights are a real row
// of the 0.5B model, rounded here by the rule of minllm/quantize.py, and every
// number is read from the sweep, the evaluations and the benchmark.

import { during, mix, place, set, show, trace } from "./stage.js";

export const sources = {
  row: "/runs/quant/row.json",
  small: "/runs/quant/sweep-base.json",
  large: "/runs/quant/sweep-qwen.json",
  bench: "/runs/bench/quant-mps.json",
  site: "/models/base-q4/meta.json",
  e16: "/runs/eval/gsm8k-qwen.json",
  e8: "/runs/eval/gsm8k-qwen-q8.json",
  e6: "/runs/eval/gsm8k-qwen-6bit.json",
  e5: "/runs/eval/gsm8k-qwen-5bit.json",
  e4: "/runs/eval/gsm8k-qwen-q4.json",
};

const BITS = 4;
const VALUES = 2 ** BITS;
const minus = (string) => string.replace("-", "−");
// a change of the loss, as the note's tables write it
const signed = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(Math.abs(v) < 10 ? 3 : 1);

// quantize.quantize for one group: the offset is its smallest weight, the scale a 15th of its range
function fit(weights) {
  const offset = Math.min(...weights);
  const scale = (Math.max(...weights) - offset) / (VALUES - 1);
  return { offset, scale, codes: weights.map((w) => Math.round((w - offset) / scale)) };
}

export default ({ row, small, large, bench, site, e16, e8, e6, e5, e4 }) => {
  const size = row.group; // weights that share a scale
  const first = row.weights.slice(0, size);
  const group = fit(first);
  const whole = fit(row.weights);
  const setting = (sweep, bits) => sweep.settings.find((r) => r.bits === bits && r.group === size);
  const added = (sweep, bits) => setting(sweep, bits).loss - sweep.loss;
  const megabytes = (bits) => (large.parameters * bits) / 8e6;
  const floats = bench.cases.find((c) => c.where === "MLX" && c.bits === 16);

  return {
    subtitle: "A weight in a few bits instead of sixteen",
    scenes: [
      {
        seconds: 6,
        caption: `The 0.5B model is ${Math.round(large.parameters / 1e6)} million weights at 16 bits each: ${Math.round(floats.file_mb)} MB.\nQuantization rounds each weight to one of a few values.`,
        draw(s) {
          const cell = 44;
          const left = 640 - (16 * cell + 15 * 6) / 2;
          s.text(640, 150, "one weight of the 0.5B model, as its file stores it", { size: 24, anchor: "middle", fill: "ink2" });
          const cells = Array.from({ length: 16 }, (_, i) => s.rect(left + i * (cell + 6), 174, cell, cell, { fill: "s1", r: 6 }));
          const before = s.text(640, 262, "16 bits", { size: 26, weight: 600, anchor: "middle" });
          const after = s.text(left, 262, `${BITS} bits: one of ${VALUES} values`, { size: 26, weight: 600, fill: "s2" });
          const count = s.text(640, 372, `${large.parameters.toLocaleString("en")} weights of 16 bits`, { size: 26, anchor: "middle" });
          const bar = s.rect(left, 396, 0, 40, { fill: "s1", r: 6 });
          const file = s.text(640, 492, "", { size: 40, weight: 700, anchor: "middle" });
          return (t) => {
            cells.forEach((c, i) => {
              const kept = i < BITS;
              const u = during(t, 3.4, 4.2);
              c.setAttribute("class", kept && t > 3.8 ? "f-s2" : "f-s1");
              show(c, during(t, 0.3 + i * 0.03, 0.6 + i * 0.03) * (kept ? 1 : mix(1, 0.14, u)));
            });
            show(before, Math.min(during(t, 0.8, 1.2), 1 - during(t, 3.4, 3.8)));
            show(after, during(t, 3.8, 4.3));
            const u = during(t, 1.6, 2.8);
            show(count, during(t, 1.4, 1.8));
            set(bar, { width: 794 * u });
            file.textContent = `${Math.round(floats.file_mb * u)} MB`;
            show(file, during(t, 1.6, 2.0));
          };
        },
      },
      {
        seconds: 8.5,
        caption: `A group of ${size} weights has a scale and an offset. Its ${VALUES} values\nspan the group, and every weight is rounded to the nearest.`,
        draw(s) {
          const x = (v) => 140 + (1000 * (v - group.offset)) / ((VALUES - 1) * group.scale);
          const at = (code) => x(group.offset + code * group.scale);
          const title = s.text(640, 128, `one group: ${size} weights in a row of the 0.5B model`, { size: 22, anchor: "middle", fill: "muted" });
          const values = Array.from({ length: VALUES }, (_, k) => ({ line: s.line(at(k), 146, at(k), 412, { stroke: "s2", width: 1.5 }), name: s.text(at(k), 442, String(k), { size: 22, anchor: "middle", fill: "s2", mono: true }) }));
          // every weight: a tick where it lies, then a brick on the value it is rounded to
          const stacked = new Array(VALUES).fill(0);
          const marks = first.map((w, i) => ({ mark: s.rect(0, 0, 3, 44, { fill: "s1", r: 1.5 }), from: x(w), to: at(group.codes[i]), level: stacked[group.codes[i]]++ }));
          const notes = s.group();
          notes.line(at(0), 454, at(0), 468, { stroke: "muted" });
          notes.text(at(0) - 20, 496, `offset ${minus(group.offset.toFixed(4))}, the smallest weight`, { size: 22, fill: "ink2" });
          notes.line(at(VALUES - 2), 462, at(VALUES - 1), 462, { stroke: "muted" });
          notes.line(at(VALUES - 2), 454, at(VALUES - 2), 470, { stroke: "muted" });
          notes.line(at(VALUES - 1), 454, at(VALUES - 1), 470, { stroke: "muted" });
          notes.text(at(VALUES - 1) + 20, 496, `scale ${group.scale.toFixed(4)}, the step between values`, { size: 22, anchor: "end", fill: "ink2" });
          const rule = s.text(640, 552, "weight = code × scale + offset", { size: 30, anchor: "middle", mono: true });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            values.forEach((v, k) => {
              show(v.line, during(t, 1.8 + k * 0.07, 2.1 + k * 0.07));
              show(v.name, during(t, 1.9 + k * 0.07, 2.2 + k * 0.07));
            });
            marks.forEach((m, i) => {
              const u = during(t, 4.0 + i * 0.02, 4.9 + i * 0.02);
              set(m.mark, { x: mix(m.from - 1.5, m.to - 12, u), y: mix(152, 394 - m.level * 10, u), width: mix(3, 24, u), height: mix(44, 7, u) });
              show(m.mark, during(t, 0.4 + i * 0.012, 0.7 + i * 0.012));
            });
            show(notes, during(t, 3.1, 3.6));
            show(rule, during(t, 6.2, 6.8));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "If a whole row shared one scale, its few extreme weights would\nstretch the steps. A small group rarely contains them.",
        draw(s) {
          const low = whole.offset;
          const high = low + (VALUES - 1) * whole.scale;
          const x = (v) => 140 + (1000 * (v - low)) / (high - low);
          const title = s.text(140, 124, `The whole row of ${row.weights.length} weights shares one scale`, { size: 24, weight: 600 });
          const step = s.text(1140, 124, `${VALUES} values, ${whole.scale.toFixed(4)} apart`, { size: 24, anchor: "end", fill: "s2", weight: 600 });
          // where the weights of the row lie
          const bins = 100;
          const counts = new Array(bins).fill(0);
          for (const w of row.weights) counts[Math.min(bins - 1, Math.floor(((w - low) / (high - low)) * bins))]++;
          const peak = Math.max(...counts);
          const bars = counts.map((c, b) => ({ h: (104 * c) / peak, bar: s.rect(141 + b * 10, 250, 8, 0, { fill: "s1", r: 2 }) }));
          s.line(140, 250, 1140, 250);
          const rowValues = Array.from({ length: VALUES }, (_, k) => s.line(x(low + k * whole.scale), 136, x(low + k * whole.scale), 258, { stroke: "s2", width: 1.5 }));
          const title2 = s.text(140, 318, `A group of ${size} has a scale of its own`, { size: 24, weight: 600 });
          const step2 = s.text(1140, 318, `${VALUES} values, ${group.scale.toFixed(4)} apart`, { size: 24, anchor: "end", fill: "s2", weight: 600 });
          const ticks = first.map((w) => s.rect(x(w) - 1.5, 338, 3, 62, { fill: "s1", r: 1.5 }));
          const floor = s.line(140, 404, 1140, 404);
          const groupValues = Array.from({ length: VALUES }, () => s.line(0, 330, 0, 412, { stroke: "s2", width: 1.5 }));
          const shrink = s.text(640, 478, `In groups of ${size}, the steps of this row shrink by ${Math.round(100 * (1 - 1 / row.ratio))}% on average.`, { size: 26, anchor: "middle" });
          const price = s.text(640, 540, `The price is a scale and an offset per group: ${BITS} + 32 / ${size} = ${BITS + 32 / size} bits per weight.`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            bars.forEach(({ h, bar }, b) => {
              const u = during(t, 0.4 + b * 0.006, 1.0 + b * 0.006);
              set(bar, { y: 250 - h * u, height: h * u });
            });
            rowValues.forEach((line, k) => show(line, during(t, 1.3 + k * 0.04, 1.5 + k * 0.04)));
            show(step, during(t, 1.9, 2.3));
            show(title2, during(t, 2.5, 2.9));
            show(floor, during(t, 2.5, 2.9));
            ticks.forEach((tick, i) => show(tick, during(t, 2.6 + i * 0.006, 2.9 + i * 0.006)));
            // the same 16 values, drawn in from the row's range to the group's
            const u = during(t, 3.4, 4.5);
            groupValues.forEach((line, k) => {
              const at = mix(x(low + k * whole.scale), x(group.offset + k * group.scale), u);
              set(line, { x1: at, x2: at });
              show(line, during(t, 3.0, 3.3));
            });
            show(step2, during(t, 4.5, 4.9));
            show(shrink, during(t, 5.0, 5.4));
            show(price, during(t, 5.5, 5.9));
          };
        },
      },
      {
        seconds: 8.5,
        caption: "Eight 4-bit codes are packed into one 32-bit word for storage,\nthe first code in the lowest bits. MLX reads this layout.",
        draw(s) {
          const codes = group.codes.slice(0, 32 / BITS);
          const column = (i) => 276 + i * 104;
          const word = codes.reduce((sum, code, i) => sum + code * VALUES ** i, 0);
          const title = s.text(640, 140, `the first ${codes.length} codes of the group`, { size: 22, anchor: "middle", fill: "muted" });
          const chips = codes.map((code, i) => {
            const chip = s.chip(0, 160, String(code), { size: 34, fill: "s2", stroke: null, ink: "surface" });
            place(chip, column(i) - chip.w / 2, 160);
            return chip;
          });
          const slots = codes.map((_, i) => s.rect(column(i) - 48, 360, 96, 60, { fill: "tint", stroke: "axis", r: 6 }));
          const bits = codes.map((code, i) => s.text(column(i), 268, code.toString(2).padStart(BITS, "0"), { size: 28, anchor: "middle", mono: true }));
          const ends = [s.text(column(0) - 48, 454, "bit 31", { size: 22, fill: "muted" }), s.text(column(codes.length - 1) + 48, 454, "bit 0", { size: 22, anchor: "end", fill: "muted" })];
          const value = s.text(640, 522, `one 32-bit word: 0x${word.toString(16).toUpperCase().padStart(8, "0")}`, { size: 28, anchor: "middle", mono: true });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            chips.forEach((chip, i) => show(chip, during(t, 0.4 + i * 0.1, 0.7 + i * 0.1)));
            slots.forEach((slot, i) => show(slot, during(t, 2.4 + i * 0.04, 2.8 + i * 0.04)));
            ends.forEach((end) => show(end, during(t, 2.6, 3.0)));
            bits.forEach((node, i) => {
              // code i takes bits 4i to 4i + 3, so the first one ends up on the right:
              // down to a lane of its own, across, and down into its place
              const a = 3.1 + i * 0.38;
              const across = during(t, a + 0.12, a + 0.4);
              set(node, { x: mix(column(i), column(codes.length - 1 - i), across), y: mix(mix(268, 326, during(t, a, a + 0.14)), 400, during(t, a + 0.38, a + 0.52)) });
              show(node, during(t, 1.3 + i * 0.1, 1.6 + i * 0.1));
            });
            show(value, during(t, 6.4, 6.9));
          };
        },
      },
      {
        seconds: 8,
        caption: `Eight bits are free. Below 4 bits the model falls apart:\nat 2 bits the loss of the 0.5B model rises by ${added(large, 2).toFixed(1)}.`,
        draw(s) {
          const widths = [8, 6, 5, 4, 3, 2];
          const FLOOR = 0.001; // what is added below this, or taken away, is drawn on the bottom line
          const TOP = 20;
          const px = (i) => 190 + (570 * (i + 0.5)) / widths.length;
          const py = (v) => 490 - (340 * Math.log(Math.max(v, FLOOR) / FLOOR)) / Math.log(TOP / FLOOR);
          s.text(190, 128, "loss added by rounding", { size: 22, fill: "muted" });
          for (const tick of [0.001, 0.01, 0.1, 1, 10]) {
            s.line(190, py(tick), 760, py(tick), { stroke: "grid", width: 1.5 });
            s.text(178, py(tick) + 8, tick === FLOOR ? `≤ +${tick}` : `+${tick}`, { size: 22, anchor: "end", fill: "muted" });
          }
          s.line(190, 490, 760, 490);
          widths.forEach((bits, i) => s.text(px(i), 522, String(bits), { size: 22, anchor: "middle", fill: "muted" }));
          s.text(475, 558, `bits per code, in groups of ${size}`, { size: 22, anchor: "middle", fill: "muted" });
          const series = [
            { sweep: small, name: "14M model", stroke: "s1", at: 0.6 },
            { sweep: large, name: "0.5B model", stroke: "s2", at: 2.6 },
          ].map((m, j) => {
            const points = widths.map((bits, i) => [px(i), py(added(m.sweep, bits))]);
            const path = s.path(points, { stroke: m.stroke });
            const key = s.group();
            key.circle(842, 142 + j * 38, 8, { fill: m.stroke });
            key.text(862, 150 + j * 38, `${m.name}, from a loss of ${m.sweep.loss.toFixed(3)}`, { size: 22 });
            return { ...m, path, key, dots: points.map(([a, b]) => s.circle(a, b, 7, { fill: m.stroke })) };
          });
          const table = [8, 4, 2].map((bits, i) => {
            const g = s.group();
            const y = 268 + i * 44;
            g.text(842, y, `${bits} bits`, { size: 24 });
            g.text(1060, y, signed(added(small, bits)), { size: 24, anchor: "end", fill: "s1", weight: 600, mono: true });
            g.text(1200, y, signed(added(large, bits)), { size: 24, anchor: "end", fill: "s2", weight: 600, mono: true });
            return g;
          });
          // the model of the site's own page: the 14M one, at 4 bits
          const mine = s.circle(px(widths.indexOf(site.quant.bits)), py(site.loss - site.loss_before), 14, { stroke: "ink" });
          mine.setAttribute("class", "s-ink");
          mine.setAttribute("fill", "none");
          const note = s.lines(842, 444, ["Circled: the 14M model at 4 bits,", `a file of ${site.megabytes.toFixed(1)} MB. This site runs it.`], { size: 22, fill: "ink2" });
          return (t) => {
            for (const m of series) {
              show(m.key, during(t, m.at - 0.3, m.at + 0.1));
              trace(m.path, during(t, m.at, m.at + 1.8));
              m.dots.forEach((dot, i) => show(dot, during(t, m.at + i * 0.36 - 0.1, m.at + i * 0.36 + 0.1)));
            }
            table.forEach((g, i) => show(g, during(t, 4.6 + i * 0.4, 5.0 + i * 0.4)));
            show(mine, during(t, 5.8, 6.2));
            note.forEach((line) => show(line, during(t, 5.8, 6.3)));
          };
        },
      },
      {
        seconds: 8,
        caption: "Six bits keep the model whole. Four bits cost it a third of\nwhat it could do, and the loss did not say so.",
        draw(s) {
          const runs = [
            { name: "16-bit floats", score: e16, loss: large.loss, mb: megabytes(16) },
            ...[8, 6, 5, 4].map((bits, i) => ({ name: `${bits} bits`, score: [e8, e6, e5, e4][i], loss: setting(large, bits).loss, mb: megabytes(setting(large, bits).bits_per_weight) })),
          ];
          const title = s.text(640, 124, `share of the ${e16.n.toLocaleString("en")} GSM8K test problems that the 0.5B model solves`, { size: 22, anchor: "middle", fill: "muted" });
          s.line(190, 430, 1130, 430);
          const rowNames = [s.text(70, 500, "file", { size: 22, fill: "muted" }), s.text(70, 536, "loss", { size: 22, fill: "muted" })];
          const columns = runs.map((r, i) => {
            const x = 300 + i * 180;
            const h = 500 * r.score.mean;
            const reach = 500 * r.score.error;
            const last = i === runs.length - 1;
            return {
              ...r,
              h,
              reach,
              bar: s.rect(x - 48, 430, 96, 0, { fill: last ? "s2" : "s1", r: 6 }),
              // the 95% interval of the share
              interval: s.line(x, 430 - h - reach, x, 430 - h + reach, { stroke: "ink", width: 2.5 }),
              value: s.text(x, 430 - h - reach - 12, "", { size: 26, weight: 600, anchor: "middle" }),
              label: s.text(x, 464, r.name, { size: 22, anchor: "middle", weight: last ? 700 : 400 }),
              file: s.text(x, 500, `${Math.round(r.mb)} MB`, { size: 22, anchor: "middle", fill: "ink2" }),
              held: s.text(x, 536, r.loss.toFixed(3), { size: 22, anchor: "middle", fill: "ink2" }),
            };
          });
          const [five, four] = runs.slice(-2);
          const step = s.lines(1210, 150, ["from 5 bits to 4:", `loss ${signed(four.loss - five.loss)}`, `solved ${minus((100 * (four.score.mean - five.score.mean)).toFixed(1))} points`], { size: 22, anchor: "end", fill: "s2", weight: 600 });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            columns.forEach((c, i) => {
              const u = during(t, 0.5 + i * 0.4, 1.4 + i * 0.4);
              set(c.bar, { y: 430 - c.h * u, height: c.h * u });
              c.value.textContent = `${(100 * c.score.mean * u).toFixed(1)}%`;
              set(c.value, { y: 430 - c.h * u - c.reach - 12 });
              show(c.value, during(t, 0.5 + i * 0.4, 0.8 + i * 0.4));
              show(c.label, during(t, 0.5 + i * 0.4, 0.8 + i * 0.4));
              show(c.interval, during(t, 1.3 + i * 0.4, 1.6 + i * 0.4));
              show(c.file, during(t, 3.5 + i * 0.1, 3.9 + i * 0.1));
              show(c.held, during(t, 4.4 + i * 0.1, 4.8 + i * 0.1));
            });
            show(rowNames[0], during(t, 3.4, 3.8));
            show(rowNames[1], during(t, 4.3, 4.7));
            step.forEach((line) => show(line, during(t, 5.4, 5.9)));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "This repository expands the codes for every token and is slow.\nMLX computes on the packed codes, and fewer bits are faster.",
        draw(s) {
          const places = [
            { where: "this repository", fill: "s1", at: 0.5 },
            { where: "MLX", fill: "s2", at: 2.8 },
          ];
          const widths = [16, 8, 4];
          const of = (where, bits) => bench.cases.find((c) => c.where === where && c.bits === bits);
          const charts = [
            { x: 250, title: "Tokens per second, one sequence", value: (c) => c.batch_1, format: (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v))) },
            { x: 770, title: "Memory at its peak, GB", value: (c) => c.peak_gb, format: (v) => v.toFixed(2) },
          ];
          places.forEach((p, j) => {
            s.circle(260 + j * 230, 118, 8, { fill: p.fill });
            s.text(278 + j * 230, 126, p.where, { size: 22 });
          });
          widths.forEach((bits, i) => s.text(232, 250 + i * 104, bits === 16 ? "16-bit floats" : `${bits} bits`, { size: 22, anchor: "end" }));
          const bars = [];
          for (const chart of charts) {
            s.text(chart.x, 176, chart.title, { size: 24, weight: 600 });
            s.line(chart.x, 196, chart.x, 500);
            const peak = Math.max(...bench.cases.map(chart.value));
            widths.forEach((bits, i) =>
              places.forEach((p, j) => {
                const v = chart.value(of(p.where, bits));
                const y = 206 + i * 104 + j * 40;
                bars.push({ w: (330 * v) / peak, at: p.at + i * 0.25, bar: s.rect(chart.x, y, 0, 32, { fill: p.fill, r: 6 }), value: s.text(chart.x + (330 * v) / peak + 10, y + 24, chart.format(v), { size: 22, fill: "ink2" }) });
              }),
            );
          }
          const fast = of("MLX", 4);
          const sum = s.text(640, 552, `MLX at 4 bits: ${(fast.batch_1 / floats.batch_1).toFixed(1)} times the speed of 16-bit floats, in ${Math.round((100 * fast.peak_gb) / floats.peak_gb)}% of the memory`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            for (const b of bars) {
              set(b.bar, { width: b.w * during(t, b.at, b.at + 1.0) });
              show(b.value, during(t, b.at + 0.8, b.at + 1.1));
            }
            show(sum, during(t, 4.9, 5.4));
          };
        },
      },
    ],
  };
};
