// Stage 7 as a film: a score is the share of held-out problems answered
// right, drawing the problems again shows how sure it is, and two models are
// compared problem by problem. Every score is counted or resampled here from
// the per-problem results that the evaluations saved in runs/eval/.

import { during, mix, show } from "./stage.js";

export const sources = {
  published: "/runs/eval/gsm8k-qwen.json",
  lora: "/runs/eval/gsm8k-sft-gsm8k-lora.json",
  full: "/runs/eval/gsm8k-sft-gsm8k.json",
  wordsBase: "/runs/eval/words-base.json",
  wordsTuned: "/runs/eval/words-sft-words.json",
  training: "/runs/sft-gsm8k-lora/log.jsonl",
};

const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js
const DRAWS = 2000; // how often the problems are drawn again, as in evaluate.paired_difference

// A small random number generator with a seed, so that every frame shows the same draws.
function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The bootstrap: the mean of n values drawn with replacement, DRAWS times,
// and the 95% of them in the middle. Values are in points of a percentage.
function resample(values, n, seed) {
  const rng = random(seed);
  const means = Array.from({ length: DRAWS }, () => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += values[Math.floor(rng() * values.length)];
    return (100 * sum) / n;
  });
  const sorted = [...means].sort((a, b) => a - b);
  return { means, low: sorted[Math.floor(0.025 * DRAWS)], high: sorted[Math.floor(0.975 * DRAWS)] };
}

// tasks.gsm8k_number of the repository: the number after the last ####, or else the last number of the reply.
function answer(reply) {
  const marked = reply.includes("####");
  const numbers = (marked ? reply.split("####").at(-1) : reply).match(/-?\d[\d,]*\.?\d*/g);
  return numbers ? (marked ? numbers[0] : numbers.at(-1)).replaceAll(",", "").replace(/\.$/, "") : "none";
}

function wrap(string, n) {
  const rows = [""];
  for (const word of string.split(" ")) {
    if (rows.at(-1) && rows.at(-1).length + 1 + word.length > n) rows.push(word);
    else rows[rows.length - 1] += (rows.at(-1) ? " " : "") + word;
  }
  return rows;
}

// The last two lines of a reply as they fit a card, with its LaTeX markup written out.
function ending(reply, n) {
  const lines = reply.replace(/\\\(|\\\)/g, "").replace(/\\boxed\{([^}]*)\}/g, "$1").split("\n").map((line) => line.trim()).filter(Boolean);
  const last = wrap(lines.at(-1), n - 2);
  if (last.length > 1) return [`… ${last.at(-2)}`, last.at(-1)];
  const words = (lines.at(-2) || "").split(" ");
  let tail = "";
  while (words.length && tail.length + words.at(-1).length + 1 <= n - 2) tail = `${words.pop()} ${tail}`.trim();
  return [`… ${tail}`, last[0]];
}

const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
const count = (v) => Math.round(v).toLocaleString("en");
const percent = (v) => `${(100 * v).toFixed(1)}%`;
const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;

// A histogram that fills as the draws come in: bars over [low, high) of the given width in points.
function histogram(s, { means, px, floor, tall, low, high, width, color }) {
  const bins = Math.floor((high - low) / width);
  const bin = means.map((m) => Math.min(bins - 1, Math.max(0, Math.floor((m - low) / width))));
  const total = new Array(bins).fill(0);
  for (const b of bin) total[b]++;
  const peak = Math.max(...total);
  const bars = total.map((_, b) => s.rect(px(low + b * width) + 1, floor, px(low + width) - px(low) - 2, 0, { fill: color, r: 2 }));
  return (drawn) => {
    const counts = new Array(bins).fill(0);
    for (let k = 0; k < drawn; k++) counts[bin[k]]++;
    bars.forEach((bar, b) => {
      const h = (tall * counts[b]) / peak;
      bar.setAttribute("height", h.toFixed(1));
      bar.setAttribute("y", (floor - h).toFixed(1));
    });
  };
}

export default ({ published, lora, full, wordsBase, wordsTuned, training }) => {
  const n = published.n;
  const right = published.scores.filter(Boolean).length;
  const first = [published.samples[0], lora.samples.find((sample) => sample.prompt === published.samples[0].prompt)];
  // The seeds of the draws are fixed, so that every frame shows the same ones.
  // With this many draws an end of an interval moves by about 0.2 points
  // from one seed to another, and these seeds were chosen to give the ends
  // that docs/07-evaluation.md reports from scripts/evaluate.py.
  // how sure one score is: the test set drawn again, and a test set of a hundred
  const whole = resample(published.scores, n, 1);
  const hundred = resample(published.scores, 100, 3);
  // two models on the same problems: the differences, problem by problem
  const pairs = (a, b) => a.scores.map((score, i) => b.scores[i] - score);
  const ways = { diff: pairs(lora, full), ...resample(pairs(lora, full), n, 7) };
  const tuning = { diff: pairs(published, lora), ...resample(pairs(published, lora), n, 8) };
  const verdict = (d) => (d.low > 0 || d.high < 0 ? "zero is outside: a real difference" : "zero is inside: within noise");
  const apart = ways.low > 0 || ways.high < 0;
  const only = { lora: ways.diff.filter((d) => d < 0).length, full: ways.diff.filter((d) => d > 0).length };
  const both = lora.scores.filter((score, i) => score && full.scores[i]).length;
  const held = training.filter((row) => row.val_loss !== undefined).map((row) => row.val_loss);
  const gain = { words: 100 * (wordsTuned.mean - wordsBase.mean), gsm8k: 100 * (lora.mean - published.mean) };

  return {
    subtitle: "A score, and how far to trust it",
    scenes: [
      {
        seconds: 8,
        caption: "An evaluation asks problems the model was not trained on.\nA program reads each reply and gives it a score.",
        draw(s) {
          const question = s.group();
          question.rect(90, 108, 1100, 148, { fill: "tint", stroke: "axis" });
          question.lines(112, 144, wrap(first[0].prompt.split("\n\n")[0], 79), { size: 22, mono: true, leading: 1.36 });
          const cards = first.map((sample, i) => {
            const x = 90 + i * 570;
            const card = s.group();
            card.rect(x, 282, 530, 150, { fill: "raised", stroke: "axis" });
            card.text(x + 22, 320, ["the model as published", "fine-tuned through LoRA"][i], { size: 24, weight: 600 });
            card.text(x + 508, 320, `${count(sample.reply.length)} characters`, { size: 22, anchor: "end", fill: "muted" });
            // the end of the reply, with the number that the program takes as the answer
            const number = answer(sample.reply);
            const rows = ending(sample.reply, 36);
            const row = rows.findLastIndex((line) => line.includes(number));
            let read = null;
            rows.forEach((line, k) => {
              const y = 366 + k * 32;
              if (k !== row) return card.text(x + 22, y, line, { size: 22, mono: true });
              const at = line.lastIndexOf(number);
              card.text(x + 22, y, line.slice(0, at), { size: 22, mono: true });
              read = card.text(x + 22 + at * 22 * MONO, y, number, { size: 22, mono: true, weight: 700 });
              card.text(x + 22 + (at + number.length) * 22 * MONO, y, line.slice(at + number.length), { size: 22, mono: true });
            });
            const reads = s.group();
            reads.text(x + 22, 486, "the answer it reads", { size: 24, fill: "ink2" });
            reads.chip(x + 250, 457, number, { size: 24 });
            const score = s.group();
            score.text(x + 372, 486, "score", { size: 24, fill: "ink2" });
            score.chip(x + 446, 457, String(sample.score), { size: 24, fill: sample.score ? "s3" : "tint", stroke: sample.score ? null : "axis", ink: sample.score ? "surface" : "ink" });
            return { card, read, reads, score };
          });
          return (t) => {
            show(question, during(t, 0.3, 0.9));
            cards.forEach((c, i) => {
              show(c.card, during(t, 1.4 + i * 0.7, 2 + i * 0.7));
              c.read?.setAttribute("class", `film-mono ${t >= 3.6 + i * 0.4 ? "f-s2" : "f-ink"}`);
              show(c.reads, during(t, 3.6 + i * 0.4, 4.1 + i * 0.4));
              show(c.score, during(t, 4.8 + i * 0.4, 5.3 + i * 0.4));
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `The score is the share of the test problems answered right:\n${percent(published.mean)} of ${count(n)} for the model as published.`,
        draw(s) {
          // every test problem as a cell, column by column in the order of the test set
          const [columns, rows, pitch] = [60, 22, 18];
          const [x, y] = [640 - (columns * pitch - 3) / 2, 116];
          const ground = s.group();
          const filled = Array.from({ length: columns }, () => s.group());
          const seen = []; // after each column: how many problems were asked, and how many answered right
          let correct = 0;
          published.scores.forEach((score, i) => {
            const [column, row] = [Math.floor(i / rows), i % rows];
            ground.rect(x + column * pitch, y + row * pitch, pitch - 3, pitch - 3, { fill: "grid", r: 3 });
            if (score) filled[column].rect(x + column * pitch, y + row * pitch, pitch - 3, pitch - 3, { fill: "s3", r: 3 });
            correct += score ? 1 : 0;
            seen[column] = { asked: i + 1, right: correct };
          });
          const sum = s.text(640, 560, "", { size: 30, weight: 600, anchor: "middle" });
          return (t) => {
            show(ground, during(t, 0.3, 0.9));
            const upto = Math.min(columns, Math.floor(mix(0, columns, during(t, 1.2, 4.6)) + 1e-9));
            filled.forEach((g, column) => show(g, column < upto ? 1 : 0));
            const now = upto ? seen[upto - 1] : { asked: 0, right: 0 };
            sum.textContent = upto ? `${count(now.right)} of ${count(now.asked)} right: ${percent(now.right / now.asked)}` : "";
          };
        },
      },
      {
        seconds: 9,
        caption: "The problems are a sample, so the score is an estimate.\nDrawing them again shows how far it could have landed.",
        draw(s) {
          const [low, high] = [30, 64];
          const px = (v) => 140 + (1000 * (v - low)) / (high - low);
          // a bar holds a whole number of problems, so that no bar gets more of the possible scores than its neighbour
          const tests = [
            { draws: whole, size: n, floor: 280, low, high, width: 700 / n, from: 1, to: 3.4 },
            { draws: hundred, size: 100, floor: 478, low: low - 0.5, high: high + 0.5, width: 1, from: 4.4, to: 6.4 },
          ].map((test) => {
            s.line(px(low), test.floor, px(high), test.floor, { stroke: "axis" });
            return {
              ...test,
              fill: histogram(s, { means: test.draws.means, px, floor: test.floor, tall: 130, low: test.low, high: test.high, width: test.width, color: "s1" }),
              name: s.text(140, test.floor - 96, `a test of ${count(test.size)} problems`, { size: 24, weight: 600 }),
              drawn: s.text(140, test.floor - 62, "", { size: 22, fill: "muted" }),
              span: s.line(px(test.draws.low), test.floor + 9, px(test.draws.high), test.floor + 9, { stroke: "s2", width: 6 }),
              range: s.lines(1140, test.floor - 96, ["95% of the draws score", `${test.draws.low.toFixed(test.size < 1000 ? 0 : 1)}% to ${test.draws.high.toFixed(test.size < 1000 ? 0 : 1)}%`], { size: 24, anchor: "end", leading: 1.4 }),
            };
          });
          for (const test of tests) test.range[1].setAttribute("font-weight", 700);
          for (const tick of [30, 40, 50, 60]) s.text(px(tick), 478 + 38, `${tick}%`, { size: 22, anchor: "middle", fill: "muted" });
          s.text(640, 554, "share answered right", { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            for (const test of tests) {
              const drawn = Math.round(DRAWS * during(t, test.from, test.to));
              test.fill(drawn);
              show(test.name, during(t, test.from - 0.6, test.from - 0.1));
              show(test.drawn, during(t, test.from - 0.6, test.from - 0.1));
              test.drawn.textContent = `drawn again ${count(drawn)} times`;
              show(test.span, during(t, test.to + 0.2, test.to + 0.7));
              for (const line of test.range) show(line, during(t, test.to + 0.2, test.to + 0.7));
            }
          };
        },
      },
      {
        seconds: 8,
        caption: "Two models scored on the same problems share the luck of\nthe draw. Comparing problem by problem removes it.",
        draw(s) {
          const [shown, pitch, x, y] = [36, 24, 330, 128];
          const sides = [lora, full];
          s.text(90, y + 17, "through LoRA", { size: 24, fill: "ink2" });
          s.text(90, y + 45, "every weight", { size: 24, fill: "ink2" });
          const columns = Array.from({ length: shown }, (_, i) => {
            const differ = sides[0].scores[i] !== sides[1].scores[i];
            const mark = s.rect(x + i * pitch - 3, y - 4, pitch, 56, { fill: "surface", stroke: "s2", r: 5, width: 2.5 });
            const g = s.group();
            sides.forEach((side, k) => g.rect(x + i * pitch, y + k * 28, pitch - 6, 20, { fill: side.scores[i] ? "s3" : "grid", r: 4 }));
            return { g, mark, differ };
          });
          const note = s.text(x - 3, y + 92, `the first ${shown} of the ${count(n)} test problems, marked where only one model is right`, { size: 22, fill: "muted" });
          // all of the problems, by what the two models did on them
          const [bx, by, bw, bh] = [90, 286, 1100, 58];
          const parts = [
            { n: only.lora, fill: "s2", ink: "surface", label: "" },
            { n: both, fill: "s3", ink: "surface", label: " both right" },
            { n: n - both - only.lora - only.full, fill: "tint", ink: "ink", label: " both wrong" },
            { n: only.full, fill: "s1", ink: "surface", label: "" },
          ];
          const frame = s.rect(bx, by, bw, bh, { fill: "surface", stroke: "axis", r: 0, width: 1.5 });
          let from = 0;
          for (const part of parts) {
            part.x = bx + (bw * from) / n;
            part.w = (bw * part.n) / n;
            part.bar = s.rect(part.x, by, 0, bh, { fill: part.fill, r: 0 });
            part.text = s.text(part.x + part.w / 2, by + 38, `${count(part.n)}${part.label}`, { size: 24, weight: 700, anchor: "middle", fill: part.ink });
            from += part.n;
          }
          const ends = [s.text(bx, by + bh + 34, "only LoRA right", { size: 24, fill: "ink2" }), s.text(bx + bw, by + bh + 34, "only every weight right", { size: 24, anchor: "end", fill: "ink2" })];
          const same = n - only.lora - only.full;
          const sum = s.lines(640, 468, [`On ${count(same)} problems the two score the same.`, `The difference between them is ${only.lora} against ${only.full}.`], { size: 28, anchor: "middle", leading: 1.45 });
          return (t) => {
            const split = during(t, 1.8, 2.6);
            columns.forEach((c, i) => {
              show(c.g, during(t, 0.3 + i * 0.02, 0.7 + i * 0.02) * (c.differ ? 1 : mix(1, 0.45, split)));
              show(c.mark, c.differ ? split : 0);
            });
            show(note, split);
            show(frame, during(t, 2.9, 3.3));
            const grown = bw * during(t, 3.2, 4.8);
            for (const part of parts) {
              part.bar.setAttribute("width", Math.min(part.w, Math.max(0, grown - (part.x - bx))).toFixed(1));
              show(part.text, grown >= part.x - bx + part.w ? 1 : 0);
            }
            for (const end of ends) show(end, during(t, 4.8, 5.3));
            sum.forEach((line, i) => show(line, during(t, 5.4 + i * 0.4, 5.9 + i * 0.4)));
          };
        },
      },
      {
        seconds: 8,
        caption: `Full fine-tuning scores ${signed(100 * mean(ways.diff))} points against LoRA, and zero\nis ${apart ? "outside" : "inside"} the interval: ${apart ? "the two can be told apart." : "they cannot be told apart."}`,
        draw(s) {
          const [low, high, floor] = [-16, 10, 456];
          const px = (v) => 140 + (1000 * (v - low)) / (high - low);
          s.line(px(low), floor, px(high), floor, { stroke: "axis" });
          for (const tick of [-15, -10, -5, 5, 10]) s.text(px(tick), floor + 46, signed(tick).replace(".0", ""), { size: 22, anchor: "middle", fill: "muted" });
          s.text(px(0), floor + 46, "no difference", { size: 22, anchor: "middle", weight: 600 });
          s.text(640, floor + 84, "difference in points, with the same problems drawn again", { size: 22, anchor: "middle", fill: "muted" });
          const sets = [
            { d: ways, color: "s2", from: 0.8, to: 2.6, x: px(0) + 26, y: 196, anchor: "start", name: "every weight, against LoRA" },
            { d: tuning, color: "s1", from: 4, to: 5.6, x: 140, y: 136, anchor: "start", name: "LoRA, against the model as published" },
          ].map((set) => ({
            ...set,
            fill: histogram(s, { means: set.d.means, px, floor, tall: 190, low, high, width: 400 / n, color: set.color }), // a bar is four problems wide
            span: s.line(px(set.d.low), floor + 9, px(set.d.high), floor + 9, { stroke: set.color, width: 6 }),
            text: s.lines(set.x, set.y, [set.name, `${signed(100 * mean(set.d.diff))} points`, `95% of the draws: ${signed(set.d.low)} to ${signed(set.d.high)}`, verdict(set.d)], { size: 24, anchor: set.anchor, leading: 1.4 }),
          }));
          const zero = s.line(px(0), 240, px(0), floor + 16, { stroke: "ink", width: 2 });
          for (const set of sets) {
            set.text[0].setAttribute("class", `f-${set.color}`);
            set.text[0].setAttribute("font-weight", 700);
            set.text[1].setAttribute("font-weight", 700);
            set.text[3].setAttribute("font-weight", 600);
          }
          return (t) => {
            show(zero, during(t, 0.3, 0.8));
            for (const set of sets) {
              set.fill(Math.round(DRAWS * during(t, set.from, set.to)));
              show(set.text[0], during(t, set.from - 0.4, set.from));
              show(set.text[1], during(t, set.to + 0.1, set.to + 0.5));
              show(set.text[2], during(t, set.to + 0.1, set.to + 0.5));
              show(set.span, during(t, set.to + 0.1, set.to + 0.5));
              show(set.text[3], during(t, set.to + 0.5, set.to + 0.9));
            }
          };
        },
      },
      {
        seconds: 8,
        caption: `Fine-tuning raised the words score by ${Math.abs(gain.words).toFixed(1)} points. On GSM8K\nit cost ${Math.abs(gain.gsm8k).toFixed(1)} points, while its held-out loss was falling.`,
        draw(s) {
          const [floor, scale, wide] = [470, 4.5, 112];
          const groups = [
            { title: "Words: requested words used", change: gain.words, from: 0.5, x: 120, bars: [["14M base", wordsBase, "s1"], ["fine-tuned", wordsTuned, "s2"]] },
            { title: "GSM8K: problems solved", change: gain.gsm8k, from: 2.8, x: 510, bars: [["as published", published, "s1"], ["LoRA", lora, "s2"], ["every weight", full, "s2"]] },
          ].map((group) => {
            const width = group.bars.length * (wide + 34) - 34;
            s.line(group.x - 16, floor, group.x + width + 16, floor, { stroke: "axis" });
            s.text(group.x + width / 2, 128, group.title, { size: 24, weight: 600, anchor: "middle" });
            return {
              ...group,
              delta: s.text(group.x + width / 2, 162, `fine-tuning: ${signed(group.change)} points`, { size: 24, weight: 700, anchor: "middle", fill: group.change > 0 ? "s3" : "s2" }),
              bars: group.bars.map(([name, run, color], i) => {
                const x = group.x + i * (wide + 34);
                s.text(x + wide / 2, floor + 34, name, { size: 22, anchor: "middle", fill: "ink2" });
                const [top, err] = [100 * run.mean * scale, 100 * run.error * scale];
                return {
                  top,
                  bar: s.rect(x, floor, wide, 0, { fill: color, r: 6 }),
                  whisker: s.line(x + wide / 2, floor - top - err, x + wide / 2, floor - top + err, { stroke: "ink", width: 3 }),
                  labels: [s.text(x + wide / 2, floor - top - err - 40, percent(run.mean), { size: 26, weight: 700, anchor: "middle" }), s.text(x + wide / 2, floor - top - err - 12, `± ${(100 * run.error).toFixed(1)}`, { size: 22, anchor: "middle", fill: "muted" })],
                };
              }),
            };
          });
          const loss = s.lines(1060, 286, ["held-out loss", "of the LoRA run", `${held[0].toFixed(3)} → ${held.at(-1).toFixed(3)}`], { size: 24, anchor: "middle", fill: "ink2", leading: 1.4 });
          loss[2].setAttribute("class", "f-ink");
          loss[2].setAttribute("font-weight", 700);
          loss[2].setAttribute("font-size", 30);
          loss[2].setAttribute("y", 366);
          return (t) => {
            for (const group of groups) {
              group.bars.forEach((b, i) => {
                const u = during(t, group.from + i * 0.3, group.from + 1 + i * 0.3);
                b.bar.setAttribute("height", (b.top * u).toFixed(1));
                b.bar.setAttribute("y", (floor - b.top * u).toFixed(1));
                show(b.whisker, during(t, group.from + 1 + i * 0.3, group.from + 1.3 + i * 0.3));
                for (const label of b.labels) show(label, during(t, group.from + 1 + i * 0.3, group.from + 1.3 + i * 0.3));
              });
              show(group.delta, during(t, group.from + 1.8, group.from + 2.3));
            }
            for (const line of loss) show(line, during(t, 5.4, 6));
          };
        },
      },
    ],
  };
};
