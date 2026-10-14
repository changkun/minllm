// The side topic 8b as a film: stage 8's DPO on the 0.5B model, with the
// verifier of stage 9 deciding which reply of a pair is preferred. The pair
// shown, the counts, the curves of three runs and every score are logged in
// runs/dpo/ and runs/eval/. The eight replies of the first scene stand for
// any problem's, marked as a problem with right and wrong replies is.

import { during, mix, show, trace } from "./stage.js";

export const sources = {
  pairs: "/runs/dpo/pairs.json",
  summary: "/runs/dpo/summary.json",
  meta: "/runs/dpo/dpo-meta.json",
  fast: "/runs/dpo/dpo.jsonl",
  held: "/runs/dpo/dpo-nll.jsonl",
  slow: "/runs/dpo/dpo-lr1e-5.jsonl",
  start: "/runs/eval/gsm8k-sft-gsm8k-lora.json",
  self: "/runs/eval/gsm8k-self-gsm8k.json",
  grpo: "/runs/eval/gsm8k-grpo-gsm8k.json",
  slowest: "/runs/eval/gsm8k-dpo-gsm8k-lr2e-6.json",
  slowTest: "/runs/eval/gsm8k-dpo-gsm8k-lr1e-5.json",
  once: "/runs/eval/gsm8k-dpo-gsm8k-1.json",
  fastTest: "/runs/eval/gsm8k-dpo-gsm8k.json",
  heldTest: "/runs/eval/gsm8k-dpo-gsm8k-nll.json",
};

const percent = (v, digits = 0) => `${(100 * v).toFixed(digits)}%`;
const count = (v) => v.toLocaleString("en");
const nats = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(0)}`;
const MARKS = [true, false, false, true, false, false, true, false]; // right or wrong, for the eight replies of the first scene

// Text in lines of at most width characters, a line of the text staying a line.
function wrap(text, width) {
  return text.split("\n").flatMap((paragraph) => {
    const lines = [""];
    for (const word of paragraph.split(/\s+/)) {
      const last = lines[lines.length - 1];
      if (last && last.length + 1 + word.length > width) lines.push(word);
      else lines[lines.length - 1] = last ? `${last} ${word}` : word;
    }
    return lines;
  });
}

// A running mean over five steps: single steps of 16 pairs are noisy.
const smooth = (rows, key) => rows.map((_, j) => rows.slice(Math.max(0, j - 4), j + 1).reduce((sum, r) => sum + r[key], 0) / Math.min(j + 1, 5));

// Bars that stand on a floor, each with its score, the interval of the score and a name.
function standing(s, x, floor, tall, rows, start) {
  const made = rows.map(([name, result, accent], i) => {
    const left = x + i * 142;
    const h = tall(result.mean);
    s.lines(left + 48, floor + 32, name, { size: 20, anchor: "middle", fill: "ink2", leading: 1.2 });
    const bar = s.rect(left, floor, 96, 0, { fill: accent, r: 6 });
    const over = s.group();
    over.text(left + 48, floor - h - tall(result.error) - 10, percent(result.mean, 1), { size: 22, weight: 700, anchor: "middle" });
    over.line(left + 48, floor - h - tall(result.error), left + 48, floor - h + tall(result.error), { stroke: "ink", width: 2.5 });
    return { bar, h, over };
  });
  s.line(x - 20, floor, x + rows.length * 142 - 26, floor, { stroke: "axis" });
  return (t) =>
    made.forEach(({ bar, h, over }, i) => {
      const u = during(t, start + i * 0.35, start + 0.85 + i * 0.35);
      bar.setAttribute("y", (floor - h * u).toFixed(1));
      bar.setAttribute("height", (h * u).toFixed(1));
      show(over, during(t, start + 0.65 + i * 0.35, start + 1.05 + i * 0.35));
    });
}

export default ({ pairs, summary, meta, fast, held, slow, start, self, grpo, slowest, slowTest, once, fastTest, heldTest }) => {
  const example = summary.example;
  const problem = wrap(example.problem.split("\n\n")[0], 64);
  const trained = (log) => log.filter((r) => "gain_preferred" in r);
  const runs = [
    { name: "lr 1e-5", rows: trained(slow), color: "s3" },
    { name: "lr 5e-5", rows: trained(fast), color: "s2" },
    { name: "5e-5 + the right reply's loss", rows: trained(held), color: "s1" },
  ];
  const steps = Math.max(...fast.map((r) => r.step));
  const perPass = Math.floor(pairs.pairs / meta.args.batch_size);
  const end = summary.runs;
  const dpos = [slowest, slowTest, once, fastTest, heldTest];
  const none = dpos.every((r) => r.mean < start.mean);

  return {
    subtitle: "A right reply preferred to a wrong one, by the verifier",
    scenes: [
      {
        seconds: 7.5,
        caption: "Stage 8's DPO, with stage 9's verifier as the annotator:\nof two replies to one problem, the right one is preferred.",
        draw(s) {
          const ask = s.box(80, 250, 230, 96, "a problem", "of GSM8K's training set");
          const replies = MARKS.map((right, i) => {
            const g = s.group();
            const y = 156 + i * 50;
            g.rect(420, y, 260, 38, { fill: "raised", stroke: "axis", r: 8 });
            g.text(440, y + 27, `reply ${i + 1}`, { size: 22, fill: "ink2" });
            const mark = g.text(660, y + 28, right ? "right" : "wrong", { size: 22, weight: 700, anchor: "end", fill: right ? "s3" : "muted" });
            return { g, mark, y, right };
          });
          const fan = s.group();
          for (const r of replies) fan.line(310, 298, 420, r.y + 19, { stroke: "grid", width: 2 });
          const first = replies.findIndex((r) => r.right);
          const second = replies.findIndex((r) => !r.right);
          const pair = s.group();
          pair.box(820, 214, 360, 84, "preferred", "a right reply", { stroke: "s3" });
          pair.box(820, 326, 360, 84, "the other", "a wrong one", { stroke: "axis" });
          pair.arrow(690, replies[first].y + 19, 810, 256);
          pair.arrow(690, replies[second].y + 19, 810, 368);
          const note = s.text(1000, 460, `up to ${pairs.args.per_prompt} pairs a problem`, { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            show(ask, during(t, 0.3, 0.7));
            show(fan, during(t, 0.8, 1.2));
            replies.forEach((r, i) => {
              show(r.g, during(t, 0.9 + i * 0.12, 1.3 + i * 0.12));
              show(r.mark, during(t, 2.2 + i * 0.12, 2.6 + i * 0.12));
            });
            show(pair, during(t, 3.8, 4.4));
            show(note, during(t, 4.8, 5.2));
          };
        },
      },
      {
        seconds: 8,
        caption: `${count(pairs.replies)} replies to ${pairs.problems} problems gave ${count(pairs.pairs)} pairs.\nThe problems and the seed are those of self-training's first round.`,
        draw(s) {
          const kinds = [
            ["only right replies", pairs.problems_all_right, "muted"],
            ["only wrong replies", pairs.problems_none_right, "muted"],
            ["right and wrong ones", pairs.problems_with_both, "s2"],
          ];
          const FULL = 640;
          s.text(120, 150, `${pairs.problems} problems, ${pairs.args.samples} replies each, ${percent(pairs.right, 1)} of them right`, { size: 26, weight: 600 });
          const rows = kinds.map(([name, n, fill], i) => {
            const g = s.group();
            const y = 196 + i * 72;
            g.text(120, y + 32, name, { size: 24, fill: "ink2" });
            g.rect(440, y, FULL, 44, { fill: "tint", r: 6 });
            const w = (FULL * n) / pairs.problems;
            return { g, w, bar: g.rect(440, y, 0, 44, { fill, r: 6 }), value: g.text(440 + w + 12, y + 32, String(n), { size: 26, weight: 700 }) };
          });
          const gives = s.lines(640, 452, [`Only the ${pairs.problems_with_both} problems with both say which reply is better.`, `They give ${count(pairs.pairs)} pairs.`], { size: 26, anchor: "middle", fill: "ink", leading: 1.4 });
          return (t) => {
            rows.forEach((row, i) => {
              show(row.g, during(t, 0.4 + i * 0.5, 0.8 + i * 0.5));
              const u = during(t, 0.6 + i * 0.5, 1.4 + i * 0.5);
              row.bar.setAttribute("width", (row.w * u).toFixed(1));
              row.value.setAttribute("x", (440 + row.w * u + 12).toFixed(1));
            });
            gives.forEach((line, i) => show(line, during(t, 3.4 + i * 0.5, 3.9 + i * 0.5)));
          };
        },
      },
      {
        seconds: 8,
        caption: "A right reply and a wrong one look alike.\nThis wrong one forgot to add the pencil.",
        draw(s) {
          const ask = s.lines(640, 128, problem, { size: 24, anchor: "middle", fill: "ink2", leading: 1.3 });
          const cards = [
            [example.preferred, "preferred, right", "s3"],
            [example.other, "the other, wrong", "muted"],
          ].map(([text, tag, color], i) => {
            const g = s.group();
            const x = 60 + i * 600;
            g.rect(x, 210, 560, 250, { fill: "raised", stroke: i ? "axis" : "s3" });
            g.text(x + 24, 250, tag, { size: 22, weight: 700, fill: color });
            wrap(text, 38).forEach((line, j) => g.text(x + 24, 296 + j * 34, line, { size: 22, mono: true, fill: line.startsWith("####") ? "ink" : "ink2", weight: line.startsWith("####") ? 700 : 400 }));
            return g;
          });
          return (t) => {
            ask.forEach((line, i) => show(line, during(t, 0.3 + i * 0.3, 0.7 + i * 0.3)));
            cards.forEach((card, i) => show(card, during(t, 1.4 + i * 1.2, 1.9 + i * 1.2)));
          };
        },
      },
      {
        seconds: 10,
        caption: `DPO asks only that the right reply gain more than the wrong one.\nAt 5e-5 the right replies fell too, by ${Math.abs(end.dpo.gain_preferred).toFixed(1)} nats over two passes.`,
        draw(s) {
          const panels = [
            ["gain_preferred", "the right replies", 90],
            ["gain_other", "the wrong replies", 690],
          ].map(([key, title, x]) => {
            const values = runs.flatMap((run) => smooth(run.rows, key));
            const low = Math.floor(Math.min(...values) / 10) * 10;
            const high = Math.max(10, Math.ceil(Math.max(...values) / 10) * 10);
            const ticks = key === "gain_other" ? [0, -40, -80] : [10, 0, -10, -20];
            const { px, py } = s.axes(x + 60, 170, 440, 270, {
              x: { domain: [0, steps], ticks: [0, perPass, steps], title: "step" },
              y: { domain: [low, high], ticks: ticks.filter((v) => v >= low && v <= high), format: (v) => (v ? nats(v) : "0") },
            });
            s.text(x + 60, 140, `${title}: log p − log p_start, in nats`, { size: 22, weight: 600 });
            const lines = runs.map((run) => {
              const ys = smooth(run.rows, key);
              const points = run.rows.map((r, j) => [px(r.step), py(ys[j])]);
              return { path: s.path(points, { stroke: run.color, width: 4 }), value: s.text(points.at(-1)[0] + 10, points.at(-1)[1] + 8, nats(ys.at(-1)), { size: 20, weight: 600 }) };
            });
            return lines;
          });
          const key = s.group();
          runs.forEach((run, i) => {
            key.line(240 + i * 300, 562, 280 + i * 300, 562, { stroke: run.color, width: 4 });
            key.text(292 + i * 300, 570, run.name, { size: 22, fill: "ink2" });
          });
          return (t) => {
            show(key, during(t, 0.2, 0.6));
            panels.forEach((lines) =>
              lines.forEach(({ path, value }, i) => {
                const from = 0.5 + i * 2.2;
                show(path, t > from ? 1 : 0);
                trace(path, during(t, from, from + 2));
                show(value, during(t, from + 1.9, from + 2.3));
              }),
            );
          };
        },
      },
      {
        seconds: 9.5,
        caption: `On the test set ${none ? "no DPO run beats the start" : "DPO hardly moves"}. GRPO, which sampled\nas many replies as it learned, gained ${(100 * (grpo.mean - start.mean)).toFixed(1)} points.`,
        draw(s) {
          const FLOOR = 470;
          const tall = (v) => 520 * v;
          s.text(60, 128, `GSM8K test problems solved, of ${count(start.n)}`, { size: 24, weight: 600 });
          const bars = standing(
            s,
            80,
            FLOOR,
            tall,
            [
              [["the start"], start, "muted"],
              [["self-", "training"], self, "s1"],
              [["DPO,", "2e-6"], slowest, "s2"],
              [["DPO,", "1e-5"], slowTest, "s2"],
              [["5e-5,", "one pass"], once, "s2"],
              [["5e-5,", "two passes"], fastTest, "s2"],
              [["5e-5 + right", "reply's loss"], heldTest, "s2"],
              [["GRPO"], grpo, "s1"],
            ],
            0.4,
          );
          return (t) => bars(t);
        },
      },
    ],
  };
};
