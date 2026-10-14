// Side topic 9b as a film: stage 9's GRPO, run from 6a's model with the
// calculator answering the calls while the replies are sampled. The
// conversation of the second scene and the replies of the third are
// illustrations of the format; their results are computed here as
// tools.calculate computes them, and their advantages as
// rl.group_advantages does. The training curves, the held-out scores and
// every test score are the logged ones of scripts/grpo.py,
// scripts/evaluate_tools.py and scripts/analyze_tools.py.

import { clamp, during, mix, place, set, show, trace } from "./stage.js";

export const sources = {
  summary: "/runs/eval/tools-summary.json",
  run: "/runs/rlvr/grpo-gsm8k-tools.json",
  log: "/runs/rlvr/grpo-gsm8k-tools.jsonl",
  stage9Run: "/runs/rlvr/grpo-gsm8k.json",
  stage9Log: "/runs/rlvr/grpo-gsm8k.jsonl",
  plain: "/runs/eval/gsm8k-sft-gsm8k-lora.json",
  grpo: "/runs/eval/gsm8k-grpo-gsm8k.json",
};

// The first problem of GSM8K's training set, and eight replies to it with
// calls, made up to show a group.
const PROBLEM = "Natalia sold clips to 48 of her friends in April, …";
const ANSWER = 72;
const REPLIES = [
  [["48/2", "48+24"], "72"],
  [["48*2", "48+96"], "144"],
  [["48/2", "24+48"], "72"],
  [["48/2"], "24"],
  [["48+48/2"], "72"],
  [["48/2", "48-24"], "24"],
  [["48/2", "48+24"], "72"],
  [["48*2"], "96"],
];

// The reply of the second scene, as the note writes it, and whose each line is.
const CONVERSATION = [
  ["<|im_start|>assistant", "prompt"],
  ["Natalia sold 48/2 =", "model"],
  ["<tool_call>", "model"],
  ['{"name": "calculate", "arguments": {"expression": "48/2"}}', "model"],
  ["</tool_call><|im_end|>", "model"],
  ["<|im_start|>user", "tool"],
  ["<tool_response>", "tool"],
  ["24", "result"],
  ["</tool_response><|im_end|>", "tool"],
  ["<|im_start|>assistant", "tool"],
  ["24 clips in May. …", "again"],
];

const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js
const wide = (string, size) => [...string].length * size * MONO;
const count = (v) => Math.round(v).toLocaleString("en");
// a share as the note writes it: one decimal, none when it is whole
const percent = (v) => `${(100 * v).toFixed(1).replace(/\.0$/, "")}%`;
const whole = (v) => `${Math.round(100 * v)}%`;
const points = (v) => Math.abs(100 * v).toFixed(1);
const signed = (v) => `${v < 0 ? "−" : "+"}${points(v)}`;
const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;

// The value of an expression of numbers, + - * / and parentheses, as tools.calculate reads one.
function value(expression) {
  const tokens = expression.replace(/,/g, "").match(/\d*\.?\d+|[-+*/()]/g) || [];
  let i = 0;
  const atom = () => {
    const token = tokens[i++];
    if (token === "-") return -atom();
    if (token !== "(") return Number(token);
    const v = sum();
    i++;
    return v;
  };
  const product = () => {
    let v = atom();
    while (tokens[i] === "*" || tokens[i] === "/") v = tokens[i++] === "*" ? v * atom() : v / atom();
    return v;
  };
  const sum = () => {
    let v = product();
    while (tokens[i] === "+" || tokens[i] === "-") v = tokens[i++] === "+" ? v + product() : v - product();
    return v;
  };
  return sum();
}

// rl.group_advantages: reward minus the group's mean, over the group's standard deviation.
function groupAdvantages(rewards) {
  const center = mean(rewards);
  const spread = Math.sqrt(rewards.reduce((sum, r) => sum + (r - center) ** 2, 0) / (rewards.length - 1));
  return rewards.map((r) => (r - center) / (spread + 1e-6));
}

// The frame of a chart, with its labels at 22 pixels. Returns { px, py }.
function frame(s, x, y, w, h, across, up) {
  const part = (v, { domain }) => (v - domain[0]) / (domain[1] - domain[0]);
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

// How much of the line through points lies before the place k on the axis that xs are on.
function before(points, xs, k) {
  let total = 0;
  let done = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    total += d;
    done += d * clamp((k - xs[i - 1]) / (xs[i] - xs[i - 1]));
  }
  return done / total;
}

// A short line in a series' color, as a legend's key.
function key(s, x, y, fill) {
  return s.line(x, y - 8, x + 36, y - 8, { stroke: fill, width: 4 });
}

export default ({ summary, run, log, stage9Run, stage9Log, plain, grpo }) => {
  const { iterations, prompts, samples, eval_n: heldOut } = run.args;
  const tuned = summary.runs["gsm8k-tools-sft-gsm8k-tools-lora"]; // 6a's model
  const after = summary.runs["gsm8k-tools-grpo-gsm8k-tools"]; // after GRPO with the tool
  const cut = summary.runs["gsm8k-tools-grpo-gsm8k-tools-no-execute"]; // the same, every call failing
  const compare = (from, to) => Object.values(summary.comparisons).find((c) => c.from === from && c.to === to);
  const form = compare("gsm8k-sft-gsm8k-lora", "gsm8k-tools-sft-gsm8k-tools-lora"); // 6a against stage 6
  const withTool = compare("gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-grpo-gsm8k-tools");
  const without = compare("gsm8k-sft-gsm8k-lora", "gsm8k-grpo-gsm8k");
  const failing = compare("gsm8k-tools-grpo-gsm8k-tools", "gsm8k-tools-grpo-gsm8k-tools-no-execute");
  const failingBefore = compare("gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora-no-execute");
  const calls = Math.round(tuned.calls_per_problem * tuned.n);
  const changed = summary.grpo;

  // the two runs, iteration by iteration
  const steps = log.filter((r) => "reward" in r);
  const steps9 = stage9Log.filter((r) => "reward" in r);
  const evals = log.filter((r) => "held_out" in r);
  const evals9 = stage9Log.filter((r) => "held_out" in r);
  const running = (rows, key) => rows.map((_, j) => mean(rows.slice(Math.max(0, j - 4), j + 1).map((r) => r[key])));
  const third = Math.round(iterations / 3); // the first and the last third of the iterations are compared
  const first = (rows, key) => mean(rows.slice(0, third).map((r) => r[key]));
  const last = (rows, key) => mean(rows.slice(-third).map((r) => r[key]));
  const sameDraw = run.args.seed === stage9Run.args.seed && run.args.prompts === stage9Run.args.prompts;
  const across = { domain: [0, iterations], ticks: [0, third, 2 * third, iterations], format: String, title: "iteration" };

  return {
    subtitle: "The calculator runs while the replies are sampled",
    scenes: [
      {
        seconds: 7,
        caption: "6a learned the form of a call, and solved no more problems.\nFine-tuning showed where calls go, not whether a call helped.",
        draw(s) {
          const head = s.text(70, 136, `6a's model on GSM8K's ${count(tuned.n)} test problems`, { size: 22, fill: "muted" });
          const tiles = [
            [tuned.calls_per_problem.toFixed(1), "calls a problem, on average"],
            [percent(tuned.with_a_call), "of problems with a call"],
            [count(tuned.malformed), `of ${count(calls)} calls malformed`],
          ].map(([number, what], i) => {
            const g = s.group();
            g.text(70, 216 + i * 116, number, { size: 48, weight: 700, fill: "s2" });
            g.text(70, 254 + i * 116, what, { size: 24, fill: "ink2" });
            return g;
          });
          // right: what is solved, with its 95% interval
          const [L, R, BASE, TALL, TOP] = [720, 1190, 460, 280, 0.5];
          const py = (v) => BASE - (TALL * v) / TOP;
          const chart = s.group();
          chart.text(L, 136, "solved, with 95% intervals", { size: 22, fill: "muted" });
          for (const tick of [0.1, 0.2, 0.3, 0.4]) {
            chart.line(L, py(tick), R, py(tick), { stroke: "grid", width: 1.5 });
            chart.text(L - 12, py(tick) + 8, whole(tick), { size: 22, anchor: "end", fill: "muted" });
          }
          chart.line(L, BASE, R, BASE, { stroke: "axis" });
          const bars = [
            { run: plain, name: "stage 6", fill: "s1", x: 780, from: 1.9 },
            { run: tuned, name: "6a, with calls", fill: "s2", x: 990, from: 2.3 },
          ].map((b) => {
            const mid = b.x + 70;
            const bar = s.rect(b.x, BASE, 140, 0, { fill: b.fill, r: 6 });
            const whisker = s.group();
            whisker.line(mid, py(b.run.mean - b.run.error), mid, py(b.run.mean + b.run.error), { stroke: "ink", width: 2 });
            for (const v of [b.run.mean - b.run.error, b.run.mean + b.run.error]) whisker.line(mid - 12, py(v), mid + 12, py(v), { stroke: "ink", width: 2 });
            return {
              ...b,
              bar,
              whisker,
              value: s.text(mid, py(b.run.mean + b.run.error) - 14, percent(b.run.mean), { size: 26, weight: 700, anchor: "middle" }),
              label: s.text(mid, BASE + 32, b.name, { size: 24, weight: 600, anchor: "middle", fill: b.fill }),
            };
          });
          const difference = s.text(L, 548, `difference ${signed(form.difference)} points`, { size: 24, weight: 600 });
          const interval = s.text(L, 580, `95% interval ${signed(form.low)} to ${signed(form.high)}`, { size: 22, fill: "ink2" });
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            tiles.forEach((g, i) => show(g, during(t, 0.5 + i * 0.45, 0.9 + i * 0.45)));
            show(chart, during(t, 1.6, 1.9));
            for (const b of bars) {
              const u = during(t, b.from, b.from + 1.0);
              set(b.bar, { y: BASE - (TALL * b.run.mean * u) / TOP, height: (TALL * b.run.mean * u) / TOP });
              show(b.label, during(t, b.from - 0.2, b.from + 0.1));
              show(b.whisker, during(t, b.from + 1.0, b.from + 1.3));
              show(b.value, during(t, b.from + 1.0, b.from + 1.3));
            }
            show(difference, during(t, 3.8, 4.2));
            show(interval, during(t, 4.0, 4.4));
          };
        },
      },
      {
        seconds: 8.5,
        caption: "A reply is now several turns, the model's and the tool's.\nOnly the model's tokens are trained on, as it sampled them.",
        draw(s) {
          const [X, Y, LEAD] = [100, 140, 31];
          const fills = { prompt: "muted", model: "ink", tool: "muted", result: "s3", again: "ink" };
          const rows = CONVERSATION.map(([string, kind], i) => {
            const g = s.group();
            g.text(X, Y + i * LEAD, string, { size: 22, mono: true, fill: fills[kind], weight: kind === "result" ? 700 : 400 });
            return { g, kind, i };
          });
          const prompt = s.text(X + wide(CONVERSATION[0][0], 22) + 24, Y, "the end of the prompt", { size: 22, fill: "muted" });
          // the call goes to the calculator, and its result comes back as the tool's turn
          const calculator = s.group();
          calculator.rect(960, 246, 240, 84, { fill: "raised", stroke: "axis" });
          calculator.text(1080, 278, "the calculator", { size: 22, anchor: "middle", fill: "ink2" });
          calculator.text(1080, 314, "48/2", { size: 24, anchor: "middle", mono: true });
          const to = s.arrow(X + wide(CONVERSATION[3][0], 22) + 12, Y + 3 * LEAD - 6, 954, 278);
          const FROM = 1060 - X; // where the result starts, from its place in the conversation
          // whose tokens are whose
          const BX = 902;
          const brace = (top, bottom, stroke, title, sub) => {
            const g = s.group();
            const [a, b] = [Y + top * LEAD - 18, Y + bottom * LEAD + 8];
            g.line(BX - 12, a, BX, a, { stroke, width: 3 });
            g.line(BX, a, BX, b, { stroke, width: 3 });
            g.line(BX - 12, b, BX, b, { stroke, width: 3 });
            g.text(BX + 20, (a + b) / 2 - 4, title, { size: 24, weight: 600, fill: stroke === "s2" ? "s2" : "ink2" });
            g.text(BX + 20, (a + b) / 2 + 26, sub, { size: 22, fill: "ink2" });
            return g;
          };
          const marks = [
            brace(1, 4, "s2", "the model's tokens", "marked 1: trained on"),
            brace(5, 9, "axis", "the tool's turn", "marked 0: target −100"),
            brace(10, 10, "s2", "the model's again", "marked 1: trained on"),
          ];
          const notes = [
            s.text(X, 528, "The model's ids are kept as it sampled them; only the tool's turn is encoded.", { size: 22, fill: "ink2" }),
            s.text(X, 562, "rl.batch sets the targets marked 0 to −100, as stage 6 does for the prompt.", { size: 22, fill: "ink2" }),
          ];
          return (t) => {
            for (const row of rows) {
              if (row.kind === "prompt") show(row.g, during(t, 0.2, 0.5));
              else if (row.kind === "model") show(row.g, during(t, 0.6 + (row.i - 1) * 0.35, 0.9 + (row.i - 1) * 0.35));
              else if (row.kind === "tool") show(row.g, during(t, 3.0, 3.4));
              else if (row.kind === "again") show(row.g, during(t, 4.2, 4.6));
              else {
                show(row.g, during(t, 3.0, 3.2));
                place(row.g, FROM * (1 - during(t, 3.1, 3.9)), 0);
              }
            }
            show(prompt, during(t, 0.3, 0.7));
            const asked = during(t, 2.2, 2.6) * (1 - during(t, 4.8, 5.1));
            show(calculator, asked);
            show(to, asked);
            marks.forEach((g, i) => show(g, during(t, 5.2 + i * 0.25, 5.6 + i * 0.25)));
            notes.forEach((g, i) => show(g, during(t, 6.0 + i * 0.25, 6.4 + i * 0.25)));
          };
        },
      },
      {
        seconds: 7,
        caption: `GRPO as stage 9 ran it: ${samples} replies to each of ${prompts} problems.\nThe calculator runs in each. A right final number scores 1.`,
        draw(s) {
          const [SIZE, PAD, GAP] = [22, 7, 22];
          const chipW = (string) => wide(string, SIZE) + 2 * PAD;
          const style = {
            call: { size: SIZE, pad: PAD, fill: "s2", stroke: null, ink: "surface" },
            result: { size: SIZE, pad: PAD, fill: "tint", stroke: "s3", ink: "s3" },
            answer: { size: SIZE, pad: PAD, fill: "raised", stroke: "ink", ink: "ink" },
          };
          // each reply: its calls with the calculator's results, round by round, and then its answer
          const lanes = REPLIES.slice(0, samples).map(([made, answer]) => [...made.map((call) => ({ call, result: String(value(call)) })), { answer: `#### ${answer}`, number: Number(answer) }]);
          const rewards = lanes.map((lane) => (lane.at(-1).number === ANSWER ? 1 : 0));
          const advantages = groupAdvantages(rewards);
          const cellW = (cell) => (cell.call ? chipW(cell.call) + 4 + chipW(cell.result) : chipW(cell.answer));
          const rounds = Math.max(...lanes.map((lane) => lane.length));
          const widths = Array.from({ length: rounds }, (_, k) => Math.max(...lanes.map((lane) => (lane[k] ? cellW(lane[k]) : 0))));
          const X0 = 90;
          const lefts = widths.map((_, k) => X0 + widths.slice(0, k).reduce((a, b) => a + b + GAP, 0));
          const ROW = (i) => 192 + i * 44;
          const begins = (k) => 0.7 + k * 0.7; // the second at which round k is sampled
          const problem = s.chip(70, 100, PROBLEM, { size: 22 });
          const answer = s.text(1210, 128, `GSM8K's answer: ${ANSWER}`, { size: 24, anchor: "end", fill: "ink2" });
          const head = s.text(X0, 176, "sampled replies, every call answered", { size: 22, fill: "muted" });
          const heads = [s.text(680, 176, "reward", { size: 22, anchor: "middle", fill: "muted" }), s.text(880, 176, "advantage", { size: 22, anchor: "end", fill: "muted" }), s.text(912, 176, "after one step", { size: 22, fill: "muted" })];
          const rows = lanes.map((lane, i) => {
            const y = ROW(i);
            const better = advantages[i] > 0;
            const cells = lane.map((cell, k) => {
              if (cell.answer) return { k, said: s.chip(lefts[k], y, cell.answer, style.answer) };
              const said = s.chip(lefts[k], y, cell.call, style.call);
              return { k, said, heard: s.chip(lefts[k] + said.w + 4, y, cell.result, style.result) };
            });
            return {
              cells,
              reward: s.text(680, y + 27, String(rewards[i]), { size: 24, anchor: "middle", mono: true, weight: rewards[i] ? 700 : 400, fill: rewards[i] ? "s3" : "ink2" }),
              advantage: s.text(880, y + 27, `${advantages[i] < 0 ? "−" : "+"}${Math.abs(advantages[i]).toFixed(2)}`, { size: 24, anchor: "end", mono: true, weight: better ? 700 : 400, fill: better ? "s3" : "ink2" }),
              step: s.arrow(925, y + (better ? 32 : 6), 925, y + (better ? 6 : 32), { stroke: better ? "s3" : "muted", width: 3 }),
              word: s.text(945, y + 27, better ? "likelier" : "rarer", { size: 22, fill: better ? "s3" : "muted" }),
            };
          });
          const scale = s.text(640, 578, `one of ${prompts} problems in an iteration, for ${iterations} iterations`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(problem, during(t, 0.2, 0.5));
            show(answer, during(t, 0.3, 0.6));
            show(head, during(t, 0.4, 0.8));
            show(heads[0], during(t, 2.7, 3.0));
            show(heads[1], during(t, 3.6, 3.9));
            show(heads[2], during(t, 4.5, 4.8));
            rows.forEach((row, i) => {
              for (const cell of row.cells) {
                show(cell.said, during(t, begins(cell.k), begins(cell.k) + 0.25));
                if (cell.heard) show(cell.heard, during(t, begins(cell.k) + 0.35, begins(cell.k) + 0.55));
              }
              show(row.reward, during(t, 2.8 + i * 0.08, 3.1 + i * 0.08));
              show(row.advantage, during(t, 3.7 + i * 0.08, 4.0 + i * 0.08));
              show(row.step, during(t, 4.6, 4.9));
              show(row.word, during(t, 4.6, 4.9));
            });
            show(scale, during(t, 4.9, 5.3));
          };
        },
      },
      {
        seconds: 7,
        caption: `Both runs drew the same ${prompts} problems at every iteration.\nThe share right rose alike in both, following the problems.`,
        draw(s) {
          const [X, Y, WIDE, TALL] = [150, 150, 600, 300];
          const at = (k) => X + (WIDE * k) / iterations;
          const bands = [
            [0, third],
            [iterations - third, iterations],
          ].map(([a, b]) => s.rect(at(a), Y, at(b) - at(a), TALL, { fill: "tint", r: 0 }));
          const { px, py } = frame(s, X, Y, WIDE, TALL, across, { domain: [0, 0.8], ticks: [0, 0.2, 0.4, 0.6, 0.8], format: whole, title: "replies that are right: each iteration, and a running mean over 5" });
          const T = 820;
          const runs = [
            { rows: steps9, fill: "s1", name: "stage 9", y: 316 },
            { rows: steps, fill: "s2", name: "with the tool", y: 256 },
          ].map((r) => {
            const xs = r.rows.map((row) => row.iteration);
            const rawAt = r.rows.map((row) => [px(row.iteration), py(row.reward)]);
            const smoothAt = running(r.rows, "reward").map((v, j) => [px(xs[j]), py(v)]);
            const label = s.group();
            key(label, T, r.y, r.fill);
            label.text(T + 48, r.y, r.name, { size: 24, weight: 600 });
            return {
              ...r,
              xs,
              rawAt,
              smoothAt,
              raw: s.path(rawAt, { stroke: r.fill, width: 1.5 }),
              smooth: s.path(smoothAt, { stroke: r.fill, width: 4 }),
              label,
              early: s.text(1120, r.y + 2, whole(first(r.rows, "reward")), { size: 28, weight: 700, anchor: "end", fill: r.fill }),
              late: s.text(1205, r.y + 2, whole(last(r.rows, "reward")), { size: 28, weight: 700, anchor: "end", fill: r.fill }),
            };
          });
          const heads = [s.text(T, 170, "mean over the iterations", { size: 22, fill: "muted" }), s.text(1120, 206, `first ${third}`, { size: 22, anchor: "end", fill: "muted" }), s.text(1205, 206, `last ${third}`, { size: 22, anchor: "end", fill: "muted" })];
          const same = s.group();
          if (sameDraw) same.lines(T, 410, [`Both drew their ${prompts} problems`, `from seed ${run.args.seed}, in the same order.`], { size: 22, fill: "ink2" });
          return (t) => {
            const k = iterations * during(t, 0.6, 3.6);
            for (const r of runs) {
              show(r.label, during(t, 0.3, 0.7));
              trace(r.raw, before(r.rawAt, r.xs, k));
              trace(r.smooth, before(r.smoothAt, r.xs, k));
              show(r.raw, 0.4 * during(t, 0.6, 0.7));
              show(r.smooth, during(t, 0.6, 0.7));
              show(r.early, during(t, 3.8, 4.2));
              show(r.late, during(t, 4.2, 4.6));
            }
            show(bands[0], during(t, 3.8, 4.2));
            show(bands[1], during(t, 4.2, 4.6));
            show(heads[0], during(t, 3.6, 4.0));
            show(heads[1], during(t, 3.8, 4.2));
            show(heads[2], during(t, 4.2, 4.6));
            show(same, during(t, 4.8, 5.2));
          };
        },
      },
      {
        seconds: 6.5,
        caption: `On ${heldOut} held-out problems stage 9's run rose, this one did not.\nOn ${heldOut} problems, either change could be noise.`,
        draw(s) {
          const [X, Y, WIDE, TALL] = [150, 150, 560, 300];
          const { px, py } = frame(s, X, Y, WIDE, TALL, across, { domain: [0.4, 0.56], ticks: [0.4, 0.45, 0.5, 0.55], format: whole, title: `solved, of ${heldOut} held-out problems` });
          const T = 830;
          const runs = [
            { rows: evals9, fill: "s1", name: "stage 9, without the tool", y: 214 },
            { rows: evals, fill: "s2", name: "with the tool", y: 360 },
          ].map((r) => {
            const xs = r.rows.map((row) => row.iteration);
            const pts = r.rows.map((row) => [px(row.iteration), py(row.held_out)]);
            const [a, b] = [r.rows[0].held_out, r.rows.at(-1).held_out];
            const moved = Math.round(Math.abs(b - a) * heldOut);
            const label = s.group();
            key(label, T, r.y, r.fill);
            label.text(T + 48, r.y, r.name, { size: 24, weight: 600 });
            const values = s.group();
            values.text(T, r.y + 40, `${percent(a)} → ${percent(b)}`, { size: 28, weight: 700, fill: r.fill });
            values.text(T, r.y + 74, `${count(moved)} ${b > a ? "more" : "fewer"} of the ${heldOut} solved`, { size: 22, fill: "ink2" });
            return {
              ...r,
              xs,
              pts,
              label,
              values,
              line: s.path(pts, { stroke: r.fill, width: 4 }),
              dots: pts.map(([x, y]) => s.circle(x, y, 7, { fill: r.fill })),
              end: s.text(px(iterations) + 14, py(b) + 9, percent(b), { size: 26, weight: 700, fill: r.fill }),
            };
          });
          return (t) => {
            const k = iterations * during(t, 0.6, 2.8);
            for (const r of runs) {
              show(r.label, during(t, 0.3, 0.7));
              trace(r.line, before(r.pts, r.xs, k));
              show(r.line, during(t, 0.6, 0.7));
              r.dots.forEach((dot, i) => show(dot, t > 0.4 && k >= r.xs[i] ? 1 : 0));
              show(r.end, during(t, 2.8, 3.1));
              show(r.values, during(t, 3.0, 3.4));
            }
          };
        },
      },
      {
        seconds: 7.5,
        caption: `With the tool, GRPO did not move the test score: ${signed(withTool.difference)} points,\nwithin noise. The same recipe gained stage 9's model ${points(without.difference)}.`,
        draw(s) {
          const FLOOR = 466;
          const tall = (v) => 560 * v;
          const pairs = [
            { name: "without the tool", fill: "s1", from: plain, to: grpo, names: ["stage 6", "after GRPO"], x: 100, at: 0.3 },
            { name: "with the tool", fill: "s2", from: tuned, to: after, names: ["6a", "after GRPO"], x: 440, at: 0.9 },
          ].map((p) => {
            const group = s.text(p.x + 135, FLOOR + 70, p.name, { size: 24, weight: 600, anchor: "middle", fill: p.fill });
            const bars = [p.from, p.to].map((result, j) => {
              const x = p.x + 150 * j;
              const mid = x + 60;
              s.text(mid, FLOOR + 34, p.names[j], { size: 22, anchor: "middle", fill: "ink2" });
              const over = s.group();
              over.line(mid, FLOOR - tall(result.mean + result.error), mid, FLOOR - tall(result.mean - result.error), { stroke: "ink", width: 2.5 });
              over.text(mid, FLOOR - tall(result.mean) - 54, percent(result.mean), { size: 28, weight: 700, anchor: "middle" });
              over.text(mid, FLOOR - tall(result.mean) - 28, `± ${percent(result.error)}`, { size: 22, anchor: "middle", fill: "muted" });
              return { result, over, bar: s.rect(x, FLOOR, 120, 0, { fill: p.fill, r: 6 }) };
            });
            // what GRPO added, as the top of its bar, over the level it started from
            const added = s.rect(p.x + 150, FLOOR - tall(p.from.mean), 120, 0, { fill: "s3", r: 0 });
            const level = s.line(p.x, FLOOR - tall(p.from.mean), p.x + 270, FLOOR - tall(p.from.mean), { stroke: "muted", width: 1.5 });
            // the whiskers lie over the bars, and the level over both
            for (const b of bars) b.over.g.parentNode.appendChild(b.over.g);
            level.parentNode.appendChild(level);
            return { ...p, group, bars, added, level, gain: p.to.mean - p.from.mean };
          });
          s.line(80, FLOOR, 720, FLOOR, { stroke: "axis" });
          const R = 780;
          const head = s.text(R, 160, "GRPO's change, 95% interval", { size: 22, fill: "muted" });
          const changes = [
            [without, "s1", "without the tool", ""],
            [withTool, "s2", "with it", ": within noise"],
          ].map(([c, fill, which, verdict], i) => {
            const g = s.group();
            g.text(R, 228 + i * 110, `${signed(c.difference)} points`, { size: 40, weight: 700, fill });
            g.text(R, 262 + i * 110, `${signed(c.low)} to ${signed(c.high)}, ${which}${verdict}`, { size: 22, fill: "ink2" });
            return g;
          });
          const turned = s.group();
          turned.text(R, 452, "problems turned right, and wrong", { size: 22, fill: "muted" });
          turned.text(R, 486, `${count(changed["stage 9"].gained)} and ${count(changed["stage 9"].lost)}, without the tool`, { size: 22, fill: "ink2" });
          turned.text(R, 518, `${count(changed["with the tool"].gained)} and ${count(changed["with the tool"].lost)}, with it`, { size: 22, fill: "ink2" });
          return (t) => {
            const up = during(t, 2.6, 3.2);
            for (const p of pairs) {
              show(p.group, during(t, p.at - 0.2, p.at + 0.2));
              const [start, end] = p.bars;
              const h = tall(p.from.mean) * during(t, p.at, p.at + 0.9);
              set(start.bar, { y: FLOOR - h, height: h });
              show(start.over, during(t, p.at + 0.8, p.at + 1.1));
              const rise = tall(p.from.mean) * during(t, p.at + 0.5, p.at + 1.4) + tall(p.gain) * up;
              set(end.bar, { y: FLOOR - rise, height: rise });
              set(p.added, { y: FLOOR - tall(p.from.mean) - tall(p.gain) * up, height: tall(p.gain) * up });
              show(p.level, during(t, 2.3, 2.6));
              show(end.over, during(t, 3.2, 3.5));
            }
            show(head, during(t, 3.4, 3.7));
            changes.forEach((g, i) => show(g, during(t, 3.5 + i * 0.4, 3.9 + i * 0.4)));
            show(turned, during(t, 4.5, 4.9));
          };
        },
      },
      {
        seconds: 7,
        caption: `It learned to call less: ${after.calls_per_problem.toFixed(2)} calls a problem, from ${tuned.calls_per_problem.toFixed(2)}.\nWith every call failing it solves ${percent(cut.mean)}, ${points(failing.difference)} points less.`,
        draw(s) {
          // left: what changed in its calls and its replies
          const [X, FULL] = [70, 330];
          const head = s.text(X, 140, "6a's model, before and after GRPO", { size: 22, fill: "muted" });
          const measures = [
            { name: "calls per test problem", from: tuned.calls_per_problem, to: after.calls_per_problem, full: 4, format: (v) => v.toFixed(2) },
            { name: "test problems with a call", from: tuned.with_a_call, to: after.with_a_call, full: 1, format: percent },
            {
              name: "its own tokens per reply, in training",
              from: first(steps, "tokens"),
              to: last(steps, "tokens"),
              full: 200,
              format: (v) => v.toFixed(0),
              sub: `first and last ${third} iterations; stage 9's: ${first(steps9, "tokens").toFixed(0)} → ${last(steps9, "tokens").toFixed(0)}`,
            },
          ].map((m, i) => {
            const y = 196 + i * 110;
            const g = s.group();
            g.text(X, y, m.name, { size: 22, fill: "ink2" });
            g.rect(X, y + 14, (FULL * m.from) / m.full, 36, { fill: "tint", r: 6 });
            if (m.sub) g.text(X, y + 80, m.sub, { size: 22, fill: "muted" });
            return { ...m, g, bar: s.rect(X, y + 14, 0, 36, { fill: "s2", r: 6 }), value: s.text(X + FULL + 18, y + 41, "", { size: 24, weight: 600 }) };
          });
          // right: what is solved when every call fails
          const [RX, RFULL, TOP] = [720, 400, 0.5];
          const right = s.text(RX, 140, `solved after GRPO, of ${count(after.n)} test problems`, { size: 22, fill: "muted" });
          const rows = [
            { name: "calls run", fill: "s2", y: 196 },
            { name: "every call failing", fill: "axis", y: 296 },
          ].map((r) => ({ ...r, label: s.text(RX, r.y, r.name, { size: 22, fill: "ink2" }), bar: s.rect(RX, r.y + 14, 0, 40, { fill: r.fill, r: 6 }), value: s.text(RX, r.y + 44, "", { size: 26, weight: 700 }) }));
          const bar = (r, share) => {
            const w = (RFULL * share) / TOP;
            set(r.bar, { width: w });
            set(r.value, { x: RX + w + 12 });
            r.value.textContent = percent(share);
          };
          const less = s.group();
          less.text(RX, 430, `${points(failing.difference)} points less`, { size: 40, weight: 700 });
          less.text(RX, 466, `95% interval ${points(failing.high)} to ${points(failing.low)}`, { size: 22, fill: "ink2" });
          less.text(RX, 500, `before GRPO, ${points(failingBefore.difference)} points less`, { size: 22, fill: "muted" });
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            measures.forEach((m, i) => {
              show(m.g, during(t, 0.3 + i * 0.4, 0.7 + i * 0.4));
              show(m.value, during(t, 0.3 + i * 0.4, 0.7 + i * 0.4));
              show(m.bar, during(t, 0.3 + i * 0.4, 0.7 + i * 0.4));
              const now = mix(m.from, m.to, during(t, 1.0 + i * 0.45, 2.0 + i * 0.45));
              set(m.bar, { width: (FULL * now) / m.full });
              m.value.textContent = `${m.format(m.from)} → ${m.format(now)}`;
            });
            show(right, during(t, 2.6, 3.0));
            const [run_, fail] = rows;
            show(run_.label, during(t, 2.7, 3.0));
            show(run_.value, during(t, 2.8, 3.1));
            bar(run_, after.mean * during(t, 2.8, 3.5));
            show(fail.label, during(t, 3.5, 3.8));
            show(fail.bar, during(t, 3.5, 3.7));
            show(fail.value, during(t, 3.5, 3.8));
            bar(fail, mix(after.mean, cut.mean, during(t, 3.8, 4.7)));
            show(less, during(t, 4.8, 5.2));
          };
        },
      },
    ],
  };
};
