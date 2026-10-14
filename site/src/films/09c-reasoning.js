// Side topic 9c as a film: reasoning, thinking longer to be right more
// often. A reasoning model's long replies against stage 9's short ones; GRPO
// from the published model with room to think; votes over samples and
// thinking cut at a budget, without training; and a 1.5B reasoning model's
// thinking imitated by the 0.5B model, which learns to think at length and
// not to stop. The problem of the vote and its answer are GSM8K's; its
// sixteen sampled answers, the replies quoted, every curve and every score
// are those logged in runs/reasoning/, runs/rlvr/ and runs/sft-gsm8k-traces/.

import { svg } from "../labs/kit.js";
import { clamp, during, mix, place, set, show, trace } from "./stage.js";

export const sources = {
  summary: "/runs/reasoning/summary.json",
  published: "/runs/reasoning/test-time-qwen.json",
  tuned: "/runs/reasoning/test-time-grpo-qwen.json",
  teacher: "/runs/reasoning/teacher-train.json",
  run: "/runs/rlvr/grpo-qwen.json",
  log: "/runs/rlvr/grpo-qwen.jsonl",
  stage9Run: "/runs/rlvr/grpo-gsm8k.json",
  stage9Log: "/runs/rlvr/grpo-gsm8k.jsonl",
  studentLog: "/runs/sft-gsm8k-traces/log.jsonl",
  studentTimes: "/runs/reasoning/test-time-sft-gsm8k-traces.json",
};

// The 19th test problem of GSM8K and its answer, whose sixteen sampled answers the vote is shown on.
const VOTED = { index: 18, question: ["Claire makes a 3 egg omelet every morning for breakfast.", "How many dozens of eggs will she eat in 4 weeks?"], answer: 7 };
const TRACES = 601; // the traces the student was fine-tuned on, as the note states; the rest of those kept measured its loss
const WINDOW = 20; // the first and the last iterations of GRPO compared, as the note compares them

const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js
const count = (v) => Math.round(v).toLocaleString("en");
const percent = (v, digits = 1) => `${(100 * v).toFixed(digits)}%`;
const points = (v) => Math.abs(100 * v).toFixed(1);
const signed = (v) => `${v < 0 ? "−" : "+"}${points(v)}`;
const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
const running = (rows, key) => rows.map((_, j) => mean(rows.slice(Math.max(0, j - 4), j + 1).map((r) => r[key])));

// reasoning.majority: the most common answer, ties going to the one given first; null if none was read.
function majority(numbers) {
  const counted = new Map();
  for (const n of numbers) if (n !== null) counted.set(n, (counted.get(n) || 0) + 1);
  if (!counted.size) return null;
  const top = Math.max(...counted.values());
  return numbers.find((n) => counted.get(n) === top);
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

// How much of the line through points lies before the place k on the axis that xs are on.
function before(points, xs, k) {
  let total = 0;
  let done = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    total += d;
    done += d * clamp((k - xs[i - 1]) / (xs[i] - xs[i - 1]));
  }
  return total ? done / total : 1;
}

// A line through points, redrawn as given, so that a dashed line can grow; hidden when there is nothing to draw.
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

// A short line in a series' color, as a legend's key; dashed if asked.
function key(s, x, y, stroke, dash = null) {
  const line = s.line(x, y - 8, x + 36, y - 8, { stroke, width: 4 });
  if (dash) line.setAttribute("stroke-dasharray", dash);
  return line;
}

// A line of text in pieces, each with its own fill and weight: [[string, fill, weight]].
function spans(s, x, y, parts, { size = 22, mono = false, anchor = "start" } = {}) {
  const node = svg("text", { x, y, "font-size": size, "text-anchor": anchor, class: mono ? "film-mono" : "" }, s.g);
  for (const [string, fill = "ink", weight = 400] of parts) {
    const span = svg("tspan", { class: `f-${fill}`, "font-weight": weight }, node);
    span.textContent = string;
  }
  return node;
}

function wrap(text, width) {
  const lines = [""];
  for (const word of text.split(/\s+/)) {
    const last = lines[lines.length - 1];
    if (last && last.length + 1 + word.length > width) lines.push(word);
    else lines[lines.length - 1] = last ? `${last} ${word}` : word;
  }
  return lines;
}

export default ({ summary, published, tuned, teacher, run, log, stage9Run, stage9Log, studentLog, studentTimes }) => {
  const { iterations, max_new_tokens: limit } = run.args;
  const limit9 = stage9Run.args.max_new_tokens;
  const steps = log.filter((r) => "reward" in r);
  const steps9 = stage9Log.filter((r) => "reward" in r);
  const first = (key) => mean(steps.slice(0, WINDOW).map((r) => r[key]));
  const last = (key) => mean(steps.slice(-WINDOW).map((r) => r[key]));

  const runs = summary.runs;
  const base = runs["gsm8k-qwen"]; // the published 0.5B model
  const grpo = runs["gsm8k-grpo-qwen"]; // after GRPO with room to think
  const student = runs["gsm8k-sft-gsm8k-traces"]; // fine-tuned on the teacher's thinking
  const compare = (from, to) => Object.values(summary.comparisons).find((c) => c.from === from && c.to === to);
  const gain = compare("gsm8k-qwen", "gsm8k-grpo-qwen");

  // the student on the first test problems: its greedy reply, the same cut at the largest budget and made to answer, and the vote of its samples
  const n = published.n;
  const CUT = Math.max(...Object.keys(studentTimes.budgets).map(Number));
  const voted = studentTimes.majority[studentTimes.ks.indexOf(studentTimes.samples)];

  return {
    subtitle: "Thinking longer, to be right more often",
    scenes: [
      {
        seconds: 7,
        caption: "Reasoning models write a long chain of thought first.\nStage 9 used their method, GRPO. Its replies stayed short.",
        draw(s) {
          const MAX = teacher.max_new_tokens;
          const [X, WIDE] = [150, 1000];
          const at = (v) => X + (WIDE * v) / MAX;
          const axis = s.group();
          axis.text(X, 130, "tokens per reply", { size: 22, fill: "muted" });
          for (const tick of [0, limit9, limit, 1024, MAX]) {
            axis.text(at(tick), 166, count(tick), { size: 22, anchor: "middle", fill: "muted" });
            axis.line(at(tick), 212, at(tick), 262, { stroke: "grid", width: 1.5 });
          }
          // a reasoning model: its replies to the training problems, and words from one of them
          const thinker = s.text(X, 204, `a reasoning model, 1.5B, on ${count(teacher.n)} training problems`, { size: 24, weight: 600 });
          const long = s.rect(X, 216, 0, 40, { fill: "s1", r: 6 });
          const think = s.text(X + 12, 243, "<think>", { size: 22, mono: true, fill: "surface" });
          const longValue = s.text(at(teacher.tokens_mean) + 14, 245, `${count(teacher.tokens_mean)} on average`, { size: 24, weight: 600, fill: "s1" });
          const sample = teacher.samples.find((r) => r.closed && r.right) ?? teacher.samples[0];
          const sentences = sample.text.split("</think>")[0].split(/(?<=[.?!])\s+/).map((x) => x.trim());
          const clip = (x, most = 68) => (x.length > most ? `${x.slice(0, x.lastIndexOf(" ", most - 2)).replace(/[,;:]$/, "")} …` : x);
          const asks = (x) => x.endsWith("?") && x.length < 68;
          const quotes = [sentences[0], sentences.find((x) => x.startsWith("Wait")), sentences.find((x) => /another way/.test(x) && asks(x)) ?? sentences.find(asks)].filter(Boolean).map((x) => clip(x));
          quotes.push(`</think> … \\boxed{${sample.number}}`);
          const from = s.text(X, 306, "from one of its replies", { size: 22, fill: "muted" });
          const said = quotes.map((q, i) => s.text(X, 340 + i * 32, q, { size: 22, mono: true, fill: i === quotes.length - 1 ? "s1" : "ink2" }));
          // stage 9: the tokens of its sampled replies, iteration by iteration
          const terse = s.group();
          terse.text(X, 476, "stage 9: GRPO from a model fine-tuned to be terse", { size: 24, weight: 600 });
          const short = s.rect(X, 488, 0, 40, { fill: "muted", r: 6 });
          terse.line(at(limit9), 482, at(limit9), 536, { stroke: "ink", width: 3 });
          const shortValue = s.text(at(limit9) + 16, 516, "", { size: 24, weight: 600, fill: "ink2" });
          const which = s.text(at(limit9) + 16, 560, "", { size: 22, fill: "muted" });
          const average9 = mean(steps9.map((r) => r.tokens));
          return (t) => {
            show(axis, during(t, 0.2, 0.6));
            show(thinker, during(t, 0.3, 0.7));
            const grown = during(t, 0.6, 3.0);
            set(long, { width: (at(teacher.tokens_mean) - X) * grown });
            show(think, grown > 0.15 ? 1 : 0);
            show(longValue, during(t, 2.8, 3.2));
            show(from, during(t, 0.9, 1.3));
            said.forEach((line, i) => show(line, during(t, 1.0 + i * 0.55, 1.4 + i * 0.55)));
            show(terse, during(t, 3.1, 3.5));
            const k = steps9.length * during(t, 3.5, 4.9);
            const j = clamp(Math.ceil(k) - 1, 0, steps9.length - 1);
            const settled = during(t, 4.9, 5.2);
            const tokens = mix(steps9[j].tokens, average9, settled);
            set(short, { width: (at(tokens) - X) * during(t, 3.4, 3.6) });
            show(shortValue, during(t, 3.5, 3.8));
            show(which, during(t, 3.5, 3.8));
            shortValue.textContent = `${count(tokens)} tokens per sampled reply`;
            which.textContent = settled > 0.5 ? `on average over ${steps9.length} iterations, a limit of ${limit9}` : `iteration ${j + 1} of ${steps9.length}, a limit of ${limit9}`;
          };
        },
      },
      {
        seconds: 7.5,
        caption: `With room for ${limit} tokens, the replies did not grow longer.\nOne run of GRPO gained ${points(gain.difference)} points, at the edge of noise.`,
        draw(s) {
          const [X, Y, WIDE, TALL] = [150, 170, 540, 300];
          const { px, py } = frame(
            s,
            X,
            Y,
            WIDE,
            TALL,
            { domain: [0, iterations], ticks: [0, iterations / 3, (2 * iterations) / 3, iterations], format: String, title: "iteration, a running mean over 5" },
            { domain: [0, limit], ticks: [0, limit / 4, limit / 2, (3 * limit) / 4, limit], format: String, title: "tokens per sampled reply" },
          );
          const ceiling = s.line(X, py(limit), X + WIDE, py(limit), { stroke: "s2", width: 2.5 });
          ceiling.setAttribute("stroke-dasharray", "8 6");
          s.text(X + WIDE, Y - 16, `limit ${limit}`, { size: 22, anchor: "end", fill: "s2" });
          const ninth = s.group();
          const floor9 = ninth.line(X, py(limit9), px(steps9.length), py(limit9), { stroke: "muted", width: 2.5 });
          floor9.setAttribute("stroke-dasharray", "8 6");
          ninth.text(X + 12, py(limit9) + 30, `stage 9's limit, ${limit9}`, { size: 22, fill: "muted" });
          const lines = [
            { rows: steps, fill: "s2", width: 4 },
            { rows: steps9, fill: "muted", width: 3.5 },
          ].map((l) => {
            const xs = l.rows.map((r) => r.iteration);
            const at = running(l.rows, "tokens").map((v, j) => [px(xs[j]), py(v)]);
            return { xs, at, path: s.path(at, { stroke: l.fill, width: l.width }) };
          });
          const R = 780;
          const legend = s.group();
          key(legend, R, 186, "s2");
          legend.text(R + 48, 186, "from the published model", { size: 24, weight: 600 });
          key(legend, R, 222, "muted");
          legend.text(R + 48, 222, "stage 9, from the fine-tuned model", { size: 22, fill: "ink2" });
          const head = s.text(R, 284, `the first and the last ${WINDOW} iterations`, { size: 22, fill: "muted" });
          const told = [
            [`${count(first("tokens"))} → ${count(last("tokens"))}`, "tokens per sampled reply", 326],
            [`${percent(first("finished"), 0)} → ${percent(last("finished"), 0)}`, `of the replies ended within ${limit}`, 410],
          ].map(([value, what, y]) => {
            const g = s.group();
            g.text(R, y, value, { size: 34, weight: 700, fill: "s2" });
            g.text(R, y + 30, what, { size: 22, fill: "ink2" });
            return g;
          });
          const test = s.group();
          test.text(R, 494, `on ${count(base.n)} test problems, greedy`, { size: 22, fill: "muted" });
          spans(test, R, 532, [[percent(base.mean), "ink2", 700], [" → ", "ink", 700], [percent(grpo.mean), "s2", 700]], { size: 34 });
          test.text(R, 564, `${signed(gain.difference)} points, 95% interval ${signed(gain.low)} to ${signed(gain.high)}`, { size: 22, fill: "ink2" });
          return (t) => {
            show(legend, during(t, 0.3, 0.7));
            show(ninth, during(t, 0.5, 0.9));
            const k = iterations * during(t, 0.6, 3.6);
            for (const l of lines) {
              trace(l.path, before(l.at, l.xs, k));
              show(l.path, during(t, 0.6, 0.7));
            }
            show(head, during(t, 3.7, 4.0));
            told.forEach((g, i) => show(g, during(t, 3.8 + i * 0.5, 4.2 + i * 0.5)));
            show(test, during(t, 5.0, 5.4));
          };
        },
      },
      {
        seconds: 8,
        caption: `${published.samples} replies are sampled, and the most common answer is taken.\nAt least one of ${published.samples} is right far more often than the vote.`,
        draw(s) {
          const answers = published.numbers[VOTED.index];
          const name = (v) => (v === null ? "none" : String(v));
          const counted = new Map();
          for (const v of answers) counted.set(name(v), (counted.get(name(v)) || 0) + 1);
          const order = [...counted.keys()].sort((a, b) => counted.get(b) - counted.get(a)); // ties stay in the order first given
          const vote = majority(answers);
          const voteRight = vote === VOTED.answer;
          const problem = s.group();
          problem.lines(70, 132, VOTED.question, { size: 22, fill: "ink2", leading: 1.3 });
          const head = s.text(70, 210, `${answers.length} replies at temperature ${published.temperature}, their answers`, { size: 22, fill: "muted" });
          const ROW = (r) => 226 + r * 48;
          const seen = new Map();
          const chips = answers.map((v, i) => {
            const label = name(v);
            const j = seen.get(label) || 0;
            seen.set(label, j + 1);
            const chip = s.chip(0, 0, label, { size: 22, pad: 10 });
            const right = v === VOTED.answer;
            return { chip, right, grid: [70 + (i % 4) * 120, ROW(Math.floor(i / 4))], tally: [150 + j * (chip.w + 6), ROW(order.indexOf(label))] };
          });
          const tallies = order.map((label, r) => s.text(130, ROW(r) + 26, `${counted.get(label)}×`, { size: 24, weight: 700, anchor: "end" }));
          const topW = chips.filter((c) => name(c.chip.label.textContent) === order[0]).reduce((sum, c) => sum + c.chip.w + 6, 0);
          const marker = s.text(150 + topW + 12, ROW(0) + 26, `the vote, ${voteRight ? "right" : "wrong"}`, { size: 24, weight: 700, fill: voteRight ? "s3" : "s2" });
          const verdicts = s.group();
          verdicts.text(70, 500, `GSM8K's answer: ${VOTED.answer}`, { size: 22, fill: "ink2" });
          verdicts.text(70, 532, `the greedy reply to this problem: ${published.greedy.scores[VOTED.index] ? "right" : "wrong"}`, { size: 22, fill: "ink2" });

          // the vote and at least one right, over k samples, for both models
          const [X, Y, WIDE, TALL] = [720, 150, 300, 270];
          const chart = s.group();
          const { px, py } = frame(chart, X, Y, WIDE, TALL, { domain: [1, published.samples], log: true, ticks: published.ks, format: String, title: "samples per problem" }, { domain: [0.3, 0.9], ticks: [0.3, 0.5, 0.7, 0.9], format: (v) => percent(v, 0), title: `right, of the first ${n} problems` });
          const ends = [];
          const series = [
            [published, "ink2"],
            [tuned, "s2"],
          ].flatMap(([m, fill]) => {
            const greedy = chart.line(px(1), py(m.greedy.right), px(published.samples), py(m.greedy.right), { stroke: fill, width: 2.5 });
            greedy.setAttribute("stroke-dasharray", "2 6");
            return ["majority", "pass"].map((curve) => {
              const at = m.ks.map((k, i) => [px(k), py(m[curve][i])]);
              const path = s.path(at, { stroke: fill, width: curve === "pass" ? 3 : 3.5 });
              if (curve === "pass") path.setAttribute("stroke-dasharray", "9 6");
              const end = s.text(X + WIDE + 12, 0, percent(m[curve].at(-1)), { size: 24, weight: 700, fill });
              ends.push({ end, y: at.at(-1)[1] });
              return { at, path, end };
            });
          });
          // the labels at the end, kept apart
          ends.sort((a, b) => a.y - b.y);
          ends.forEach((e, i) => {
            if (i > 0) e.y = Math.max(e.y, ends[i - 1].y + 27);
            set(e.end, { y: e.y + 8 });
          });
          const legend = s.group();
          key(legend, X, 528, "ink2");
          legend.text(X + 46, 528, "as published", { size: 22, fill: "ink2" });
          key(legend, X + 200, 528, "s2");
          legend.text(X + 246, 528, "after GRPO", { size: 22, fill: "ink2" });
          key(legend, X, 564, "ink");
          legend.text(X + 46, 564, "the vote", { size: 22, fill: "ink2" });
          key(legend, X + 150, 564, "ink", "9 6");
          legend.text(X + 196, 564, "at least one", { size: 22, fill: "ink2" });
          key(legend, X + 340, 564, "ink", "2 6");
          legend.text(X + 386, 564, "greedy", { size: 22, fill: "ink2" });
          return (t) => {
            show(problem, during(t, 0.2, 0.5));
            show(head, during(t, 0.4, 0.7));
            const sorted = during(t, 2.4, 3.4);
            chips.forEach((c, i) => {
              show(c.chip, during(t, 0.6 + i * 0.1, 0.8 + i * 0.1));
              place(c.chip, mix(c.grid[0], c.tally[0], sorted), mix(c.grid[1], c.tally[1], sorted));
              const marked = c.right && t > 3.5;
              c.chip.ground.setAttribute("class", marked ? "f-s3" : "f-tint s-axis");
              c.chip.label.setAttribute("class", `film-mono ${marked ? "f-surface" : "f-ink"}`);
            });
            tallies.forEach((g) => show(g, during(t, 3.3, 3.6)));
            show(marker, during(t, 3.5, 3.8));
            show(verdicts, during(t, 3.7, 4.0));
            show(chart, during(t, 3.9, 4.3));
            show(legend, during(t, 3.9, 4.3));
            const x = mix(px(1), px(published.samples), during(t, 4.3, 5.8));
            for (const line of series) {
              shape(line.path, t > 4.3 ? upto(line.at, x) : []);
              show(line.end, during(t, 5.8, 6.2));
            }
          };
        },
      },
      {
        seconds: 7,
        caption: "Neither model writes its thinking apart from its answer.\nA cut at a budget falls in the middle of a worked solution.",
        draw(s) {
          const MAX = published.max_new_tokens;
          const [X, WIDE] = [150, 980];
          const at = (v) => X + (WIDE * v) / MAX;
          // the published model's greedy reply to the first problem, in its numbered steps; a step's tokens are taken as its share of the characters
          const shown = published.samples_shown.find((r) => r.problem === 0);
          const length = published.greedy.lengths[0];
          const text = shown.reply;
          const starts = [0, ...[...text.matchAll(/\n\n(?:\d+\.|Thus|So,|Therefore)/g)].map((m) => m.index), text.length];
          const boxed = (text.match(/\\boxed\{([^}]*)\}/) || [])[1] ?? "";
          const head = s.text(X, 126, `the published model's greedy reply to the first test problem, ${count(length)} tokens`, { size: 22, fill: "muted" });
          const parts = starts.slice(0, -1).map((a, i) => {
            const [x0, x1] = [at((length * a) / text.length), at((length * starts[i + 1]) / text.length)];
            const g = s.group();
            g.rect(x0 + 1.5, 140, x1 - x0 - 3, 40, { fill: "ink2", r: 4 });
            const step = /^\n\n(\d+)\./.exec(text.slice(a, a + 6));
            const label = step ? step[1] : i === starts.length - 2 ? boxed : "";
            if (label) g.text((x0 + x1) / 2, 168, label, { size: 22, weight: 600, anchor: "middle", fill: "surface" });
            return g;
          });
          const end = at(length);
          const veil = s.rect(end, 136, 0, 48, { fill: "surface", r: 0 });
          veil.setAttribute("fill-opacity", 0.75);
          const cue = s.chip(0, 192, "####", { size: 22, pad: 8, fill: "raised", stroke: "ink" });
          const told = s.text(0, 218, "", { size: 22, fill: "ink2" });

          // what is right at each budget, for both models; the cut stops at every budget below the limit, and the last point is the reply uncut
          const budgets = Object.keys(published.budgets).map(Number).sort((a, b) => a - b);
          const stops = budgets.filter((b) => b < MAX);
          const [Y, TALL] = [280, 200];
          const ticks = budgets.filter((b, i) => i === 0 || at(b) - at(budgets[i - 1]) > 44);
          const { py } = frame(s, X, Y, WIDE, TALL, { domain: [0, MAX], ticks, format: (b) => (b >= MAX ? `${b}, uncut` : String(b)), title: "tokens allowed, then the answer asked for" }, { domain: [0, 0.5], ticks: [0, 0.2, 0.4], format: (v) => percent(v, 0), title: `right, of the first ${n} problems` });
          const models = [
            [published, "ink2", 7],
            [tuned, "s2", 5],
          ].map(([m, fill, r]) => {
            const pts = budgets.map((b) => [at(b), py(m.budgets[b].right)]);
            return { pts, path: s.path(pts, { stroke: fill, width: 3 }), dots: pts.map(([x, y]) => s.circle(x, y, r, { fill })) };
          });
          // the scores at the larger budgets, both models side by side
          const marks = budgets
            .filter((b) => b >= 128)
            .map((b) => ({ b, label: spans(s, at(b) - 14, Math.min(py(published.budgets[b].right), py(tuned.budgets[b].right)) - 16, [[percent(published.budgets[b].right), "ink2", 700], ["  "], [percent(tuned.budgets[b].right), "s2", 700]], { size: 22, anchor: "end" }) }));
          const legend = s.group();
          key(legend, 760, 430, "ink2");
          legend.text(806, 430, "as published", { size: 22, fill: "ink2" });
          key(legend, 760, 462, "s2");
          legend.text(806, 462, "after GRPO", { size: 22, fill: "ink2" });
          // the cut, on the reply and on the chart
          const blade = s.line(0, 134, 0, 188, { stroke: "ink", width: 3 });
          const marker = s.line(0, Y, 0, Y + TALL, { stroke: "ink", width: 2 });
          marker.setAttribute("stroke-dasharray", "4 5");

          const T0 = 0.8;
          const STEP = 0.75;
          const when = (b) => (b < MAX ? T0 + stops.indexOf(b) * STEP : T0 + (stops.length - 1) * STEP + 0.9);
          return (t) => {
            show(head, during(t, 0.2, 0.5));
            parts.forEach((g, i) => show(g, during(t, 0.2 + i * 0.08, 0.5 + i * 0.08)));
            show(legend, during(t, 0.4, 0.8));
            let v = stops[0];
            for (let i = 1; i < stops.length; i++) v = mix(v, stops[i], during(t, T0 + i * STEP - 0.3, T0 + i * STEP));
            const x = at(v);
            const on = during(t, T0 - 0.3, T0);
            for (const line of [blade, marker]) {
              set(line, { x1: x, x2: x });
              show(line, on);
            }
            set(veil, { x, width: Math.max(0, end - x) });
            show(veil, on);
            const cut = v < length;
            place(cue, x + 8, 192);
            show(cue, cut ? on : 0);
            set(told, { x: cut ? x + 8 + cue.w + 10 : x - 14, "text-anchor": cut ? "start" : "end" });
            told.textContent = cut ? `cut at ${count(v)}, and the answer asked for` : "uncut: the reply's own answer";
            show(told, on);
            // the line follows the cut, and then reaches the reply uncut
            const reach = v + (MAX - stops.at(-1)) * during(t, when(MAX) - 0.5, when(MAX));
            for (const m of models) {
              trace(m.path, before(m.pts, budgets, reach));
              show(m.path, on);
              m.dots.forEach((dot, i) => show(dot, during(t, when(budgets[i]), when(budgets[i]) + 0.15)));
            }
            for (const mark of marks) show(mark.label, during(t, when(mark.b), when(mark.b) + 0.3));
          };
        },
      },
      {
        seconds: 7,
        caption: `A 1.5B reasoning model thinks through ${count(teacher.n)} training problems.\nOf its right replies that ended, ${TRACES} fine-tune the 0.5B model.`,
        draw(s) {
          const MAX = teacher.max_new_tokens;
          const head = s.text(70, 140, `the teacher, 1.5B: one reply to each of ${count(teacher.n)} training problems`, { size: 22, fill: "muted" });
          // every reply as a cell: kept (it ended and was right), cut off at the limit, or ended and wrong
          const COLS = 40;
          const PITCH = 15;
          const kinds = teacher.tokens.map((tokens, i) => (tokens < MAX ? (teacher.scores[i] ? "s1" : "muted") : "axis"));
          const cells = kinds.map((fill, i) => s.rect(70 + (i % COLS) * PITCH, 160 + Math.floor(i / COLS) * PITCH, PITCH - 3, PITCH - 3, { fill, r: 2 }));
          const legend = s.group();
          [
            ["s1", "ended and right: kept", 70],
            ["axis", `cut off at ${count(MAX)}`, 330],
            ["muted", "ended, wrong", 530],
          ].forEach(([fill, label, x]) => {
            legend.rect(x, 540, 18, 18, { fill, r: 3 });
            legend.text(x + 28, 557, label, { size: 22, fill: "ink2" });
          });
          const NUM = 840;
          const losses = studentLog.filter((r) => "val_loss" in r);
          const rows = [
            [percent(teacher.right), "right"],
            [percent(teacher.ended), `ended within ${count(MAX)} tokens`],
            [count(teacher.tokens_mean), "tokens on average"],
            [count(teacher.kept), "kept: closed, ended, right"],
            [count(TRACES), "fine-tune the 0.5B student"],
            [losses.at(-1).val_loss.toFixed(3), `loss on the other ${count(teacher.kept - TRACES)}, from ${losses[0].val_loss.toFixed(3)}`],
          ].map(([value, what], i) => {
            const g = s.group();
            const y = 204 + i * 68;
            g.text(NUM, y, value, { size: 34, weight: 700, anchor: "end", fill: i < 4 ? "s1" : "s3" });
            g.text(NUM + 18, y, what, { size: 22, fill: "ink2" });
            return g;
          });
          return (t) => {
            show(head, during(t, 0.2, 0.5));
            const written = cells.length * during(t, 0.3, 2.6);
            const sifted = during(t, 3.0, 3.5);
            cells.forEach((cell, i) => show(cell, (i < written ? 1 : 0) * (kinds[i] === "s1" ? 1 : 1 - 0.7 * sifted)));
            show(legend, during(t, 0.6, 1.0));
            [1.0, 1.6, 2.2, 3.0, 3.8, 4.4].forEach((at, i) => show(rows[i], during(t, at, at + 0.4)));
          };
        },
      },
      {
        seconds: 8,
        caption: "The student thinks like the teacher, and does not stop.\nIt learned the form of long thinking, not when it is done.",
        draw(s) {
          const X = 80;
          const head = s.text(X, 136, `the end of the student's first reply in a loop, cut off at ${count(student.max_new_tokens)} tokens`, { size: 22, fill: "muted" });
          // its last paragraphs, the first of them begun before the excerpt and left out
          const paragraphs = student.loop_example.split(/\n\n+/).slice(1).map((p) => p.replace(/\s+/g, " ").trim()).filter(Boolean);
          const LEAD = 30;
          let y = 178;
          const lines = [];
          for (const p of paragraphs) {
            let wrapped = wrap(p, 60);
            if (wrapped.length > 2) wrapped = [`${wrapped[0]} …`];
            for (const line of wrapped) {
              const again = /^But hold on, the question says/.exec(line);
              const answer = /the answer should be/.test(line);
              const parts = again ? [[again[0], "s2", 700], [line.slice(again[0].length), "ink2"]] : [[line, answer ? "s3" : "ink2", answer ? 700 : 400]];
              lines.push({ node: spans(s, X, y, parts, { mono: true }), y, chars: line.length, answer });
              y += LEAD;
            }
            y += 10;
          }
          const answered = lines.find((l) => l.answer);
          const right = answered ? s.text(X + answered.chars * 22 * MONO + 18, answered.y, "right", { size: 22, weight: 700, fill: "s3" }) : null;
          // back to where it began
          const [top, bottom] = [lines[0], lines.at(-1)];
          const RX = 940;
          const loop = s.group();
          loop.path([[X + bottom.chars * 22 * MONO + 12, bottom.y - 7], [RX, bottom.y - 7], [RX, top.y - 7]], { stroke: "s2", width: 3 });
          loop.arrow(RX, top.y - 7, X + 60 * 22 * MONO + 16, top.y - 7, { stroke: "s2", width: 3 });
          loop.lines(RX + 20, (top.y + bottom.y) / 2, ["and again,", "until the limit"], { size: 22, fill: "s2" });

          // how often each ran into its limit, and in a loop
          const keys = s.group();
          keys.rect(X, 446, 18, 18, { fill: "axis", r: 3 });
          keys.text(X + 28, 462, "cut off at the limit", { size: 22, fill: "ink2" });
          keys.rect(X + 260, 446, 18, 18, { fill: "ink2", r: 3 });
          keys.text(X + 288, 462, "in a loop: its last 200 characters written before", { size: 22, fill: "ink2" });
          const [BX, FULL] = [290, 460];
          const teacher192 = summary.first.teacher;
          const bars = [
            { name: "the student", fill: "s3", total: student.n, cut: student.cut_off, looping: student.cut_off_looping, limit: student.max_new_tokens, y: 480, from: 4.8 },
            { name: "the teacher", fill: "s1", total: summary.first.n, cut: teacher192.cut_off, looping: teacher192.cut_off_looping, limit: teacher192.max_new_tokens, y: 530, from: 5.3 },
          ].map((b) => {
            const g = s.group();
            g.text(X, b.y + 23, b.name, { size: 24, weight: 600, fill: b.fill });
            g.rect(BX, b.y, FULL, 30, { fill: "tint", r: 4 });
            const words = `${count(b.cut)} of ${count(b.total)} cut off at ${count(b.limit)}, ${b.looping ? `${count(b.looping)} in a loop` : "none in a loop"}`;
            return { ...b, g, cutBar: s.rect(BX, b.y, 0, 30, { fill: "axis", r: 4 }), loopBar: s.rect(BX, b.y, 0, 30, { fill: "ink2", r: 4 }), words: s.text(BX + FULL + 18, b.y + 23, words, { size: 22, fill: "ink2" }) };
          });
          return (t) => {
            show(head, during(t, 0.2, 0.5));
            lines.forEach((l, i) => show(l.node, during(t, 0.4 + i * 0.32, 0.7 + i * 0.32)));
            if (right) show(right, during(t, 3.0, 3.3));
            show(loop, during(t, 3.6, 4.1));
            show(keys, during(t, 4.4, 4.8));
            for (const b of bars) {
              show(b.g, during(t, b.from - 0.2, b.from + 0.1));
              const u = during(t, b.from, b.from + 0.7);
              set(b.cutBar, { width: (FULL * b.cut * u) / b.total });
              set(b.loopBar, { width: (FULL * b.looping * u) / b.total });
              show(b.words, during(t, b.from + 0.5, b.from + 0.8));
            }
          };
        },
      },
      {
        seconds: 7,
        caption: `Cut short, the student gains a little. A vote of ${studentTimes.samples} gains more.\nNeither reaches the published model's greedy reply.`,
        draw(s) {
          const FLOOR = 470;
          const tall = (v) => 540 * v;
          const head = s.text(90, 140, `solved, of ${count(base.n)} test problems, greedy`, { size: 22, fill: "muted" });
          const bars = [
            ["as published", base, "ink2"],
            ["after GRPO", grpo, "s2"],
            ["the student", student, "s3"],
          ].map(([name, result, fill], i) => {
            const x = 100 + i * 175;
            const mid = x + 60;
            const over = s.group();
            over.line(mid, FLOOR - tall(result.mean + result.error), mid, FLOOR - tall(result.mean - result.error), { stroke: "ink", width: 2.5 });
            over.text(mid, FLOOR - tall(result.mean + result.error) - 12, percent(result.mean), { size: 26, weight: 700, anchor: "middle" });
            return { result, over, from: 0.3 + i * 0.35, bar: s.rect(x, FLOOR, 120, 0, { fill, r: 6 }), label: s.text(mid, FLOOR + 32, name, { size: 24, weight: 600, anchor: "middle", fill }) };
          });
          for (const b of bars) b.over.g.parentNode.appendChild(b.over.g); // the whiskers lie over the bars
          s.line(80, FLOOR, 640, FLOOR, { stroke: "axis" });

          // on the first problems: the student greedy, cut and made to answer, and voting; the published model greedy
          const R = 720;
          const wide = (v) => 800 * v;
          const right = s.text(R, 140, `the first ${n} test problems`, { size: 22, fill: "muted" });
          const greedy = studentTimes.greedy.right;
          const rows = [
            { name: "the student, greedy", fill: "s3", from: 0, to: greedy, y: 186, at: 1.9 },
            { name: `the student, cut at ${count(CUT)} tokens`, fill: "s3", from: greedy, to: studentTimes.budgets[CUT].right, y: 268, at: 2.6 },
            { name: `the student, a vote of ${studentTimes.samples}`, fill: "s3", from: 0, to: voted, y: 350, at: 3.3 },
            { name: "the published model, greedy", fill: "ink2", from: 0, to: published.greedy.right, y: 432, at: 4.0 },
          ].map((r) => ({ ...r, label: s.text(R, r.y, r.name, { size: 22, fill: "ink2" }), bar: s.rect(R, r.y + 12, 0, 36, { fill: r.fill, r: 6 }), value: s.text(R, r.y + 39, "", { size: 24, weight: 700, anchor: "end", fill: "surface" }) }));
          // where the published model's greedy reply is, beside the student's bars
          const level = s.line(R + wide(published.greedy.right), 192, R + wide(published.greedy.right), 404, { stroke: "ink2", width: 2.5 });
          level.setAttribute("stroke-dasharray", "6 6");
          return (t) => {
            show(head, during(t, 0.2, 0.5));
            for (const b of bars) {
              const h = tall(b.result.mean) * during(t, b.from, b.from + 0.9);
              set(b.bar, { y: FLOOR - h, height: h });
              show(b.label, during(t, b.from - 0.2, b.from + 0.1));
              show(b.over, during(t, b.from + 0.8, b.from + 1.1));
            }
            show(right, during(t, 1.8, 2.1));
            for (const r of rows) {
              show(r.label, during(t, r.at - 0.2, r.at + 0.1));
              const v = r.from ? mix(r.from, r.to, during(t, r.at + 0.4, r.at + 1.1)) : r.to * during(t, r.at, r.at + 0.7);
              show(r.bar, during(t, r.at - 0.1, r.at + 0.1));
              set(r.bar, { width: wide(v) });
              set(r.value, { x: R + wide(v) - 12 });
              r.value.textContent = percent(v);
              show(r.value, during(t, r.at + 0.3, r.at + 0.6));
            }
            show(level, during(t, 4.6, 4.9));
          };
        },
      },
    ],
  };
};
