// Stage 9 as a film: a program checks the answer, the replies to one prompt
// are compared with each other, and the better ones become likelier. The
// stories of the loop are an illustration; their rewards and advantages are
// computed here as tasks.words_score and rl.group_advantages compute them.
// The curves, the story that cheats and every score are the logged ones.

import { Tokenizer } from "../labs/bpe.js";
import { svg } from "../labs/kit.js";
import { clamp, during, mix, set, show, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  run: "/runs/rlvr/grpo-words.json",
  log: "/runs/rlvr/grpo-words.jsonl",
  stories: "/runs/rlvr/stories.json",
  cheat: "/runs/eval/words-grpo-words.json",
  mathRun: "/runs/rlvr/grpo-gsm8k.json",
  mathLog: "/runs/rlvr/grpo-gsm8k.jsonl",
  published: "/runs/eval/gsm8k-qwen.json",
  tuned: "/runs/eval/gsm8k-sft-gsm8k-lora.json",
  grpo: "/runs/eval/gsm8k-grpo-gsm8k.json",
};

// One prompt of the words task and eight stories for it, made up to show the loop.
const PROMPT = "Write a short story using these words: another, easy, sleep.";
const REPLIES = [
  "It was easy to sleep in another bed…",
  "Tom played with his red ball all day…",
  "The bear went to sleep in his cave…",
  "Mia had another easy day at the park…",
  "Sam could not sleep. It was not easy…",
  "Another bird sat in the tall tree…",
  "The sun was hot and the cat was sad…",
  "Ben found another easy place to sleep…",
];

const MONO = 0.602; // the width of a character of the monospaced font, in ems
const percent = (v, digits = 0) => `${(100 * v).toFixed(digits)}%`;
const mean = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
const signed = (v, digits = 2) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(digits)}`;

// The words a prompt of the words task asks for.
const asked = (prompt) => prompt.slice(prompt.indexOf(":") + 1).replace(/\.\s*$/, "").split(",").map((word) => word.trim());
// tasks.words_score: the requested words that the reply uses, as whole words.
const found = (words, reply) => words.filter((word) => (reply.toLowerCase().match(/[a-z]+/g) || []).includes(word));
// rl.group_advantages: reward minus the group's mean, over the group's standard deviation.
function groupAdvantages(rewards) {
  const center = mean(rewards);
  const spread = Math.sqrt(rewards.reduce((sum, r) => sum + (r - center) ** 2, 0) / (rewards.length - 1));
  return { center, spread, advantages: rewards.map((r) => (r - center) / (spread + 1e-6)) };
}

// A line of monospaced text in pieces, so that single words can be marked. Returns { label, spans }.
function pieces(stage, x, y, string, size) {
  const label = svg("text", { x, y, "font-size": size, class: "film-mono" }, stage.g);
  const spans = string.split(/([A-Za-z]+)/).filter(Boolean).map((piece) => {
    const span = svg("tspan", { class: "f-ink" }, label);
    span.textContent = piece;
    return { span, word: piece.toLowerCase() };
  });
  return { label, spans };
}
// Mark the given words of such a line, or none of them.
function mark(line, words, on) {
  for (const { span, word } of line.spans) {
    const marked = on && words.includes(word);
    set(span, { class: marked ? "f-s2" : "f-ink", "font-weight": marked ? "700" : "400" });
  }
}
// A reply as a chip whose words can be marked.
function reply(stage, x, y, string, size = 22) {
  const w = [...string].length * size * MONO + 28;
  stage.rect(x, y, w, size * 1.7, { fill: "tint", stroke: "axis", r: 8, width: 1.5 });
  return pieces(stage, x + 12, y + size * 1.2, string, size);
}

// A box with a title, and a line below it.
function card(stage, x, y, w, h, title, sub, { fill = "raised", stroke = "axis" } = {}) {
  const g = stage.group();
  g.rect(x, y, w, h, { fill, stroke, r: 12 });
  g.text(x + w / 2, y + h / 2 + (sub ? -4 : 9), title, { size: 26, weight: 600, anchor: "middle" });
  if (sub) g.text(x + w / 2, y + h / 2 + 26, sub, { size: 22, anchor: "middle", fill: "ink2" });
  return g;
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

const ROW = (i) => 178 + i * 50; // where the replies of the loop are, in both of its scenes

export default ({ vocabulary, run, log, stories, cheat, mathRun, mathLog, published, tuned, grpo }) => {
  // the loop, on the made-up group
  const words = asked(PROMPT);
  const group = REPLIES.slice(0, run.args.samples).map((text) => ({ text, found: found(words, text) }));
  const rewards = group.map((r) => r.found.length / words.length);
  const { center, spread, advantages } = groupAdvantages(rewards);

  // the words run: the held-out score, and running means over 5 iterations of what every iteration logged
  const evals = log.filter((r) => "held_out" in r);
  const steps = log.filter((r) => "reward" in r);
  const running = (key) => steps.map((_, j) => mean(steps.slice(Math.max(0, j - 4), j + 1).map((r) => r[key])));
  const signal = running("informative_groups");
  const tokens = running("tokens");
  const limit = cheat.max_new_tokens;

  // the story the unleashed model writes for the first test prompt
  const story = cheat.samples[0];
  const wanted = asked(story.prompt);
  const length = new Tokenizer(vocabulary).encode(story.reply).length;

  // GSM8K
  const gain = grpo.mean - tuned.mean;
  const rightAt = (rows) => mean(rows.map((r) => r.reward));
  const sampled = mathLog.filter((r) => "reward" in r);

  return {
    subtitle: "A program checks the answer, and the check is the reward",
    scenes: [
      {
        seconds: 6,
        caption: "For some tasks a program can check the answer.\nThen no reward model has to be learned from people.",
        draw(s) {
          const cards = [
            ["A math problem", "#### 18", "has a number"],
            ["Code", "assert f(2) == 4", "has tests"],
            ["A story", "another, easy, sleep", "uses the requested words"],
          ].map(([title, example, check], i) => {
            const g = s.group();
            const x = 90 + i * 380;
            g.rect(x, 112, 340, 190, { fill: "raised", stroke: "axis" });
            g.text(x + 170, 156, title, { size: 28, weight: 600, anchor: "middle" });
            g.chips(x + 170, 180, [example], { center: true, size: 24 });
            g.text(x + 170, 270, check, { size: 24, anchor: "middle", fill: "ink2" });
            return g;
          });
          const flow = [
            card(s, 250, 368, 200, 92, "a reply", "", { fill: "tint" }),
            s.arrow(462, 414, 528, 414),
            card(s, 540, 368, 220, 92, "a verifier", "a program", { stroke: "s2" }),
            s.arrow(772, 414, 838, 414),
            card(s, 850, 368, 200, 92, "its reward", "", { fill: "tint" }),
          ];
          const point = s.text(640, 540, "The measure of success is the training signal.", { size: 24, anchor: "middle", fill: "muted" });
          return (t) => {
            cards.forEach((g, i) => show(g, during(t, 0.3 + i * 0.6, 0.8 + i * 0.6)));
            flow.forEach((part, i) => show(part, during(t, 2.5 + i * 0.3, 2.9 + i * 0.3)));
            show(point, during(t, 3.9, 4.4));
          };
        },
      },
      {
        seconds: 8.5,
        caption: `The model writes ${group.length} stories for the same prompt.\nA verifier scores each: the share of the requested words used.`,
        draw(s) {
          const prompt = s.chips(640, 100, [PROMPT], { center: true, size: 24 })[0];
          const heads = [s.text(650, 166, "requested words found", { size: 22, fill: "muted" }), s.text(1180, 166, "reward", { size: 22, anchor: "end", fill: "muted" })];
          const rows = group.map((r, i) => {
            const text = s.group();
            const verdict = s.group();
            const y = ROW(i);
            const line = reply(text, 90, y, r.text);
            const asks = verdict.chips(650, y, words, { size: 22, pad: 9, gap: 6 });
            asks.forEach((c, j) => {
              const there = r.found.includes(words[j]);
              c.ground.setAttribute("class", there ? "f-s3" : "f-surface s-axis");
              c.label.setAttribute("class", `film-mono ${there ? "f-surface" : "f-muted"}`);
            });
            verdict.text(960, y + 27, `${r.found.length} of ${words.length}`, { size: 24, fill: "ink2" });
            verdict.text(1180, y + 27, rewards[i].toFixed(2), { size: 24, anchor: "end", mono: true });
            return { text, verdict, line };
          });
          return (t) => {
            show(prompt, during(t, 0.3, 0.8));
            heads.forEach((h) => show(h, during(t, 3.2, 3.7)));
            rows.forEach((row, i) => {
              const checked = during(t, 3.5 + i * 0.36, 3.9 + i * 0.36);
              show(row.text, during(t, 1.0 + i * 0.24, 1.4 + i * 0.24));
              show(row.verdict, checked);
              mark(row.line, group[i].found, checked > 0.5);
            });
          };
        },
      },
      {
        seconds: 9,
        caption: "A reply's advantage is its reward minus the group's mean,\nover the spread. One step makes the better replies likelier.",
        draw(s) {
          const X = 615; // the bars of the rewards start here,
          const FULL = 220; // and a reward of 1 is this long
          const at = (reward) => X + FULL * reward;
          const named = s.chips(640, 100, ["advantage = (reward − mean) / standard deviation"], { center: true, size: 24 })[0];
          const valued = s.chips(640, 100, [`advantage = (reward − ${center.toFixed(2)}) / ${spread.toFixed(2)}`], { center: true, size: 24 })[0];
          s.text(X, 166, "reward", { size: 22, fill: "muted" });
          const heads = [s.text(1030, 166, "advantage", { size: 22, anchor: "end", fill: "muted" }), s.text(1066, 166, "after one step", { size: 22, fill: "muted" })];
          const rows = group.map((r, i) => {
            const y = ROW(i);
            const better = advantages[i] > 0;
            s.chip(60, y, r.text, { size: 22, ink: "ink2" });
            s.rect(X, y + 8, FULL, 22, { fill: "tint", r: 4 });
            return {
              bar: s.rect(X, y + 8, 0, 22, { fill: "s1", r: 4 }),
              value: s.text(X + FULL + 12, y + 27, rewards[i].toFixed(2), { size: 22, fill: "ink2", mono: true }),
              gap: s.rect(at(center), y + 8, 0, 22, { fill: better ? "s3" : "axis", r: 0 }),
              advantage: s.text(1030, y + 27, signed(advantages[i]), { size: 24, anchor: "end", fill: better ? "s3" : "ink2", weight: better ? 700 : 400, mono: true }),
              step: s.arrow(1080, y + (better ? 32 : 6), 1080, y + (better ? 6 : 32), { stroke: better ? "s3" : "muted", width: 3 }),
              word: s.text(1100, y + 27, better ? "likelier" : "rarer", { size: 22, fill: better ? "s3" : "muted" }),
            };
          });
          const middle = s.line(at(center), 148, at(center), 148, { stroke: "ink", width: 2.5 });
          const middleName = s.text(at(center) + 10, 166, `mean ${center.toFixed(2)}`, { size: 22 });
          return (t) => {
            const down = during(t, 1.8, 2.6);
            show(named, 1 - during(t, 4.0, 4.4));
            show(valued, during(t, 4.2, 4.6));
            set(middle, { y2: mix(148, 574, down) });
            show(middle, down > 0 ? 1 : 0);
            show(middleName, during(t, 2.4, 2.8));
            show(heads[0], during(t, 4.2, 4.6));
            show(heads[1], during(t, 6.0, 6.4));
            rows.forEach((row, i) => {
              const grown = during(t, 0.5 + i * 0.06, 1.3 + i * 0.06);
              const apart = during(t, 3.0, 4.0);
              const edge = mix(at(center), at(rewards[i]), apart);
              set(row.bar, { width: FULL * rewards[i] * grown });
              show(row.value, grown);
              set(row.gap, { x: Math.min(edge, at(center)), width: Math.abs(edge - at(center)) });
              show(row.advantage, during(t, 4.4 + i * 0.14, 4.8 + i * 0.14));
              show(row.step, during(t, 6.1, 6.6));
              show(row.word, during(t, 6.3, 6.8));
            });
          };
        },
      },
      {
        seconds: 7.5,
        caption: `In ${run.minutes.toFixed(0)} minutes the story model goes from ${percent(evals[0].held_out)} to ${percent(evals.at(-1).held_out)}.\nAs it learns, fewer groups hold replies that differ in reward.`,
        draw(s) {
          const most = run.args.iterations;
          const px = (k) => 150 + (560 * k) / most;
          const py = (v) => 420 - 290 * v;
          for (const tick of [0, 0.5, 1]) {
            s.line(150, py(tick), 710, py(tick), { stroke: tick ? "grid" : "axis", width: tick ? 1.5 : 2 });
            s.text(138, py(tick) + 8, percent(tick), { size: 22, anchor: "end", fill: "muted" });
          }
          for (const tick of [0, most / 3, (2 * most) / 3, most]) s.text(px(tick), 450, tick.toFixed(0), { size: 22, anchor: "middle", fill: "muted" });
          s.text(430, 482, `iterations of ${run.args.prompts} prompts with ${run.args.samples} stories each`, { size: 22, anchor: "middle", fill: "muted" });

          const signalAt = steps.map((r, j) => [px(r.iteration), py(signal[j])]);
          const scoreAt = evals.map((r) => [px(r.iteration), py(r.held_out)]);
          const signalLine = s.path(signalAt, { stroke: "s1", width: 3 });
          const scoreLine = s.path(scoreAt, { stroke: "s2", width: 4 });
          const dots = scoreAt.map(([x, y]) => s.circle(x, y, 6, { fill: "s2" }));
          const ends = s.group();
          const score = ends.text(728, py(evals.at(-1).held_out) + 12, "", { size: 32, weight: 700, fill: "s2" });
          ends.text(810, py(evals.at(-1).held_out) + 10, "of the requested words used", { size: 22, fill: "ink2" });
          const share = ends.text(728, py(signal.at(-1)) + 12, "", { size: 32, weight: 700, fill: "s1" });
          ends.text(810, py(signal.at(-1)) + 10, "of the groups still carry signal", { size: 22, fill: "ink2" });
          ends.text(810, py(signal.at(-1)) + 40, "equal rewards in a group teach nothing", { size: 22, fill: "muted" });

          // the length of the stories, against the limit
          const wide = (n) => (420 * n) / limit;
          s.text(150, 548, "tokens per story", { size: 22, fill: "ink2" });
          s.rect(340, 526, wide(limit), 28, { fill: "tint", r: 4 });
          const bar = s.rect(340, 526, 0, 28, { fill: "s1", r: 4 });
          const count = s.text(0, 548, "", { size: 22, anchor: "end", fill: "surface", weight: 700 });
          s.line(340 + wide(limit), 518, 340 + wide(limit), 562, { stroke: "ink", width: 2.5 });
          s.text(352 + wide(limit), 548, `the limit of ${limit}`, { size: 22, fill: "ink2" });
          return (t) => {
            const k = most * during(t, 0.8, 4.8);
            const j = clamp(Math.round(k) - 1, 0, steps.length - 1);
            trace(scoreLine, before(scoreAt, evals.map((r) => r.iteration), k));
            trace(signalLine, before(signalAt, steps.map((r) => r.iteration), k));
            show(signalLine, during(t, 0.8, 1.0));
            dots.forEach((dot, i) => show(dot, t > 0.5 && k >= evals[i].iteration ? 1 : 0));
            show(ends, during(t, 0.5, 1.0));
            score.textContent = percent(evals.findLast((r) => r.iteration <= k).held_out);
            share.textContent = percent(signal[j]);
            set(bar, { width: wide(tokens[j]) });
            set(count, { x: 340 + wide(tokens[j]) - 10 });
            count.textContent = tokens[j].toFixed(0);
          };
        },
      },
      {
        seconds: 7.5,
        caption: "The verifier counts words. The model learned to produce words.\nNothing in the reward says that the text should be a story.",
        draw(s) {
          const prompt = s.chip(60, 100, story.prompt, { size: 24 });
          const lines = wrap(story.reply, 64).map((line, i) => pieces(s, 60, 184 + i * 30, line, 22));
          lines.forEach((line) => mark(line, wanted, true));
          const verdict = s.group();
          verdict.text(1000, 206, "the verifier finds", { size: 22, fill: "muted" });
          verdict.text(1000, 240, `${found(wanted, story.reply).length} of ${wanted.length} words`, { size: 26, weight: 600 });
          const reward = s.group();
          reward.text(1000, 312, "reward", { size: 22, fill: "muted" });
          reward.text(1000, 366, story.score.toFixed(2), { size: 52, weight: 700, fill: "s3" });
          const cut = s.group();
          cut.text(1000, 438, "length", { size: 22, fill: "muted" });
          cut.text(1000, 472, `${length} tokens`, { size: 26, weight: 600 });
          if (length >= limit) cut.text(1000, 502, "cut off at the limit", { size: 22, fill: "ink2" });
          return (t) => {
            show(prompt, during(t, 0.3, 0.8));
            lines.forEach((line, i) => show(line.label, during(t, 0.9 + i * 0.26, 1.2 + i * 0.26)));
            show(verdict, during(t, 4.4, 4.9));
            show(reward, during(t, 4.8, 5.3));
            show(cut, during(t, 5.2, 5.7));
          };
        },
      },
      {
        seconds: 8,
        caption: "A stricter verifier closes one hole and leaves the other.\nA KL penalty keeps the stories ordinary, at a lower score.",
        draw(s) {
          const models = [
            ["sft-words", ["fine-tuned, the start"], 0.4],
            ["grpo-words", ["GRPO"], 0.8],
            ["grpo-words-finish", ["GRPO, reward only", "if the story ends"], 2.8],
            ["grpo-words-kl", ["GRPO through LoRA", "with a KL penalty"], 4.6],
          ];
          const strangest = Math.max(...models.map(([name]) => stories[name].strangeness));
          const measures = [
            ["words_used", "requested words used", 1, (v) => percent(v)],
            ["cut_off", "stories cut off", 1, (v) => percent(v, v < 0.01 ? 1 : 0)],
            ["strangeness", "strangeness", strangest, (v) => v.toFixed(2)],
          ];
          const FULL = 180;
          const left = (j) => 370 + j * 290;
          measures.forEach(([, title], j) => s.text(left(j), 132, title, { size: 24, weight: 600 }));
          const rows = models.map(([name, label, start], i) => {
            const g = s.group();
            const y = 172 + i * 104;
            g.lines(60, label.length > 1 ? y + 12 : y + 27, label, { size: 24, leading: 1.2 });
            const cells = measures.map(([key, , full, format], j) => {
              const w = (FULL * stories[name][key]) / full;
              g.rect(left(j), y, FULL, 36, { fill: "tint", r: 4 });
              return { w, bar: g.rect(left(j), y, 0, 36, { fill: "s1", r: 4 }), value: g.text(left(j) + w + 10, y + 27, format(stories[name][key]), { size: 24, weight: 600 }) };
            });
            return { g, cells, start };
          });
          return (t) => {
            const now = rows.findLastIndex((row) => t >= row.start);
            rows.forEach((row, i) => {
              show(row.g, during(t, row.start, row.start + 0.4));
              row.cells.forEach((cell, j) => {
                const u = during(t, row.start + 0.2 + j * 0.15, row.start + 1.1 + j * 0.15);
                set(cell.bar, { width: cell.w * u, class: i > 0 && i === now ? "f-s2" : "f-s1" });
                set(cell.value, { x: left(j) + cell.w * u + 10 });
                show(cell.value, during(t, row.start + 0.2, row.start + 0.5));
              });
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `On GSM8K the verdict alone adds ${(100 * gain).toFixed(1)} points to the 0.5B model.\nIt solves more problems, and stays below the published model.`,
        draw(s) {
          const FLOOR = 500;
          const tall = (v) => 560 * v;
          const bars = [
            ["as published", published, 0.4],
            ["fine-tuned", tuned, 1.0],
            ["after GRPO", grpo, 1.6],
          ].map(([name, result, start], i) => {
            const x = 130 + i * 210;
            s.text(x + 65, FLOOR + 34, name, { size: 24, anchor: "middle", fill: "ink2" });
            return { x, result, start, h: tall(result.mean), bar: s.rect(x, FLOOR, 130, 0, { fill: "s1", r: 6 }) };
          });
          // what GRPO added to the fine-tuned model, as the top of its bar
          const added = s.rect(130 + 2 * 210, FLOOR - tall(tuned.mean), 130, 0, { fill: "s3", r: 0 });
          const level = s.line(130 + 210, FLOOR - tall(tuned.mean), 130 + 2 * 210 + 130, FLOOR - tall(tuned.mean), { stroke: "muted", width: 1.5 });
          // every score with its 95% interval
          for (const b of bars) {
            b.over = s.group();
            b.over.line(b.x + 65, FLOOR - tall(b.result.mean + b.result.error), b.x + 65, FLOOR - tall(b.result.mean - b.result.error), { stroke: "ink", width: 2.5 });
            b.over.text(b.x + 65, FLOOR - b.h - 54, percent(b.result.mean, 1), { size: 28, weight: 700, anchor: "middle" });
            b.over.text(b.x + 65, FLOOR - b.h - 28, `± ${percent(b.result.error, 1)}`, { size: 22, anchor: "middle", fill: "muted" });
          }
          s.line(110, FLOOR, 700, FLOOR, { stroke: "axis" });
          s.text(405, FLOOR + 70, `${grpo.n.toLocaleString("en")} test problems solved, greedy replies`, { size: 22, anchor: "middle", fill: "muted" });
          const told = s.group();
          told.text(780, 250, `${signed(100 * gain, 1)} points`, { size: 54, weight: 700, fill: "s3" });
          told.lines(780, 300, [`from ${mathRun.args.iterations} iterations of ${mathRun.args.prompts} problems`, `with ${mathRun.args.samples} replies each, in ${mathRun.minutes.toFixed(0)} minutes`], { size: 24, fill: "ink2" });
          const training = s.group();
          training.lines(780, 420, ["Sampled replies that are right:", `${percent(rightAt(sampled.slice(0, 10)))} in the first 10 iterations,`, `${percent(rightAt(sampled.slice(-10)))} in the last 10.`], { size: 24, fill: "ink2" });
          return (t) => {
            // the bar of GRPO first rises to where it started, and then by what it added
            const up = during(t, 3.0, 3.6) * tall(gain);
            bars.forEach(({ bar, h, over, start, result }) => {
              const last = result === grpo;
              const now = last ? tall(tuned.mean) * during(t, start, start + 1.0) + up : h * during(t, start, start + 1.0);
              set(bar, { y: FLOOR - now, height: now });
              show(over, last ? during(t, 3.5, 3.9) : during(t, start + 0.8, start + 1.2));
            });
            set(added, { y: FLOOR - tall(tuned.mean) - up, height: up });
            show(level, during(t, 2.4, 2.8));
            show(told, during(t, 3.7, 4.2));
            show(training, during(t, 4.6, 5.1));
          };
        },
      },
    ],
  };
};
