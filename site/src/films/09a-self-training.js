// The side topic 9a as a film: the loop of stage 9 with fine-tuning on the
// right replies in place of the policy gradient, what that does to the story
// model round after round, and why it does nothing on GSM8K. The stories of
// the second scene are an illustration, chosen as selftrain.keep would choose.
// The rounds, the two replies to a test problem and every score are logged.

import { clamp, during, mix, place, set, show, trace } from "./stage.js";

export const sources = {
  meta: "/runs/self/words-meta.json",
  rounds: "/runs/self/words.jsonl",
  stories: "/runs/self/stories.json",
  grpoRun: "/runs/rlvr/grpo-words.json",
  grpoLog: "/runs/rlvr/grpo-words.jsonl",
  mathRounds: "/runs/self/gsm8k.jsonl",
  reference: "/runs/sft-gsm8k-lora/meta.json",
  wordsStart: "/runs/eval/words-sft-words.json",
  wordsSelf: "/runs/eval/words-self-words.json",
  wordsLeashed: "/runs/eval/words-grpo-words-kl.json",
  wordsGrpo: "/runs/eval/words-grpo-words.json",
  mathStart: "/runs/eval/gsm8k-sft-gsm8k-lora.json",
  mathSelf: "/runs/eval/gsm8k-self-gsm8k.json",
  mathGrpo: "/runs/eval/gsm8k-grpo-gsm8k.json",
};

// One prompt of the words task and four stories for it, made up to show what is kept.
const PROMPT = "Write a short story using these words: another, easy, sleep.";
const REPLIES = [
  { text: "Mia had another easy day at the park…", finished: true },
  { text: "Tom played with his red ball all day…", finished: true },
  { text: "Ben found another easy place to sleep…", finished: false },
  { text: "It was easy to sleep in another bed…", finished: true },
];

const percent = (v, digits = 0) => `${(100 * v).toFixed(digits)}%`;
const points = (v) => (100 * v).toFixed(1);

// The words a prompt of the words task asks for.
const asked = (prompt) => prompt.slice(prompt.indexOf(":") + 1).replace(/\.\s*$/, "").split(",").map((word) => word.trim());
// tasks.words_score: the requested words that the reply uses, as whole words.
const found = (words, reply) => words.filter((word) => (reply.toLowerCase().match(/[a-z]+/g) || []).includes(word));
// selftrain.keep: of the replies to one prompt, those that ended and are entirely right, each text once, at most per_prompt.
function keep(replies, scores, perPrompt) {
  const seen = new Set();
  return replies.map((r, i) => {
    const kept = r.finished && scores[i] === 1 && !seen.has(r.text) && seen.size < perPrompt;
    if (kept) seen.add(r.text);
    return kept;
  });
}

// A box with a title, and a line below it.
function card(stage, x, y, w, h, title, sub, { fill = "raised", stroke = "axis" } = {}) {
  const g = stage.group();
  g.rect(x, y, w, h, { fill, stroke, r: 12 });
  g.text(x + w / 2, y + h / 2 + (sub ? -4 : 9), title, { size: 26, weight: 600, anchor: "middle" });
  if (sub) g.text(x + w / 2, y + h / 2 + 26, sub, { size: 22, anchor: "middle", fill: "ink2" });
  return g;
}

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

// How much of the line through places lies before k on the axis that xs are on.
function before(places, xs, k) {
  let total = 0;
  let done = 0;
  for (let i = 1; i < places.length; i++) {
    const d = Math.hypot(places[i][0] - places[i - 1][0], places[i][1] - places[i - 1][1]);
    total += d;
    done += d * clamp((k - xs[i - 1]) / (xs[i] - xs[i - 1]));
  }
  return done / total;
}

// Where a point is that has gone the part u of the way through corners.
function along(corners, u) {
  const lengths = corners.slice(1).map(([x, y], i) => Math.hypot(x - corners[i][0], y - corners[i][1]));
  let left = clamp(u) * lengths.reduce((sum, d) => sum + d, 0);
  for (let i = 0; i < lengths.length; i++) {
    if (left <= lengths[i]) return [mix(corners[i][0], corners[i + 1][0], left / lengths[i]), mix(corners[i][1], corners[i + 1][1], left / lengths[i])];
    left -= lengths[i];
  }
  return corners.at(-1);
}

// Bars that stand on a floor, each with its score, the interval of the score and a name.
// rows: [name lines, result of an evaluation, accent]. Returns what sets them to a time.
function standing(s, x, floor, tall, rows, start) {
  const made = rows.map(([name, result, accent], i) => {
    const left = x + i * 130;
    const h = tall(result.mean);
    s.lines(left + 48, floor + 32, name, { size: 22, anchor: "middle", fill: "ink2", leading: 1.2 });
    const bar = s.rect(left, floor, 96, 0, { fill: accent ? "s2" : "s1", r: 6 });
    const over = s.group();
    over.text(left + 48, floor - h - tall(result.error) - 10, percent(result.mean, 1), { size: 24, weight: 700, anchor: "middle" });
    over.line(left + 48, floor - h - tall(result.error), left + 48, floor - h + tall(result.error), { stroke: "ink", width: 2.5 });
    return { bar, h, over };
  });
  s.line(x - 16, floor, x + rows.length * 130 - 18, floor, { stroke: "axis" });
  return (t) =>
    made.forEach(({ bar, h, over }, i) => {
      const u = during(t, start + i * 0.4, start + 0.9 + i * 0.4);
      set(bar, { y: floor - h * u, height: h * u });
      show(over, during(t, start + 0.7 + i * 0.4, start + 1.1 + i * 0.4));
    });
}

export default ({ meta, rounds, stories, grpoRun, grpoLog, mathRounds, reference, wordsStart, wordsSelf, wordsLeashed, wordsGrpo, mathStart, mathSelf, mathGrpo }) => {
  // what is kept, on the made-up group
  const words = asked(PROMPT);
  const group = REPLIES.slice(0, meta.args.samples).map((r) => ({ ...r, found: found(words, r.text) }));
  const kept = keep(group, group.map((r) => r.found.length / words.length), meta.args.per_prompt);

  // the rounds on the words task, and GRPO's run from the same start, by the stories sampled so far
  const perRound = meta.args.prompts * meta.args.samples;
  const ours = rounds.map((r) => ({ n: r.round * perRound, score: r.held_out, kept: r.kept }));
  const theirs = grpoLog.filter((r) => "held_out" in r).map((r) => ({ n: r.iteration * grpoRun.args.prompts * grpoRun.args.samples, score: r.held_out }));
  const most = Math.max(ours.at(-1).n, theirs.at(-1).n);

  // a test problem of GSM8K that the model at the start gets wrong and the self-trained one right
  const which = mathStart.samples.findIndex((sample, i) => sample.score === 0 && mathSelf.samples[i].score === 1);
  const problem = mathStart.samples[which].prompt.split("\n\n")[0].split(/(?<=[.?])\s+/);
  const firstRound = mathRounds.find((r) => r.round === 1);

  return {
    subtitle: "Keep the answers that are right, and fine-tune on them",
    scenes: [
      {
        seconds: 7,
        caption: "Self-training keeps stage 9's replies and its verifier.\nIt replaces the policy gradient with plain fine-tuning.",
        draw(s) {
          const Y = 240;
          const rail = [
            [185, Y + 50],
            [1075, Y + 50],
            [1075, Y + 196],
            [185, Y + 196],
            [185, Y + 50],
          ];
          s.arrow(300, Y + 50, 350, Y + 50);
          s.arrow(580, Y + 50, 640, Y + 50);
          const again = s.group();
          again.line(1075, Y + 100, 1075, Y + 196, { stroke: "muted", width: 2.5 });
          again.line(1075, Y + 196, 185, Y + 196, { stroke: "muted", width: 2.5 });
          again.arrow(185, Y + 196, 185, Y + 104);
          again.text(630, Y + 240, "then the better model answers again, round after round", { size: 24, anchor: "middle", fill: "ink2" });
          const between = s.arrow(890, Y + 50, 950, Y + 50);
          const dot = s.circle(0, 0, 10, { fill: "s2" });
          const same = [card(s, 70, Y, 230, 100, "answer", "several times"), card(s, 350, Y, 230, 100, "verify", "a program")];
          const sameName = s.text(325, Y - 24, "as in stage 9", { size: 22, anchor: "middle", fill: "muted" });
          const gradient = card(s, 640, Y, 560, 100, "policy-gradient step", "the better replies likelier", { fill: "tint" });
          const simpler = [card(s, 640, Y, 250, 100, "keep", "what is right", { stroke: "s2" }), card(s, 950, Y, 250, 100, "fine-tune", "on what was kept", { stroke: "s2" })];
          const simplerName = s.text(920, Y - 24, "in place of the policy gradient", { size: 22, anchor: "middle", fill: "s2" });
          return (t) => {
            same.forEach((box, i) => show(box, during(t, 0.3 + i * 0.4, 0.7 + i * 0.4)));
            show(sameName, during(t, 0.9, 1.3));
            show(gradient, Math.min(during(t, 1.1, 1.5), 1 - during(t, 2.3, 2.8)));
            simpler.forEach((box, i) => show(box, during(t, 2.7 + i * 0.3, 3.1 + i * 0.3)));
            show(between, during(t, 2.9, 3.3));
            show(simplerName, during(t, 3.0, 3.4));
            show(again, during(t, 3.6, 4.0));
            const [x, y] = along(rail, during(t, 3.9, 5.4));
            set(dot, { cx: x, cy: y });
            show(dot, t > 3.9 && t < 5.4 ? 1 : 0);
          };
        },
      },
      {
        seconds: 8.5,
        caption: "A reply is kept only if it is entirely right and ends itself.\nA story with two of its three words is thrown away.",
        draw(s) {
          const ROW = (i) => 182 + i * 62;
          const prompt = s.chips(640, 100, [PROMPT], { center: true, size: 24 })[0];
          const heads = [s.text(640, 168, "requested words found", { size: 22, fill: "muted" }), s.text(950, 168, "the story", { size: 22, fill: "muted" })];
          const rows = group.map((r, i) => {
            const y = ROW(i);
            const all = s.group();
            const text = all.group();
            const verdict = all.group();
            text.chip(60, y, r.text, { size: 22 });
            const asks = verdict.chips(640, y, words, { size: 22, pad: 9, gap: 6 });
            asks.forEach((c, j) => {
              const there = r.found.includes(words[j]);
              c.ground.setAttribute("class", there ? "f-s3" : "f-surface s-axis");
              c.label.setAttribute("class", `film-mono ${there ? "f-surface" : "f-muted"}`);
            });
            verdict.text(950, y + 27, r.finished ? "ends" : "is cut off", { size: 22, fill: r.finished ? "ink" : "muted" });
            const fate = all.text(1088, y + 27, kept[i] ? "kept" : "thrown away", { size: 24, weight: kept[i] ? 700 : 400, fill: kept[i] ? "s3" : "muted" });
            return { all, text, verdict, fate };
          });
          // what was kept goes on to be the data of a fine-tuning
          const first = kept.indexOf(true);
          const data = s.chip(60, ROW(first), group[first].text, { size: 22, stroke: "s3" });
          const on = s.group();
          on.arrow(data.w + 76, 508, data.w + 140, 508);
          card(on, data.w + 154, 462, 330, 92, "fine-tune on it", "as if it were data", { stroke: "s2" });
          return (t) => {
            show(prompt, during(t, 0.3, 0.8));
            heads.forEach((h) => show(h, during(t, 1.9, 2.3)));
            rows.forEach((row, i) => {
              show(row.text, during(t, 0.9 + i * 0.2, 1.3 + i * 0.2));
              show(row.verdict, during(t, 2.2 + i * 0.5, 2.6 + i * 0.5));
              show(row.fate, during(t, 2.5 + i * 0.5, 2.9 + i * 0.5));
              show(row.all, kept[i] ? 1 : mix(1, 0.35, during(t, 4.6, 5.2)));
            });
            show(data, t > 5.0 ? 1 : 0);
            place(data, 60, mix(ROW(first), 489, during(t, 5.0, 6.0)));
            show(on, during(t, 5.8, 6.3));
          };
        },
      },
      {
        seconds: 8.5,
        caption: "On the words task it works, and more slowly than GRPO.\nEvery round keeps more stories, and later rounds add less.",
        draw(s) {
          const px = (n) => 140 + (540 * n) / most;
          const py = (v) => 470 - (330 * (v - 0.3)) / 0.7;
          for (const tick of [0.4, 0.6, 0.8, 1]) {
            s.line(140, py(tick), 680, py(tick), { stroke: "grid", width: 1.5 });
            s.text(128, py(tick) + 8, percent(tick), { size: 22, anchor: "end", fill: "muted" });
          }
          s.line(140, 470, 680, 470, { stroke: "axis" });
          for (const r of ours.filter((_, i) => i % 2 === 0)) s.text(px(r.n), 500, r.n.toLocaleString("en"), { size: 22, anchor: "middle", fill: "muted" });
          s.text(410, 532, "stories sampled", { size: 22, anchor: "middle", fill: "muted" });
          s.text(140, 124, "requested words used, held-out prompts", { size: 22, fill: "muted" });

          const spot = (r) => [px(r.n), py(r.score)];
          const theirLine = s.path(theirs.map(spot), { stroke: "s1", width: 3 });
          const theirDots = theirs.map((r) => s.circle(px(r.n), py(r.score), 5, { fill: "s1" }));
          const theirEnd = s.group();
          theirEnd.text(694, py(theirs.at(-1).score) + 2, percent(theirs.at(-1).score), { size: 28, weight: 700, fill: "s1" });
          theirEnd.text(694, py(theirs.at(-1).score) + 28, "GRPO", { size: 22, fill: "ink2" });
          const ourLine = s.path(ours.map(spot), { stroke: "s2", width: 4 });
          const ourDots = ours.map((r) => s.circle(px(r.n), py(r.score), 6, { fill: "s2" }));
          // what a round added, below the middle of its piece of the line
          const gains = ours.slice(1).map((r, i) => s.text((px(r.n) + px(ours[i].n)) / 2, (py(r.score) + py(ours[i].score)) / 2 + 40, `+${(100 * (r.score - ours[i].score)).toFixed(0)}`, { size: 22, anchor: "middle", fill: "ink2" }));
          const ourEnd = s.group();
          ourEnd.text(694, py(ours.at(-1).score) + 2, percent(ours.at(-1).score), { size: 28, weight: 700, fill: "s2" });
          ourEnd.text(694, py(ours.at(-1).score) + 28, "self-training", { size: 22, fill: "ink2" });

          // what every round kept of the stories it sampled
          const head = s.text(900, 150, `kept, of ${perRound.toLocaleString("en")} stories`, { size: 22, fill: "muted" });
          const keeps = ours.slice(1).map((r, i) => {
            const y = 172 + i * 50;
            const g = s.group();
            g.text(900, y + 25, `round ${i + 1}`, { size: 22, fill: "ink2" });
            g.rect(996, y, 150, 34, { fill: "tint", r: 4 });
            g.text(1220, y + 25, r.kept.toLocaleString("en"), { size: 22, anchor: "end", weight: 600 });
            return { g, w: (150 * r.kept) / perRound, bar: g.rect(996, y, 0, 34, { fill: "s2", r: 4 }) };
          });
          return (t) => {
            const far = most * during(t, 0.5, 1.9);
            trace(theirLine, before(theirs.map(spot), theirs.map((r) => r.n), far));
            show(theirLine, t > 0.4 ? 1 : 0);
            theirDots.forEach((dot, i) => show(dot, t > 0.4 && far >= theirs[i].n ? 1 : 0));
            show(theirEnd, during(t, 1.9, 2.3));
            const at = (i) => 2.6 + (i - 1) * 0.55; // when round i starts to be drawn
            const done = ours.slice(1).reduce((sum, _, i) => sum + during(t, at(i + 1), at(i + 1) + 0.45), 0);
            trace(ourLine, before(ours.map(spot), ours.map((r) => r.n), done * perRound));
            show(ourLine, t > 2.2 ? 1 : 0);
            ourDots.forEach((dot, i) => show(dot, i ? during(t, at(i) + 0.3, at(i) + 0.5) : during(t, 2.1, 2.4)));
            gains.forEach((gain, i) => show(gain, during(t, at(i + 1) + 0.35, at(i + 1) + 0.6)));
            show(head, during(t, 2.4, 2.8));
            keeps.forEach(({ g, w, bar }, i) => {
              show(g, during(t, at(i + 1), at(i + 1) + 0.3));
              set(bar, { width: w * during(t, at(i + 1) + 0.1, at(i + 1) + 0.55) });
            });
            show(ourEnd, during(t, 5.9, 6.3));
          };
        },
      },
      {
        seconds: 7,
        caption: "GRPO needed a penalty to keep writing stories.\nSelf-training ends in the same place without one.",
        draw(s) {
          const models = [
            ["sft-words", ["fine-tuned, the start"], 0.4],
            ["grpo-words", ["GRPO"], 0.9],
            ["grpo-words-kl", ["GRPO through LoRA", "with a KL penalty"], 1.4],
            ["self-words", ["self-training"], 3.2],
          ];
          const strangest = Math.max(...models.map(([name]) => stories[name].strangeness));
          const measures = [
            ["words_used", "requested words used", 1, (v) => percent(v)],
            ["cut_off", "stories cut off", 1, (v) => percent(v)],
            ["strangeness", "strangeness", strangest, (v) => v.toFixed(2)],
          ];
          const FULL = 180;
          const left = (j) => 370 + j * 290;
          measures.forEach(([, title], j) => s.text(left(j), 132, title, { size: 24, weight: 600 }));
          const rows = models.map(([name, label, start], i) => {
            const g = s.group();
            const y = 172 + i * 104;
            const mine = name === "self-words";
            g.lines(60, label.length > 1 ? y + 12 : y + 27, label, { size: 24, leading: 1.2, weight: mine ? 700 : 400 });
            const cells = measures.map(([key, , full, format], j) => {
              const w = (FULL * stories[name][key]) / full;
              g.rect(left(j), y, FULL, 36, { fill: "tint", r: 4 });
              return { w, bar: g.rect(left(j), y, 0, 36, { fill: mine ? "s2" : "s1", r: 4 }), value: g.text(left(j) + w + 10, y + 27, format(stories[name][key]), { size: 24, weight: 600 }) };
            });
            return { g, cells, start };
          });
          return (t) => {
            rows.forEach((row) => {
              show(row.g, during(t, row.start, row.start + 0.4));
              row.cells.forEach((cell, j) => {
                const u = during(t, row.start + 0.2 + j * 0.15, row.start + 1.1 + j * 0.15);
                set(cell.bar, { width: cell.w * u });
                set(cell.value, { x: left(j) + cell.w * u + 10 });
                show(cell.value, during(t, row.start + 0.2, row.start + 0.5));
              });
            });
          };
        },
      },
      {
        seconds: 8.5,
        caption: "A right solution to a math problem and a wrong one look alike.\nFine-tuning on right ones teaches what the model does already.",
        draw(s) {
          const asking = s.lines(640, 126, problem, { size: 24, anchor: "middle", fill: "ink2", leading: 1.3 });
          const replies = [mathStart.samples[which], mathSelf.samples[which]].map((sample, i) => {
            const g = s.group();
            const x = 60 + i * 600;
            const right = sample.score === 1;
            const lines = wrap(sample.reply, 38);
            g.rect(x, 186, 560, 240, { fill: "raised", stroke: "axis" });
            const tag = g.text(x + 536, 224, right ? "right" : "wrong", { size: 24, weight: 700, anchor: "end", fill: right ? "s3" : "muted" });
            g.text(x + 24, 224, `reply ${i + 1}`, { size: 22, fill: "muted" });
            lines.forEach((line, j) => g.text(x + 24, 262 + j * 30, line, { size: 22, mono: true, fill: j === lines.length - 1 ? (right ? "s3" : "ink") : "ink2", weight: j === lines.length - 1 ? 700 : 400 }));
            return { g, tag };
          });
          const FULL = 900; // a loss of 1
          const losses = [
            ["loss on its kept replies, round 1", firstRound.loss, "s2"],
            ["loss on reference solutions", reference.val_loss, "s1"],
          ].map(([name, loss, fill], i) => {
            const g = s.group();
            const y = 462 + i * 56;
            g.text(60, y + 25, name, { size: 22, fill: "ink2" });
            const value = g.text(380 + FULL * loss + 12, y + 26, loss.toFixed(2), { size: 24, weight: 700 });
            return { g, w: FULL * loss, value, bar: g.rect(380, y, 0, 36, { fill, r: 4 }) };
          });
          return (t) => {
            asking.forEach((line, i) => show(line, during(t, 0.3 + i * 0.3, 0.7 + i * 0.3)));
            replies.forEach(({ g, tag }, i) => {
              show(g, during(t, 1.3 + i * 0.6, 1.8 + i * 0.6));
              show(tag, during(t, 3.0 + i * 0.3, 3.4 + i * 0.3));
            });
            losses.forEach(({ g, w, value, bar }, i) => {
              const u = during(t, 4.6 + i * 0.7, 5.5 + i * 0.7);
              show(g, during(t, 4.4 + i * 0.7, 4.8 + i * 0.7));
              set(bar, { width: w * u });
              set(value, { x: 380 + w * u + 12 });
            });
          };
        },
      },
      {
        seconds: 7.5,
        caption: `On the words task it gains ${points(wordsSelf.mean - wordsStart.mean)} points, and GRPO gains ${points(wordsGrpo.mean - wordsStart.mean)}.\nOn GSM8K it does nothing, and GRPO gains ${points(mathGrpo.mean - mathStart.mean)} points.`,
        draw(s) {
          const FLOOR = 492;
          const tall = (v) => 300 * v;
          s.text(60, 128, "requested words used", { size: 24, weight: 600 });
          s.text(60, 158, `${wordsSelf.n.toLocaleString("en")} test prompts`, { size: 22, fill: "muted" });
          const stories_ = standing(
            s,
            76,
            FLOOR,
            tall,
            [
              [["fine-tuned"], wordsStart, false],
              [["self-training"], wordsSelf, true],
              [["GRPO with", "a KL penalty"], wordsLeashed, false],
              [["GRPO"], wordsGrpo, false],
            ],
            0.4,
          );
          s.text(720, 128, "GSM8K problems solved", { size: 24, weight: 600 });
          s.text(720, 158, `${mathSelf.n.toLocaleString("en")} test problems`, { size: 22, fill: "muted" });
          const math = standing(
            s,
            736,
            FLOOR,
            tall,
            [
              [["fine-tuned"], mathStart, false],
              [["self-training"], mathSelf, true],
              [["GRPO"], mathGrpo, false],
            ],
            3.2,
          );
          return (t) => {
            stories_(t);
            math(t);
          };
        },
      },
    ],
  };
};
