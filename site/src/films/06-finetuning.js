// Stage 6 as a film: a conversation in the chat format becomes training data,
// the loss counts only on the reply, and the same training runs on every
// weight or on two thin matrices beside a frozen one. The numbers are those
// of the logged fine-tuning runs and of the scores of the story model.

import { Tokenizer } from "../labs/bpe.js";
import { during, mix, place, show, spelled, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  words: "/runs/sft-words/meta.json",
  before: "/runs/eval/words-base.json",
  after: "/runs/eval/words-sft-words.json",
  full: "/runs/sft-gsm8k/meta.json",
  lora: "/runs/sft-gsm8k-lora/meta.json",
  fullLog: "/runs/sft-gsm8k/log.jsonl",
  loraLog: "/runs/sft-gsm8k-lora/log.jsonl",
};

const START = "<|im_start|>";
const END = "<|im_end|>";
const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js

// The example of the note: an instruction, and a story that is a right answer to it.
const TALK = [
  { role: "user", content: "Write a short story using these words: garden, kitten." },
  { role: "assistant", content: "Lily found a kitten in the garden. They played all day." },
];

// chat.training_example of the repository: the tokens of a conversation, and
// which of them the loss counts. last marks the end of a line of the format.
function example(tok, messages) {
  const tokens = [];
  for (const m of messages) {
    const header = [tok.special.get(START), ...tok.encode(`${m.role}\n`)];
    const content = [...tok.encode(m.content), tok.special.get(END)];
    header.forEach((id, i) => tokens.push({ id, target: false, last: i === header.length - 1 }));
    for (const id of content) tokens.push({ id, target: m.role === "assistant", last: false });
    for (const id of tok.encode("\n")) tokens.push({ id, target: false, last: true });
  }
  return tokens;
}

const percent = (v) => `${(100 * v).toFixed(1)}%`;
const count = (v) => Math.round(v).toLocaleString("en");
const millions = (v) => `${(v / 1e6).toFixed(v < 1e8 ? 1 : 0)}M`;
// The start of a reply, cut at a word.
const excerpt = (reply, n) => {
  const flat = reply.replace(/\s+/g, " ");
  return flat.length <= n ? flat : `${flat.slice(0, flat.lastIndexOf(" ", n - 1))} …`;
};

// A line of monospaced text in which the given words are marked.
function marked(s, x, y, string, words, size) {
  let at = x;
  return string.split(new RegExp(`\\b(${words.join("|")})\\b`, "i")).map((part, i) => {
    const node = s.text(at, y, part, { size, mono: true, fill: i % 2 ? "s2" : "ink", weight: i % 2 ? 700 : 400 });
    at += part.length * size * MONO;
    return node;
  });
}

export default ({ vocabulary, words, before, after, full, lora, fullLog, loraLog }) => {
  const tok = new Tokenizer(vocabulary);
  const spell = (id) => spelled(tok.text(id));
  const tokens = example(tok, TALK);
  const targets = tokens.filter((token) => token.target).length;
  const all = (run) => run.scores.filter((score) => score === 1).length / run.n; // stories that use every word
  const held = (log) => log.filter((row) => row.val_loss !== undefined).map((row) => [row.step, row.val_loss]);
  const runs = [
    { name: "every weight", meta: full, curve: held(fullLog), color: "s1" },
    { name: `LoRA, rank ${lora.lora.r}`, meta: lora, curve: held(loraLog), color: "s2" },
  ];
  const steps = full.step;

  return {
    subtitle: "Teaching a pretrained model what to do",
    scenes: [
      {
        seconds: 7,
        caption: "Fine-tuning continues training on examples of what is wanted:\nconversations, written in the chat format of stage 5.",
        draw(s) {
          const size = 26;
          const step = size * MONO;
          const width = (TALK[1].content.length + END.length) * step;
          const x = 640 - width / 2;
          const rows = TALK.map((m, i) => {
            const y = 206 + i * 142;
            return {
              content: s.text(x, y + 48, m.content, { size, mono: true }),
              label: s.text(x + width, y, ["what is asked", "what is wanted: the reply"][i], { size: 24, anchor: "end", fill: "ink2" }),
              format: [
                s.text(x, y, START, { size, mono: true, fill: "muted" }),
                s.text(x + START.length * step, y, m.role, { size, mono: true, fill: "s1" }),
                s.text(x + m.content.length * step, y + 48, END, { size, mono: true, fill: "muted" }),
              ],
            };
          });
          const note = s.text(640, 496, `${START} opens a turn and ${END} closes it. Each is one token.`, { size: 24, anchor: "middle", fill: "muted" });
          return (t) => {
            rows.forEach((row, i) => {
              show(row.content, during(t, 0.3 + i * 0.9, 0.9 + i * 0.9));
              show(row.label, during(t, 0.5 + i * 0.9, 1.1 + i * 0.9));
              for (const mark of row.format) show(mark, during(t, 2.8 + i * 0.5, 3.4 + i * 0.5));
            });
            show(note, during(t, 4.4, 5));
          };
        },
      },
      {
        seconds: 8,
        caption: "The loss counts only where the model should learn to write:\nthe reply, and the token that ends it.",
        draw(s) {
          // the conversation as tokens, one row for each line of the format
          const lines = [[]];
          for (const token of tokens) {
            lines.at(-1).push(token);
            if (token.last) lines.push([]);
          }
          const rows = lines.filter((line) => line.length).map((line, i) => s.chips(0, [118, 172, 248, 302][i], line.map((token) => spell(token.id)), { size: 22, pad: 6, gap: 4 }));
          const left = 640 - Math.max(...rows.map((row) => row.at(-1).x + row.at(-1).w)) / 2;
          for (const row of rows) for (const c of row) place(c, c.x + left, c.y);
          const chips = rows.flat();
          const rowOf = rows.flatMap((row, k) => row.map(() => k));
          const legend = [
            ["context", "the instruction and the format: read, and ignored by the loss", "tint", "muted", 378],
            ["target", `the reply, and the ${END} that ends it`, "s2", "surface", 428],
          ].map(([name, what, fill, ink, y]) => {
            const g = s.group();
            g.chip(left, y, name, { size: 22, pad: 8, fill, ink, stroke: fill === "s2" ? null : "axis" });
            g.text(left + 126, y + 27, what, { size: 24, fill: "ink2" });
            return g;
          });
          const sum = s.text(640, 530, `${targets} of the ${tokens.length} tokens are targets`, { size: 30, weight: 600, anchor: "middle" });
          const order = tokens.map((token, i) => (token.target ? tokens.slice(0, i).filter((other) => other.target).length : -1));
          return (t) => {
            const dim = during(t, 2, 2.8);
            chips.forEach((c, i) => {
              const on = order[i] >= 0 && t >= 3.6 + order[i] * 0.14;
              show(c, during(t, 0.3 + rowOf[i] * 0.3, 0.8 + rowOf[i] * 0.3) * (tokens[i].target ? 1 : mix(1, 0.6, dim)));
              c.ground.setAttribute("class", on ? "f-s2 s-s2" : "f-tint s-axis");
              c.label.setAttribute("class", `film-mono ${on ? "f-surface" : tokens[i].target ? "f-ink" : dim > 0.5 ? "f-muted" : "f-ink"}`);
            });
            show(legend[0], during(t, 2.2, 2.8));
            show(legend[1], during(t, 3.6, 4.2));
            show(sum, during(t, 5.7, 6.3));
          };
        },
      },
      {
        seconds: 9,
        caption: "Fine-tuning changes a weight matrix W into W + ΔW.\nLoRA freezes W and trains ΔW as two thin matrices, B and A.",
        draw(s) {
          const [x, y, d, r] = [100, 190, 220, 34];
          // W, a matrix of the pretrained model
          const weight = s.group();
          weight.rect(x, y, d, d, { fill: "tint", stroke: "axis" });
          for (let i = 1; i < 5; i++) {
            weight.line(x + (i * d) / 5, y + 4, x + (i * d) / 5, y + d - 4, { stroke: "grid", width: 1.5 });
            weight.line(x + 4, y + (i * d) / 5, x + d - 4, y + (i * d) / 5, { stroke: "grid", width: 1.5 });
          }
          weight.text(x + d / 2, y + d / 2 + 20, "W", { size: 56, weight: 700, anchor: "middle", fill: "ink2" });
          const pretrained = s.text(x + d / 2, y + d + 40, "pretrained", { size: 24, anchor: "middle", fill: "ink2" });
          const frozen = s.text(x + d / 2, y + d + 40, "frozen", { size: 24, weight: 600, anchor: "middle" });
          const plus = s.text(x + d + 35, y + d / 2 + 16, "+", { size: 44, anchor: "middle", fill: "muted" });
          // the change to it: a matrix of the same size, or the product of two thin ones
          const x2 = x + d + 70;
          const change = s.group();
          change.rect(x2, y, d, d, { fill: "s2" });
          change.text(x2 + d / 2, y + d / 2 + 20, "ΔW", { size: 56, weight: 700, anchor: "middle", fill: "surface" });
          change.text(x2 + d / 2, y + d + 40, "as large as W, trained", { size: 24, anchor: "middle", fill: "ink2" });
          const thin = s.group();
          thin.rect(x2, y, r, d, { fill: "s2", r: 6 });
          thin.rect(x2 + r + 26, y, d, r, { fill: "s2", r: 6 });
          thin.text(x2 + r / 2, y + d + 40, "B", { size: 30, weight: 700, anchor: "middle", fill: "s2" });
          thin.text(x2 + r + 26 + d / 2, y + r + 42, "A", { size: 30, weight: 700, anchor: "middle", fill: "s2" });
          thin.lines(x2 + r + 26 + d / 2, y + 150, ["two thin matrices", `of rank ${lora.lora.r}, trained`], { size: 24, anchor: "middle", fill: "ink2" });
          // how many numbers each way trains, in the whole model
          const [cx, cw] = [790, 400];
          s.text(cx, 150, "weights trained, in the 0.5B model", { size: 22, fill: "muted" });
          const share = lora.trainable_parameters / full.trainable_parameters;
          const sides = [
            { y: 200, name: "every weight", n: full.trainable_parameters, part: 1, color: "s1" },
            { y: 362, name: `through LoRA, rank ${lora.lora.r}`, n: lora.trainable_parameters, part: share, color: "s2" },
          ].map((side) => ({
            ...side,
            label: s.text(cx, side.y, side.name, { size: 24, fill: "ink2" }),
            number: s.text(cx, side.y + 48, "", { size: 40, weight: 700, fill: side.color }),
            track: s.rect(cx, side.y + 66, cw, 14, { fill: "tint", r: 4 }),
            bar: s.rect(cx, side.y + 66, 0, 14, { fill: side.color, r: 4 }),
          }));
          const part = s.text(cx, 362 + 116, `${percent(share)} of the weights`, { size: 24, weight: 600 });
          return (t) => {
            show(weight, during(t, 0.3, 0.9));
            show(plus, during(t, 0.9, 1.4));
            // the full matrix leaves before the two thin ones come
            const leaving = 1 - during(t, 4, 4.5);
            const lowRank = during(t, 4.5, 5.1);
            show(change, during(t, 0.9, 1.5) * leaving);
            show(thin, lowRank);
            show(pretrained, during(t, 0.5, 1.1) * leaving);
            show(frozen, lowRank);
            sides.forEach((side, i) => {
              const [from, to] = i ? [5.2, 6.6] : [2, 3.4];
              const u = during(t, from, to);
              for (const node of [side.label, side.number, side.track, side.bar]) show(node, during(t, from - 0.2, from + 0.3));
              side.number.textContent = count(side.n * u);
              side.bar.setAttribute("width", (cw * side.part * u).toFixed(1));
            });
            show(part, during(t, 6.6, 7.2));
          };
        },
      },
      {
        seconds: 8,
        caption: `Fine-tuned for ${count(words.step)} steps, the story model answers the\ninstruction. It uses ${percent(after.mean)} of the words, up from ${percent(before.mean)}.`,
        draw(s) {
          const asked = after.samples[0];
          const wanted = asked.prompt.match(/words: (.+)\.$/)[1].split(", ");
          const replies = [before.samples.find((sample) => sample.prompt === asked.prompt), asked];
          const names = ["base model", "fine-tuned"];
          const prompt = s.chips(640, 108, [asked.prompt], { center: true, size: 24 })[0];
          const said = replies.map((sample, i) => {
            const g = s.group();
            g.text(90, 202 + i * 44, names[i], { size: 24, fill: "ink2" });
            marked(g, 250, 202 + i * 44, excerpt(sample.reply, 70), wanted, 22);
            return g;
          });
          const [x, w] = [250, 800];
          const groups = [
            { title: "requested words used", y: 300, values: [before.mean, after.mean] },
            { title: "stories that use all of them", y: 436, values: [all(before), all(after)] },
          ].map((group) => ({
            heading: s.text(90, group.y, group.title, { size: 24, weight: 600 }),
            bars: group.values.map((value, i) => {
              const y = group.y + 18 + i * 40;
              s.rect(x, y, w, 28, { fill: "tint", r: 6 });
              return { value, bar: s.rect(x, y, 0, 28, { fill: i ? "s2" : "s1", r: 6 }), label: s.text(x, y + 22, "", { size: 24, weight: 700 }), name: s.text(90, y + 22, names[i], { size: 24, fill: "ink2" }) };
            }),
          }));
          return (t) => {
            show(prompt, during(t, 0.3, 0.8));
            said.forEach((g, i) => show(g, during(t, 0.9 + i * 0.9, 1.5 + i * 0.9)));
            groups.forEach((group, k) => {
              const u = during(t, 3.2 + k * 1.3, 4.4 + k * 1.3);
              for (const b of group.bars) {
                b.bar.setAttribute("width", (w * b.value * u).toFixed(1));
                b.label.setAttribute("x", (x + w * b.value * u + 12).toFixed(1));
                b.label.textContent = percent(b.value * u);
              }
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `The 0.5B model learns grade-school math twice, ${steps} steps each:\nonce with every weight trained, once through LoRA.`,
        draw(s) {
          const [x, y, w, h] = [170, 140, 740, 360];
          const [low, high] = [0.55, 0.85];
          const px = (step) => x + (w * step) / steps;
          const py = (loss) => y + h * (1 - (loss - low) / (high - low));
          for (const tick of [0.6, 0.7, 0.8]) {
            s.line(x, py(tick), x + w, py(tick), { stroke: "grid", width: 1.5 });
            s.text(x - 14, py(tick) + 8, tick.toFixed(1), { size: 22, anchor: "end", fill: "muted" });
          }
          s.line(x, y + h, x + w, y + h, { stroke: "axis" });
          for (const tick of [0, steps / 3, (2 * steps) / 3, steps]) s.text(px(tick), y + h + 32, count(tick), { size: 22, anchor: "middle", fill: "muted" });
          s.text(x + w / 2, y + h + 68, "training step", { size: 22, anchor: "middle", fill: "muted" });
          s.text(x, y - 18, "loss on held-out problems", { size: 22, fill: "muted" });
          const from = s.text(px(0) + 18, py(runs[0].curve[0][1]) + 8, runs[0].curve[0][1].toFixed(3), { size: 24, weight: 700 });
          const drawn = runs.map((run, i) => {
            const points = run.curve.map(([step, loss]) => [px(step), py(loss)]);
            const path = s.path(points, { stroke: run.color, width: 4 });
            // how far along the line each of its points is
            let far = 0;
            const along = points.map(([a, b], k) => (far += k ? Math.hypot(a - points[k - 1][0], b - points[k - 1][1]) : 0) / path.total);
            const end = run.curve.at(-1)[1];
            const at = py(end) + (i ? 30 : -4); // the two ends are close: one name above its line, one below
            return {
              path,
              along,
              dots: points.map(([a, b]) => s.circle(a, b, 6, { fill: run.color })),
              value: s.text(x + w + 22, at, end.toFixed(3), { size: 24, weight: 700, fill: run.color }),
              name: s.text(x + w + 100, at, run.name, { size: 24, fill: "ink2" }),
            };
          });
          return (t) => {
            show(from, during(t, 0.5, 1));
            for (const run of drawn) {
              const u = during(t, 1, 4.2);
              trace(run.path, u);
              run.dots.forEach((dot, k) => show(dot, u >= run.along[k] - 0.001 ? 1 : 0));
              show(run.name, during(t, 0.5, 1));
              show(run.value, during(t, 4.2, 4.8));
            }
          };
        },
      },
      {
        seconds: 7,
        caption: "LoRA takes half the time and half the memory,\nand ends at a slightly lower held-out loss.",
        draw(s) {
          const measures = [
            { title: "weights trained", value: (meta) => meta.trainable_parameters, format: millions },
            { title: "minutes", value: (meta) => meta.minutes, format: (v) => v.toFixed(1) },
            { title: "memory, GB", value: (meta) => meta.memory_gb, format: (v) => v.toFixed(0) },
            { title: "held-out loss", value: (meta) => meta.val_loss, format: (v) => v.toFixed(3) },
          ];
          const [floor, tall, wide] = [480, 280, 88];
          const sides = [runs[1], runs[0]]; // LoRA on the left of each pair
          sides.forEach((run, i) => {
            const at = [330, 700][i];
            s.rect(at, 116, 24, 24, { fill: run.color, r: 5 });
            s.text(at + 36, 137, i ? "every weight trained" : `through ${run.name}`, { size: 24, fill: "ink2" });
          });
          const panels = measures.map((measure, k) => {
            const x = 112 + k * 275;
            const peak = Math.max(...sides.map((run) => measure.value(run.meta)));
            s.line(x - 10, floor, x + 2 * wide + 34, floor, { stroke: "axis" });
            s.text(x + wide + 12, floor + 40, measure.title, { size: 24, anchor: "middle" });
            return sides.map((run, i) => {
              const value = measure.value(run.meta);
              const bx = x + i * (wide + 24);
              return { value, format: measure.format, height: (tall * value) / peak, bar: s.rect(bx, floor, wide, 0, { fill: run.color, r: 6 }), label: s.text(bx + wide / 2, floor, "", { size: 24, weight: 700, anchor: "middle" }) };
            });
          });
          return (t) => {
            panels.forEach((bars, k) => {
              const u = during(t, 0.5 + k * 0.9, 1.7 + k * 0.9);
              for (const b of bars) {
                b.bar.setAttribute("height", (b.height * u).toFixed(1));
                b.bar.setAttribute("y", (floor - b.height * u).toFixed(1));
                b.label.setAttribute("y", (floor - b.height * u - 12).toFixed(1));
                b.label.textContent = b.format(b.value);
                show(b.label, during(t, 1.3 + k * 0.9, 1.7 + k * 0.9));
              }
            });
          };
        },
      },
    ],
  };
};
