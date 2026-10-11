// Stage 5 as a film: a published model is three files, and each of them
// becomes something this repository already has. Its weights are a file of
// named tensors, its tokenizer is the one of stage 1 with other numbers, its
// chat is a format of text, and tests hold it against the reference. The
// speeds are those of the benchmarks on the 0.5B model.

import { during, mix, set, show, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  small: "/models/base-q4/meta.json",
  qwen: "/runs/bench/inference-qwen-mps.json",
  step: "/runs/bench/train-qwen-mps.json",
  before: "/runs/bench/train-mps.json",
};

const NAME = "Qwen2.5-0.5B-Instruct";
const MONO = 0.602; // the width of a character of the monospaced font in ems, as stage.js has it
const whole = (n) => Math.round(n).toLocaleString("en");
const plural = (n, word) => `${whole(n)} ${word}${n === 1 ? "" : "s"}`;

// A box with a title and a line below it.
function card(s, x, y, w, title, sub, { mono = false, fill = "raised" } = {}) {
  const g = s.group();
  g.rect(x, y, w, 88, { fill, stroke: "axis", r: 12 });
  g.text(x + w / 2, y + 38, title, { size: 26, weight: 600, anchor: "middle", mono });
  g.text(x + w / 2, y + 70, sub, { size: 22, anchor: "middle", fill: "ink2" });
  return g;
}

// A line of monospaced text from x, in pieces of [string, color]. Returns the pieces.
function typed(s, x, y, parts, size = 22) {
  let at = x;
  return parts.map(([string, fill = "ink"]) => {
    const node = s.text(at, y, string, { size, mono: true, fill });
    at += [...string].length * size * MONO;
    return node;
  });
}

export default ({ vocabulary, small, qwen, step, before }) => {
  const times = qwen.parameters / small.parameters;
  const gain = (row) => row.uncached_seconds / row.cached_seconds;
  const lengths = [...qwen.length].sort((a, b) => a.batch - b.batch || a.prompt_tokens - b.prompt_tokens);
  const alone = qwen.batch.find((row) => row.batch === 1);
  const most = qwen.batch.at(-1);
  const usual = (bench) => bench.configs.find((config) => config.compile && config.amp); // both switches of stage 3

  return {
    subtitle: "A published model in the same code",
    scenes: [
      {
        seconds: 6.5,
        caption: "The model of stage 3 writes stories and knows nothing else.\nThe later stages need one that already knows something.",
        draw(s) {
          // two squares, as large as the models are
          const BIG = 330;
          const LITTLE = BIG / Math.sqrt(times);
          const FLOOR = 452;
          s.rect(300 - LITTLE / 2, FLOOR - LITTLE, LITTLE, LITTLE, { fill: "s1", r: 6 });
          s.text(300, 490, "the model of stage 3", { size: 26, weight: 600, anchor: "middle" });
          s.text(300, 524, `${whole(small.parameters)} parameters`, { size: 24, anchor: "middle", mono: true });
          s.text(300, 558, "trained here, on stories", { size: 22, anchor: "middle", fill: "ink2" });
          const other = s.group();
          const square = other.rect(820, FLOOR, 0, 0, { fill: "s2", r: 6 });
          other.text(820, 490, NAME, { size: 26, weight: 600, anchor: "middle" });
          const count = other.text(820, 524, "", { size: 24, anchor: "middle", mono: true });
          const more = s.text(820, 558, `${Math.round(times)} times as many, and published`, { size: 22, anchor: "middle", weight: 700, fill: "s2" });
          return (t) => {
            const side = mix(LITTLE, BIG, during(t, 1.6, 3.6));
            show(other, during(t, 1.2, 1.6));
            set(square, { x: 820 - side / 2, y: FLOOR - side, width: side, height: side });
            count.textContent = `${whole(small.parameters * (side / LITTLE) ** 2)} parameters`;
            show(more, during(t, 3.8, 4.2));
          };
        },
      },
      {
        seconds: 7,
        caption: "A published model is three files. Each becomes something\nthis repository has already: Config, Transformer, Tokenizer.",
        draw(s) {
          const rows = [
            ["config.json", "the shape of the model", "Config", "24 layers, width 896, 14 heads"],
            ["model.safetensors", `${whole(qwen.parameters)} numbers`, "Transformer", "the code of stage 2, with these weights"],
            ["vocab.json, merges.txt", "a vocabulary and its merges", "Tokenizer", "the byte pair encoding of stage 1"],
          ].map(([file, holds, thing, what], i) => {
            const y = 122 + i * 132;
            return { file: card(s, 100, y, 400, file, holds, { mono: true, fill: "tint" }), arrow: s.arrow(516, y + 44, 644, y + 44), thing: card(s, 660, y, 520, thing, what) };
          });
          const open = s.text(640, 552, "Nothing about the model stays hidden behind a library.", { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            rows.forEach((row, i) => {
              const from = 0.4 + i * 1.2;
              show(row.file, during(t, from, from + 0.4));
              show(row.arrow, during(t, from + 0.4, from + 0.7));
              show(row.thing, during(t, from + 0.6, from + 1));
            });
            show(open, during(t, 4.3, 4.8));
          };
        },
      },
      {
        seconds: 7,
        caption: "The weights are a file of named tensors: a header, then raw\nnumbers. Each name is translated, and all 290 find a place.",
        draw(s) {
          // the three parts of a safetensors file, not to scale
          const parts = [
            [100, 190, "8 bytes", "the header's length"],
            [300, 420, "a header, as JSON", "per tensor: name, type, shape, bytes"],
            [730, 450, "raw numbers", "the tensors, one after another"],
          ].map(([x, w, title, sub], i) => {
            const g = s.group();
            g.rect(x, 122, w, 64, { fill: i === 2 ? "tint" : "raised", stroke: "axis", r: 8 });
            g.text(x + w / 2, 163, title, { size: 24, weight: 600, anchor: "middle" });
            g.text(x + w / 2, 218, sub, { size: 22, anchor: "middle", fill: "muted" });
            return g;
          });
          const theirs = s.group();
          theirs.text(100, 300, "a name in the file", { size: 22, fill: "muted" });
          typed(theirs, 100, 338, [["model.", "s2"], ["layers.0."], ["self_attn", "s2"], [".q_proj.weight"]], 24);
          const ours = s.group();
          ours.arrow(118, 356, 118, 420);
          ours.text(142, 396, "to_minllm drops a prefix and renames four parts", { size: 22, fill: "ink2" });
          ours.text(100, 458, "its name in this repository", { size: 22, fill: "muted" });
          typed(ours, 100, 496, [["layers.0."], ["attn", "s2"], [".q_proj.weight"]], 24);
          const shape = s.text(100, 548, "the query projection of the first block, 896 by 896", { size: 22, fill: "ink2" });
          const all = s.group();
          const count = all.text(800, 366, "", { size: 72, weight: 700 });
          all.text(800, 406, "named tensors in the file", { size: 24, fill: "ink2" });
          const placed = s.group();
          placed.circle(816, 462, 16, { fill: "s3" });
          placed.path([[808, 462], [814, 469], [825, 455]], { stroke: "surface", width: 4 });
          placed.text(844, 470, "every one finds its place", { size: 24, weight: 600 });
          placed.text(800, 512, "A missing, spare or misshapen", { size: 22, fill: "ink2" });
          placed.text(800, 542, "tensor would raise an error.", { size: 22, fill: "ink2" });
          return (t) => {
            parts.forEach((g, i) => show(g, during(t, 0.3 + i * 0.4, 0.7 + i * 0.4)));
            show(theirs, during(t, 1.7, 2.1));
            show(ours, during(t, 2.5, 3));
            show(shape, during(t, 3, 3.4));
            show(all, during(t, 3.5, 3.9));
            count.textContent = String(Math.round(290 * during(t, 3.5, 4.4)));
            show(placed, during(t, 4.5, 5));
          };
        },
      },
      {
        seconds: 6.5,
        caption: "The tokenizer is the byte pair encoding of stage 1.\nWhat differs is bookkeeping, such as how tokens are numbered.",
        draw(s) {
          const mine = 256 + vocabulary.merges.length + vocabulary.special_tokens.length;
          s.text(620, 150, "stage 1", { size: 26, weight: 600, anchor: "middle" });
          s.text(960, 150, "Qwen2.5", { size: 26, weight: 600, anchor: "middle" });
          s.line(100, 172, 1180, 172, { stroke: "axis" });
          const rows = [
            ["the byte “!” is", `token ${"!".charCodeAt(0)}`, "token 0"],
            ["the space is", `token ${" ".charCodeAt(0)}`, "token 220"],
            ["special tokens", String(vocabulary.special_tokens.length), "22"],
            ["tokens it knows", whole(mine), "151,665"],
          ].map(([what, here, there], i) => {
            const g = s.group();
            const y = 218 + i * 68;
            g.text(100, y, what, { size: 24 });
            g.text(620, y, here, { size: 26, anchor: "middle", mono: true });
            g.text(960, y, there, { size: 26, anchor: "middle", mono: true });
            g.line(100, y + 24, 1180, y + 24, { stroke: "grid", width: 1.5 });
            return g;
          });
          const table = s.text(100, 510, "Qwen's embedding table has 151,936 rows.", { size: 24, fill: "ink2" });
          const spare = s.text(100, 546, "The last 271 are spare: never produced, and never trained.", { size: 24, fill: "ink2" });
          return (t) => {
            rows.forEach((g, i) => show(g, during(t, 0.5 + i * 0.7, 0.9 + i * 0.7)));
            show(table, during(t, 3.6, 4));
            show(spare, during(t, 4.1, 4.5));
          };
        },
      },
      {
        seconds: 7,
        caption: "A chat model still predicts the next token. It was trained\non conversations written in one fixed format of text.",
        draw(s) {
          const START = ["<|im_start|>", "s2"];
          const END = ["<|im_end|>", "s2"];
          s.rect(84, 122, 1112, 290, { fill: "tint", stroke: "axis", r: 12 });
          const lines = [
            [START, ["system"]],
            [["You are Qwen, created by Alibaba Cloud. You are a helpful assistant."], END],
            [START, ["user"]],
            [["What is the capital of France?"], END],
            [START, ["assistant"]],
          ].map((parts, i) => {
            const g = s.group();
            typed(g, 110, 162 + i * 44, parts);
            return g;
          });
          const GOES = "the model goes on from here, token by token, up to its own ";
          const answer = s.text(110, 162 + 5 * 44, "", { size: 22, mono: true, fill: "muted" });
          const stop = s.text(110 + GOES.length * 22 * MONO, 162 + 5 * 44, END[0], { size: 22, mono: true, fill: "s2" });
          const notes = s.lines(110, 484, ["The orange markers are control tokens, and only the code writes them.", "Typed by a user, the same characters stay ordinary text."], { size: 24, fill: "ink2", leading: 1.5 });
          return (t) => {
            lines.forEach((g, i) => show(g, during(t, 0.3 + i * 0.45, 0.6 + i * 0.45)));
            answer.textContent = GOES.slice(0, Math.round(GOES.length * during(t, 2.7, 4.1)));
            show(stop, during(t, 4.1, 4.4));
            notes.forEach((line, i) => show(line, during(t, 4.5 + i * 0.3, 4.9 + i * 0.3)));
          };
        },
      },
      {
        seconds: 6.5,
        caption: "Loading without an error does not prove the model is right.\nFive tests compare it with the reference implementation.",
        draw(s) {
          card(s, 100, 116, 420, "this repository", "the code of stage 2, with the weights");
          card(s, 760, 116, 420, "transformers", "the reference implementation");
          s.arrow(624, 160, 536, 160);
          s.arrow(656, 160, 744, 160);
          s.text(640, 138, "must agree", { size: 22, anchor: "middle", fill: "muted" });
          const checks = [
            ["the rendered conversation, character for character", "test_chat_format_matches_reference"],
            ["the token ids of that conversation", "test_chat_encoding"],
            ["the token ids of 14 texts, from English to emoji", "test_tokenizer_matches_reference"],
            ["the ten likeliest next tokens, logits within 0.001", "test_logits_match_reference"],
            ["32 tokens of greedy generation, token for token", "test_generation_matches_reference"],
          ].map(([what, test], i) => {
            const y = 268 + i * 62;
            const said = s.group();
            said.text(160, y + 8, what, { size: 24 });
            said.text(1180, y + 8, test, { size: 22, anchor: "end", mono: true, fill: "muted" });
            const done = s.group();
            done.circle(120, y, 16, { fill: "s3" });
            return { said, done, tick: done.path([[112, y], [118, y + 7], [129, y - 7]], { stroke: "surface", width: 4 }) };
          });
          return (t) => {
            checks.forEach((check, i) => {
              show(check.said, during(t, 0.5 + i * 0.3, 0.9 + i * 0.3));
              show(check.done, during(t, 2.2 + i * 0.45, 2.5 + i * 0.45));
              trace(check.tick, during(t, 2.4 + i * 0.45, 2.7 + i * 0.45));
            });
          };
        },
      },
      {
        seconds: 7.5,
        caption: `The cache pays once the prompt is long. One sequence generates\n${whole(alone.tokens_per_second)} tokens per second, and a batch of ${most.batch} generates ${whole(most.tokens_per_second)}.`,
        draw(s) {
          const peak = Math.max(...lengths.map(gain));
          s.text(80, 144, `Faster with the cache, for ${lengths[0].new_tokens} new tokens`, { size: 26, weight: 600 });
          const rows = lengths.map((row, i) => {
            const y = 182 + i * 68;
            const same = gain(row) < 1.5;
            s.text(80, y + 29, `${plural(row.batch, "prompt")} of ${whole(row.prompt_tokens)} tokens`, { size: 22 });
            return {
              w: (230 * gain(row)) / peak,
              bar: s.rect(366, y, 0, 40, { fill: same ? "s2" : "s1", r: 5 }),
              said: s.text(366, y + 29, same ? "no gain" : `${gain(row).toFixed(gain(row) < 10 ? 1 : 0)}× faster`, { size: 24, weight: 700, fill: same ? "s2" : "ink" }),
            };
          });
          s.line(760, 120, 760, 540, { stroke: "grid", width: 1.5 });
          s.text(800, 144, "Tokens per second", { size: 26, weight: 600 });
          const training = usual(step);
          const speeds = [
            [alone.tokens_per_second, ["generating one sequence"]],
            [most.tokens_per_second, [`generating a batch of ${most.batch}`]],
            [training.tokens_per_second, [`in a training step of ${whole(step.batch_size * step.seq_len)} tokens,`, `which takes ${training.memory_gb.toFixed(0)} GB of memory.`, `The 14M model: ${whole(usual(before).tokens_per_second)}.`]],
          ].map(([value, about], i) => {
            const g = s.group();
            const y = 216 + i * 112;
            return { g, value, number: g.text(800, y, "", { size: 46, weight: 700 }), about: g.lines(800, y + 34, about, { size: 22, fill: "ink2" }) };
          });
          return (t) => {
            rows.forEach((row, i) => {
              const u = during(t, 0.4 + i * 0.45, 1.2 + i * 0.45);
              set(row.bar, { width: row.w * u });
              set(row.said, { x: 366 + row.w * u + 12 });
              show(row.said, during(t, 0.9 + i * 0.45, 1.3 + i * 0.45));
            });
            speeds.forEach((speed, i) => {
              const from = 2.8 + i * 0.7;
              show(speed.g, during(t, from, from + 0.4));
              speed.number.textContent = whole(speed.value * during(t, from, from + 0.9));
            });
          };
        },
      },
    ],
  };
};
