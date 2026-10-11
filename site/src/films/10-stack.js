// Stage 10 as a film: a model of this repository is written out as the
// directory that other tools read, checked against them, run elsewhere and
// served, and every stage is set next to the library used for it in practice.
// The note logs no run of its own, so the film shows what is written and
// compared and leaves the check's numbers out. The shape of the model is read
// from the model the site runs, and the two speeds are those of stage 4.

import { Tokenizer } from "../labs/bpe.js";
import { clamp, during, mix, place, set, show, spelled } from "./stage.js";

export const sources = { vocabulary: "/tokenizer.json", model: "/models/base-q4/meta.json", bench: "/runs/bench/inference-mps.json" };

const MODEL = "sft-words"; // the model the note exports
const REPLY = "Once upon a time, there was a little kitten named Fluffy."; // what it writes there, as the note quotes it

// The table of the note, a row per stage.
const MAP = [
  ["1", "Tokenizer", "tokenizer.py", "tokenizers, tiktoken, SentencePiece"],
  ["2", "Model", "model.py", "the model classes of transformers"],
  ["3", "Pretraining", "scripts/train.py", "accelerate, datasets, plain PyTorch"],
  ["4", "Inference", "inference.py", "vLLM, llama.cpp, MLX, TGI"],
  ["5", "Pretrained weights", "pretrained.py", "from_pretrained, the Hub, safetensors"],
  ["6", "Fine-tuning", "finetune.py, lora.py", "peft, trl, Unsloth"],
  ["7", "Evaluation", "evaluate.py, tasks.py", "lm-evaluation-harness, Inspect"],
  ["8", "RLHF", "reward.py, ppo.py, dpo.py", "trl, OpenRLHF"],
  ["9", "RLVR", "rl.py, scripts/grpo.py", "verl, OpenRLHF, trl"],
  ["10", "Serving", "scripts/serve.py", "vLLM, an OpenAI-compatible endpoint"],
];

// A box with a name and a line about it, both large enough to read in a film.
function card(s, x, y, w, h, title, sub, { stroke = "axis", ink = "ink", size = 26 } = {}) {
  const g = s.group();
  g.rect(x, y, w, h, { fill: "raised", stroke, r: 12 });
  g.text(x + w / 2, y + h / 2 + (sub ? -6 : 9), title, { size, weight: 600, anchor: "middle", fill: ink });
  if (sub) g.text(x + w / 2, y + h / 2 + 26, sub, { size: 22, anchor: "middle", fill: "ink2" });
  return g;
}

export default ({ vocabulary, model, bench }) => {
  const tok = new Tokenizer(vocabulary);
  const cfg = model.config;
  // the rule of export.config_json: which reference class has this shape
  const architecture = cfg.n_experts ? "MixtralForCausalLM" : cfg.qkv_bias ? "Qwen2ForCausalLM" : "LlamaForCausalLM";
  const batch = (n) => bench.batch.find((b) => b.batch === n);

  return {
    subtitle: "From this repository to the tools everyone uses",
    scenes: [
      {
        seconds: 6.5,
        caption: "Stage 5 took a published model apart into three files.\nExport is the same step backwards.",
        draw(s) {
          const source = card(s, 70, 276, 260, 116, `out/${MODEL}`, "a model trained here");
          const step = s.group();
          step.arrow(345, 334, 585, 334);
          step.text(465, 313, "scripts/export.py", { size: 22, anchor: "middle", fill: "ink2", mono: true });
          const directory = s.group();
          directory.rect(600, 120, 610, 428, { fill: "raised", stroke: "axis" });
          directory.text(624, 164, `out/export/${MODEL}/`, { size: 24, fill: "muted", mono: true });
          const files = [
            [["model.safetensors"], "the weights, under the reference names"],
            [["config.json"], "the model's shape, in the reference vocabulary"],
            [["tokenizer.json", "tokenizer_config.json"], "the tokenizer of stage 1, and the chat format"],
          ].map(([names, what], i) => {
            const g = s.group();
            g.chips(624, 192 + i * 124, names, { size: 24, gap: 10 });
            g.text(624, 266 + i * 124, what, { size: 22, fill: "ink2" });
            return g;
          });
          const dot = s.circle(345, 334, 9, { fill: "s2" });
          return (t) => {
            show(source, during(t, 0.3, 0.8));
            show(step, during(t, 0.9, 1.3));
            show(directory, during(t, 1.2, 1.7));
            // a file at a time goes along the arrow
            let along = -1;
            files.forEach((g, i) => {
              const leaves = 1.8 + i * 0.9;
              if (t > leaves && t < leaves + 0.6) along = during(t, leaves, leaves + 0.6);
              show(g, during(t, leaves + 0.5, leaves + 0.9));
            });
            set(dot, { cx: mix(345, 585, Math.max(0, along)) });
            show(dot, along < 0 ? 0 : 1);
          };
        },
      },
      {
        seconds: 7.5,
        caption: "config.json gives the model's shape in the reference names.\nTo other tools, the model of stage 3 is a very small Llama.",
        draw(s) {
          s.text(430, 128, "minllm/model.py", { size: 22, anchor: "end", fill: "muted", mono: true });
          s.text(546, 128, "config.json", { size: 22, fill: "muted", mono: true });
          const pairs = [
            ["vocab_size", "vocab_size"],
            ["dim", "hidden_size"],
            ["n_layers", "num_hidden_layers"],
            ["n_heads", "num_attention_heads"],
            ["n_kv_heads", "num_key_value_heads"],
            ["mlp_dim", "intermediate_size"],
          ].map(([ours, theirs]) => [`${ours} = ${cfg[ours]}`, `"${theirs}": ${cfg[ours]}`]);
          pairs.push([`qkv_bias = ${cfg.qkv_bias ? "True" : "False"}`, `"architectures": ["${architecture}"]`]);
          const rows = pairs.map(([ours, theirs], i) => {
            const y = 146 + i * 50;
            const last = i === pairs.length - 1;
            const left = s.chip(0, y, ours, { size: 24 });
            place(left, 430 - left.w, y);
            return {
              left,
              arrow: s.arrow(446, y + left.h / 2, 530, y + left.h / 2),
              right: s.chip(546, y, theirs, last ? { size: 24, fill: "s2", stroke: null, ink: "surface" } : { size: 24 }),
              at: last ? 3.9 : 0.4 + i * 0.5,
            };
          });
          const other = s.text(640, 544, "With biases on query, key and value it is a Qwen2ForCausalLM.", { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            for (const r of rows) {
              show(r.left, during(t, r.at, r.at + 0.3));
              show(r.arrow, during(t, r.at + 0.15, r.at + 0.45));
              show(r.right, during(t, r.at + 0.3, r.at + 0.6));
            }
            show(other, during(t, 5.0, 5.6));
          };
        },
      },
      {
        seconds: 8,
        caption: "The check loads the exported directory with transformers\nand compares it against this repository's own code.",
        draw(s) {
          const sides = [card(s, 70, 112, 320, 100, "this repository", "the model, in its own code"), card(s, 890, 112, 320, 100, "transformers", "the exported directory")];
          const rails = [s.line(230, 212, 230, 530), s.line(1050, 212, 1050, 530)];
          const rungs = [
            ["token ids", "identical"],
            ["chat prompt", "identical"],
            ["logits", "the same, up to rounding"],
            ["greedy continuation", "identical"],
          ].map(([what, verdict], i) => {
            const y = 280 + i * 82;
            const g = s.group();
            const chip = g.chip(0, y - 22, what, { size: 26 });
            place(chip, 640 - chip.w / 2, y - 22);
            g.line(230, y, chip.x - 8, y);
            g.line(chip.x + chip.w + 8, y, 1050, y);
            return { g, chip, y, at: 1.4 + i * 1.15, dots: [g.circle(230, y, 9), g.circle(1050, y, 9)], verdict: s.text(chip.x + chip.w + 22, y - 12, verdict, { size: 22, weight: 600, fill: "s3" }) };
          });
          return (t) => {
            sides.forEach((box, i) => show(box, during(t, 0.3 + i * 0.3, 0.8 + i * 0.3)));
            rails.forEach((rail) => show(rail, during(t, 0.9, 1.3)));
            for (const r of rungs) {
              show(r.g, during(t, r.at, r.at + 0.3));
              // each side's result runs in from its rail and meets the other at what is compared
              const u = during(t, r.at + 0.2, r.at + 0.9);
              set(r.dots[0], { cx: mix(230, r.chip.x - 8, u) });
              set(r.dots[1], { cx: mix(1050, r.chip.x + r.chip.w + 8, u) });
              r.dots.forEach((dot) => show(dot, Math.min(during(t, r.at + 0.1, r.at + 0.3), 1 - during(t, r.at + 0.85, r.at + 1.0))));
              const same = t >= r.at + 0.9;
              r.chip.ground.setAttribute("class", same ? "f-s3 s-s3" : "f-tint s-axis");
              r.chip.label.setAttribute("class", `film-mono ${same ? "f-surface" : "f-ink"}`);
              show(r.verdict, during(t, r.at + 0.9, r.at + 1.2));
            }
          };
        },
      },
      {
        seconds: 6.5,
        caption: "MLX runs the directory as it is and writes the same story.\nvLLM and llama.cpp start from it too, and were not tried here.",
        draw(s) {
          const directory = card(s, 70, 284, 290, 112, `out/export/${MODEL}`, "one directory", { size: 24 });
          const tools = [
            ["transformers", ["loads it for the check", "and agrees with this code"], true],
            ["MLX", ["runs it as it is: the same", "story, token for token"], true],
            ["vLLM", ["serves it on Linux with an", "NVIDIA GPU: not tried here"], false],
            ["llama.cpp", ["needs one more conversion,", "to GGUF: not tried here"], false],
          ].map(([name, what, tried], i) => {
            const y = 116 + i * 112;
            const g = s.group();
            g.arrow(372, 340, 548, y + 44, { stroke: tried ? "s3" : "muted" });
            const box = s.group();
            card(box, 560, y, 240, 88, name, "", { stroke: tried ? "s3" : "axis", ink: tried ? "ink" : "ink2" });
            box.lines(824, y + 36, what, { size: 22, fill: tried ? "ink2" : "muted" });
            return { arrow: g, box, y: y + 44, tried, at: 0.9 + i * 0.85, dot: s.circle(372, 340, 9, { fill: "s3" }) };
          });
          return (t) => {
            show(directory, during(t, 0.3, 0.8));
            for (const tool of tools) {
              show(tool.arrow, during(t, tool.at, tool.at + 0.3));
              show(tool.box, during(t, tool.at + 0.5, tool.at + 0.9));
              // the directory goes where it was run
              const u = during(t, tool.at + 0.1, tool.at + 0.7);
              set(tool.dot, { cx: mix(372, 548, u), cy: mix(340, tool.y, u) });
              show(tool.dot, tool.tried ? Math.min(during(t, tool.at, tool.at + 0.15), 1 - during(t, tool.at + 0.6, tool.at + 0.75)) : 0);
            }
          };
        },
      },
      {
        seconds: 8.5,
        caption: "A model becomes useful to other programs behind an HTTP API.\nserve.py speaks OpenAI's chat completions and streams tokens.",
        draw(s) {
          const tokens = tok.encode(REPLY).map((id) => tok.text(id));
          const ends = [card(s, 70, 110, 270, 130, "a client", "any OpenAI library"), card(s, 940, 110, 270, 130, "scripts/serve.py", "the model, over HTTP", { size: 24 })];
          const request = s.group();
          request.arrow(352, 150, 928, 150);
          request.text(640, 136, "POST /v1/chat/completions", { size: 22, anchor: "middle", fill: "ink2", mono: true });
          const asked = s.circle(352, 150, 9, { fill: "s2" });
          const back = s.arrow(928, 210, 352, 210);
          const each = Math.min(0.25, 3.3 / tokens.length);
          const leaves = (i) => 2.3 + i * each;
          const done = leaves(tokens.length - 1) + 0.8;
          const chunks = tokens.map((token) => s.chip(0, 191, spelled(token), { size: 22, fill: "s2", stroke: null, ink: "surface", pad: 9 }));
          const line = (content) => `data: {"choices": [{"index": 0, "delta": {"content": ${JSON.stringify(content)}}}], …}`;
          const log = s.lines(203, 304, [line(tokens[0]), line(tokens[1]), "…", "data: [DONE]"], { size: 22, mono: true, leading: 1.5 });
          const label = s.text(640, 474, "the reply, as the client receives it", { size: 22, anchor: "middle", fill: "muted" });
          const reply = s.chips(640, 494, tokens.map(spelled), { center: true, size: 22, gap: 6, pad: 9 });
          return (t) => {
            ends.forEach((box, i) => show(box, during(t, 0.2 + i * 0.2, 0.7 + i * 0.2)));
            show(request, during(t, 0.8, 1.2));
            set(asked, { cx: mix(352, 928, during(t, 1.1, 1.9)) });
            show(asked, Math.min(during(t, 1.0, 1.15), 1 - during(t, 1.8, 1.95)));
            show(back, during(t, 1.9, 2.3));
            chunks.forEach((chip, i) => {
              const u = during(t, leaves(i), leaves(i) + 0.6);
              place(chip, mix(928 - chip.w, 352, u), 191);
              show(chip, Math.min(during(t, leaves(i) - 0.1, leaves(i) + 0.05), 1 - during(t, leaves(i) + 0.5, leaves(i) + 0.6)));
              show(reply[i], during(t, leaves(i) + 0.5, leaves(i) + 0.7));
            });
            show(label, during(t, leaves(0) + 0.3, leaves(0) + 0.7));
            show(log[0], during(t, leaves(0) + 0.5, leaves(0) + 0.8));
            show(log[1], during(t, leaves(1) + 0.5, leaves(1) + 0.8));
            show(log[2], during(t, leaves(2) + 0.5, leaves(2) + 0.8));
            show(log[3], during(t, done, done + 0.3));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "This server handles one request at a time. Stage 4 measured\nwhat that wastes: a second sequence in a batch is almost free.",
        draw(s) {
          const one = batch(1);
          const many = batch(64);
          const span = 520; // a panel's width, and the time the four requests take one after another
          const length = span / 4;
          const panels = [
            { x: 70, at: 330, title: "serve.py: one request at a time", starts: [0, 1, 2, 3].map((i) => i * length), length, run: one, unit: "tokens per second, 1 sequence", fill: "ink" },
            // in a batch of 4 a request takes this much longer than alone: stage 4's two timings
            { x: 690, at: 950, title: "a production server: one running batch", starts: [0, 40, 80, 210], length: (length * batch(4).seconds) / one.seconds, run: many, unit: `tokens per second, ${many.batch} in a batch`, fill: "s2" },
          ].map((p) => {
            s.text(p.x, 130, p.title, { size: 24, weight: 600 });
            const bars = p.starts.map((start, j) => {
              s.rect(p.x, 150 + j * 36, span, 26, { fill: "tint", r: 6 });
              return s.rect(p.x + start, 150 + j * 36, 0, 26, { fill: "s1", r: 6 });
            });
            s.arrow(p.x, 306, p.x + span, 306);
            s.text(p.x + span, 336, "time", { size: 22, anchor: "end", fill: "muted" });
            return { ...p, bars, number: s.text(p.at, 492, "", { size: 60, weight: 700, anchor: "middle", fill: p.fill }), what: s.text(p.at, 530, p.unit, { size: 22, anchor: "middle", fill: "ink2" }) };
          });
          // the two speeds are of this repository's code, not of a production server
          const from = s.text(640, 412, "what stage 4 measured with this repository's code, on the 14M model", { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            const now = span * during(t, 0.8, 3.8); // how far time has come, on both panels alike
            for (const p of panels) {
              p.bars.forEach((bar, j) => set(bar, { width: clamp(now - p.starts[j], 0, p.length) }));
              p.number.textContent = Math.round(mix(0, p.run.tokens_per_second, during(t, 4.2, 5.4))).toLocaleString("en");
              show(p.number, during(t, 4.0, 4.4));
              show(p.what, during(t, 4.0, 4.4));
            }
            show(from, during(t, 3.7, 4.1));
          };
        },
      },
      {
        seconds: 9,
        caption: "The libraries add scale, coverage and years of fixed bugs.\nThey do not add ideas: each row computes the same thing.",
        draw(s) {
          s.text(106, 128, "Stage", { size: 22, fill: "muted" });
          s.text(350, 128, "Here", { size: 22, fill: "muted" });
          s.text(766, 128, "In practice", { size: 22, fill: "muted" });
          s.line(70, 142, 1210, 142, { stroke: "axis" });
          const rows = MAP.map(([n, stage, here, practice], i) => {
            const y = 176 + i * 43;
            const left = s.group();
            left.text(92, y, n, { size: 22, weight: 700, anchor: "end", fill: "s2" });
            left.text(106, y, stage, { size: 22, weight: 600 });
            left.text(350, y, here, { size: 22, mono: true });
            const right = s.group();
            right.arrow(706, y - 7, 750, y - 7);
            right.text(766, y, practice, { size: 22, fill: "ink2" });
            return { left, right };
          });
          return (t) => {
            rows.forEach((r, i) => {
              show(r.left, during(t, 0.4 + i * 0.5, 0.8 + i * 0.5));
              show(r.right, during(t, 0.65 + i * 0.5, 1.05 + i * 0.5));
            });
          };
        },
      },
    ],
  };
};
