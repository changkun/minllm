// Side topic 4b as a film: speculative decoding. One text at a time, a pass
// of the model costs about the same over one new token or five. A cheap draft
// proposes k tokens, the model reads them all in one pass, keeps them up to
// the first it would not have picked, and adds one of its own. The round, the
// lookup and the parts of a pass are illustrations, as in the note's figure;
// the times of stage 4, the speeds, the costs of a round and the replies
// compared are read from runs/bench/.

import { svg } from "../labs/kit.js";
import { during, mix, place, set, show } from "./stage.js";

export const sources = {
  stage4: "/runs/bench/inference-mps.json",
  summary: "/runs/bench/speculative-summary.json",
  big: "/runs/bench/speculative-qwen2.5-7b-instruct-mps.json",
  bigBy15: "/runs/bench/speculative-qwen2.5-7b-instruct-draft-1.5b-mps.json",
};

// The note's round: the text, what the draft proposes, and what the model would pick at each place.
const PROMPT = "Natalia sold 48/2 =";
const DRAFT = ["24", "clips", "in", "April"];
const OWN = ["24", "clips", "in", "May"];
// The illustration of prompt lookup: a problem of GSM8K, and a reply that ends in words the problem has.
const PROBLEM = ["Natalia sold clips to 48 of her friends in April, and then she", "sold half as many clips in May. How many clips did Natalia sell", "altogether in April and May?"];
const REPLY = "Natalia sold 48/2 = 24 clips in";
const TAIL = "clips in"; // the reply's final tokens
const FOLLOWED = ["May", ".", "How", "many"]; // what followed them in the problem, k = 4
const IM_END = 151645; // Qwen's end of a turn: it ends a drafted reply in the raw files and is not part of the reply
const K = 4; // the draft length the note compares at
const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js

const times = (v) => `${v.toFixed(2)}×`;
const percent = (v) => `${Math.round(100 * v)}%`;
const greedy = (run, draft, k) => run.cases.find((c) => c.temperature === 0 && c.draft === draft && c.k === k);

// A chip as kept, as not kept, as the model's own, or as it was made.
function paint(chip, ground, ink, weight = 400) {
  chip.ground.setAttribute("class", ground);
  chip.label.setAttribute("class", `film-mono ${ink}`);
  chip.label.setAttribute("font-weight", weight);
}
// A chip centred on x.
function centred(s, x, y, string, options) {
  const chip = s.chip(0, y, string, options);
  place(chip, x - chip.w / 2, y);
  return chip;
}
// The outline of a rectangle, with nothing inside, so that what is under it shows.
const outline = (s, x, y, w, h, stroke) => svg("rect", { x, y, width: w, height: h, rx: 10, fill: "none", class: `s-${stroke}`, "stroke-width": 2 }, s.g);
const dashed = (line, dash = "5 6") => line.setAttribute("stroke-dasharray", dash);

export default ({ stage4, summary, big, bigBy15 }) => {
  // stage 4: one text of the most tokens on the GPU, each pass over one new token (cached) or over the whole text so far (not)
  const timed = stage4.length.filter((r) => r.batch === 1).reduce((a, b) => (b.new_tokens > a.new_tokens ? b : a));
  const perPass = { one: (1000 * timed.cached_seconds) / timed.new_tokens, whole: (1000 * timed.uncached_seconds) / timed.new_tokens };
  const small = `${Math.round(stage4.parameters / 1e6)}M`;

  const run = {
    big: summary["speculative-qwen2.5-7b-instruct-mps"],
    bigBy15: summary["speculative-qwen2.5-7b-instruct-draft-1.5b-mps"],
    mid: summary["speculative-qwen2.5-1.5b-instruct-mps"],
    story: summary["speculative-wide-mps"],
    story32: summary["speculative-wide-mps-float32"],
    storyCpu: summary["speculative-wide-cpu-bfloat16"],
    mid32: summary["speculative-qwen2.5-1.5b-instruct-mps-float32"],
  };
  const drafted = (r) => r.cases.filter((c) => c.temperature === 0 && c.draft === "model");
  const fastest = Math.max(...drafted(run.big).map((c) => c.speedup), ...drafted(run.bigBy15).map((c) => c.speedup));
  const storySlower = drafted(run.story).every((c) => c.speedup < 1);

  return {
    subtitle: "A cheap draft, checked by the model in one pass",
    scenes: [
      {
        seconds: 7,
        caption: "One text at a time, a pass goes to launching operations and\nreading weights. Over five new tokens it costs about the same.",
        draw(s) {
          const WIDE = 600; // the pixels of a pass over one new token
          const PARTS = [
            ["launching operations", "ink2", 0.52, 60],
            ["reading weights", "axis", 0.46, 350],
            ["arithmetic", "s3", 0.02, 580], // for each new token
          ];
          const rows = [
            [1, 140, ["24"], "a pass over one new token"],
            [5, 300, ["24", "clips", "in", "May", "."], "a pass over five new tokens"],
          ].map(([n, y, tokens, title], r) => {
            const g = s.group();
            g.text(60, y, title, { size: 24, weight: 600 });
            const before = g.chip(60, y + 16, PROMPT, { size: 24, pad: 10, ink: "muted" });
            g.chips(60 + before.w + 8, y + 16, tokens, { size: 24, pad: 10, fill: "raised", stroke: "ink" });
            let x = 60;
            const parts = PARTS.map(([, fill, share], i) => {
              const w = WIDE * share * (i === 2 ? n : 1);
              const part = { x, w, rect: s.rect(x, y + 76, 0, 34, { fill, r: 3 }) };
              x += w;
              return part;
            });
            return { g, parts, from: 0.3 + r * 1.9 };
          });
          const level = s.line(60 + WIDE, 206, 60 + WIDE, 420, { stroke: "ink", width: 2 });
          dashed(level);
          const legend = s.group();
          for (const [name, fill, , x] of PARTS) {
            legend.rect(x, 445, 18, 18, { fill, r: 3 });
            legend.text(x + 28, 461, name, { size: 22, fill: "ink2" });
          }
          legend.text(60, 504, "an illustration of where the time of a pass goes", { size: 22, fill: "muted" });

          const R = 850;
          const measured = s.group();
          measured.text(R, 140, "measured in stage 4:", { size: 22, fill: "muted" });
          measured.text(R, 168, `the ${small} model, one text, GPU`, { size: 22, fill: "muted" });
          const values = [
            [perPass.one, 250, "a pass over one new token"],
            [perPass.whole, 370, "a pass over the whole text so far"],
          ].map(([v, y, what]) => {
            const g = s.group();
            g.text(R, y + 32, what, { size: 22, fill: "ink2" });
            return { v, g, value: s.text(R, y, "", { size: 38, weight: 700 }) };
          });
          return (t) => {
            for (const row of rows) {
              show(row.g, during(t, row.from, row.from + 0.4));
              const spans = [[0.4, 0.9], [0.9, 1.3], [1.3, 1.5]];
              row.parts.forEach((p, i) => set(p.rect, { width: p.w * during(t, row.from + spans[i][0], row.from + spans[i][1]) }));
            }
            show(legend, during(t, 0.9, 1.3));
            show(level, during(t, 3.8, 4.1));
            show(measured, during(t, 4.1, 4.5));
            for (const [i, v] of values.entries()) {
              const from = 4.3 + i * 0.4;
              show(v.g, during(t, from, from + 0.3));
              show(v.value, during(t, from, from + 0.3));
              v.value.textContent = `${(v.v * during(t, from, from + 0.6)).toFixed(1)} ms`;
            }
          };
        },
      },
      {
        seconds: 8.5,
        caption: "A draft proposes k tokens. The model reads them in one pass,\nkeeps them up to the first it would not pick, and adds one.",
        draw(s) {
          const SIZE = 26;
          const slot = (i) => 495 + i * 150; // the centre of a place after the text
          const [D, P, M, V, R] = [146, 222, 300, 378, 410];
          const heads = s.group();
          heads.text(60, 128, "the text", { size: 22, fill: "muted" });
          heads.text(430, 128, `the draft proposes k = ${K} tokens, a small pass each`, { size: 22, fill: "s1" });
          heads.chip(60, D, PROMPT, { size: SIZE, pad: 10 });
          const draft = DRAFT.map((w) => centred(s, 0, D, w, { size: SIZE, pad: 10, stroke: "s1", ink: "s1" }));
          draft.forEach((c, i) => place(c, slot(i) - c.w / 2, D));
          const last = draft.at(-1);
          const strike = s.line(last.x - 6, D + last.h / 2, last.x + last.w + 6, D + last.h / 2, { stroke: "ink2", width: 3 });

          // one pass of the model over the text and the four draft tokens
          const swept = s.rect(60, P, 0, 46, { fill: "grid", r: 10 });
          const band = s.group();
          outline(band, 60, P, 1100, 46, "ink2");
          band.text(80, P + 31, "the model: one pass over the text and all four", { size: 22, weight: 600 });
          const down = Array.from({ length: K + 1 }, (_, i) => s.arrow(slot(i), P + 50, slot(i), M - 6, { stroke: "muted", width: 2 }));
          const asks = s.group();
          asks.lines(60, M + 18, ["the model's own token", "at each place"], { size: 22, fill: "ink2", leading: 1.2 });
          const picks = [...OWN, "…"].map((w, i) => centred(s, slot(i), M, w, { size: SIZE, pad: 10, fill: "raised", stroke: "ink2", ink: i === K ? "muted" : "ink" }));
          const verdicts = [...OWN.slice(0, -1).map(() => ["kept", "s3"]), ["its own", "ink"], ["if all are kept", "muted"]].map(([w, fill], i) => s.text(slot(i), V, w, { size: 22, anchor: "middle", fill, weight: i === K - 1 ? 700 : 400 }));

          // what the round leaves: the text, the kept tokens, and the model's own
          const result = s.chips(60, R, [PROMPT, ...OWN], { size: SIZE, pad: 10 });
          result.slice(1, -1).forEach((c) => paint(c, "f-s3 s-s3", "f-surface"));
          paint(result.at(-1), "f-raised s-ink", "f-ink", 700);
          const kept = OWN.length - 1;
          const told = s.text(60, 512, `${kept} kept, 1 added: ${kept + 1} tokens for 1 pass of the model`, { size: 26, weight: 700 });
          const bounds = s.text(60, 552, "A pass yields 1 token if the first is not kept, and k + 1 if all are.", { size: 22, fill: "ink2" });
          const CHECK = (i) => 4.0 + i * 0.45; // when the draft token at place i is checked
          return (t) => {
            show(heads, during(t, 0.2, 0.5));
            draft.forEach((c, i) => {
              show(c, during(t, 0.6 + i * 0.45, 0.85 + i * 0.45));
              const checked = t >= CHECK(i);
              if (i < K - 1) paint(c, checked ? "f-s3 s-s3" : "f-tint s-s1", checked ? "f-surface" : "f-s1");
              else paint(c, checked ? "f-tint s-axis" : "f-tint s-s1", checked ? "f-muted" : "f-s1");
            });
            show(strike, during(t, CHECK(K - 1), CHECK(K - 1) + 0.25));
            show(swept, during(t, 2.3, 2.4));
            set(swept, { width: 1100 * during(t, 2.4, 3.2) });
            show(band, during(t, 2.3, 2.6));
            show(asks, during(t, 3.2, 3.5));
            picks.forEach((c, i) => {
              const from = 3.2 + i * 0.1;
              show(down[i], during(t, from, from + 0.3));
              show(c, during(t, from, from + 0.3) * (i === K ? mix(1, 0.4, during(t, CHECK(K - 1), CHECK(K - 1) + 0.3)) : 1));
            });
            paint(picks[K - 1], t >= CHECK(K - 1) ? "f-raised s-ink" : "f-raised s-ink2", "f-ink", t >= CHECK(K - 1) ? 700 : 400);
            verdicts.forEach((v, i) => show(v, during(t, CHECK(Math.min(i, K - 1)) + (i === K ? 0.3 : 0), CHECK(Math.min(i, K - 1)) + (i === K ? 0.6 : 0.3))));
            result.forEach((c, i) => show(c, during(t, 5.9 + i * 0.06, 6.2 + i * 0.06)));
            show(told, during(t, 6.4, 6.7));
            show(bounds, during(t, 6.6, 6.9));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "A smaller model with the same tokenizer drafts. Prompt lookup\nproposes what followed an earlier copy of the final tokens.",
        draw(s) {
          // a smaller model: Qwen2.5's sizes share one vocabulary
          const top = s.group();
          top.text(60, 136, "A smaller model with the same tokenizer", { size: 26, weight: 700, fill: "s1" });
          const sizes = [
            ["0.5B", 60],
            ["1.5B", 250],
            ["7B", 440],
          ].map(([name, x]) => {
            const g = s.group();
            g.box(x, 160, 110, 56, name, "", { size: 26, stroke: "s1" });
            return g;
          });
          const links = s.group();
          links.arrow(174, 188, 246, 188, { stroke: "s1" });
          links.arrow(364, 188, 436, 188, { stroke: "s1" });
          links.path([[115, 220], [115, 244], [495, 244]], { stroke: "s1", width: 2.5 });
          links.arrow(495, 244, 495, 222, { stroke: "s1" });
          const vocabulary = s.text(60, 284, "Qwen2.5's sizes share one vocabulary.", { size: 22, fill: "ink2" });
          const round = s.group();
          round.text(680, 136, "a round: k small passes, then one", { size: 22, fill: "muted" });
          const passes = Array.from({ length: K }, (_, i) => s.rect(680 + i * 50, 166, 40, 44, { fill: "s1", r: 6 }));
          const model = s.group();
          model.box(890, 160, 230, 56, "the model", "", { size: 24, stroke: "ink2" });
          const cheaper = s.text(680, 284, "It pays only if the draft is much cheaper.", { size: 22, fill: "ink2" });

          // prompt lookup: the reply's final tokens, found earlier, and what followed them there
          const CW = 22 * MONO;
          const X = 210;
          const bottom = s.group();
          bottom.text(60, 340, "Prompt lookup: no model at all", { size: 26, weight: 700, fill: "s2" });
          bottom.text(1220, 340, "an illustration", { size: 22, anchor: "end", fill: "muted" });
          const span = (line, start, length, y, fill) => s.rect(X + start * CW - 3, y - 21, length * CW + 6, 30, { fill, stroke: "s2", r: 5, width: 2 });
          const at = PROBLEM[1].indexOf(TAIL);
          const after = PROBLEM[1].indexOf(FOLLOWED[0], at + TAIL.length);
          const upto = PROBLEM[1].indexOf(FOLLOWED.at(-1), after) + FOLLOWED.at(-1).length;
          const ending = span(REPLY, REPLY.lastIndexOf(TAIL), TAIL.length, 482, "tint");
          const earlier = span(PROBLEM[1], at, TAIL.length, 408, "tint");
          const next = span(PROBLEM[1], after, upto - after, 408, "s2");
          next.setAttribute("fill-opacity", 0.25);
          const texts = s.group();
          texts.text(60, 380, "the problem", { size: 22, fill: "muted" });
          PROBLEM.forEach((line, i) => texts.text(X, 380 + i * 28, line, { size: 22, mono: true, fill: "ink2" }));
          texts.text(60, 482, "the reply", { size: 22, fill: "muted" });
          texts.text(X, 482, REPLY, { size: 22, mono: true });
          const CX = 680;
          const copy = s.arrow(X + ((after + upto) / 2) * CW, 416, CX + 40, 452, { stroke: "s2" });
          const proposed = s.chips(CX, 456, FOLLOWED, { size: 22, pad: 8, gap: 6, stroke: "s2", ink: "s2" });
          const label = s.text(proposed.at(-1).x + proposed.at(-1).w + 16, 482, `proposed, k = ${K}`, { size: 22, fill: "s2" });
          const how = s.group();
          how.text(60, 538, "It finds the last earlier place of the text's final tokens, and proposes what followed.", { size: 22, fill: "ink2" });
          how.text(60, 574, "It costs nothing but the search.", { size: 22, fill: "ink2" });
          return (t) => {
            show(top, during(t, 0.2, 0.5));
            sizes.forEach((g, i) => show(g, during(t, 0.4 + i * 0.15, 0.7 + i * 0.15)));
            show(links, during(t, 1.0, 1.4));
            show(vocabulary, during(t, 1.2, 1.5));
            show(round, during(t, 1.4, 1.7));
            passes.forEach((p, i) => show(p, during(t, 1.6 + i * 0.25, 1.75 + i * 0.25)));
            show(model, during(t, 2.7, 3.0));
            show(cheaper, during(t, 3.0, 3.3));
            show(bottom, during(t, 3.2, 3.5));
            show(texts, during(t, 3.3, 3.6));
            show(ending, during(t, 3.9, 4.2));
            show(earlier, during(t, 4.3, 4.6));
            show(next, during(t, 4.7, 5.0));
            show(copy, during(t, 4.8, 5.1));
            proposed.forEach((c, i) => show(c, during(t, 5.0 + i * 0.12, 5.25 + i * 0.12)));
            show(label, during(t, 5.4, 5.7));
            show(how, during(t, 5.6, 5.9));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `It pays where the model's pass is slow: the 7B model is up to\n${fastest.toFixed(2)} times as fast. The story models are ${storySlower ? "slower at every k" : "mostly slower"}.`,
        draw(s) {
          const pairs = [
            [run.big, "Qwen2.5-7B", "drafted by the 0.5B model"],
            [run.bigBy15, "Qwen2.5-7B", "drafted by the 1.5B model"],
            [run.mid, "Qwen2.5-1.5B", "drafted by the 0.5B model"],
            [run.story, `the 42M story model`, "drafted by the 14M one"],
          ];
          const [BAR, MS] = [430, 3.2]; // where the bars of a pass begin, and the pixels of a millisecond
          const [X0, PER] = [740, 150]; // where the speed is 0, and the pixels of the model alone's speed
          const x = (v) => X0 + PER * v;
          const frame = s.group();
          frame.text(BAR, 136, "a pass of the model alone", { size: 22, fill: "muted" });
          frame.text(X0, 136, "speed with the draft, a dot for each k", { size: 22, fill: "muted" });
          for (const v of [0, 1, 2]) {
            const line = frame.line(x(v), 176, x(v), 530, { stroke: v === 1 ? "ink2" : "grid", width: v === 1 ? 2.5 : 1.5 });
            if (v === 1) dashed(line);
            frame.text(x(v), 558, `${v}×`, { size: 22, anchor: "middle", fill: "muted" });
          }
          frame.text(x(1), 168, "the model alone", { size: 22, anchor: "middle", fill: "ink2" });
          const rows = pairs.map(([r, name, by], i) => {
            const y = 220 + i * 92;
            const ms = 1000 / r.alone;
            const g = s.group();
            g.text(60, y - 4, name, { size: 24, weight: 700 });
            g.text(60, y + 26, by, { size: 22, fill: "s1" });
            const cases = drafted(r);
            const best = cases.reduce((a, b) => (b.speedup > a.speedup ? b : a));
            return {
              g,
              ms,
              y,
              bar: s.rect(BAR, y - 18, 0, 30, { fill: "ink2", r: 4 }),
              value: s.text(BAR, y + 5, `${ms.toFixed(ms < 10 ? 1 : 0)} ms`, { size: 22, weight: 600 }),
              dots: cases.map((c) => ({ c, dot: s.circle(x(1), y, 8, { fill: "s1", stroke: "surface" }) })),
              best: s.text(1220, y + 9, times(best.speedup), { size: 28, weight: 700, anchor: "end", fill: best.speedup > 1 ? "ink" : "muted" }),
            };
          });
          return (t) => {
            show(frame, during(t, 0.2, 0.5));
            rows.forEach((row, i) => {
              show(row.g, during(t, 0.3 + i * 0.12, 0.6 + i * 0.12));
              const grown = during(t, 0.7 + i * 0.12, 1.5 + i * 0.12);
              set(row.bar, { width: MS * row.ms * grown });
              set(row.value, { x: BAR + MS * row.ms * grown + 12 });
              show(row.value, during(t, 1.2 + i * 0.12, 1.5 + i * 0.12));
              const from = 2.0 + i * 0.5;
              row.dots.forEach(({ c, dot }, j) => {
                const u = during(t, from + j * 0.06, from + 0.6 + j * 0.06);
                show(dot, during(t, from + j * 0.06, from + 0.15 + j * 0.06));
                set(dot, { cx: mix(x(1), x(c.speedup), u) });
              });
              show(row.best, during(t, from + 0.7, from + 1.0));
            });
          };
        },
      },
      {
        seconds: 7.5,
        caption: "A round has to cost less than it yields. Prompt lookup costs\nalmost nothing: it pays on GSM8K and not on the stories.",
        draw(s) {
          const groups = [
            ["Qwen2.5-7B", "on GSM8K", [[run.big, "model", "the 0.5B model"], [run.bigBy15, "model", "the 1.5B model"], [run.big, "lookup", "prompt lookup"]]],
            ["Qwen2.5-1.5B", "on GSM8K", [[run.mid, "model", "the 0.5B model"], [run.mid, "lookup", "prompt lookup"]]],
            ["the 42M story model", "on story openings", [[run.story, "model", "the 14M model"], [run.story, "lookup", "prompt lookup"]]],
          ];
          const [X0, PER] = [520, 48]; // where a cost of 0 is, and the pixels of a token of the model alone
          const x = (v) => X0 + PER * v;
          const frame = s.group();
          [
            ["s1", "yields, a smaller model's draft", 60],
            ["s2", "yields, prompt lookup", 450],
            ["ink2", "costs", 750],
          ].forEach(([fill, name, at]) => {
            frame.circle(at + 9, 122, 9, { fill });
            frame.text(at + 28, 130, name, { size: 22, fill: "ink2" });
          });
          frame.text(X0, 174, `in tokens of the model alone, k = ${K}, greedy`, { size: 22, fill: "muted" });
          frame.text(1110, 174, "kept", { size: 22, anchor: "end", fill: "muted" });
          frame.text(1220, 174, "speed", { size: 22, anchor: "end", fill: "muted" });
          for (let v = 0; v <= 10; v += 2) {
            frame.line(x(v), 190, x(v), 534, { stroke: "grid", width: 1.5 });
            frame.text(x(v), 562, String(v), { size: 22, anchor: "middle", fill: "muted" });
          }
          const rows = [];
          let y = 214;
          for (const [target, on, drafts] of groups) {
            frame.text(60, y + 8, target, { size: 24, weight: 700 });
            frame.text(60, y + 52, on, { size: 22, fill: "muted" });
            for (const [r, kind, name] of drafts) {
              const c = greedy(r, kind, K);
              const fill = kind === "model" ? "s1" : "s2";
              const g = s.group();
              g.text(330, y + 8, name, { size: 22, fill });
              rows.push({
                c,
                g,
                link: s.line(x(0), y, x(0), y, { stroke: "axis", width: 4 }),
                cost: s.circle(x(0), y, 9, { fill: "ink2", stroke: "surface" }),
                yields: s.circle(x(0), y, 9, { fill, stroke: "surface" }),
                kept: s.text(1110, y + 8, percent(c.kept), { size: 22, anchor: "end", fill: "ink2" }),
                speed: s.text(1220, y + 8, times(c.speedup), { size: 24, weight: 700, anchor: "end", fill: c.speedup > 1 ? "ink" : "muted" }),
              });
              y += 44;
            }
            y += 16;
          }
          return (t) => {
            show(frame, during(t, 0.2, 0.5));
            rows.forEach((row, i) => {
              const from = 0.6 + i * 0.3;
              show(row.g, during(t, from, from + 0.3));
              const u = during(t, from + 0.2, from + 1.2);
              const [a, b] = [x(row.c.round_cost * u), x(row.c.tokens_per_pass * u)];
              set(row.cost, { cx: a });
              set(row.yields, { cx: b });
              set(row.link, { x1: Math.min(a, b), x2: Math.max(a, b) });
              for (const dot of [row.cost, row.yields, row.link]) show(dot, during(t, from + 0.1, from + 0.3));
              show(row.kept, during(t, from + 1.0, from + 1.3));
              show(row.speed, during(t, from + 1.1, from + 1.4));
            });
          };
        },
      },
      {
        seconds: 7,
        caption: "In 16-bit floats on the GPU, replies with a draft part from\nthe model's own, mostly at the same token whatever the draft.",
        draw(s) {
          // the 7B model's replies, greedy, k = 4: the model alone, the two smaller drafts and prompt lookup
          const own = (raw) => raw.cases.find((c) => c.temperature === 0 && c.draft === "none").outputs.map((o) => o.length);
          const lengths = (raw, draft) => greedy(raw, draft, K).outputs.map((o) => o.length - (o.at(-1) === IM_END ? 1 : 0));
          const lanes = [
            { fill: "ink2", parted: own(big).map(() => null), lengths: own(big) },
            { fill: "s1", parted: greedy(run.big, "model", K).parted_at, lengths: lengths(big, "model") },
            { fill: "s1", parted: greedy(run.bigBy15, "model", K).parted_at, lengths: lengths(bigBy15, "model") },
            { fill: "s2", parted: greedy(run.big, "lookup", K).parted_at, lengths: lengths(big, "lookup") },
          ];
          const LIMIT = run.big.max_new_tokens;
          const [X0, X1] = [200, 980];
          const x = (v) => X0 + ((X1 - X0) * v) / LIMIT;
          const [TOP, LANE, PITCH, GROUP] = [196, 10, 13, 74];
          const frame = s.group();
          frame.text(60, 128, `the 7B model's ${run.big.n} replies, greedy, k = ${K}, in 16-bit floats on the GPU`, { size: 22, fill: "muted" });
          [
            ["ink2", "the model alone", 60],
            ["s1", "the 0.5B, then the 1.5B draft", 290],
            ["s2", "prompt lookup", 676],
            ["axis", "after it parts", 900],
          ].forEach(([stroke, name, at]) => {
            frame.line(at, 158, at + 32, 158, { stroke, width: 8 });
            frame.text(at + 44, 166, name, { size: 22, fill: "ink2" });
          });
          for (const v of [0, 64, 128, 192, 256].filter((v) => v <= LIMIT)) {
            frame.line(x(v), TOP - 8, x(v), TOP + 4 * GROUP + 4, { stroke: "grid", width: 1.5 });
            frame.text(x(v), 566, String(v), { size: 22, anchor: "middle", fill: "muted" });
          }
          frame.text(X1 + 34, 566, "tokens", { size: 22, fill: "muted" });
          const replies = own(big).map((_, r) => {
            const y = TOP + r * GROUP;
            frame.text(60, y + 32, `reply ${r + 1}`, { size: 22, fill: "ink2" });
            const where = [...new Set(lanes.slice(1).map((l) => l.parted[r]).filter((p) => p !== null))].sort((a, b) => a - b);
            const marks = where.map((p) => {
              const line = s.line(x(p), y - 5, x(p), y + 3 * PITCH + LANE + 5, { stroke: "ink", width: 2 });
              dashed(line, "3 4");
              return { p, line };
            });
            const told = s.text(1020, y + 32, where.length ? `parts at ${where.join(" and ")}` : "the same", { size: 22, weight: 600 });
            const bars = lanes.map((lane, j) => {
              const at = y + j * PITCH;
              const end = lane.lengths[r];
              const parted = lane.parted[r];
              return { end, parted, same: s.rect(X0, at, 0, LANE, { fill: lane.fill, r: 2 }), own: s.rect(X0, at, 0, LANE, { fill: "axis", r: 2 }) };
            });
            return { marks, told, bars, last: where.length ? where.at(-1) : LIMIT };
          });
          for (const r of replies) for (const m of r.marks) s.g.append(m.line); // the marks lie over the lanes
          const SWEEP = [0.8, 3.8];
          const reach = (p) => mix(SWEEP[0], SWEEP[1], p / LIMIT); // the second the sweep gets to token p
          return (t) => {
            show(frame, during(t, 0.2, 0.5));
            const now = LIMIT * during(t, SWEEP[0], SWEEP[1]);
            for (const r of replies) {
              for (const b of r.bars) {
                const cut = b.parted === null ? b.end : b.parted;
                set(b.same, { width: x(Math.min(now, cut)) - X0 });
                const from = x(Math.min(now, cut));
                set(b.own, { x: from, width: Math.max(0, x(Math.min(now, b.end)) - from) });
                show(b.same, now > 0 ? 1 : 0);
                show(b.own, b.parted !== null && now > b.parted ? 1 : 0);
              }
              for (const m of r.marks) show(m.line, during(t, reach(m.p), reach(m.p) + 0.2));
              show(r.told, during(t, reach(r.last), reach(r.last) + 0.3));
            }
          };
        },
      },
      {
        seconds: 6.5,
        caption: "The check is to blame: a near tie, tipped in the last bits.\nIn 32-bit floats, every reply is the model's own.",
        draw(s) {
          const groups = [
            ["the 42M story model", "on story openings", [
              ["the 14M draft", "model", [run.story, run.story32, run.storyCpu]],
              ["prompt lookup", "lookup", [run.story, run.story32, run.storyCpu]],
            ]],
            ["Qwen2.5-1.5B", "on GSM8K", [
              ["the 0.5B draft", "model", [run.mid, run.mid32, null]],
              ["prompt lookup", "lookup", [run.mid, run.mid32, null]],
            ]],
          ];
          const COLUMNS = [520, 760, 1000];
          const [SQUARE, STEP] = [18, 22];
          const frame = s.group();
          frame.rect(60, 113, 18, 18, { fill: "s3", r: 3 });
          frame.text(88, 130, "a reply that is the model's own, token for token", { size: 22, fill: "ink2" });
          frame.rect(640, 113, 18, 18, { fill: "axis", r: 3 });
          frame.text(668, 130, "a reply that parts from it", { size: 22, fill: "ink2" });
          frame.text(60, 178, `greedy, k = ${K}`, { size: 22, fill: "muted" });
          ["GPU, 16-bit floats", "GPU, 32-bit floats", "CPU, 16-bit floats"].forEach((name, i) => frame.text(COLUMNS[i] + (10 * STEP - 4) / 2, 178, name, { size: 22, weight: 600, anchor: "middle", fill: i === 1 ? "s3" : "ink" }));
          const cells = [];
          let y = 222;
          for (const [target, on, drafts] of groups) {
            frame.text(60, y + 2, target, { size: 24, weight: 700 });
            frame.text(60, y + 32, on, { size: 22, fill: "muted" });
            for (const [name, kind, runs] of drafts) {
              frame.text(330, y + 2, name, { size: 22, fill: kind === "model" ? "s1" : "s2" });
              runs.forEach((r, col) => {
                const x = COLUMNS[col];
                if (!r) {
                  cells.push({ col, g: frame.text(x, y + 2, "not measured", { size: 22, fill: "muted" }), squares: [] });
                  return;
                }
                const c = greedy(r, kind, K);
                const g = s.group();
                g.text(x, y + 34, `${c.same} of ${r.n}`, { size: 22, weight: 600, fill: c.same === r.n ? "s3" : "ink2" });
                const squares = c.parted_at.map((p, i) => s.rect(x + i * STEP, y - 16, SQUARE, SQUARE, { fill: p === null ? "s3" : "axis", r: 3 }));
                cells.push({ col, g, squares });
              });
              y += 76;
            }
            y += 20;
          }
          const fp32 = [run.story32, run.mid32].flatMap((r) => r.cases.filter((c) => c.temperature === 0).map((c) => [c.same, r.n]));
          const sum = (i) => fp32.reduce((total, pair) => total + pair[i], 0);
          const all = s.text(60, 578, `In 32-bit floats on the GPU: ${sum(0)} of ${sum(1)} replies the model's own, over every draft and k measured.`, { size: 22, fill: "ink2" });
          const FROM = [0.6, 2.0, 3.2]; // when each column fills
          return (t) => {
            show(frame, during(t, 0.2, 0.5));
            for (const cell of cells) {
              const from = FROM[cell.col];
              cell.squares.forEach((q, i) => show(q, during(t, from + i * 0.07, from + 0.2 + i * 0.07)));
              show(cell.g, during(t, from + 0.8, from + 1.1));
            }
            show(all, during(t, 4.3, 4.6));
          };
        },
      },
    ],
  };
};
