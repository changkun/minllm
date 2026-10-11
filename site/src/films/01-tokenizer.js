// Stage 1 as a film: text becomes bytes, the pairs of bytes that occur most
// become tokens, and a sentence becomes the numbers a model reads. The merges
// are the ones stage 1 learned, replayed by bpe.js.

import { Tokenizer } from "../labs/bpe.js";
import { during, mix, show, spelled } from "./stage.js";

export const sources = { vocabulary: "/tokenizer.json", bench: "/runs/bench/tokenizer.json" };

const SENTENCE = "Once upon a time, there was a little kitten.";
const WORD = " kitten";

export default ({ vocabulary, bench }) => {
  const tok = new Tokenizer(vocabulary);
  const ids = tok.encode(SENTENCE);
  const bytes = new TextEncoder().encode(SENTENCE).length;
  const spell = (id) => spelled(tok.text(id));
  const size = bench.sizes.find((s) => s.vocab_size === tok.size);

  return {
    subtitle: "From text to the numbers a model reads",
    scenes: [
      {
        seconds: 7,
        caption: "A model computes with numbers. Text starts as bytes:\none or more per character, each a number below 256.",
        draw(s) {
          const letters = [..."Once upon a time"];
          const top = s.chips(640, 240, letters.map(spelled), { center: true, size: 40, gap: 6 });
          const codes = top.map((c, i) => s.text(c.x + c.w / 2, 390, String(letters[i].charCodeAt(0)), { size: 26, anchor: "middle", fill: "s1", mono: true }));
          const ticks = top.map((c) => s.line(c.x + c.w / 2, 320, c.x + c.w / 2, 356, { stroke: "axis" }));
          const count = s.text(640, 470, `${letters.length} characters, ${letters.length} bytes`, { size: 26, anchor: "middle", fill: "muted" });
          return (t) => {
            show(count, during(t, 4, 4.6));
            top.forEach((c, i) => show(c, during(t, 0.3 + i * 0.08, 0.6 + i * 0.08)));
            codes.forEach((c, i) => show(c, during(t, 2.2 + i * 0.08, 2.6 + i * 0.08)));
            ticks.forEach((c, i) => show(c, during(t, 2.0 + i * 0.08, 2.4 + i * 0.08)));
          };
        },
      },
      {
        seconds: 9,
        caption: `Training counts which two neighbours occur together most often\nand makes the pair a new token. Then again, ${tok.merges.length.toLocaleString("en")} times.`,
        draw(s) {
          // the first merges that were learned, as rows: left + right -> new
          const rows = tok.merges.slice(0, 7).map(([a, b], i) => {
            const g = s.group();
            const y = 124 + i * 58;
            g.text(250, y + 34, `merge ${i + 1}`, { size: 24, fill: "muted" });
            const left = g.chip(400, y, spell(a), { size: 28 });
            g.text(400 + left.w + 16, y + 34, "+", { size: 28, fill: "muted" });
            const right = g.chip(400 + left.w + 48, y, spell(b), { size: 28 });
            g.arrow(400 + left.w + 48 + right.w + 20, y + 24, 400 + left.w + 48 + right.w + 90, y + 24);
            g.chip(400 + left.w + 48 + right.w + 106, y, spell(256 + i), { size: 28, fill: "s2", stroke: null, ink: "surface" });
            g.text(960, y + 34, `token ${256 + i}`, { size: 24, fill: "muted", mono: true });
            return g;
          });
          const count = s.text(1030, 560, "", { size: 26, anchor: "end", fill: "ink2" });
          return (t) => {
            rows.forEach((g, i) => show(g, during(t, 0.4 + i * 0.55, 0.9 + i * 0.55)));
            const merges = Math.round(mix(7, tok.merges.length, during(t, 4.8, 7.6)));
            count.textContent = t < 4.6 ? "" : `${merges.toLocaleString("en")} merges: a vocabulary of ${(256 + merges + (merges === tok.merges.length ? tok.special.size : 0)).toLocaleString("en")} tokens`;
          };
        },
      },
      {
        seconds: 10,
        caption: "New text is encoded by replaying the merges\nin the order in which they were learned.",
        draw(s) {
          // every state of the word, laid out centered; a state fades into the next
          const states = tok.steps(WORD);
          const rows = states.map((state) => {
            const g = s.group();
            g.row = g.chips(640, 280, state.map(spell), { center: true, size: 46, gap: 10 });
            return g;
          });
          const label = s.text(640, 450, "", { size: 26, anchor: "middle", fill: "muted" });
          const each = 8.4 / states.length;
          return (t) => {
            const at = Math.min(states.length - 1, Math.floor(Math.max(0, t - 0.6) / each));
            rows.forEach((g, i) => show(g, i === at ? 1 - during(t, 0.6 + (i + 1) * each - 0.25, 0.6 + (i + 1) * each) * (i < states.length - 1 ? 1 : 0) : 0));
            // the pair that merges next, marked
            const next = states[at + 1];
            const pair = next ? states[at].findIndex((id, i) => next[i] !== id) : -1;
            rows[at].row.forEach((c, i) => c.ground.setAttribute("class", i === pair || i === pair + 1 ? "f-s2 s-s2" : "f-tint s-axis"));
            rows[at].row.forEach((c, i) => c.label.setAttribute("class", `film-mono ${i === pair || i === pair + 1 ? "f-surface" : "f-ink"}`));
            label.textContent = next ? `${states[at].length} tokens: merge “${spell(states[at][pair])}” and “${spell(states[at][pair + 1])}”` : `1 token, number ${states[at][0]}, from ${states[0].length} bytes`;
          };
        },
      },
      {
        seconds: 8,
        caption: `The sentence is ${bytes} bytes, and ${ids.length} tokens.\nThese numbers are all the model will ever see of it.`,
        draw(s) {
          const row = s.chips(640, 230, ids.map(spell), { center: true, size: 28, gap: 6, pad: 9 });
          row.forEach((c, i) => c.ground.setAttribute("class", i % 2 ? "f-tint s-axis" : "f-raised s-axis"));
          const numbers = row.map((c, i) => s.text(c.x + c.w / 2, 340, String(ids[i]), { size: 22, anchor: "middle", fill: "s1", mono: true }));
          const sequence = s.text(640, 460, `[${ids.join(", ")}]`, { size: 30, anchor: "middle", mono: true });
          return (t) => {
            row.forEach((c, i) => show(c, during(t, 0.3 + i * 0.12, 0.6 + i * 0.12)));
            numbers.forEach((n, i) => show(n, during(t, 2.4 + i * 0.12, 2.8 + i * 0.12)));
            show(sequence, during(t, 4.6, 5.4));
          };
        },
      },
      {
        seconds: 7,
        caption: `On stories it has not seen, a token of this vocabulary\nstands for ${size.bytes_per_token.toFixed(2)} bytes on average.`,
        draw(s) {
          // bytes per token against the size of the vocabulary, from the benchmark
          const shown = bench.sizes.filter((b) => b.vocab_size <= 16384);
          const peak = Math.max(...shown.map((b) => b.bytes_per_token));
          const bars = shown.map((b, i) => {
            const x = 300 + i * 120;
            const h = (300 * b.bytes_per_token) / peak;
            const mine = b.vocab_size === tok.size;
            return {
              h,
              bar: s.rect(x, 480 - h, 72, h, { fill: mine ? "s2" : "s1", r: 6 }),
              value: s.text(x + 36, 470 - h, b.bytes_per_token.toFixed(2), { size: 22, anchor: "middle", fill: mine ? "ink" : "ink2", weight: mine ? 700 : 400 }),
              name: s.text(x + 36, 514, b.vocab_size.toLocaleString("en"), { size: 22, anchor: "middle", fill: mine ? "ink" : "muted", weight: mine ? 700 : 400 }),
            };
          });
          s.line(280, 480, 1010, 480, { stroke: "axis" });
          s.text(640, 556, "bytes per token, by the size of the vocabulary", { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            bars.forEach((b, i) => {
              const u = during(t, 0.4 + i * 0.25, 1.4 + i * 0.25);
              b.bar.setAttribute("height", (b.h * u).toFixed(1));
              b.bar.setAttribute("y", (480 - b.h * u).toFixed(1));
              show(b.value, during(t, 1.2 + i * 0.25, 1.6 + i * 0.25));
            });
          };
        },
      },
    ],
  };
};
