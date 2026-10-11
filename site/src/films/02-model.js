// Stage 2 as a film: a token id becomes a vector, the blocks add to it what
// attention and the MLP compute, and the last vector becomes a score for
// every token. The attention weights and the prediction are those of the
// model that stage 3 trains.

import { Tokenizer } from "../labs/bpe.js";
import { during, mix, place, set, show, spelled, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  meta: "/models/base-q4/meta.json",
  attention: "/runs/base/attention.json",
  next: "/runs/base/next-token.json",
};

const percent = (v) => `${Math.round(100 * v)}%`;
const millions = (n) => `${(n / 1e6).toFixed(1)}M`;
const count = (n) => n.toLocaleString("en");
const name = (token) => token.trim() || token;

// Numbers that look random and are the same on every call.
function noise(seed) {
  let a = Math.imul(seed + 1, 2654435761);
  return () => {
    a = (a + 1831565813) | 0;
    let v = Math.imul(a ^ (a >>> 15), 1 | a);
    v = (v + Math.imul(v ^ (v >>> 7), 61 | v)) ^ v;
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

export default ({ vocabulary, meta, attention, next }) => {
  const tok = new Tokenizer(vocabulary);
  const spell = (id) => spelled(tok.text(id));
  const { dim, vocab_size: vocab, n_layers: layers, n_heads: heads, n_kv_heads: kv, mlp_dim: hidden } = meta.config;

  // the prompt whose next token the trained model was asked for
  const asked = next.prompts[0];
  const ids = tok.encode(asked.prompt);
  const peak = Math.max(...asked.logits);
  const weights = asked.logits.map((l) => Math.exp(l - peak));
  const sum = weights.reduce((a, b) => a + b, 0);
  const likely = weights.map((w, id) => ({ id, p: w / sum })).sort((a, b) => b.p - a.p).slice(0, 6);

  // where the parameters are, from the sizes of the model
  const parts = [
    { name: "embedding", n: vocab * dim, fill: "muted" },
    { name: "attention", n: layers * (2 * dim * dim + 2 * (dim / heads) * kv * dim), fill: "s1" },
    { name: "MLP", n: layers * 3 * dim * hidden, fill: "s2" },
  ];
  const total = meta.parameters;

  // The sentence in two rows, and above every token how much the reading token takes from it.
  const T = attention.tokens;
  const second = T.indexOf(" She");
  const most = (h, reader) => h.weights[reader].map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]);
  function reading(s, h, readers, arrive) {
    const TALL = 110;
    const upper = s.chips(640, 262, T.slice(1, second).map(spelled), { center: true, size: 24, gap: 6, pad: 8 });
    const lower = s.chips(640, 470, T.slice(second).map(spelled), { center: true, size: 24, gap: 6, pad: 8 });
    const chip = [null, ...upper, ...lower]; // by the place of the token in the sentence
    const floor = (i) => (i < second ? 252 : 460);
    const bars = chip.map((c, i) => c && s.rect(c.x + 4, floor(i), c.w - 8, 0, { fill: "s1", r: 4 }));
    const labels = chip.map((c, i) => c && s.text(c.x + c.w / 2, floor(i), "", { size: 22, anchor: "middle", fill: "ink2" }));
    s.text(70, 126, `layer ${h.layer}, head ${h.head}`, { size: 26, weight: 600 });
    const readout = s.text(640, 566, "", { size: 26, anchor: "middle", fill: "ink2" });
    const said = (reader) => {
      const [[w1, i1], [w2, i2]] = most(h, reader);
      return `“${name(T[reader])}” reads ${percent(w1)} from “${name(T[i1])}”${w2 >= 0.2 ? ` and ${percent(w2)} from “${name(T[i2])}”` : ""}`;
    };
    const sees = (reader, i) => (reader < 0 || i <= reader ? 1 : 0.3); // a token cannot read from later ones
    return (t) => {
      let k = -1;
      readers.forEach((r, i) => (k = t >= r.at ? i : k));
      const now = k >= 0 ? readers[k].token : -1;
      const before = k > 0 ? readers[k - 1].token : -1;
      const u = k >= 0 ? during(t, readers[k].at, readers[k].at + 1) : 0;
      const marked = u > 0.5 ? now : before; // the token that is marked as reading
      const clear = marked === now ? 2 * u - 1 : 1 - 2 * u; // what is written fades out and in again
      for (let i = 1; i < T.length; i++) {
        const w = mix(before >= 0 ? h.weights[before][i] : 0, now >= 0 ? h.weights[now][i] : 0, u);
        set(bars[i], { y: floor(i) - TALL * w, height: TALL * w });
        const written = marked >= 0 ? h.weights[marked][i] : 0;
        labels[i].textContent = written >= 0.15 ? percent(written) : "";
        set(labels[i], { y: floor(i) - TALL * w - 8 });
        show(labels[i], clear);
        show(chip[i], during(t, arrive + 0.03 * i, arrive + 0.3 + 0.03 * i) * mix(sees(before, i), sees(now, i), u));
        chip[i].ground.setAttribute("class", i === marked ? "f-s2 s-s2" : "f-tint s-axis");
        chip[i].label.setAttribute("class", `film-mono ${i === marked ? "f-surface" : "f-ink"}`);
      }
      readout.textContent = marked >= 0 ? said(marked) : "";
      show(readout, clear);
    };
  }
  const [pronoun, previous] = attention.heads;
  const she = T.indexOf(" She");
  const her = T.indexOf(" her");
  const [[toName, named]] = most(pronoun, she);

  return {
    subtitle: "From token ids to a score for every next token",
    scenes: [
      {
        seconds: 7,
        caption: `The model first turns each token id into a vector:\none row of ${dim} learned numbers from a table of ${count(vocab)} rows.`,
        draw(s) {
          const CELLS = 6;
          const row = s.chips(640, 118, ids.map(spell), { center: true, size: 30, gap: 10 });
          const numbers = row.map((c, i) => s.text(c.x + c.w / 2, 206, String(ids[i]), { size: 24, anchor: "middle", fill: "s1", mono: true }));
          const ticks = row.map((c) => s.line(c.x + c.w / 2, 220, c.x + c.w / 2, 244, { stroke: "axis" }));
          // a vector as a column of cells: the same id gives the same column
          const columns = row.map((c, i) => {
            const value = noise(ids[i]);
            return Array.from({ length: CELLS }, (_, k) => {
              const g = s.group();
              g.rect(c.x + c.w / 2 - 24, 254 + k * 38, 48, 32, { fill: "raised", stroke: "axis", r: 5, width: 1.5 });
              g.rect(c.x + c.w / 2 - 24, 254 + k * 38, 48, 32, { fill: "s1", r: 5 }).setAttribute("opacity", (0.12 + 0.8 * value()).toFixed(2));
              return g;
            });
          });
          const said = s.text(640, 540, `${ids.length} tokens in, ${ids.length} vectors of ${dim} numbers out`, { size: 26, anchor: "middle", fill: "ink2" });
          return (t) => {
            row.forEach((c, i) => show(c, during(t, 0.3 + i * 0.08, 0.6 + i * 0.08)));
            numbers.forEach((n, i) => show(n, during(t, 1.5 + i * 0.08, 1.9 + i * 0.08)));
            ticks.forEach((l, i) => show(l, during(t, 2.7 + i * 0.12, 3.0 + i * 0.12)));
            columns.forEach((cells, i) => cells.forEach((g, k) => show(g, during(t, 2.8 + i * 0.12 + k * 0.1, 3.1 + i * 0.12 + k * 0.1))));
            show(said, during(t, 4.7, 5.2));
          };
        },
      },
      {
        seconds: 8.5,
        caption: `The vectors run through ${layers} blocks as the residual stream.\nA block reads a copy, computes, and adds its result back.`,
        draw(s) {
          const Y = 440; // where the stream runs
          const blocks = [
            { x: 320, title: "attention", note: "tokens exchange information", from: 1.5 },
            { x: 720, title: "MLP", note: "each token on its own", from: 3.3 },
          ];
          const all = s.group();
          place(all, 0, 24);
          all.rect(250, 155, 800, 350, { fill: "surface", stroke: "axis", r: 14 });
          const block = all.text(250, 138, "", { size: 26, weight: 600 });
          const marks = Array.from({ length: layers }, (_, i) => all.rect(1050 - (layers - i) * 30 + 8, 116, 22, 22, { fill: "tint", stroke: "axis", r: 4, width: 1.5 }));
          all.text(70, Y - 22, "residual stream", { size: 24, fill: "ink2" });
          all.line(70, Y, 1160, Y, { stroke: "s1", width: 5 });
          all.arrow(1150, Y, 1200, Y, { stroke: "s1", width: 5 });
          for (const p of blocks) {
            p.g = all.group();
            p.g.text(p.x + 120, 208, p.note, { size: 22, anchor: "middle", fill: "muted" });
            p.g.arrow(p.x + 20, Y - 6, p.x + 20, 311);
            p.g.text(p.x + 8, 380, "read", { size: 22, anchor: "end", fill: "muted" });
            p.g.arrow(p.x + 220, 307, p.x + 220, Y - 18);
            p.g.text(p.x + 234, 380, "add", { size: 22, fill: "muted" });
            p.frame = p.g.rect(p.x, 225, 240, 80, { fill: "raised", stroke: "axis", r: 12 });
            p.g.text(p.x + 120, 275, p.title, { size: 28, weight: 600, anchor: "middle" });
            p.g.circle(p.x + 220, Y, 15, { fill: "surface", stroke: "s1" });
            p.g.text(p.x + 220, Y + 9, "+", { size: 26, anchor: "middle", fill: "s1" });
            p.copy = all.circle(0, 0, 8, { fill: "s2" });
          }
          const vector = all.circle(0, Y, 11, { fill: "s2" });
          return (t) => {
            blocks.forEach((p, i) => {
              show(p.g, during(t, 0.3 + i * 0.3, 0.8 + i * 0.3));
              // the copy goes up into the part, and what the part computed comes down
              const up = during(t, p.from, p.from + 0.5);
              const down = during(t, p.from + 1, p.from + 1.5);
              set(p.copy, { cx: down > 0 ? p.x + 220 : p.x + 20, cy: down > 0 ? mix(307, Y, down) : mix(Y, 307, up) });
              show(p.copy, up > 0 && down < 1 && !(up === 1 && down === 0) ? 1 : 0);
              p.frame.setAttribute("class", up === 1 && down === 0 ? "f-raised s-s2" : "f-raised s-axis");
            });
            // the vector itself stays on the stream, and is larger after every addition
            const x = mix(mix(mix(mix(mix(100, 340, during(t, 1, 1.5)), 540, during(t, 2, 3)), 740, during(t, 3, 3.3)), 940, during(t, 3.8, 4.8)), 1130, during(t, 4.8, 5.3));
            set(vector, { cx: x, r: 11 + 3 * during(t, 2.9, 3.1) + 3 * during(t, 4.7, 4.9) });
            show(vector, during(t, 0.8, 1));
            const k = 1 + Math.round((layers - 1) * during(t, 5.5, 6.9));
            block.textContent = `block ${k} of ${layers}`;
            marks.forEach((m, i) => m.setAttribute("class", i + 1 < k ? "f-s1 s-s1" : i + 1 === k ? "f-s2 s-s2" : "f-tint s-axis"));
          };
        },
      },
      {
        seconds: 8.5,
        caption: `In attention a token reads from itself and earlier tokens.\nIn this head “${name(T[she])}” puts ${percent(toName)} of its attention on “${name(T[named])}”.`,
        draw: (s) => reading(s, pronoun, [{ at: 1.6, token: she }, { at: 4.8, token: her }], 0.2),
      },
      {
        seconds: 7.5,
        caption: "Another head tends to look at the token just before.\nNobody told these heads what to do.",
        draw: (s) => reading(s, previous, [{ at: 0.6, token: she }, { at: 2.6, token: her }, { at: 4.6, token: T.indexOf(" ball") }], -1),
      },
      {
        seconds: 8,
        caption: `The MLP then processes each token's vector on its own.\nThe MLPs hold ${percent(parts[2].n / total)} of the model's ${millions(total)} parameters.`,
        draw(s) {
          const some = s.chips(640, 108, T.slice(6, 11).map(spelled), { center: true, size: 26, gap: 44 });
          const left = some[0].x - 20;
          const right = some.at(-1).x + some.at(-1).w + 20;
          const mlp = s.group();
          mlp.rect(left, 196, right - left, 62, { fill: "raised", stroke: "axis", r: 12 });
          mlp.text((left + right) / 2, 237, "MLP", { size: 28, weight: 600, anchor: "middle" });
          for (const c of some) {
            mlp.arrow(c.x + c.w / 2, 156, c.x + c.w / 2, 194);
            mlp.arrow(c.x + c.w / 2, 260, c.x + c.w / 2, 298);
          }
          const dots = some.map((c) => s.circle(c.x + c.w / 2, 0, 8, { fill: "s2" }));
          const alone = s.text(640, 336, "every vector goes through alone, and the result is added to it", { size: 22, anchor: "middle", fill: "muted" });
          // one bar for all the parameters, cut where they are
          const X = 140;
          const WIDE = 1000;
          const title = s.text(X, 404, `${count(total)} parameters`, { size: 24, fill: "ink2" });
          let at = X;
          const cuts = parts.map((p) => {
            const w = (WIDE * p.n) / total;
            const cut = { w, x: at, bar: s.rect(at, 420, 0, 50, { fill: p.fill, r: 4 }), g: s.group() };
            cut.g.text(at + w / 2, 502, p.name, { size: 24, anchor: "middle", weight: p.fill === "s2" ? 700 : 400 });
            cut.g.text(at + w / 2, 534, `${millions(p.n)}, ${percent(p.n / total)}`, { size: 22, anchor: "middle", fill: "ink2" });
            at += w;
            return cut;
          });
          return (t) => {
            some.forEach((c, i) => show(c, during(t, 0.2 + i * 0.08, 0.5 + i * 0.08)));
            show(mlp, during(t, 0.8, 1.3));
            dots.forEach((d) => {
              set(d, { cy: mix(156, 298, during(t, 1.6, 2.8)) });
              show(d, Math.min(during(t, 1.5, 1.6), 1 - during(t, 2.8, 3)));
            });
            show(alone, during(t, 2.6, 3.1));
            show(title, during(t, 3.3, 3.7));
            cuts.forEach((c, i) => {
              const u = during(t, 3.7 + i * 0.7, 4.5 + i * 0.7);
              set(c.bar, { width: Math.max(0, c.w * u - 3) });
              show(c.bar, u > 0 ? 1 : 0);
              show(c.g, during(t, 4.2 + i * 0.7, 4.6 + i * 0.7));
            });
          };
        },
      },
      {
        seconds: 8,
        caption: `The last vector becomes a score for each of the ${count(vocab)} tokens.\nAfter “${name(tok.text(ids.at(-2)))}${tok.text(ids.at(-1))}” the trained model gives “${name(tok.text(likely[0].id))}” ${percent(likely[0].p)}.`,
        draw(s) {
          const row = s.chips(640, 104, ids.map(spell), { center: true, size: 24, gap: 6, pad: 8 });
          const last = row.at(-1);
          const down = s.arrow(last.x + last.w / 2, 150, last.x + last.w / 2, 184, { stroke: "s2" });
          // every score, in the order of the vocabulary
          const low = Math.min(...asked.logits);
          const scores = s.path(asked.logits.map((l, id) => [100 + (1080 * id) / (vocab - 1), 262 - (70 * (l - low)) / (peak - low)]), { stroke: "s1", width: 1.5 });
          const base = s.line(100, 264, 1180, 264, { stroke: "axis" });
          const all = s.text(100, 300, `${count(asked.logits.length)} scores, in the order of the vocabulary`, { size: 22, fill: "muted" });
          const best = s.circle(100 + (1080 * likely[0].id) / (vocab - 1), 192, 7, { fill: "s2" });
          const told = s.lines(100, 360, ["as probabilities,", "the most likely", "next tokens"], { size: 24, fill: "ink2" });
          const wide = 560 / likely[0].p;
          const bars = likely.map((l, i) => {
            const y = 320 + i * 42;
            const chip = s.chip(0, y, spell(l.id), { size: 22, pad: 8 });
            place(chip, 480 - chip.w, y);
            return {
              w: wide * l.p,
              chip,
              bar: s.rect(496, y + 4, 0, 30, { fill: i ? "s1" : "s2", r: 5 }),
              value: s.text(0, y + 27, percent(l.p), { size: 24, weight: i ? 400 : 700, fill: i ? "ink2" : "ink" }),
            };
          });
          return (t) => {
            row.forEach((c, i) => show(c, during(t, 0.2 + i * 0.05, 0.5 + i * 0.05)));
            const marked = t >= 1;
            last.ground.setAttribute("class", marked ? "f-s2 s-s2" : "f-tint s-axis");
            last.label.setAttribute("class", `film-mono ${marked ? "f-surface" : "f-ink"}`);
            show(down, during(t, 1, 1.4));
            show(base, during(t, 1.2, 1.5));
            trace(scores, during(t, 1.4, 3.2));
            show(scores, t > 1.4 ? 1 : 0);
            show(all, during(t, 2.8, 3.2));
            show(best, during(t, 3.2, 3.5));
            told.forEach((line) => show(line, during(t, 3.4, 3.8)));
            bars.forEach((b, i) => {
              const u = during(t, 3.6 + i * 0.25, 4.5 + i * 0.25);
              show(b.chip, during(t, 3.5 + i * 0.25, 3.8 + i * 0.25));
              set(b.bar, { width: b.w * u });
              set(b.value, { x: 508 + b.w * u });
              show(b.value, during(t, 4.2 + i * 0.25, 4.6 + i * 0.25));
            });
          };
        },
      },
    ],
  };
};
