// Stage 3 as a film: every position of a text predicts the token after it,
// the loss says how wrong the predictions are, and a step moves the weights.
// The schedule, the loss curve and what the model wrote are those of the
// reference run in runs/base/.

import { Tokenizer } from "../labs/bpe.js";
import { during, mix, place, set, show, spelled, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  meta: "/models/base-q4/meta.json",
  attention: "/runs/base/attention.json",
  next: "/runs/base/next-token.json",
  log: "/runs/base/log.jsonl",
  samples: "/runs/base/samples.json",
};

const percent = (v) => `${Math.round(100 * v)}%`;
const count = (n) => n.toLocaleString("en");
const mark = (chip, fill, ink = "surface") => {
  chip.ground.setAttribute("class", fill ? `f-${fill} s-${fill}` : "f-tint s-axis");
  chip.label.setAttribute("class", `film-mono f-${fill ? ink : "ink"}`);
};

// Text cut at its spaces into lines of at most so many characters each, with a mark where it is cut off.
function wrap(text, widths) {
  const lines = [];
  let rest = text;
  widths.forEach((width, i) => {
    const room = i + 1 < widths.length ? width : width - 2;
    const cut = rest.length <= room ? rest.length : rest.lastIndexOf(" ", room);
    lines.push(rest.slice(0, cut));
    rest = rest.slice(cut).trimStart();
  });
  if (rest) lines[lines.length - 1] += " …";
  return lines;
}

export default ({ vocabulary, meta, attention, next, log, samples }) => {
  const tok = new Tokenizer(vocabulary);
  const spell = (id) => spelled(tok.text(id));

  // a prompt the trained model was asked about, and the token that follows it in the sentence
  const asked = next.prompts[0];
  const ids = tok.encode(asked.prompt);
  const came = tok.encode(attention.text)[ids.length];
  const peak = Math.max(...asked.logits);
  const weights = asked.logits.map((l) => Math.exp(l - peak));
  const sum = weights.reduce((a, b) => a + b, 0);
  const likely = weights.map((w, id) => ({ id, p: w / sum })).sort((a, b) => b.p - a.p).slice(0, 5);
  const given = weights[came] / sum;

  // the reference run
  const trained = log.filter((e) => "train_loss" in e);
  const evaluated = log.filter((e) => "val_loss" in e);
  const steps = evaluated.at(-1).step;
  const minutes = (evaluated.at(-1).elapsed / 60).toFixed(1);
  const top = Math.max(...trained.map((e) => e.lr));
  const warm = trained.find((e) => e.lr === top).step;
  const low = trained.at(-1).lr;
  const tail = trained.filter((e) => e.step > steps - 200);
  const fitted = tail.reduce((a, e) => a + e.train_loss, 0) / tail.length;
  const at = (step) => evaluated.find((e) => e.step === step);
  const early = evaluated[1];

  return {
    subtitle: "Predict the next token, measure, step",
    scenes: [
      {
        seconds: 8,
        caption: "The task is to read some text and predict the next token.\nNobody labels anything. The text is its own supervision.",
        draw(s) {
          const n = ids.length - 1;
          // a column for every position: the token that is read, and below it the one to predict
          const read = ids.slice(0, n).map((id) => s.chip(0, 0, spell(id), { size: 26 }));
          const target = ids.slice(1).map((id) => s.chip(0, 0, spell(id), { size: 26 }));
          const columns = [];
          let x = 200;
          for (let i = 0; i < n; i++) {
            const w = Math.max(read[i].w, target[i].w);
            columns.push(x + w / 2);
            x += w + 14;
          }
          const home = (chips, i) => columns[i] - chips[i].w / 2;
          read.forEach((c, i) => place(c, home(read, i), 180));
          s.text(70, 210, "input", { size: 24, fill: "ink2" });
          const name = s.text(70, 400, "target", { size: 24, fill: "ink2" });
          const arrows = columns.map((c) => s.arrow(c, 234, c, 360));
          const said = s.lines(640, 490, [`${n} positions, ${n} predictions from one pass through the model.`, `A window of ${meta.config.max_seq_len} tokens gives ${meta.config.max_seq_len} of them.`], { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            read.forEach((c, i) => show(c, during(t, 0.3 + i * 0.08, 0.6 + i * 0.08)));
            show(name, during(t, 1.6, 2));
            // the targets are the same text, one token further on
            const moved = during(t, 1.8, 3.2);
            target.forEach((c, i) => {
              const from = i + 1 < n ? home(read, i + 1) : home(target, i);
              place(c, mix(from, home(target, i), moved), i + 1 < n ? mix(180, 370, moved) : 370);
              show(c, i + 1 < n ? during(t, 1.7, 1.9) : during(t, 3, 3.4));
            });
            arrows.forEach((a, i) => show(a, during(t, 3.3 + i * 0.06, 3.6 + i * 0.06)));
            // one position after the other: what has been read so far, and what is to come
            const k = t < 4.2 ? -1 : Math.min(n - 1, Math.floor(n * during(t, 4.2, 6.2)));
            read.forEach((c, i) => c.ground.setAttribute("class", i <= k ? "f-tint s-s1" : "f-tint s-axis"));
            target.forEach((c, i) => mark(c, i === k ? "s2" : null));
            said.forEach((line) => show(line, during(t, 5.8, 6.3)));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "The loss is the negative log of the probability\ngiven to the token that actually came next.",
        draw(s) {
          const row = s.chips(640, 104, [...ids.map(spell), spell(came)], { center: true, size: 24, gap: 6, pad: 8 });
          const actual = row.at(-1);
          mark(actual, "s3");
          const whose = s.lines(90, 252, ["the trained model's", "probabilities for", "the next token"], { size: 24, fill: "ink2" });
          const wide = 520 / likely[0].p;
          const bars = likely.map((l, i) => {
            const y = 196 + i * 46;
            const chip = s.chip(0, y, spell(l.id), { size: 22, pad: 8 });
            place(chip, 480 - chip.w, y);
            return { l, chip, w: wide * l.p, bar: s.rect(496, y + 4, 0, 30, { fill: "s1", r: 5 }), value: s.text(0, y + 27, percent(l.p), { size: 24, fill: "ink2" }) };
          });
          const formula = s.text(640, 484, `loss = −ln ${given.toFixed(3)} = ${(-Math.log(given)).toFixed(2)}`, { size: 34, anchor: "middle", weight: 600, mono: true });
          const fresh = s.text(640, 542, `Untrained, every token gets 1/${count(next.vocab_size)}: the loss is ln ${count(next.vocab_size)} = ${Math.log(next.vocab_size).toFixed(2)}.`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            row.forEach((c, i) => show(c, c === actual ? during(t, 3.3, 3.7) : during(t, 0.2 + i * 0.05, 0.5 + i * 0.05)));
            whose.forEach((line) => show(line, during(t, 1, 1.4)));
            bars.forEach((b, i) => {
              const u = during(t, 1.3 + i * 0.25, 2.2 + i * 0.25);
              show(b.chip, during(t, 1.2 + i * 0.25, 1.5 + i * 0.25));
              set(b.bar, { width: b.w * u });
              set(b.value, { x: 508 + b.w * u });
              show(b.value, during(t, 1.9 + i * 0.25, 2.3 + i * 0.25));
              // the one that came is the one that counts
              const right = b.l.id === came && t >= 3.5;
              b.bar.setAttribute("class", right ? "f-s3" : "f-s1");
              mark(b.chip, right ? "s3" : null);
              b.value.setAttribute("font-weight", right ? 700 : 400);
            });
            show(formula, during(t, 4.3, 4.8));
            show(fresh, during(t, 5.2, 5.7));
          };
        },
      },
      {
        seconds: 8.5,
        caption: "One step measures the loss on a batch, finds how each weight\nshould move to lower it, and moves it a little.",
        draw(s) {
          const Y = 205;
          const parts = [
            { title: "batch", code: "get_batch()", note: ["windows cut from", "the text at random"] },
            { title: "loss", code: "model(x, y)", note: ["how wrong the", "predictions are"] },
            { title: "gradient", code: "loss.backward()", note: ["which way each", "weight should move"] },
            { title: "update", code: "optimizer.step()", note: ["AdamW moves", "each weight a little"] },
          ];
          parts.forEach((p, i) => {
            p.x = 100 + i * 280;
            p.g = s.group();
            p.ground = p.g.rect(p.x, Y - 36, 220, 72, { fill: "raised", stroke: "axis", r: 12 });
            p.g.text(p.x + 110, Y + 10, p.title, { size: 28, weight: 600, anchor: "middle" });
            p.g.text(p.x + 110, Y + 76, p.code, { size: 22, anchor: "middle", fill: "s1", mono: true });
            p.g.lines(p.x + 110, Y + 116, p.note, { size: 22, anchor: "middle", fill: "muted" });
            if (i) p.g.arrow(p.x - 54, Y, p.x - 6, Y);
          });
          // the way back: the next step starts with a new batch
          const corners = [[1166, Y], [1200, Y], [1200, 420], [70, 420], [70, Y], [94, Y]];
          const back = s.group();
          corners.slice(0, -2).forEach(([x1, y1], i) => back.line(x1, y1, corners[i + 1][0], corners[i + 1][1], { stroke: "muted", width: 2.5 }));
          back.arrow(70, Y, 96, Y);
          const way = [[94, Y], ...corners];
          const lengths = way.slice(1).map(([x, y], i) => Math.hypot(x - way[i][0], y - way[i][1]));
          const around = lengths.reduce((a, b) => a + b, 0);
          const dot = s.circle(0, 0, 10, { fill: "s2" });
          const laps = [[1.5, 3.7], [3.7, 4.7], [4.7, 5.3]];
          const step = s.text(640, 484, "", { size: 34, weight: 600, anchor: "middle" });
          const took = s.text(640, 536, `The reference run takes ${count(steps)} steps, in ${minutes} minutes.`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            parts.forEach((p, i) => show(p.g, during(t, 0.2 + i * 0.25, 0.6 + i * 0.25)));
            show(back, during(t, 1, 1.4));
            let k = -1;
            laps.forEach(([from], i) => (k = t >= from ? i : k));
            const lap = k < 0 ? 0 : during(t, laps[k][0], laps[k][1]);
            const on = k >= 0 && t < laps.at(-1)[1];
            // where the dot is, that far along the way around
            let d = lap * around;
            let x = way[0][0];
            let y = way[0][1];
            for (let i = 0; i < lengths.length; i++) {
              const u = Math.min(1, d / lengths[i]);
              x = mix(way[i][0], way[i + 1][0], u);
              y = mix(way[i][1], way[i + 1][1], u);
              d -= lengths[i];
              if (d <= 0) break;
            }
            set(dot, { cx: x, cy: y });
            show(dot, on ? 1 : 0);
            parts.forEach((p) => p.ground.setAttribute("class", on && y === Y && x >= p.x && x <= p.x + 220 ? "f-raised s-s2" : "f-raised s-axis"));
            // three steps one by one, then all the others
            const n = t < laps.at(-1)[1] ? k + 1 : Math.round(mix(laps.length, steps, during(t, 5.3, 6.4)));
            step.textContent = n > 0 ? `step ${count(n)}` : "";
            show(took, during(t, 6.2, 6.6));
          };
        },
      },
      {
        seconds: 7,
        caption: `The learning rate ramps up over the first ${warm} steps,\nthen follows a cosine down to a tenth of its peak.`,
        draw(s) {
          const X = 190;
          const TOP = 176;
          const WIDE = 900;
          const TALL = 290;
          const { px, py } = s.axes(X, TOP, WIDE, TALL, { x: { domain: [0, steps] }, y: { domain: [0, top] } });
          for (const v of [0, top / 2, top]) {
            if (v) s.line(X, py(v), X + WIDE, py(v), { stroke: "grid", width: 1.5 });
            s.text(X - 14, py(v) + 8, String(v), { size: 22, anchor: "end", fill: "muted" });
          }
          for (const v of [0, steps / 3, (2 * steps) / 3, steps]) s.text(px(v), TOP + TALL + 32, count(v), { size: 22, anchor: "middle", fill: "muted" });
          s.text(X, TOP - 52, "learning rate", { size: 22, fill: "muted" });
          s.text(X + WIDE / 2, TOP + TALL + 68, "step", { size: 22, anchor: "middle", fill: "muted" });
          const curve = s.path(trained.map((e) => [px(e.step), py(e.lr)]), { stroke: "s2", width: 4 });
          const high = s.group();
          high.circle(px(warm), py(top), 7, { fill: "s2" });
          high.text(px(warm) + 18, py(top) - 14, `${top} at step ${warm}`, { size: 24, weight: 600 });
          const end = s.group();
          end.circle(px(steps), py(low), 7, { fill: "s2" });
          end.text(px(steps), py(top * 0.44), `${low.toFixed(4)}, 1/${Math.round(top / low)} of the peak`, { size: 24, weight: 600, anchor: "end" });
          const large = s.text(px(warm) + 40, py(top * 0.5) - 14, "large steps early, to make progress", { size: 22, fill: "ink2" });
          const small = s.text(px(steps), py(top * 0.56), "small steps late, to settle", { size: 22, anchor: "end", fill: "ink2" });
          return (t) => {
            trace(curve, during(t, 0.6, 3.6));
            show(curve, t > 0.6 ? 1 : 0);
            show(high, during(t, 1.1, 1.5));
            show(large, during(t, 1.8, 2.3));
            show(small, during(t, 3.1, 3.6));
            show(end, during(t, 3.6, 4));
          };
        },
      },
      {
        seconds: 8,
        caption: `${count(steps)} steps took ${minutes} minutes. The validation loss fell from\n${evaluated[0].val_loss.toFixed(2)} to ${early.val_loss.toFixed(2)} in the first ${early.step} steps, and then to ${evaluated.at(-1).val_loss.toFixed(2)}.`,
        draw(s) {
          const X = 150;
          const TOP = 140;
          const WIDE = 800;
          const TALL = 360;
          const { px, py } = s.axes(X, TOP, WIDE, TALL, { x: { domain: [0, steps] }, y: { domain: [1, 10], log: true } });
          for (const v of [1, 2, 5, 10]) {
            if (v > 1) s.line(X, py(v), X + WIDE, py(v), { stroke: "grid", width: 1.5 });
            s.text(X - 14, py(v) + 8, String(v), { size: 22, anchor: "end", fill: "muted" });
          }
          for (const v of [0, steps / 3, (2 * steps) / 3, steps]) s.text(px(v), TOP + TALL + 32, count(v), { size: 22, anchor: "middle", fill: "muted" });
          s.text(X, TOP - 18, "loss", { size: 22, fill: "muted" });
          s.text(X + WIDE / 2, TOP + TALL + 64, "step", { size: 22, anchor: "middle", fill: "muted" });
          s.line(430, 184, 470, 184, { stroke: "s1", width: 4 });
          s.text(484, 192, "training loss, every 50 steps", { size: 22, fill: "ink2" });
          s.circle(450, 222, 7, { fill: "s2" });
          s.text(484, 230, "validation loss, on stories it has not seen", { size: 22, fill: "ink2" });
          const points = trained.map((e) => [px(e.step), py(e.train_loss)]);
          const curve = s.path(points, { stroke: "s1", width: 3.5 });
          // how far along the curve every logged step is, to show an evaluation when the line gets there
          let run = 0;
          const reached = points.map(([x, y], i) => (run += i ? Math.hypot(x - points[i - 1][0], y - points[i - 1][1]) : 0) / curve.total);
          const dots = evaluated.map((e) => {
            const i = trained.findIndex((p) => p.step >= e.step);
            return { when: reached[i], dot: s.circle(px(e.step), py(e.val_loss), 6.5, { fill: "s2" }) };
          });
          const before = s.text(px(0) + 20, py(evaluated[0].val_loss) + 2, `${evaluated[0].val_loss.toFixed(2)} before training`, { size: 24, weight: 600 });
          const soon = s.text(px(early.step) + 16, py(early.val_loss) - 14, `${early.val_loss.toFixed(2)} at step ${early.step}`, { size: 24, weight: 600 });
          const ends = s.group();
          ends.text(990, 300, "validation", { size: 22, fill: "muted" });
          ends.text(990, 342, evaluated.at(-1).val_loss.toFixed(2), { size: 36, weight: 700, fill: "s2" });
          ends.text(990, 400, "training", { size: 22, fill: "muted" });
          ends.text(990, 442, fitted.toFixed(2), { size: 36, weight: 700, fill: "s1" });
          ends.text(990, 496, `after ${minutes} minutes`, { size: 22, fill: "muted" });
          return (t) => {
            const part = during(t, 0.8, 5.2);
            trace(curve, part);
            show(curve, t > 0.8 ? 1 : 0);
            dots.forEach((d, i) => show(d.dot, i ? (part >= d.when ? 1 : 0) : during(t, 0.3, 0.6)));
            show(before, during(t, 0.4, 0.8));
            show(soon, part >= dots[1].when ? during(t, 2, 2.4) : 0);
            show(ends, during(t, 5.4, 6));
          };
        },
      },
      {
        seconds: 9.5,
        caption: `What the model wrote after “${samples.prompt}” during the run.\nGrammar arrives first, and sense takes longer.`,
        draw(s) {
          const SIZE = 22;
          const X = 250;
          const letter = SIZE * 0.602; // the width of a character in the monospaced font
          const room = Math.floor((1216 - X) / letter);
          const shown = [0, early.step, steps / 3, steps].map((step, i) => {
            const y = 168 + i * 110;
            // the characters a page cannot show are left out
            const wrote = [...samples.samples.find((e) => e.step === step).text.slice(samples.prompt.length)].filter((c) => c >= " " && c !== "�").join("").replace(/ +/g, " ");
            const lines = wrap(wrote, [room - samples.prompt.length, room]);
            const g = s.group();
            g.text(70, y, `step ${count(step)}`, { size: 26, weight: 700 });
            g.text(70, y + 34, `loss ${at(step).val_loss.toFixed(2)}`, { size: 22, fill: "muted" });
            g.text(X, y, samples.prompt, { size: SIZE, fill: "muted", mono: true });
            return { g, lines, from: 0.5 + i * 1.9, nodes: [g.text(X + samples.prompt.length * letter, y, "", { size: SIZE, mono: true }), g.text(X, y + 34, "", { size: SIZE, mono: true })] };
          });
          return (t) => {
            for (const row of shown) {
              show(row.g, during(t, row.from - 0.3, row.from));
              // written out, as it was generated: one piece after the other
              let letters = Math.round((row.lines[0].length + row.lines[1].length) * during(t, row.from, row.from + 1.5) ** 0.8);
              row.nodes.forEach((node, i) => {
                node.textContent = row.lines[i].slice(0, letters);
                letters = Math.max(0, letters - row.lines[i].length);
              });
            }
          };
        },
      },
    ],
  };
};
