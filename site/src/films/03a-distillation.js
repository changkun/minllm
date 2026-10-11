// Side topic 3a as a film: the text names one token, a trained model answers
// with a probability for every token, and a small model can be trained on that
// answer. The predictions are the 42M teacher's own, and the losses are those
// of the logged runs.

import { Tokenizer } from "../labs/bpe.js";
import { during, mix, set, show, spelled, trace } from "./stage.js";

export const sources = {
  vocabulary: "/tokenizer.json",
  targets: "/runs/distill/targets.json",
  next: "/runs/distill/next-token.json",
  text: "/runs/base/log.jsonl",
  mixed: "/runs/distill/distilled.jsonl",
  alone: "/runs/distill/teacher-only.jsonl",
  timed: "/runs/distill/distilled-20min.jsonl",
  teacher: "/runs/moe/wide.jsonl", // the teacher is the wide model of side topic 2a
};

const percent = (p) => `${(100 * p).toFixed(p > 0 && p < 0.0995 ? 1 : 0)}%`;
const whole = (n) => n.toLocaleString("en");

// Rows of bars from x to the right: a token, a bar as long as its share, and the share.
function rows(s, x, y, items, scale, gap = 48) {
  return items.map(({ name, p, fill = "s1" }, i) => {
    const g = s.group();
    const top = y + i * gap;
    g.text(x - 14, top + 24, name, { size: 24, anchor: "end", mono: true });
    return { g, x, p, w: p * scale, bar: g.rect(x, top, 0, 30, { fill, r: 4 }), value: g.text(x, top + 23, "", { size: 22, fill: "ink2" }) };
  });
}

// A row at a part u of its length, from 0 to 1.
function grow(row, u) {
  set(row.bar, { width: row.w * u });
  set(row.value, { x: row.x + row.w * u + 10 });
  row.value.textContent = percent(row.p * u);
}

// A box with a title and a line below it.
function card(s, x, y, w, title, sub) {
  s.rect(x, y, w, 88, { fill: "raised", stroke: "axis", r: 12 });
  s.text(x + w / 2, y + 38, title, { size: 26, weight: 600, anchor: "middle" });
  s.text(x + w / 2, y + 70, sub, { size: 22, anchor: "middle", fill: "ink2" });
}

// The frame of a chart, as Stage.axes draws it, with labels of the size the films use.
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
  s.text(x + w / 2, y + h + 62, across.title, { size: 22, anchor: "middle", fill: "muted" });
  s.text(x, y - 16, up.title, { size: 22, fill: "muted" });
  return { px, py };
}

// A chip as the one that is talked about.
function mark(chip) {
  chip.ground.setAttribute("class", "f-s2 s-s2");
  chip.label.setAttribute("class", "film-mono f-surface");
}

export default ({ vocabulary, targets, next, text, mixed, alone, timed, teacher }) => {
  const tok = new Tokenizer(vocabulary);
  const pieces = (string) => tok.encode(string).map((id) => spelled(tok.text(id)));
  const [first, , third] = targets.positions;
  const held = targets.held_out;

  const evaluated = (log) => log.filter((row) => row.val_loss !== undefined);
  const last = (log) => evaluated(log).at(-1);
  const end = { text: last(text), mixed: last(mixed), alone: last(alone), timed: last(timed), teacher: last(teacher) };
  const loss = (row) => row.val_loss.toFixed(3);
  // how far a student came from the model without a teacher towards the teacher
  const way = (row) => (end.text.val_loss - row.val_loss) / (end.text.val_loss - end.teacher.val_loss);
  const minutes = Math.round(end.text.elapsed / 60);
  const slower = end.mixed.elapsed / end.mixed.step / (end.text.elapsed / end.text.step);

  return {
    subtitle: "A small model learns from a larger one",
    scenes: [
      {
        seconds: 7.5,
        caption: `The text names one token out of ${whole(next.vocab_size)} as what came next.\nA trained model answers with a probability for every token.`,
        draw(s) {
          const row = s.chips(640, 116, [...pieces(first.context), spelled(first.label)], { center: true, size: 28, gap: 6, pad: 10 });
          mark(row.at(-1));
          const said = s.text(150, 232, "What the text says", { size: 26, weight: 600 });
          const thought = s.text(700, 232, "What the teacher says", { size: 26, weight: 600 });
          const label = rows(s, 264, 258, first.top.map((t) => ({ name: t.token.trim(), p: t.token === first.label ? 1 : 0, fill: "s2" })), 290);
          const answer = rows(s, 814, 258, first.top.map((t) => ({ name: t.token.trim(), p: t.probability, fill: t.token === first.label ? "s2" : "s1" })), 290);
          return (t) => {
            row.forEach((c, i) => show(c, i < row.length - 1 ? during(t, 0.3 + i * 0.06, 0.6 + i * 0.06) : during(t, 1, 1.4)));
            show(said, during(t, 1.5, 1.9));
            label.forEach((r) => (show(r.g, during(t, 1.5, 1.9)), grow(r, during(t, 1.7, 2.5))));
            show(thought, during(t, 2.9, 3.3));
            answer.forEach((r, i) => (show(r.g, during(t, 2.9 + i * 0.12, 3.3 + i * 0.12)), grow(r, during(t, 3.2 + i * 0.12, 4.4 + i * 0.12))));
          };
        },
      },
      {
        seconds: 8,
        caption: `Here the text goes on with “${third.label.trim()}”, which the teacher gave ${percent(third.label_probability)}.\nThe label reports an unlikely turn as if it were the answer.`,
        draw(s) {
          const words = third.context.split(" ").slice(-6).join(" ");
          const row = s.chips(640, 116, ["…", ...pieces(` ${words}`), spelled(third.label)], { center: true, size: 28, gap: 6, pad: 10 });
          mark(row.at(-1));
          s.text(150, 222, "What the teacher says", { size: 26, weight: 600 });
          const answer = rows(s, 264, 246, [...third.top.map((t) => ({ name: t.token.trim(), p: t.probability })), { name: third.label.trim(), p: third.label_probability, fill: "s2" }], 560, 46);
          const came = s.text(264 + 70, 246 + 6 * 46 + 23, "what came next in the text", { size: 22, fill: "s2" });
          const often = s.text(740, 276, "", { size: 60, weight: 700 });
          const first_ = s.lines(740, 314, ["of the time, the token that came", "next is the teacher's first choice"], { size: 24, fill: "ink2" });
          const many = s.text(740, 440, "", { size: 60, weight: 700 });
          const spread = s.lines(740, 478, ["tokens get more than 1% from", "the teacher, on average"], { size: 24, fill: "ink2" });
          const over = s.text(740, 554, `over ${whole(held.tokens)} tokens of held-out stories`, { size: 22, fill: "muted" });
          return (t) => {
            row.forEach((c, i) => show(c, i < row.length - 1 ? during(t, 0.3 + i * 0.06, 0.6 + i * 0.06) : during(t, 1, 1.4)));
            answer.forEach((r, i) => {
              const mine = i === answer.length - 1;
              show(r.g, mine ? during(t, 3, 3.4) : during(t, 1.4 + i * 0.1, 1.8 + i * 0.1));
              grow(r, mine ? during(t, 3.2, 3.8) : during(t, 1.6 + i * 0.1, 2.8 + i * 0.1));
            });
            show(came, during(t, 3.6, 4));
            often.textContent = `${Math.round(100 * held.label_is_first_choice * during(t, 4.2, 5))}%`;
            show(often, during(t, 4.2, 4.6));
            first_.forEach((line) => show(line, during(t, 4.2, 4.6)));
            many.textContent = (held.above_one_percent * during(t, 5, 5.8)).toFixed(1);
            show(many, during(t, 5, 5.4));
            spread.forEach((line) => show(line, during(t, 5, 5.4)));
            show(over, during(t, 5.4, 5.8));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "The loss compares the student's prediction with the teacher's.\nA temperature above 1 flattens both first. Here it stays at 1.",
        draw(s) {
          // the teacher's whole prediction after the first prompt, at a temperature
          const logits = next.prompts[0].logits;
          const peak = Math.max(...logits);
          const at = (T) => {
            const e = logits.map((v) => Math.exp((v - peak) / T));
            const sum = e.reduce((a, b) => a + b, 0);
            return e.map((v) => v / sum);
          };
          const order = at(1).map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 9).map(([, i]) => i);
          const BASE = 498;
          const FULL = 300; // the height of a bar that holds everything
          s.text(150, 140, `The teacher's prediction after “${next.prompts[0].prompt.split(" ").slice(-4).join(" ")}”`, { size: 24, fill: "ink2" });
          s.text(150, 174, "softmax(logits / T)", { size: 22, fill: "muted", mono: true });
          const dial = s.text(1130, 160, "", { size: 46, weight: 700, anchor: "end", fill: "s2" });
          s.line(140, BASE, 1140, BASE, { stroke: "axis" });
          const bars = [...order, -1].map((id, i) => {
            const x = 160 + i * 96 + (id < 0 ? 40 : 0);
            return {
              id,
              x,
              bar: s.rect(x, BASE, 64, 0, { fill: id < 0 ? "axis" : "s1", r: 4 }),
              value: s.text(x + 32, BASE, "", { size: 22, anchor: "middle", fill: "ink2" }),
              name: s.text(x + 32, BASE + 32, id < 0 ? "the other" : tok.text(id).trim(), { size: 22, anchor: "middle", mono: id >= 0, fill: id < 0 ? "muted" : "ink" }),
            };
          });
          s.text(bars.at(-1).x + 32, BASE + 60, `${whole(logits.length - order.length)} tokens`, { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            const T = mix(mix(mix(1, 2, during(t, 1.5, 2.4)), 4, during(t, 2.9, 3.7)), 1, during(t, 4.3, 5.2));
            const p = at(T);
            const shown = order.reduce((sum, id) => sum + p[id], 0);
            const up = during(t, 0.3, 1.1);
            dial.textContent = `T = ${T.toFixed(1)}`;
            for (const b of bars) {
              const share = b.id < 0 ? 1 - shown : p[b.id];
              const h = FULL * share * up;
              set(b.bar, { y: BASE - h, height: h });
              set(b.value, { y: BASE - h - 10 });
              b.value.textContent = percent(share * up);
              show(b.value, during(t, 0.3, 0.7));
            }
          };
        },
      },
      {
        seconds: 7.5,
        caption: "A weight alpha mixes the loss on the teacher's answer with\nthe loss on the label. At 1 the student never sees a label.",
        draw(s) {
          const label = s.group();
          card(label, 80, 122, 290, "the text", "the token that came next");
          label.arrow(382, 166, 478, 166);
          label.text(430, 150, "1 − alpha", { size: 22, anchor: "middle", fill: "s1" });
          const student = s.group();
          card(student, 490, 122, 300, "the student, 14M", "one prediction");
          const answer = s.group();
          card(answer, 910, 122, 290, "the teacher, 42M", "its whole prediction");
          answer.arrow(898, 166, 802, 166);
          answer.text(850, 150, "alpha", { size: 22, anchor: "middle", fill: "s2" });
          const rule = s.text(640, 290, "objective = alpha * soft + (1 - alpha) * loss", { size: 28, anchor: "middle", mono: true });
          const LEFT = 240;
          const WIDE = 800;
          const track = s.group();
          track.rect(LEFT, 350, WIDE, 44, { fill: "s1", r: 6 });
          const part = track.rect(LEFT, 350, 0, 44, { fill: "s2", r: 6 });
          const dial = track.text(640, 334, "", { size: 24, anchor: "middle", weight: 600 });
          const stops = [
            [0, "the text alone, stage 3"],
            [0.5, "half text, half teacher"],
            [1, "the teacher alone"],
          ].map(([alpha, name]) => {
            const x = LEFT + alpha * WIDE;
            track.line(x, 394, x, 410, { stroke: "axis" });
            return { alpha, value: track.text(x, 440, `alpha ${alpha}`, { size: 22, anchor: "middle", mono: true }), name: track.text(x, 472, name, { size: 22, anchor: "middle" }) };
          });
          const from = s.text(LEFT, 540, "orange: the loss on the teacher's answer", { size: 22, fill: "s2" });
          const told = s.text(LEFT + WIDE, 540, "blue: the loss on the label", { size: 22, anchor: "end", fill: "s1" });
          return (t) => {
            const alpha = mix(mix(0, 0.5, during(t, 3, 3.9)), 1, during(t, 4.6, 5.5));
            show(label, during(t, 0.3, 0.7) * mix(1, 0.3, during(t, 4.6, 5.5)));
            show(student, during(t, 0.7, 1.1));
            show(answer, during(t, 1.1, 1.5));
            show(rule, during(t, 1.8, 2.3));
            show(track, during(t, 2.5, 3));
            show(from, during(t, 3, 3.4));
            show(told, during(t, 2.5, 3));
            set(part, { width: WIDE * alpha });
            dial.textContent = `alpha = ${alpha.toFixed(2)}`;
            for (const stop of stops) {
              const here = Math.abs(alpha - stop.alpha) < 0.02;
              stop.name.setAttribute("class", here ? "f-ink" : "f-muted");
              stop.name.setAttribute("font-weight", here ? 700 : 400);
              stop.value.setAttribute("class", `film-mono ${here ? "f-ink" : "f-muted"}`);
            }
          };
        },
      },
      {
        seconds: 8.5,
        caption: "The student is the 14M model of stage 3, on the same batches.\nThe only difference is what it is asked to predict.",
        draw(s) {
          const steps = end.text.step;
          const across = { domain: [500, steps], ticks: [500, 2000, 4000, steps], format: whole, title: "training step" };
          const up = { domain: [1.35, 2.25], ticks: [1.4, 1.6, 1.8, 2.0, 2.2], format: (v) => v.toFixed(1), title: "validation loss" };
          const { px, py } = frame(s, 140, 144, 1020, 356, across, up);
          s.rect(634, 150, 532, 158, { fill: "surface", r: 0 }); // the ground of the legend
          const curve = (log, stroke, width) => s.path(evaluated(log).filter((row) => row.step >= 500).map((row) => [px(row.step), py(row.val_loss)]), { stroke, width });
          const goal = s.line(px(500), py(end.teacher.val_loss), px(steps), py(end.teacher.val_loss), { stroke: "muted", width: 2 });
          set(goal, { "stroke-dasharray": "3 9" });
          const runs = [
            { name: "from the text alone", row: end.text, stroke: "s1", path: curve(text, "s1", 3), from: 0.6 },
            { name: "from the teacher alone", row: end.alone, stroke: "ink2", path: curve(alone, "ink2", 3), from: 2.2 },
            { name: "half text, half teacher", row: end.mixed, stroke: "s2", path: curve(mixed, "s2", 4.5), from: 3.8 },
            { name: "the teacher itself, 42M", row: end.teacher, stroke: "muted", path: goal, from: 5.4 },
          ].map((run, i) => {
            const g = s.group();
            const y = 176 + i * 40;
            const swatch = g.line(650, y - 8, 694, y - 8, { stroke: run.stroke, width: 4 });
            if (run.path === goal) set(swatch, { "stroke-dasharray": "3 9" });
            g.text(708, y, run.name, { size: 24, weight: run.stroke === "s2" ? 700 : 400 });
            return { ...run, g, value: g.text(1150, y, loss(run.row), { size: 24, anchor: "end", weight: run.stroke === "s2" ? 700 : 400, mono: true }) };
          });
          return (t) => {
            for (const run of runs) {
              show(run.g, during(t, run.from - 0.3, run.from + 0.1));
              if (run.path === goal) show(goal, during(t, run.from, run.from + 0.6));
              else trace(run.path, during(t, run.from, run.from + 1.4));
              show(run.value, run.path === goal ? during(t, run.from + 0.4, run.from + 0.8) : during(t, run.from + 1.1, run.from + 1.5));
            }
          };
        },
      },
      {
        seconds: 8.5,
        caption: `The student gets ${Math.round(100 * way(end.mixed))}% of the way to its teacher. In the same\n${minutes} minutes, the text alone still ends lower.`,
        draw(s) {
          // from the model without a teacher, on the left, to the teacher, on the right
          const LEFT = 300;
          const WIDE = 680;
          const at = (row) => LEFT + WIDE * way(row);
          s.text(100, 140, `After the same ${whole(end.text.step)} steps`, { size: 26, weight: 600 });
          s.text(LEFT - 16, 222, "the text alone", { size: 22, anchor: "end", fill: "ink2" });
          s.text(LEFT - 16, 252, loss(end.text), { size: 24, anchor: "end", weight: 700, mono: true });
          s.text(LEFT + WIDE + 16, 222, "the teacher, 42M", { size: 22, fill: "ink2" });
          s.text(LEFT + WIDE + 16, 252, loss(end.teacher), { size: 24, weight: 700, mono: true });
          s.rect(LEFT, 220, WIDE, 16, { fill: "tint", stroke: "axis", r: 8, width: 1.5 });
          const come = s.rect(LEFT, 220, 0, 16, { fill: "s2", r: 8 });
          const only = s.group();
          only.circle(at(end.alone), 228, 9, { fill: "ink2" });
          only.text(at(end.alone), 196, `the teacher alone: ${loss(end.alone)}`, { size: 22, anchor: "middle", fill: "ink2" });
          const dot = s.circle(LEFT, 228, 12, { fill: "s2", stroke: "surface" });
          const both = s.text(at(end.mixed), 274, `half text, half teacher: ${loss(end.mixed)}`, { size: 22, anchor: "middle" });
          const far = s.text(at(end.mixed), 310, "", { size: 28, anchor: "middle", weight: 700, fill: "s2" });

          const later = s.group();
          later.text(100, 396, `After the same ${minutes} minutes`, { size: 26, weight: 600 });
          later.text(1180, 396, "validation loss", { size: 22, anchor: "end", fill: "muted" });
          const most = Math.max(end.text.step, end.timed.step);
          const pair = [
            { name: "the text alone", row: end.text, fill: "s1", bold: end.text.val_loss <= end.timed.val_loss },
            { name: "half text, half teacher", row: end.timed, fill: "s2", bold: end.timed.val_loss < end.text.val_loss },
          ].map((run, i) => {
            const y = 420 + i * 54;
            later.text(400, y + 25, run.name, { size: 22, anchor: "end" });
            return {
              ...run,
              w: (440 * run.row.step) / most,
              bar: later.rect(416, y, 0, 34, { fill: run.fill, r: 5 }),
              steps: later.text(416, y + 25, "", { size: 22, fill: "ink2" }),
              value: later.text(1180, y + 25, loss(run.row), { size: 24, anchor: "end", weight: run.bold ? 700 : 400, fill: run.bold ? "ink" : "ink2", mono: true }),
            };
          });
          const cost = s.text(416, 562, `A step with the teacher takes ${slower.toFixed(1)} times as long.`, { size: 22, fill: "muted" });
          return (t) => {
            const u = during(t, 0.8, 2.6);
            set(come, { width: WIDE * way(end.mixed) * u });
            set(dot, { cx: LEFT + WIDE * way(end.mixed) * u });
            far.textContent = `${Math.round(100 * way(end.mixed) * u)}% of the way`;
            show(far, during(t, 0.8, 1.2));
            show(both, during(t, 2.4, 2.8));
            show(only, during(t, 3, 3.4));
            show(later, during(t, 0.3, 0.7));
            pair.forEach((run, i) => {
              const v = during(t, 4, 5.4); // both for the same time
              set(run.bar, { width: run.w * v });
              set(run.steps, { x: 416 + run.w * v + 12 });
              run.steps.textContent = `${whole(Math.round(run.row.step * v))} steps`;
              show(run.steps, during(t, 4, 4.3));
              show(run.value, during(t, 5.4 + i * 0.2, 5.8 + i * 0.2));
            });
            show(cost, during(t, 6.1, 6.5));
          };
        },
      },
    ],
  };
};
