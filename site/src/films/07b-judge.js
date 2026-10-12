// Side topic 7b as a film: a language model as the judge of other replies.
// Its verdict is the probability of the token that a started reply goes on
// with. It is measured where the true verdict is known: GSM8K solutions whose
// answers are known, and the comparisons of stage 8's annotator. Every number
// is read from the runs that scripts/judge.py and scripts/analyze_judge.py
// wrote to runs/judge/; the problem and the solution shown come from stage 7.

import { during, mix, place, set, show, spelled } from "./stage.js";

export const sources = {
  solutions: "/runs/eval/gsm8k-qwen.json",
  small: "/runs/judge/gsm8k-qwen.json",
  smallRef: "/runs/judge/gsm8k-qwen-reference.json",
  large: "/runs/judge/gsm8k-qwen2.5-1.5b-instruct.json",
  largeRef: "/runs/judge/gsm8k-qwen2.5-1.5b-instruct-reference.json",
  stories: "/runs/judge/stories-qwen.json",
  storiesTaste: "/runs/judge/stories-qwen-rubric.json",
  storiesLarge: "/runs/judge/stories-qwen2.5-1.5b-instruct.json",
  storiesLargeTaste: "/runs/judge/stories-qwen2.5-1.5b-instruct-rubric.json",
  summary: "/runs/judge/summary.json",
  reward: "/runs/rlhf/reward.json",
};

const BIN = 0.025; // the width of a bar of the histogram of probabilities

const count = (v) => Math.round(v).toLocaleString("en");
const percent = (v) => `${(100 * v).toFixed(1)}%`;
const whole = (v) => `${Math.round(100 * v)}%`;
const clip = (string, n) => ([...string].length > n ? `${[...string].slice(0, n - 1).join("")}…` : string);
const dashed = (line) => set(line, { "stroke-dasharray": "6 8" });

function wrap(string, n) {
  const rows = [""];
  for (const word of string.split(" ")) {
    if (rows.at(-1) && rows.at(-1).length + 1 + word.length > n) rows.push(word);
    else rows[rows.length - 1] += (rows.at(-1) ? " " : "") + word;
  }
  return rows;
}

// A box with a title and a line below it, the line at the smallest size a film uses.
function labelled(s, x, y, w, h, title, sub, { size = 24, stroke = "axis" } = {}) {
  const g = s.group();
  g.rect(x, y, w, h, { fill: "raised", stroke, r: 12 });
  g.text(x + w / 2, y + h / 2 - 4, title, { size, weight: 600, anchor: "middle" });
  g.text(x + w / 2, y + h / 2 + 24, sub, { size: 22, anchor: "middle", fill: "ink2" });
  return g;
}

// A card that stands for a story: gray lines in place of its text.
function story(s, x, y, w, name, widths) {
  const g = s.group();
  g.rect(0, 0, w, 70, { fill: "raised", stroke: "axis" });
  g.text(16, 28, name, { size: 22, weight: 600, fill: "ink2" });
  widths.forEach((part, k) => g.rect(16, 40 + k * 15, (w - 32) * part, 9, { fill: "grid", r: 4 }));
  place(g, x, y);
  return g;
}

export default ({ solutions, small, smallRef, large, largeRef, stories, storiesTaste, storiesLarge, storiesLargeTaste, summary, reward }) => {
  // The solution that the film follows: the shortest problem among those that stage 7 kept in
  // full, of those that are wrong and that the 0.5B judge approves. The kept ones are the first
  // problems, in the order the judge read them, so the sample's place is its place in the judge's runs.
  const kept = solutions.samples.map((sample, i) => ({ ...sample, i, problem: sample.prompt.split("\n\nShow your work")[0].replace(/\s+/g, " ").trim() }));
  const shown = [...kept].filter((x) => small.truth[x.i] === 0 && small.p_correct[x.i] >= 0.5).sort((a, b) => a.problem.length - b.problem.length)[0] || kept[0];
  const p = small.p_correct[shown.i];
  const lines = shown.reply.split("\n").map((line) => line.trim());
  const opening = lines.find((line) => line) || "";
  const boxed = lines.find((line) => line.includes("\\boxed")) || lines.filter((line) => line).at(-1) || "";

  // the four judges of GSM8K solutions, and the one whose cut is moved
  const gsm = [
    { run: small, name: "0.5B" },
    { run: smallRef, name: "0.5B, told the answer" },
    { run: large, name: "1.5B" },
    { run: largeRef, name: "1.5B, told the answer" },
  ];
  const ranked = summary["gsm8k-qwen-reference"];
  // the half of the problems that the cut is measured on, as analyze_judge.py splits them
  const half = smallRef.p_correct.map((x, i) => ({ x, right: smallRef.truth[i] === 1 })).filter((_, i) => i % 2 === 1);
  const agrees = (cut) => half.filter((r) => r.x >= cut === r.right).length / half.length;

  // the story judges, each pair in both orders
  const slotJudges = [
    { run: stories, name: "0.5B judge", key: "stories-qwen" },
    { run: storiesLarge, name: "1.5B judge", key: "stories-qwen2.5-1.5b-instruct" },
  ];
  const favored = (run) => (run.picks_slot_a >= 0.5 ? "A" : "B"); // the slot a judge names more often
  const pA = (run, i) => [run.p_first_as_a[i], 1 - run.p_first_as_b[i]]; // P(A) as given and swapped
  const steady = (run, i) => pA(run, i).every((x) => x >= 0.5) || pA(run, i).every((x) => x < 0.5); // the same slot in both orders
  // the same story in both orders, as scripts/judge.py counts a pair as consistent
  const agreesWithItself = (run, i) => run.p_first_as_a[i] >= 0.5 === run.p_first_as_b[i] >= 0.5;
  const pair = Math.max(0, stories.p_first_as_a.findIndex((_, i) => slotJudges.every((j) => steady(j.run, i))));
  const larger = summary["stories-qwen2.5-1.5b-instruct"];

  return {
    subtitle: "A language model gives the verdict",
    scenes: [
      {
        seconds: 6.5,
        caption: "A program can check a number. Whether a story is good is a\njudgment, so a language model is asked for the verdict.",
        draw(s) {
          const top = s.group();
          top.text(70, 138, "stage 7: a program checks the number", { size: 24, weight: 600 });
          labelled(top, 70, 156, 300, 70, "a solution", "to a math problem");
          top.arrow(384, 191, 516, 191);
          labelled(top, 530, 156, 300, 70, "a program", "compares the number");
          top.arrow(844, 191, 956, 191);
          top.box(970, 156, 248, 70, "right or wrong", "", { size: 24, stroke: "s3" });
          const head = s.text(70, 300, "most tasks: no program can say", { size: 24, weight: 600 });
          const card = s.group();
          card.rect(70, 318, 300, 130, { fill: "raised", stroke: "axis" });
          card.text(90, 352, "a story", { size: 24, weight: 600, fill: "ink2" });
          [0.9, 0.95, 0.82, 0.55].forEach((part, k) => card.rect(90, 368 + k * 18, 260 * part, 10, { fill: "grid", r: 5 }));
          card.text(70, 478, "gray lines stand in for its text", { size: 22, fill: "muted" });
          const into = s.arrow(384, 383, 516, 383);
          const none = s.group();
          dashed(none.rect(530, 338, 300, 90, { fill: "surface", stroke: "axis" }));
          none.text(680, 392, "no program", { size: 24, anchor: "middle", fill: "muted" });
          const judge = s.group();
          labelled(judge, 530, 338, 300, 90, "a judge", "a language model", { size: 26, stroke: "s2" });
          judge.text(680, 458, "reads the task, the reply, what to look for", { size: 22, anchor: "middle", fill: "ink2" });
          const out = s.group();
          out.arrow(844, 383, 956, 383);
          out.box(970, 348, 248, 70, "a verdict", "", { size: 26, stroke: "s2" });
          const known = s.lines(640, 536, ["The judges are measured where the true verdict is known:", `${count(small.n)} GSM8K solutions and ${count(stories.n)} pairs of stories.`], { size: 24, anchor: "middle", fill: "ink2", leading: 1.4 });
          return (t) => {
            show(top, during(t, 0.2, 0.8));
            show(head, during(t, 1.5, 1.9));
            show(card, during(t, 1.7, 2.1));
            show(into, during(t, 2.1, 2.4));
            show(none, during(t, 2.3, 2.6) * (1 - during(t, 3.0, 3.3)));
            show(judge, during(t, 3.3, 3.7));
            show(out, during(t, 3.8, 4.2));
            known.forEach((line, i) => show(line, during(t, 4.3 + i * 0.2, 4.7 + i * 0.2)));
          };
        },
      },
      {
        seconds: 7.5,
        caption: 'The judge\'s reply is started as {"correct": and its next\ntoken, true or false, is read as a probability from 0 to 1.',
        draw(s) {
          const N = 42; // characters to a row of the prompt
          const rows = [
            ["Problem:", "ink2"],
            ...wrap(shown.problem, N).slice(0, 3).map((row) => [clip(row, N), "ink"]),
            ["Solution:", "ink2"],
            [clip(opening, N), "ink"],
            [clip(boxed, N), "ink"],
            ["Is the final answer of the solution", "ink2"],
            ["correct? Reply with a JSON object, …", "ink2"],
            ["<|im_start|>assistant", "muted"],
          ];
          const card = s.group();
          card.text(70, 128, "what the 0.5B judge reads, shortened", { size: 22, fill: "muted" });
          card.rect(70, 142, 600, 46 + rows.length * 30, { fill: "tint", stroke: "axis" });
          rows.forEach(([row, fill], i) => card.text(92, 176 + i * 30, row, { size: 22, mono: true, fill }));
          const started = s.text(92, 176 + rows.length * 30, '{"correct":', { size: 22, mono: true, weight: 700, fill: "s2" });
          // the two verdicts, as the next token
          const FULL = 300;
          const head = s.text(720, 160, "its next token, one of two", { size: 22, fill: "muted" });
          const bars = [p, 1 - p].map((value, j) => {
            const y = 206 + j * 44;
            return { value, name: s.text(720, y, spelled([" true", " false"][j]), { size: 22, mono: true }), bar: s.rect(810, y - 20, 0, 28, { fill: j ? "axis" : "s1", r: 4 }), label: s.text(810, y, "", { size: 22, fill: "ink2" }) };
          });
          // the verdict on a line from 0 to 1, approved above one half
          const [L, R, Y] = [740, 1180, 384];
          const at = (v) => mix(L, R, v);
          const scale = s.group();
          scale.text(720, 316, "the verdict, approved above one half", { size: 22, fill: "muted" });
          scale.rect(at(0.5), Y - 12, at(1) - at(0.5), 24, { fill: "tint", r: 4 });
          scale.line(L, Y, R, Y, { stroke: "axis" });
          [0, 0.5, 1].forEach((v) => {
            scale.line(at(v), Y - 12, at(v), Y + 12, { stroke: "axis" });
            scale.text(at(v), Y + 40, String(v), { size: 22, anchor: "middle", fill: "muted" });
          });
          const dot = s.circle(at(0), Y, 10, { fill: "s1", stroke: "surface" });
          const value = s.text(at(0), Y - 24, "", { size: 24, weight: 700, anchor: "middle" });
          const approves = s.text(720, 476, `The judge ${p >= 0.5 ? "approves" : "rejects"} it.`, { size: 24 });
          const truth = s.text(720, 514, `Stage 7's program scored it ${small.truth[shown.i] ? "right" : "wrong"}.`, { size: 24, weight: 700, fill: small.truth[shown.i] ? "s3" : "s2" });
          const last = s.text(640, 566, "Nothing is generated, and nothing can come out in another form.", { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(card, during(t, 0.2, 0.7));
            show(started, during(t, 0.9, 1.3));
            show(head, during(t, 1.3, 1.7));
            bars.forEach((b, j) => {
              const u = during(t, 1.6 + j * 0.15, 2.5 + j * 0.15);
              show(b.name, during(t, 1.4 + j * 0.15, 1.7 + j * 0.15));
              set(b.bar, { width: FULL * b.value * u });
              set(b.label, { x: 810 + FULL * b.value * u + 10 });
              b.label.textContent = percent(b.value * u);
              show(b.label, during(t, 1.6 + j * 0.15, 1.8 + j * 0.15));
            });
            show(scale, during(t, 2.7, 3.1));
            const u = during(t, 3.0, 3.9);
            set(dot, { cx: at(p * u) });
            set(value, { x: at(p * u) });
            value.textContent = (p * u).toFixed(2);
            show(dot, during(t, 3.0, 3.2));
            show(value, during(t, 3.0, 3.2));
            show(approves, during(t, 4.0, 4.4));
            show(truth, during(t, 4.6, 5.0));
            show(last, during(t, 5.2, 5.6));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `The 0.5B judge approves nearly all, and the 1.5B is lenient.\nTold the right answer, the 1.5B judge is right on ${percent(largeRef.accuracy)}.`,
        draw(s) {
          const WIDE = 270;
          const title = s.text(70, 130, `${count(small.n)} solutions by the 0.5B model to GSM8K problems, ${percent(small.truly_right)} of them right`, { size: 22, fill: "muted" });
          const heads = s.group();
          heads.text(380, 176, "approves right ones", { size: 22, fill: "ink2" });
          heads.text(700, 176, "approves wrong ones", { size: 22, fill: "ink2" });
          heads.text(1220, 176, "agrees with the truth", { size: 22, anchor: "end", fill: "ink2" });
          const rows = gsm.map((judge, i) => {
            const y = 196 + i * 84;
            const bar = (x, v, fill) => {
              const w = WIDE * v;
              const inside = w > 100;
              return { w, bar: s.rect(x, y + 4, 0, 38, { fill, r: 6 }), label: s.text(inside ? x + w - 10 : x + w + 10, y + 31, percent(v), { size: 22, weight: 700, anchor: inside ? "end" : "start", fill: inside ? "surface" : "ink" }) };
            };
            return {
              from: 0.7 + i * 0.85,
              name: s.text(70, y + 31, judge.name, { size: 24, weight: 600 }),
              right: bar(380, judge.run.approves_right, "s3"),
              wrong: bar(700, judge.run.approves_wrong, "s2"),
              agrees: s.text(1220, y + 35, percent(judge.run.accuracy), { size: 34, weight: 700, anchor: "end", fill: i === gsm.length - 1 ? "s1" : "ink" }),
            };
          });
          const last = s.text(70, 556, `A judge that approves everything agrees ${percent(small.truly_right)} of the time.`, { size: 24, fill: "ink2" });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            show(heads, during(t, 0.4, 0.8));
            for (const row of rows) {
              show(row.name, during(t, row.from, row.from + 0.3));
              [row.right, row.wrong].forEach((b, k) => {
                const u = during(t, row.from + 0.1 + k * 0.15, row.from + 0.7 + k * 0.15);
                set(b.bar, { width: b.w * u });
                show(b.label, during(t, row.from + 0.6 + k * 0.15, row.from + 0.8 + k * 0.15));
              });
              show(row.agrees, during(t, row.from + 0.8, row.from + 1.1));
            }
            show(last, during(t, 4.9, 5.3));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `The 0.5B judge's probabilities rank well, but its cut at one\nhalf is wrong. A cut set on half the problems gives ${percent(ranked.accuracy_at_threshold)}.`,
        draw(s) {
          const lo = Math.floor(10 * Math.min(...half.map((r) => r.x))) / 10;
          const [X0, X1, UP, DOWN, TALL] = [130, 1170, 330, 366, 130];
          const px = (v) => X0 + ((X1 - X0) * (v - lo)) / (1 - lo);
          const bins = Math.round((1 - lo) / BIN);
          const counts = [true, false].map((right) => {
            const c = Array(bins).fill(0);
            for (const r of half) if (r.right === right) c[Math.min(bins - 1, Math.floor((r.x - lo) / BIN))] += 1;
            return c;
          });
          const most = Math.max(...counts.flat());
          const title = s.text(70, 128, `the 0.5B judge, told the answer: its probability of true, on ${count(half.length)} problems`, { size: 22, fill: "muted" });
          const key = s.group();
          key.rect(926, 112, 20, 20, { fill: "s3", r: 4 });
          key.text(954, 129, "right", { size: 22, fill: "ink2" });
          key.rect(1048, 112, 20, 20, { fill: "s2", r: 4 });
          key.text(1076, 129, "wrong", { size: 22, fill: "ink2" });
          // what the cut approves, behind the bars
          const band = s.rect(px(0.5), 184, X1 - px(0.5), 326, { fill: "tint", r: 0 });
          const frame = s.group();
          frame.line(X0, UP, X1, UP, { stroke: "axis" });
          frame.line(X0, DOWN, X1, DOWN, { stroke: "axis" });
          for (let v = lo; v <= 1.0001; v += 0.1) frame.text(px(v), UP + 25, v.toFixed(1), { size: 22, anchor: "middle", fill: "muted" });
          const bars = counts.flatMap((c, k) =>
            c.map((n, j) => {
              const h = (TALL * n) / most;
              const x = px(lo + j * BIN) + 1.5;
              return { h, k, j, bar: s.rect(x, k ? DOWN : UP, px(lo + BIN) - px(lo) - 3, 0, { fill: k ? "s2" : "s3", r: 2 }) };
            }),
          );
          // the cut, above and below the row of probabilities
          const cut = [s.line(px(0.5), 184, px(0.5), UP, { stroke: "ink", width: 3 }), s.line(px(0.5), DOWN, px(0.5), 510, { stroke: "ink", width: 3 })];
          const below = s.text(px(0.5) - 10, 172, "rejected", { size: 22, anchor: "end", fill: "ink2" });
          const above = s.text(px(0.5) + 10, 172, "approved", { size: 22, fill: "ink2" });
          const ranks = s.text(70, 540, `A right solution gets the higher probability ${percent(ranked.auc)} of the time.`, { size: 24, fill: "ink2" });
          const counter = s.text(70, 574, "", { size: 26, weight: 700 });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            show(key, during(t, 0.3, 0.7));
            show(frame, during(t, 0.4, 0.8));
            for (const b of bars) {
              const u = during(t, 0.6 + b.j * 0.03, 1.3 + b.j * 0.03);
              set(b.bar, b.k ? { height: b.h * u } : { y: UP - b.h * u, height: b.h * u });
            }
            show(ranks, during(t, 2.0, 2.4));
            const at = mix(0.5, ranked.threshold, during(t, 3.6, 5.2));
            const x = px(at);
            for (const part of cut) set(part, { x1: x, x2: x });
            set(band, { x, width: X1 - x });
            set(below, { x: x - 10 });
            set(above, { x: x + 10 });
            for (const thing of [...cut, band, below, above, counter]) show(thing, during(t, 2.7, 3.1));
            counter.textContent = `cut at ${at.toFixed(2)}: agrees with the truth on ${percent(agrees(at))}`;
          };
        },
      },
      {
        seconds: 8,
        caption: `Each pair of stories is judged in both orders. The verdict\nfollows the slot: ${favored(stories)} for the 0.5B judge, ${favored(storiesLarge)} for the 1.5B judge.`,
        draw(s) {
          const [A, B] = [220, 490]; // where the slots are
          const rowsY = [180, 266];
          const head = s.group();
          head.text(70, 128, "one pair of stories, judged in both orders", { size: 22, fill: "muted" });
          head.text(1220, 128, "gray lines stand in for the stories' text", { size: 22, anchor: "end", fill: "muted" });
          head.text(A + 125, 166, "slot A", { size: 22, anchor: "middle", fill: "ink2" });
          head.text(B + 125, 166, "slot B", { size: 22, anchor: "middle", fill: "ink2" });
          slotJudges.forEach((judge, k) => head.text(880 + k * 220, 166, judge.name, { size: 22, anchor: "middle", fill: "ink2" }));
          const labels = ["as given", "swapped"].map((name, r) => s.text(70, rowsY[r] + 43, name, { size: 24, weight: 600 }));
          const one = [0.95, 0.6];
          const two = [0.8, 0.9];
          const cards = [
            [story(s, A, rowsY[0], 250, "story 1", one), story(s, B, rowsY[0], 250, "story 2", two)],
            [story(s, A, rowsY[1], 250, "story 1", one), story(s, B, rowsY[1], 250, "story 2", two)],
          ];
          // what each judge says to the pair in each order: a slot and its probability
          const says = rowsY.map((y, r) =>
            slotJudges.map((judge, k) => {
              const a = pA(judge.run, pair)[r];
              const slot = a >= 0.5 ? "A" : "B";
              const g = s.group();
              g.chip(840 + k * 220, y + 12, slot, { size: 24, fill: slot === "A" ? "s1" : "s2", stroke: null, ink: "surface" });
              g.text(892 + k * 220, y + 44, whole(Math.max(a, 1 - a)), { size: 24, fill: "ink2" });
              return g;
            }),
          );
          const same = s.text(70, 374, "Swapped, each judge names the same slot, and so the other story.", { size: 24 });
          // every pair: the ones that get the same verdict in both orders
          const all = slotJudges.map((judge, k) => {
            const x = 70 + k * 650;
            const a = judge.run.picks_slot_a;
            const consistent = summary[judge.key].consistent_pairs;
            const g = s.group();
            g.text(x, 418, judge.name, { size: 24, weight: 700 });
            g.text(x, 446, `says ${favored(judge.run)} in ${percent(Math.max(a, 1 - a))} of its ${count(2 * judge.run.n)} verdicts`, { size: 22, fill: "ink2" });
            const rows = Array.from({ length: 10 }, () => s.group());
            const marked = [];
            for (let i = 0; i < judge.run.n; i++) {
              const cell = rows[Math.floor(i / 50) % 10].rect(x + (i % 50) * 10, 458 + Math.floor(i / 50) * 10, 8, 8, { fill: "grid", r: 1.5 });
              if (agreesWithItself(judge.run, i)) marked.push(cell);
            }
            const total = s.text(x, 580, `${count(consistent)} of ${count(judge.run.n)} pairs the same in both orders`, { size: 22, weight: 700, fill: "s3" });
            return { g, rows, marked, total };
          });
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            show(labels[0], during(t, 0.5, 0.9));
            cards[0].forEach((card) => show(card, during(t, 0.5, 0.9)));
            says[0].forEach((g, k) => show(g, during(t, 1.1 + k * 0.2, 1.5 + k * 0.2)));
            // the second row begins as the first, and the stories change places
            show(labels[1], during(t, 1.9, 2.3));
            // story 1 steps down out of the row, both move across, and story 1 steps up into slot B
            const across = during(t, 2.6, 3.2);
            const down = 84 * (during(t, 2.3, 2.6) - during(t, 3.2, 3.5));
            cards[1].forEach((card, i) => {
              show(card, during(t, 1.9, 2.3));
              place(card, i ? mix(B, A, across) : mix(A, B, across), rowsY[1] + (i ? 0 : down));
            });
            says[1].forEach((g, k) => show(g, during(t, 3.6 + k * 0.2, 4.0 + k * 0.2)));
            show(same, during(t, 4.2, 4.6));
            for (const judge of all) {
              show(judge.g, during(t, 4.7, 5.1));
              judge.rows.forEach((row, i) => show(row, during(t, 4.8 + i * 0.05, 5.1 + i * 0.05)));
              for (const cell of judge.marked) cell.setAttribute("class", t >= 5.6 ? "f-s3" : "f-grid");
              show(judge.total, during(t, 5.6, 6.0));
            }
          };
        },
      },
      {
        seconds: 6.5,
        caption: `Averaged over both orders, the 1.5B judge agrees with the\nannotator on ${percent(storiesLarge.agrees_with_annotator)}. Stage 8's reward model agrees on ${percent(reward.agrees_with_annotator)}.`,
        draw(s) {
          const [X, WIDE] = [420, 640];
          const title = s.text(70, 128, `${count(stories.n)} pairs, both orders averaged: how often the verdict is the annotator's`, { size: 22, fill: "muted" });
          // what a coin would agree on, behind the bars
          const coin = s.group();
          dashed(coin.line(X + WIDE / 2, 146, X + WIDE / 2, 404, { stroke: "ink", width: 2 }));
          coin.text(X + WIDE / 2 + 12, 412, "a coin: 50%", { size: 22, fill: "ink2" });
          const rows = [
            { v: stories.agrees_with_annotator, name: "0.5B judge", y: 156 },
            { v: storiesTaste.agrees_with_annotator, name: "0.5B, told the taste", y: 220 },
            { v: storiesLarge.agrees_with_annotator, name: "1.5B judge", y: 284 },
            { v: storiesLargeTaste.agrees_with_annotator, name: "1.5B, told the taste", y: 348 },
            { v: reward.agrees_with_annotator, name: "stage 8's reward model", y: 428, fill: "s1" },
          ].map((row, i) => ({
            ...row,
            from: i < 4 ? 0.5 + i * 0.35 : 2.8,
            label: s.text(70, row.y + 28, row.name, { size: 24, weight: i === 4 ? 700 : 400 }),
            bar: s.rect(X, row.y, 0, 38, { fill: row.fill || "s2", r: 6 }),
            value: s.text(X, row.y + 28, "", { size: 24, weight: 700, anchor: "end", fill: "surface" }),
          }));
          const where = s.text(70, 552, `Where the 1.5B judge gives the same verdict in both orders, it agrees on ${count(larger.agrees_where_consistent * larger.consistent_pairs)} of ${count(larger.consistent_pairs)}.`, { size: 22, fill: "ink2" });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            for (const row of rows) {
              const u = during(t, row.from, row.from + 0.7);
              show(row.label, during(t, row.from - 0.2, row.from + 0.1));
              set(row.bar, { width: WIDE * row.v * u });
              set(row.value, { x: X + WIDE * row.v * u - 12 });
              row.value.textContent = percent(row.v * u);
              show(row.value, during(t, row.from + 0.3, row.from + 0.5));
            }
            show(coin, during(t, 2.1, 2.5));
            show(where, during(t, 4.0, 4.4));
          };
        },
      },
      {
        seconds: 7,
        caption: "Each fault has its remedy: a reference answer, both orders,\nand a threshold set on labeled examples.",
        draw(s) {
          const [FLOOR, TALL] = [470, 250];
          const remedies = [
            { title: "a reference answer", sub: "1.5B judge, GSM8K solutions", names: ["not told", "told"], values: [large.accuracy, largeRef.accuracy], measure: "agrees with the truth" },
            { title: "both orders", sub: "1.5B judge, pairs of stories", names: ["one order", "both"], values: [storiesLarge.agrees_with_annotator_as_shown, storiesLarge.agrees_with_annotator], measure: "agrees with the annotator" },
            { title: "a threshold", sub: "0.5B judge, told the answer", names: ["at 0.5", `at ${ranked.threshold.toFixed(2)}`], values: [ranked.accuracy_at_half, ranked.accuracy_at_threshold], measure: "agrees, on the other half" },
          ].map((remedy, c) => {
            const x = 70 + c * 395;
            const g = s.group();
            g.text(x, 140, remedy.title, { size: 28, weight: 700 });
            g.text(x, 172, remedy.sub, { size: 22, fill: "ink2" });
            g.line(x, FLOOR, x + 350, FLOOR, { stroke: "axis" });
            g.text(x, 550, remedy.measure, { size: 22, fill: "muted" });
            return {
              g,
              from: 0.4 + c * 1.1,
              bars: remedy.values.map((v, k) => {
                const at = x + 30 + k * 160;
                return {
                  h: TALL * v,
                  v,
                  bar: s.rect(at, FLOOR, 130, 0, { fill: k ? "s1" : "axis", r: 6 }),
                  value: s.text(at + 65, FLOOR, "", { size: 26, weight: 700, anchor: "middle", fill: k ? "ink" : "ink2" }),
                  name: s.text(at + 65, FLOOR + 32, remedy.names[k], { size: 22, anchor: "middle", fill: "ink2" }),
                };
              }),
            };
          });
          return (t) => {
            for (const remedy of remedies) {
              show(remedy.g, during(t, remedy.from, remedy.from + 0.4));
              remedy.bars.forEach((b, k) => {
                const from = remedy.from + 0.2 + k * 0.4;
                const u = during(t, from, from + 0.6);
                set(b.bar, { y: FLOOR - b.h * u, height: b.h * u });
                set(b.value, { y: FLOOR - b.h * u - 10 });
                b.value.textContent = percent(b.v * u);
                show(b.value, during(t, from, from + 0.2));
                show(b.name, during(t, from, from + 0.3));
              });
            }
          };
        },
      },
    ],
  };
};
