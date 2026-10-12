// Side topic 7a as a film: a model is scored on multiple-choice questions
// without writing a word. After "Answer:" it has a probability for each of
// four letters, and the likeliest is its answer. The same model and the same
// questions then give other scores when the question is asked or the answer
// is read another way, and other answers when the options change places.
// Every number is read from the summary that scripts/analyze_choice.py makes
// of the per-question results in runs/eval/.

import { during, set, show, spelled, trace } from "./stage.js";

export const sources = { summary: "/runs/eval/mmlu-summary.json" };

const LETTERS = "ABCD";
const GUESS = 1 / LETTERS.length; // what guessing scores
const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js

const sum = (values) => values.reduce((a, b) => a + b, 0);
const likeliest = (values) => values.indexOf(Math.max(...values));
const count = (v) => Math.round(v).toLocaleString("en");
const percent = (v) => `${(100 * v).toFixed(1)}%`;
const whole = (v) => `${Math.round(100 * v)}%`;
const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;
const plural = (k, word) => `${k} ${word}${k === 1 ? "" : "s"}`;
const clip = (string, n) => ([...string].length > n ? `${[...string].slice(0, n - 1).join("")}…` : string);
const token = (letter) => spelled(` ${letter}`); // a letter as the token that follows "Answer:"
// How long a run took, in the unit that suits it.
const lasted = (seconds) => (seconds < 90 ? plural(Math.round(seconds), "second") : plural(Math.round(seconds / 60), "minute"));

function wrap(string, n) {
  const rows = [""];
  for (const word of string.split(" ")) {
    if (rows.at(-1) && rows.at(-1).length + 1 + word.length > n) rows.push(word);
    else rows[rows.length - 1] += (rows.at(-1) ? " " : "") + word;
  }
  return rows;
}

// How a written reply begins, as it fits a card of the given rows: its markup left out.
function opening(reply, n, rows) {
  const plain = reply.replace(/\\[()[\]]|\*\*|`/g, "").replace(/\\[a-z]+\{([^}]*)\}/g, "$1").replace(/\s+/g, " ").trim();
  const all = wrap(plain, n).map((row) => clip(row, n));
  if (all.length <= rows) return all;
  return [...all.slice(0, rows - 1), `${clip(all[rows - 1], n - 2)} …`];
}

// A line of dashes, for what a score is held against.
const dashed = (line) => set(line, { "stroke-dasharray": "3 7" });

export default ({ summary }) => {
  const { n, runs, written, calibration } = summary;
  const letter = runs["qwen-letter-plain-5shot"]; // the protocol of the authors of MMLU
  const story = runs["base-letter-plain-0shot"];
  const chat0 = runs["qwen-letter-chat-0shot"];
  const q4 = runs["qwen-q4-letter-plain-5shot"];
  const tuned = runs["sft-gsm8k-lora-letter-plain-5shot"];

  // The question that the film follows: the shortest of those that the summary shows in full.
  // shown.orders has the probabilities of its four letters, as published and with the options moved round.
  const shown = [...summary.shown].sort((a, b) => a.question.length - b.question.length)[0];
  const asked = { question: clip(shown.question, 34), options: shown.options.map((option) => clip(option, 9)), answer: shown.answer };
  const prompt = [asked.question, ...asked.options.map((option, j) => `${LETTERS[j]}. ${option}`), "Answer:"];

  // the story model: which letter it says, and how often that letter is right
  const picks = story.picked_letter;
  const answers = summary.right_letter;
  const favorite = likeliest(picks);

  // how the question is asked: four prompts, and the share of the probability that is on a letter
  const asks = [
    { run: letter, how: "text" },
    { run: runs["qwen-letter-plain-0shot"], how: "text" },
    { run: chat0, how: "chat turn" },
    { run: runs["qwen-letter-chat-5shot"], how: "chat turn" },
  ];
  // What the model begins with in a chat turn, where that is not a letter: the tokens it finds
  // likeliest there if the run has kept them, and else the first words of its written replies.
  const first = { tokens: Boolean(chat0.first_tokens), all: [...(chat0.first_tokens || written.first_words || [])].sort((a, b) => b[1] - a[1]).map(([word]) => word.trim()) };
  const begins = first.all.filter((word) => word && !/^\W*[A-D]\b/.test(word)).slice(0, 2).map((word) => clip(word, 5));

  // how the answer is read: a reply that the rule found no answer in, if there is one
  const samples = written.samples || [];
  const unread = samples.filter((sample) => sample.read < 0);
  const sample = unread.find((x) => !x.reply.includes("\\") && x.reply.startsWith("To ")) || unread.find((x) => !x.reply.includes("\\")) || unread[0] || samples[0] || { reply: "", read: -1 };
  // each option's text as the continuation: the sum of its log-probabilities, the first of three rules
  const rules = Object.values(summary.cloze.rules).map((rule) => rule.mean);
  const cloze = { ...summary.cloze.rules.sum, logprobs: shown.cloze.logprobs };

  // the options moved round by 0, 1, 2 and 3 places: what stood at place p stands at (p + k) mod 4
  const orders = summary.orders.scores;
  const { same_option_in_all: same, right_in_all: always, right_in_some: sometimes, right_in_none: never } = summary.orders;

  // how sure and how often right: the questions in groups by the probability of the pick
  // among the four letters, those of at least 20 questions as scripts/make_figures.py draws them
  const levels = calibration.groups.filter((group) => group.share * n >= 20);

  // other models, each against the model as published on the same questions
  const [bits, kept] = [q4.against_published, tuned.against_published];
  const verdict = (d) => (d.low > 0 || d.high < 0 ? "a real difference" : "within noise");
  const moves = (d) => `${d.difference < 0 ? "loses" : "gains"} ${Math.abs(100 * d.difference).toFixed(1)} points`;

  return {
    subtitle: "A score without letting the model write",
    scenes: [
      {
        seconds: 7.5,
        caption: "The model writes nothing. After “Answer:” it has a probability\nfor each of the four letters, and the likeliest is its answer.",
        draw(s) {
          const p = shown.orders[0];
          const pick = likeliest(p);
          const score = pick === asked.answer ? 1 : 0;
          const full = Math.min(600, 300 / Math.max(...p)); // the width of a bar that holds everything
          const card = s.group();
          card.text(70, 132, `the end of the prompt, after ${letter.shots} solved examples`, { size: 22, fill: "muted" });
          card.rect(70, 148, 500, 224, { fill: "tint", stroke: "axis" });
          const lines = prompt.map((line, i) => card.text(92, 186 + i * 32, line, { size: 22, mono: true }));
          const pass = s.group();
          pass.arrow(586, 268, 760, 268);
          pass.text(673, 252, "one forward pass", { size: 22, anchor: "middle", fill: "ink2" });
          const header = s.text(780, 186, "probability of the next token", { size: 22, fill: "muted" });
          // a bar beside each option: the probability of its letter
          const bars = p.map((value, j) => {
            const y = 218 + j * 32;
            return { value, w: full * value, name: s.text(780, y, token(LETTERS[j]), { size: 22, mono: true }), bar: s.rect(836, y - 19, 0, 24, { fill: "s1", r: 4 }), label: s.text(836, y, "", { size: 22, fill: "ink2" }) };
          });
          const other = s.text(780, 346, `any other token: ${percent(Math.max(0, 1 - sum(p)))}`, { size: 22, fill: "muted" });
          const said = s.group();
          said.text(180, 461, "the likeliest letter is its answer", { size: 24, fill: "ink2" });
          said.chip(498, 432, LETTERS[pick], { size: 24, fill: "s2", stroke: null, ink: "surface" });
          const truth = s.group();
          truth.text(620, 461, "the right answer", { size: 24, fill: "ink2" });
          truth.chip(800, 432, LETTERS[asked.answer], { size: 24, fill: "s3", stroke: null, ink: "surface" });
          const scored = s.group();
          scored.text(922, 461, "score", { size: 24, fill: "ink2" });
          scored.chip(990, 432, String(score), { size: 24, fill: score ? "s3" : "tint", stroke: score ? null : "axis", ink: score ? "surface" : "ink" });
          const last = s.text(640, 540, "Nothing is sampled, and no reply has to be parsed.", { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(card, during(t, 0.3, 0.8));
            show(pass, during(t, 1.3, 1.7));
            show(header, during(t, 1.5, 1.9));
            bars.forEach((b, j) => {
              const u = during(t, 1.8 + j * 0.12, 2.8 + j * 0.12);
              show(b.name, during(t, 1.6 + j * 0.12, 1.9 + j * 0.12));
              show(b.label, during(t, 1.8 + j * 0.12, 2.0 + j * 0.12));
              set(b.bar, { width: b.w * u });
              set(b.label, { x: 836 + b.w * u + 10 });
              b.label.textContent = percent(b.value * u);
              b.bar.setAttribute("class", j === pick && t >= 3.6 ? "f-s2" : "f-s1");
            });
            show(other, during(t, 3.0, 3.4));
            // in the prompt: the option it answers with, and then the one that is right
            lines.forEach((line, i) => {
              const color = i - 1 === asked.answer && t >= 4.5 ? "s3" : i - 1 === pick && t >= 3.6 ? "s2" : "ink";
              line.setAttribute("class", `film-mono f-${color}`);
              line.setAttribute("font-weight", color === "ink" ? 400 : 700);
            });
            show(said, during(t, 3.6, 4.0));
            show(truth, during(t, 4.5, 4.9));
            show(scored, during(t, 5.1, 5.5));
            show(last, during(t, 5.5, 5.9));
          };
        },
      },
      {
        seconds: 7,
        caption: `The 0.5B model answers ${percent(letter.mean)} of the ${count(n)} questions right.\nThe 14M story model says ${LETTERS[favorite]} to ${percent(picks[favorite])} of them and scores ${percent(story.mean)}.`,
        draw(s) {
          const ALL = 900; // the width of a bar for all of the questions
          const guess = 70 + ALL * GUESS;
          const title = s.text(70, 128, `share of the ${count(n)} questions answered right`, { size: 22, fill: "muted" });
          const rows = [
            { run: letter, name: "0.5B model as published", color: "s1", y: 190, from: 0.5 },
            { run: story, name: "14M story model", color: "s2", y: 300, from: 2.5 },
          ].map((row) => {
            const w = ALL * row.run.mean;
            return {
              ...row,
              w,
              label: s.text(70, row.y - 14, row.name, { size: 24, weight: 600 }),
              bar: s.rect(70, row.y, 0, 36, { fill: row.color, r: 6 }),
              value: s.text(70 + w - 12, row.y + 26, `${percent(row.run.mean)} ± ${percent(row.run.error)}`, { size: 24, weight: 700, anchor: "end", fill: "surface" }),
              took: s.text(Math.max(70 + w, guess) + 16, row.y + 26, `in ${lasted(row.run.seconds)}${row.run.mean < GUESS ? ", below guessing" : ""}`, { size: 22, fill: "ink2" }),
            };
          });
          const chance = s.group();
          dashed(chance.line(guess, 184, guess, 344, { stroke: "ink", width: 2 }));
          chance.text(guess, 374, `guessing: ${whole(GUESS)}`, { size: 22, anchor: "middle", fill: "ink2" });
          const says = s.lines(70, 452, [`It says ${LETTERS[favorite]} to ${percent(picks[favorite])} of the questions.`, `${LETTERS[favorite]} is the right answer for ${percent(answers[favorite])} of them.`], { size: 26, leading: 1.5 });
          // letter by letter: how often it is right, and how often the story model says it
          const [FLOOR, TALL] = [500, 300];
          const key = s.group();
          key.circle(732, 121, 8, { fill: "axis" });
          key.text(750, 129, "the letter of the right answer", { size: 22 });
          key.circle(732, 153, 8, { fill: "s2" });
          key.text(750, 161, "the letter the story model says", { size: 22 });
          key.line(716, FLOOR, 1220, FLOOR, { stroke: "axis" });
          const pairs = [...LETTERS].flatMap((name, j) => {
            const x = 724 + j * 125;
            key.text(x + 59, FLOOR + 34, name, { size: 24, weight: 600, anchor: "middle" });
            return [
              [answers[j], "axis", x, 3.8],
              [picks[j], "s2", x + 62, 4.3],
            ].map(([value, color, bx, from]) => ({ value, from: from + j * 0.08, bar: s.rect(bx, FLOOR, 56, 0, { fill: color, r: 4 }), label: s.text(bx + 28, FLOOR, "", { size: 22, anchor: "middle", fill: "ink2" }) }));
          });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            for (const row of rows) {
              const u = during(t, row.from + 0.2, row.from + 1.1);
              show(row.label, during(t, row.from, row.from + 0.4));
              set(row.bar, { width: row.w * u });
              set(row.value, { x: 70 + row.w * u - 12 });
              show(row.value, during(t, row.from + 0.9, row.from + 1.2));
              show(row.took, during(t, row.from + 1.1, row.from + 1.5));
            }
            show(chance, during(t, 1.9, 2.3));
            show(key, during(t, 3.5, 3.9));
            for (const b of pairs) {
              const u = during(t, b.from, b.from + 0.7);
              set(b.bar, { y: FLOOR - TALL * b.value * u, height: TALL * b.value * u });
              set(b.label, { y: FLOOR - TALL * b.value * u - 8 });
              b.label.textContent = percent(b.value * u);
              show(b.label, during(t, b.from, b.from + 0.2));
            }
            says.forEach((line, i) => show(line, during(t, 4.9 + i * 0.2, 5.3 + i * 0.2)));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `Asked in a chat turn, the model wants to begin a sentence.\nOnly ${whole(chat0.on_letters)} of its probability is on a letter. It scores ${percent(chat0.mean)}.`,
        draw(s) {
          const ends = prompt.slice(-4);
          const row = (i) => 174 + i * 34;
          // the question as text, which ends where the answer goes
          const text = s.group();
          text.text(70, 128, "as text to continue", { size: 24, weight: 600 });
          text.rect(70, 142, 540, 152, { fill: "tint", stroke: "axis" });
          ends.forEach((line, i) => text.text(92, row(i), line, { size: 22, mono: true }));
          const next = s.group();
          const tokens = next.chips(92 + [...ends.at(-1)].length * 22 * MONO + 12, row(3) - 26, [...LETTERS].map(token), { size: 22, pad: 8, gap: 6, fill: "s1", stroke: null, ink: "surface" });
          next.text(tokens.at(-1).x + tokens.at(-1).w + 14, row(3), "the next token", { size: 22, fill: "muted" });
          // and in the chat format, where the assistant's turn opens after it
          const chat = s.group();
          chat.text(670, 128, "as a user's turn in a chat", { size: 24, weight: 600 });
          chat.rect(670, 142, 540, 152, { fill: "tint", stroke: "axis" });
          [ends.at(-2), `${ends.at(-1)}<|im_end|>`, "<|im_start|>assistant"].forEach((line, i) => chat.text(692, row(i), line, { size: 22, mono: true }));
          const reply = s.group();
          const words = reply.chips(692, row(3) - 26, begins, { size: 22, pad: 8, gap: 6, fill: "raised" });
          reply.text(words.length ? words.at(-1).x + words.at(-1).w + 14 : 692, row(3), `its first ${first.tokens ? "token" : "word"}, when not a letter`, { size: 22, fill: "muted" });
          const head = s.group();
          head.text(430, 328, "of all it could say next, the share on the four letters", { size: 22, fill: "muted" });
          head.text(1100, 328, "score", { size: 22, anchor: "end", fill: "muted" });
          const rows = asks.map((ask, i) => {
            const y = 342 + i * 56;
            const inChat = ask.run.format === "chat";
            const part = ask.run.on_letters;
            const g = s.group();
            g.text(70, y + 28, `${ask.how}, ${ask.run.shots ? `${ask.run.shots} solved examples` : "no examples"}`, { size: 24 });
            g.rect(430, y, 540, 40, { fill: "tint", stroke: "axis", r: 6 });
            return {
              ...ask,
              part,
              g,
              from: inChat ? 4.0 + (i - 2) * 0.4 : 1.3 + i * 0.3,
              bar: s.rect(430, y, 0, 40, { fill: inChat ? "s2" : "s1", r: 6 }),
              value: s.text(442, y + 28, "", { size: 22, weight: 700, fill: "surface" }),
              rest: part < 0.7 ? s.text(430 + 540 * part + 14, y + 28, "something else", { size: 22, fill: "muted" }) : null,
              score: s.text(1100, y + 29, percent(ask.run.mean), { size: 26, weight: 700, anchor: "end" }),
              error: s.text(1110, y + 28, `± ${percent(ask.run.error)}`, { size: 22, fill: "muted" }),
            };
          });
          return (t) => {
            show(text, during(t, 0.3, 0.7));
            show(next, during(t, 0.8, 1.2));
            show(head, during(t, 1.1, 1.5));
            show(chat, during(t, 2.9, 3.3));
            show(reply, during(t, 3.4, 3.8));
            for (const r of rows) {
              const u = during(t, r.from + 0.2, r.from + 1.0);
              show(r.g, during(t, r.from, r.from + 0.3));
              set(r.bar, { width: 540 * r.part * u });
              r.value.textContent = whole(r.part * u);
              show(r.value, during(t, r.from + 0.5, r.from + 0.8));
              if (r.rest) show(r.rest, during(t, r.from + 0.9, r.from + 1.2));
              show(r.score, during(t, r.from + 1.0, r.from + 1.3));
              show(r.error, during(t, r.from + 1.0, r.from + 1.3));
            }
          };
        },
      },
      {
        seconds: 7.5,
        caption: `Reading the answer off a written reply, or scoring each\noption's own text, gives the same model ${percent(written.mean)} and ${percent(cloze.mean)}.`,
        draw(s) {
          const WIDE = 366;
          const p = shown.orders[0];
          const minus = (v) => v.toFixed(2).replace("-", "−");
          const columns = [
            {
              title: "the letter",
              sub: "the probability of its letter",
              rows: p.map((v, j) => `${token(LETTERS[j])}${percent(v).padStart(9)}`),
              marked: likeliest(p),
              reads: `its answer: ${LETTERS[likeliest(p)]}`,
              run: letter,
              note: `all ${count(n)} questions`,
            },
            {
              title: "a written reply",
              sub: "what a rule reads off the reply",
              rows: opening(sample.reply, 24, 4),
              marked: -1,
              reads: `the answer it reads: ${sample.read < 0 ? "none" : LETTERS[sample.read]}`,
              run: written,
              note: `no answer in ${count(written.unreadable)} of ${count(written.n)}`,
            },
            {
              title: "each option's text",
              sub: "the log-probability of its text",
              rows: asked.options.map((option, j) => `${option.padEnd(11)}${minus(cloze.logprobs[j]).padStart(7)}`),
              marked: likeliest(cloze.logprobs),
              reads: `its answer: ${asked.options[likeliest(cloze.logprobs)]}`,
              run: cloze,
              note: Math.max(...rules) > cloze.mean ? `up to ${percent(Math.max(...rules))} by other rules` : `${count(n)} questions, no options shown`,
            },
          ].map((column, c) => {
            const x = 60 + c * 397;
            const g = s.group();
            g.text(x + 2, 132, column.title, { size: 26, weight: 700 });
            g.text(x + 2, 164, column.sub, { size: 22, fill: "ink2" });
            g.rect(x, 182, WIDE, 150, { fill: "tint", stroke: "axis" });
            // the whole width is every question, with a mark at what guessing scores
            g.rect(x, 462, WIDE, 20, { fill: "grid", r: 4 });
            const mark = s.group();
            mark.line(x + WIDE * GUESS, 454, x + WIDE * GUESS, 490, { stroke: "ink", width: 2 });
            if (!c) mark.text(x + WIDE * GUESS, 516, "guessing", { size: 22, anchor: "middle", fill: "ink2" });
            return {
              ...column,
              g,
              mark,
              from: 0.3 + c * 1.7,
              lines: column.rows.map((line, i) => g.text(x + 18, 216 + i * 32, line, { size: 22, mono: true })),
              read: s.text(x + 2, 368, column.reads, { size: 22, fill: "ink2" }),
              score: s.text(x + 2, 440, "", { size: 48, weight: 700 }),
              error: s.text(x + 176, 440, `± ${percent(column.run.error)}`, { size: 24, fill: "muted" }),
              bar: s.rect(x, 462, 0, 20, { fill: c ? "s2" : "s1", r: 4 }),
              last: s.text(x + 2, 552, column.note, { size: 22, fill: "muted" }),
            };
          });
          return (t) => {
            for (const c of columns) {
              const u = during(t, c.from + 1.0, c.from + 1.7);
              show(c.g, during(t, c.from, c.from + 0.4));
              c.lines.forEach((line, i) => {
                const on = i === c.marked && t >= c.from + 0.8;
                line.setAttribute("class", `film-mono f-${on ? "s2" : "ink"}`);
                line.setAttribute("font-weight", on ? 700 : 400);
              });
              show(c.read, during(t, c.from + 0.8, c.from + 1.2));
              c.score.textContent = percent(c.run.mean * u);
              show(c.score, during(t, c.from + 1.0, c.from + 1.2));
              set(c.bar, { width: WIDE * c.run.mean * u });
              show(c.mark, during(t, c.from + 1.5, c.from + 1.8));
              show(c.error, during(t, c.from + 1.6, c.from + 1.9));
              show(c.last, during(t, c.from + 1.6, c.from + 2.0));
            }
          };
        },
      },
      {
        seconds: 8.5,
        caption: `With the options moved round, the score stays the same.\nFor ${percent(1 - same)} of the questions, the pick depends on the letter.`,
        draw(s) {
          const TURNS = [1.9, 3.1, 4.3]; // the seconds at which the options move on by one place
          const MOVE = 0.6;
          const base = (place) => 208 + place * 50; // where the text of a row stands
          const p = shown.orders;
          const full = Math.min(700, 280 / Math.max(...p.flat()));
          const moved = (k) => (k ? `moved round by ${plural(k, "place")}` : "the options as published");
          const question = s.text(70, 128, asked.question, { size: 22, mono: true });
          const state = s.text(70, 160, "", { size: 22 });
          const places = [...LETTERS].map((name, r) => ({
            name: s.text(70, base(r), `${name}.`, { size: 24, weight: 600, mono: true }),
            bar: s.rect(250, base(r) - 22, 0, 30, { fill: "s1", r: 4 }),
            value: s.text(250, base(r), "", { size: 22, fill: "ink2" }),
          }));
          const options = asked.options.map((option, j) => s.text(118, base(j), option, { size: 24, mono: true, fill: j === asked.answer ? "s3" : "ink", weight: j === asked.answer ? 700 : 400 }));
          // every order: its score over all questions, and the option the model picks at this one
          const title = s.text(690, 128, `all ${count(n)} questions, share answered right`, { size: 22, fill: "muted" });
          const here = s.text(1220, 160, "its pick here", { size: 22, anchor: "end", fill: "muted" });
          const scores = orders.map((run, k) => {
            const picked = (likeliest(p[k]) - k + LETTERS.length) % LETTERS.length; // the option under the letter it picks
            return {
              run,
              from: k ? TURNS[k - 1] + 0.5 : 1.0,
              label: s.text(690, base(k), k ? `moved by ${plural(k, "place")}` : "as published", { size: 22 }),
              bar: s.rect(880, base(k) - 22, 0, 30, { fill: "s1", r: 4 }),
              value: s.text(880, base(k) + 1, "", { size: 24, weight: 600 }),
              pick: s.text(1220, base(k), asked.options[picked], { size: 22, mono: true, anchor: "end", fill: picked === asked.answer ? "s3" : "ink", weight: picked === asked.answer ? 700 : 400 }),
            };
          });
          // all of the questions, by how many of the four orders they are answered right in
          const [bx, by, bw, bh] = [70, 404, 1150, 48];
          const frame = s.rect(bx, by, bw, bh, { fill: "surface", stroke: "axis", r: 0, width: 1.5 });
          const parts = [
            { part: always, fill: "s3", ink: "surface", label: "right in all four orders", anchor: "start" },
            { part: sometimes, fill: "s2", ink: "surface", label: "right in some orders, wrong in others", anchor: "middle" },
            { part: never, fill: "tint", ink: "ink", label: "right in none", anchor: "end" },
          ];
          let before = 0;
          for (const part of parts) {
            part.x = bx + bw * before;
            part.w = bw * part.part;
            part.bar = s.rect(part.x, by, 0, bh, { fill: part.fill, r: 0 });
            part.value = s.text(part.x + part.w / 2, by + 33, percent(part.part), { size: 24, weight: 700, anchor: "middle", fill: part.ink });
            part.name = s.text(part.x + (part.anchor === "start" ? 0 : part.anchor === "middle" ? part.w / 2 : part.w), by + bh + 32, part.label, { size: 22, anchor: part.anchor, fill: "ink2" });
            before += part.part;
          }
          const last = s.text(640, 544, `It picks the same option in all four orders for ${percent(same)} of the questions.`, { size: 26, anchor: "middle" });
          return (t) => {
            const turned = sum(TURNS.map((turn) => during(t, turn, turn + MOVE))); // how many places the options have moved by
            const k = TURNS.filter((turn) => t >= turn + MOVE / 2).length; // the order that is being shown
            const [from, part] = [Math.min(TURNS.length - 1, Math.floor(turned)), turned - Math.min(TURNS.length - 1, Math.floor(turned))];
            show(question, during(t, 0.2, 0.6));
            show(state, during(t, 0.3, 0.7));
            state.textContent = moved(k);
            state.setAttribute("class", k ? "f-s2" : "f-ink2");
            const now = p[from].map((value, r) => value + (p[from + 1][r] - value) * part); // the four probabilities, on the way from one order to the next
            places.forEach((place, r) => {
              const value = now[r] * during(t, 0.5 + r * 0.08, 1.2 + r * 0.08);
              show(place.name, during(t, 0.3 + r * 0.08, 0.6 + r * 0.08));
              set(place.bar, { width: full * value });
              place.bar.setAttribute("class", now[r] === Math.max(...now) && t >= 1.3 ? "f-s2" : "f-s1");
              set(place.value, { x: 250 + full * value + 10 });
              place.value.textContent = whole(value);
              show(place.value, during(t, 0.5 + r * 0.08, 0.7 + r * 0.08));
            });
            options.forEach((option, j) => {
              // an option goes down a row, and from the last row out and in again at the first
              const place = (j + turned) % LETTERS.length;
              const over = place - (LETTERS.length - 1);
              set(option, { y: over <= 0 ? base(place) : over < 0.5 ? base(LETTERS.length - 1) + 50 * over : base(0) - 50 * (1 - over) });
              show(option, during(t, 0.3 + j * 0.08, 0.6 + j * 0.08) * (over <= 0 ? 1 : Math.abs(1 - 2 * over)));
            });
            show(title, during(t, 0.8, 1.2));
            show(here, during(t, 1.3, 1.7));
            for (const score of scores) {
              const u = during(t, score.from, score.from + 0.6);
              show(score.label, during(t, score.from, score.from + 0.3));
              set(score.bar, { width: 300 * score.run.mean * u });
              set(score.value, { x: 880 + 300 * score.run.mean * u + 10 });
              score.value.textContent = percent(score.run.mean * u);
              show(score.value, during(t, score.from, score.from + 0.2));
              show(score.pick, during(t, score.from + 0.3, score.from + 0.6));
            }
            show(frame, during(t, 5.2, 5.5));
            const grown = bw * during(t, 5.4, 6.4);
            for (const part of parts) {
              set(part.bar, { width: Math.min(part.w, Math.max(0, grown - (part.x - bx))) });
              show(part.value, grown >= part.x - bx + part.w - 0.5 ? 1 : 0);
              show(part.name, grown >= part.x - bx + part.w - 0.5 ? 1 : 0);
            }
            show(last, during(t, 6.5, 6.9));
          };
        },
      },
      {
        seconds: 6.5,
        caption: `The probability also says how sure the model is. It gives\nits pick ${percent(calibration.confidence)} on average, and is right ${percent(letter.mean)} of the time.`,
        draw(s) {
          const [x, y, w, h] = [150, 160, 400, 320];
          const px = (v) => x + (w * (v - GUESS)) / (1 - GUESS);
          const py = (v) => y + h * (1 - v);
          const frame = s.group();
          for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
            frame.line(x, py(tick), x + w, py(tick), { stroke: "grid", width: 1.5 });
            frame.text(x - 12, py(tick) + 8, whole(tick), { size: 22, anchor: "end", fill: "muted" });
          }
          frame.line(x, y + h, x + w, y + h, { stroke: "axis" });
          for (const tick of [0.25, 0.5, 0.75, 1]) frame.text(px(tick), y + h + 30, whole(tick), { size: 22, anchor: "middle", fill: "muted" });
          frame.text(x + w / 2, y + h + 64, "the probability it gives its pick", { size: 22, anchor: "middle", fill: "muted" });
          frame.text(x, y - 18, "share answered right", { size: 22, fill: "muted" });
          // where a model would be that is right as often as it is sure
          const even = s.group();
          even.line(px(GUESS), py(GUESS), px(1), py(1), { stroke: "axis", width: 2.5 });
          even.text(px(0.75) - 16, py(0.84), "as sure as right", { size: 22, anchor: "end", fill: "muted" });
          const points = levels.map((level) => [px(level.confidence), py(level.right)]);
          const gaps = levels.map((level) => s.line(px(level.confidence), py(level.right), px(level.confidence), py(level.confidence), { stroke: "s2", width: 3 }));
          const curve = s.path(points, { stroke: "s1", width: 4 });
          const dots = points.map(([a, b]) => s.circle(a, b, 7, { fill: "s1", stroke: "surface" }));
          const surest = levels.at(-1);
          const facts = [
            { at: 186, label: "what it gives its pick, on average", value: calibration.confidence, form: percent, fill: "ink", from: 1.4 },
            { at: 300, label: "how often its pick is right", value: letter.mean, form: percent, fill: "s1", from: 2.4 },
            { at: 414, label: "what lies between them, on average", value: calibration.gap, form: (v) => `${(100 * v).toFixed(1)} points`, fill: "s2", from: 3.6 },
          ].map((fact) => ({ ...fact, name: s.text(640, fact.at, fact.label, { size: 22, fill: "ink2" }), number: s.text(640, fact.at + 50, "", { size: 48, weight: 700, fill: fact.fill }) }));
          const last = s.text(640, 536, `Where it says ${whole(surest.confidence)}, it is right ${whole(surest.right)} of the time.`, { size: 24, fill: "ink2" });
          return (t) => {
            show(frame, during(t, 0.2, 0.6));
            show(even, during(t, 0.7, 1.1));
            trace(curve, during(t, 1.2, 2.8));
            dots.forEach((dot, i) => show(dot, during(t, 1.2 + i * 0.2, 1.5 + i * 0.2)));
            gaps.forEach((line, i) => show(line, during(t, 3.2 + i * 0.05, 3.5 + i * 0.05)));
            for (const fact of facts) {
              show(fact.name, during(t, fact.from, fact.from + 0.4));
              show(fact.number, during(t, fact.from, fact.from + 0.4));
              fact.number.textContent = fact.form(fact.value * during(t, fact.from, fact.from + 0.8));
            }
            show(last, during(t, 4.5, 4.9));
          };
        },
      },
      {
        seconds: 7,
        caption: `At 4 bits the 0.5B model ${moves(bits)}: ${verdict(bits)}.\nFine-tuned on GSM8K it ${moves(kept)}: ${verdict(kept)}.`,
        draw(s) {
          const [FLOOR, TALL] = [470, 560];
          const title = s.text(70, 128, `share of the ${count(n)} questions answered right`, { size: 22, fill: "muted" });
          const chance = s.group();
          dashed(chance.line(180, FLOOR - TALL * GUESS, 686, FLOOR - TALL * GUESS, { stroke: "ink2", width: 2 }));
          chance.text(70, FLOOR - TALL * GUESS - 6, "guessing", { size: 22, fill: "ink2" });
          chance.text(70, FLOOR - TALL * GUESS + 22, whole(GUESS), { size: 22, fill: "ink2" });
          s.line(180, FLOOR, 686, FLOOR, { stroke: "axis" });
          const bars = [
            { run: letter, name: "as published", fill: "s1" },
            { run: q4, name: "at 4 bits", fill: "s2" },
            { run: tuned, name: "fine-tuned", fill: "s2" },
          ].map((model, i) => {
            const at = 260 + i * 170;
            const [top, reach] = [TALL * model.run.mean, TALL * model.run.error];
            return {
              ...model,
              top,
              reach,
              from: 0.5 + i * 0.5,
              bar: s.rect(at - 55, FLOOR, 110, 0, { fill: model.fill, r: 6 }),
              // the 95% interval of the share
              interval: s.line(at, FLOOR - top - reach, at, FLOOR - top + reach, { stroke: "ink", width: 2.5 }),
              value: s.text(at, FLOOR, "", { size: 26, weight: 600, anchor: "middle" }),
              label: s.text(at, FLOOR + 34, model.name, { size: 22, anchor: "middle" }),
            };
          });
          // each against the model as published, question by question with the questions drawn again, and the same change where a model has to write
          const changes = [
            { name: "at 4 bits", d: bits, other: q4.on_gsm8k, at: 176, from: 2.6 },
            { name: "fine-tuned on GSM8K through LoRA", d: kept, other: tuned.on_gsm8k, at: 384, from: 4.0 },
          ].map((change) => ({
            ...change,
            lines: [
              s.text(740, change.at, change.name, { size: 24, weight: 700, fill: "s2" }),
              s.text(740, change.at + 46, `${signed(100 * change.d.difference)} points`, { size: 38, weight: 700 }),
              s.text(740, change.at + 82, `95% interval ${signed(100 * change.d.low)} to ${signed(100 * change.d.high)}: ${verdict(change.d)}`, { size: 22 }),
              s.text(740, change.at + 114, `on GSM8K, where it writes: ${signed(100 * (change.other.mean - change.other.published))} points`, { size: 22, fill: "ink2" }),
            ],
          }));
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            show(chance, during(t, 0.3, 0.7));
            for (const b of bars) {
              const u = during(t, b.from, b.from + 0.9);
              set(b.bar, { y: FLOOR - b.top * u, height: b.top * u });
              set(b.value, { y: FLOOR - b.top * u - b.reach - 12 });
              b.value.textContent = percent(b.run.mean * u);
              show(b.value, during(t, b.from, b.from + 0.3));
              show(b.label, during(t, b.from, b.from + 0.3));
              show(b.interval, during(t, b.from + 0.8, b.from + 1.1));
            }
            for (const change of changes) change.lines.forEach((line, i) => show(line, during(t, change.from + i * 0.3, change.from + 0.4 + i * 0.3)));
          };
        },
      },
    ],
  };
};
