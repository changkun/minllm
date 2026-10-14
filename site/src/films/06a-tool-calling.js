// Side topic 6a as a film: a tool call is text that the model writes and a
// program runs. GSM8K's solutions mark every calculation for a calculator, so
// each mark becomes a call, and stage 6's LoRA recipe trains on them. The
// conversations shown are the fine-tuned model's own, and every number is read
// from the runs of scripts/finetune.py, scripts/evaluate_tools.py and
// scripts/count_calculations.py. The problem's text comes from stage 6's run.

import { during, place, set, show } from "./stage.js";

export const sources = {
  summary: "/runs/eval/tools-summary.json",
  calculations: "/runs/eval/calculations.json",
  trained: "/runs/sft-gsm8k-tools-lora/meta.json",
  stage6: "/runs/sft-gsm8k-lora/meta.json",
  plain: "/runs/eval/gsm8k-sft-gsm8k-lora.json",
  calling: "/runs/eval/gsm8k-tools-sft-gsm8k-tools-lora.json",
  failing: "/runs/eval/gsm8k-tools-sft-gsm8k-tools-lora-no-execute.json",
};

const MONO = 0.602; // the width of a monospaced character in ems, as in stage.js
const count = (v) => Math.round(v).toLocaleString("en");
const percent = (v) => `${(100 * v).toFixed(1)}%`;
const points = (v) => (100 * v).toFixed(1);
const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(100 * v).toFixed(1)}`;
const wide = (string, size) => [...string].length * size * MONO;

// Pieces of monospaced text in a row, each in its own color: [string, fill, weight].
function spans(s, x, y, parts, size = 22) {
  let at = x;
  return parts.map(([string, fill = "ink", weight = 400]) => {
    const node = s.text(at, y, string, { size, mono: true, fill, weight });
    at += wide(string, size);
    return node;
  });
}

// The value of an expression of numbers, + - * / and parentheses, as tools.calculate reads one.
function value(expression) {
  const tokens = expression.replace(/,/g, "").match(/\d*\.?\d+|[-+*/()]/g) || [];
  let i = 0;
  const atom = () => {
    const token = tokens[i++];
    if (token === "-") return -atom();
    if (token !== "(") return Number(token);
    const v = sum();
    i++;
    return v;
  };
  const product = () => {
    let v = atom();
    while (tokens[i] === "*" || tokens[i] === "/") v = tokens[i++] === "*" ? v * atom() : v / atom();
    return v;
  };
  const sum = () => {
    let v = product();
    while (tokens[i] === "+" || tokens[i] === "-") v = tokens[i++] === "+" ? v + product() : v - product();
    return v;
  };
  return sum();
}

// The model's turns in a conversation: the calls of each, the tool's answer to
// them, what the model wrote first in its next turn, and the answer it gave.
function turns(conversation) {
  return conversation.flatMap((m, i) => {
    if (m.role !== "assistant") return [];
    const [heard, next] = [conversation[i + 1], conversation[i + 2]];
    return [
      {
        text: m.content,
        calls: [...m.content.matchAll(/"expression": "([^"]*)"/g)].map((x) => x[1]),
        result: heard && (heard.content.match(/<tool_response>\n([\s\S]*?)\n<\/tool_response>/) || [])[1],
        wrote: next && (next.content.match(/^\$?(-?[\d,]*\.?\d+)/) || [])[1],
        answer: (m.content.match(/####\s*(\S+)/) || [])[1],
      },
    ];
  });
}

function wrap(string, n) {
  const rows = [""];
  for (const word of string.split(" ")) {
    if (rows.at(-1) && rows.at(-1).length + 1 + word.length > n) rows.push(word);
    else rows[rows.length - 1] += (rows.at(-1) ? " " : "") + word;
  }
  return rows;
}

// A bar of a share that grows from x: its length is full for 1.
function grow(bar, label, x, full, share, u, inside = false) {
  set(bar, { width: full * share * u });
  set(label, { x: inside ? x + full * share * u - 12 : x + full * share * u + 12 });
  label.textContent = percent(share * u);
}

export default ({ summary, calculations, trained, stage6, plain, calling, failing }) => {
  const runs = summary.runs;
  const tuned = runs["gsm8k-tools-sft-gsm8k-tools-lora"];
  const cut = runs["gsm8k-tools-sft-gsm8k-tools-lora-no-execute"];
  const told = runs["gsm8k-tools-qwen-system-markers"]; // the tool instructions tokenized as Qwen's tokenizer does
  const compare = (from, to) => Object.values(summary.comparisons).find((c) => c.from === from && c.to === to);
  const listed = compare("gsm8k-qwen", "gsm8k-tools-qwen-system-markers"); // the published model, told about the tool
  const gain = compare("gsm8k-sft-gsm8k-lora", "gsm8k-tools-sft-gsm8k-tools-lora"); // the tool, against stage 6
  const written = calculations["gsm8k-sft-gsm8k-lora"]; // stage 6's model's own arithmetic
  const made = trained.data;
  const calls = Math.round(tuned.calls_per_problem * tuned.n);

  // the first test problem: its text, and the model's turns on it with the calls run and failing
  const problem = plain.samples[0].prompt.split("\n\n")[0];
  const opening = `${problem.split(/(?<=\.) /).slice(0, 2).join(" ")} …`;
  const egg = calling.samples[0];
  const eggTurns = turns(egg.conversation);
  const setup = eggTurns[0].text.split("\n<tool_call>")[0].split(": ").at(-1);
  const eggFailing = turns(failing.samples[0].conversation);

  return {
    subtitle: "The model writes the call, a program runs it",
    scenes: [
      {
        seconds: 8,
        caption: "A call is text: the model writes JSON between <tool_call>\ntags and ends its turn. The result comes in the next turn.",
        draw(s) {
          const X = 190;
          const row = (y, role, parts) => {
            const g = s.group();
            if (role) g.text(X - 20, y, role, { size: 22, anchor: "end", fill: "s1" });
            spans(g, X, y, parts);
            return g;
          };
          const tools = row(150, "system", [["# Tools … <tools>", "muted"], ['{… "name": "calculate", …}'], ["</tools> …", "muted"]]);
          const listedHere = s.text(1210, 150, "listed for the model", { size: 22, anchor: "end", fill: "muted" });
          const asked = row(196, "user", [["Natalia sold clips to 48 of her friends in April, …"]]);
          const writes = [
            row(242, "assistant", [["Natalia sold 48/2 ="]]),
            row(274, "", [["<tool_call>", "s2"]]),
            row(306, "", [['{"name": "calculate", "arguments": {"expression": "48/2"}}', "s2"]]),
            row(338, "", [["</tool_call>", "s2"], ["<|im_end|>", "muted"]]),
          ];
          const ends = s.text(X + wide("</tool_call><|im_end|>", 22) + 24, 338, "the turn ends", { size: 22, fill: "muted" });
          const program = s.group();
          program.rect(990, 372, 226, 80, { fill: "raised", stroke: "axis" });
          program.text(1103, 402, "the program", { size: 22, anchor: "middle", fill: "ink2" });
          program.text(1103, 436, "calculate 48/2", { size: 22, anchor: "middle", mono: true });
          const to = s.arrow(X + wide('{"name": "calculate", "arguments": {"expression": "48/2"}}', 22) + 12, 314, 1070, 366);
          const back = [row(384, "user", [["<tool_response>", "muted"]]), row(448, "", [["</tool_response>", "muted"]])];
          const result = s.group();
          spans(result, 0, 416, [["24", "s3", 700]]);
          const goes = row(494, "assistant", [["24 clips in May. …"]]);
          return (t) => {
            show(tools, during(t, 0.2, 0.6));
            show(listedHere, during(t, 0.4, 0.8));
            show(asked, during(t, 0.8, 1.2));
            writes.forEach((g, i) => show(g, during(t, 1.5 + i * 0.35, 1.8 + i * 0.35)));
            show(ends, during(t, 2.9, 3.3));
            show(to, during(t, 3.4, 3.8));
            show(program, during(t, 3.4, 3.8));
            for (const g of back) show(g, during(t, 4.0, 4.4));
            // the result leaves the program and comes into the conversation, along its own line
            show(result, during(t, 4.2, 4.4));
            place(result, X + (940 - X) * (1 - during(t, 4.3, 5.1)), 0);
            show(goes, during(t, 5.4, 5.8));
          };
        },
      },
      {
        seconds: 7,
        caption: `GSM8K's solutions mark every calculation for a calculator.\nEach mark becomes a call: ${count(made.conversations)} solutions, ${count(made.calls)} calls.`,
        draw(s) {
          const head = s.text(120, 140, "a solution of GSM8K's training set, as its authors marked it", { size: 22, fill: "muted" });
          const mark = (expression, result) => [["<<", "muted"], [expression, "s2", 700], ["=", "muted"], [result, "s3", 700], [">>", "muted"]];
          const solution = [
            [["Natalia sold 48/2 = "], ...mark("48/2", "24"), ["24 clips in May."]],
            [["Natalia sold 48+24 = "], ...mark("48+24", "72"), ["72 clips altogether in April and May."]],
            [["#### 72"]],
          ].map((parts, i) => {
            const g = s.group();
            spans(g, 120, 186 + i * 34, parts);
            return g;
          });
          const down = s.group();
          down.arrow(640, 276, 640, 318);
          down.text(660, 304, "each mark becomes a call", { size: 22, fill: "ink2" });
          const X = 190;
          const conversation = [
            ["assistant", [["Natalia sold 48/2 = "], ["<tool_call>", "s2"], ['{… "expression": "', "s2"], ["48/2", "s2", 700], ['"}', "s2"], ["</tool_call>", "s2"]]],
            ["user", [["<tool_response>", "muted"], ["24", "s3", 700], ["</tool_response>", "muted"]]],
            ["assistant", [["24 clips in May. Natalia sold 48+24 = "], ["<tool_call>…", "s2"]]],
          ].map(([role, parts], i) => {
            const g = s.group();
            g.text(X - 20, 364 + i * 40, role, { size: 22, anchor: "end", fill: "s1" });
            spans(g, X, 364 + i * 40, parts);
            return g;
          });
          const total = s.text(640, 524, "", { size: 28, weight: 600, anchor: "middle" });
          const agree = made.calculator_disagrees === 0 ? "the calculator gives the marked result for every one of them" : `the calculator disagrees with ${count(made.calculator_disagrees)} of the marks`;
          const checked = s.text(640, 562, agree, { size: 22, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            solution.forEach((g, i) => show(g, during(t, 0.4 + i * 0.25, 0.8 + i * 0.25)));
            show(down, during(t, 1.6, 2.0));
            conversation.forEach((g, i) => show(g, during(t, 2.2 + i * 0.6, 2.6 + i * 0.6)));
            const u = during(t, 4.0, 4.9);
            show(total, during(t, 3.9, 4.1));
            total.textContent = `${count(made.conversations * u)} solutions become conversations with ${count(made.calls * u)} calls`;
            show(checked, during(t, 4.6, 5.0));
          };
        },
      },
      {
        seconds: 6.5,
        caption: "Stage 6's LoRA recipe, on the same problems: the two models\ndiffer only in who does the arithmetic.",
        draw(s) {
          const X = 400;
          const learned = { size: 22, pad: 6, fill: "s2", stroke: null, ink: "surface" };
          const read = { size: 22, pad: 6, fill: "tint", stroke: "axis", ink: "ink2" };
          const label = (y, name, sub, fill) => {
            const g = s.group();
            g.text(70, y, name, { size: 26, weight: 700, fill });
            g.text(70, y + 32, sub, { size: 22, fill: "ink2" });
            return g;
          };
          const plainLabel = label(176, "stage 6", "the arithmetic written out", "s1");
          const plainRow = s.group();
          const [, answer] = plainRow.chips(X, 150, ["Natalia sold 48/2 =", "24", "clips in May."], { gap: 4, ...learned });
          const plainNote = s.group();
          const at = answer.x + answer.w / 2;
          plainNote.arrow(at, 232, at, 194);
          plainNote.text(at + 16, 232, "computed by the model, and learned", { size: 22, fill: "ink2" });
          const toolLabel = label(326, "with calls", "the arithmetic by the tool", "s2");
          const first = s.group();
          first.chips(X, 300, ["Natalia sold 48/2 =", '<tool_call>{… "48/2"}</tool_call>'], { gap: 4, ...learned });
          const second = s.group();
          const [heard] = second.chips(X, 348, ["<tool_response>24</tool_response>", "24 clips in May."], { gap: 4, ...learned });
          set(heard.ground, { class: "f-tint s-axis", "stroke-width": 1.5 });
          set(heard.label, { class: "f-ink2 film-mono" });
          const toolNote = s.group();
          const under = heard.x + heard.w / 2;
          toolNote.arrow(under, 432, under, 394);
          toolNote.text(under + 16, 432, "computed by the calculator, and only read", { size: 22, fill: "ink2" });
          const legend = s.group();
          const one = legend.chip(70, 478, "learned", learned);
          legend.text(70 + one.w + 12, 503, "the assistant's turns, calls included", { size: 22, fill: "ink2" });
          const two = legend.chip(660, 478, "read", read);
          legend.text(660 + two.w + 12, 503, "the problem and the tool's results", { size: 22, fill: "ink2" });
          const same = stage6.lora.r === trained.lora.r && stage6.step === trained.step;
          const recipe = s.text(640, 568, same ? `Both: LoRA of rank ${trained.lora.r}, ${count(trained.step)} steps of ${trained.args.batch_size}, on the same problems` : `stage 6: rank ${stage6.lora.r}, ${stage6.step} steps; with calls: rank ${trained.lora.r}, ${trained.step} steps`, { size: 24, weight: 600, anchor: "middle" });
          return (t) => {
            show(plainLabel, during(t, 0.2, 0.6));
            show(plainRow, during(t, 0.2, 0.6));
            show(plainNote, during(t, 0.9, 1.3));
            show(toolLabel, during(t, 1.6, 2.0));
            show(first, during(t, 1.6, 2.0));
            show(second, during(t, 2.1, 2.5));
            show(toolNote, during(t, 2.7, 3.1));
            show(legend, during(t, 3.4, 3.8));
            show(recipe, during(t, 4.1, 4.5));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "The program runs the loop in rounds. A turn with a call gets\nits result and goes on; a turn without one is the answer.",
        draw(s) {
          const SIZE = 22;
          const PAD = 7;
          const chipW = (string) => wide(string, SIZE) + 2 * PAD;
          // conversations whose calls and results are short enough to stand in a row, one call to a turn
          const lanes = calling.samples
            .map((sample) => turns(sample.conversation))
            .filter((lane) => lane.every((turn) => turn.calls.length <= 1 && [turn.calls[0], turn.result, turn.answer].every((x) => !x || x.length <= 7)))
            .slice(0, 5)
            .map((lane) => lane.map((turn) => (turn.calls.length ? { call: turn.calls[0], result: turn.result } : { answer: `#### ${turn.answer}` })));
          const rounds = Math.max(...lanes.map((lane) => lane.length));
          const widths = Array.from({ length: rounds }, (_, k) => Math.max(...lanes.map((lane) => (lane[k] ? (lane[k].call ? chipW(lane[k].call) + 4 + chipW(lane[k].result) : chipW(lane[k].answer)) : 0))));
          const X0 = 250;
          const lefts = widths.map((_, k) => X0 + widths.slice(0, k).reduce((a, b) => a + b + 26, 0));
          const style = {
            call: { size: SIZE, pad: PAD, fill: "s2", stroke: null, ink: "surface" },
            result: { size: SIZE, pad: PAD, fill: "tint", stroke: "s3", ink: "s3" },
            answer: { size: SIZE, pad: PAD, fill: "raised", stroke: "ink", ink: "ink" },
          };
          const begins = (k) => 0.9 + k * 0.9; // the second at which round k is generated
          const head = s.text(70, 130, "the fine-tuned model's conversations, one turn a round", { size: 22, fill: "muted" });
          const open = s.text(1210, 130, "", { size: 24, weight: 700, anchor: "end", fill: "s2" });
          const heads = lefts.map((x, k) => s.text(x, 178, `round ${k + 1}`, { size: 22, fill: "muted" }));
          const rows = lanes.map((lane, i) => {
            const y = 200 + i * 60;
            const name = s.text(70, y + 26, `conversation ${i + 1}`, { size: 22, fill: "ink2" });
            const cells = lane.map((turn, k) => {
              if (turn.answer) return { k, turn: s.chip(lefts[k], y, turn.answer, style.answer) };
              const said = s.chip(lefts[k], y, turn.call, style.call);
              return { k, turn: said, result: s.chip(lefts[k] + said.w + 4, y, turn.result, style.result) };
            });
            return { name, cells, length: lane.length };
          });
          const legend = s.group();
          let x = 70;
          const shown = lanes[0]; // its first call, that call's result, and its answer stand for the three
          for (const [string, kind, what] of [
            [shown[0].call, "call", "a call"],
            [shown[0].result, "result", "the tool's result"],
            [shown.at(-1).answer, "answer", "a turn without a call: the answer"],
          ]) {
            const c = legend.chip(x, 530, string, style[kind]);
            legend.text(x + c.w + 12, 556, what, { size: 22, fill: "ink2" });
            x += c.w + 12 + what.length * 11 + 30;
          }
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            show(legend, during(t, 0.2, 0.6));
            heads.forEach((h, k) => {
              show(h, during(t, begins(k) - 0.2, begins(k)));
              h.setAttribute("class", t >= begins(k) && t < begins(k + 1) && k < rounds ? "f-ink" : "f-muted");
            });
            let still = 0;
            for (const row of rows) {
              show(row.name, during(t, 0.3, 0.7));
              for (const cell of row.cells) {
                show(cell.turn, during(t, begins(cell.k), begins(cell.k) + 0.3));
                if (cell.result) show(cell.result, during(t, begins(cell.k) + 0.4, begins(cell.k) + 0.65));
              }
              const done = t >= begins(row.length - 1) + 0.3;
              row.name.setAttribute("class", done ? "f-muted" : "f-ink2");
              if (!done) still += 1;
            }
            show(open, during(t, 0.6, 1.0));
            open.textContent = still ? `${still} of ${rows.length} conversations open` : "every conversation answered";
          };
        },
      },
      {
        seconds: 6.5,
        caption: `The published model, told about the tool, calls it in ${percent(told.with_a_call)}\nof problems. The fine-tuned model calls in ${percent(tuned.with_a_call)}.`,
        draw(s) {
          const [X, FULL] = [70, 1000];
          const head = s.text(70, 132, `problems with a call, of ${count(tuned.n)} GSM8K test problems`, { size: 22, fill: "muted" });
          const bars = [
            { y: 186, name: "0.5B as published, told about the tool", share: told.with_a_call, fill: "s1", from: 0.3 },
            { y: 344, name: "fine-tuned on solutions with calls", share: tuned.with_a_call, fill: "s2", from: 1.8 },
          ].map((b) => ({
            ...b,
            title: s.text(X, b.y, b.name, { size: 26, weight: 600 }),
            track: s.rect(X, b.y + 16, FULL, 40, { fill: "tint", r: 6 }),
            bar: s.rect(X, b.y + 16, 0, 40, { fill: b.fill, r: 6 }),
            label: s.text(X, b.y + 45, "", { size: 24, weight: 700, anchor: b.share > 0.5 ? "end" : "start", fill: b.share > 0.5 ? "surface" : "ink" }),
          }));
          const loses = s.text(X, 290, `It solves ${points(-listed.difference)} points fewer than without the tool listed (95% interval ${points(-listed.high)} to ${points(-listed.low)}).`, { size: 22, fill: "ink2" });
          const tiles = [
            [tuned.calls_per_problem.toFixed(1), "calls in a problem, on average"],
            [count(tuned.malformed), `of ${count(calls)} calls malformed`],
            [count(tuned.next_turn_begins_with_it), `of ${count(tuned.results)} results begin the next turn`],
          ].map(([number, what], i) => {
            const g = s.group();
            g.text(70 + i * 380, 478, number, { size: 44, weight: 700, fill: "s2" });
            g.text(70 + i * 380, 514, what, { size: 22, fill: "ink2" });
            return g;
          });
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            for (const b of bars) {
              for (const node of [b.title, b.track, b.label]) show(node, during(t, b.from, b.from + 0.3));
              grow(b.bar, b.label, X, FULL, b.share, during(t, b.from + 0.2, b.from + 1.0), b.share > 0.5);
            }
            show(loses, during(t, 1.3, 1.7));
            tiles.forEach((g, i) => show(g, during(t, 3.1 + i * 0.4, 3.5 + i * 0.4)));
          };
        },
      },
      {
        seconds: 8,
        caption: "It solves as many problems as stage 6's model, within noise:\nits mistakes are in what it calculates, not how.",
        draw(s) {
          // left: the share solved, with its 95% interval
          const [BASE, TALL, TOP] = [440, 280, 0.5];
          const py = (v) => BASE - (TALL * v) / TOP;
          const frame = s.group();
          frame.text(70, 136, "GSM8K solved, with 95% intervals", { size: 22, fill: "muted" });
          for (const tick of [0.1, 0.2, 0.3, 0.4]) {
            frame.line(140, py(tick), 520, py(tick), { stroke: "grid", width: 1.5 });
            frame.text(128, py(tick) + 8, `${Math.round(100 * tick)}%`, { size: 22, anchor: "end", fill: "muted" });
          }
          frame.line(140, BASE, 520, BASE, { stroke: "axis" });
          const bars = [
            { run: { mean: plain.mean, error: plain.error }, name: "stage 6", fill: "s1", x: 170, from: 0.5 },
            { run: tuned, name: "with calls", fill: "s2", x: 350, from: 0.8 },
          ].map((b) => {
            const mid = b.x + 70;
            const bar = s.rect(b.x, BASE, 140, 0, { fill: b.fill, r: 6 });
            const whisker = s.group(); // over the bar, which hides the lower half otherwise
            whisker.line(mid, py(b.run.mean - b.run.error), mid, py(b.run.mean + b.run.error), { stroke: "ink", width: 2 });
            for (const v of [b.run.mean - b.run.error, b.run.mean + b.run.error]) whisker.line(mid - 12, py(v), mid + 12, py(v), { stroke: "ink", width: 2 });
            return {
              ...b,
              bar,
              whisker,
              value: s.text(mid, py(b.run.mean + b.run.error) - 14, percent(b.run.mean), { size: 24, weight: 700, anchor: "middle" }),
              label: s.text(mid, BASE + 32, b.name, { size: 24, weight: 600, anchor: "middle", fill: b.fill }),
            };
          });
          const difference = s.text(70, 538, `difference ${signed(gain.difference)} points`, { size: 24, weight: 600 });
          const interval = s.text(70, 572, `95% interval ${signed(gain.low)} to ${signed(gain.high)}`, { size: 22, fill: "ink2" });
          // right: the first test problem, with the calls run
          const R = 640;
          const where = s.text(R, 136, "the first test problem, with the calls run", { size: 22, fill: "muted" });
          const asked = s.lines(R, 172, wrap(opening, 50), { size: 22, fill: "ink2", leading: 1.3 });
          const set_ = s.text(R, 282, setup, { size: 22, mono: true, weight: 700 });
          const wrong = s.text(R, 314, "the eggs laid times the eggs eaten", { size: 22, weight: 600, fill: "ink" });
          const steps = eggTurns
            .filter((turn) => turn.calls.length)
            .map((turn, i) => {
              const g = s.group();
              const y = 362 + i * 36;
              g.text(R, y, turn.calls[0], { size: 24, mono: true, fill: "s2" });
              g.text(R + 120, y, "→", { size: 24, fill: "muted" });
              g.text(R + 160, y, turn.result, { size: 24, mono: true, weight: 700, fill: "s3" });
              return g;
            });
          const last = eggTurns.at(-1);
          const answered = s.group();
          answered.text(R, 520, `#### ${last.answer}`, { size: 24, mono: true, weight: 700 });
          answered.text(R + 160, 520, `the answer is ${egg.number}`, { size: 24, fill: "ink2" });
          return (t) => {
            show(frame, during(t, 0.2, 0.5));
            for (const b of bars) {
              const u = during(t, b.from, b.from + 1.0);
              set(b.bar, { y: BASE - (TALL * b.run.mean * u) / TOP, height: (TALL * b.run.mean * u) / TOP });
              show(b.label, during(t, b.from - 0.2, b.from + 0.1));
              show(b.whisker, during(t, b.from + 1.0, b.from + 1.3));
              show(b.value, during(t, b.from + 1.0, b.from + 1.3));
            }
            show(difference, during(t, 2.2, 2.6));
            show(interval, during(t, 2.4, 2.8));
            show(where, during(t, 2.9, 3.3));
            for (const line of asked) show(line, during(t, 3.0, 3.4));
            show(set_, during(t, 3.6, 3.9));
            show(wrong, during(t, 3.9, 4.2));
            steps.forEach((g, i) => show(g, during(t, 4.2 + i * 0.25, 4.5 + i * 0.25)));
            show(answered, during(t, 5.4, 5.8));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `When every call fails, it solves ${percent(cut.mean)}: it gets an error\nand writes numbers of its own, ${percent(cut.wrong_calculations / cut.written_calculations)} of them wrong.`,
        draw(s) {
          // left: what is solved, and how much of its own arithmetic is wrong
          const [X, FULL, TOP] = [270, 260, 0.5];
          const groups = [
            {
              title: "GSM8K solved",
              y: 140,
              from: 0.3,
              rows: [
                { name: "calls run", share: tuned.mean, fill: "s2" },
                { name: "every call fails", share: cut.mean, fill: "axis" },
              ],
            },
            {
              title: "written calculations that are wrong",
              y: 320,
              from: 4.2,
              rows: [
                { name: "stage 6's model", share: written.wrong / written.calculations, fill: "s1" },
                { name: "every call fails", share: cut.wrong_calculations / cut.written_calculations, fill: "axis" },
              ],
            },
          ].map((group) => ({
            ...group,
            heading: s.text(70, group.y, group.title, { size: 24, weight: 600 }),
            bars: group.rows.map((row, i) => {
              const y = group.y + 20 + i * 50;
              return {
                ...row,
                name: s.text(70, y + 27, row.name, { size: 22, fill: "ink2" }),
                bar: s.rect(X, y, 0, 36, { fill: row.fill, r: 6 }),
                label: s.text(X, y + 27, "", { size: 24, weight: 700 }),
              };
            }),
          }));
          const counted = s.text(70, 476, `${count(written.wrong)} of ${count(written.calculations)}, and ${count(cut.wrong_calculations)} of ${count(cut.written_calculations)}`, { size: 22, fill: "muted" });
          // right: the first test problem, every call answered with an error
          const R = 660;
          const where = s.group();
          where.text(R, 140, "the first test problem, every call answered with", { size: 22, fill: "muted" });
          where.text(R, 172, eggFailing[0].result, { size: 22, mono: true, fill: "muted" });
          where.text(R, 222, "its call", { size: 22, fill: "muted" });
          where.text(R + 200, 222, "what it writes next", { size: 22, fill: "muted" });
          const steps = eggFailing
            .filter((turn) => turn.calls.length && turn.wrote !== undefined)
            .map((turn, i) => {
              const g = s.group();
              const y = 262 + i * 38;
              const right = value(turn.calls[0]);
              const ok = Math.abs(Number(turn.wrote.replace(/,/g, "")) - right) < 1e-6;
              g.text(R, y, turn.calls[0], { size: 24, mono: true, fill: "s2" });
              g.text(R + 150, y, "→", { size: 24, fill: "muted" });
              g.text(R + 200, y, turn.wrote, { size: 24, mono: true, weight: ok ? 400 : 700 });
              if (!ok) g.text(R + 290, y, `not ${String(right).replace("-", "−")}`, { size: 22, fill: "ink2" });
              return g;
            });
          const answered = s.group();
          answered.text(R, 262 + steps.length * 38 + 14, `#### ${eggFailing.at(-1).answer}`, { size: 24, mono: true, weight: 700 });
          answered.text(R + 200, 262 + steps.length * 38 + 14, `the answer is ${failing.samples[0].number}`, { size: 24, fill: "ink2" });
          return (t) => {
            for (const group of groups) {
              show(group.heading, during(t, group.from, group.from + 0.3));
              group.bars.forEach((b, i) => {
                const from = group.from + 0.2 + i * 0.4;
                show(b.name, during(t, from, from + 0.3));
                show(b.label, during(t, from, from + 0.3));
                grow(b.bar, b.label, X, FULL / TOP, b.share, during(t, from, from + 0.8));
              });
            }
            show(counted, during(t, 5.4, 5.8));
            show(where, during(t, 1.8, 2.2));
            steps.forEach((g, i) => show(g, during(t, 2.3 + i * 0.3, 2.6 + i * 0.3)));
            show(answered, during(t, 2.4 + steps.length * 0.3, 2.8 + steps.length * 0.3));
          };
        },
      },
    ],
  };
};
