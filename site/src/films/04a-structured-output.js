// Side topic 4a as a film: a reply held to a form. Asked for JSON, the 0.5B
// model mostly writes prose. Held to the form, every token that would leave it
// is ruled out before the pick. The form is an automaton over bytes, and a
// token is allowed if all of its bytes can be read. The prediction of the
// second scene and the automaton of the third are illustrations; the states
// of the fourth are the first form's own, and every number is read from the
// runs of scripts/evaluate_structured.py and scripts/bench_constrain.py.

import { svg } from "../labs/kit.js";
import { during, mix, set, show, spelled } from "./stage.js";

export const sources = {
  thinkingAsked: "/runs/structured/thinking-asked.json",
  thinkingHeld: "/runs/structured/thinking-constrained.json",
  answerAsked: "/runs/structured/answer-asked.json",
  answerHeld: "/runs/structured/answer-constrained.json",
  firstAsked: "/runs/structured/answer-first-asked.json",
  firstHeld: "/runs/structured/answer-first-constrained.json",
  textAsked: "/runs/structured/reason-then-answer-asked.json",
  textHeld: "/runs/structured/reason-then-answer-constrained.json",
  free: "/runs/eval/gsm8k-qwen.json",
  bench: "/runs/bench/constrain-mps.json",
};

// The first form as the prompt shows it, and as schema.regex writes it.
const ASK = ["Answer with a JSON object and nothing else,", "in this form:", '{"reasoning": "<your work>", "answer": <number>}'];
const FORM = String.raw`\{"reasoning": "(CHARACTER)*", "answer": -?(0|[1-9][0-9]*)(\.[0-9]+)?\}`;
const START = '{"reasoning": "'; // where the first form begins: one byte after another

const count = (v) => Math.round(v).toLocaleString("en");
const percent = (v) => `${(100 * v).toFixed(1)}%`;
const clip = (string, n) => ([...string].length > n ? `${[...string].slice(0, n - 1).join("")}…` : string);
const share = (values, test) => values.filter(test).length / values.length;
function median(values) {
  const v = [...values].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
const dashed = (line) => set(line, { "stroke-dasharray": "3 7" });

function wrap(string, n) {
  const rows = [""];
  for (const word of string.split(" ")) {
    if (rows.at(-1) && rows.at(-1).length + 1 + word.length > n) rows.push(word);
    else rows[rows.length - 1] += (rows.at(-1) ? " " : "") + word;
  }
  return rows;
}
// A reply as plain text: its markup left out, its white space as single spaces.
const plain = (reply) => reply.replace(/\\[()[\]]|\*\*|`/g, "").replace(/\\[a-z]+\{([^}]*)\}/g, "$1").replace(/\s+/g, " ").trim();

// A run keeps every n/30th reply as a sample, with its row: [JSON, JSON in a fence, in the form, tokens].
const sampled = (run) => run.samples.map((sample, k) => ({ ...sample, row: run.rows[k * Math.max(1, Math.floor(run.n / 30))] }));
// A reply ran into the limit if it has as many tokens as were allowed, as scripts/evaluate_structured.py counts it.
const limited = (run) => (row) => row[3] >= run.max_new_tokens - 2;

// The line and the head of an arrow of Stage.arrow, in a color.
function paint(arrow, color) {
  arrow.children[0].setAttribute("class", `s-${color}`);
  arrow.children[1].setAttribute("class", `f-${color}`);
}
// A state of an automaton, with a second ring where a text may end. Returns { x, y, r, ring, rings }.
function state(s, x, y, r, accepting = false) {
  const ring = s.circle(x, y, r, { fill: "raised", stroke: "ink2" });
  const rings = accepting ? [ring, svg("circle", { cx: x, cy: y, r: r - 6, fill: "none", class: "s-ink2", "stroke-width": 2 }, s.g)] : [ring];
  return { x, y, r, ring, rings };
}
// An edge from rim to rim.
function edge(s, a, b) {
  const [dx, dy] = [b.x - a.x, b.y - a.y];
  const d = Math.hypot(dx, dy);
  return s.arrow(a.x + (a.r * dx) / d, a.y + (a.r * dy) / d, b.x - ((b.r + 2) * dx) / d, b.y - ((b.r + 2) * dy) / d, { stroke: "axis", width: 2.5 });
}
// An edge from a state back to itself, below it.
function loop(s, a) {
  const g = svg("g", {}, s.g);
  const [sx, sy, ex, ey] = [a.x - a.r * 0.5, a.y + a.r * 0.87, a.x + a.r * 0.5, a.y + a.r * 0.87];
  const [c1, c2] = [[a.x - 48, a.y + 84], [a.x + 48, a.y + 84]];
  svg("path", { d: `M${sx} ${sy} C${c1} ${c2} ${ex} ${ey}`, fill: "none", class: "s-axis", "stroke-width": 2.5 }, g);
  const angle = Math.atan2(ey - c2[1], ex - c2[0]);
  const corner = (turn) => `${ex - 11 * Math.cos(angle + turn)},${ey - 11 * Math.sin(angle + turn)}`;
  svg("polygon", { points: `${ex},${ey} ${corner(0.45)} ${corner(-0.45)}`, class: "f-axis" }, g);
  return g;
}
// A state as the one the text is in, or not.
const current = (node, on, color = "s1") => node.ring.setAttribute("class", on ? `f-${color} s-${color}` : "f-raised s-ink2");
// A chip as read, as leading nowhere, as not reached, or as it was.
function look(chip, how) {
  const [ground, ink] = { read: ["f-s1 s-s1", "f-surface"], nowhere: ["f-s2 s-s2", "f-surface"], unread: ["f-tint s-axis", "f-muted"] }[how] || ["f-tint s-axis", "f-ink"];
  chip.ground.setAttribute("class", ground);
  chip.label.setAttribute("class", `film-mono ${ink}`);
}

export default ({ thinkingAsked, thinkingHeld, answerAsked, answerHeld, firstAsked, firstHeld, textAsked, textHeld, free, bench }) => {
  const n = thinkingAsked.n;
  // the first form, asked: a reply in prose and one in the form, as they begin
  const asked = sampled(thinkingAsked);
  const prose = asked.find((x) => !x.row[0] && !x.row[1] && !x.reply.slice(0, 200).includes("\\")) || asked[0];
  const formed = asked.find((x) => x.row[2] && x.reply.trim().startsWith('{"')) || asked[0];
  const opening = wrap(plain(prose.reply), 47).slice(0, 3);
  opening[2] = clip(`${opening[2]} …`, 48);

  // the four forms, each asked in the prompt and held to
  const forms = [
    { asked: thinkingAsked, held: thinkingHeld, name: "reasoning, then the answer", note: `cramped: a median of ${median(thinkingHeld.rows.map((r) => r[3]))} tokens, ${median(thinkingAsked.rows.map((r) => r[3]))} asked` },
    { asked: answerAsked, held: answerHeld, name: "the answer alone", note: "a guess: no room to work it out" },
    { asked: firstAsked, held: firstHeld, name: "the answer, then reasoning", note: "a guess: the answer comes before the work" },
    { asked: textAsked, held: textHeld, name: "free text, then the answer as JSON", note: `closest, ${Math.round(100 * (free.mean - textHeld.mean))} points short of stage 7` },
  ];

  // free text, then JSON: the replies that ran into the limit, and how often they are right
  const cutAsked = textAsked.rows.filter(limited(textAsked)).length;
  const cutHeld = textHeld.rows.map((row, i) => ({ row, right: textHeld.scores[i] }));
  const atLimit = cutHeld.filter((x) => limited(textHeld)(x.row));
  const below = cutHeld.filter((x) => !limited(textHeld)(x.row));
  const rightAt = share(atLimit, (x) => x.right > 0);
  const rightBelow = share(below, (x) => x.right > 0);
  // a reply held to the form that ran into the limit: the end of its text, and the JSON line it had to write
  const hurried = sampled(textHeld).filter((x) => limited(textHeld)(x.row));
  const ending = hurried.find((x) => !x.reply.trimEnd().split("\n").slice(0, -1).join(" ").slice(-200).includes("\\")) || hurried[0];
  const endLines = ending.reply.trimEnd().split("\n");
  const tail = wrap(plain(endLines.slice(0, -1).join("\n")), 26).slice(-4);
  tail[0] = `…${tail[0]}`;

  const timing = thinkingHeld.timing;
  const slower = bench.constrained.seconds / bench.free.seconds - 1;
  const perSecond = (run) => (bench.steps * bench.batch) / run.seconds;

  return {
    subtitle: "A reply that cannot leave its form",
    scenes: [
      {
        seconds: 7,
        caption: `Asked for a JSON object, the 0.5B model mostly writes prose.\nOnly ${percent(thinkingAsked.in_form)} of its ${count(n)} replies have the asked form.`,
        draw(s) {
          const prompt = s.group();
          prompt.text(60, 128, `asked after each of the ${count(n)} problems of GSM8K`, { size: 22, fill: "muted" });
          prompt.rect(60, 142, 690, 120, { fill: "tint", stroke: "axis" });
          ASK.forEach((line, i) => prompt.text(80, 178 + i * 32, line, { size: 22, mono: true }));
          const reply = s.group();
          reply.text(60, 306, "a reply in prose, as it begins", { size: 22, fill: "muted" });
          reply.rect(60, 320, 690, 120, { fill: "raised", stroke: "axis" });
          opening.forEach((line, i) => reply.text(80, 356 + i * 32, line, { size: 22, mono: true, fill: "s2" }));
          const json = s.group();
          json.text(60, 494, "a reply in the asked form, as it begins", { size: 22, fill: "muted" });
          json.text(60, 532, clip(formed.reply.trim().split("\n")[0], 50), { size: 22, mono: true, fill: "s1" });

          // every reply a square: in the form, JSON of another shape, or no JSON at all
          const KINDS = [
            { fill: "s1", name: "in the asked form", test: (row) => row[2] },
            { fill: "axis", name: "JSON, not in the asked form", test: (row) => !row[2] && (row[0] || row[1]) },
            { fill: "s2", name: "not JSON, mostly prose", test: (row) => !row[0] && !row[1] },
          ];
          const kinds = thinkingAsked.rows.map((row) => KINDS.findIndex((kind) => kind.test(row))).sort((a, b) => a - b);
          const [COLS, PITCH, X, Y] = [42, 10, 800, 150];
          const title = s.text(X, 128, `the ${count(n)} replies, a square each`, { size: 22, fill: "muted" });
          const lines = [];
          kinds.forEach((k, i) => {
            const r = Math.floor(i / COLS);
            lines[r] ||= s.group();
            lines[r].rect(X + (i % COLS) * PITCH, Y + r * PITCH, 8, 8, { fill: KINDS[k].fill, r: 1.5 });
          });
          const legend = KINDS.map((kind, k) => {
            const g = s.group();
            const y = 500 + k * 32;
            g.rect(X, y - 15, 16, 16, { fill: kind.fill, r: 3 });
            g.text(X + 26, y, kind.name, { size: 22 });
            g.text(1220, y, percent(share(kinds, (x) => x === k)), { size: 22, weight: 700, anchor: "end" });
            return g;
          });
          return (t) => {
            show(prompt, during(t, 0.2, 0.6));
            show(reply, during(t, 1.0, 1.4));
            show(json, during(t, 1.6, 2.0));
            show(title, during(t, 1.8, 2.2));
            lines.forEach((line, r) => show(line, during(t, 2.0 + (r * 2.2) / lines.length, 2.2 + (r * 2.2) / lines.length)));
            legend.forEach((g, k) => show(g, during(t, 4.3 + k * 0.25, 4.7 + k * 0.25)));
          };
        },
      },
      {
        seconds: 6.5,
        caption: "Before each pick, every token that would leave the form gets\nprobability zero. The model chooses among the rest.",
        draw(s) {
          // made up: twenty tokens, the likeliest first, and which of them the form allows
          const raw = [0.26, 0.17, 0.12, 0.09, 0.07, 0.055, 0.045, 0.035, 0.03, 0.025, 0.02, 0.017, 0.014, 0.012, 0.01, 0.008, 0.007, 0.006, 0.005, 0.004];
          const allowed = [2, 5, 9, 14];
          const total = raw.reduce((a, b) => a + b, 0);
          const kept = allowed.reduce((sum, i) => sum + raw[i], 0);
          const [BASE, FULL, X, PITCH] = [410, 400, 260, 46];
          const picked = allowed[0];
          s.text(60, 128, "the prediction for the next token", { size: 22, fill: "muted" });
          s.text(1220, 128, "an illustration, not measured", { size: 22, anchor: "end", fill: "muted" });
          s.line(X - 10, BASE, X + 19 * PITCH + 42, BASE, { stroke: "axis" });
          s.text(X - 20, BASE - 8, "probability", { size: 22, anchor: "end", fill: "ink2" });
          // where a ruled-out token stood before, as an outline
          const ghosts = s.group();
          raw.forEach((p, i) => {
            if (!allowed.includes(i)) set(ghosts.rect(X + i * PITCH, BASE - (FULL * p) / total, 32, (FULL * p) / total, { fill: "surface", stroke: "axis", r: 3, width: 1.5 }), { "stroke-dasharray": "4 4" });
          });
          const bars = raw.map((p, i) => ({ before: p / total, after: allowed.includes(i) ? p / kept : 0, on: allowed.includes(i), bar: s.rect(X + i * PITCH, BASE, 32, 0, { fill: "s1", r: 3 }) }));
          const mask = s.group();
          mask.text(X - 20, BASE + 31, "allowed", { size: 22, anchor: "end", fill: "ink2" });
          bars.forEach((b, i) => mask.rect(X + i * PITCH, BASE + 12, 32, 22, { fill: b.on ? "s3" : "grid", r: 3 }));
          const mark = s.text(X + picked * PITCH + 16, BASE, "picked", { size: 22, weight: 700, anchor: "middle", fill: "s2" });
          const steps = [
            [100, "1  Score", "every token, as in stage 4"],
            [480, "2  Rule out", "a score of −∞, so probability 0"],
            [860, "3  Pick", "among the rest, by their scores"],
          ].map(([x, head, sub]) => {
            const g = s.group();
            g.text(x, 506, head, { size: 26, weight: 700 });
            g.text(x, 540, sub, { size: 22, fill: "ink2" });
            return g;
          });
          return (t) => {
            const rise = during(t, 0.2, 1.1);
            const fall = during(t, 2.0, 2.8);
            const grow = during(t, 3.0, 3.8);
            bars.forEach((b, i) => {
              const p = b.on ? mix(b.before, b.after, grow) : b.before * (1 - fall);
              set(b.bar, { y: BASE - FULL * p * rise, height: FULL * p * rise });
              b.bar.setAttribute("class", i === picked && t >= 4.0 ? "f-s2" : "f-s1");
            });
            show(mask, during(t, 1.4, 1.8));
            show(ghosts, during(t, 2.0, 2.4));
            const top = bars[picked].after * FULL;
            set(mark, { y: BASE - top - 12 });
            show(mark, during(t, 4.0, 4.4));
            show(steps[0], during(t, 0.3, 0.7));
            show(steps[1], during(t, 1.5, 1.9));
            show(steps[2], during(t, 4.0, 4.4));
          };
        },
      },
      {
        seconds: 8,
        caption: "The form is a regular expression, compiled into an automaton:\na table of where each byte leads, or that it leads nowhere.",
        draw(s) {
          const regex = s.group();
          regex.text(60, 128, "the first form as a regular expression", { size: 22, fill: "muted" });
          regex.text(60, 166, FORM, { size: 22, mono: true });
          // the automaton that automaton.Automaton builds for a smaller form, its states as it numbers them
          const toy = s.group();
          toy.text(60, 214, "an illustration: the automaton of a smaller form", { size: 22, fill: "s2" });
          toy.text(600, 214, String.raw`\{"a": (0|[1-9][0-9]*)\}`, { size: 22, mono: true });
          const R = 22;
          const at = [[110, 334], [202, 334], [294, 334], [386, 334], [478, 334], [570, 334], [662, 334], [800, 254], [800, 414], [1110, 254], [960, 414]];
          toy.arrow(62, 334, 110 - R - 2, 334, { stroke: "axis", width: 2.5 });
          const nodes = at.map(([x, y], i) => state(toy, x, y, R, i === 9));
          const EDGES = [
            [0, 1, "{", 156, 320, "middle"],
            [1, 2, '"', 248, 320, "middle"],
            [2, 3, "a", 340, 320, "middle"],
            [3, 4, '"', 432, 320, "middle"],
            [4, 5, ":", 524, 320, "middle"],
            [5, 6, "␣", 616, 320, "middle"],
            [6, 7, "0", 718, 284, "end"],
            [6, 8, "1-9", 718, 404, "end"],
            [7, 9, "}", 955, 242, "middle"],
            [8, 9, "}", 930, 318, "middle"],
            [8, 10, "0-9", 880, 446, "middle"],
            [10, 9, "}", 1052, 358, "start"],
          ];
          const edges = new Map(EDGES.map(([a, b, bytes, lx, ly, anchor]) => {
            const line = edge(toy, nodes[a], nodes[b]);
            toy.text(lx, ly, bytes, { size: 22, mono: true, anchor, fill: "ink2" });
            return [`${a} ${b}`, line];
          }));
          loop(toy, nodes[10]);
          toy.text(1000, 478, "0-9", { size: 22, mono: true, fill: "ink2" });
          for (const node of nodes) toy.g.append(...node.rings); // above the edges
          // a text read byte by byte, the states it passes through, and a text that leads nowhere
          const TEXT = '{"a": 18}';
          const PATH = [0, 1, 2, 3, 4, 5, 6, 8, 10, 9];
          const STEP = 0.36;
          const read = s.chips(110, 452, [...TEXT].map(spelled), { size: 22, pad: 7, gap: 5 });
          const status = s.text(420, 478, "", { size: 22 });
          const other = s.chip(110, 502, "T", { size: 22, pad: 7 });
          const nowhere = s.text(420, 528, "T leads nowhere: no text of the form begins so", { size: 22, fill: "s2" });
          const away = s.group();
          away.arrow(110, 334 - R, 110, 256, { stroke: "s2", width: 2.5 });
          away.text(124, 278, "T", { size: 22, mono: true, fill: "s2" });
          const table = s.text(60, 578, `The first form's table has ${timing.states} rows, one per state, and a column for every byte.`, { size: 22, fill: "ink2" });
          return (t) => {
            show(regex, during(t, 0.2, 0.6));
            show(toy, during(t, 0.8, 1.2));
            const later = t >= 5.0; // the second text, from the first state again
            const k = Math.max(0, Math.min(TEXT.length, Math.floor((t - 1.6) / STEP) + 1)); // bytes of the first text read
            read.forEach((chip, i) => {
              show(chip, during(t, 1.2, 1.5));
              look(chip, i < k ? "read" : "");
            });
            show(other, during(t, 5.0, 5.2));
            look(other, t >= 5.3 ? "nowhere" : "");
            const done = k === TEXT.length && !later;
            nodes.forEach((node, i) => current(node, t >= 1.2 && i === (later ? 0 : PATH[k]), done && i === 9 ? "s3" : "s1"));
            for (const [key, line] of edges) {
              const j = PATH.findIndex((p, m) => `${p} ${PATH[m + 1]}` === key);
              paint(line, !later && j >= 0 && j < k ? "s1" : "axis");
            }
            show(away, during(t, 5.3, 5.6));
            show(nowhere, during(t, 5.3, 5.6));
            status.textContent = k === TEXT.length ? "complete: it ends where a text may end" : `${k} of ${TEXT.length} bytes read`;
            status.setAttribute("class", k === TEXT.length ? "f-s3" : "f-ink2");
            show(status, during(t, 1.6, 1.9));
            show(table, during(t, 5.9, 6.3));
          };
        },
      },
      {
        seconds: 8.5,
        caption: "A token is allowed if all its bytes can be read. A reply may\nend only where complete, and near the limit it is led there.",
        draw(s) {
          // the first sixteen states of the first form's automaton: from each, one byte leads on
          const x = (i) => 96 + 68 * i;
          const CY = 196;
          s.text(60, 128, "the start of the first form's automaton: from each state, one byte leads on", { size: 22, fill: "muted" });
          const nodes = Array.from({ length: START.length + 1 }, (_, i) => ({ x: x(i), y: CY, r: 12 }));
          const edges = [...START].map((byte, i) => {
            s.text((x(i) + x(i + 1)) / 2, CY - 12, spelled(byte), { size: 22, mono: true, anchor: "middle", fill: "ink2" });
            return edge(s, nodes[i], nodes[i + 1]);
          });
          for (const node of nodes) node.ring = s.circle(node.x, node.y, node.r, { fill: "raised", stroke: "ink2" });
          s.text(x(START.length) + 26, CY + 8, "…", { size: 22, fill: "ink2" });
          const away = s.group();
          away.arrow(x(2) + 6, CY + 11, x(2) + 30, CY + 42, { stroke: "s2", width: 2.5 });
          away.text(x(2) + 38, CY + 40, "␣", { size: 22, mono: true, fill: "s2" });

          // three tokens, each read from the state the text is in
          const head = s.group();
          head.text(96, 262, "token", { size: 22, fill: "muted" });
          head.text(330, 262, "its bytes, one after another", { size: 22, fill: "muted" });
          const row = (i) => 280 + i * 54;
          const rows = [
            { token: '{"', bytes: [...'{"'], verdict: "allowed: both bytes can be read", fill: "s3" },
            { token: " reasoning", bytes: [..." reasoning"], verdict: "ruled out: a space leads nowhere", fill: "s2" },
            { token: "<|im_end|>", bytes: null, verdict: "allowed only where the text is complete", fill: "ink2" },
          ].map((r, i) => {
            const g = s.group();
            g.chip(96, row(i), spelled(r.token), { size: 22, pad: 8 });
            const chips = r.bytes ? g.chips(330, row(i), r.bytes.map(spelled), { size: 22, pad: 6, gap: 4 }) : [];
            if (!r.bytes) g.text(330, row(i) + 26, "none: a control token is no text", { size: 22, fill: "muted" });
            return { ...r, g, chips, verdict: s.text(700, row(i) + 26, r.verdict, { size: 22, fill: r.fill }) };
          });
          const allowed = s.text(60, 460, `At the first step, ${count(timing.allowed_at_start)} tokens of the whole vocabulary are allowed.`, { size: 24, weight: 600 });

          // the budget of a reply, and its last tokens, which are led to the nearest end of the form
          const limit = s.group();
          const [WIDE, LAST] = [640, 0.9]; // the bar of a reply's tokens, and where its last ones begin
          limit.text(60, 504, `Near the limit of ${thinkingHeld.max_new_tokens} tokens, a token is allowed only if the end is still in reach.`, { size: 22, fill: "ink2" });
          limit.rect(60, 518, WIDE, 22, { fill: "grid", r: 4 });
          const used = s.rect(60, 518, 0, 22, { fill: "s1", r: 4 });
          const closing = s.rect(60 + WIDE * LAST, 518, 0, 22, { fill: "s2", r: 4 });
          const led = s.text(60 + WIDE + 20, 536, "its last tokens are led to the end", { size: 22, fill: "s2" });
          const shortest = s.text(60, 576, `In tokens, not bytes: the shortest text of the form is ${timing.bytes_to_end} bytes and ${timing.tokens_to_end} tokens.`, { size: 22, fill: "ink2" });
          return (t) => {
            // the walk of {" from the first state
            const walked = [during(t, 1.3, 1.6), during(t, 1.7, 2.0)];
            edges.forEach((line, i) => paint(line, i < 2 && walked[i] >= 0.5 ? "s1" : "axis"));
            const at = walked[1] >= 0.5 ? 2 : walked[0] >= 0.5 ? 1 : 0;
            nodes.forEach((node, i) => current(node, t >= 0.9 && i === at));
            show(away, during(t, 2.9, 3.2));
            rows.forEach((r, i) => {
              const from = 0.9 + i * 1.5;
              show(r.g, during(t, from, from + 0.4));
              r.chips.forEach((chip, j) => look(chip, i === 0 ? (walked[j] >= 0.5 ? "read" : "") : j === 0 ? (t >= 2.9 ? "nowhere" : "") : t >= 2.9 ? "unread" : ""));
              show(r.verdict, during(t, from + 0.9, from + 1.2));
            });
            show(allowed, during(t, 4.6, 5.0));
            show(limit, during(t, 5.2, 5.6));
            set(used, { width: WIDE * LAST * during(t, 5.4, 6.3) });
            set(closing, { width: WIDE * (1 - LAST) * during(t, 6.3, 6.6) });
            show(led, during(t, 6.3, 6.6));
            show(shortest, during(t, 6.5, 6.9));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "Held to the form, every reply fits. The answer alone or first\nis a guess, and reasoning inside a string is cramped.",
        draw(s) {
          const [X, PER] = [670, 1040]; // where the bars begin, and the pixels of a share of 1
          s.text(60, 128, "the reply is", { size: 22, fill: "muted" });
          s.text(640, 128, "in the form", { size: 22, anchor: "end", fill: "muted" });
          s.text(X, 128, "right answers:", { size: 22, fill: "muted" });
          s.rect(X + 160, 113, 18, 18, { fill: "axis", r: 3 });
          s.text(X + 186, 128, "asked", { size: 22 });
          s.rect(X + 268, 113, 18, 18, { fill: "s1", r: 3 });
          s.text(X + 294, 128, "held to the form", { size: 22 });
          const rows = forms.map((form, i) => {
            const y = 186 + i * 96;
            const g = s.group();
            g.text(60, y, form.name, { size: 24, weight: 600 });
            g.text(640, y, percent(form.asked.in_form), { size: 22, anchor: "end", fill: "ink2" });
            const bar = (run, dy, fill) => ({ run, w: PER * run.mean, bar: s.rect(X, y + dy - 17, 0, 20, { fill, r: 3 }), value: s.text(X, y + dy, "", { size: 22, fill: fill === "s1" ? "ink" : "ink2", weight: fill === "s1" ? 700 : 400 }) });
            return {
              g,
              from: 0.5 + i * 0.25,
              asked: bar(form.asked, 0, "axis"),
              held: bar(form.held, 30, "s1"),
              fits: s.text(640, y + 30, percent(form.held.in_form), { size: 22, anchor: "end", weight: 700, fill: "s1" }),
              note: s.text(60, y + 30, form.note, { size: 22, fill: "s2" }),
            };
          });
          const line = s.group();
          const x7 = X + PER * free.mean;
          dashed(line.line(x7, 150, x7, 530, { stroke: "ink", width: 2 }));
          line.text(1220, 560, `stage 7, free to write: ${percent(free.mean)}`, { size: 22, anchor: "end", fill: "ink2" });
          return (t) => {
            rows.forEach((r, i) => {
              show(r.g, during(t, r.from, r.from + 0.4));
              for (const [b, from] of [[r.asked, 1.2], [r.held, 2.6]]) {
                const u = during(t, from + i * 0.12, from + 0.8 + i * 0.12);
                set(b.bar, { width: b.w * u });
                set(b.value, { x: X + b.w * u + 10 });
                b.value.textContent = percent(b.run.mean * u);
                show(b.value, during(t, from + i * 0.12, from + 0.2 + i * 0.12));
              }
              show(r.fits, during(t, 2.6 + i * 0.12, 3.0 + i * 0.12));
              show(r.note, during(t, 4.0 + i * 0.35, 4.4 + i * 0.35));
            });
            show(line, during(t, 3.6, 4.0));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `Free text, then JSON, comes closest. A reply may stop only\nafter its JSON line, and ${count(atLimit.length)} of ${count(textHeld.n)} run into the limit.`,
        draw(s) {
          // the replies to the fourth form by their length, in groups of 32 tokens, and those at the limit apart
          const LIMIT = textHeld.max_new_tokens;
          const GROUP = 32;
          const groups = Math.floor(LIMIT / GROUP);
          const counts = (run) => {
            const c = Array(groups + 1).fill(0);
            for (const row of run.rows) c[limited(run)(row) ? groups : Math.min(groups - 1, Math.floor(row[3] / GROUP))]++;
            return c;
          };
          const [askedCounts, heldCounts] = [counts(textAsked), counts(textHeld)];
          const top = Math.ceil(Math.max(...askedCounts, ...heldCounts) / 100) * 100;
          const [X, BASE, H, PITCH] = [130, 470, 300, 34];
          const py = (v) => BASE - (H * v) / top;
          const at = (i) => (i < groups ? X + i * PITCH : X + groups * PITCH + 26);
          const frame = s.group();
          for (let v = 0; v <= top; v += 100) {
            frame.line(X - 6, py(v), at(groups) + 46, py(v), { stroke: "grid", width: 1.5 });
            frame.text(X - 14, py(v) + 8, count(v), { size: 22, anchor: "end", fill: "muted" });
          }
          frame.line(X - 6, BASE, at(groups) + 46, BASE, { stroke: "axis" });
          for (const v of [0, 128, 256, 384]) frame.text(X + (v / GROUP) * PITCH, BASE + 30, String(v), { size: 22, anchor: "middle", fill: "muted" });
          frame.text(at(groups) + 18, BASE + 30, String(LIMIT), { size: 22, anchor: "middle", fill: "s2", weight: 700 });
          frame.text(X + (groups * PITCH) / 2, BASE + 62, "tokens in a reply", { size: 22, anchor: "middle", fill: "muted" });
          frame.text(at(groups) + 18, BASE + 62, "the limit", { size: 22, anchor: "middle", fill: "s2" });
          frame.rect(X, 113, 18, 18, { fill: "axis", r: 3 });
          frame.text(X + 26, 128, "asked", { size: 22 });
          frame.rect(X + 110, 113, 18, 18, { fill: "s1", r: 3 });
          frame.text(X + 136, 128, "held to the form", { size: 22 });
          const bars = [
            [askedCounts, "axis", 2, 0.6],
            [heldCounts, "s1", 17, 1.8],
          ].flatMap(([c, fill, dx, from]) => c.map((v, i) => ({ v, from: from + i * 0.03, bar: s.rect(at(i) + dx, BASE, 15, 0, { fill, r: 2 }) })));
          const peak = s.group();
          peak.text(at(groups) - 10, py(heldCounts[groups]) - 12, count(heldCounts[groups]), { size: 22, weight: 700, anchor: "end" });
          peak.text(at(groups) - 10, py(askedCounts[groups]) - 6, count(askedCounts[groups]), { size: 22, anchor: "end", fill: "ink2" });

          // how such a reply ends
          const card = s.group();
          card.text(820, 128, "how a reply at the limit ends", { size: 22, fill: "muted" });
          card.rect(820, 142, 400, 190, { fill: "raised", stroke: "axis" });
          tail.forEach((line, i) => card.text(838, 176 + i * 30, line, { size: 22, mono: true, fill: "ink2" }));
          card.text(838, 312, endLines.at(-1), { size: 22, mono: true, weight: 700, fill: "s1" });
          const cut = s.group();
          cut.text(820, 382, "replies that run into the limit", { size: 22, fill: "ink2" });
          cut.text(820, 422, `${count(cutAsked)} asked, ${count(atLimit.length)} held`, { size: 32, weight: 700 });
          const right = s.group();
          right.text(820, 474, "right answers, held to the form", { size: 22, fill: "ink2" });
          const shares = [
            [820, rightAt, "s2", "at the limit"],
            [1010, rightBelow, "s1", "the others"],
          ].map(([x, v, fill, name]) => {
            right.text(x, 550, name, { size: 22, fill: "ink2" });
            return { v, value: s.text(x, 516, "", { size: 34, weight: 700, fill }) };
          });
          return (t) => {
            show(frame, during(t, 0.2, 0.6));
            for (const b of bars) {
              const h = (H * b.v * during(t, b.from, b.from + 0.6)) / top;
              set(b.bar, { y: BASE - h, height: h });
            }
            show(peak, during(t, 2.6, 3.0));
            show(card, during(t, 3.2, 3.6));
            show(cut, during(t, 3.9, 4.3));
            show(right, during(t, 4.6, 5.0));
            for (const x of shares) {
              x.value.textContent = percent(x.v * during(t, 4.6, 5.4));
              show(x.value, during(t, 4.6, 4.9));
            }
          };
        },
      },
      {
        seconds: 6.5,
        caption: `The constraint makes generation ${Math.round(100 * slower)}% slower: a lookup per row\nand a stack of masks, made once per state.`,
        draw(s) {
          const [X, PER] = [420, 50]; // where the bars begin, and the pixels of a second
          s.text(60, 128, `deciding every step: ${bench.steps} steps of ${bench.batch} prompts, the fastest of ${bench.repeats} runs`, { size: 22, fill: "muted" });
          const steps = [
            ["without a constraint", bench.free, "axis", 0.4],
            ["with a constraint", bench.constrained, "s1", 1.0],
          ].map(([name, run, fill, from], i) => {
            const y = 160 + i * 64;
            const w = PER * run.seconds;
            return {
              w,
              from,
              label: s.text(60, y + 28, name, { size: 24, weight: 600 }),
              bar: s.rect(X, y, 0, 40, { fill, r: 5 }),
              value: s.text(X + w - 12, y + 28, `${run.seconds.toFixed(1)} s`, { size: 22, weight: 700, anchor: "end", fill: fill === "s1" ? "surface" : "ink" }),
              rate: s.text(X + w + 16, y + 28, `${count(perSecond(run))} tokens per second`, { size: 22, fill: "ink2" }),
            };
          });
          const cost = s.group();
          cost.text(X, 346, `${Math.round(100 * slower)}%`, { size: 48, weight: 700, fill: "s2" });
          cost.text(X + 90, 346, "more time for the same steps", { size: 24, fill: "ink2" });
          s.text(60, 412, "building the index, once per form", { size: 22, fill: "muted" });
          const builds = [
            ["the first form", thinkingHeld.timing],
            ["the answer alone", answerHeld.timing],
          ].map(([name, run], i) => {
            const y = 436 + i * 64;
            const w = PER * run.index;
            return {
              w,
              from: 2.6 + i * 0.5,
              label: s.text(60, y + 28, `${name}, ${run.states} states`, { size: 24, weight: 600 }),
              bar: s.rect(X, y, 0, 40, { fill: "s1", r: 5 }),
              value: s.text(X + w + 16, y + 28, `${run.index.toFixed(1)} s`, { size: 22, weight: 700 }),
            };
          });
          return (t) => {
            for (const r of [...steps, ...builds]) {
              show(r.label, during(t, r.from, r.from + 0.3));
              set(r.bar, { width: r.w * during(t, r.from + 0.1, r.from + 0.9) });
              show(r.value, during(t, r.from + 0.7, r.from + 1.0));
              if (r.rate) show(r.rate, during(t, r.from + 0.8, r.from + 1.1));
            }
            show(cost, during(t, 2.0, 2.4));
          };
        },
      },
    ],
  };
};
