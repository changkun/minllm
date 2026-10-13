// Chapter 3c as a film: a run of N parameters on D tokens costs about 6 N D
// operations, so at a fixed budget a larger model sees fewer tokens. Eight
// sizes of stage 3's model are trained at four budgets; at every budget the
// loss against size is a valley whose floor is the best size, near 20 tokens
// per parameter. One law fitted on the smaller budgets predicts the largest,
// and says what stage 3's own choice of size cost. The runs, the fits and the
// predictions are those of runs/scaling/summary.json.

import { svg } from "../labs/kit.js";
import { during, mix, set, show, trace } from "./stage.js";

export const sources = {
  summary: "/runs/scaling/summary.json",
  base: "/runs/base/log.jsonl",
};

const millions = (n) => `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`; // parameters: 0.36M, 7.77M, 14.2M
const tokens = (n) => `${n < 1e7 ? (n / 1e6).toFixed(1) : Math.round(n / 1e6)}M`; // tokens: 8.2M, 34M, 215M
const budget = (key) => key.replace("+", ""); // 3e+14 as 3e14
const rate = (v) => v.toExponential(0).replace("+", ""); // 0.004 as 4e-3
const per = (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v)));
function operations(v) {
  const e = Math.floor(Math.log10(v));
  return `${(v / 10 ** e).toFixed(1)}e${e}`;
}
// The loss that a fitted L(N, D) = E + A / N^alpha + B / D^beta predicts, as minllm/scaling.py computes it.
const law = (fit, n, d) => fit.E + fit.A / n ** fit.alpha + fit.B / d ** fit.beta;
// Values from a to b, evenly apart on a log scale.
const spread = (a, b, n = 80) => Array.from({ length: n + 1 }, (_, i) => a * (b / a) ** (i / n));

// The frame of a chart, with its labels at 22 pixels. Returns { px, py }.
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
  if (across.title) s.text(x + w / 2, y + h + 62, across.title, { size: 22, anchor: "middle", fill: "muted" });
  if (up.title) s.text(x, y - 16, up.title, { size: 22, fill: "muted" });
  return { px, py };
}

// How far along a line through points each point is, from 0 to 1.
function along(points) {
  let run = 0;
  const at = points.map(([x, y], i) => (run += i ? Math.hypot(x - points[i - 1][0], y - points[i - 1][1]) : 0));
  return at.map((d) => d / (run || 1));
}

export default ({ summary, base }) => {
  const { valleys, sweep, stage3 } = summary;
  const keys = Object.keys(valleys).sort((a, b) => Number(a) - Number(b));
  const last = keys.at(-1);
  const largest = valleys[last].runs.slice().sort((a, b) => a.params - b.params);
  const dimension = (name) => Number(name.split("-d").at(-1));

  // the family, and the best learning rate of each size in the sweep
  const family = [...new Set(sweep.map((r) => r.dim))]
    .sort((a, b) => a - b)
    .map((dim) => {
      const tried = sweep.filter((r) => r.dim === dim).sort((a, b) => a.lr - b.lr);
      return { dim, params: tried[0].params, tried, best: tried.reduce((a, b) => (b.loss < a.loss ? b : a)) };
    });
  const rates = [...new Set(sweep.map((r) => r.lr))].sort((a, b) => a - b);
  const ours = family.find((f) => f.params === stage3.params);

  // the runs of every budget, and the floors of the valleys
  const runs = keys.flatMap((key) => valleys[key].runs);
  const ratios = keys.map((key) => valleys[key].tokens / valleys[key].params);

  // the law fitted without the largest budget, and what it said about that budget's runs
  const early = summary.fit_without_largest;
  const fitted = keys.slice(0, -1).reduce((n, key) => n + valleys[key].runs.length, 0);
  const misses = summary.predictions.map((p) => ({ ...p, miss: p.predicted - p.loss })).sort((a, b) => a.params - b.params);
  const worst = Math.max(...misses.map((p) => Math.abs(p.miss)));
  const high = misses.every((p) => p.miss > 0);

  // stage 3's choice against the law's, at its compute
  const best = stage3.best;
  const cost = stage3.predicted - best.predicted;
  const peak = Math.max(...base.filter((e) => "lr" in e).map((e) => e.lr));
  const tuned = peak / ours.best.lr === 0.5 ? "at half its best learning rate" : `at learning rate ${rate(peak)}, its best ${rate(ours.best.lr)}`;

  return {
    subtitle: "How large a model, and how much text, for a budget",
    scenes: [
      {
        seconds: 7.5,
        caption: "A run of N parameters on D tokens costs about C = 6 N D.\nAt a fixed C, a larger model sees fewer tokens.",
        draw(s) {
          // a run as a box N wide and D high: at one budget its area stays
          const X = 130;
          const BASE = 520;
          const wide = (n) => (130 * n) / largest[0].params;
          const tall = (d) => (360 * d) / largest[0].tokens;
          const box = s.rect(X, BASE, 0, 0, { fill: "tint", stroke: "s1", r: 6, width: 3 });
          const inside = s.text(0, 0, "", { size: 28, weight: 700, anchor: "middle" });
          const unit = s.text(0, 0, "operations", { size: 22, anchor: "middle", fill: "ink2" });
          const across = s.text(0, BASE + 36, "", { size: 24, anchor: "middle" });
          const up = s.text(X, 0, "", { size: 24 });
          const sized = s.group();
          sized.text(690, 196, "C = 6 N D", { size: 60, weight: 700, mono: true });
          sized.text(690, 244, "6 operations per parameter per token", { size: 24, fill: "ink2" });
          const forward = s.group();
          forward.text(690, 326, "2", { size: 44, weight: 700, fill: "s1" });
          forward.text(740, 322, "forward: a multiply and an add", { size: 24 });
          const backward = s.group();
          backward.text(690, 396, "4", { size: 44, weight: 700, fill: "s1" });
          backward.lines(740, 392, ["backward: a gradient for the input", "and one for the weight"], { size: 24 });
          const said = s.lines(690, 500, [`The four runs at a budget of ${budget(last)},`, "as boxes of N by D: their area stays."], { size: 24, fill: "ink2" });
          const moves = [
            [2.2, 2.7],
            [3.0, 3.5],
            [3.8, 4.3],
          ];
          return (t) => {
            // where among the runs the box is, and the run whose numbers are shown
            const at = moves.reduce((sum, [a, b]) => sum + during(t, a, b), 0);
            const i = Math.min(largest.length - 2, Math.floor(at));
            const u = at - i;
            const n = largest[i].params ** (1 - u) * largest[i + 1].params ** u;
            const d = largest[i].tokens ** (1 - u) * largest[i + 1].tokens ** u;
            const run = largest[Math.round(at)];
            const w = wide(n);
            const h = tall(d);
            set(box, { y: BASE - h, width: w, height: h });
            set(inside, { x: X + w / 2, y: BASE - h / 2 - 2 });
            set(unit, { x: X + w / 2, y: BASE - h / 2 + 26 });
            inside.textContent = operations(6 * run.params * run.tokens);
            set(across, { x: X + w / 2 });
            across.textContent = `N = ${millions(run.params)} parameters`;
            set(up, { y: BASE - h - 16 });
            up.textContent = `D = ${tokens(run.tokens)} tokens`;
            for (const thing of [box, inside, unit, across, up]) show(thing, during(t, 0.4, 0.9));
            show(sized, during(t, 0.2, 0.6));
            show(forward, during(t, 0.9, 1.3));
            show(backward, during(t, 1.4, 1.8));
            said.forEach((line) => show(line, during(t, 2.2, 2.6)));
          };
        },
      },
      {
        seconds: 7,
        caption: `Stage 3's model has eight widths, ${millions(family[0].params)} to ${millions(family.at(-1).params)} parameters.\nEach has its best learning rate: small ones take larger steps.`,
        draw(s) {
          const X = 250;
          const Y = 160;
          const W = 900;
          const H = 280;
          const cx = (i) => X + (W * (i + 0.5)) / family.length;
          const k = family.indexOf(ours);
          const marked = s.group();
          marked.rect(cx(k) - 54, Y - 4, 108, H + 84, { fill: "tint", r: 10 });
          marked.text(cx(k), Y - 16, "stage 3", { size: 22, weight: 600, anchor: "middle", fill: "s2" });
          const { py } = frame(s, X, Y, W, H, { domain: [0, 1], ticks: [] }, { domain: [rates[0] / 1.4, rates.at(-1) * 1.4], log: true, ticks: rates, format: rate, title: "learning rate" });
          s.text(X - 12, Y + H + 32, "width", { size: 22, anchor: "end", fill: "muted" });
          s.text(X - 12, Y + H + 64, "parameters", { size: 22, anchor: "end", fill: "muted" });
          const columns = family.map((f, i) => {
            const g = s.group();
            g.text(cx(i), Y + H + 32, String(f.dim), { size: 22, anchor: "middle", fill: "ink2" });
            g.text(cx(i), Y + H + 64, millions(f.params), { size: 22, anchor: "middle", fill: "ink2" });
            for (const r of f.tried) g.circle(cx(i), py(r.lr), 9, { fill: "surface", stroke: "muted" });
            return g;
          });
          const line = s.path(
            family.map((f, i) => [cx(i), py(f.best.lr)]),
            { stroke: "s2", width: 3 },
          );
          const bests = family.map((f, i) => s.circle(cx(i), py(f.best.lr), 11, { fill: "s2" }));
          const note = s.text(640, 566, "Each size is trained briefly at rates that double, and the best is kept.", { size: 22, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(marked, during(t, 0.2, 0.6));
            columns.forEach((g, i) => show(g, during(t, 0.4 + i * 0.28, 0.7 + i * 0.28)));
            bests.forEach((dot, i) => show(dot, during(t, 3.0 + i * 0.12, 3.3 + i * 0.12)));
            trace(line, during(t, 3.4, 4.6));
            show(line, t > 3.4 ? 1 : 0);
            show(note, during(t, 4.4, 4.8));
          };
        },
      },
      {
        seconds: 7,
        caption: "Every size is trained at every budget it can reach, on as\nmany tokens as the budget buys, each with its own schedule.",
        draw(s) {
          const cx = (i) => 310 + i * 120;
          const rowY = (r) => 200 + r * 70;
          const title = s.text(70, 132, `${runs.length} runs, each on the tokens its budget buys, C / 6N`, { size: 24, fill: "ink2" });
          const head = s.group();
          head.text(70, 184, "budget", { size: 22, fill: "muted" });
          family.forEach((f, i) => head.text(cx(i), 184, millions(f.params), { size: 22, anchor: "middle", fill: "muted" }));
          keys.forEach((_, r) => family.forEach((__, i) => head.rect(cx(i) - 54, rowY(r), 108, 56, { fill: "tint", r: 8 })));
          const rows = keys.map((key, r) => {
            const label = s.text(70, rowY(r) + 38, budget(key), { size: 28, weight: 700 });
            const cells = valleys[key].runs.map((run) => {
              const i = family.findIndex((f) => f.dim === dimension(run.name));
              const g = s.group();
              const ground = g.rect(cx(i) - 54, rowY(r), 108, 56, { fill: "raised", stroke: "axis", r: 8, width: 1.5 });
              g.text(cx(i), rowY(r) + 37, tokens(run.tokens), { size: 26, anchor: "middle" });
              return { g, i, ground, twice: run.passes > 1 };
            });
            return { label, cells };
          });
          const twice = s.group();
          twice.rect(70, 494, 40, 28, { fill: "raised", stroke: "s2", r: 6, width: 3 });
          twice.text(126, 516, "reads part of the 131M tokens of TinyStories a second time", { size: 24, fill: "ink2" });
          const limits = s.text(70, 566, "No run reads the text more than 1.7 times, and none has under 8M tokens.", { size: 24, fill: "ink2" });
          return (t) => {
            show(title, during(t, 0.2, 0.6));
            show(head, during(t, 0.4, 0.8));
            rows.forEach((row, r) => {
              const from = 0.8 + r * 0.7;
              show(row.label, during(t, from, from + 0.3));
              row.cells.forEach((c, j) => {
                show(c.g, during(t, from + 0.1 + j * 0.08, from + 0.4 + j * 0.08));
                const lit = c.twice && t >= 4.0;
                c.ground.setAttribute("class", lit ? "f-raised s-s2" : "f-raised s-axis");
                c.ground.setAttribute("stroke-width", lit ? 3 : 1.5);
              });
            });
            show(twice, during(t, 4.0, 4.4));
            show(limits, during(t, 4.6, 5.0));
          };
        },
      },
      {
        seconds: 8,
        caption: "At every budget, the loss against size is a valley.\nIts floor is the best size for that budget.",
        draw(s) {
          const X = 150;
          const Y = 160;
          const W = 700;
          const H = 340;
          const across = { domain: [0.25e6, 40e6], log: true, ticks: [0.3e6, 1e6, 3e6, 10e6, 30e6], format: (v) => `${v / 1e6}M`, title: "parameters, on a log scale" };
          const up = { domain: [1.35, 2.45], ticks: [1.4, 1.6, 1.8, 2.0, 2.2, 2.4], format: (v) => v.toFixed(1), title: "loss at the end of the run" };
          const { px, py } = frame(s, X, Y, W, H, across, up);
          const shades = [0.4, 0.6, 0.8, 1];
          const floors = keys.map((key) => [px(valleys[key].params), py(valleys[key].loss)]);
          const joined = s.path(floors, { stroke: "s2", width: 2.5 });
          const drawn = keys.map((key, b) => {
            const sorted = valleys[key].runs.slice().sort((p, q) => p.params - q.params);
            const points = sorted.map((r) => [px(r.params), py(r.loss)]);
            const line = s.path(points, { stroke: "s1", width: 3 });
            line.setAttribute("opacity", shades[b]);
            const dots = points.map(([x, y]) => s.circle(x, y, 6.5, { fill: "s1" }));
            const [x, y] = points.at(-1);
            const label = s.text(x + 16, y + 8, `C = ${budget(key)}`, { size: 22, fill: "ink2" });
            return { line, dots, reached: along(points), label, from: 0.5 + b * 1.05, shade: shades[b] };
          });
          const marks = keys.map((key, b) => s.circle(floors[b][0], floors[b][1], 9, valleys[key].bracketed ? { fill: "s2" } : { fill: "surface", stroke: "s2" }));
          const sure = s.group();
          sure.circle(970, 236, 9, { fill: "s2" });
          sure.lines(992, 244, ["floor: the best size", "for its budget"], { size: 22, fill: "ink2" });
          const unsure = s.group();
          unsure.circle(970, 336, 9, { fill: "surface", stroke: "s2" });
          unsure.text(992, 344, "a floor less sure", { size: 22, fill: "ink2" });
          const loose = keys.some((key) => !valleys[key].bracketed);
          return (t) => {
            drawn.forEach((v) => {
              const part = during(t, v.from, v.from + 0.8);
              trace(v.line, part);
              show(v.line, t > v.from ? v.shade : 0);
              v.dots.forEach((dot, i) => show(dot, v.shade * during(t, v.from + 0.8 * v.reached[i] - 0.1, v.from + 0.8 * v.reached[i] + 0.1)));
              show(v.label, during(t, v.from + 0.6, v.from + 0.9));
            });
            marks.forEach((m, b) => show(m, during(t, drawn[b].from + 0.9, drawn[b].from + 1.2)));
            trace(joined, during(t, 5.0, 5.7));
            show(joined, t > 5.0 ? 1 : 0);
            show(sure, during(t, drawn[0].from + 0.9, drawn[0].from + 1.3));
            show(unsure, loose ? during(t, drawn.at(-1).from + 0.9, drawn.at(-1).from + 1.3) : 0);
          };
        },
      },
      {
        seconds: 6.5,
        caption: `The best split stays near Chinchilla's 20 tokens per\nparameter: ${ratios.slice(0, -1).map(per).join(", ")}, and ${per(ratios.at(-1))} at the largest budget.`,
        draw(s) {
          const BX = 660;
          const UNIT = 16; // pixels for a token per parameter
          const rowY = (r) => 206 + r * 72;
          const chinchilla = s.group();
          chinchilla.line(BX + 20 * UNIT, 156, BX + 20 * UNIT, rowY(keys.length - 1) + 50, { stroke: "s2", width: 2.5 });
          chinchilla.text(BX + 20 * UNIT, 144, "Chinchilla's 20", { size: 22, weight: 600, anchor: "middle", fill: "s2" });
          const head = s.group();
          head.text(70, 180, "budget", { size: 22, fill: "muted" });
          head.text(200, 180, "its floor", { size: 22, fill: "muted" });
          head.text(BX, 180, "tokens per parameter", { size: 22, fill: "muted" });
          const rows = keys.map((key, r) => {
            const v = valleys[key];
            const g = s.group();
            g.text(70, rowY(r) + 26, budget(key), { size: 28, weight: 700 });
            g.text(200, rowY(r) + 25, `${millions(v.params)} parameters on ${tokens(v.tokens)} tokens`, { size: 24, fill: "ink2" });
            const bar = s.rect(BX, rowY(r), 0, 34, { fill: "s1", r: 5 });
            const value = s.text(1110, rowY(r) + 27, "", { size: 28, weight: 700, anchor: "end" });
            return { g, bar, value, ratio: ratios[r], sure: v.bracketed };
          });
          const note = s.text(640, 530, `At ${budget(last)} the smallest run is also the lowest, so its floor is less sure.`, { size: 24, anchor: "middle", fill: "ink2" });
          return (t) => {
            show(head, during(t, 0.2, 0.5));
            show(chinchilla, during(t, 0.4, 0.8));
            rows.forEach((row, r) => {
              const from = 0.7 + r * 0.55;
              const u = during(t, from + 0.2, from + 0.8);
              show(row.g, during(t, from, from + 0.3));
              set(row.bar, { width: row.ratio * UNIT * u });
              show(row.value, during(t, from + 0.2, from + 0.5));
              row.value.textContent = String(Math.round(row.ratio * u));
              const open = !row.sure && t >= 3.7;
              row.bar.setAttribute("class", open ? "f-surface s-s1" : "f-s1");
              row.bar.setAttribute("stroke-width", open ? 3 : 0);
            });
            show(note, rows.some((row) => !row.sure) ? during(t, 3.7, 4.1) : 0);
          };
        },
      },
      {
        seconds: 7.5,
        caption: `One law, fitted on the three smaller budgets, predicts\nthe four runs at ${budget(last)} within ${worst.toFixed(3)}.`,
        draw(s) {
          const formula = s.text(640, 150, "", { size: 38, anchor: "middle", mono: true });
          const piece = (text, attributes = {}) => (svg("tspan", attributes, formula).textContent = text);
          piece("L(N, D) = E + A / N");
          piece("α", { dy: -14, "font-size": 26 });
          piece(" + B / D", { dy: 14 });
          piece("β", { dy: -14, "font-size": 26 });
          const constants = s.text(640, 204, `fitted on the ${fitted} runs below ${budget(last)}: E ${early.E.toFixed(2)}, α ${early.alpha.toFixed(2)}, β ${early.beta.toFixed(2)}`, { size: 24, anchor: "middle", fill: "ink2" });
          const chart = s.group();
          const X = 200;
          const Y = 280;
          const W = 600;
          const H = 210;
          const TOP = 1.53;
          const across = { domain: [6.6e6, 34e6], log: true, ticks: misses.map((p) => p.params), format: millions, title: `parameters, at ${budget(last)} operations` };
          const up = { domain: [1.425, TOP], ticks: [1.44, 1.46, 1.48, 1.5, 1.52], format: (v) => v.toFixed(2), title: "loss" };
          const { px, py } = frame(chart, X, Y, W, H, across, up);
          const C = Number(last);
          const curve = s.path(
            spread(...across.domain)
              .map((n) => [n, law(early, n, C / (6 * n))])
              .filter(([, l]) => l <= TOP)
              .map(([n, l]) => [px(n), py(l)]),
            { stroke: "s2", width: 3 },
          );
          const pairs = misses.map((p) => ({
            ring: s.circle(px(p.params), py(p.predicted), 8, { fill: "surface", stroke: "s2" }),
            gap: s.line(px(p.params), py(p.predicted), px(p.params), py(p.loss), { stroke: "muted", width: 2 }),
            dot: s.circle(px(p.params), py(p.loss), 8, { fill: "s1" }),
          }));
          const legend = s.group();
          legend.circle(880, 296, 8, { fill: "surface", stroke: "s2" });
          legend.text(900, 304, "predicted by the law", { size: 22, fill: "ink2" });
          const measured = s.group();
          measured.circle(880, 336, 8, { fill: "s1" });
          measured.text(900, 344, "measured", { size: 22, fill: "ink2" });
          const off = s.text(870, 404, high ? "the law is high by" : "the law misses by", { size: 22, fill: "muted" });
          const rows = misses.map((p, i) => {
            const g = s.group();
            const most = Math.abs(p.miss) === worst;
            g.text(870, 440 + i * 34, millions(p.params), { size: 22, fill: "muted" });
            g.text(1060, 440 + i * 34, Math.abs(p.miss).toFixed(3), { size: 24, weight: most ? 700 : 400, anchor: "end", fill: most ? "s2" : "ink" });
            return g;
          });
          return (t) => {
            show(formula, during(t, 0.2, 0.6));
            show(constants, during(t, 0.7, 1.1));
            show(chart, during(t, 1.0, 1.4));
            trace(curve, during(t, 1.4, 2.6));
            show(curve, t > 1.4 ? 1 : 0);
            show(legend, during(t, 1.6, 2.0));
            pairs.forEach((pair, i) => {
              show(pair.ring, during(t, 2.0 + i * 0.2, 2.3 + i * 0.2));
              // a measured loss arrives where the run ended, after the law had said where it would
              const from = 3.0 + i * 0.45;
              show(pair.dot, during(t, from, from + 0.3));
              show(pair.gap, during(t, from + 0.2, from + 0.4));
              show(rows[i], during(t, from + 0.2, from + 0.5));
            });
            show(measured, during(t, 3.0, 3.4));
            show(off, during(t, 3.0, 3.4));
          };
        },
      },
      {
        seconds: 8,
        caption: `Stage 3's run is ${stage3.params > best.params ? "larger" : "smaller"} than the law's choice at its compute.\nNear its floor the valley is flat: it costs ${cost.toFixed(3)} in loss.`,
        draw(s) {
          const X = 170;
          const Y = 170;
          const W = 580;
          const H = 320;
          const TOP = 1.56;
          const across = { domain: [2.8e6, 36e6], log: true, ticks: [3e6, 10e6, 30e6], format: (v) => `${v / 1e6}M`, title: `parameters, at stage 3's ${operations(stage3.flops)} operations` };
          const up = { domain: [1.44, TOP], ticks: [1.44, 1.46, 1.48, 1.5, 1.52, 1.54, 1.56], format: (v) => v.toFixed(2), title: "loss" };
          const chart = s.group();
          const { px, py } = frame(chart, X, Y, W, H, across, up);
          const at = (n) => law(summary.fit, n, stage3.flops / (6 * n));
          const curve = s.path(
            spread(...across.domain)
              .map((n) => [n, at(n)])
              .filter(([, l]) => l <= TOP)
              .map(([n, l]) => [px(n), py(l)]),
            { stroke: "s2", width: 3 },
          );
          const level = s.line(px(best.params), py(best.predicted), px(stage3.params) + 24, py(best.predicted), { stroke: "s3", width: 2 });
          const rise = s.line(px(stage3.params), py(best.predicted), px(stage3.params), py(stage3.predicted), { stroke: "s2", width: 3 });
          const chosen = s.circle(px(best.params), py(best.predicted), 9, { fill: "s3" });
          const ring = s.circle(0, 0, 9, { fill: "surface", stroke: "s2" });
          const ran = s.circle(px(stage3.params), py(stage3.loss), 9, { fill: "s1" });
          const block = (y, mark, title, detail) => {
            const g = s.group();
            g.circle(822, y - 9, 9, mark);
            g.text(844, y, title, { size: 26, weight: 600 });
            g.text(844, y + 34, detail, { size: 22, fill: "ink2" });
            return g;
          };
          const law_ = block(196, { fill: "s3" }, `the law's choice: ${best.predicted.toFixed(3)}`, `${millions(best.params)} parameters on ${tokens(best.tokens)} tokens`);
          const split = block(296, { fill: "surface", stroke: "s2" }, `stage 3's split: ${stage3.predicted.toFixed(3)}`, `${millions(stage3.params)} parameters on ${tokens(stage3.tokens)} tokens`);
          const real = block(396, { fill: "s1" }, `stage 3 measured: ${stage3.loss.toFixed(3)}`, tuned);
          const gap = s.group();
          gap.text(822, 510, cost.toFixed(3), { size: 40, weight: 700, fill: "s2" });
          gap.text(822, 546, `for ${(stage3.params / best.params).toFixed(1)} times the best size`, { size: 22, fill: "ink2" });
          return (t) => {
            show(chart, during(t, 0.2, 0.5));
            trace(curve, during(t, 0.4, 1.8));
            show(curve, t > 0.4 ? 1 : 0);
            show(chosen, during(t, 1.8, 2.1));
            show(law_, during(t, 1.8, 2.2));
            // from the law's choice along the valley to stage 3's size
            const n = Math.exp(mix(Math.log(best.params), Math.log(stage3.params), during(t, 2.5, 3.3)));
            set(ring, { cx: px(n), cy: py(at(n)) });
            show(ring, during(t, 2.3, 2.5));
            show(split, during(t, 3.2, 3.6));
            show(level, during(t, 3.6, 4.0));
            show(rise, during(t, 3.6, 4.0));
            show(gap, during(t, 3.8, 4.2));
            show(ran, during(t, 4.6, 4.9));
            show(real, during(t, 4.6, 5.0));
          };
        },
      },
    ],
  };
};
