// The side topic 2a as a film: a router sends each token to two of eight
// experts, the parameters grow with the experts and the compute with the two
// that run, and what that bought in the runs of runs/moe/, against the dense
// model of stage 3.

import { during, mix, place, set, show, spelled, trace } from "./stage.js";

export const sources = {
  meta: "/models/base-q4/meta.json",
  dense: "/runs/base/log.jsonl",
  mixture: "/runs/moe/moe.jsonl",
  unbalanced: "/runs/moe/no-balance.jsonl",
  wide: "/runs/moe/wide.jsonl",
  mixture20: "/runs/moe/moe-20min.jsonl",
  wide20: "/runs/moe/wide-20min.jsonl",
  routing: "/runs/moe/routing.json",
};

const percent = (v) => `${Math.round(100 * v)}%`;
const millions = (n) => `${(n / 1e6).toFixed(1)}M`;
const count = (n) => n.toLocaleString("en");
const loss = (v) => v.toFixed(3);

export default ({ meta, dense, mixture, unbalanced, wide, mixture20, wide20, routing }) => {
  const { dim, n_layers: layers, mlp_dim: hidden } = meta.config;
  const { n_experts: experts, n_active: active } = routing;

  // The sizes of the three models. The two experts that run are half as wide
  // as the MLP they replace, and the wide model has all the experts' width.
  const narrow = hidden / active;
  const mlp = (width) => 3 * dim * width;
  const router = dim * experts;
  const models = [
    { name: "dense", held: meta.parameters, used: meta.parameters, fill: "ink2", note: `one MLP of width ${count(hidden)} in every block` },
    {
      name: "mixture",
      held: meta.parameters + layers * (experts * mlp(narrow) + router - mlp(hidden)),
      used: meta.parameters + layers * (active * mlp(narrow) + router - mlp(hidden)),
      fill: "s2",
      note: `${experts} experts of width ${count(narrow)} in every block, of which ${active} run`,
    },
    { name: "dense, wide", held: meta.parameters + layers * (mlp(experts * narrow) - mlp(hidden)), fill: "s1", note: `one MLP of width ${count(experts * narrow)} in every block` },
  ];
  models[2].used = models[2].held;

  const evaluations = (log) => log.filter((e) => "val_loss" in e);
  const end = (log) => evaluations(log).at(-1);
  const steps = end(dense).step;
  const runs = [dense, mixture, wide].map((log, i) => ({ ...models[i], log, end: end(log) }));
  const timed = [dense, mixture20, wide20].map((log, i) => ({ ...models[i], end: end(log) }));
  const minutes = (e) => e.elapsed / 60;

  // every expert's share of its block's work, at every step that was logged
  const shares = (log) => log.filter((e) => e.expert_share);
  const idle = (e) => e.expert_share.flat().filter((v) => v < 0.01).length;
  const sides = [
    { title: "with the balance loss", log: shares(mixture), x: 200 },
    { title: "without it", log: shares(unbalanced), x: 750 },
  ];

  // what the experts of the last block get, on text the mixture has not seen
  const block = routing.blocks.at(-1);
  const gets = (token) => block.favorites.findIndex((list) => list.some((f) => f.token === token));
  const named = [gets(" their"), gets(" have"), gets('?"')]; // the possessives, the auxiliary verbs, the ends of speech

  return {
    subtitle: "More parameters than a token uses",
    scenes: [
      {
        seconds: 8.5,
        caption: `A mixture puts ${experts} experts where the one MLP was.\nFor each token a router picks the ${active} that run.`,
        draw(s) {
          const MID = 342;
          // which experts three tokens are sent to: a picture of the rule, not a measurement
          const picks = [[1, 4], [0, 6], [2, 7]];
          const starts = [1.4, 3.6, 5.2];
          const paces = [2, 1.4, 1.4];
          const tokens = picks.map((_, i) => s.chip(70, 258 + i * 60, "token", { size: 28 }));
          const into = tokens.map((c) => ({ x1: c.x + c.w + 6, y1: c.y + c.h / 2, x2: 294, y2: MID + (c.y + c.h / 2 - MID) / 4 }));
          const wires = into.map((a) => s.arrow(a.x1, a.y1, a.x2, a.y2));
          const box = s.box(300, 302, 150, 80, "router", "", { size: 28 });
          const rule = s.lines(375, 420, ["a score for every", `expert: the top ${active} run`], { size: 22, anchor: "middle", fill: "muted" });
          const row = Array.from({ length: experts }, (_, i) => {
            const y = 108 + i * 60;
            const g = s.group();
            const e = { g, y: y + 23, out: MID + (i - (experts - 1) / 2) * 6, in: MID + (i - (experts - 1) / 2) * 4 };
            e.from = g.arrow(452, e.out, 654, e.y);
            e.ground = g.rect(660, y, 240, 46, { fill: "raised", stroke: "axis", r: 10 });
            e.label = g.text(780, y + 31, `expert ${i + 1}`, { size: 24, weight: 600, anchor: "middle" });
            e.to = s.arrow(906, e.y, 1018, e.in, { stroke: "s2" });
            return e;
          });
          const sum = s.group();
          sum.circle(1042, MID, 22, { fill: "surface", stroke: "axis" });
          sum.text(1042, MID + 10, "+", { size: 30, anchor: "middle", fill: "ink2" });
          sum.arrow(1066, MID, 1200, MID);
          sum.lines(1134, 384, ["their", "weighted sum"], { size: 22, anchor: "middle", fill: "muted" });
          const dots = [0, 1, 2, 3].map(() => s.circle(0, 0, 8, { fill: "s2" }));
          const at = (dot, x1, y1, x2, y2, u) => set(dot, { cx: mix(x1, x2, u), cy: mix(y1, y2, u) });
          return (t) => {
            tokens.forEach((c, i) => show(c, during(t, 0.2 + i * 0.1, 0.5 + i * 0.1)));
            wires.forEach((w, i) => show(w, during(t, 0.4 + i * 0.1, 0.7 + i * 0.1)));
            show(box, during(t, 0.5, 0.9));
            rule.forEach((line) => show(line, during(t, 0.9, 1.3)));
            show(sum, during(t, 1, 1.4));
            let k = -1;
            starts.forEach((start, i) => (k = t >= start ? i : k));
            const u = k < 0 ? 0 : (t - starts[k]) / paces[k]; // how far the token on its way has come
            const chosen = k < 0 ? [] : picks[k];
            // the choice holds until the next token leaves
            const held = during(u, 0.4, 0.5) * (k >= 0 && k + 1 < starts.length ? 1 - during(t, starts[k + 1] - 0.25, starts[k + 1]) : 1);
            tokens.forEach((c, i) => {
              c.ground.setAttribute("class", i === k ? "f-s2 s-s2" : "f-tint s-axis");
              c.label.setAttribute("class", `film-mono ${i === k ? "f-surface" : "f-ink"}`);
            });
            row.forEach((e, i) => {
              const runs = chosen.includes(i);
              show(e.g, during(t, 0.6 + i * 0.07, 1 + i * 0.07) * (runs ? 1 : mix(1, 0.3, held)));
              e.ground.setAttribute("class", runs && held > 0.5 ? "f-s2 s-s2" : "f-raised s-axis");
              e.label.setAttribute("class", runs && held > 0.5 ? "f-surface" : "f-ink");
              show(e.to, runs ? Math.min(held, during(u, 0.55, 0.65)) : 0);
            });
            const [a, b] = chosen.length ? chosen.map((i) => row[i]) : [row[0], row[0]];
            const arrow = into[Math.max(0, k)];
            const leaving = during(u, 0, 0.2);
            const routed = during(u, 0.25, 0.5);
            const summed = during(u, 0.6, 0.85);
            const out = during(u, 0.85, 1);
            at(dots[0], arrow.x1, arrow.y1, arrow.x2, arrow.y2, leaving);
            show(dots[0], k >= 0 && u < 0.25 ? 1 : 0);
            [a, b].forEach((e, i) => {
              if (summed > 0) at(dots[1 + i], 906, e.y, 1020, e.in, summed);
              else at(dots[1 + i], 452, e.out, 654, e.y, routed);
              show(dots[1 + i], k >= 0 && ((u >= 0.25 && u < 0.5) || (summed > 0 && summed < 1)) ? 1 : 0);
            });
            at(dots[3], 1066, MID, 1190, MID, out);
            show(dots[3], out > 0 && out < 1 ? 1 : 0);
          };
        },
      },
      {
        seconds: 7,
        caption: "Parameters grow with the number of experts.\nThe compute per token grows only with the number that run.",
        draw(s) {
          const X = 330;
          const scale = 700 / Math.max(...models.map((m) => m.held));
          s.rect(X, 112, 34, 24, { fill: "tint", stroke: "axis", r: 5 });
          s.text(X + 46, 132, "parameters held", { size: 22, fill: "ink2" });
          s.rect(X + 290, 112, 34, 24, { fill: "ink2", r: 5 });
          s.text(X + 336, 132, "used for one token", { size: 22, fill: "ink2" });
          const rows = models.map((m, i) => {
            const y = 176 + i * 130;
            const g = s.group();
            g.text(X - 24, y + 35, m.name, { size: 26, anchor: "end", weight: m.fill === "s2" ? 700 : 400 });
            g.text(X, y + 84, m.note, { size: 22, fill: "muted" });
            return {
              g,
              m,
              from: [0.3, 1.7, 4.1][i],
              held: s.rect(X, y, 0, 52, { fill: "tint", stroke: "axis", r: 8 }),
              used: s.rect(X, y, 0, 52, { fill: m.fill, r: 8 }),
              inside: s.text(X + 14, y + 35, millions(m.used), { size: 24, weight: 600, fill: "surface" }),
              after: s.text(X + scale * m.held + 14, y + 35, millions(m.held), { size: 26, weight: 600 }),
            };
          });
          return (t) => {
            for (const r of rows) {
              show(r.g, during(t, r.from, r.from + 0.4));
              // the mixture holds its parameters first, and then uses a part of them
              const held = during(t, r.from + 0.2, r.from + (r.m.fill === "s2" ? 1.3 : 1));
              const used = r.m.fill === "s2" ? during(t, r.from + 1.4, r.from + 2) : held;
              set(r.held, { width: scale * r.m.held * held });
              set(r.used, { width: scale * r.m.used * used });
              show(r.held, held > 0 ? 1 : 0);
              show(r.used, used > 0 ? 1 : 0);
              show(r.inside, during(used, 0.8, 1));
              show(r.after, during(held, 0.8, 1));
            }
          };
        },
      },
      {
        seconds: 9,
        caption: `Nothing in the loss asks a router to use all its experts.\nWithout a balance loss, ${idle(sides[1].log.at(-1))} of ${layers * experts} end below 1% of the work.`,
        draw(s) {
          const TOP = 160;
          const PITCH = 42;
          for (const side of sides) {
            s.text(side.x + (experts * PITCH - 6) / 2, 132, side.title, { size: 26, weight: 600, anchor: "middle" });
            s.text(side.x - 16, 132, "block", { size: 22, anchor: "end", fill: "muted" });
            side.cells = Array.from({ length: layers }, (_, b) => {
              s.text(side.x - 16, TOP + b * PITCH + 26, String(b + 1), { size: 22, anchor: "end", fill: "muted" });
              return Array.from({ length: experts }, (_, e) => ({
                ground: s.rect(side.x + e * PITCH, TOP + b * PITCH, 36, 36, { fill: "raised", stroke: "axis", r: 6, width: 1.5 }),
                share: s.rect(side.x + e * PITCH, TOP + b * PITCH, 36, 36, { fill: "s1", r: 6 }),
              }));
            });
            side.count = s.text(side.x + (experts * PITCH - 6) / 2, 530, "", { size: 24, anchor: "middle", fill: "ink2" });
          }
          s.text(628, 308, "step", { size: 22, anchor: "middle", fill: "muted" });
          const step = s.text(628, 346, "", { size: 30, weight: 600, anchor: "middle" });
          s.text(640, 570, "a row is a block, a square an expert, shaded by its share of the block's work", { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            // the first steps slowly: that is when the routers pick their favorites
            const far = during(t, 1.2, 6.8) ** 2;
            for (const side of sides) {
              const entry = side.log[Math.round((side.log.length - 1) * far)];
              step.textContent = count(entry.step);
              side.cells.forEach((cells, b) =>
                cells.forEach((cell, e) => {
                  const share = entry.expert_share[b][e];
                  show(cell.share, share < 0.01 ? 0 : Math.sqrt(Math.min(1, 2 * share)));
                  cell.ground.setAttribute("class", share < 0.01 ? "f-raised s-s2" : "f-raised s-axis");
                  cell.ground.setAttribute("stroke-width", share < 0.01 ? 3 : 1.5);
                }),
              );
              side.count.textContent = `${idle(entry)} of ${layers * experts} experts below 1%`;
            }
          };
        },
      },
      {
        seconds: 8,
        caption: `Over the same ${count(steps)} steps the extra parameters help:\nthe mixture ends at ${loss(runs[1].end.val_loss)}, the dense model at ${loss(runs[0].end.val_loss)}.`,
        draw(s) {
          const first = steps / 4; // the chart starts where the three curves have come apart
          const curves = runs.map((r) => ({ ...r, points: evaluations(r.log).filter((e) => e.step >= first) }));
          const values = curves.flatMap((c) => c.points.map((e) => e.val_loss));
          const domain = [Math.floor(Math.min(...values) * 20) / 20, Math.ceil(Math.max(...values) * 20) / 20];
          const X = 150;
          const TOP = 140;
          const WIDE = 730;
          const TALL = 360;
          const { px, py } = s.axes(X, TOP, WIDE, TALL, { x: { domain: [first, steps] }, y: { domain } });
          for (let v = Math.ceil(domain[0] * 10 + 0.01) / 10; v < domain[1]; v += 0.1) {
            s.line(X, py(v), X + WIDE, py(v), { stroke: "grid", width: 1.5 });
            s.text(X - 14, py(v) + 8, v.toFixed(1), { size: 22, anchor: "end", fill: "muted" });
          }
          for (const at of [steps / 3, (2 * steps) / 3, steps]) s.text(px(at), TOP + TALL + 32, count(at), { size: 22, anchor: "middle", fill: "muted" });
          s.text(X, TOP - 18, "validation loss", { size: 22, fill: "muted" });
          s.text(X + WIDE / 2, TOP + TALL + 66, "step", { size: 22, anchor: "middle", fill: "muted" });
          curves.forEach((c, i) => {
            s.line(470, 172 + i * 36, 510, 172 + i * 36, { stroke: c.fill, width: 4 });
            s.text(524, 180 + i * 36, `${c.name}: ${millions(c.held)}${c.used < c.held ? `, ${millions(c.used)} for a token` : ""}`, { size: 22, fill: "ink2" });
            c.path = s.path(c.points.map((e) => [px(e.step), py(e.val_loss)]), { stroke: c.fill, width: c.fill === "s2" ? 4.5 : 3.5 });
            c.dot = s.circle(px(steps), py(c.end.val_loss), 6, { fill: c.fill });
            c.value = s.text(px(steps) + 18, py(c.end.val_loss) + 8, loss(c.end.val_loss), { size: 24, weight: 700, fill: c.fill });
            c.who = s.text(px(steps) + 96, py(c.end.val_loss) + 8, c.name, { size: 24, fill: "ink2" });
          });
          return (t) => {
            curves.forEach((c, i) => {
              trace(c.path, during(t, 0.6, 4));
              show(c.path, t > 0.6 ? 1 : 0);
              show(c.dot, during(t, 3.9, 4.1));
              show(c.value, during(t, 4.1 + i * 0.4, 4.5 + i * 0.4));
              show(c.who, during(t, 4.1 + i * 0.4, 4.5 + i * 0.4));
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `Given the same ${Math.round(minutes(timed[0].end))} minutes, the mixture takes ${count(timed[1].end.step)} steps\nand ends at ${loss(timed[1].end.val_loss)}, behind the dense model's ${loss(timed[0].end.val_loss)}.`,
        draw(s) {
          const X = 330;
          const best = Math.min(...timed.map((r) => r.end.val_loss));
          s.text(X, 176, "steps", { size: 22, fill: "muted" });
          s.text(1010, 176, "minutes", { size: 22, anchor: "end", fill: "muted" });
          s.text(1190, 176, "validation loss", { size: 22, anchor: "end", fill: "muted" });
          const rows = timed.map((r, i) => {
            const y = 212 + i * 118;
            s.text(X - 24, y + 36, r.name, { size: 26, anchor: "end", weight: r.fill === "s2" ? 700 : 400 });
            return {
              r,
              w: (520 * r.end.step) / steps,
              bar: s.rect(X, y, 0, 54, { fill: r.fill, r: 8 }),
              steps: s.text(X, y + 36, "", { size: 24, fill: "ink2" }),
              minutes: s.text(1010, y + 36, "", { size: 24, anchor: "end", fill: "ink2" }),
              loss: s.text(1190, y + 37, loss(r.end.val_loss), { size: 30, weight: 700, anchor: "end", fill: r.end.val_loss === best ? "s3" : "ink" }),
            };
          });
          return (t) => {
            // the three run side by side for the same time
            const u = during(t, 0.6, 3.6);
            rows.forEach((row, i) => {
              set(row.bar, { width: row.w * u });
              set(row.steps, { x: X + row.w * u + 14 });
              row.steps.textContent = count(Math.round((row.r.end.step * u) / 50) * 50);
              row.minutes.textContent = (minutes(row.r.end) * u).toFixed(1);
              show(row.loss, during(t, 3.9 + i * 0.3, 4.4 + i * 0.3));
            });
          };
        },
      },
      {
        seconds: 8,
        caption: `In block ${layers} the experts sort tokens by what they do:\npossessives, auxiliary verbs, the ends of speech.`,
        draw(s) {
          const peak = Math.max(...block.share);
          s.text(70, 128, `block ${layers}`, { size: 26, weight: 600 });
          s.text(200, 128, "share of the work", { size: 22, fill: "muted" });
          s.text(410, 128, `the tokens it gets most reliably, in ${count(routing.tokens)} tokens of unseen stories`, { size: 22, fill: "muted" });
          const rows = block.share.map((share, e) => {
            const y = 148 + e * 54;
            const g = s.group();
            const row = { g, w: (130 * share) / peak, name: g.text(70, y + 27, `expert ${e + 1}`, { size: 24 }), bar: g.rect(200, y + 6, 0, 26, { fill: "s1", r: 5 }), chips: [] };
            g.text(344, y + 27, percent(share), { size: 22, fill: "ink2" });
            let at = 410;
            for (const f of block.favorites[e].slice(0, 7)) {
              const chip = g.chip(at, y, spelled(f.token), { size: 22, pad: 8 });
              if (at + chip.w > 1216) {
                chip.g.remove();
                break;
              }
              row.chips.push(chip);
              at += chip.w + 6;
            }
            return row;
          });
          return (t) => {
            rows.forEach((row, e) => {
              show(row.g, during(t, 0.3 + e * 0.15, 0.7 + e * 0.15));
              set(row.bar, { width: row.w * during(t, 0.5 + e * 0.15, 1.3 + e * 0.15) });
              const turn = named.indexOf(e);
              const marked = turn >= 0 && t >= 3 + turn * 1.1;
              row.name.setAttribute("class", marked ? "f-s2" : "f-ink");
              row.name.setAttribute("font-weight", marked ? 700 : 400);
              for (const chip of row.chips) chip.ground.setAttribute("class", marked ? "f-tint s-s2" : "f-tint s-axis");
            });
          };
        },
      },
    ],
  };
};
