// Stage 8 as a film: comparisons of two replies, a reward model that learns
// them, and three ways to use what was learned: keep the best of several
// stories, train on the reward with a penalty for drifting, or train on the
// comparisons directly. The numbers are those of the logged runs in runs/rlhf/.

import { during, mix, show, trace } from "./stage.js";

export const sources = {
  preferences: "/runs/rlhf/preferences.json",
  reward: "/runs/rlhf/reward.json",
  best: "/runs/rlhf/best_of_n.json",
  ppo: "/runs/rlhf/ppo.jsonl",
  free: "/runs/rlhf/ppo-no-kl.jsonl",
  run: "/runs/rlhf/ppo.json",
  dpo: "/runs/rlhf/dpo.json",
  stories: "/runs/rlhf/stories.json",
  asked: "/runs/eval/taste-sft-words.json",
};

const count = (v) => Math.round(v).toLocaleString("en");
const share = (v) => `${(100 * v).toFixed(0)}%`;
const signed = (v, digits = 2) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(digits)}`;
const plain = (v) => String(v).replace("-", "−");

// The three ways to use what the annotator said, with the one a scene is about marked.
function ways(s, now) {
  ["1  keep the best of n", "2  PPO, on the reward", "3  DPO, on the comparisons"].forEach((name, i) => {
    const x = [250, 640, 1030][i];
    s.text(x, 128, name, { size: 24, weight: i === now ? 700 : 400, anchor: "middle", fill: i === now ? "ink" : "muted" });
    if (i === now) s.line(x - 130, 142, x + 130, 142, { stroke: "s2", width: 4 });
  });
}

// The frame of a chart with its ticks: where a value is on the stage.
function chart(s, { x, y, w, h, across, up, ticks, steps, format = plain, title }) {
  const px = (v) => x + (w * (v - across[0])) / (across[1] - across[0]);
  const py = (v) => y + h * (1 - (v - up[0]) / (up[1] - up[0]));
  for (const tick of ticks) {
    s.line(x, py(tick), x + w, py(tick), { stroke: "grid", width: 1.5 });
    s.text(x - 12, py(tick) + 8, format(tick), { size: 22, anchor: "end", fill: "muted" });
  }
  s.line(x, y + h, x + w, y + h, { stroke: "axis" });
  for (const step of steps) s.text(px(step), y + h + 30, plain(step), { size: 22, anchor: "middle", fill: "muted" });
  s.text(x, y - 16, title, { size: 22, fill: "ink2" });
  return { px, py };
}

export default ({ preferences, reward, best, ppo, free, run, dpo, stories, asked }) => {
  const pairs = preferences.comparisons - preferences.held_out;
  const last = (log, key) => log.at(-1)[key];
  const most = best.n.at(-1); // the largest number of stories that was tried
  const bestOfMost = best.picked_by_reward_model.at(-1);
  const models = [
    { key: "sft-words", name: "fine-tuned, the start", color: "muted" },
    { key: "ppo", name: "PPO", color: "s1" },
    { key: "ppo-no-kl", name: "PPO without the penalty", color: "s2" },
    { key: "dpo", name: "DPO", color: "s1" },
  ];
  const loose = stories["ppo-no-kl"];
  const highest = models.every((m) => stories[m.key].taste <= loose.taste);

  return {
    subtitle: "From comparisons of two replies to a better model",
    scenes: [
      {
        seconds: 7,
        caption: "For much of what is wanted there is no right answer to write\ndown. Shown two replies, a person can say which is better.",
        draw(s) {
          const prompt = s.chips(640, 108, [asked.samples[0].prompt], { center: true, size: 24 })[0];
          // two stories, as the lines of their text
          const cards = [[392, 410, 350, 398, 236], [400, 366, 408, 174]].map((lines, i) => {
            const x = 170 + i * 500;
            const g = s.group();
            g.arrow(640 + (i ? 60 : -60), 154, x + 220 + (i ? -70 : 70), 178);
            const ground = g.rect(x, 184, 440, 174, { fill: "raised", stroke: "axis" });
            g.text(x + 20, 220, `story ${"ab"[i]}`, { size: 24, weight: 600, fill: "ink2" });
            lines.forEach((w, k) => g.rect(x + 20, 240 + k * 22, w, 11, { fill: "grid", r: 5 }));
            return { g, ground };
          });
          const chosen = s.text(170 + 500 + 220, 396, "✓ preferred", { size: 26, weight: 700, anchor: "middle", fill: "s2" });
          const total = s.text(640, 478, "", { size: 36, weight: 700, anchor: "middle" });
          const who = s.lines(640, 520, ["by a program that plays the annotator. It has a hidden taste,", `and picks the story it likes more in ${share(preferences.annotator_consistency)} of the comparisons.`], { size: 24, anchor: "middle", fill: "ink2", leading: 1.4 });
          return (t) => {
            show(prompt, during(t, 0.3, 0.8));
            cards.forEach((card, i) => show(card.g, during(t, 0.9 + i * 0.3, 1.5 + i * 0.3)));
            cards[1].ground.setAttribute("class", t >= 2.6 ? "f-raised s-s2" : "f-raised s-axis");
            cards[1].ground.setAttribute("stroke-width", t >= 2.6 ? 4 : 2);
            show(chosen, during(t, 2.6, 3.1));
            show(total, during(t, 3.6, 4));
            total.textContent = `${count(preferences.comparisons * during(t, 3.6, 4.8))} comparisons`;
            who.forEach((line, i) => show(line, during(t, 4.6 + i * 0.2, 5.1 + i * 0.2)));
          };
        },
      },
      {
        seconds: 7,
        caption: "A reward model learns from the comparisons to give any story\na score. The taste itself it was never shown.",
        draw(s) {
          // what the reward model is
          const parts = [s.box(120, 124, 320, 56, "a prompt and a story", "", { size: 24 }), s.arrow(280, 186, 280, 214), s.box(120, 220, 320, 64, "reward model", "", { size: 26, stroke: "s2" }), s.arrow(280, 290, 280, 318)];
          const out = s.group();
          out.chips(280, 324, [signed(reward.held_out_rewards[0])], { center: true, size: 26 });
          out.text(360, 356, "its reward", { size: 22, fill: "muted" });
          const found = s.lines(90, 444, [`After ${count(pairs)} comparisons it agrees, on`, "ones it has not seen, with the annotator", `in ${share(reward.agrees_with_annotator)} and with their taste in ${share(reward.agrees_with_taste)}.`], { size: 24, leading: 1.4 });
          // every held-out story: the reward it gets against the taste that training never saw
          const { px, py } = chart(s, { x: 660, y: 150, w: 500, h: 350, across: [-2.6, 1.6], up: [-3.2, 3.2], ticks: [-2, 0, 2], steps: [-2, -1, 0, 1], title: "reward, for held-out stories" });
          s.text(910, 566, "the annotator's hidden taste", { size: 22, anchor: "middle", fill: "muted" });
          const groups = Array.from({ length: 20 }, () => s.group());
          reward.held_out_tastes.forEach((taste, i) => groups[i % groups.length].circle(px(taste), py(reward.held_out_rewards[i]), 4, { fill: "s1" }).setAttribute("opacity", 0.4));
          return (t) => {
            parts.forEach((part, i) => show(part, during(t, 0.3 + i * 0.25, 0.7 + i * 0.25)));
            show(out, during(t, 1.3, 1.7));
            groups.forEach((g, i) => show(g, during(t, 2 + i * 0.1, 2.3 + i * 0.1)));
            found.forEach((line, i) => show(line, during(t, 4.4 + i * 0.2, 4.9 + i * 0.2)));
          };
        },
      },
      {
        seconds: 6,
        caption: "The simplest use trains nothing: write several stories,\nscore them, keep the best. Every request pays for all of them.",
        draw(s) {
          ways(s, 0);
          const [floor, unit, wide] = [486, 165, 84];
          s.text(90, 190, "mean taste of the story kept", { size: 22, fill: "ink2" });
          s.rect(486, 172, 22, 22, { fill: "s2", r: 5 });
          s.text(518, 190, "picked by the reward model", { size: 22, fill: "ink2" });
          s.line(850, 183, 876, 183, { stroke: "s3", width: 5 });
          s.text(888, 190, "picked by the annotator", { size: 22, fill: "ink2" });
          s.line(230, floor, 1050, floor, { stroke: "axis" });
          s.text(640, floor + 72, "stories written for each request", { size: 22, anchor: "middle", fill: "muted" });
          const bars = best.n.map((n, i) => {
            const x = 300 + i * 160 - wide / 2;
            const [mine, theirs] = [best.picked_by_reward_model[i] * unit, best.picked_by_taste[i] * unit];
            s.text(x + wide / 2, floor + 34, String(n), { size: 24, anchor: "middle", weight: 600 });
            return {
              mine,
              bar: s.rect(x, floor, wide, 0, { fill: "s2", r: 6 }),
              value: s.text(x + wide / 2, floor - mine + 30, signed(best.picked_by_reward_model[i]), { size: 24, weight: 700, anchor: "middle", fill: "surface" }),
              mark: s.line(x - 8, floor - theirs, x + wide + 8, floor - theirs, { stroke: "s3", width: 5 }),
              ideal: s.text(x + wide / 2, floor - theirs - 12, signed(best.picked_by_taste[i]), { size: 22, anchor: "middle", fill: "s3", weight: 600 }),
            };
          });
          return (t) => {
            bars.forEach((b, i) => {
              const u = during(t, 0.4 + i * 0.4, 1.2 + i * 0.4);
              b.bar.setAttribute("height", (b.mine * u).toFixed(1));
              b.bar.setAttribute("y", (floor - b.mine * u).toFixed(1));
              show(b.value, during(t, 1 + i * 0.4, 1.3 + i * 0.4));
              show(b.mark, during(t, 3.2 + i * 0.1, 3.6 + i * 0.1));
              show(b.ideal, during(t, 3.2 + i * 0.1, 3.6 + i * 0.1));
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `PPO trains the model on the reward. The taste, which training\nnever sees, climbs with it${stories.ppo.taste > bestOfMost ? `, past the best of ${most}` : ""}.`,
        draw(s) {
          ways(s, 1);
          const iterations = ppo.length;
          const panels = [
            { key: "reward", title: "reward from the reward model", x: 130, up: [-0.5, 3], ticks: [0, 1, 2, 3] },
            { key: "taste", title: "the annotator's taste", x: 720, up: [0, 1.5], ticks: [0, 0.5, 1, 1.5] },
          ].map((panel) => {
            const { px, py } = chart(s, { ...panel, y: 216, w: 380, h: 270, across: [1, iterations], steps: [1, 20, 40, 60] });
            s.text(panel.x + 190, 552, "iteration of PPO", { size: 22, anchor: "middle", fill: "muted" });
            const line = s.path(ppo.map((row) => [px(row.iteration), py(row[panel.key])]), { stroke: "s1", width: 4 });
            return {
              line,
              px,
              py,
              from: s.text(px(1) + 12, py(ppo[0][panel.key]) - 10, signed(ppo[0][panel.key]), { size: 24, weight: 700 }),
              to: s.text(px(iterations) + 14, py(last(ppo, panel.key)) + 8, signed(last(ppo, panel.key)), { size: 24, weight: 700, fill: "s1" }),
            };
          });
          // what keeping the best of the most stories gave, as a line to cross
          const { px, py } = panels[1];
          const bar = s.group();
          bar.line(px(1), py(bestOfMost), px(iterations), py(bestOfMost), { stroke: "ink2", width: 2 });
          bar.text(px(1) + 8, py(bestOfMost) - 10, `best of ${most}: ${signed(bestOfMost)}`, { size: 22, fill: "ink2" });
          return (t) => {
            for (const panel of panels) {
              trace(panel.line, during(t, 1, 4));
              show(panel.from, during(t, 0.5, 1));
              show(panel.to, during(t, 4, 4.5));
            }
            show(bar, during(t, 4.4, 5));
          };
        },
      },
      {
        seconds: 7,
        caption: "A penalty on the KL divergence holds the model near its start.\nWithout it the model drifts and drops the instruction.",
        draw(s) {
          ways(s, 1);
          const iterations = ppo.length;
          const rule = s.text(640, 186, `loss = policy loss + ${run.args.kl} · KL(model ‖ model at the start)`, { size: 24, anchor: "middle", mono: true });
          const sides = [
            { log: ppo, color: "s1", name: "with the penalty" },
            { log: free, color: "s2", name: "without it" },
          ];
          const panels = [
            { key: "kl", title: "KL divergence from the starting model", x: 130, up: [0, 1.5], ticks: [0, 0.5, 1, 1.5], digits: (v) => v.toFixed(2) },
            { key: "words", title: "requested words used", x: 770, up: [0, 0.3], ticks: [0, 0.1, 0.2, 0.3], format: share, digits: share },
          ].map((panel) => {
            const { px, py } = chart(s, { ...panel, y: 248, w: 340, h: 240, across: [1, iterations], steps: [1, 20, 40, 60] });
            s.text(panel.x + 170, 554, "iteration of PPO", { size: 22, anchor: "middle", fill: "muted" });
            return sides.map((side) => {
              const [x, y] = [px(iterations) + 14, py(last(side.log, panel.key)) + 8];
              return {
                line: s.path(side.log.map((row) => [px(row.iteration), py(row[panel.key])]), { stroke: side.color, width: 4 }),
                end: s.text(x, y, panel.digits(last(side.log, panel.key)), { size: 24, weight: 700, fill: side.color }),
                name: s.text(x, y + 28, panel.key === "kl" ? side.name : "", { size: 22, fill: side.color }),
              };
            });
          });
          return (t) => {
            show(rule, during(t, 0.3, 0.9));
            panels.forEach((lines) =>
              lines.forEach((side, i) => {
                const [from, to] = [1.2 + i * 1.8, 2.8 + i * 1.8];
                trace(side.line, during(t, from, to));
                show(side.end, during(t, to, to + 0.4));
                show(side.name, during(t, to, to + 0.4));
              }),
            );
          };
        },
      },
      {
        seconds: 6,
        caption: `DPO skips the reward model and the sampling. It trains on\nthe comparisons themselves, and one pass takes ${dpo.minutes.toFixed(1)} minutes.`,
        draw(s) {
          ways(s, 2);
          const [w, h] = [220, 60];
          const row = (y, names, places, mark) =>
            names.map((name, i) => {
              const g = s.group();
              g.box(places[i], y, w, h, name, "", { size: 24, stroke: name === mark ? "s2" : "axis" });
              if (i) g.arrow(places[i - 1] + w + 10, y + h / 2, places[i] - 10, y + h / 2);
              return g;
            });
          const long = row(196, ["comparisons", "reward model", "PPO", "a better model"], [90, 383, 676, 970], "");
          const short = row(352, ["comparisons", "DPO", "a better model"], [90, 530, 970], "DPO");
          const times = [s.text(676 + w / 2, 296, `${run.minutes.toFixed(1)} minutes`, { size: 22, anchor: "middle", fill: "muted" }), s.text(530 + w / 2, 452, `${dpo.minutes.toFixed(1)} minutes`, { size: 24, weight: 700, anchor: "middle", fill: "s2" })];
          const rule = s.lines(640, 516, ["The preferred reply must gain more probability than the other,", "both measured against the model at the start."], { size: 24, anchor: "middle", fill: "ink2", leading: 1.4 });
          return (t) => {
            const fade = mix(1, 0.6, during(t, 1.8, 2.4));
            long.forEach((g, i) => show(g, during(t, 0.3 + i * 0.25, 0.7 + i * 0.25) * fade));
            short.forEach((g, i) => show(g, during(t, 1.9 + i * 0.3, 2.4 + i * 0.3)));
            show(times[0], during(t, 1.2, 1.6) * fade);
            show(times[1], during(t, 3, 3.5));
            rule.forEach((line, i) => show(line, during(t, 3.4 + i * 0.2, 3.9 + i * 0.2)));
          };
        },
      },
      {
        seconds: 8,
        caption: `Without the penalty the taste is ${highest ? "highest" : "high"} and the instruction\nis gone. The penalty protects what the reward leaves out.`,
        draw(s) {
          const measures = [
            { title: "taste", x: 400, value: (m) => m.taste, format: (v) => signed(v) },
            { title: "requested words used", x: 660, value: (m) => m.words_used, format: share },
            { title: "strangeness", x: 960, value: (m) => m.strangeness, format: (v) => v.toFixed(2) },
          ];
          const wide = 150;
          const rows = models.map((model, i) => s.text(90, 216 + i * 62, model.name, { size: 24, weight: model.key === "ppo-no-kl" ? 700 : 400 }));
          const panels = measures.map((measure) => {
            const title = s.text(measure.x, 152, measure.title, { size: 24, weight: 600 });
            const peak = Math.max(...models.map((model) => measure.value(stories[model.key])));
            const bars = models.map((model, i) => {
              const value = measure.value(stories[model.key]);
              const y = 190 + i * 62;
              return { width: (wide * value) / peak, bar: s.rect(measure.x, y, 0, 36, { fill: model.color, r: 6 }), label: s.text(measure.x, y + 27, measure.format(value), { size: 24, weight: 700 }) };
            });
            return { x: measure.x, title, bars };
          });
          const notes = s.lines(90, 482, ["Strangeness is the loss that the pretrained story model gives the stories.", `${share(loose.same_opening)} of the stories without the penalty begin “${loose.most_common_opening}”.`], { size: 24, fill: "ink2", leading: 1.5 });
          return (t) => {
            rows.forEach((row, i) => show(row, during(t, 0.3 + i * 0.1, 0.7 + i * 0.1)));
            panels.forEach((panel, k) => {
              const u = during(t, 0.9 + k * 1.3, 1.9 + k * 1.3);
              show(panel.title, during(t, 0.7 + k * 1.3, 1.1 + k * 1.3));
              for (const b of panel.bars) {
                b.bar.setAttribute("width", (b.width * u).toFixed(1));
                b.label.setAttribute("x", (panel.x + b.width * u + 10).toFixed(1));
                show(b.label, during(t, 1.5 + k * 1.3, 1.9 + k * 1.3));
              }
            });
            notes.forEach((line, i) => show(line, during(t, 5 + i * 0.5, 5.5 + i * 0.5)));
          };
        },
      },
    ],
  };
};
