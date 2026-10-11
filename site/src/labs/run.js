// The reference run of stage 3, to be read off: the loss at any step, and
// what the model wrote when asked to continue "Once upon a time" then.

import { chart } from "./chart.js";
import { html, json, jsonl } from "./kit.js";

export default async function mount(el) {
  const log = await jsonl("/runs/base/log.jsonl");
  const { prompt, samples } = await json("/runs/base/samples.json");
  const train = log.filter((r) => "train_loss" in r);
  const evals = log.filter((r) => "val_loss" in r);
  const steps = evals.at(-1).step;
  const tokens = 64 * 256; // per step: the batch of the reference run
  const readout = html("ul", { class: "readout" });
  const sample = html("p", { class: "sample" });
  const view = chart(el, {
    series: [
      { name: "training loss, mean of 50 steps", color: "s1", points: train.map((r) => [r.step, r.train_loss]) },
      { name: "validation loss, every 500 steps", color: "s2", points: evals.map((r) => [r.step, r.val_loss]), dots: true },
    ],
    x: { domain: [0, steps], ticks: [0, 1000, 2000, 3000, 4000, 5000, 6000], format: (t) => t.toLocaleString("en"), title: "training step" },
    y: { domain: [1.3, 9], ticks: [1.5, 2, 3, 4, 6, 8], log: true },
    onhover(step) {
      const t = train.reduce((best, r) => (Math.abs(r.step - step) < Math.abs(best.step - step) ? r : best));
      const e = evals.filter((r) => r.step <= Math.max(step, 0)).at(-1);
      readout.replaceChildren(
        html("li", {}, "Step ", html("b", { text: t.step.toLocaleString("en") })),
        html("li", {}, "Tokens seen ", html("b", { text: `${((t.step * tokens) / 1e6).toFixed(0)}M` })),
        html("li", {}, "Training loss ", html("b", { text: t.train_loss.toFixed(3) })),
        html("li", {}, "Learning rate ", html("b", { text: t.lr.toExponential(1) })),
        html("li", {}, `Validation loss at step ${e.step.toLocaleString("en")} `, html("b", { text: e.val_loss.toFixed(3) })),
        html("li", {}, "As uncertain as a choice among ", html("b", { text: `${Math.exp(e.val_loss).toFixed(1)} tokens` })),
      );
      const s = samples.filter((x) => x.step <= e.step).at(-1);
      sample.replaceChildren(html("b", { text: `Asked at step ${s.step.toLocaleString("en")} to continue “${prompt}”: ` }), s.text);
    },
  });
  el.append(readout, sample, html("p", { class: "lab-note", text: "Move over the chart. The curve is runs/base/log.jsonl, and the texts are what the training run printed at each evaluation, in runs/base/samples.json." }));
  view.show(2000);
}
