// GRPO's advantage on one group of replies: mark replies right or wrong and
// see what each gets, by the arithmetic of rl.group_advantages.

import { choice, html } from "./kit.js";

export default function mount(el) {
  const state = { right: [true, false, false, true, false, false, false, true], others: 0.5 };
  const rows = html("div", { class: "bars" });
  const note = html("p", { class: "sample" });
  el.append(
    html("div", { class: "controls" }, html("span", { class: "lab-note", text: "Eight replies to one prompt. Click a reply to change the verifier's verdict." })),
    html("div", { class: "controls" }, choice("The other prompts of the batch are", [["easy: 90% right", 0.9], ["mixed: 50%", 0.5], ["hard: 10%", 0.1]], 0.5, (v) => ((state.others = v), draw()))),
    rows,
    note,
  );
  function draw() {
    const r = state.right.map(Number);
    const n = r.length;
    const mean = r.reduce((s, v) => s + v, 0) / n;
    const std = Math.sqrt(r.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
    const group = r.map((v) => (v - mean) / (std + 1e-6));
    // a baseline over the whole batch: this group and seven others of the chosen difficulty
    const batch = (mean + 7 * state.others) / 8;
    const cell = (v, limit) => {
      const fill = html("span", { class: `swing ${v >= 0 ? "up" : "down"}` });
      fill.style.width = `${(50 * Math.min(1, Math.abs(v) / limit)).toFixed(1)}%`;
      return html("span", { class: "swing-track" }, fill, html("span", { class: "swing-value", text: `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}` }));
    };
    rows.replaceChildren(
      html("div", { class: "adv-row adv-head" }, html("span", {}), html("span", { text: "against its own group (GRPO)" }), html("span", { text: "against the batch's mean" })),
      ...r.map((v, i) => {
        const verdict = html("button", { type: "button", "aria-pressed": String(Boolean(v)), text: `reply ${i + 1}: ${v ? "right" : "wrong"}` });
        verdict.addEventListener("click", () => ((state.right[i] = !state.right[i]), draw()));
        return html("div", { class: "adv-row" }, verdict, cell(group[i], 2.5), cell(v - batch, 1));
      }),
    );
    const k = r.filter(Boolean).length;
    note.textContent =
      std === 0
        ? `All eight replies earned the same, so every advantage is zero and this group teaches nothing. Against the batch's mean they would all be pushed ${mean > batch ? "up" : "down"}, right or wrong alike, only because the prompt is ${mean > batch ? "easier" : "harder"} than the others.`
        : `${k} of 8 are right. Within the group, the right replies are pushed up and the wrong ones down, whatever the other prompts are like. Against the batch's mean, ${mean > batch + 0.05 ? "this prompt looks easy, so even its wrong replies lose little" : mean < batch - 0.05 ? "this prompt looks hard, so its right replies gain a lot and its wrong ones lose more than they should" : "the two agree, because this prompt is about as hard as the rest"}.`;
  }
  draw();
}
