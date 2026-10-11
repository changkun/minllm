// Two attention heads of the trained model on one sentence: pick a token and
// see how much it reads from each token before it.

import { choice, html, json } from "./kit.js";

export default async function mount(el) {
  const data = await json("/runs/base/attention.json");
  const state = { head: 0, token: data.tokens.indexOf(" She") };
  const sentence = html("div", { class: "attn" });
  const note = html("p", { class: "lab-note" });
  el.append(
    html("div", { class: "controls" }, choice("Head", data.heads.map((h, i) => [`layer ${h.layer}, head ${h.head}`, i]), 0, (v) => ((state.head = v), draw())), html("span", { class: "lab-note", text: "Point at a token, or click one to keep it." })),
    sentence,
    note,
    html("p", { class: "lab-note", text: "The weights are those of the trained 14M model, saved by scripts/dump_attention.py in runs/base/attention.json. Layers and heads count from zero." }),
  );

  function draw(preview = state.token) {
    const weights = data.heads[state.head].weights[preview];
    sentence.replaceChildren(
      ...data.tokens.map((token, i) => {
        const w = i <= preview ? weights[i] : 0;
        const chip = html("span", { class: `attn-token${i === preview ? " reading" : ""}${i > preview ? " later" : ""}`, text: token === "<|endoftext|>" ? "<eot>" : token });
        chip.style.setProperty("--w", w.toFixed(3));
        chip.addEventListener("pointerenter", () => draw(i));
        chip.addEventListener("click", () => ((state.token = i), draw()));
        return chip;
      }),
    );
    const ranked = weights.map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    note.textContent = `“${data.tokens[preview].trim()}” reads ${ranked.map(([w, i]) => `${(100 * w).toFixed(0)}% from “${data.tokens[i].trim() || data.tokens[i]}”`).join(", ")}. Tokens after it are out of its sight.`;
  }
  sentence.addEventListener("pointerleave", () => draw());
  draw();
}
