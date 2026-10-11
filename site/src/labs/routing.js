// What each expert of the trained mixture gets: its share of the work and the
// tokens it receives most reliably, block by block.

import { html, json, slider } from "./kit.js";

export default async function mount(el) {
  const data = await json("/runs/moe/routing.json");
  const table = html("div", { class: "bars" });
  const readout = html("ul", { class: "readout" });
  el.append(html("div", { class: "controls" }, slider("Block", { min: 1, max: data.blocks.length, value: data.blocks.length }, draw)), table, readout, html("p", { class: "lab-note", text: `From ${data.tokens.toLocaleString("en")} tokens of held-out stories, in runs/moe/routing.json. A token is listed for the expert that most often gets it first, with how often that is.` }));
  function draw(block) {
    const b = data.blocks[block - 1];
    const peak = Math.max(...b.share);
    table.replaceChildren(
      ...b.share.map((share, e) => {
        const fill = html("span", { class: "bar-fill" });
        fill.style.width = `${((100 * share) / peak).toFixed(1)}%`;
        const tokens = b.favorites[e].slice(0, 9).map((f) => html("span", { class: "piece", title: `first for ${(100 * f.rate).toFixed(0)}% of its ${f.count} occurrences`, text: f.token.replaceAll("\n", "↵") }));
        return html("div", { class: "route-row" }, html("span", { class: "bar-label", text: `expert ${e + 1}` }), html("span", { class: "bar-track" }, fill), html("span", { class: "bar-value", text: `${(100 * share).toFixed(0)}%` }), html("span", { class: "route-tokens" }, ...tokens));
      }),
    );
    readout.replaceChildren(
      html("li", {}, "Its first expert follows from the token alone ", html("b", { text: `${(100 * b.by_token).toFixed(0)}%` }), " of the time"),
      html("li", {}, "and is the expert of the token before ", html("b", { text: `${(100 * b.by_previous).toFixed(0)}%` }), ` (chance: ${(100 * b.by_previous_baseline).toFixed(0)}%)`),
    );
  }
  draw(data.blocks.length);
}
