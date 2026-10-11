// What generating a text costs in attention work, token by token, with and
// without the cache: one cell is one query reading one key.

import { html, svg } from "./kit.js";

const T = 12;
const CELL = 18;

export default function mount(el) {
  let at = 5;
  let timer = null;
  const figure = svg("svg", { viewBox: "0 0 720 286", role: "img", "aria-label": "Attention work per generated token" });
  const readout = html("ul", { class: "readout" });
  const step = html("button", { type: "button", text: "Next token" });
  const play = html("button", { type: "button", class: "primary", text: "Play" });
  const reset = html("button", { type: "button", text: "Reset" });
  el.append(html("div", { class: "controls" }, step, play, reset), figure, readout, html("p", { class: "lab-note", text: "One block and one head are drawn. The model has 8 blocks, and every one of them does this for every new token." }));

  function triangle(left, title, cached) {
    svg("text", { x: left, y: 16, class: "f-ink", "font-weight": 600 }, figure).textContent = title;
    for (let r = 0; r < T; r++) {
      for (let c = 0; c <= r; c++) {
        const done = r < at; // rows of tokens that exist
        const now = cached ? r === at - 1 : done; // computed for the newest token
        svg("rect", { x: left + c * CELL, y: 30 + r * CELL, width: CELL - 2, height: CELL - 2, rx: 3, class: now ? "f-s2" : done ? "f-s1" : "f-tint", "fill-opacity": now ? 1 : done ? 0.3 : 1 }, figure);
      }
    }
    svg("text", { x: left, y: 30 + T * CELL + 18, class: "f-muted" }, figure).textContent = cached ? "Orange: computed for this token. Blue: kept from before." : "Orange: computed again for this token, every row.";
  }

  function draw() {
    figure.replaceChildren();
    triangle(40, "Without a cache", false);
    triangle(400, "With a cache", true);
    const sum = (f) => Array.from({ length: at }, (_, i) => f(i + 1)).reduce((s, v) => s + v, 0);
    readout.replaceChildren(
      html("li", {}, "Tokens so far ", html("b", { text: String(at) })),
      html("li", {}, "Scores computed for this token: ", html("b", { text: String((at * (at + 1)) / 2) }), " without, ", html("b", { text: String(at) }), " with"),
      html("li", {}, "In all: ", html("b", { text: String(sum((t) => (t * (t + 1)) / 2)) }), " without, ", html("b", { text: String(sum((t) => t)) }), " with"),
    );
    step.disabled = at >= T;
  }
  const stop = () => (clearInterval(timer), (timer = null), (play.textContent = "Play"));
  step.addEventListener("click", () => (at < T && at++, draw()));
  reset.addEventListener("click", () => (stop(), (at = 1), draw()));
  play.addEventListener("click", () => {
    if (timer) return stop();
    if (at >= T) at = 1;
    play.textContent = "Stop";
    draw();
    timer = setInterval(() => (at >= T ? stop() : (at++, draw())), 450);
  });
  draw();
}
