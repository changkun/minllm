// Type text and see the tokens that the trained tokenizer of stage 1 makes
// of it, then watch one chunk being merged step by step.

import { Tokenizer } from "./bpe.js";
import { html, json } from "./kit.js";

const show = (text) => text.replaceAll("\n", "↵\n");

export default async function mount(el) {
  const tok = new Tokenizer(await json("/tokenizer.json"));
  const input = html("textarea", { rows: 3, spellcheck: "false", "aria-label": "Text to tokenize" });
  input.value = "Once upon a time, there was a little kitten who loved the garden. Its name was Xylophone!";
  const tokens = html("div", { class: "tokens", "aria-live": "polite" });
  const readout = html("ul", { class: "readout" });
  const pieces = html("div", { class: "pieces" });
  const caption = html("p", { class: "lab-note" });
  const step = html("button", { type: "button", text: "Step" });
  const play = html("button", { type: "button", class: "primary", text: "Play" });
  const reset = html("button", { type: "button", text: "Reset" });
  el.append(
    input,
    tokens,
    readout,
    html("p", { class: "lab-note", text: "Each colored piece is one token. Click one to watch its chunk being merged, pair by pair, in the order in which the merges were learned." }),
    html("div", { class: "controls" }, step, play, reset),
    pieces,
    caption,
  );

  let states = [];
  let at = 0;
  let timer = null;

  function drawSteps() {
    const ids = states[at];
    const next = states[at + 1];
    // the pair that the next step merges: the first place where the next state differs
    let pair = -1;
    if (next) pair = ids.findIndex((id, i) => next[i] !== id);
    pieces.replaceChildren(...ids.map((id, i) => html("span", { class: `piece${next && (i === pair || i === pair + 1 || isSamePair(ids, i, pair)) ? " next" : ""}`, title: `token ${id}`, text: show(tok.text(id)) })));
    if (!next) caption.textContent = at === 0 ? "Nothing to merge: this chunk is a single byte, or no pair of its bytes was learned." : `Done after ${at} ${at === 1 ? "merge" : "merges"}: ${ids.length} ${ids.length === 1 ? "token" : "tokens"}, from ${states[0].length} bytes.`;
    else {
      const made = next.find((id, i) => ids[i] !== id);
      caption.textContent = `Step ${at + 1} of ${states.length - 1}: “${tok.text(ids[pair])}” and “${tok.text(ids[pair + 1])}” become token ${made}, the ${ordinal(made - 255)} merge that training learned.`;
    }
    step.disabled = !next;
  }

  // every occurrence of the pair merges in the same step
  function isSamePair(ids, i, pair) {
    if (pair < 0) return false;
    const [a, b] = [ids[pair], ids[pair + 1]];
    return (ids[i] === a && ids[i + 1] === b) || (ids[i - 1] === a && ids[i] === b);
  }

  function watch(chunk) {
    clearInterval(timer);
    timer = null;
    play.textContent = "Play";
    states = tok.steps(chunk);
    at = 0;
    drawSteps();
  }

  function draw() {
    const text = input.value;
    const chunks = text.match(tok.pattern) || [];
    tokens.replaceChildren();
    let count = 0;
    for (const chunk of chunks) {
      for (const id of tok.steps(chunk).at(-1)) {
        const span = html("span", { class: `tok tok-${count++ % 2}`, title: `token ${id}`, text: show(tok.text(id)) });
        span.addEventListener("click", () => watch(chunk));
        tokens.append(span);
      }
    }
    const bytes = new TextEncoder().encode(text).length;
    readout.replaceChildren(
      html("li", {}, "Characters ", html("b", { text: String([...text].length) })),
      html("li", {}, "Bytes ", html("b", { text: String(bytes) })),
      html("li", {}, "Tokens ", html("b", { text: String(count) })),
      html("li", {}, "Bytes per token ", html("b", { text: count ? (bytes / count).toFixed(2) : "0" })),
      html("li", {}, "Vocabulary ", html("b", { text: `${tok.size.toLocaleString("en")} tokens` })),
    );
    // start with the chunk that takes the most merges
    watch(chunks.reduce((best, c) => (tok.steps(c).length > tok.steps(best).length ? c : best), chunks[0] || ""));
  }

  step.addEventListener("click", () => {
    if (at < states.length - 1) at++;
    drawSteps();
  });
  reset.addEventListener("click", () => watch(tok.decode(states[0])));
  play.addEventListener("click", () => {
    if (timer) return watch(tok.decode(states[0]));
    if (at >= states.length - 1) at = 0;
    drawSteps();
    play.textContent = "Stop";
    timer = setInterval(() => {
      if (at >= states.length - 1) {
        clearInterval(timer);
        timer = null;
        play.textContent = "Play";
        return;
      }
      at++;
      drawSteps();
    }, 700);
  });
  input.addEventListener("input", draw);
  draw();
}

function ordinal(n) {
  const tail = n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th";
  return `${n.toLocaleString("en")}${tail}`;
}
