// The story model of stage 3, writing in the browser: the 4-bit file of
// docs/10a-quantization.md, run by model.js in a worker. Every token it
// writes can be asked what else could have stood there.

import { REPO, html, percent, slider } from "./kit.js";

const PROMPTS = ["Once upon a time", "The tiny robot was sad because", "Lily and her dog went to the"];
const reasons = { end: "The model ended the story itself.", length: "Stopped at the length asked for.", context: "Stopped at 256 tokens, the longest text the model was trained on.", stopped: "Stopped." };

export default async function mount(el) {
  const state = { temperature: 0.8, topK: 50, tokens: 120, writing: false, started: false };
  const worker = new Worker(new URL("./generate-worker.js", import.meta.url), { type: "module" });

  const prompt = html("input", { type: "text", value: PROMPTS[0], "aria-label": "The prompt", maxlength: 300 });
  const presets = PROMPTS.map((text) => html("button", { type: "button", text, onclick: () => ((prompt.value = text), go(false)) }));
  const write = html("button", { type: "button", class: "primary", text: "Write a story" });
  const one = html("button", { type: "button", text: "One more token" });
  const stop = html("button", { type: "button", text: "Stop", disabled: "" });
  const story = html("p", { class: "story" });
  const status = html("p", { class: "lab-note", text: "The model is an 8 MB file. It is fetched when you ask for a story, and then runs on this device." });
  const speed = html("b", { text: "–" });
  const count = html("b", { text: "0" });
  const readout = html("ul", { class: "readout" }, html("li", {}, "Tokens written ", count), html("li", {}, "Tokens per second ", speed));
  const bars = html("div", { class: "bars" });
  const about = html("p", { class: "lab-note" });
  const choice = html("div", { class: "choice" }, about, bars);

  el.append(
    html("div", { class: "controls" }, prompt),
    html("div", { class: "controls presets" }, ...presets),
    html(
      "div",
      { class: "controls" },
      slider("Temperature", { min: 0, max: 1.5, step: 0.05, value: state.temperature, format: (v) => v.toFixed(2) }, (v) => (state.temperature = v)),
      slider("Top-k", { min: 0, max: 100, value: state.topK, format: (v) => (v ? String(v) : "off") }, (v) => (state.topK = v)),
      slider("Length", { min: 20, max: 250, step: 10, value: state.tokens }, (v) => (state.tokens = v)),
    ),
    html("div", { class: "controls" }, write, one, stop),
    story,
    readout,
    choice,
    status,
    html("p", { class: "lab-note" }, "This is the 14M model that stage 3 trained, with its weights rounded to 4 bits as in 10a. The page runs it with ", html("a", { href: `${REPO}/blob/main/site/src/labs/model.js`, text: "a few hundred lines of plain JavaScript" }), " that compute what minllm/model.py computes, and a test holds the two to the same numbers. Nothing is sent anywhere."),
  );

  let tokens = []; // the spans of the tokens written, each with the choice it was drawn from
  let shown = null;

  // The choice behind one written token: the likeliest candidates, and which one was drawn.
  function show(span) {
    shown?.classList.remove("asked");
    shown = span;
    span?.classList.add("asked");
    if (!span) return bars.replaceChildren(), (about.textContent = "");
    about.textContent = span.candidates.some((c) => c.drawn && c.piece === "<|endoftext|>") ? "The last choice: the model drew the token that ends a story." : "What could have stood here, and what was drawn:";
    const rows = [...bars.children];
    span.candidates.forEach((c, i) => {
      // The rows stay and only change, so that the bars move from one choice to the next.
      const row = rows[i] || bars.appendChild(html("div", { class: "bar-row" }, html("span", { class: "bar-label" }), html("span", { class: "bar-track" }, html("span", { class: "bar-fill" })), html("span", { class: "bar-value" })));
      row.classList.toggle("drawn", c.drawn);
      row.children[0].textContent = c.piece.trim() ? c.piece.trim() : JSON.stringify(c.piece);
      row.children[1].firstChild.style.width = `${(100 * c.p).toFixed(2)}%`;
      row.children[2].textContent = c.p < 0.001 ? "<0.1%" : percent(c.p, c.p < 0.1 ? 1 : 0);
    });
    for (const row of rows.slice(span.candidates.length)) row.remove();
  }

  function busy(on) {
    state.writing = on;
    write.disabled = one.disabled = on;
    stop.disabled = !on;
    for (const b of presets) b.disabled = on;
  }

  // more: go on from what is written, by one token. Otherwise start again from the prompt.
  function go(more) {
    if (state.writing) return;
    if (!more || !state.started) {
      story.replaceChildren(html("span", { class: "prompted", text: prompt.value }));
      tokens = [];
      show(null);
      count.textContent = "0";
    }
    busy(true);
    worker.postMessage({ type: "write", prompt: prompt.value, tokens: more ? 1 : state.tokens, temperature: state.temperature, topK: state.topK, topP: 1, more: more && state.started });
    state.started = true;
  }

  worker.onmessage = ({ data }) => {
    if (data.type === "loading") {
      status.textContent = data.unpacking ? "Unpacking the weights…" : `Fetching the model: ${(data.received / 1e6).toFixed(1)} of ${(data.total / 1e6).toFixed(1)} MB`;
    } else if (data.type === "token") {
      const span = html("span", { class: "written", tabindex: "0" });
      span.candidates = data.candidates;
      if (data.end) {
        span.classList.add("ended");
        span.textContent = "■";
        span.style.setProperty("--doubt", 0);
      } else {
        // A token can end in the middle of a character. The text so far is right as a whole, so what is new goes to this token.
        const before = tokens.reduce((n, t) => n + t.textContent.length, 0);
        span.textContent = data.text.endsWith("�") ? "" : data.text.slice(before);
        // How dark a token is drawn: by its surprise, -log2 of its probability, up to six bits.
        span.style.setProperty("--doubt", Math.min(1, -Math.log2(data.p) / 6).toFixed(3));
        count.textContent = String(tokens.length + 1);
        speed.textContent = data.perSecond.toFixed(0);
        tokens.push(span);
      }
      span.addEventListener("pointerenter", () => show(span));
      span.addEventListener("focus", () => show(span));
      story.append(span);
      show(span);
      status.textContent = "Point at a word, or tap it, to see what it was chosen from. A darker word was a less likely pick.";
    } else if (data.type === "done") {
      busy(false);
      if (data.reason !== "length" || tokens.length !== 1) status.textContent = `${reasons[data.reason]} Point at a word, or tap it, to see what it was chosen from. A darker word was a less likely pick.`;
      if (data.reason === "end" || data.reason === "context") state.started = false;
    } else if (data.type === "error") {
      busy(false);
      status.textContent = `The model could not be run: ${data.message}`;
    }
  };
  worker.onerror = (event) => {
    busy(false);
    status.textContent = `The model could not be run: ${event.message || "this browser does not start module workers"}`;
  };

  write.addEventListener("click", () => go(false));
  one.addEventListener("click", () => go(true));
  stop.addEventListener("click", () => worker.postMessage({ type: "stop" }));
  prompt.addEventListener("keydown", (event) => event.key === "Enter" && go(false));
  prompt.addEventListener("input", () => (state.started = false));
}
