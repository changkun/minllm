// A film: a chapter told in a few scenes, each a drawing that is a function
// of time. The same code plays on the page and is rendered to a video file by
// site/render.mjs, frame by frame, so the two never differ.
//
// A film is a module in this directory. It names the logged runs it reads and
// returns its scenes:
//
//     export const sources = { log: "/runs/base/log.jsonl" };
//     export default (data) => ({
//       subtitle: "one line under the title",
//       scenes: [{ seconds: 7, caption: "what is being shown", draw(s) { ...; return (t) => { ... }; } }],
//     });
//
// draw gets a Stage (stage.js) to build the scene on once, and returns what
// sets it to second t of the scene. The title, the caption, the fades between
// scenes and the last card are made here.

import { html, json, jsonl, svg } from "../labs/kit.js";
import { H, Stage, W, during, show } from "./stage.js";

const TITLE = 2.8; // seconds of the title card
const END = 3.6; // and of the last card
const FADE = 0.4;
const TOP = 96; // a scene draws below this line, and above the caption
export const FLOOR = 590;

async function load(sources = {}) {
  const entries = await Promise.all(Object.entries(sources).map(async ([name, path]) => [name, await (path.endsWith(".jsonl") ? jsonl : json)(path)]));
  return Object.fromEntries(entries);
}

// about: { n, title, next } of the chapter. Returns { root, duration, draw(t) }.
export async function film(name, about) {
  const module = await import(`./${name}.js`);
  const made = module.default(await load(module.sources));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "film-stage", role: "img", "aria-label": `${about.n} ${about.title}, as a film` });
  svg("rect", { width: W, height: H, class: "f-surface" }, root);

  const title = {
    seconds: TITLE,
    draw(s) {
      s.text(W / 2, 300, about.n, { size: 150, weight: 700, anchor: "middle", fill: "s2" });
      s.text(W / 2, 400, about.title, { size: 68, weight: 700, anchor: "middle" });
      if (made.subtitle) s.text(W / 2, 462, made.subtitle, { size: 30, anchor: "middle", fill: "ink2" });
      s.text(W / 2, 640, "minllm · a language model from raw text, on a laptop", { size: 22, anchor: "middle", fill: "muted" });
      return () => {};
    },
  };
  const end = {
    seconds: END,
    draw(s) {
      s.text(W / 2, 300, about.next ? "Next" : "That is the pipeline", { size: about.next ? 30 : 54, weight: about.next ? 400 : 700, anchor: "middle", fill: about.next ? "muted" : "ink" });
      if (about.next) s.text(W / 2, 372, about.next, { size: 60, weight: 700, anchor: "middle" });
      s.text(W / 2, 470, "minllm.latere.site", { size: 30, anchor: "middle", fill: "s1" });
      s.text(W / 2, 640, "Changkun Ou, Latere AI", { size: 22, anchor: "middle", fill: "muted" });
      return () => {};
    },
  };

  let start = 0;
  const scenes = [title, ...made.scenes, end].map((scene) => {
    const stage = new Stage(root);
    const lines = (scene.caption || "").split("\n");
    const caption = lines.map((line, i) => stage.text(W / 2, 652 + 40 * i - 20 * (lines.length - 1), line, { size: 30, anchor: "middle", fill: "ink2" }));
    const built = { start, end: start + scene.seconds, stage, update: scene.draw(stage), caption };
    start = built.end;
    return built;
  });
  const duration = start;
  // What stays through the scenes: which chapter this is, and how far the film has come.
  const head = new Stage(root);
  head.text(40, 56, about.n, { size: 26, weight: 700, fill: "s2" });
  head.text(40 + 18 * about.n.length + 12, 56, about.title, { size: 26, weight: 600 });
  const progress = svg("rect", { y: H - 5, height: 5, class: "f-s2" }, root);

  function draw(t) {
    t = Math.min(duration, Math.max(0, t));
    for (const scene of scenes) {
      const on = t >= scene.start && (t < scene.end || scene.end === duration);
      scene.stage.g.setAttribute("display", on ? "inline" : "none");
      if (!on) continue;
      // the first scene is there from the start, and the last one stays
      const arriving = scene.start === 0 ? 1 : during(t, scene.start, scene.start + FADE);
      const leaving = scene.end === duration ? 1 : 1 - during(t, scene.end - FADE, scene.end);
      show(scene.stage, Math.min(arriving, leaving));
      scene.update(t - scene.start);
    }
    show(head, Math.min(during(t, TITLE - FADE, TITLE), 1 - during(t, duration - END - FADE, duration - END)));
    progress.setAttribute("width", ((W * t) / duration).toFixed(1));
  }
  draw(0);
  return { root, duration, draw, top: TOP };
}

const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

// The film in a page, with what plays it. el carries the chapter in its data attributes.
export default async function mount(el) {
  const { root, duration, draw } = await film(el.dataset.film, { n: el.dataset.n, title: el.dataset.title, next: el.dataset.next });
  let t = 0;
  let playing = false;
  let last = 0;
  const play = html("button", { type: "button", class: "primary", text: "Play" });
  const bar = html("input", { type: "range", min: 0, max: duration, step: 0.01, value: 0, "aria-label": "Where the film is" });
  const time = html("span", { class: "film-time" });
  const large = html("button", { type: "button", text: "Enlarge" });
  function seek(to) {
    t = Math.min(duration, Math.max(0, to));
    draw(t);
    bar.value = t;
    time.textContent = `${clock(t)} / ${clock(duration)}`;
  }
  function frame(now) {
    if (!playing) return;
    seek(t + (now - last) / 1000);
    last = now;
    if (t >= duration) return stop();
    requestAnimationFrame(frame);
  }
  function stop() {
    playing = false;
    play.textContent = t >= duration ? "Play again" : "Play";
  }
  function toggle() {
    if (playing) return stop();
    if (t >= duration) seek(0);
    playing = true;
    play.textContent = "Pause";
    last = performance.now();
    requestAnimationFrame(frame);
  }
  play.addEventListener("click", toggle);
  root.addEventListener("click", toggle);
  bar.addEventListener("input", () => seek(Number(bar.value)));
  large.addEventListener("click", () => {
    const on = el.classList.toggle("large");
    large.textContent = on ? "Close" : "Enlarge";
    document.body.style.overflow = on ? "hidden" : "";
  });
  el.replaceChildren(root, html("div", { class: "film-controls" }, play, bar, time, large));
  seek(TITLE * 0.7); // the title, as the picture to start from
  t = 0;
}
