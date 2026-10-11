// The page around a note: the pipeline as the header, the note with the list
// of its sections beside it, and the way on at the foot.

import { ASSETS } from "./assets.mjs";
import { REPO, pages, stages } from "./structure.mjs";

const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

// The mark: a line through two stages, as in the header.
const MARK = `<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" class="mark-ground"/><path d="M8 16h16" class="mark-line"/><circle cx="9" cy="16" r="4.2" class="mark-dot"/><circle cx="23" cy="16" r="4.2" class="mark-dot"/></svg>`;
const ICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23eb6834'/%3E%3Cpath d='M8 16h16' stroke='white' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='9' cy='16' r='4.2' fill='white'/%3E%3Ccircle cx='23' cy='16' r='4.2' fill='white'/%3E%3C/svg%3E";

// The chain of stages with the side topics below the stage they branch from.
export function thread(current) {
  const item = (s) => {
    const here = s.slug === current || (s.leaves || []).some((l) => l.slug === current);
    const leaves = (s.leaves || []).map((l) => `<a class="leaf${l.slug === current ? " current" : ""}" href="/${l.slug}/" title="${l.n} ${l.short}"><span class="num">${l.n}</span> ${l.short}</a>`).join("");
    return `<li class="${here ? "here" : ""}"><a class="stage${s.slug === current ? " current" : ""}" href="/${s.slug}/" title="${s.n} ${s.short}"><span class="dot">${s.n}</span><span class="name">${s.short}</span></a>${leaves}</li>`;
  };
  return `<nav class="thread" aria-label="The pipeline"><ol>${stages.map(item).join("")}</ol></nav>`;
}

export function shell({ title, description, current, body }) {
  return `<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(title)}</title>
<meta name="description" content="${escape(description)}">
<meta name="author" content="Changkun Ou">
<script>
// The theme is set before the page is drawn: the reader's choice, or light. "auto" follows the system.
(function () {
  var choice = "light";
  try { choice = localStorage.getItem("theme") || "light"; } catch (e) {}
  var dark = choice === "dark" || (choice === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.documentElement.dataset.choice = choice;
})();
</script>
<link rel="icon" href="${ICON}">
<link rel="stylesheet" href="${ASSETS}/style.css">
</head>
<body>
<header class="top">
<a class="brand" href="/" aria-label="minllm, to the start">${MARK}<span>minllm</span></a>
${thread(current)}
<div class="tools">
<a class="tool" href="/code/">Code</a>
<button class="tool theme" type="button" aria-label="Change the theme"><span class="theme-name">Light</span></button>
<a class="tool repo" href="${REPO}">GitHub</a>
</div>
</header>
${body}
<footer class="foot">
<p>minllm is a minimal pipeline for making a language model, written to be read. Every number on this site comes from a run that is logged in the <a href="${REPO}/tree/main/runs">repository</a>.</p>
<p>By <a href="https://changkun.de">Changkun Ou</a>, Latere AI · <a href="${REPO}">Code on GitHub</a> · <a href="${REPO}/blob/main/LICENSE">MIT license</a></p>
</footer>
<dialog class="viewer" aria-label="A figure, enlarged">
<div class="viewer-bar"><span class="viewer-hint">Drag to move, scroll or pinch to zoom</span><button type="button" data-zoom="out" aria-label="Zoom out">−</button><button type="button" data-zoom="in" aria-label="Zoom in">+</button><button type="button" data-zoom="fit">Fit</button><button type="button" class="primary" data-zoom="close">Close</button></div>
<div class="viewer-stage"><div class="viewer-content"></div></div>
</dialog>
<script type="module" src="${ASSETS}/site.js"></script>
</body>
</html>
`;
}

// A link to a page of the chain, as a button with its number.
const way = (p, label) => `<a class="way" href="/${p.slug}/"><span class="way-label">${label}</span><span class="way-name"><span class="num">${p.n}</span> ${p.title}</span></a>`;

export function page(note, notes) {
  const index = pages.findIndex((p) => p.slug === note.slug);
  const at = pages[index];
  let kicker = note.kicker || "";
  const ways = [];
  if (at) {
    const stageIndex = stages.findIndex((s) => s.slug === (at.kind === "leaf" ? at.parent.slug : at.slug));
    const previous = at.kind === "leaf" ? stages[stageIndex] : stages[stageIndex - 1];
    const next = stages[stageIndex + 1];
    kicker = `<p class="kicker">${at.kind === "leaf" ? `Side topic of <a href="/${at.parent.slug}/">stage ${at.parent.n}, ${notes[at.parent.slug].title}</a>` : `Stage ${at.n} of ${stages.length}`}</p>`;
    if (previous) ways.push(way({ ...previous, title: notes[previous.slug].title }, at.kind === "leaf" ? "Back to the stage" : "Previous stage"));
    for (const leaf of at.leaves || []) ways.push(way({ ...leaf, title: notes[leaf.slug].title }, "Side topic of this stage"));
    if (next) ways.push(way({ ...next, title: notes[next.slug].title }, "Next stage"));
  }
  const toc = note.headings.length ? `<aside class="toc"><p class="toc-title">On this page</p><ol>${note.headings.map((h) => `<li><a href="#${h.id}">${escape(h.text)}</a></li>`).join("")}</ol></aside>` : "";
  const body = `<main class="note${note.wide ? " note-wide" : ""}">
<div class="note-head">
${kicker}
<h1>${at ? `<span class="num">${at.n}</span> ` : ""}${escape(note.title)}</h1>
${note.files ? `<div class="files">${note.files}</div>` : ""}
</div>
<div class="note-grid">
<article class="prose">
${note.html}
</article>
${toc}
</div>
${ways.length ? `<nav class="ways" aria-label="Where to go from here">${ways.join("")}</nav>` : ""}
</main>`;
  return shell({ title: `${at ? at.n + ". " : ""}${note.title} · minllm`, description: `${note.title}: part of minllm, a minimal pipeline for making a language model.`, current: note.slug, body });
}
