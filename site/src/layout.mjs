// The page around a note: the pipeline as the header, the note with the list
// of its sections beside it, and the way on at the foot.

import { ASSETS } from "./assets.mjs";
import { REPO, pages, stages } from "./structure.mjs";

const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

// The mark: a line through two stages, as in the header.
const MARK = `<svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" class="mark-ground"/><path d="M8 16h16" class="mark-line"/><circle cx="9" cy="16" r="4.2" class="mark-dot"/><circle cx="23" cy="16" r="4.2" class="mark-dot"/></svg>`;
// The company the author works at, with its mark. The mark is Latere's own, from
// latere.ai/onboarding/brand/latere-mark.svg, and takes the color of the text.
export const LATERE = `<a class="latere" href="https://latere.ai"><svg class="latere-mark" viewBox="139.68 271.24 750.69 440.74" aria-hidden="true"><g fill="currentColor"><path d="M728.1 280.1C701.8 282.6 670.6 292.5 639.8 308.1C601.3 327.7 563.4 354.4 526.5 388C510.2 402.8 469 444.2 456.3 458.4C424.8 493.7 402.4 525.4 392.5 548.9C390.6 553.4 389 557.6 389 558.2C389 558.8 391.2 555.9 393.8 551.9C410.6 526.3 460.3 472.9 498 439.9C561.8 384.2 622.4 345.2 671.5 328.4C701.9 317.9 722.1 314.5 747.5 315.3C765 315.9 771.4 317 786 321.6C796.3 324.8 800.4 326.8 809.8 333.3C832.3 348.8 846.9 376.1 850 408.4C853.5 446.2 837.8 496.9 809.5 539.5C788.6 570.9 756.4 603.6 723 627.7C672.5 664.1 610.6 686.5 548.3 691C528.7 692.4 506 691.1 483.5 687.4C466.9 684.6 445.4 679.3 437.1 676.1C428.9 672.8 429.8 674.4 438.6 678.8C477.3 698.1 529.5 707.1 579.5 703C667.3 695.7 753.5 649.9 813.6 578.5C848.4 537.2 872.5 487.1 880.6 439.5C882.9 425.9 883 396.1 880.7 382.9C873.9 343.6 852.7 313.1 820 295.7C795.6 282.7 761.5 276.9 728.1 280.1Z"/><path d="M379 289.7C359.1 291 338.7 294.2 324 298.1C250.2 318 196.1 359 166.2 417.8C158.9 432.1 154 446.3 150.3 463.5C147.9 475.3 147.6 478.4 147.7 496.5C147.7 514 148 518 150.3 528.6C156.2 556.5 168 581.1 185.6 602C218.1 640.8 265 662.9 320.4 665.7C351.7 667.2 396.7 658.5 430 644.5C456.2 633.5 482.9 617.8 503 601.7C512.9 593.7 531.6 575.4 539.5 566C545.2 559.2 555 545 555 543.6C555 543.2 550.7 546.8 545.4 551.4C516.8 576.5 466.5 606.8 432.5 619.3C369.6 642.5 309.1 642.4 262.7 619C250 612.6 241.6 606.7 231.6 597.1C209.9 576.1 195.9 551.1 189.4 521C187.9 514.2 187.5 508.3 187.5 492C187.5 473.4 187.8 470.5 190.2 460.5C207.3 388.6 269 333.4 356 312.1C404.8 300.2 450.8 300 498.4 311.5C503.9 312.8 508.6 313.8 508.7 313.6C509.3 313 490.3 305.1 481 302.1C453.6 293.2 408.2 287.6 379 289.7Z"/><path d="M435.5 336.6C392.4 339.8 359.8 348.5 327.4 365.4C282.1 389 252.6 421.6 241.4 460.5C232.7 490.8 237.3 527.6 252.8 552C269.8 578.9 295.9 595 331.5 600.7C345.5 602.9 375.3 602.1 390.4 599.1C404.5 596.2 419.3 591.4 430.6 585.8C439.9 581.2 451.5 574.2 450.9 573.5C450.7 573.4 446.5 574.6 441.5 576.2C419.7 583.6 391 587.2 370.6 585.2C343.8 582.5 325.7 576.6 306.5 564.3C270.1 540.8 258.1 496.1 277 454.4C297.7 408.6 348.8 371.9 415.5 355C459.9 343.7 496.2 342.3 528.5 350.7C529.8 351 530.2 350.9 529.5 350.2C526.2 347 498.8 340 481.5 338C471.5 336.9 443.3 336 435.5 336.6Z"/><path d="M711.5 343.1C665.2 350.1 619.6 373.4 560.5 420.6C508.7 461.9 461 514.4 440.2 553.4C437.8 557.7 436.1 561.6 436.2 562.1C436.4 562.5 440 558.5 444.2 553.2C469.8 520.9 509.8 482.2 548 452.4C607.6 405.9 661.9 378.1 704.8 371.9C716.6 370.2 736.5 371.2 746 374C768.5 380.4 783.8 394.6 790.6 415.2C793.1 422.7 793.4 425 793.4 438C793.4 448.5 792.9 454.9 791.6 461C783.6 497.8 763.5 531.9 728.5 567.9C702.7 594.4 675 614.7 642 631.1C588.9 657.5 542.9 668.9 488.1 669.1C467.1 669.1 466.8 670 486.8 672.7C503.6 674.9 536.8 674.9 554.3 672.6C587.6 668.3 625.2 657.4 657 642.7C673.9 635 697.7 620.7 714 608.7C770.6 566.9 808.1 514 820.5 458.5C823 447.3 823.6 421.3 821.5 410.5C815.5 378.8 796 356.2 766.6 347.1C751.4 342.4 727.7 340.7 711.5 343.1Z"/><path d="M677 399.9C657.1 404.2 634.5 414 614.7 427C608.7 431 604 434.4 604.3 434.6C604.5 434.8 608.5 433.2 613.1 431.1C625.3 425.4 646.5 418.4 658.6 416C692 409.6 713 419.2 718.1 443.3C720 452.1 718.7 467.1 715.1 477.9C711.1 489.7 701.8 508 693.3 520.7C673.5 550.4 643.2 580.5 611.5 601.8C585.1 619.5 564.5 629 527 640.9C521.1 642.7 539.7 641.4 553 639C575.6 634.9 596.2 627.6 622.3 614.3C687 581.3 732.8 528.4 745.6 471.9C746.8 466.9 747.3 459.9 747.4 450.5C747.4 438.4 747.1 435.6 745.1 430C739.7 414.5 729.7 404.7 714.6 400.1C705.8 397.4 689.3 397.3 677 399.9Z"/><circle cx="564" cy="499.2" r="29.4"/></g></svg>Latere AI</a>`;
const ICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23eb6834'/%3E%3Cpath d='M8 16h16' stroke='white' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='9' cy='16' r='4.2' fill='white'/%3E%3Ccircle cx='23' cy='16' r='4.2' fill='white'/%3E%3C/svg%3E";

// The chain of stages with the side topics below the stage they branch from.
export function thread(current) {
  const item = (s) => {
    const here = s.slug === current || (s.leaves || []).some((l) => l.slug === current);
    const leaves = (s.leaves || []).map((l) => `<a class="leaf${l.slug === current ? " current" : ""}" href="/${l.slug}/" title="${l.n} ${l.short}"><span class="num">${l.n}</span> ${l.short}</a>`).join("");
    return `<li class="${here ? "here" : ""}"><span class="row"><a class="stage${s.slug === current ? " current" : ""}" href="/${s.slug}/" title="${s.n} ${s.short}"><span class="dot">${s.n}</span><span class="name">${s.short}</span></a></span>${leaves ? `<span class="leaves">${leaves}</span>` : ""}</li>`;
  };
  return `<nav class="thread" data-fit="current" aria-label="The pipeline"><ol>${stages.map(item).join("")}</ol></nav>`;
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
<a class="tool wide" href="/films/">Films</a>
<a class="tool code" href="/code/">Code</a>
<button class="tool theme" type="button" aria-label="Change the theme"><span class="theme-name">Light</span></button>
<a class="tool repo" href="${REPO}">GitHub</a>
</div>
<script>
// The pipeline shows as much as fits: every name, then the name of the stage being read,
// then the numbers alone. Run before the page is drawn, and again when the window changes.
(function () {
  var nav = document.querySelector(".thread"), list = nav.firstChild, levels = ["names", "current", "numbers", "bare"];
  function fit() {
    for (var i = 0; i < levels.length; i++) {
      nav.dataset.fit = levels[i];
      if (list.scrollWidth <= list.clientWidth + 1) return;
    }
    var here = nav.querySelector("li.here");
    if (here) nav.scrollLeft = here.offsetLeft - (nav.clientWidth - here.offsetWidth) / 2;
  }
  fit();
  addEventListener("resize", fit);
  if (document.fonts) document.fonts.ready.then(fit);
})();
</script>
</header>
${body}
<footer class="foot">
<p>minllm is a minimal pipeline for making a language model, written to be read. Every number on this site comes from a run that is logged in the <a href="${REPO}/tree/main/runs">repository</a>.</p>
<p>By <a href="https://changkun.de">Changkun Ou</a>, ${LATERE} · <a href="${REPO}">Code on GitHub</a> · <a href="${REPO}/blob/main/LICENSE">MIT license</a></p>
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
