// Build the website into site/dist: the notes of docs/ as pages, the code
// they explain, the logged runs that the interactive figures read, and a
// landing page.
//
//     cd site && npm ci && npm run build      # then serve site/dist
//
// The notes are rendered as they are. Nothing about the site is written into
// them: links are rewritten here, the figures are put into the pages so that
// they follow the site's theme, and the interactive figures are placed by
// src/labs.mjs.

import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import MarkdownIt from "markdown-it";
import { createHighlighter } from "shiki";

import { ASSETS, assets } from "./src/assets.mjs";
import { codeIndex, codePage, sources } from "./src/code.mjs";
import { about, filmed, films } from "./src/films.mjs";
import { labs } from "./src/labs.mjs";
import { landing } from "./src/landing.mjs";
import { page } from "./src/layout.mjs";
import { REPO, pages } from "./src/structure.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const dist = join(here, "dist");

const themes = { light: "github-light", dark: "github-dark" };
const highlighter = await createHighlighter({ themes: Object.values(themes), langs: ["python", "sh", "json", "yaml"] });
const md = new MarkdownIt({
  html: false,
  highlight(code, lang) {
    if (!["python", "sh", "json", "yaml"].includes(lang)) return ""; // plain blocks: program output
    return highlighter.codeToHtml(code, { lang, themes, defaultColor: false });
  },
});

const slugify = (text) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const files = sources(root); // the Python files that get a page of their own

// A link of a note, as the site needs it: another note becomes its page, a
// Python file its code page, and anything else opens on GitHub.
function href(target, base) {
  if (/^(https?:|#|mailto:)/.test(target)) return target;
  const path = join("/", base, target).slice(1); // relative to the repository's root
  const note = path.match(/^docs\/([0-9a-z-]+)\.md(#.*)?$/);
  if (note) return `/${note[1]}/${note[2] || ""}`;
  if (files.includes(path)) return `/code/${path}/`;
  return `${REPO}/${/\.[a-z]+$/.test(path) ? "blob" : "tree"}/main/${path.replace(/\/$/, "")}`;
}

// A figure of docs/figures as part of the page. The file carries its own
// light and dark colors for GitHub; here its classes take the site's.
function figure(name, alt) {
  const source = readFileSync(join(root, "docs", "figures", name), "utf8");
  const width = source.match(/ width="(\d+)"/)[1];
  const svg = source
    .replace(/<style>.*?<\/style>/s, "")
    .replace(/ width="\d+" height="\d+"/, "")
    .replace("<svg ", `<svg class="fig" aria-label="${alt.replace(/"/g, "&quot;")}" `);
  return `<figure class="figure" data-figure="${name}" style="max-width:${width}px">${svg.trim()}<button class="zoom" type="button" aria-label="Open this figure large">Enlarge</button></figure>`;
}

// Render markdown that lives in the directory `base` of the repository.
export function render(source, base) {
  const env = { base, headings: [] };
  const tokens = md.parse(source, env);
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === "heading_open" && t.tag === "h2") {
      const text = tokens[i + 1].content;
      const id = slugify(text);
      t.attrSet("id", id);
      env.headings.push({ id, text });
    }
    if (t.type === "inline") {
      for (const child of t.children) {
        if (child.type === "link_open") child.attrSet("href", href(child.attrGet("href"), base));
      }
    }
    if (t.type === "table_open") t.attrSet("class", "table");
  }
  let html = md.renderer.render(tokens, md.options, env);
  // A figure is a paragraph that holds one image, and a table gets a frame that scrolls on a narrow screen.
  html = html.replace(/<p><img src="figures\/([^"]+)" alt="([^"]*)"[^>]*><\/p>/g, (_, name, alt) => figure(name, alt));
  html = html.replace(/<table class="table">/g, '<div class="table-frame"><table class="table">').replace(/<\/table>/g, "</table></div>");
  return { html, headings: env.headings };
}

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });

const section = (lab) => `<section class="lab" id="try-${lab.name}" data-lab="${lab.name}"><h3 class="lab-title"><span>${lab.title}</span><button class="lab-large" type="button">Enlarge</button></h3><div class="lab-body"><noscript>This figure is interactive and needs JavaScript.</noscript></div></section>`;

const notes = {};
const explains = {}; // code file -> the notes that link to it
for (const p of pages) {
  const source = readFileSync(join(root, "docs", `${p.slug}.md`), "utf8");
  const title = source.match(/^# (.+)$/m)[1];
  const body = source.replace(/^# .+\n/, "");
  // The first paragraph of a note lists its code, scripts and tests.
  const [head, ...rest] = body.trim().split(/\n\n/);
  notes[p.slug] = { ...p, title: title.replace(/^\w+\. /, ""), files: render(head, "docs").html, ...render(rest.join("\n\n"), "docs") };
  for (const [, target] of source.matchAll(/\]\(\.\.\/([^)#]+\.py)\)/g)) (explains[target] ||= new Set()).add(p.slug);
}

const titles = Object.fromEntries(Object.values(notes).map((note) => [note.slug, note.title]));
// The element that a chapter's film is played in.
function screen(slug) {
  const { n, title, next } = about(slug, titles);
  return `<section class="film" data-film="${slug}" data-n="${n}" data-title="${title}" data-next="${next}"><noscript>This chapter has a film, which needs JavaScript.</noscript></section>`;
}

for (const note of Object.values(notes)) {
  let html = note.html;
  for (const lab of labs.filter((l) => l.note === note.slug)) {
    // A lab follows a figure of the note, or closes the section before a heading.
    const place = lab.after ? new RegExp(`(<figure class="figure" data-figure="${lab.after}"[^>]*>.*?</figure>)`, "s") : new RegExp(`(<h2 id="${lab.before}">)`);
    if (!place.test(html)) throw new Error(`${note.slug}: no place for the lab ${lab.name}`);
    html = html.replace(place, lab.after ? `$1\n${section(lab)}` : `${section(lab)}\n$1`);
  }
  if (filmed(note.slug)) html = `${screen(note.slug)}\n${html}`; // the film of a chapter is the first thing in its note
  mkdirSync(join(dist, note.slug), { recursive: true });
  writeFileSync(join(dist, note.slug, "index.html"), page({ ...note, html }, notes));
}

writeFileSync(join(dist, "index.html"), landing(readFileSync(join(root, "README.md"), "utf8"), notes, render, labs, section));

// Every interactive figure on one page, each with the note it belongs to.
const gallery = labs.map((lab) => `<p class="kicker"><a href="/${lab.note}/">${notes[lab.note].n} ${notes[lab.note].title}</a></p>\n${section(lab)}`).join("\n");
mkdirSync(join(dist, "interactive"), { recursive: true });
writeFileSync(join(dist, "interactive", "index.html"), page({ slug: null, title: "Interactive figures", files: "", html: `<p>The figures of this site that can be tried, in the order of the pipeline. Each reads what it shows from the runs logged in the repository.</p>\n${gallery}`, headings: [] }, notes));

// Every film on one page, in the order of the pipeline.
const reel = films().map((slug) => `<h2 id="${slug}"><span class="num">${notes[slug].n}</span> <a href="/${slug}/">${notes[slug].title}</a></h2>\n${screen(slug)}`).join("\n");
mkdirSync(join(dist, "films"), { recursive: true });
writeFileSync(join(dist, "films", "index.html"), page({ slug: null, title: "Films", files: "", html: `<p>Every chapter as a film of about a minute, in the order of the pipeline. A film is drawn by the page from the runs logged in the repository, so it can be stopped and moved to any second. The note below each title has the whole chapter. The same films are video files, attached to the <a href="${REPO}/releases">releases on GitHub</a>.</p>\n${reel}`, headings: films().map((slug) => ({ id: slug, text: `${notes[slug].n} ${notes[slug].title}` })) }, notes));

// The code: one page per Python file, and a list of them all.
const explained = (path) => [...(explains[path] || [])].map((slug) => notes[slug]);
for (const path of files) {
  const source = readFileSync(join(root, path), "utf8");
  const code = highlighter.codeToHtml(source.replace(/\n$/, ""), { lang: "python", themes, defaultColor: false, transformers: [{ line(node, line) { node.properties.id = `L${line}`; } }] });
  mkdirSync(join(dist, "code", path), { recursive: true });
  writeFileSync(join(dist, "code", path, "index.html"), codePage({ path, source, code, notes: explained(path) }, notes));
}
writeFileSync(join(dist, "code", "index.html"), codeIndex(files.map((path) => ({ path, source: readFileSync(join(root, path), "utf8"), notes: explained(path) })), notes));

writeFileSync(join(dist, "404.html"), page({ slug: null, title: "Not found", files: "", html: '<p>There is no page at this address. <a href="/">The pipeline</a> lists all of them.</p>', headings: [] }, notes));

cpSync(join(root, "docs", "figures"), join(dist, "figures"), { recursive: true });
cpSync(join(root, "runs"), join(dist, "runs"), { recursive: true }); // what the interactive figures read
cpSync(join(here, "public"), dist, { recursive: true }); // files served as they are: the tokenizer of stage 1
for (const path of assets) {
  mkdirSync(dirname(join(dist, ASSETS, path)), { recursive: true });
  cpSync(join(here, "src", path), join(dist, ASSETS, path));
}

console.log(`built ${Object.keys(notes).length} notes, ${films().length} films, ${files.length} code pages and ${labs.length} interactive figures into ${dist}, ${readdirSync(join(dist, "figures")).length} figures`);
