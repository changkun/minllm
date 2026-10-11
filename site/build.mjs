// Build the website into site/dist: the notes of docs/ as pages, the figures,
// the logged runs that the interactive figures read, and a landing page.
//
//     cd site && npm ci && npm run build      # then serve site/dist
//
// The notes are rendered as they are. Nothing about the site is written into
// them: links are rewritten here, and the interactive figures are placed by
// src/labs.mjs.

import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import MarkdownIt from "markdown-it";
import { createHighlighter } from "shiki";

import { labs } from "./src/labs.mjs";
import { landing } from "./src/landing.mjs";
import { page } from "./src/layout.mjs";
import { REPO, pages } from "./src/structure.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const dist = join(here, "dist");

const highlighter = await createHighlighter({ themes: ["github-light", "github-dark"], langs: ["python", "sh", "json", "yaml"] });
const md = new MarkdownIt({
  html: false,
  highlight(code, lang) {
    if (!["python", "sh", "json", "yaml"].includes(lang)) return ""; // plain blocks: program output
    return highlighter.codeToHtml(code, { lang, themes: { light: "github-light", dark: "github-dark" }, defaultColor: false });
  },
});

const slugify = (text) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

// A link of a note, as the site needs it: another note becomes its page, and
// a file of the repository opens on GitHub.
function href(target, base) {
  if (/^(https?:|#|mailto:)/.test(target)) return target;
  const path = join("/", base, target).slice(1); // relative to the repository's root
  const note = path.match(/^docs\/([0-9a-z-]+)\.md(#.*)?$/);
  if (note) return `/${note[1]}/${note[2] || ""}`;
  if (path.startsWith("docs/figures/")) return `/figures/${path.slice("docs/figures/".length)}`;
  return `${REPO}/${/\.[a-z]+$/.test(path) ? "blob" : "tree"}/main/${path.replace(/\/$/, "")}`;
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
        if (child.type === "image") {
          child.attrSet("src", href(child.attrGet("src"), base));
          child.attrSet("loading", "lazy");
        }
      }
    }
    if (t.type === "table_open") t.attrSet("class", "table");
  }
  let html = md.renderer.render(tokens, md.options, env);
  // A figure is a paragraph that holds one image, and a table gets a frame that scrolls on a narrow screen.
  html = html.replace(/<p>(<img [^>]+>)<\/p>/g, '<figure class="figure">$1</figure>');
  html = html.replace(/<table class="table">/g, '<div class="table-frame"><table class="table">').replace(/<\/table>/g, "</table></div>");
  return { html, headings: env.headings };
}

rmSync(dist, { recursive: true, force: true });
mkdirSync(join(dist, "assets"), { recursive: true });

const notes = {};
for (const p of pages) {
  const source = readFileSync(join(root, "docs", `${p.slug}.md`), "utf8");
  const title = source.match(/^# (.+)$/m)[1];
  const body = source.replace(/^# .+\n/, "");
  // The first paragraph of a note lists its code, scripts and tests.
  const [files, ...rest] = body.trim().split(/\n\n/);
  notes[p.slug] = { ...p, title: title.replace(/^\w+\. /, ""), files: render(files, "docs").html, ...render(rest.join("\n\n"), "docs") };
}

for (const note of Object.values(notes)) {
  let html = note.html;
  for (const lab of labs.filter((l) => l.note === note.slug)) {
    const figure = new RegExp(`(<figure class="figure"><img src="/figures/${lab.after}"[^>]*></figure>)`);
    if (!figure.test(html)) throw new Error(`${note.slug}: no figure ${lab.after} to put the lab ${lab.name} after`);
    html = html.replace(figure, `$1\n<section class="lab" data-lab="${lab.name}"><h3 class="lab-title">${lab.title}</h3><div class="lab-body"><noscript>This figure is interactive and needs JavaScript.</noscript></div></section>`);
  }
  mkdirSync(join(dist, note.slug), { recursive: true });
  writeFileSync(join(dist, note.slug, "index.html"), page({ ...note, html }, notes));
}

writeFileSync(join(dist, "index.html"), landing(readFileSync(join(root, "README.md"), "utf8"), notes, render));
writeFileSync(join(dist, "404.html"), page({ slug: null, title: "Not found", files: "", html: '<p>There is no page at this address. <a href="/">The pipeline</a> lists all of them.</p>', headings: [] }, notes));

cpSync(join(root, "docs", "figures"), join(dist, "figures"), { recursive: true });
cpSync(join(root, "runs"), join(dist, "runs"), { recursive: true }); // what the interactive figures read
cpSync(join(here, "data"), join(dist, "data"), { recursive: true });
cpSync(join(here, "src", "style.css"), join(dist, "assets", "style.css"));
cpSync(join(here, "src", "site.js"), join(dist, "assets", "site.js"));
cpSync(join(here, "src", "labs"), join(dist, "assets", "labs"), { recursive: true });

console.log(`built ${Object.keys(notes).length} notes and ${labs.length} interactive figures into ${dist}, ${readdirSync(join(dist, "figures")).length} figures`);
