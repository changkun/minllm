// The code of the repository as pages: every Python file with its lines
// numbered, the notes that explain it, and a list of all files.

import { readdirSync } from "node:fs";
import { join } from "node:path";

import { page } from "./layout.mjs";
import { REPO } from "./structure.mjs";

const DIRECTORIES = [
  ["minllm", "The library, one concept per file"],
  ["scripts", "One entry point per step of the pipeline"],
  ["tests", "Checks for the claims the notes make"],
];

const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

export function sources(root) {
  return DIRECTORIES.flatMap(([dir]) => readdirSync(join(root, dir)).filter((name) => name.endsWith(".py") && name !== "__init__.py").sort().map((name) => `${dir}/${name}`));
}

// What a file says about itself: the first paragraph of its docstring. A test
// file has none, and is named after the file it tests.
function summary(source, path, files) {
  const doc = source.match(/^(?:#![^\n]*\n)?\s*"""\s*([\s\S]*?)"""/);
  if (!doc) {
    const tested = `minllm/${path.replace(/^tests\/test_/, "")}`;
    if (files.some((f) => f.path === tested)) return `Tests of ${tested}.`;
    const file = files.find((f) => f.path === path);
    return file.notes.length ? `Tests of ${file.notes.map((n) => `${n.n} ${n.title}`).join(", ")}.` : "";
  }
  const text = doc[1].trim().split(/\n\s*\n/)[0].replace(/\s+/g, " ");
  return text.length > 190 ? `${text.slice(0, 187).replace(/\s\S*$/, "")}…` : text;
}

export function codePage({ path, source, code, notes: explained }, notes) {
  const lines = source.split("\n").length - 1;
  const where = explained.length ? `<p class="kicker">Explained in ${explained.map((n) => `<a href="/${n.slug}/">${n.n} ${n.title}</a>`).join(", ")}</p>` : `<p class="kicker"><a href="/code/">All code</a></p>`;
  const html = `<p class="code-actions"><a class="button" href="/code/">All files</a> <a class="button" href="${REPO}/blob/main/${path}">View on GitHub</a> <span class="lab-note">${lines.toLocaleString("en")} lines</span></p>\n<div class="code-file">${code}</div>`;
  return page({ slug: null, title: path, kicker: where, files: "", html, headings: [], wide: true }, notes);
}

export function codeIndex(files, notes) {
  const groups = DIRECTORIES.map(([dir, what]) => {
    const rows = files.filter((f) => f.path.startsWith(`${dir}/`)).map((f) => `<tr><td><a href="/code/${f.path}/"><code>${f.path.slice(dir.length + 1)}</code></a></td><td>${escape(summary(f.source, f.path, files))}</td><td class="count">${(f.source.split("\n").length - 1).toLocaleString("en")}</td></tr>`);
    return `<h2 id="${dir}"><code>${dir}/</code></h2>\n<p>${what}.</p>\n<div class="table-frame"><table class="table code-list"><thead><tr><th>File</th><th>What it is</th><th>Lines</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  });
  const html = `<p>Every Python file of the repository, readable here. The notes link to these pages, and each page links back to the notes that explain it.</p>\n${groups.join("\n")}`;
  return page({ slug: null, title: "Code", files: "", html, headings: DIRECTORIES.map(([dir]) => ({ id: dir, text: `${dir}/` })) }, notes);
}
