// Which chapters have a film, and what a film is told about its chapter.
// The build puts a film at the top of its note, and tools/render.mjs makes a
// video file of it.

import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { pages, stages } from "./structure.mjs";

const src = dirname(fileURLToPath(import.meta.url));

export const filmed = (slug) => existsSync(join(src, "films", `${slug}.js`));
export const films = () => pages.filter((p) => filmed(p.slug)).map((p) => p.slug);

// titles: the title of every chapter by its slug. Returns { n, title, next },
// where next is the stage that follows in the chain, also after a side topic.
export function about(slug, titles) {
  const page = pages.find((p) => p.slug === slug);
  const stage = page.kind === "leaf" ? page.parent : page;
  const next = stages[stages.findIndex((s) => s.slug === stage.slug) + 1];
  return { n: page.n, title: titles[slug], next: next ? `${next.n} ${titles[next.slug]}` : "" };
}
