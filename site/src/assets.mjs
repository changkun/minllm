// The address of the site's own files: the stylesheet, the page's script
// and the interactive figures. The host serves everything under /assets/ as
// files that never change, to be kept for a year, so the address carries a
// hash of their contents and changes whenever one of them does.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const src = dirname(fileURLToPath(import.meta.url));

// What is served, by its path below the address.
export const assets = ["style.css", "site.js", ...readdirSync(join(src, "labs")).sort().map((name) => `labs/${name}`)];

const hash = createHash("sha256");
for (const path of assets) hash.update(path).update(readFileSync(join(src, path)));
export const ASSETS = `/assets/${hash.digest("hex").slice(0, 10)}`;
