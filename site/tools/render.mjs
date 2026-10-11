// Render the films of the chapters to video files: every frame is the film's
// own drawing at that time, photographed in headless Chrome and handed to
// ffmpeg. Needs Chrome and ffmpeg on this machine. It reads the sources, so
// there is nothing to build first.
//
//     node tools/render.mjs                         # every film, into ../out/films/<chapter>.mp4
//     node tools/render.mjs 01-tokenizer            # one
//     node tools/render.mjs 01-tokenizer --sheet    # twelve of its frames in one picture, to look it over
//     node tools/render.mjs 01-tokenizer --at 12.5  # one frame, at that second
//     node tools/render.mjs --check                 # no pictures: draw every half second of every film and
//                                                   # report text that leaves the stage or lies on other text

import { spawn } from "node:child_process";
import { once } from "node:events";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { about, films } from "../src/films.mjs";
import { pages } from "../src/structure.mjs";
import { browser } from "./browser.mjs";

const site = join(dirname(fileURLToPath(import.meta.url)), "..");
const root = join(site, "..");
const out = join(root, "out", "films");
const FPS = 30;
const SIZE = { width: 1280, height: 720, scale: 1.5 }; // 1920 by 1080 pixels

const args = process.argv.slice(2);
const sheet = args.includes("--sheet");
const check = args.includes("--check");
const at = args.includes("--at") ? Number(args[args.indexOf("--at") + 1]) : null;
const names = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--at");
const titles = Object.fromEntries(pages.map((p) => [p.slug, readFileSync(join(root, "docs", `${p.slug}.md`), "utf8").match(/^# \w+\. (.+)$/m)[1]]));

// What a film asks for, served from where it is in the repository: the
// site's sources, the logged runs, and the files the site serves as they are.
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".jsonl": "text/plain" };
const page = (name) => `<!doctype html>
<html lang="en" data-theme="light">
<meta charset="utf-8">
<link rel="stylesheet" href="/src/style.css">
<body class="film-page">
<script type="module">
import { film } from "/src/films/film.js";
window.film = await film(${JSON.stringify(name)}, ${JSON.stringify(about(name, titles))});
document.body.append(window.film.root);
</script>
`;
const server = createServer((request, response) => {
  const url = decodeURIComponent(new URL(request.url, "http://x").pathname);
  if (url.startsWith("/film/")) return response.writeHead(200, { "content-type": "text/html" }).end(page(url.slice(6)));
  const path = url.startsWith("/src/") ? join(site, url) : url.startsWith("/runs/") ? join(root, url) : join(site, "public", url);
  if (!existsSync(path) || statSync(path).isDirectory()) return response.writeHead(404).end();
  response.writeHead(200, { "content-type": types[extname(path)] || "application/octet-stream" });
  createReadStream(path).pipe(response);
}).listen(0, "127.0.0.1");
await once(server, "listening");
const base = `http://127.0.0.1:${server.address().port}`;

// Run in the page: what is wrong with the film as it is drawn now. Text is
// measured where it is shown, and whatever is nearly faded out is left alone.
function wrong() {
  const stage = document.querySelector(".film-stage");
  const box = stage.getBoundingClientRect();
  const scale = 1280 / box.width;
  const shown = [];
  for (const text of stage.querySelectorAll("text")) {
    let opacity = 1;
    let hidden = !text.textContent.trim();
    for (let node = text; node !== stage; node = node.parentNode) {
      opacity *= Number(node.getAttribute("opacity") ?? 1);
      if (node.getAttribute("display") === "none" || node.getAttribute("visibility") === "hidden") hidden = true;
    }
    if (hidden || opacity < 0.5) continue;
    const r = text.getBoundingClientRect();
    shown.push({ string: text.textContent, left: (r.left - box.left) * scale, right: (r.right - box.left) * scale, top: (r.top - box.top) * scale, bottom: (r.bottom - box.top) * scale });
  }
  const found = [];
  for (const [i, a] of shown.entries()) {
    if (a.left < 8 || a.right > 1272 || a.top < 4 || a.bottom > 716) found.push(`“${a.string}” leaves the stage`);
    for (const b of shown.slice(i + 1)) {
      const across = Math.min(a.right, b.right) - Math.max(a.left, b.left);
      const down = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      if (across > 6 && down > 0.35 * Math.min(a.bottom - a.top, b.bottom - b.top)) found.push(`“${a.string}” lies on “${b.string}”`);
    }
  }
  return found;
}

mkdirSync(out, { recursive: true });
const chrome = await browser();
const tab = await chrome.page(sheet ? { ...SIZE, scale: 1 / 3 } : at === null ? SIZE : { ...SIZE, scale: 1 });
for (const name of names.length ? names : films()) {
  await tab.open(`${base}/film/${name}`);
  let duration;
  for (let tries = 0; !duration; tries++) {
    duration = await tab.run("window.film && window.film.duration");
    if (!duration) await new Promise((r) => setTimeout(r, 100));
    if (tries > 50) throw new Error(`${name}: the film did not load: ${(await tab.errors()).join("; ") || "no error was reported"}`);
  }
  if (check) {
    // The same complaint through many frames is reported once, with the first second it holds at.
    const seen = new Map();
    for (let t = 0; t <= duration; t += 0.5) {
      try {
        await tab.run(`window.film.draw(${t})`);
      } catch (error) {
        seen.set(`drawing fails: ${error.message.split("\n")[0]}`, seen.get(error.message) ?? t);
        continue;
      }
      for (const complaint of await tab.run(`(${wrong})()`)) if (!seen.has(complaint)) seen.set(complaint, t);
    }
    console.log(`${name}  ${duration.toFixed(1)} s  ${seen.size ? "" : "nothing found"}`);
    for (const [complaint, t] of seen) console.log(`  at ${t.toFixed(1)} s: ${complaint}`);
    if (seen.size) process.exitCode = 1;
    continue;
  }
  if (at !== null) {
    await tab.run(`window.film.draw(${at})`);
    writeFileSync(join(out, `${name}-${at}.png`), await tab.picture());
    console.log(join(out, `${name}-${at}.png`));
    continue;
  }
  const times = sheet ? Array.from({ length: 12 }, (_, i) => ((i + 0.85) * duration) / 12) : Array.from({ length: Math.round(duration * FPS) }, (_, i) => i / FPS);
  const target = join(out, `${name}.${sheet ? "png" : "mp4"}`);
  const input = ["-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-"];
  const output = sheet ? ["-vf", "tile=4x3:padding=4:color=gray", "-frames:v", "1"] : ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"];
  const ffmpeg = spawn("ffmpeg", [...input, ...output, target], { stdio: ["pipe", "inherit", "inherit"] });
  for (const t of times) {
    await tab.run(`window.film.draw(${t})`);
    if (!ffmpeg.stdin.write(await tab.picture())) await once(ffmpeg.stdin, "drain");
  }
  ffmpeg.stdin.end();
  await once(ffmpeg, "close");
  console.log(`${target}  ${duration.toFixed(1)} s, ${times.length} frames, ${(statSync(target).size / 1e6).toFixed(2)} MB`);
}
chrome.close();
server.close();
