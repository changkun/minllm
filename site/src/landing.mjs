// The landing page: what the repository is, the pipeline as a map, and the
// README's own tables of what gets built and what came out.

import { shell } from "./layout.mjs";
import { REPO, phases, stages } from "./structure.mjs";

// The sections of the README by their headings.
function sections(readme) {
  const parts = {};
  let name = "intro";
  for (const line of readme.split("\n")) {
    const heading = line.match(/^## (.+)$/);
    if (heading) name = heading[1];
    else parts[name] = (parts[name] || "") + line + "\n";
  }
  return parts;
}

// The map: two rows of stages by phase, each a link, with the side topics below.
function map(notes) {
  const box = (s) => {
    const leaves = (s.leaves || []).map((l) => `<a class="map-leaf" href="/${l.slug}/"><span class="num">${l.n}</span> ${l.short}</a>`).join("");
    return `<li><a class="map-stage" href="/${s.slug}/"><span class="num">${s.n}</span><span class="map-name">${s.short}</span></a>${leaves}</li>`;
  };
  const labels = phases.map((p) => `<p class="map-label" style="grid-column:${p.from + 1}/${p.to + 1}">${p.label}</p>`).join("");
  return `<div class="map-labels">${labels}</div><ol class="map-row">${stages.map(box).join("")}</ol>`;
}

// section: the element of an interactive figure, as the notes have it.
export function landing(readme, notes, render, labs, section) {
  const s = sections(readme);
  const intro = s.intro.split("\n\n").filter((p) => p.trim() && !p.startsWith("#") && !p.startsWith("!["));
  const part = (name) => render(s[name].replace(/^### /gm, "### "), "").html;
  const body = `<main class="landing">
<section class="hero">
<h1>A language model, from raw text to a served, post-trained model, on a laptop</h1>
<p class="byline">By <a href="https://changkun.de">Changkun Ou</a>, Latere AI</p>
${render(intro.slice(0, 2).join("\n\n"), "").html}
<p class="hero-actions"><a class="button primary" href="/${stages[0].slug}/">Start at stage 1</a> <a class="button" href="#try-generate">Run the model here</a> <a class="button" href="/code/">Read the code</a> <a class="button" href="${REPO}">GitHub</a></p>
</section>
<section class="map" aria-label="The pipeline">
${map(notes)}
<p class="map-note">Ten stages in order. A side topic hangs below the stage it changes, and the chain does not depend on it.</p>
</section>
<section class="landing-lab">
<p class="map-note">What the pipeline makes: the model of stages 1 to 3, here in the 4-bit file of 10a, with the sampler of stage 4.</p>
${section(labs.find((lab) => lab.name === "generate"))}
</section>
<section class="tries" aria-label="Interactive figures">
<p class="tries-title">Try it in the browser</p>
${labs.map((lab) => `<a class="button" href="/${lab.note}/#try-${lab.name}"><span class="num">${notes[lab.note].n}</span> ${lab.title}</a>`).join("\n")}
<a class="button" href="/interactive/">All on one page</a>
</section>
<section class="prose landing-prose">
<h2 id="stages">What gets built</h2>
${part("Stages")}
<h2 id="results">What came out</h2>
${part("What came out")}
<h2 id="quick-start">Quick start</h2>
${part("Quick start")}
<h2 id="two-models">Two models</h2>
${part("Two models")}
<h2 id="related-work">Related work</h2>
${part("Related work")}
</section>
</main>`;
  return shell({ title: "minllm · a minimal pipeline for making a language model", description: "From raw text to a tokenizer, a pretrained model, inference, fine-tuning, RLHF and RLVR, on a laptop and written to be read.", current: null, body });
}

