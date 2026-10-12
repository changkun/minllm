// Chapter 3b as a film: a page of a crawl is mostly menus, a rule for lines
// and five rules for pages keep its prose, copies of three kinds are removed,
// and a tenth of the crawl is left. Then the same model is trained on three
// versions of the text and measured on held-out text of every kind. The
// page, the counts, what the models wrote and their losses are those of
// runs/data/.

import { svg } from "../labs/kit.js";
import { during, mix, place, set, show } from "./stage.js";

export const sources = {
  funnel: "/runs/data/funnel.json",
  cross: "/runs/data/cross.json",
  raw: "/runs/data/raw.jsonl",
  rules: "/runs/data/rules.jsonl",
  clean: "/runs/data/clean.jsonl",
};

const PROMPT = "Once upon a time"; // what scripts/train.py asks a model to go on from
const LETTER = 0.602; // the width of a character of the monospaced font, in ems

const count = (n) => Math.round(n).toLocaleString("en");
const millions = (n) => `${count(n / 1e6)}M`;
const share = (v) => (v === 1 ? "100%" : `${(100 * v).toFixed(1)}%`);
const cut = (text, room) => (text.length > room ? `${text.slice(0, room - 1).trimEnd()}…` : text);

// Numbers between 0 and 1 that are the same on every run.
function seeded(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let x = Math.imul(a ^ (a >>> 15), 1 | a);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

// Text cut at its spaces into lines of at most so many characters each, with a mark where it is cut off.
function wrap(text, widths) {
  const lines = [];
  let rest = text;
  widths.forEach((width, i) => {
    const room = i + 1 < widths.length ? width : width - 2;
    const cut = rest.length <= room ? rest.length : rest.lastIndexOf(" ", room);
    lines.push(rest.slice(0, cut));
    rest = rest.slice(cut).trimStart();
  });
  if (rest) lines[lines.length - 1] += " …";
  return lines;
}

// The example page as the crawl gives it, in two columns: where every line is, and its text cut to its column.
const PAGE = { size: 22, top: 156, floor: 584, columns: [{ x: 60, room: 30 }, { x: 500, room: 54 }] };
function spots(example) {
  const half = Math.ceil(example.lines.length / 2);
  const pitch = (PAGE.floor - PAGE.top) / (half - 1);
  return example.lines.map(([text, kept], i) => {
    const column = PAGE.columns[Math.floor(i / half)];
    return { i, kept, text, shown: cut(text, column.room), x: column.x, y: PAGE.top + (i % half) * pitch };
  });
}
const written = (s, spot) => s.text(spot.x, spot.y, spot.shown, { size: PAGE.size, mono: true });
const tone = (node, fill) => node.setAttribute("class", `film-mono f-${fill}`);

export default ({ funnel, cross, raw, rules, clean }) => {
  const { example, train } = funnel;
  const steps = train.steps;
  // what is left after every step: the crawl, its English pages, their prose, the pages that pass, and those without copies
  const [crawl, english, , passed, exact, near, left] = steps;

  // the page
  const page = spots(example);
  const first = page.findIndex((l) => l.kept);
  const stay = page.filter((l) => l.kept);
  const priced = (l) => !l.kept && l.text.includes("$");

  // the rules for a page, in the order in which prepare.verdict checks them
  const reasons = [
    ["too short", "fewer than 50 words of prose"],
    ["not English", "under 2 of the 8 commonest words, or other alphabets"],
    ["not words", "few words with a letter, or words of odd lengths"],
    ["symbols", "more than one # or ... per ten words"],
    ["repeats itself", "over 30% of its lines occur more than once"],
  ].map(([name, when]) => ({ name, when, pages: train.dropped_by_rule[name] }));
  const most = Math.max(...reasons.map((r) => r.pages));

  // the three versions of the text, and the model trained on each
  const models = [
    { name: "raw", key: "web-raw", log: raw, what: "English pages as extracted", brief: "pages as extracted", text: english },
    { name: "rules", key: "web-rules", log: rules, what: "after the rules for lines and pages", brief: "after the rules", text: passed },
    { name: "clean", key: "web-clean", log: clean, what: "also without copies", brief: "also without copies", text: left },
  ].map((m) => ({ ...m, last: m.log.filter((e) => "sample" in e).at(-1) }));
  const trained = models[0].last.step;
  const kinds = [
    { key: "raw", name: "raw", note: "with menus" },
    { key: "rules", name: "rules", note: "prose" },
    { key: "clean", name: "clean", note: "no copies" },
    { key: "stories", name: "stories", note: "seen by none" },
  ].map((k) => ({ ...k, best: Math.min(...models.map((m) => cross.loss[m.key][k.key])) }));

  return {
    subtitle: "From pages of the web to text worth training on",
    scenes: [
      {
        seconds: 6,
        caption: `This is one of ${count(crawl.pages)} pages in eight files of Common Crawl.\nThe text of a page is mostly menus, links and the footer.`,
        draw(s) {
          const address = s.text(60, 120, example.url, { size: 22, fill: "muted", mono: true });
          const ground = s.rect(page[first].x - 8, page[first].y - 20, page[first].shown.length * PAGE.size * LETTER + 16, 27, { fill: "tint", stroke: "s2", r: 6 });
          const lines = page.map((spot) => written(s, spot));
          const said = s.text(1220, 120, `${page.length} lines, and the first sentence is line ${first + 1}`, { size: 22, anchor: "end", fill: "ink2" });
          return (t) => {
            show(address, during(t, 0.2, 0.5));
            lines.forEach((line, i) => show(line, during(t, 0.4 + i * 0.05, 0.65 + i * 0.05)));
            show(ground, during(t, 3, 3.4));
            show(said, during(t, 3, 3.4));
          };
        },
      },
      {
        seconds: 7.5,
        caption: "A line is kept if it has at least five words and ends like\na sentence. Menus fall away, and so does the list of prices.",
        draw(s) {
          s.text(60, 120, example.url, { size: 22, fill: "muted", mono: true });
          const lines = page.map((spot) => ({ ...spot, node: written(s, spot) }));
          // where the sentences go when nothing is left between them
          lines.filter((l) => l.kept).forEach((l, k) => (l.to = 214 + k * 36));
          const tally = s.group();
          tally.text(60, 238, `${stay.length} of ${page.length} lines stay`, { size: 36, weight: 700 });
          tally.text(60, 276, "as the prose of this page", { size: 22, fill: "ink2" });
          const prices = s.lines(60, 372, [`${page.filter(priced).length} lines of prices fall too:`, "a list of prices ends in no period."], { size: 22, fill: "ink2" });
          return (t) => {
            for (const l of lines) {
              const judged = 0.5 + l.i * 0.05;
              if (l.kept) {
                set(l.node, { y: mix(l.y, l.to, during(t, 4.3, 5.1)) });
                continue;
              }
              // the prices are looked at before they go
              const from = priced(l) ? 3.5 : judged + 0.2;
              const gone = during(t, from, from + 0.6);
              set(l.node, { y: l.y + 44 * gone ** 3 });
              show(l.node, 1 - gone);
              tone(l.node, t < judged ? "ink" : priced(l) ? "s2" : "muted");
            }
            prices.forEach((line) => show(line, during(t, 2.6, 3)));
            show(tally, during(t, 5.1, 5.6));
          };
        },
      },
      {
        seconds: 7,
        caption: "Then a whole page is judged by the prose it has left.\nMost pages that go are too short: lists of products or links.",
        draw(s) {
          const X = 830;
          const WIDE = 280;
          const head = s.text(70, 128, `The crawl tags ${count(english.pages)} of ${count(crawl.pages)} pages as English. A page is dropped as`, { size: 24, fill: "ink2" });
          const rows = reasons.map((r, i) => {
            const y = 168 + i * 54;
            const g = s.group();
            g.text(70, y + 22, r.name, { size: 26, weight: 600 });
            g.text(262, y + 21, r.when, { size: 22, fill: "ink2" });
            return { ...r, g, w: (WIDE * r.pages) / most, bar: g.rect(X, y, 0, 28, { fill: r.pages === most ? "s2" : "s1", r: 5 }), value: g.text(X, y + 22, "", { size: 24, weight: r.pages === most ? 700 : 400 }) };
          });
          const sum = s.text(70, 478, `${count(english.pages - passed.pages)} pages go, and ${count(passed.pages)} stay.`, { size: 28, weight: 600 });
          const addresses = s.text(70, 544, "", { size: 24, fill: "ink2" });
          svg("tspan", {}, addresses).textContent = `In them, ${count(train.addresses_replaced)} e-mail and IP addresses become `;
          svg("tspan", { class: "film-mono f-s1" }, addresses).textContent = "email@example.com";
          svg("tspan", {}, addresses).textContent = " and ";
          svg("tspan", { class: "film-mono f-s1" }, addresses).textContent = "0.0.0.0";
          return (t) => {
            show(head, during(t, 0.2, 0.6));
            rows.forEach((r, i) => {
              const u = during(t, 0.9 + i * 0.45, 1.8 + i * 0.45);
              show(r.g, during(t, 0.7 + i * 0.45, 1 + i * 0.45));
              set(r.bar, { width: r.w * u });
              set(r.value, { x: X + r.w * u + 12 });
              r.value.textContent = count(r.pages * u);
            });
            show(sum, during(t, 3.9, 4.3));
            show(addresses, during(t, 4.7, 5.1));
          };
        },
      },
      {
        seconds: 8,
        caption: "Copies are of three kinds, and each needs its own method:\nthe same page, nearly the same page, a line on many pages.",
        draw(s) {
          const random = seeded(11);
          const TOP = 156;
          const WIDE = 80;
          const TALL = 104;
          const text = (n) => Array.from({ length: n }, () => 0.4 + 0.6 * random());
          // a page as a picture: a sheet, and a stroke for every line of it
          const sheet = (on, x, widths, marked = -1) => {
            const g = on.group();
            g.rect(x, TOP, WIDE, TALL, { fill: "raised", stroke: "axis", r: 6, width: 1.5 });
            const strokes = widths.map((u, j) => g.line(x + 11, TOP + 17 + j * 14, x + 11 + (WIDE - 22) * u, TOP + 17 + j * 14, { stroke: j === marked ? "s2" : "muted", width: 4 }));
            return { g, strokes };
          };
          const column = (cx, title, method, n, noun) => {
            const g = s.group();
            g.text(cx, 132, title, { size: 26, weight: 600, anchor: "middle" });
            const how = s.text(cx, 338, method, { size: 22, anchor: "middle", fill: "ink2" });
            const value = s.text(cx, 392, "", { size: 38, weight: 700, anchor: "middle" });
            const name = s.text(cx, 424, noun, { size: 22, anchor: "middle", fill: "muted" });
            return { g, how, value, name, n };
          };

          // the same page twice
          const twice = column(250, "the same page twice", "a hash of its text", passed.pages - exact.pages, "exact copies");
          const same = text(6);
          sheet(twice.g, 250 - 96, same);
          const copy = sheet(twice.g, 250 + 16, same);
          twice.g.text(250, TOP + 62, "=", { size: 30, anchor: "middle", fill: "muted" });

          // nearly the same page: a signature each, cut into bands
          const nearly = column(640, "nearly the same page", "MinHash: 128 numbers in 16 bands", exact.pages - near.pages, "near copies");
          const alike = text(6);
          sheet(nearly.g, 640 - 96, alike);
          const other = sheet(nearly.g, 640 + 16, alike.map((u, j) => (j === 2 ? 1.4 - u : u)), 2);
          nearly.g.text(640, TOP + 62, "≈", { size: 30, anchor: "middle", fill: "muted" });
          // pages 80% alike agree on a band of 8 numbers with a chance of 0.8 to the 8th
          const agree = Array.from({ length: 16 }, () => random() < 0.8 ** 8);
          if (!agree.includes(true)) agree[5] = true;
          const bands = agree.map((_, b) => [0, 16].map((down) => s.rect(640 - 142 + b * 18, TOP + TALL + 14 + down, 14, 11, { fill: "axis", r: 3 })));

          // the same line on many pages
          const shared = column(1030, "a line on many pages", "counted over all pages", train.shared_lines, "shared lines");
          const notices = [1030 - 136, 1030 - 40, 1030 + 56].map((x) => sheet(shared.g, x, [...text(5), 0.92], 5).strokes[5]);

          const [line, on] = train.most_shared_lines[0];
          const quote = s.chip(0, 462, cut(line, 84), { size: 22, stroke: "s2" });
          place(quote, 640 - quote.w / 2, 462);
          const where = s.text(640, 544, `the line that most pages share: it is on ${count(on)} of ${count(near.pages)} pages`, { size: 24, anchor: "middle", fill: "ink2" });

          // how a kind is found, and how many of it there are, counted up
          const counted = (t, c, from, to) => {
            show(c.how, during(t, from, from + 0.4));
            show(c.name, during(t, from + 0.3, from + 0.7));
            c.value.textContent = t < from + 0.3 ? "" : count(c.n * during(t, from + 0.3, to));
          };
          return (t) => {
            show(twice.g, during(t, 0.2, 0.6));
            show(copy.g, 1 - 0.8 * during(t, 1.1, 1.6));
            counted(t, twice, 0.9, 1.9);

            show(nearly.g, during(t, 1.9, 2.3));
            bands.forEach((pair, b) =>
              pair.forEach((cell) => {
                show(cell, during(t, 2.4 + b * 0.04, 2.6 + b * 0.04));
                cell.setAttribute("class", agree[b] && t >= 3.3 ? "f-s3" : "f-axis");
              }),
            );
            show(other.g, 1 - 0.8 * during(t, 3.5, 4));
            counted(t, nearly, 3.2, 4.2);

            show(shared.g, during(t, 4.1, 4.5));
            notices.forEach((stroke) => show(stroke, 1 - 0.8 * during(t, 5.2, 5.7)));
            counted(t, shared, 4.8, 5.7);

            show(quote, during(t, 5.6, 6));
            show(where, during(t, 5.8, 6.2));
          };
        },
      },
      {
        seconds: 7.5,
        caption: `${share(left.characters / crawl.characters)} of the crawl's characters are left. Two steps do most\nof it: the language tag, and the rule for lines.`,
        draw(s) {
          const X = 450;
          const WIDE = 340;
          const names = [`eight files of the crawl's ${count(funnel.files_in_crawl)}`, "tagged English", "lines of prose", "pages that pass the rules", "without exact copies", "without near copies", "without shared lines"];
          s.text(1110, 122, "pages", { size: 22, anchor: "end", fill: "muted" });
          s.text(1212, 122, "left", { size: 22, anchor: "end", fill: "muted" });
          const rows = steps.map((step, i) => {
            const y = 146 + i * 58;
            const g = s.group();
            g.text(70, y + 23, names[i], { size: 22 });
            return {
              g,
              now: step,
              was: steps[i - 1] || { pages: 0, characters: 0 },
              bar: g.rect(X, y, 0, 30, { fill: "s1", r: 5 }),
              size: g.text(X, y + 23, "", { size: 22, fill: "ink2" }),
              pages: g.text(1110, y + 23, "", { size: 22, anchor: "end", fill: "ink2" }),
              part: g.text(1212, y + 24, "", { size: 26, weight: i === steps.length - 1 ? 700 : 600, anchor: "end" }),
            };
          });
          return (t) => {
            rows.forEach((r, i) => {
              const from = 0.4 + i * 0.65;
              // a step starts from what the one before it left
              const u = during(t, from + 0.1, from + 0.75);
              const characters = mix(r.was.characters, r.now.characters, u);
              show(r.g, during(t, from, from + 0.3));
              set(r.bar, { width: (WIDE * characters) / crawl.characters });
              r.bar.setAttribute("class", (i === 1 || i === 2) && t >= 5.4 ? "f-s2" : "f-s1");
              set(r.size, { x: X + (WIDE * characters) / crawl.characters + 12 });
              r.size.textContent = `${millions(characters)}${i ? "" : " characters"}`;
              r.pages.textContent = count(mix(r.was.pages, r.now.pages, u));
              r.part.textContent = share(characters / crawl.characters);
              r.part.setAttribute("class", (i === 1 || i === 2) && t >= 5.4 ? "f-s2" : "f-ink");
            });
          };
        },
      },
      {
        seconds: 7,
        caption: `The same 14M model trains for ${count(trained)} steps on each version.\nGiven the opening of a story, the raw one writes a menu.`,
        draw(s) {
          const SIZE = 22;
          const X = 70;
          const room = Math.floor((1216 - X) / (SIZE * LETTER));
          const shown = models.map((m, i) => {
            const y = 150 + i * 150;
            // the characters a page cannot show are left out
            const wrote = [...m.last.sample.slice(PROMPT.length)].filter((c) => c >= " " && c !== "�").join("").replace(/ +/g, " ");
            const lines = wrap(wrote, [room - PROMPT.length, room]);
            const g = s.group();
            g.text(X, y, m.name, { size: 28, weight: 700 });
            g.text(X + 96, y, `${m.what}: ${millions(m.text.characters)} characters`, { size: 22, fill: "muted" });
            g.text(X, y + 42, PROMPT, { size: SIZE, fill: "muted", mono: true });
            return { g, lines, from: 0.5 + i * 1.5, nodes: [g.text(X + PROMPT.length * SIZE * LETTER, y + 42, "", { size: SIZE, mono: true }), g.text(X, y + 76, "", { size: SIZE, mono: true })] };
          });
          return (t) => {
            for (const row of shown) {
              show(row.g, during(t, row.from - 0.3, row.from));
              // written out, as it was generated: one piece after the other
              let letters = Math.round((row.lines[0].length + row.lines[1].length) * during(t, row.from, row.from + 1.4) ** 0.8);
              row.nodes.forEach((node, i) => {
                node.textContent = row.lines[i].slice(0, letters);
                letters = Math.max(0, letters - row.lines[i].length);
              });
            }
          };
        },
      },
      {
        seconds: 8.5,
        caption: "Each model is best on the kind of text it was trained on.\nA loss can be compared only on the same text: down a column.",
        draw(s) {
          const centre = (c) => 548 + c * 186;
          const cursor = s.rect(0, 134, 164, 292, { fill: "tint", stroke: "s2", r: 10 });
          const frame = s.group();
          frame.text(90, 164, "trained on", { size: 22, fill: "muted" });
          frame.text(centre(1.5), 120, "loss on held-out text of each kind", { size: 22, anchor: "middle", fill: "muted" });
          kinds.forEach((k, c) => {
            frame.text(centre(c), 164, k.name, { size: 26, weight: 600, anchor: "middle" });
            frame.text(centre(c), 192, k.note, { size: 22, anchor: "middle", fill: "muted" });
          });
          const rows = models.map((m, r) => {
            const y = 250 + r * 66;
            const g = s.group();
            g.text(90, y, m.name, { size: 28, weight: 600 });
            g.text(90, y + 27, m.brief, { size: 22, fill: "muted" });
            return { g, cells: kinds.map((k, c) => ({ best: cross.loss[m.key][k.key] === k.best, node: g.text(centre(c), y + 8, cross.loss[m.key][k.key].toFixed(2), { size: 32, anchor: "middle" }) })) };
          });
          const said = s.lines(640, 470, ["On raw pages the raw model is best: menus are easy to predict.", "On prose, and on stories that none of them saw, cleaning helps.", "Removing copies adds little in a sample this small."], { size: 24, anchor: "middle", fill: "ink2", leading: 1.42 });
          return (t) => {
            show(frame, during(t, 0.2, 0.6));
            // the column that is looked at: raw pages, then prose, then the stories
            const at = 2 * during(t, 3.4, 4.2) + during(t, 4.9, 5.5);
            set(cursor, { x: centre(at) - 82 });
            show(cursor, Math.min(during(t, 2, 2.4), 1 - during(t, 6.4, 6.8)));
            rows.forEach((row, r) => {
              show(row.g, during(t, 0.6 + r * 0.4, 1 + r * 0.4));
              row.cells.forEach((cell, c) => {
                const marked = cell.best && t >= 2.2 && at >= c - 0.5;
                cell.node.setAttribute("class", marked ? "f-s3" : "f-ink");
                cell.node.setAttribute("font-weight", marked ? 700 : 400);
              });
            });
            show(said[0], during(t, 2.4, 2.8));
            show(said[1], during(t, 4.2, 4.6));
            show(said[2], during(t, 5.9, 6.3));
          };
        },
      },
    ],
  };
};
