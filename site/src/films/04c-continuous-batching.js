// The side topic 4c as a film: batches that start and end together against
// a running batch that requests join and leave, what every row of it keeps
// on its own, and what it gained in the benchmark of runs/bench/. The reply
// lengths of the first two scenes and the positions of the third are made up
// for the drawing, as in docs/figures/batching-timeline.svg; every
// measurement is from the benchmark.

import { clamp, during, set, show } from "./stage.js";

export const sources = {
  qwen: "/runs/bench/batching-qwen-mps.json",
  story: "/runs/bench/batching-sft-words-mps.json",
};

const LENGTHS = [5, 11, 4, 8, 6, 12, 3, 9, 7, 5, 10, 4, 6, 8, 5, 9]; // of sixteen requests, in the order they arrive
const SLOTS = 4;
const count = (n) => n.toLocaleString("en");
const seconds = (v) => (v < 0.1 ? v.toFixed(2) : v.toFixed(1));

// Where every reply runs: [slot, first step, steps], and the steps a row waits after it.
function scheduled(running) {
  const placed = [];
  if (running) {
    const free = Array(SLOTS).fill(0);
    for (const n of LENGTHS) {
      const slot = free.indexOf(Math.min(...free));
      placed.push({ slot, from: free[slot], n, idle: 0 });
      free[slot] += n;
    }
    return { placed, end: Math.max(...free) };
  }
  let start = 0;
  for (let i = 0; i < LENGTHS.length; i += SLOTS) {
    const batch = LENGTHS.slice(i, i + SLOTS);
    const longest = Math.max(...batch);
    batch.forEach((n, slot) => placed.push({ slot, from: start, n, idle: longest - n }));
    start += longest;
  }
  return { placed, end: start };
}

// The slots of a batch over the steps, drawn up to a step that moves.
function timeline(s, top, running, unit) {
  const X = 200;
  const { placed, end } = scheduled(running);
  for (let slot = 0; slot < SLOTS; slot++) s.text(X - 16, top + slot * 40 + 25, `slot ${slot + 1}`, { size: 20, anchor: "end", fill: "muted" });
  const bars = placed.map((p) => ({
    ...p,
    reply: s.rect(X + p.from * unit, top + p.slot * 40, 0, 32, { fill: running ? "s2" : "s1", r: 5 }),
    wait: p.idle ? s.rect(X + (p.from + p.n) * unit, top + p.slot * 40, 0, 32, { fill: "grid", r: 5 }) : null,
  }));
  const cursor = s.line(X, top - 8, X, top + SLOTS * 40, { stroke: "ink2", width: 2 });
  return {
    end,
    draw(step) {
      for (const b of bars) {
        set(b.reply, { width: Math.max(0, unit * clamp(step - b.from, 0, b.n) - 3) });
        if (b.wait) set(b.wait, { width: Math.max(0, unit * clamp(step - b.from - b.n, 0, b.idle) - 3) });
      }
      const at = X + unit * Math.min(step, end);
      set(cursor, { x1: at, x2: at });
      show(cursor, step > 0 && step < end ? 1 : 0);
    },
  };
}

// Pairs of bars, the batches that end together above the running batch, each pair with its own scale.
function pairs(s, top, rows) {
  const X = 470;
  const WIDE = 560;
  const made = rows.map(([name, still, running], i) => {
    const y = top + i * 116;
    const peak = Math.max(still, running);
    const g = s.group();
    g.lines(X - 24, y + 26, name, { size: 24, anchor: "end", leading: 1.25 });
    return {
      g,
      bars: [still, running].map((v, k) => ({
        w: Math.max(6, (WIDE * v) / peak),
        bar: s.rect(X, y + k * 44, 0, 36, { fill: k ? "s2" : "s1", r: 6 }),
        value: s.text(X + Math.max(6, (WIDE * v) / peak) + 12, y + k * 44 + 27, `${seconds(v)} s`, { size: 24, weight: 600 }),
      })),
    };
  });
  return (t, start) =>
    made.forEach((row, i) => {
      const from = start + i * 0.9;
      show(row.g, during(t, from, from + 0.4));
      row.bars.forEach((b, k) => {
        set(b.bar, { width: b.w * during(t, from + 0.2 + k * 0.3, from + 0.9 + k * 0.3) });
        show(b.value, during(t, from + 0.8 + k * 0.3, from + 1.1 + k * 0.3));
      });
    });
}

function legend(s, y) {
  const g = s.group();
  g.rect(200, y - 20, 30, 24, { fill: "s1", r: 5 });
  g.text(242, y, "batches that start and end together", { size: 22, fill: "ink2" });
  g.rect(690, y - 20, 30, 24, { fill: "s2", r: 5 });
  g.text(732, y, "a running batch", { size: 22, fill: "ink2" });
  return g;
}

export default ({ qwen, story }) => {
  const run = (bench, batching, slots, arrival) => bench.runs.find((r) => r.batching === batching && r.slots === slots && r.arrival === arrival);
  const both = (bench, slots, arrival, key) => [run(bench, "static", slots, arrival)[key], run(bench, "continuous", slots, arrival)[key]];
  const qwenOnce = both(qwen, 16, "at once", "seconds");
  const qwenFirst16 = both(qwen, 16, "at random", "first_token_mean");
  const qwenFirst64 = both(qwen, 64, "at random", "first_token_mean");
  const storyOnce = both(story, 64, "at once", "seconds");
  const storyFirst = both(story, 64, "at random", "first_token_mean");
  const storySteps = both(story, 64, "at once", "steps");
  const differing = qwen.replies_differing_from_the_first_run["continuous 16 at once"];
  const UNIT = 22;

  return {
    subtitle: "Requests join and leave a batch between any two steps",
    scenes: [
      {
        seconds: 8.5,
        caption: "Stage 4's batches start and end together. A row whose reply has ended\nwaits for the longest one, and a new request waits for the whole batch.",
        draw(s) {
          const still = timeline(s, 200, false, UNIT);
          s.text(200, 160, "four slots, sixteen requests; a column is a decode step", { size: 22, fill: "muted" });
          const key = s.group();
          key.rect(200, 400, 30, 24, { fill: "s1", r: 5 });
          key.text(242, 420, "writing a reply", { size: 22, fill: "ink2" });
          key.rect(470, 400, 30, 24, { fill: "grid", r: 5 });
          key.text(512, 420, "done, waiting for the others", { size: 22, fill: "ink2" });
          return (t) => {
            still.draw(still.end * during(t, 0.6, 7.4));
            show(key, during(t, 1.5, 2));
          };
        },
      },
      {
        seconds: 8,
        caption: "A running batch gives a free slot to the next request at the next step.\nThe same replies take fewer steps, and none waits for another's.",
        draw(s) {
          s.text(200, 132, "together", { size: 22, fill: "muted" });
          const still = timeline(s, 146, false, UNIT);
          s.text(200, 362, "running", { size: 22, fill: "muted" });
          const moving = timeline(s, 376, true, UNIT);
          return (t) => {
            still.draw(still.end);
            moving.draw(moving.end * during(t, 0.6, 6.4));
          };
        },
      },
      {
        seconds: 8.5,
        caption: "Every row is at its own position. It takes its own rotary angles,\nand a mask hides the keys beyond its own length.",
        draw(s) {
          const lengths = [6, 15, 10, 3]; // the cache of every slot, before this step
          const WIDTH = 18;
          const X = 260;
          const rows = lengths.map((n, slot) => {
            const y = 170 + slot * 80;
            s.text(X - 24, y + 30, `slot ${slot + 1}`, { size: 22, anchor: "end", fill: "muted" });
            const cells = Array.from({ length: WIDTH }, (_, j) => s.rect(X + j * 40, y, 34, 44, { fill: j < n ? "s1" : "tint", r: 5 }));
            const fresh = s.rect(X + n * 40, y, 34, 44, { fill: "s2", r: 5 });
            const hidden = s.rect(X + (n + 1) * 40 - 3, y - 4, (WIDTH - n - 1) * 40 + 2, 52, { fill: "surface", stroke: "axis", r: 6 });
            const where = s.text(X + WIDTH * 40 + 20, y + 30, `position ${n}`, { size: 22, weight: 600 });
            return { cells, fresh, hidden, where };
          });
          const key = s.group();
          key.rect(X, 520, 30, 24, { fill: "s1", r: 5 });
          key.text(X + 42, 540, "keys and values in the cache", { size: 22, fill: "ink2" });
          key.rect(X + 400, 520, 30, 24, { fill: "s2", r: 5 });
          key.text(X + 442, 540, "this step's new token", { size: 22, fill: "ink2" });
          const masked = s.group();
          masked.rect(X + 700, 520, 30, 24, { fill: "surface", stroke: "axis", r: 5 });
          masked.text(X + 742, 540, "hidden by the mask", { size: 22, fill: "ink2" });
          return (t) => {
            rows.forEach((r, i) => {
              r.cells.forEach((c, j) => show(c, during(t, 0.3 + i * 0.2, 0.7 + i * 0.2)));
              show(r.where, during(t, 1.8 + i * 0.2, 2.2 + i * 0.2));
              show(r.fresh, during(t, 3 + i * 0.15, 3.4 + i * 0.15));
              show(r.hidden, 0.85 * during(t, 4.6, 5.3));
            });
            show(key, during(t, 0.8, 1.2));
            show(masked, during(t, 4.6, 5.3));
          };
        },
      },
      {
        seconds: 9,
        caption: `Qwen's replies to GSM8K run to ${qwen.reply_tokens.max} tokens, ${Math.round(qwen.reply_tokens.mean)} on average. With 16 slots the\nrunning batch ends the ${qwen.requests} requests sooner, and a request waits less.`,
        draw(s) {
          const key = legend(s, 130);
          const bars = pairs(s, 180, [
            [["all requests done,", "all there at the start"], ...qwenOnce],
            [["first token, arriving", `at random, ${qwen.rate} a second`], ...qwenFirst16],
            [["the same,", "with 64 slots"], ...qwenFirst64],
          ]);
          return (t) => {
            show(key, during(t, 0.2, 0.6));
            bars(t, 0.6);
          };
        },
      },
      {
        seconds: 9.5,
        caption: `The story model's replies are alike in length, and each prefill costs a step.\nWith 64 slots, ${count(storySteps[1])} steps and ${count(story.requests)} prefills end later than ${count(storySteps[0])} steps.`,
        draw(s) {
          const key = legend(s, 130);
          const bars = pairs(s, 180, [
            [["all requests done,", "all there at the start"], ...storyOnce],
            [["first token, arriving", `at random, ${story.rate} a second`], ...storyFirst],
          ]);
          return (t) => {
            show(key, during(t, 0.2, 0.6));
            bars(t, 0.6);
          };
        },
      },
      {
        seconds: 8,
        caption: `In bfloat16, a sum depends on the shape of the batch, and a greedy reply with it:\n${differing} of Qwen's ${qwen.requests} replies changed somewhere. The story model's ${count(story.requests)} did not.`,
        draw(s) {
          const SIDE = 16;
          const cells = Array.from({ length: qwen.requests }, (_, i) => {
            const x = 330 + (i % SIDE) * 38;
            const y = 110 + Math.floor(i / SIDE) * 26;
            return s.rect(x, y, 32, 20, { fill: i < differing ? "s2" : "tint", r: 4 });
          });
          const note = s.text(640, 560, `greedy replies of the running batch, against batches of 16 that end together`, { size: 22, anchor: "middle", fill: "muted" });
          return (t) => {
            cells.forEach((c, i) => show(c, during(t, 0.3 + (i / qwen.requests) * 2.5, 0.6 + (i / qwen.requests) * 2.5)));
            show(note, during(t, 3, 3.5));
          };
        },
      },
    ],
  };
};
