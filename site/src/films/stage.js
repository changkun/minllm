// What a scene of a film draws with: a group on a stage of 1280 by 720, a few
// shapes, and the arithmetic of a change over time. Nothing here measures the
// page it is on, so a frame comes out the same wherever it is drawn.

import { svg } from "../labs/kit.js";

export const W = 1280;
export const H = 720;

export const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
export const mix = (a, b, u) => a + (b - a) * u;
// How far a change that lasts from second a to second b has come at second t: 0 before, 1 after, eased between.
export function during(t, a, b) {
  const u = clamp((t - a) / (b - a));
  return u * u * (3 - 2 * u);
}

const MONO = 0.602; // the width of a character of the monospaced font, in ems

// A token as it is shown: a space at its start or end is made visible.
export const spelled = (string) => string.replaceAll(" ", "␣").replaceAll("\n", "↵");

export class Stage {
  constructor(parent) {
    this.g = svg("g", {}, parent);
  }

  // A stage inside this one, to move or fade several shapes as one.
  group() {
    return new Stage(this.g);
  }

  text(x, y, string, { size = 28, weight = 400, anchor = "start", fill = "ink", mono = false } = {}) {
    const node = svg("text", { x, y, "font-size": size, "font-weight": weight, "text-anchor": anchor, class: `f-${fill}${mono ? " film-mono" : ""}` }, this.g);
    node.textContent = string;
    return node;
  }

  // Several lines of text below each other, the first at y.
  lines(x, y, strings, { leading = 1.35, ...options } = {}) {
    const size = options.size || 28;
    return strings.map((string, i) => this.text(x, y + i * size * leading, string, options));
  }

  rect(x, y, w, h, { fill = "tint", stroke = null, r = 10, width = 2 } = {}) {
    return svg("rect", { x, y, width: w, height: h, rx: r, class: `f-${fill}${stroke ? ` s-${stroke}` : ""}`, "stroke-width": stroke ? width : 0 }, this.g);
  }

  line(x1, y1, x2, y2, { stroke = "axis", width = 2 } = {}) {
    return svg("line", { x1, y1, x2, y2, class: `s-${stroke}`, "stroke-width": width, "stroke-linecap": "round" }, this.g);
  }

  // A line with a head at its end.
  arrow(x1, y1, x2, y2, { stroke = "muted", width = 2.5 } = {}) {
    const g = svg("g", {}, this.g);
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = 11;
    svg("line", { x1, y1, x2: x2 - 0.6 * head * Math.cos(angle), y2: y2 - 0.6 * head * Math.sin(angle), class: `s-${stroke}`, "stroke-width": width, "stroke-linecap": "round" }, g);
    const corner = (turn) => `${x2 - head * Math.cos(angle + turn)},${y2 - head * Math.sin(angle + turn)}`;
    svg("polygon", { points: `${x2},${y2} ${corner(0.45)} ${corner(-0.45)}`, class: `f-${stroke}` }, g);
    return g;
  }

  // A box with a title in it, and a second line below the title if given.
  box(x, y, w, h, title, sub = "", { fill = "raised", stroke = "axis", size = 26, ink = "ink" } = {}) {
    const g = svg("g", {}, this.g);
    svg("rect", { x, y, width: w, height: h, rx: 12, class: `f-${fill} s-${stroke}`, "stroke-width": 2 }, g);
    const title_ = svg("text", { x: x + w / 2, y: y + h / 2 + (sub ? -4 : size * 0.35), "font-size": size, "font-weight": 600, "text-anchor": "middle", class: `f-${ink}` }, g);
    title_.textContent = title;
    if (sub) svg("text", { x: x + w / 2, y: y + h / 2 + size * 0.9, "font-size": size * 0.72, "text-anchor": "middle", class: ink === "ink" ? "f-ink2" : `f-${ink}` }, g).textContent = sub;
    return g;
  }

  // A piece of text in monospace on a rounded ground, as wide as its characters. Returns { g, w, h }.
  chip(x, y, string, { size = 30, fill = "tint", stroke = "axis", ink = "ink", pad = 12 } = {}) {
    const g = svg("g", { transform: `translate(${x} ${y})` }, this.g);
    const w = Math.max(1, [...string].length) * size * MONO + 2 * pad;
    const h = size * 1.7;
    const ground = svg("rect", { width: w, height: h, rx: 8, class: `f-${fill}${stroke ? ` s-${stroke}` : ""}`, "stroke-width": stroke ? 1.5 : 0 }, g);
    const label = svg("text", { x: pad, y: h / 2 + size * 0.35, "font-size": size, class: `f-${ink} film-mono` }, g);
    label.textContent = string;
    return { g, w, h, x, y, ground, label };
  }

  // Chips in a row from x, or centered on x, with their places. Returns the chips, each with its x.
  chips(x, y, strings, { gap = 8, center = false, ...options } = {}) {
    const made = strings.map((string) => this.chip(0, y, string, options));
    const total = made.reduce((sum, c) => sum + c.w, 0) + gap * (made.length - 1);
    let at = center ? x - total / 2 : x;
    for (const c of made) {
      place(c, at, y);
      at += c.w + gap;
    }
    return made;
  }

  // A line through points, which trace() can draw from its start.
  path(points, { stroke = "s1", width = 3 } = {}) {
    const node = svg("path", { d: "M" + points.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join(" L"), fill: "none", class: `s-${stroke}`, "stroke-width": width, "stroke-linejoin": "round", "stroke-linecap": "round" }, this.g);
    node.total = points.slice(1).reduce((sum, [a, b], i) => sum + Math.hypot(a - points[i][0], b - points[i][1]), 0);
    return node;
  }

  // The frame of a chart at (x, y) of w by h: lines and labels at the ticks of
  // the vertical axis, labels at those of the horizontal one. An axis is
  // { domain: [low, high], ticks, format, log, title }. Returns { px, py },
  // which give the place of a value on the stage.
  axes(x, y, w, h, { x: across, y: up }) {
    const part = (v, { domain, log }) => (log ? Math.log(v / domain[0]) / Math.log(domain[1] / domain[0]) : (v - domain[0]) / (domain[1] - domain[0]));
    const px = (v) => x + w * part(v, across);
    const py = (v) => y + h * (1 - part(v, up));
    for (const tick of up.ticks || []) {
      this.line(x, py(tick), x + w, py(tick), { stroke: "grid", width: 1.5 });
      this.text(x - 12, py(tick) + 7, (up.format || String)(tick), { size: 20, anchor: "end", fill: "muted" });
    }
    this.line(x, y + h, x + w, y + h, { stroke: "axis" });
    for (const tick of across.ticks || []) this.text(px(tick), y + h + 28, (across.format || String)(tick), { size: 20, anchor: "middle", fill: "muted" });
    if (across.title) this.text(x + w / 2, y + h + 60, across.title, { size: 20, anchor: "middle", fill: "muted" });
    if (up.title) this.text(x, y - 16, up.title, { size: 20, fill: "muted" });
    return { px, py };
  }

  circle(x, y, r, { fill = "s1", stroke = null } = {}) {
    return svg("circle", { cx: x, cy: y, r, class: `f-${fill}${stroke ? ` s-${stroke}` : ""}`, "stroke-width": stroke ? 2 : 0 }, this.g);
  }
}

const node = (thing) => thing.g || thing; // a chip, a stage, or an element

// How visible something is, from 0 to 1.
export function show(thing, amount) {
  node(thing).setAttribute("opacity", clamp(amount).toFixed(3));
}

// Where something made at a place is now. For a chip or a group: its corner.
export function place(thing, x, y) {
  node(thing).setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
  if (thing.g) Object.assign(thing, { x, y });
}

// Draw a path up to a part of its length, from 0 to 1.
export function trace(path, part) {
  path.setAttribute("stroke-dasharray", `${(path.total * clamp(part)).toFixed(1)} ${(path.total + 10).toFixed(1)}`);
}

export function set(element, attributes) {
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, typeof value === "number" ? value.toFixed(2) : value);
}
