// A line chart that can be read off: move over it and a rule shows the
// values of every series at that place.

import { html, svg } from "./kit.js";

// series: [{ name, color, points: [[x, y], ...], dots }]. x and y: { domain: [lo, hi], ticks, format, log, title }.
export function chart(el, { width = 720, height = 260, series, x, y, onhover }) {
  const pad = { left: 52, right: 16, top: 12, bottom: x.title ? 40 : 24 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  const scale = (v, { domain, log }) => (log ? (Math.log(v) - Math.log(domain[0])) / (Math.log(domain[1]) - Math.log(domain[0])) : (v - domain[0]) / (domain[1] - domain[0]));
  const px = (v) => pad.left + w * scale(v, x);
  const py = (v) => pad.top + h * (1 - scale(v, y));
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  for (const t of y.ticks) {
    svg("line", { x1: pad.left, y1: py(t), x2: pad.left + w, y2: py(t), class: "s-line" }, root);
    svg("text", { x: pad.left - 8, y: py(t) + 4, "text-anchor": "end", class: "f-muted" }, root).textContent = (y.format || String)(t);
  }
  svg("line", { x1: pad.left, y1: pad.top + h + 0.5, x2: pad.left + w, y2: pad.top + h + 0.5, class: "s-axis" }, root);
  for (const t of x.ticks) {
    svg("line", { x1: px(t), y1: pad.top + h, x2: px(t), y2: pad.top + h + 4, class: "s-axis" }, root);
    svg("text", { x: px(t), y: pad.top + h + 17, "text-anchor": "middle", class: "f-muted" }, root).textContent = (x.format || String)(t);
  }
  if (x.title) svg("text", { x: pad.left + w / 2, y: height - 4, "text-anchor": "middle", class: "f-muted" }, root).textContent = x.title;
  for (const s of series) {
    if (!s.dots) svg("path", { d: "M" + s.points.map(([a, b]) => `${px(a).toFixed(1)} ${py(b).toFixed(1)}`).join(" L"), fill: "none", class: `s-${s.color}`, "stroke-width": 2, "stroke-linejoin": "round" }, root);
    else for (const [a, b] of s.points) svg("circle", { cx: px(a), cy: py(b), r: 3.5, class: `f-${s.color} s-surface`, "stroke-width": 1.5 }, root);
  }
  // the rule that follows the pointer, and a marker on every series
  const rule = svg("line", { y1: pad.top, y2: pad.top + h, class: "s-ink", "stroke-width": 1, visibility: "hidden" }, root);
  const marks = series.map((s) => svg("circle", { r: 4.5, class: `f-${s.color} s-surface`, "stroke-width": 2, visibility: "hidden" }, root));
  const nearest = (points, v) => points.reduce((best, p) => (Math.abs(p[0] - v) < Math.abs(best[0] - v) ? p : best));
  function show(v) {
    v = Math.min(x.domain[1], Math.max(x.domain[0], v));
    rule.setAttribute("x1", px(v));
    rule.setAttribute("x2", px(v));
    rule.setAttribute("visibility", "visible");
    const at = series.map((s, i) => {
      const p = nearest(s.points, v);
      marks[i].setAttribute("cx", px(p[0]));
      marks[i].setAttribute("cy", py(p[1]));
      marks[i].setAttribute("visibility", "visible");
      return p;
    });
    onhover?.(v, at);
  }
  root.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const sx = ((event.clientX - box.left) / box.width) * width;
    const t = (sx - pad.left) / w;
    show(x.log ? Math.exp(Math.log(x.domain[0]) + t * (Math.log(x.domain[1]) - Math.log(x.domain[0]))) : x.domain[0] + t * (x.domain[1] - x.domain[0]));
  });
  const legend = html("p", { class: "legend" }, ...series.map((s) => html("span", {}, html("i", { class: `key key-${s.color}${s.dots ? " key-dot" : ""}` }), s.name)));
  el.append(legend, root);
  return { show };
}
