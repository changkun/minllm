// What the interactive figures share: making elements, reading the logged
// runs, and a few formats. Colors come from the page's stylesheet by class,
// f-* to fill and s-* to stroke, so a figure follows the light or dark theme.

const SVG = "http://www.w3.org/2000/svg";
export const REPO = "https://github.com/changkun/minllm";

export function svg(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  parent?.append(node);
  return node;
}

export function html(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  node.append(...children);
  return node;
}

async function get(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response;
}

export const json = async (path) => (await get(path)).json();
export const jsonl = async (path) => (await (await get(path)).text()).trim().split("\n").map((line) => JSON.parse(line));

// A slider with its label and its current value shown.
export function slider(label, { min, max, step = 1, value, format = String }, onchange) {
  const output = html("output", { text: format(value) });
  const input = html("input", { type: "range", min, max, step, value });
  input.addEventListener("input", () => {
    output.textContent = format(Number(input.value));
    onchange(Number(input.value));
  });
  return html("label", {}, label, input, output);
}

// A row of buttons of which one is pressed.
export function choice(label, options, value, onchange) {
  const buttons = options.map(([name, v]) => {
    const button = html("button", { type: "button", text: name, "aria-pressed": String(v === value) });
    button.addEventListener("click", () => {
      for (const b of buttons) b.setAttribute("aria-pressed", String(b === button));
      onchange(v);
    });
    return button;
  });
  return html("label", {}, label, ...buttons);
}

export const percent = (v, digits = 0) => `${(100 * v).toFixed(digits)}%`;
