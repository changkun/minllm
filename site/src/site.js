// The page's own behavior: the theme, the interactive figures, figures that
// open large, and the section being read in the list beside a note.

// ------------------------------------------------------------------ theme

const root = document.documentElement;
const names = { light: "Light", dark: "Dark", auto: "Auto" };
const system = matchMedia("(prefers-color-scheme: dark)");

function applyTheme(choice) {
  root.dataset.choice = choice;
  root.dataset.theme = choice === "dark" || (choice === "auto" && system.matches) ? "dark" : "light";
  for (const name of document.querySelectorAll(".theme-name")) name.textContent = names[choice];
}
applyTheme(root.dataset.choice || "light");
system.addEventListener("change", () => applyTheme(root.dataset.choice));
for (const button of document.querySelectorAll(".theme")) {
  button.addEventListener("click", () => {
    const next = { light: "dark", dark: "auto", auto: "light" }[root.dataset.choice];
    try {
      localStorage.setItem("theme", next);
    } catch {}
    applyTheme(next);
  });
}

// ---------------------------------------------------- interactive figures

for (const lab of document.querySelectorAll("[data-lab]")) {
  const body = lab.querySelector(".lab-body");
  import(new URL(`./labs/${lab.dataset.lab}.js`, import.meta.url)) // beside this file, whatever its address
    .then((module) => {
      body.textContent = "";
      return module.default(body);
    })
    .catch((error) => {
      body.textContent = `This figure could not be loaded: ${error.message}`;
    });
  const large = lab.querySelector(".lab-large");
  large?.addEventListener("click", () => {
    const on = lab.classList.toggle("large");
    large.textContent = on ? "Close" : "Enlarge";
    document.body.style.overflow = on ? "hidden" : "";
  });
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") document.querySelector(".lab.large .lab-large")?.click();
});

// ------------------------------------------------- a figure, opened large

const viewer = document.querySelector(".viewer");
if (viewer) {
  const stage = viewer.querySelector(".viewer-stage");
  const content = viewer.querySelector(".viewer-content");
  const view = { x: 0, y: 0, scale: 1, width: 1, height: 1 };
  const place = () => (content.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`);
  // Zoom by a factor, keeping the point (cx, cy) of the stage where it is.
  function zoom(factor, cx = stage.clientWidth / 2, cy = stage.clientHeight / 2) {
    const scale = Math.min(12, Math.max(0.2, view.scale * factor));
    view.x = cx - ((cx - view.x) * scale) / view.scale;
    view.y = cy - ((cy - view.y) * scale) / view.scale;
    view.scale = scale;
    place();
  }
  function fit() {
    view.scale = Math.min(stage.clientWidth / view.width, stage.clientHeight / view.height) * 0.96;
    view.x = (stage.clientWidth - view.width * view.scale) / 2;
    view.y = (stage.clientHeight - view.height * view.scale) / 2;
    place();
  }
  // The button under a figure opens it, and so does the figure itself.
  for (const target of document.querySelectorAll(".figure .zoom, .figure .fig")) {
    target.addEventListener("click", () => {
      const figure = target.closest(".figure").querySelector("svg").cloneNode(true);
      const [, , w, h] = figure.getAttribute("viewBox").split(" ").map(Number);
      Object.assign(view, { width: w, height: h });
      figure.removeAttribute("style");
      figure.setAttribute("width", w);
      figure.setAttribute("height", h);
      content.replaceChildren(figure);
      viewer.showModal();
      fit();
    });
  }
  viewer.addEventListener("click", (event) => {
    const action = event.target.dataset?.zoom;
    if (action === "in") zoom(1.4);
    if (action === "out") zoom(1 / 1.4);
    if (action === "fit") fit();
    if (action === "close") viewer.close();
  });
  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    const box = stage.getBoundingClientRect();
    zoom(Math.exp(-event.deltaY * (event.ctrlKey ? 0.01 : 0.0015)), event.clientX - box.left, event.clientY - box.top);
  }, { passive: false });
  stage.addEventListener("dblclick", (event) => {
    const box = stage.getBoundingClientRect();
    zoom(1.8, event.clientX - box.left, event.clientY - box.top);
  });
  // One finger or the mouse drags, two fingers pinch.
  const pointers = new Map();
  let pinch = 0;
  stage.addEventListener("pointerdown", (event) => {
    stage.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, [event.clientX, event.clientY]);
    pinch = 0;
  });
  stage.addEventListener("pointermove", (event) => {
    if (!pointers.has(event.pointerId)) return;
    const [px, py] = pointers.get(event.pointerId);
    pointers.set(event.pointerId, [event.clientX, event.clientY]);
    if (pointers.size === 1) {
      view.x += event.clientX - px;
      view.y += event.clientY - py;
      place();
    } else if (pointers.size === 2) {
      const [[ax, ay], [bx, by]] = [...pointers.values()];
      const distance = Math.hypot(ax - bx, ay - by);
      const box = stage.getBoundingClientRect();
      if (pinch) zoom(distance / pinch, (ax + bx) / 2 - box.left, (ay + by) / 2 - box.top);
      pinch = distance;
    }
  });
  for (const type of ["pointerup", "pointercancel"]) {
    stage.addEventListener(type, (event) => {
      pointers.delete(event.pointerId);
      pinch = 0;
    });
  }
}

// ------------------------------------------------- the section being read

const links = new Map([...document.querySelectorAll(".toc a")].map((a) => [a.getAttribute("href").slice(1), a]));
if (links.size) {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          for (const a of links.values()) a.classList.remove("reading");
          links.get(entry.target.id)?.classList.add("reading");
        }
      }
    },
    { rootMargin: "-80px 0px -70% 0px" },
  );
  for (const id of links.keys()) observer.observe(document.getElementById(id));
}
