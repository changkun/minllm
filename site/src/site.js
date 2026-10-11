// Loads the interactive figures of a page, and marks the section being read
// in the list beside the note.

for (const lab of document.querySelectorAll("[data-lab]")) {
  const body = lab.querySelector(".lab-body");
  import(`/assets/labs/${lab.dataset.lab}.js`)
    .then((module) => {
      body.textContent = "";
      return module.default(body);
    })
    .catch((error) => {
      body.textContent = `This figure could not be loaded: ${error.message}`;
    });
}

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
