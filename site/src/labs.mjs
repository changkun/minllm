// The interactive figures: which note each belongs to, and where it goes:
// after a figure, or before a heading, at the end of the section above it. A lab is a module in src/labs/ whose default export fills an
// element, reading what it shows from the logged runs.

export const labs = [
  { note: "01-tokenizer", before: "three-things-the-real-version-adds", name: "tokenizer", title: "Tokenize your own text" },
  { note: "10a-quantization", after: "quant-levels.svg", name: "quantize", title: "Round this row yourself" },
];
