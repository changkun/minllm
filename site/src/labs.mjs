// The interactive figures: which note each belongs to, and where it goes:
// after a figure, or before a heading, at the end of the section above it. A lab is a module in src/labs/ whose default export fills an
// element, reading what it shows from the logged runs.

export const labs = [
  { note: "01-tokenizer", before: "three-things-the-real-version-adds", name: "tokenizer", title: "Tokenize your own text" },
  { note: "02-model", after: "model-attention.svg", name: "attention", title: "See what a token reads" },
  { note: "02a-moe", after: "moe-routing.svg", name: "routing", title: "Look into each expert" },
  { note: "03-pretraining", after: "pretraining-loss.svg", name: "run", title: "Read the run off, step by step" },
  { note: "03a-distillation", after: "distill-targets.svg", name: "temperature", title: "Turn the teacher's temperature" },
  { note: "04-inference", before: "the-cache", name: "sampling", title: "Turn the three knobs of sampling" },
  { note: "04-inference", after: "inference-work.svg", name: "cache", title: "Count the work, token by token" },
  { note: "10a-quantization", after: "quant-levels.svg", name: "quantize", title: "Round this row yourself" },
];
