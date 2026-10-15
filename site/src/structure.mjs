// The shape of the site, which is the shape of the repository: ten stages in
// a chain, and side topics that hang off the stage they change. The header,
// the landing page and the previous/next links are all drawn from this.

export const REPO = "https://github.com/changkun/minllm";

export const stages = [
  { n: "1", slug: "01-tokenizer", short: "Tokenizer" },
  { n: "2", slug: "02-model", short: "Model", leaves: [{ n: "2a", slug: "02a-moe", short: "Experts" }, { n: "2b", slug: "02b-context", short: "Context" }, { n: "2c", slug: "02c-latent-attention", short: "Latent attention" }] },
  { n: "3", slug: "03-pretraining", short: "Pretraining", leaves: [{ n: "3a", slug: "03a-distillation", short: "Distillation" }, { n: "3b", slug: "03b-data", short: "Data" }, { n: "3c", slug: "03c-scaling", short: "Scaling" }] },
  { n: "4", slug: "04-inference", short: "Inference", leaves: [{ n: "4a", slug: "04a-structured-output", short: "Structured output" }, { n: "4b", slug: "04b-speculative", short: "Speculative" }, { n: "4c", slug: "04c-continuous-batching", short: "Batching" }] },
  { n: "5", slug: "05-pretrained", short: "Weights" },
  { n: "6", slug: "06-finetuning", short: "Fine-tuning", leaves: [{ n: "6a", slug: "06a-tool-calling", short: "Tool calling" }] },
  { n: "7", slug: "07-evaluation", short: "Evaluation", leaves: [{ n: "7a", slug: "07a-multiple-choice", short: "Multiple choice" }, { n: "7b", slug: "07b-judge", short: "Judge" }] },
  { n: "8", slug: "08-rlhf", short: "RLHF", leaves: [{ n: "8a", slug: "08a-safety", short: "Safety" }, { n: "8b", slug: "08b-dpo", short: "DPO" }] },
  { n: "9", slug: "09-rlvr", short: "RLVR", leaves: [{ n: "9a", slug: "09a-self-training", short: "Self-training" }, { n: "9b", slug: "09b-tool-rl", short: "Tool RL" }, { n: "9c", slug: "09c-reasoning", short: "Reasoning" }] },
  { n: "10", slug: "10-stack", short: "Export", leaves: [{ n: "10a", slug: "10a-quantization", short: "Quantization" }] },
];

// Every page in reading order: a stage, then its side topics.
export const pages = stages.flatMap((s) => [{ ...s, kind: "stage" }, ...(s.leaves || []).map((l) => ({ ...l, kind: "leaf", parent: s }))]);

// Where the first four stages end and the post-training ones begin.
export const phases = [
  { label: "From scratch: a 14M model, trained here", from: 0, to: 4 },
  { label: "Post-training: that model, and an open 0.5B one", from: 4, to: 10 },
];
