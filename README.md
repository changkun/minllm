# minllm

A minimal pipeline for making a language model, small enough to run on a
laptop and written to be read. It goes from raw text to a tokenizer, a
pretrained model, an inference engine, a fine-tuned chat model, and a model
post-trained with reinforcement learning, first from human preferences (RLHF)
and then from verifiable rewards (RLVR).

The only machine learning dependency is PyTorch, for tensors and automatic
differentiation. Everything on top of it is written here, in a few hundred
lines per stage: the tokenizer, the model and the training loop so far, and
the inference engine, the fine-tuning and the RL algorithms as the stages
below get built. There is no `transformers`, `tokenizers`, `peft`, `trl` or
`vllm` underneath.

## Stages

Each stage has a note that explains it, code that implements it, and tests
that check what the note claims.

| | Stage | What gets built | Status |
|---|---|---|---|
| 1 | [Tokenizer](docs/01-tokenizer.md) | Byte-level BPE: training, encoding, decoding, special tokens | done |
| 2 | [Model](docs/02-model.md) | Decoder-only transformer with RMSNorm, RoPE, SwiGLU and grouped-query attention | done |
| 3 | [Pretraining](docs/03-pretraining.md) | Data pipeline, AdamW, learning rate schedule, checkpoints: a model that writes stories | done |
| 4 | Inference | KV cache, sampling strategies, batching, stop conditions, a small server | next |
| 5 | Pretrained weights | Load an open 0.5B model into the same model code and match its reference outputs | planned |
| 6 | Fine-tuning | Chat format, loss masking, full fine-tuning and LoRA | planned |
| 7 | Evaluation | Batched generation over a task, scoring, and comparing two models | planned |
| 8 | RLHF | Preference data, a reward model trained on pairwise comparisons, and three ways to use it: rejection sampling, policy gradients with a KL penalty (PPO), and DPO, which skips the reward model | planned |
| 9 | RLVR | The reward model replaced by a program that checks the answer: rollouts, verifiers, policy-gradient updates with GRPO | planned |
| 10 | The practitioner's stack | Export the model and run it in vLLM, llama.cpp and MLX; map each stage to the library used for it in practice | planned |

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/changkun/minllm && cd minllm
uv sync
uv run pytest

uv run scripts/download_data.py      # TinyStories: 22 MB validation, 512 MB of training text
uv run scripts/train_tokenizer.py    # stage 1, 5 seconds
uv run scripts/encode_data.py        # text to token ids, 35 seconds
uv run scripts/train.py              # stage 3, 20 minutes
uv run scripts/sample.py             # read what it writes
```

Timings are from a MacBook Pro with an M5 Pro and 64 GB of memory, using the
Apple GPU. The code picks an NVIDIA GPU, the Apple GPU or the CPU, in that
order; set `MINLLM_DEVICE=cpu` to override. CUDA has not been tested yet.

## Two models

The stages use two models, both running on the code in
[`minllm/model.py`](minllm/model.py).

The first is trained from scratch here: 14M parameters, a vocabulary of 4,096
tokens, trained on [TinyStories](https://arxiv.org/abs/2305.07759). After 20
minutes of training it writes fluent children's stories. It carries stages 1
to 4 and stays the testbed afterwards: each post-training method is tried on
it first, where an experiment takes minutes.

The second is an open pretrained model of about 0.5B parameters from the Qwen
family, loaded into the same code from stage 5 on. Post-training on a real
task needs a model that already knows something: RLVR improves a model on a
task it sometimes gets right, and a model trained on a laptop from children's
stories does not solve arithmetic word problems at any rate worth reinforcing.
A 0.5B model gets a meaningful share of them right, and at that size sampling
and gradient steps should both still be practical on a laptop, which stage 5
will measure.

## Layout

```
minllm/      the library, one concept per file
scripts/     one entry point per step of the pipeline
tests/       checks for the claims the notes make
docs/        one note per stage
runs/        logs of the reference runs quoted in the notes
data/, out/  downloads and checkpoints, not in git
```

## Related work

- [nanoGPT](https://github.com/karpathy/nanoGPT) and
  [minbpe](https://github.com/karpathy/minbpe) by Andrej Karpathy set the
  standard for teaching by small, complete code. Stages 1 to 3 follow their
  shape with a current architecture.
- [nanochat](https://github.com/karpathy/nanochat), also by Karpathy, is a
  full pipeline from tokenizer to chat UI, sized for a rented multi-GPU node.
  minllm is sized for a laptop and puts more weight on the post-training
  stages.
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian
  Raschka covers pretraining and fine-tuning in book form.
- [The RLHF Book](https://rlhfbook.com/) by Nathan Lambert is the reference
  for the post-training stages, 6 to 9, which follow its order: instruction
  tuning, reward models, rejection sampling, policy gradients, direct
  alignment, and then RLVR.

## License

[MIT](LICENSE)
