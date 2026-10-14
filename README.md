# minllm

A minimal pipeline for making a language model, small enough to run on a
laptop and written to be read. It goes from raw text to a tokenizer, a
pretrained model, an inference engine, a fine-tuned chat model, and a model
post-trained with reinforcement learning, first from human preferences (RLHF)
and then from verifiable rewards (RLVR), and ends with that model in the
format that other tools run.

These ten stages are the pipeline, and each builds on the one before. Side
topics branch off a single stage for whoever wants to go further there: a
mixture of experts in place of the model's MLP, and windows longer than it
was trained on; distillation from a larger model, text from the web and
scaling laws for pretraining; structured output and speculative decoding in
inference; tool calling in fine-tuning; multiple-choice benchmarks and model
judges in evaluation; refusals in RLHF; self-training in place of
reinforcement learning, reinforcement learning with a tool, and reasoning;
and quantization of the exported file. The pipeline does not depend on them.

![The ten stages of the pipeline, and the side topics that branch off them](docs/figures/pipeline.svg)

The notes are also a website, [minllm.latere.site](https://minllm.latere.site).
Every chapter opens there with a film of about a minute, the model of stage 3
runs in the browser, the figures can be tried, and the code can be read
beside the notes that explain it. The films are also video files, attached to
the [releases](https://github.com/changkun/minllm/releases).

The only machine learning dependency is PyTorch, for tensors and automatic
differentiation. Everything on top of it is written here, in a few hundred
lines per stage: the tokenizer, the model, the training loop, the inference
engine, fine-tuning with LoRA, the evaluation, and the RL algorithms. There
is no `transformers`, `tokenizers`, `peft`, `trl` or `vllm` underneath.

## Stages

Each stage has a note that explains it, code that implements it, and tests
that check what the note claims. Every note ends with the papers, reports and
projects that introduced what it builds.

| | Stage | What gets built |
|---|---|---|
| 1 | [Tokenizer](docs/01-tokenizer.md) | Byte-level BPE: training, encoding, decoding, special tokens |
| 2 | [Model](docs/02-model.md) | Decoder-only transformer with RMSNorm, RoPE, SwiGLU and grouped-query attention |
| 3 | [Pretraining](docs/03-pretraining.md) | Data pipeline, AdamW, learning rate schedule, checkpoints: a model that writes stories |
| 4 | [Inference](docs/04-inference.md) | KV cache, sampling strategies, batches of unequal prompts, stop conditions |
| 5 | [Pretrained weights](docs/05-pretrained.md) | Qwen2.5-0.5B-Instruct loaded into the same model code: a safetensors reader, its tokenizer, the chat format, and tests against the reference implementation |
| 6 | [Fine-tuning](docs/06-finetuning.md) | Conversations as data, a loss on the replies only, full fine-tuning and LoRA |
| 7 | [Evaluation](docs/07-evaluation.md) | Scoring replies to held-out prompts, confidence intervals, and paired comparison of two models |
| 8 | [RLHF](docs/08-rlhf.md) | Preference comparisons, a reward model, and three ways to use it: rejection sampling, PPO with a KL penalty, and DPO, which skips the reward model |
| 9 | [RLVR](docs/09-rlvr.md) | The reward model replaced by a program that checks the answer: rollouts, verifiers, and GRPO |
| 10 | [The practitioner's stack](docs/10-stack.md) | Export to the format other tools read, a check against transformers, a run in MLX, an OpenAI-compatible server, and a map from each stage to the library used for it |

### Side topics

Each one changes what a single stage built, with a note, code and tests of
its own.

| | Side topic | Branches from | What it adds |
|---|---|---|---|
| 2a | [Mixture of experts](docs/02a-moe.md) | 2, the model | A router and eight expert MLPs in place of the MLP of every block, the load-balancing loss, three ways to hand tokens to experts, and a look at what the routers learn |
| 2b | [Longer context](docs/02b-context.md) | 2, the model | Windows longer than those trained on: what the rotary table does past them, stretched by interpolation, NTK and YaRN with and without training, measured on 3b's web model by the loss along long documents and by finding a passage again |
| 3a | [Distillation](docs/03a-distillation.md) | 3, pretraining | A second model as a teacher in the training loop: the student learns from the teacher's whole prediction at every position, and not only from the token that came next |
| 3b | [Data preparation](docs/03b-data.md) | 3, pretraining | From pages of Common Crawl to text worth training on: the crawl's language tags, rules for lines and pages, placeholders for addresses, and copies found by hash and by MinHash, with the model of stage 3 trained on each version |
| 3c | [Scaling laws](docs/03c-scaling.md) | 3, pretraining | The model of stage 3 at eight sizes and four compute budgets: the best size for each budget, how many tokens per parameter, and a law fitted on small runs that predicts larger ones |
| 4a | [Structured output](docs/04a-structured-output.md) | 4, inference | Constrained decoding: a JSON schema as a regular expression, its automaton over bytes walked by the whole vocabulary, and a mask at every step that keeps a reply in the form and lets it finish within its length |
| 4b | [Speculative decoding](docs/04b-speculative.md) | 4, inference | A cheap draft of the next tokens checked by the model in one pass, with output exactly the model's own: a smaller model or the text itself as the draft, and when it pays on a laptop |
| 6a | [Tool calling](docs/06a-tool-calling.md) | 6, fine-tuning | Fine-tuning on conversations in which the model calls a calculator and reads its result, made from GSM8K's own calculator marks, and the loop that runs the calls |
| 7a | [Multiple choice](docs/07a-multiple-choice.md) | 7, evaluation | A benchmark scored without letting the model write: the probability it gives each answer letter of MMLU's questions, and what the prompt, the order of the options and the way of reading the answer do to the score |
| 7b | [Evaluation by a judge](docs/07b-judge.md) | 7, evaluation | A language model as the judge of other replies, its verdict read as a probability, measured against GSM8K's right answers and stage 8's annotator, with its biases and their remedies |
| 8a | [Safety](docs/08a-safety.md) | 8, RLHF | Refusals of unsafe requests and of safe ones that sound alike, what fine-tuning on math does to them, and refusals mixed back into fine-tuning |
| 9a | [Self-training](docs/09a-self-training.md) | 9, RLVR | The verifier without the reinforcement learning: sample several replies, keep the ones that are right, fine-tune on them, and repeat |
| 9b | [Reinforcement learning with a tool](docs/09b-tool-rl.md) | 9, RLVR | GRPO on conversations in which the calculator of 6a runs while the replies are sampled, trained on the model's own tokens only |
| 9c | [Reasoning](docs/09c-reasoning.md) | 9, RLVR | Thinking longer to be right more often: reinforcement learning from the published model with room to think, votes over samples and budgets for thinking, and a reasoning model's thinking distilled into the 0.5B model |
| 10a | [Quantization](docs/10a-quantization.md) | 10, the export | Weights as 4-bit or 8-bit codes with a scale and an offset per group, what each bit width costs in loss and on the task, and a packed file that MLX runs as it is |

## What came out

The numbers are measured on one laptop, and the runs behind them are logged
in [`runs/`](runs/). The notes explain each one.

| Stage | Result |
|---|---|
| 3 | A 14M model trained from scratch writes coherent children's stories after 20 minutes. Compiling the model and 16-bit arithmetic make a training step about 4 times faster. |
| 4 | A KV cache is 16 times faster for 64 sequences of 255 tokens, and no faster at all for a single one, where overhead dominates. |
| 5 | The same model code loads Qwen2.5-0.5B-Instruct: identical token ids, logits within 0.001 of the reference, identical greedy output. |
| 6, 7 | Fine-tuning teaches the story model to follow an instruction: requested words used rise from 6.5% to 39.6%. Fine-tuning the 0.5B model on GSM8K's solutions lowers its loss and its accuracy, from 46.8% to 36.6%: it learned to imitate terse solutions. A token limit set too low had hidden five of those ten points. |
| 8 | A reward model trained on 3,500 noisy comparisons orders 84% of held-out pairs the way the annotator's hidden taste does. PPO with a KL penalty raises that taste from +0.21 to +1.27, and DPO to +1.06 in 80 seconds. Without the penalty the taste is highest and the stories stop making sense. |
| 9 | GRPO takes the story model from 40% to 94% of requested words in 15 minutes, by repeating them until the token limit. A stricter verifier gets endings and the same nonsense, and a KL penalty gets stories. On GSM8K, 22 minutes of GRPO add 5.9 points on the test set (95% interval 3.2 to 8.5). |
| 10 | The from-scratch model, exported, runs in MLX and writes the same story token for token. |

Two lessons recur. A loss or a reward is satisfied by whatever satisfies
it, which is not always what was meant: stages 7, 8 and 9 each show a
different case. And the way to notice is to read what the model wrote.

From the side topics:

| Side topic | Result |
|---|---|
| 2a | A mixture of eight experts per block, two of them active, has the dense model's arithmetic per token and three times its parameters. Step for step it learns more: a validation loss of 1.423 against 1.465 after 6,000 steps. Minute for minute it learns less: 1.491 against 1.465 after 20 minutes. Without the load-balancing loss, 24 of its 64 experts get less than 1% of the work. On this GPU it generates at a sixth of the speed of a dense model with the same parameters, and the gap closes as the models get wider. Exported, it loads as a Mixtral and gives the same logits. |
| 2b | Read past the 256 tokens it was trained on, 3b's web model breaks at once: its loss rises from 3.20 to 4.36. Stretched by YaRN, without a step of training, it keeps its loss level to 1,024 tokens, or to 2,048 when stretched by 8, and finds a passage it saw 1,024 tokens earlier. After 500 steps on windows of 1,024, every stretch is level to 1,024, and they differ in how far back they reach: a passage 512 tokens back is worth 1.01 to YaRN and 0.39 to the model trained without a stretch. Trained on 1,024 from the start, the model copies more over short distances and ends with a higher loss. |
| 3a | Taught by a 42M model of the same kind, the 14M model of stage 3 ends at a validation loss of 1.418 where the text alone gave 1.465, and the teacher has 1.387: 60% of the way, with no change to the model. Taught by the teacher alone, without ever being told which token came next, it reaches 1.445. A step with a teacher takes 1.7 times as long, and given the same 20 minutes the text alone still wins, 1.465 against 1.481. |
| 3b | Of 201,520 pages of Common Crawl, 1.4 billion characters, the usual rules leave a tenth. Less than half is tagged English, and of that only 31% is sentences and not menus. The 14M model trained on the raw pages learns to write menus, and has the lowest loss of all on its own held-out text. On held-out prose it has 3.52 where the model trained on the cleaned text has 3.30, and on stories that neither saw 3.35 against 3.03. Removing copies, a tenth of what the rules passed, adds 0.02, at the edge of what one run can show. |
| 3c | Trained at eight sizes from 0.36M to 30M parameters and four budgets from 3e14 to 1e16 operations, the model of stage 3 does best on 16 to 23 tokens per parameter, near Chinchilla's 20. A law L(N, D) fitted on the three smaller budgets predicts every run at the largest within 0.014. Stage 3's 14M model on 98M tokens is larger than the law would choose, 9.8M on 142M tokens, and the law puts the cost at 0.008 in loss. |
| 4a | Asked for its answer as JSON, the 0.5B model gives JSON of the form in 18.4% of GSM8K's problems; held to it by constrained decoding, in all of them, at a cost of 3% in speed. What is right depends on the form: 7.6% with the answer alone, 29.5% with the reasoning inside the object, 34.6% with free text first, against 46.8% in its own format. Held to a form that ends where it does not want to stop, the model runs on into its token limit. |
| 4b | Drafted by the 0.5B model, Qwen2.5-7B writes 2.26 times as fast on this laptop, 38.2 tokens a second against 16.9, and Qwen2.5-1.5B only 1.18 times: a round of draft and check has to cost less than it yields, and a draft a third of the model's size costs nearly as much as it yields. Prompt lookup, which costs almost nothing, makes either 1.4 to 1.5 times as fast on GSM8K. The story models of 3a are slower with any draft. In 16-bit floats on the GPU, greedy replies with a draft part from the model's own at near ties; in 32-bit floats, or on the CPU, they are the same. |
| 6a | Told about a calculator as its chat template and tokenizer write it, the published model calls it in 5.9% of GSM8K's problems, or 0.5% with the template's markers spelled out as characters, and solves 5.3 points fewer than without the tool. Fine-tuned with stage 6's recipe on the same solutions with every calculation turned into a call, it calls in 98.6%, never malformed, and solves 37.5% against 36.6% without the tool, within noise: its mistakes are in what it calculates, not how. With every call failing it solves 10.4%. |
| 7a | Asked the way the authors of MMLU asked, the 0.5B model answers 46.9% of the 14,042 questions in ten minutes, where its makers report 47.5% for its base model. Put into a chat turn, the same questions score 38.9%: there the model wants to begin a sentence, and only a third of its probability is on a letter at all. Moving the options round leaves the score where it was and changes which option the model picks on more than half of the questions, and it answers 27.6% right in every order. At 4 bits it loses 2.8 points here, against a third of its score on GSM8K. |
| 7b | As judges of GSM8K solutions, the 0.5B model approves nearly every one and the 1.5B model 77% of the wrong ones; told the right answer, the 1.5B model agrees with the truth 95.4% of the time. As judges of stories, both decide by position: the 1.5B model picks the second story 98% of the time, whatever is in it. Stage 8's 14M reward model agrees with the annotator 77.4% of the time, where the order-averaged 1.5B judge reaches 56.2%. |
| 8a | The published model refuses 79.0% of XSTest's unsafe requests and 34.4% of its safe ones. Stage 6's fine-tuning on math, which never mentioned either, leaves 28.5% and 1.2%. With 229 refusals mixed in as 3% of the fine-tuning data, it refuses 90.5% and 40.0%, with no measurable cost on GSM8K. |
| 9a | Keeping the story model's right stories and fine-tuning on them, for six rounds, takes it from 39.6% to 68.7% of requested words in 13 minutes. GRPO reached 93.6% from the same number of stories, by writing nonsense. The self-trained stories still end and read as stories, with no penalty to hold them, and match the leashed GRPO run. On GSM8K two rounds do nothing, 35.4% against 36.6% before, where GRPO gained 5.9 points from half as many replies: keeping the right replies does not say what made them right. |
| 9b | Stage 9's GRPO from 6a's model, with the calculator running in every rollout, did not move the score: 37.5% to 38.2% of GSM8K's test problems, within noise, where the same 60 iterations from stage 6's model added 5.9 points. The model learned to call less, 3.11 times a problem against 3.38, and still solves 9.7% when every call fails. Stage 9's gain was not arithmetic that the calculator had already done: of the 200 problems it turned right, 28 had a calculation written wrong before. |
| 9c | GRPO from the published model, with room for 512 tokens, took it from 46.8% to 49.1% of GSM8K's test problems, at the edge of noise, and its replies did not grow longer. The most common answer of 16 samples is right on 61.8% of the first 500 problems, against 47.0% for its greedy reply, and at least one of the 16 on 83.8%. Fine-tuned on 601 chains of thought of a 1.5B reasoning model, the 0.5B model learned to think at length and not when to stop: it solves 36.2%, 10.5 points fewer than before, and 448 of its 1,319 replies run into the limit in a loop. Cut at 2,048 tokens and made to answer, it solves 40.6% of the first 500. |
| 10a | Stored at 8 bits per weight, the 0.5B model is half the size and solves as many GSM8K problems as before (47.3% against 46.8%), and at 6 bits still as many. At 4 bits, in 278 MB, it solves 31.9%, while its replies read as fluently as ever and its held-out loss only rises from 0.83 to 0.92. This repository's own code runs the packed model 14 times slower. MLX reads the same file and runs it 2.1 times faster, in a third of the memory. |

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

uv run scripts/download_qwen.py      # stage 5: an open 0.5B model, 1 GB
uv run scripts/chat.py               # talk to it, on this repository's model code
```

The later stages build on these. Each note lists its commands, and in short:

```sh
uv run scripts/finetune.py --task words                 # stage 6, 11 minutes
uv run scripts/finetune.py --task gsm8k --lora 16       # 10 minutes
uv run scripts/evaluate.py --model out/sft-words --task words     # stage 7

uv run scripts/make_preferences.py                      # stage 8
uv run scripts/train_reward.py
uv run scripts/ppo.py                                   # 7 minutes
uv run scripts/dpo.py

uv run scripts/grpo.py --task words                     # stage 9, 15 minutes
uv run scripts/grpo.py --task gsm8k --max-new-tokens 256          # 22 minutes

uv run scripts/export.py --model out/sft-words          # stage 10
uv run scripts/serve.py --model out/sft-words
```

And four of the side topics:

```sh
uv run scripts/train.py --n-experts 8 --mlp-dim 512 --out out/moe  # 2a, 31 minutes
uv run scripts/analyze_moe.py

uv run scripts/train.py --mlp-dim 4096 --out out/wide              # 3a: a teacher, 38 minutes
uv run scripts/train.py --teacher out/wide --out out/distilled     # and its student, 34 minutes

uv run scripts/self_train.py --task words               # 9a, 13 minutes
uv run scripts/self_train.py --task gsm8k

uv run scripts/quantize.py --model qwen --bits 4        # 10a: 988 MB become 278
uv run scripts/evaluate.py --model out/qwen-q4 --task gsm8k
```

Timings are from a MacBook Pro with an M5 Pro and 64 GB of memory, using the
Apple GPU. The code picks an NVIDIA GPU, the Apple GPU or the CPU, in that
order; set `MINLLM_DEVICE=cpu` to override. CUDA has not been tested. On an
Apple GPU the process is capped at half of the recommended memory, so that a
batch that is too large fails instead of swapping the machine.

## Two models

The stages use two models, both running on the code in
[`minllm/model.py`](minllm/model.py).

The first is trained from scratch here: 14M parameters, a vocabulary of 4,096
tokens, trained on [TinyStories](https://arxiv.org/abs/2305.07759). After 20
minutes of training it writes fluent children's stories. It carries stages 1
to 4 and stays the testbed afterwards: fine-tuning, RLHF and RLVR are all run
on it, where an experiment takes minutes and can be repeated.

The second is Qwen2.5-0.5B-Instruct, an open pretrained model with 494M
parameters, loaded into the same code from stage 5 on. Post-training on a real
task needs a model that already knows something: RLVR improves a model on a
task it sometimes gets right, and a model trained on a laptop from children's
stories does not do arithmetic at any rate worth reinforcing. The 0.5B model
solves 47% of GSM8K's test problems as published. On the laptop it generates
3,100 tokens per second in a batch and trains at 4,000, which is enough for
fine-tuning in 10 minutes and a GRPO run in 22.

## Layout

```
minllm/      the library, one concept per file
scripts/     one entry point per step of the pipeline
tests/       checks for the claims the notes make
docs/        one note per stage and side topic, with figures drawn by scripts/make_figures.py
site/        the website: the notes and the code as pages, a film per chapter and interactive
             figures that read runs/, and the story model in JavaScript, tested against the
             Python one. site/tools/render.mjs renders the films to video files
runs/        logs and benchmarks, the source of every number and figure in the notes
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
