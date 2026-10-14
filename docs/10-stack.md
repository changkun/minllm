# 10. The practitioner's stack

Code: [`minllm/export.py`](../minllm/export.py) ·
Scripts: [`scripts/export.py`](../scripts/export.py),
[`scripts/check_export.py`](../scripts/check_export.py),
[`scripts/serve.py`](../scripts/serve.py)

The nine stages before this one wrote everything by hand, to see how it
works. Nobody builds products that way. This stage connects the two worlds:
it writes the models of this repository in the format everyone else's tools
read, runs them in those tools, and maps each stage to the library that
practitioners use for it.

## Export

Stage 5 took a published model apart into three files. Export is the same
step backwards. `scripts/export.py` writes a directory with:

- **`model.safetensors`**, the weights under the reference names. LoRA
  adapters are merged into their matrices first, so the result is an ordinary
  model.
- **`config.json`**, the shape of the model in the reference vocabulary. The
  architecture of stage 2 has a name there. Without biases it is a
  `LlamaForCausalLM`, and with biases on the query, key and value projections
  it is a `Qwen2ForCausalLM`. The model trained from scratch in stage 3 is, to
  every other tool, a very small Llama. (With experts in place of the MLP it
  is a `MixtralForCausalLM`: see [2a](02a-moe.md).)
- **`tokenizer.json`**, the tokenizer of stage 1 in the format of the Hugging
  Face `tokenizers` library: the same merges, the same split pattern, the
  bytes spelled in GPT-2's alphabet. And `tokenizer_config.json`, which
  carries the chat format as a template.

For a fine-tuned Qwen, the tokenizer and configuration are copied unchanged
from the download, and only the weights are new.

```sh
uv run scripts/export.py --model out/sft-words
uv run --with transformers scripts/check_export.py --model out/sft-words
```

```
tokenizer: identical ids on 3 texts
chat template: identical prompt
logits: largest difference 1.18e-05 over 102,400 values
generation: identical for 24 tokens: 'Once upon a time, there was a little kitten named Fluffy. […]'
```

The check loads the exported directory with Hugging Face transformers and
compares it against this repository's own code. As in stage 5, transformers
is installed for that one command and is not a dependency.

A fine-tuned 0.5B model exports the same way. Its adapters are merged into
the weights, so 35 MB of LoRA become a complete model of 2.0 GB in 32-bit
floats:

```sh
uv run scripts/export.py --model out/sft-gsm8k-lora
uv run --with transformers scripts/check_export.py --model out/sft-gsm8k-lora
```

```
tokenizer: identical ids on 3 texts
chat template: identical prompt
logits: largest difference 1.81e-04 over 4,710,016 values
generation: identical for 24 tokens: 'The kitten curled up on the windowsill, its fur a soft, golden brown. […]'
```

## Running it elsewhere

**MLX**, Apple's array framework, runs the exported directory as it is:

```sh
uv run --with mlx-lm python -m mlx_lm generate --model out/export/sft-words \
    --prompt "Write a short story using these words: garden, kitten." --temp 0
```

```
Once upon a time, there was a little kitten named Fluffy. Fluffy was a very
happy kitten. She loved to play in the garden with her friends. […]
Generation: 100 tokens, 1853.896 tokens-per-sec
```

The same story, token for token, as this repository's sampler writes at
temperature 0. It arrives faster: 1,854 tokens per second for a single
sequence, with mlx-lm 0.32 writing its default of 100 tokens, against 460 in
stage 4. mlx_lm sends each decode step to the GPU before it has read back
the token of the step before, so the GPU does not wait on Python; how much
of the difference that accounts for was not measured here. The check outputs
of this section and MLX's are saved in `runs/export/`.

**vLLM** serves the same directory with `vllm serve out/export/sft-words` on a
Linux machine with an NVIDIA GPU. On macOS, vLLM itself runs only on the
CPU, in an experimental build from source, so this was not tried here. For
Apple's GPU there is a separate plugin, vllm-metal, which runs vLLM on top
of MLX. It was not tried either. **llama.cpp** needs one more conversion,
from this format to its own GGUF file, with the `convert_hf_to_gguf.py`
script of its repository. That was not tried here either.

Models that run locally are usually stored in 4 to 8 bits per weight. The
side topic [10a](10a-quantization.md) writes such a file and runs it in MLX.

## Serving

A model becomes useful to other programs behind an HTTP API, and the one
everyone speaks is OpenAI's chat completions. `scripts/serve.py` implements
it in 85 lines on the standard library's HTTP server:

```sh
uv run scripts/serve.py --model out/sft-words
curl localhost:8000/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Write a short story using these words: garden, kitten."}],
  "stream": true}'
```

```
data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Once"}}], …}
data: {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": " upon"}}], …}
data: [DONE]
```

Any client library for the OpenAI API can be pointed at it.

This server handles one request at a time, which wastes what stage 4
measured: a second sequence in the batch is almost free. A production server
is built around that fact. vLLM keeps one batch running and lets requests
join and leave it between decode steps (continuous batching), allocates the
KV cache in blocks so that no memory is reserved for text that is never
written (paged attention), and reuses the cache of a prompt prefix that
several requests share.

## The map

| Stage | Here | In practice |
|---|---|---|
| 1 Tokenizer | `tokenizer.py` | `tokenizers`, `tiktoken`, SentencePiece |
| 2 Model | `model.py` | the model classes of `transformers` |
| 2a Mixture of experts | `MoE` in `model.py` | the Mixtral, Qwen3-MoE and DeepSeek classes of `transformers`; grouped kernels for the experts in training frameworks |
| 2b Longer context | `rope_table` in `model.py`, `context.py` | continued pretraining on long text, with a larger theta or YaRN; FlashAttention kernels; needle-in-a-haystack tests |
| 3 Pretraining | `scripts/train.py` | custom loops over many GPUs, with `accelerate` or plain PyTorch; `datasets` for the data |
| 3a Distillation | `distill.py`, `train.py --teacher` | a distillation loss in a custom loop; more often, fine-tuning on text that a larger model wrote |
| 3b Data preparation | `prepare.py`, `scripts/prepare_data.py` | trafilatura for the text, fastText for the language, quality classifiers; filters tested by training, as for FineWeb and DataComp-LM |
| 3c Scaling laws | `scaling.py`, `scripts/scaling.py` | laws fitted to small runs before the large one; µP, so that the learning rate carries over with width |
| 4 Inference | `inference.py` | vLLM, SGLang, llama.cpp, MLX |
| 4a Structured output | `constrain.py`, `automaton.py`, `schema.py` | Outlines, XGrammar, llama.cpp's GBNF grammars; guided decoding in vLLM and SGLang |
| 4b Speculative decoding | `speculative.py` | vLLM, SGLang, TensorRT-LLM, llama.cpp; Medusa and EAGLE drafts built into the model |
| 5 Pretrained weights | `pretrained.py` | `from_pretrained`, the Hugging Face Hub, `safetensors` |
| 6 Fine-tuning | `scripts/finetune.py`, `lora.py` | `peft`, `trl`'s `SFTTrainer`, Unsloth |
| 6a Tool calling | `tools.py`, `scripts/evaluate_tools.py` | each family's own call format, which serving software translates to an API's `tools` and `tool_calls` |
| 7 Evaluation | `evaluate.py`, `tasks.py` | `lm-evaluation-harness`, Inspect |
| 7a Multiple choice | `choice.py`, `scripts/evaluate_choice.py` | `lm-evaluation-harness` and its `loglikelihood` requests |
| 7b Evaluation by a judge | `judge.py`, `scripts/judge.py` | a large model as the judge, as in MT-Bench, AlpacaEval and Arena-Hard |
| 8 RLHF | `reward.py`, `scripts/ppo.py`, `scripts/dpo.py` | `trl` (`DPOTrainer`, reward modeling), OpenRLHF |
| 8a Safety | `safety.py`, `scripts/evaluate_safety.py`, `scripts/make_refusals.py` | refusals in the fine-tuning and preference data; XSTest for refusing too much |
| 9 RLVR | `rl.py`, `scripts/grpo.py` | verl, OpenRLHF, `trl`'s `GRPOTrainer` |
| 9a Self-training | `selftrain.py`, `scripts/self_train.py` | no library of its own: an inference engine for the samples, a filter, and the fine-tuning of stage 6 |
| 9b Reinforcement learning with a tool | `rollout` in `tools.py`, `scripts/grpo.py --task gsm8k-tools` | tools run in sandboxes during the rollouts, their output masked out of the loss (Search-R1, ReTool) |
| 9c Reasoning | `reasoning.py`, `scripts/test_time.py`, `scripts/make_traces.py` | DeepSeek-R1's recipe: long chains of thought, reinforcement learning on checked answers, small models distilled from its thinking |
| 10 Serving | `scripts/serve.py` | vLLM, an OpenAI-compatible endpoint |
| 10a Quantization | `quantize.py` | llama.cpp's GGUF files, `mlx_lm.convert -q`, bitsandbytes; GPTQ and AWQ for smarter rounding |

The libraries add scale, coverage and years of fixed bugs. They do not add
ideas: every row computes what its left column computes.

One difference of scale is worth knowing before reading an RL framework. In
this repository, sampling and training happen in the same process on the same
weights. At scale they are separate systems. An inference engine such as vLLM
produces the rollouts, a trainer consumes them, and most of the engineering
is in moving weights and samples between the two.

## Exercises

1. Export the model of stage 3 (`--model out/base`) and generate a story with
   `mlx_lm`. It has no chat training: what prompt do you have to give it?
2. Point an OpenAI client library at `scripts/serve.py` and hold a conversation
   with Qwen through it.
3. Send two requests to the server at once. Measure how long the second one
   waits. Then sketch what would have to change for both to be generated in
   one batch.
4. Change one line of `config_json` so that the export is wrong in a way that
   still loads: `rope_theta`, or `rms_norm_eps`. Which line of
   `check_export.py` catches it?

## References

- Wolf et al. (2019), [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771). The transformers library: the reference implementations and the layout of a model directory.
- Hugging Face, [safetensors](https://github.com/huggingface/safetensors). The weight file format.
- Kwon et al. (2023), [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). vLLM, the inference server.
- Yu et al. (2022), [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI. Continuous batching.
- Apple, [mlx-lm](https://github.com/ml-explore/mlx-lm). Language models on MLX, which runs the exported directory.
- Gerganov et al., [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). The file format of llama.cpp.
- OpenAI, [Chat completions API reference](https://platform.openai.com/docs/api-reference/chat). The HTTP interface that `scripts/serve.py` speaks.
- Hugging Face, [trl](https://github.com/huggingface/trl). Trainers for the post-training stages.
