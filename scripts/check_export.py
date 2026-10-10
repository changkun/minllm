"""
Check an exported model against this repository's own code, by loading it
with Hugging Face transformers: the same token ids, the same chat prompt, the
same logits, the same greedy continuation.

    uv run --with transformers scripts/check_export.py --model out/sft-words

transformers is installed for this command only. See docs/10-stack.md.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minllm import chat, checkpoint, lora
from minllm.inference import generate

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="the model that was exported")
parser.add_argument("--export", default=None, help="where it was exported to")
args = parser.parse_args()
exported = args.export or f"out/export/{os.path.basename(args.model.rstrip('/'))}"

ours, tok, meta = checkpoint.load(args.model, "cpu")
ours = lora.merge(ours)
theirs = AutoModelForCausalLM.from_pretrained(exported, dtype=torch.float32).eval()
their_tok = AutoTokenizer.from_pretrained(exported)

texts = ["Once upon a time, there was a little girl named Lily.", "  two spaces,\n\nnew lines, don't, 12345 and émojis 🙂", "<|im_start|>user\nhi<|im_end|>\n"]
for text in texts:
    assert their_tok(text)["input_ids"] == tok.encode(text), f"token ids differ for {text!r}"
print(f"tokenizer: identical ids on {len(texts)} texts")

messages = [{"role": "user", "content": "Write a short story using these words: garden, kitten."}]
if meta["family"] == "qwen":
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages
assert their_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) == chat.render(messages)
print("chat template: identical prompt")

ids = chat.encode(tok, messages)
with torch.no_grad():
    a = ours(torch.tensor([ids]))[0][0]
    b = theirs(torch.tensor([ids])).logits[0]
    difference = (a - b).abs().max().item()
    their_greedy = theirs.generate(torch.tensor([ids]), max_new_tokens=24, do_sample=False, repetition_penalty=1.0)[0, len(ids) :].tolist()
assert difference < 1e-3, f"logits differ by {difference}"
print(f"logits: largest difference {difference:.2e} over {a.numel():,} values")
our_greedy = generate(ours, [ids], 24, temperature=0)[0]
assert their_greedy[: len(our_greedy)] == our_greedy, "greedy continuations differ"
print(f"generation: identical for {len(our_greedy)} tokens: {tok.decode(our_greedy)!r}")
