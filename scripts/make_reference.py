"""
Record what the reference implementation computes, for the tests of stage 5
to compare against. This is the one place where Hugging Face transformers is
used, and it is not a dependency of the project: uv installs it for this run
only.

    uv run --with transformers scripts/make_reference.py

Writes tests/fixtures/qwen.json.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minllm.pretrained import DIR, SYSTEM_PROMPT

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces,\ttabs\tand\n\nblank lines\n",
    "don't I'll we've THEY'RE it's",
    "Numbers: 0 7 42 1234567890 3.14159 1e-9",
    "def f(x):\n    return x ** 2  # square\n",
    "naïve café, coöperate, Ångström",
    "é is e with a combining accent",
    "你好，世界！今天天气怎么样？",
    "日本語のテキストと한국어 텍스트",
    "emoji 🙂🙃 and a flag 🇩🇪 and a family 👨‍👩‍👧",
    "<|im_start|>user\nhi<|im_end|>\n<|endoftext|>",
    "https://example.com/a?b=c&d=e#f",
    "",
]
MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is 17 * 23? Think step by step."},
]

tok = AutoTokenizer.from_pretrained(DIR)
model = AutoModelForCausalLM.from_pretrained(DIR, dtype=torch.float32).eval()

chat_text = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
chat_ids = tok(chat_text)["input_ids"]
with torch.no_grad():
    logits = model(torch.tensor([chat_ids])).logits[0]
    greedy = model.generate(torch.tensor([chat_ids]), max_new_tokens=32, do_sample=False, repetition_penalty=1.0)
last = logits[-1]
top = last.topk(10)

reference = {
    "tokenizer": [{"text": text, "ids": tok(text)["input_ids"]} for text in TEXTS],
    "chat": {"messages": MESSAGES, "text": chat_text, "ids": chat_ids},
    "logits": {
        "top_ids": top.indices.tolist(),
        "top_values": top.values.tolist(),
        # the logits of token 0 at every position, to check all positions and not only the last
        "token_0": logits[:, 0].tolist(),
        "mean": logits.mean().item(),
        "std": logits.std().item(),
    },
    "greedy": greedy[0, len(chat_ids) :].tolist(),
    "greedy_text": tok.decode(greedy[0, len(chat_ids) :]),
}
with open("tests/fixtures/qwen.json", "w") as f:
    json.dump(reference, f, ensure_ascii=False, indent=1)
print("wrote tests/fixtures/qwen.json")
print(reference["greedy_text"])
