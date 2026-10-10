"""
Talk to Qwen2.5-0.5B-Instruct, running on the model code of this repository.

    uv run scripts/chat.py
    uv run scripts/chat.py --prompt "Why is the sky blue?"

See docs/05-pretrained.md.
"""

import argparse

import torch

from minllm import chat
from minllm.common import pick_device
from minllm.data import EOT
from minllm.inference import stream
from minllm.pretrained import SYSTEM_PROMPT, load_model, load_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default=None, help="answer this one message and exit")
parser.add_argument("--system", default=SYSTEM_PROMPT)
parser.add_argument("--max-tokens", type=int, default=512)
# Qwen's recommended sampling settings, from its generation_config.json
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--top-p", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
tok = load_tokenizer()
# The weights are published in bfloat16. Inference keeps them that way, which
# halves the memory and is faster; training converts them to 32 bits.
model = load_model(device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]


def reply(messages):
    """Generate the assistant's next message, printing it as it is written."""
    prompt = chat.encode(tok, messages)
    ids, shown = [], ""
    for step in stream(model, [prompt], args.max_tokens, args.temperature, args.top_k, args.top_p, stop):
        if step[0] < 0:
            break
        ids.append(step[0].item())
        text = tok.decode(ids)
        if not text.endswith("�"):  # hold back half of a multi-byte character
            print(text[len(shown) :], end="", flush=True)
            shown = text
    print()
    return tok.decode(ids)


messages = [{"role": "system", "content": args.system}]
if args.prompt:
    reply(messages + [{"role": "user", "content": args.prompt}])
else:
    print("Ctrl-D to quit.")
    while True:
        try:
            user = input("\n> ")
        except EOFError:
            break
        messages.append({"role": "user", "content": user})
        print()
        messages.append({"role": "assistant", "content": reply(messages)})
