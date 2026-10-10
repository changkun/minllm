"""
Talk to a model: Qwen2.5-0.5B-Instruct, or one fine-tuned in this repository.

    uv run scripts/chat.py
    uv run scripts/chat.py --prompt "Why is the sky blue?"
    uv run scripts/chat.py --model out/sft-words --prompt "Write a short story using these words: garden, kitten."

See docs/05-pretrained.md and docs/06-finetuning.md.
"""

import argparse

import torch

from minllm import chat, checkpoint
from minllm.common import pick_device
from minllm.data import EOT
from minllm.inference import stream_text
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen", help="qwen, or the directory of a trained model")
parser.add_argument("--prompt", default=None, help="answer this one message and exit")
parser.add_argument("--system", default=None, help="defaults to Qwen's system prompt for Qwen, and to none otherwise")
parser.add_argument("--max-tokens", type=int, default=512)
# Qwen's recommended sampling settings, from its generation_config.json
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--top-p", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = pick_device()
torch.manual_seed(args.seed)
# The weights are published in bfloat16. Inference keeps them that way, which
# halves the memory and is faster; training converts them to 32 bits.
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
if args.system is None and meta["family"] == "qwen":
    args.system = SYSTEM_PROMPT


def reply(messages):
    """Generate the assistant's next message, printing it as it is written."""
    text = ""
    sampling = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, stop=stop)
    for piece in stream_text(model, tok, chat.encode(tok, messages), args.max_tokens, **sampling):
        print(piece, end="", flush=True)
        text += piece
    print()
    return text


messages = [{"role": "system", "content": args.system}] if args.system else []
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
