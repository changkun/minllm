"""
Serve the same requests in batches that start and end together, as stage 4
generates, and in a running batch that requests join and leave, as
minllm/batching.py does. Once with every request there at the start, and
once with requests arriving at random, as they reach a server.

    uv run scripts/bench_batching.py                                   # the 0.5B model on GSM8K's test problems
    uv run scripts/bench_batching.py --model out/sft-words --task words --n 512 --rate 40

Replies are greedy. Writes runs/bench/batching-<model>-<device>.json, which
docs/04c-continuous-batching.md quotes.
"""

import argparse
import json
import os
import random
import time
from collections import deque

import torch

from minllm import chat, checkpoint, tasks
from minllm.batching import Batcher
from minllm.common import pick_device, result_path
from minllm.data import EOT
from minllm.inference import stream
from minllm.pretrained import SYSTEM_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen")
parser.add_argument("--task", choices=["gsm8k", "words"], default="gsm8k")
parser.add_argument("--n", type=int, default=256, help="requests: the first test examples")
parser.add_argument("--slots", type=int, nargs="+", default=[16, 64], help="rows of a batch")
parser.add_argument("--rate", type=float, default=4.0, help="requests per second, on average, when they arrive at random")
parser.add_argument("--max-new-tokens", type=int, default=None, help="the limit on a reply's length; the task's own by default")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
args = parser.parse_args()

device = pick_device()
model, tok, meta = checkpoint.load(args.model, device, torch.float32 if device == "cpu" else torch.bfloat16)
stop = [tok.special_tokens[chat.IM_END], tok.special_tokens[EOT]]
prompts = [chat.encode(tok, e["messages"]) for e in tasks.test_examples(args.task, args.n, SYSTEM_PROMPT)]
limit = args.max_new_tokens or tasks.MAX_NEW_TOKENS[args.task]
room = max(len(p) for p in prompts) + limit  # the longest a slot has to hold
rng = random.Random(args.seed)
gaps = [rng.expovariate(args.rate) for _ in prompts]  # Poisson arrivals: independent, at the given rate
at_random = [sum(gaps[:i]) for i in range(len(prompts))]


def sync():
    if device == "mps":
        torch.mps.synchronize()


class Clock:
    """Seconds since the first request could arrive, and what happened to every request when."""

    def __init__(self):
        self.start = time.time()
        self.first, self.end, self.tokens = {}, {}, {i: [] for i in range(len(prompts))}

    def now(self):
        return time.time() - self.start

    def wait_for(self, arrival):
        time.sleep(max(0.0, arrival - self.now()))

    def token(self, i, token):
        self.first.setdefault(i, self.now())
        if token is not None:
            self.tokens[i].append(token)


@torch.no_grad()
def static(slots, arrivals):
    """
    Batches that start and end together. When the GPU is free, the requests
    that have arrived, up to `slots` of them, start as one batch, and the
    batch runs until its longest reply has ended. Whoever arrives meanwhile
    waits for it.
    """
    clock, waiting, steps = Clock(), deque(range(len(prompts))), 0
    while waiting:
        clock.wait_for(arrivals[waiting[0]])
        batch = []
        while waiting and len(batch) < slots and arrivals[waiting[0]] <= clock.now():
            batch.append(waiting.popleft())
        for row in stream(model, [prompts[i] for i in batch], limit, temperature=0, stop=stop):
            row = row.tolist()
            steps += 1
            for i, token in zip(batch, row):
                if i in clock.end:
                    continue
                clock.token(i, token if token >= 0 else None)
                if token < 0 or len(clock.tokens[i]) == limit:
                    clock.end[i] = clock.now()
        for i in batch:
            clock.end.setdefault(i, clock.now())
    return clock, steps


@torch.no_grad()
def continuous(slots, arrivals):
    """A running batch: a request takes a free slot at the next step after it arrives, and leaves when it ends."""
    clock, waiting, steps = Clock(), deque(range(len(prompts))), 0
    batcher = Batcher(model, slots, room, stop=stop, temperature=0)
    ids = {}
    while waiting or batcher.busy():
        if not batcher.busy():
            clock.wait_for(arrivals[waiting[0]])
        while waiting and arrivals[waiting[0]] <= clock.now():
            i = waiting.popleft()
            ids[batcher.add(prompts[i], limit)] = i
        for request, token, done in batcher.step():
            clock.token(ids[request], token)
            if done:
                clock.end[ids[request]] = clock.now()
        steps += 1
    return clock, steps


def summary(clock, steps, arrivals):
    n = len(prompts)
    latency = sorted(clock.end[i] - arrivals[i] for i in range(n))
    first = sorted(clock.first[i] - arrivals[i] for i in range(n))
    tokens = sum(len(t) for t in clock.tokens.values())
    seconds = max(clock.end.values())
    return {
        "seconds": seconds,
        "steps": steps,
        "tokens": tokens,
        "tokens_per_second": tokens / seconds,
        "latency_mean": sum(latency) / n,
        "latency_median": latency[n // 2],
        "latency_p90": latency[int(0.9 * n)],
        "first_token_mean": sum(first) / n,
        "first_token_p90": first[int(0.9 * n)],
    }


results = {"model": args.model, "task": args.task, "device": device, "requests": len(prompts), "max_new_tokens": limit, "rate": args.rate, "runs": []}
replies = {}
for slots in args.slots:
    for arrival_name, arrivals in (("at once", [0.0] * len(prompts)), ("at random", at_random)):
        for name, serve in (("static", static), ("continuous", continuous)):
            sync()
            clock, steps = serve(slots, arrivals)
            sync()
            r = {"batching": name, "slots": slots, "arrival": arrival_name} | summary(clock, steps, arrivals)
            results["runs"].append(r)
            replies[name, slots, arrival_name] = [clock.tokens[i] for i in range(len(prompts))]
            print(f"{name:10s} {slots:3d} slots, {arrival_name:9s}: {r['seconds']:6.1f} s, {r['steps']:5d} steps, {r['tokens_per_second']:6.0f} tokens/s | latency {r['latency_mean']:5.1f} s mean, {r['latency_p90']:5.1f} s p90 | first token {r['first_token_mean']:5.2f} s mean", flush=True)
            if device == "mps":
                torch.mps.empty_cache()

lengths = [len(t) for t in replies["static", args.slots[0], "at once"]]
results["reply_tokens"] = {"mean": sum(lengths) / len(lengths), "max": max(lengths), "at_limit": sum(n >= limit for n in lengths)}
# Greedy replies are the same however they are batched, up to the order in
# which 16-bit sums are added: count the replies that came out otherwise.
reference = replies["static", args.slots[0], "at once"]
results["replies_differing_from_the_first_run"] = {f"{name} {slots} {arrival}": sum(a != b for a, b in zip(reference, r)) for (name, slots, arrival), r in replies.items()}
print("replies differing from the first run:", results["replies_differing_from_the_first_run"])

name = os.path.basename(args.model.rstrip("/"))
out = result_path(args.out or f"runs/bench/batching-{name}-{device}.json")
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {out}")
