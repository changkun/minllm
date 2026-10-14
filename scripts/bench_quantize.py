"""
Measure what a quantized Qwen costs in memory and in time: in this
repository's code, which expands the packed weights for every forward pass,
and in MLX, which multiplies with them as they are.

    uv run scripts/quantize.py --model qwen --bits 8 && uv run scripts/export.py --model out/qwen-q8
    uv run scripts/quantize.py --model qwen --bits 4 && uv run scripts/export.py --model out/qwen-q4
    uv run --with mlx-lm scripts/bench_quantize.py

Writes runs/bench/quant-<device>.json, which docs/10a-quantization.md quotes.
mlx-lm is installed for this command only.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

from minllm import chat, checkpoint, quantize
from minllm.common import pick_device, result_path
from minllm.inference import generate
from minllm.pretrained import DIR

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=3, help="measure every case this often and keep the fastest")
parser.add_argument("--out", default=None)
parser.add_argument("--case", default=None, help="measure this one case and print the result: the script calls itself with it")
args = parser.parse_args()

# (where it runs, bits per weight, what is loaded, the weight file)
CASES = {
    "ours-16": ("this repository", 16, "qwen", f"{DIR}/model.safetensors"),
    "ours-8": ("this repository", 8, "out/qwen-q8", "out/qwen-q8/model.safetensors"),
    "ours-4": ("this repository", 4, "out/qwen-q4", "out/qwen-q4/model.safetensors"),
    "mlx-16": ("MLX", 16, DIR, f"{DIR}/model.safetensors"),
    "mlx-8": ("MLX", 8, "out/export/qwen-q8", "out/export/qwen-q8/model.safetensors"),
    "mlx-4": ("MLX", 4, "out/export/qwen-q4", "out/export/qwen-q4/model.safetensors"),
}
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a long story about a kitten who explores a garden."},
]


def ours(spec):
    """Generation with the cached sampler of stage 4, and the GPU memory it takes."""
    device = pick_device()
    model, tok, _ = checkpoint.load(spec, device, torch.bfloat16)
    prompt = chat.encode(tok, MESSAGES)
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    result = {"weights_gb": quantize.nbytes(model) / 1e9}
    autocast = torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=device != "cpu")
    for batch in [1, 64]:
        best = float("inf")
        for _ in range(3):  # the first run warms the device up
            sync()
            start = time.time()
            with autocast:
                generate(model, [prompt] * batch, 100, temperature=0)
            sync()
            best = min(best, time.time() - start)
        result[f"batch_{batch}"] = batch * 100 / best
        if device == "mps" and batch == 1:
            # the pool of GPU memory after one sequence, as MLX is measured: weights and work
            result["peak_gb"] = torch.mps.driver_allocated_memory() / 1e9
    return result


def mlx(directory):
    """Generation of one sequence with mlx-lm, by its own count of tokens per second and of memory."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    model, tokenizer = load(directory)
    prompt = tokenizer.encode(chat.render(MESSAGES), add_special_tokens=False)
    result = {"weights_gb": mx.get_active_memory() / 1e9, "batch_1": 0.0}
    for _ in range(3):
        for response in stream_generate(model, tokenizer, prompt, max_tokens=200):
            pass
        result["batch_1"] = max(result["batch_1"], response.generation_tps)
    return result | {"peak_gb": response.peak_memory}


if args.case:
    where, _, spec, _ = CASES[args.case]
    print(json.dumps(ours(spec) if where == "this repository" else mlx(spec)))
    raise SystemExit

try:
    import mlx_lm  # noqa: F401
except ImportError:
    CASES = {case: row for case, row in CASES.items() if row[0] != "MLX"}
    print("mlx-lm is not installed: measuring this repository's code only")

rows = []
for case, (where, bits, spec, file) in CASES.items():
    # Each measurement in a process of its own, the fastest of several: see scripts/bench_moe.py.
    command = [sys.executable, __file__, "--case", case]
    runs = [json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()[-1]) for _ in range(args.repeats)]
    best = max(runs, key=lambda r: r["batch_1"])
    rows.append({"where": where, "bits": bits, "model": spec, "file_mb": os.path.getsize(file) / 1e6} | best)
    batch = f", {best['batch_64']:6.0f} in a batch of 64" if "batch_64" in best else ""
    print(f"{where:16s} {bits:2d} bits | file {rows[-1]['file_mb']:5.0f} MB | weights in memory {best['weights_gb']:.2f} GB, peak {best.get('peak_gb', 0):.2f} GB | {best['batch_1']:6.1f} tokens/s alone{batch}", flush=True)

out = result_path(args.out or f"runs/bench/quant-{pick_device()}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"device": pick_device(), "repeats": args.repeats, "cases": rows}, f, indent=1)
print(f"wrote {out}")
