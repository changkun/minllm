"""
What the scripts share: choosing a device, handing GPU memory back,
downloading a file, and loading a checkpoint of the model trained here.
"""

import os
import urllib.request

import torch

from minllm.model import Config, Transformer


def pick_device():
    """The fastest device available: an NVIDIA GPU, then Apple's GPU, then the CPU."""
    device = os.environ.get("MINLLM_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        # An Apple GPU shares its memory with the rest of the machine. Without
        # a limit, a batch that is too large does not fail: it pushes every
        # other program into swap. With one, it raises an out-of-memory error.
        torch.mps.set_per_process_memory_fraction(float(os.environ.get("MINLLM_MEMORY_FRACTION", 0.5)))
    return device


def free_memory(device):
    """
    Hand unused GPU memory back. PyTorch keeps freed blocks in a pool to reuse
    them, and reuses a block only for a tensor of the same size. A loop whose
    tensors have a different size at every step, as fine-tuning on examples of
    varying length does, fills the pool with blocks it never uses again. Such
    a run holds 2 GB of tensors at any time and still grows to 25 GB within a
    minute. Returns the size of the pool before it was emptied, in GB.
    """
    if torch.device(device).type != "mps":
        return 0.0
    pool = torch.mps.driver_allocated_memory() / 2**30
    torch.mps.empty_cache()
    return pool


def fetch(url, path, max_bytes=None):
    """Download url to path, or only its first max_bytes. Does nothing if path exists."""
    if os.path.exists(path):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    tmp = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response, open(tmp, "wb") as f:
        total, done = int(response.headers["Content-Length"]), 0
        while block := response.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r{path}: {done >> 20}/{total >> 20} MB", end="", flush=True)
    print()
    os.replace(tmp, path)
    return True


def load_checkpoint(path, device):
    """Rebuild the model saved by scripts/train.py. Returns it with the raw checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Transformer(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
