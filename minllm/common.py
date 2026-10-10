import os
import urllib.request

import torch

from minllm.model import Config, Transformer


def pick_device():
    """The fastest device available: an NVIDIA GPU, then Apple's GPU, then the CPU."""
    if device := os.environ.get("MINLLM_DEVICE"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
