import os

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


def load_checkpoint(path, device):
    """Rebuild the model saved by scripts/train.py. Returns it with the raw checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Transformer(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt
