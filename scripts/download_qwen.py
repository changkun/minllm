"""
Download Qwen2.5-0.5B-Instruct: weights, tokenizer and configuration, 1 GB.

    uv run scripts/download_qwen.py

See docs/05-pretrained.md.
"""

from minllm.pretrained import DIR, download

download()
print(f"Qwen2.5-0.5B-Instruct is in {DIR}")
