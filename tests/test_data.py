import numpy as np
import torch

from minllm.data import EOT, get_batch, read_stories


def test_read_stories(tmp_path):
    # Long enough that stories and separators straddle the 1 MB read blocks.
    stories = [f"Story {i}. " + "The cat sat. " * (i % 50 + 1) for i in range(6000)]
    path = tmp_path / "stories.txt"
    path.write_text("".join(f"\n{story}\n{EOT}" for story in stories), encoding="utf-8")
    assert path.stat().st_size > 2 << 20
    assert list(read_stories(path)) == [story.strip() for story in stories]

    head = list(read_stories(path, max_bytes=1 << 20))
    assert 0 < len(head) < len(stories) and head == [story.strip() for story in stories[: len(head)]]


def test_get_batch_targets_are_inputs_shifted_by_one():
    tokens = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(tokens, batch_size=8, seq_len=16, device="cpu", generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)  # in a stream that counts up, the next token is always one more
    assert torch.equal(x[:, 1:], y[:, :-1])
