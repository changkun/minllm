"""
Write the samples that the notes quote, by running the commands the notes
give: the stories of stage 3 and 3a, one prompt at three temperatures for
stage 4, and the story model's reply before and after stage 6's
fine-tuning on the words task; and the number of examples of that task.
Each output is saved with its command under runs/samples/, so that a
quoted sample can be checked against the run that wrote it.

    uv run scripts/samples.py
"""

import json
import os
import subprocess
import sys

from minllm import tasks

MOLE = "Write a short story using these words: mole, protect."
SAMPLES = {
    "03-stories": ["scripts/sample.py"],
    "03-robot": ["scripts/sample.py", "--prompt", "The tiny robot was sad because", "--temperature", "0.5"],
    "03a-distilled": ["scripts/sample.py", "--ckpt", "out/distilled/ckpt.pt"],
    "04-temperature-0": ["scripts/sample.py", "--n", "1", "--top-k", "0", "--temperature", "0"],
    "04-temperature-0.8": ["scripts/sample.py", "--n", "1", "--top-k", "0", "--temperature", "0.8"],
    "04-temperature-1.5": ["scripts/sample.py", "--n", "1", "--top-k", "0", "--temperature", "1.5"],
    "06-mole-before": ["scripts/chat.py", "--model", "out/base", "--prompt", MOLE, "--temperature", "0"],
    "06-mole-after": ["scripts/chat.py", "--model", "out/sft-words", "--prompt", MOLE, "--temperature", "0"],
}

os.makedirs("runs/samples", exist_ok=True)
for name, command in SAMPLES.items():
    out = subprocess.run([sys.executable, *command], check=True, capture_output=True, text=True).stdout
    with open(f"runs/samples/{name}.txt", "w", encoding="utf-8") as f:
        f.write("$ uv run " + " ".join(f'"{c}"' if " " in c else c for c in command) + "\n\n" + out)
    print(f"runs/samples/{name}.txt")

# Stage 6's words task: how many examples tasks.words_examples can cut from the training text.
made = sum(1 for _ in tasks.words_examples("data/tinystories/train.txt", tasks.content_words(), None))
with open("runs/samples/words-examples.json", "w") as f:
    json.dump({"training_text": "data/tinystories/train.txt", "examples": made}, f)
print(f"{made:,} examples of the words task in the training text")
