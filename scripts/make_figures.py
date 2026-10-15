"""
Draw the figures of the notes into docs/figures/, from the logs in runs/:
the published record, not a learner's results in out/runs/.

    uv run scripts/make_figures.py

The figures are SVG, written out by hand. Drawing them needs no plotting
library, they stay sharp at any size, and they follow the reader's light or
dark theme. Every chart reads its numbers from a file in runs/; the diagrams
are drawn from constants.
"""

import argparse
import json
import math
import os
import re
from html import escape

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="docs/figures")
parser.add_argument("--dark", action="store_true", help="draw the dark theme unconditionally, to preview it")
args = parser.parse_args()

# Colors by role: the surface, three inks, the grid, and the series hues in a
# fixed order. Each has a light and a dark value.
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
LIGHT |= dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a")
DARK = dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835")
DARK |= dict(s1="#3987e5", s2="#d95926", s3="#199e70")


def stylesheet():
    rules = lambda colors: "".join(f".f-{k}{{fill:{v}}}.s-{k}{{stroke:{v}}}" for k, v in colors.items())
    font = "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}"
    if args.dark:
        return font + rules(DARK)
    return font + rules(LIGHT) + "@media(prefers-color-scheme:dark){" + rules(DARK) + "}"


def n(value):
    """A coordinate, written compactly."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def text_width(s, size):
    return len(s) * size * 0.56  # close enough for a system sans


class Figure:
    def __init__(self, width, height, title, subtitle=""):
        self.width, self.height, self.title, self.subtitle = width, height, title, subtitle
        self.parts = []
        self.rect(0.5, 0.5, width - 1, height - 1, fill="surface", stroke="grid", rx=8)
        self.text(24, 31, title, size=14, weight=600, fill="ink")
        if subtitle:
            self.text(24, 50, subtitle, fill="ink2")

    def rect(self, x, y, w, h, fill=None, stroke=None, rx=0, opacity=None, width=1):
        cls = f"f-{fill}" if fill else ""
        cls += f" s-{stroke}" if stroke else ""
        extra = "" if fill else ' fill="none"'
        extra += f' stroke-width="{width}"' if stroke else ""
        extra += f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<rect class="{cls}" x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="{rx}"{extra}/>')

    def line(self, x1, y1, x2, y2, stroke="grid", width=1):
        self.parts.append(
            f'<line class="s-{stroke}" x1="{n(x1)}" y1="{n(y1)}" x2="{n(x2)}" y2="{n(y2)}" stroke-width="{width}" stroke-linecap="round"/>'
        )

    def path(self, d, stroke=None, fill=None, width=2, opacity=None, dash=None):
        cls = (f"s-{stroke} " if stroke else "") + (f"f-{fill}" if fill else "")
        extra = "" if fill else ' fill="none"'
        extra += f' stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"' if stroke else ""
        extra += f' {"fill" if fill else "stroke"}-opacity="{opacity}"' if opacity is not None else ""
        extra += f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path class="{cls}" d="{d}"{extra}/>')

    def polyline(self, points, stroke, width=2, opacity=None, dash=None):
        self.path("M" + " L".join(f"{n(x)} {n(y)}" for x, y in points), stroke=stroke, width=width, opacity=opacity, dash=dash)

    def dot(self, x, y, fill, r=4):
        # The ring in the surface color keeps a dot legible where it crosses a line.
        self.parts.append(f'<circle class="f-{fill} s-surface" cx="{n(x)}" cy="{n(y)}" r="{r}" stroke-width="2"/>')

    def point(self, x, y, fill, r=2, opacity=0.35):
        self.parts.append(f'<circle class="f-{fill}" cx="{n(x)}" cy="{n(y)}" r="{r}" fill-opacity="{opacity}"/>')

    def text(self, x, y, s, size=12, fill="ink2", anchor="start", weight=400, rotate=None):
        extra = f' transform="rotate({rotate} {n(x)} {n(y)})"' if rotate else ""
        extra += f' font-weight="{weight}"' if weight != 400 else ""
        self.parts.append(
            f'<text class="f-{fill}" x="{n(x)}" y="{n(y)}" font-size="{size}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'
        )

    def arrow(self, points, stroke="muted"):
        """A polyline that ends in an arrowhead."""
        self.polyline(points, stroke, width=1.5)
        (x0, y0), (x1, y1) = points[-2], points[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        wing = lambda da: (x1 - 7 * math.cos(a + da), y1 - 7 * math.sin(a + da))
        (ax, ay), (bx, by) = wing(0.45), wing(-0.45)
        self.path(f"M{n(ax)} {n(ay)} L{n(x1)} {n(y1)} L{n(bx)} {n(by)}", stroke=stroke, width=1.5)

    def box(self, x, y, w, h, title, note="", fill="surface", opacity=None, stroke="axis"):
        """A labeled box for diagrams."""
        self.rect(x, y, w, h, fill=fill, stroke=stroke, rx=6, opacity=opacity)
        if note:
            self.text(x + w / 2, y + h / 2 - 3, title, fill="ink", anchor="middle", weight=600)
            self.text(x + w / 2, y + h / 2 + 13, note, size=11, anchor="middle")
        else:
            self.text(x + w / 2, y + h / 2 + 4, title, fill="ink", anchor="middle", weight=600)

    def legend(self, x, y, entries):
        """A row of keys: [(color, label, "line" or "dot"), ...]."""
        for color, label, kind in entries:
            if kind == "line":
                self.line(x, y - 4, x + 16, y - 4, stroke=color, width=2)
            else:
                self.dot(x + 8, y - 4, color)
            self.text(x + 22, y, label)
            x += 22 + text_width(label, 12) + 18

    def save(self, name):
        os.makedirs(args.out, exist_ok=True)
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">'
            f"<title>{escape(self.title)}</title><desc>{escape(self.subtitle)}</desc><style>{stylesheet()}</style>"
        )
        with open(f"{args.out}/{name}.svg", "w") as f:
            f.write(head + "\n" + "\n".join(self.parts) + "\n</svg>\n")
        print(f"{args.out}/{name}.svg")


class Axes:
    """A plot area inside a figure, with a linear or logarithmic scale per axis."""

    def __init__(self, fig, left, top, width, height, xlim, ylim, xlog=False, ylog=False):
        self.fig, self.left, self.top, self.width, self.height = fig, left, top, width, height
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    @staticmethod
    def _scale(v, lim, log):
        if log:
            return (math.log(v) - math.log(lim[0])) / (math.log(lim[1]) - math.log(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def x(self, v):
        return self.left + self.width * self._scale(v, self.xlim, self.xlog)

    def y(self, v):
        return self.top + self.height * (1 - self._scale(v, self.ylim, self.ylog))

    def yticks(self, ticks, fmt=str):
        for t in ticks:
            self.fig.line(self.left, self.y(t), self.left + self.width, self.y(t))
            self.fig.text(self.left - 8, self.y(t) + 4, fmt(t), size=11, fill="muted", anchor="end")

    def xticks(self, ticks, fmt=str, title=""):
        bottom = self.top + self.height
        self.fig.line(self.left, bottom, self.left + self.width, bottom, stroke="axis")
        for t in ticks:
            self.fig.line(self.x(t), bottom, self.x(t), bottom + 4, stroke="axis")
            self.fig.text(self.x(t), bottom + 17, fmt(t), size=11, fill="muted", anchor="middle")
        if title:
            self.fig.text(self.left + self.width / 2, bottom + 36, title, size=11, fill="muted", anchor="middle")

    def title(self, s):
        self.fig.text(self.left, self.top - 12, s, fill="ink", weight=600)

    def line(self, xs, ys, color, width=2, opacity=None, dash=None):
        self.fig.polyline([(self.x(a), self.y(b)) for a, b in zip(xs, ys)], color, width, opacity, dash)

    def dots(self, xs, ys, color, r=4):
        for a, b in zip(xs, ys):
            self.fig.dot(self.x(a), self.y(b), color, r)

    def label(self, x, y, s, dx=8, dy=4, anchor="start", fill="ink2"):
        self.fig.text(self.x(x) + dx, self.y(y) + dy, s, anchor=anchor, fill=fill)


def nice_ticks(lo, hi, count=3):
    """Round numbers that span lo to hi, about count of them."""
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda m: abs(m * magnitude - raw)) * magnitude
    ticks, t = [], math.ceil(lo / step - 1e-9) * step
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def thousands(v):
    return f"{v / 1000:g}K" if v >= 1000 else f"{v:g}"


def load(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ 1 tokenizer


def tokenizer_vocab():
    sizes = load("runs/bench/tokenizer.json")["sizes"]
    vocab = [s["vocab_size"] for s in sizes]
    fig = Figure(720, 330, "What a larger vocabulary buys, and what it costs", "Tokenizers trained on the same 64 MB of TinyStories. The model here uses 4,096 tokens.")
    chosen = vocab.index(4096)

    left = Axes(fig, 64, 108, 270, 150, (400, 21000), (2, 4.5), xlog=True)
    left.title("Bytes per token, held-out stories")
    left.yticks([2, 3, 4], str)
    left.xticks(vocab, lambda t: f"{t:,}", "vocabulary size")
    ys = [s["bytes_per_token"] for s in sizes]
    left.line(vocab, ys, "s1")
    left.dots(vocab, ys, "s1")
    left.label(vocab[chosen], ys[chosen], f"{ys[chosen]:.2f}", dx=0, dy=-12, anchor="middle", fill="ink")
    left.label(vocab[0], ys[0], f"{ys[0]:.2f}", dx=10, dy=4)
    left.label(vocab[-1], ys[-1], f"{ys[-1]:.2f}", dx=0, dy=-12, anchor="middle")

    right = Axes(fig, 420, 108, 270, 150, (400, 21000), (1, 100000), xlog=True, ylog=True)
    right.title("Occurrences of the rarest token in training")
    right.yticks([1, 10, 100, 1000, 10000, 100000], lambda t: f"{t:,}")
    right.xticks(vocab, lambda t: f"{t:,}", "vocabulary size")
    ys = [s["last_merge_count"] for s in sizes]
    right.line(vocab, ys, "s1")
    right.dots(vocab, ys, "s1")
    right.label(vocab[chosen], ys[chosen], f"{ys[chosen]:,}", dx=10, dy=-6, fill="ink")
    right.label(vocab[0], ys[0], f"{ys[0]:,}", dx=10, dy=4)
    right.label(vocab[-1], ys[-1], f"{ys[-1]:,}", dx=10, dy=4)
    fig.save("tokenizer-vocab")


# ---------------------------------------------------------------------- 2 model


def model_architecture():
    fig = Figure(720, 520, "The model", "Every block reads the residual stream, computes, and adds its result back.")
    sx = 150  # the residual stream runs down this x

    fig.text(sx, 84, "token ids  (B, T)", anchor="middle")
    fig.arrow([(sx, 92), (sx, 108)])
    fig.box(sx - 70, 110, 140, 36, "Embedding")
    fig.line(sx, 146, sx, 420, stroke="s1", width=3)
    fig.text(sx - 14, 170, "residual stream", anchor="end", size=11, fill="muted")
    fig.text(sx - 14, 184, "(B, T, dim)", anchor="end", size=11, fill="muted")

    fig.rect(40, 196, 652, 196, stroke="grid", rx=10)
    fig.text(692, 188, "Block, repeated 8 times", anchor="end", size=11, fill="muted")

    def sublayer(y, title, note, role):
        # Branch off the stream, through the norm and the sublayer, and back in.
        fig.arrow([(sx, y), (232, y)])
        fig.box(234, y - 18, 96, 36, "RMSNorm")
        fig.arrow([(330, y), (368, y)])
        fig.box(370, y - 24, 232, 48, title, note, fill="s1", opacity=0.1)
        fig.arrow([(602, y), (650, y), (650, y + 52), (sx + 13, y + 52)])
        fig.parts.append(f'<circle class="f-surface s-s1" cx="{sx}" cy="{y + 52}" r="11" stroke-width="2"/>')
        fig.text(sx, y + 57, "+", anchor="middle", fill="ink", size=15, weight=600)
        fig.text(370, y + 41, role, size=11, fill="muted")

    sublayer(238, "Attention", "RoPE · causal mask · grouped-query", "tokens exchange information")
    sublayer(322, "MLP", "SwiGLU: down(silu(gate(x)) · up(x))", "each token is processed alone")

    fig.arrow([(sx, 420), (sx, 428)])
    fig.box(sx - 70, 430, 140, 32, "RMSNorm")
    fig.arrow([(sx + 70, 446), (268, 446)])
    fig.box(270, 430, 300, 32, "Output layer  ·  the embedding matrix again")
    fig.arrow([(570, 446), (598, 446)])
    fig.text(604, 442, "logits", fill="ink", weight=600)
    fig.text(604, 457, "(B, T, vocab)", size=11)
    fig.text(24, 496, "B sequences of T tokens. The numbers are those of the default configuration: dim 384, vocabulary 4,096.", size=11, fill="muted")
    fig.save("model-architecture")


def model_attention():
    data = load("runs/base/attention.json")
    tokens = ["<eot>" if t == "<|endoftext|>" else t.strip() or t for t in data["tokens"]]
    T, cell = len(tokens), 11
    fig = Figure(720, 446, "Two attention heads of the trained model", "Each row is a token, and the color shows how much it reads from each token before it. Stronger is more.")
    notes = {
        (1, 2): "“She” and “her” look back to “Lily” and “girl”",
        (5, 5): "Tokens tend to look at the one just before",
    }
    for i, head in enumerate(data["heads"]):
        left, top = 84 + i * 340, 104
        fig.text(left, top - 26, f"Layer {head['layer']}, head {head['head']}", fill="ink", weight=600)
        fig.text(left, top - 10, notes.get((head["layer"], head["head"]), ""), size=11)
        for r in range(T):
            fig.text(left - 6, top + r * cell + 9, tokens[r], size=9, fill="muted", anchor="end")
            fig.text(left + r * cell + 8, top + T * cell + 6, tokens[r], size=9, fill="muted", anchor="end", rotate=-90)
            for c in range(r + 1):
                # One hue; the weight sets how far the cell stands out from the surface.
                fig.rect(left + c * cell, top + r * cell, cell - 1, cell - 1, fill="grid", opacity=0.6)
                weight = head["weights"][r][c]
                if weight >= 0.02:
                    fig.rect(left + c * cell, top + r * cell, cell - 1, cell - 1, fill="s1", opacity=round(weight, 2))
    fig.text(24, 428, "The upper right is empty: a token cannot attend to the tokens after it. Head numbers count from zero.", size=11, fill="muted")
    fig.save("model-attention")


# ---------------------------------------------------------------- 3 pretraining


def pretraining_loss():
    log = [json.loads(line) for line in open("runs/base/log.jsonl")]
    train = [(r["step"], r["train_loss"]) for r in log if "train_loss" in r]
    valid = [(r["step"], r["val_loss"]) for r in log if "val_loss" in r]
    rates = [(r["step"], r["lr"]) for r in log if "lr" in r]
    steps = max(s for s, _ in valid)
    fig = Figure(720, 470, "The reference run", "Loss on a logarithmic scale. The first 500 steps do most of the work, and the rest is slow and matters.")
    fig.legend(24, 76, [("s1", "training loss, mean of 50 steps", "line"), ("s2", "validation loss, every 500 steps", "dot")])

    ax = Axes(fig, 64, 96, 600, 220, (0, steps), (1.3, 9), ylog=True)
    ax.yticks([1.5, 2, 3, 4, 6, 8], lambda t: f"{t:g}")
    ax.line(*zip(*train), "s1")
    ax.dots(*zip(*valid), "s2")
    ax.label(valid[0][0], valid[0][1], f"{valid[0][1]:.2f}, a uniform guess over 4,096 tokens", dx=12)
    ax.label(valid[-1][0], valid[-1][1], f"{valid[-1][1]:.3f}", dx=-4, dy=-12, anchor="end", fill="ink")

    lr = Axes(fig, 64, 352, 600, 60, (0, steps), (0, 0.00105))
    lr.title("Learning rate")
    lr.yticks([0, 0.001], lambda t: f"{t:g}")
    lr.xticks(range(0, steps + 1, 1000), lambda t: f"{t:,}", "training step")
    lr.line(*zip(*rates), "s1")
    fig.save("pretraining-loss")


def speed_bars(source, name, title, where):
    bench = load(source)
    configs = bench["configs"]
    fig = Figure(720, 232, title, f"Tokens per second {where}: forward, backward and optimizer step, {bench['batch_size']} × {bench['seq_len']} tokens.")
    top, row, bar = 78, 34, 18
    left, width = 214, 380
    peak = max(c["tokens_per_second"] for c in configs)
    slowest = min(c["tokens_per_second"] for c in configs)
    fig.line(left, top - 4, left, top + row * len(configs) - 8, stroke="axis")
    for i, c in enumerate(configs):
        y, w = top + i * row, width * c["tokens_per_second"] / peak
        fig.text(left - 10, y + bar / 2 + 4, c["name"], anchor="end", fill="ink")
        # Square at the baseline, rounded at the end that carries the value.
        fig.path(f"M{left} {y} H{n(left + w - 4)} a4 4 0 0 1 4 4 V{y + bar - 4} a4 4 0 0 1 -4 4 H{left} Z", fill="s1")
        k = c["tokens_per_second"] / 1000
        value = f"{k:.0f}K" if k >= 10 else f"{k:.1f}K"
        fig.text(left + w + 8, y + bar / 2 + 4, value, fill="ink")
        notes = [f"{c['tokens_per_second'] / slowest:.1f}×"] if c["tokens_per_second"] > slowest else []
        if bench.get("model") == "qwen" and c.get("memory_gb"):
            notes.append(f"{c['memory_gb']:.0f} GB")
        fig.text(left + w + 16 + text_width(value, 12), y + bar / 2 + 4, " · ".join(notes), fill="muted")
    fig.save(name)


# ------------------------------------------------------------------ 4 inference


def inference_work():
    fig = Figure(720, 352, "What generating the eighth token costs", "Each cell is one query reading one key. Rows are tokens, columns are the tokens they attend to.")
    T, cell = 8, 24

    def triangle(left, top, title, note, color, cached):
        fig.text(left, top - 30, title, fill="ink", weight=600)
        fig.text(left, top - 13, note, size=11)
        for r in range(T):
            for c in range(r + 1):
                x, y = left + c * cell, top + r * cell
                if cached and r < T - 1:
                    fig.rect(x + 0.5, y + 0.5, cell - 3, cell - 3, stroke="axis", rx=3)
                else:
                    fig.rect(x, y, cell - 2, cell - 2, fill=color, rx=3)
            fig.text(left - 8, top + r * cell + 15, str(r + 1), size=11, fill="muted", anchor="end")

    triangle(70, 112, "Without a cache", "all 8 rows are computed again: 36 cells", "s2", cached=False)
    triangle(410, 112, "With a cache", "only the new row is computed: 8 cells", "s1", cached=True)
    fig.text(410 + 4 * cell + 14, 112 + 2 * cell, "outlined: computed at earlier", size=11, fill="muted")
    fig.text(410 + 4 * cell + 14, 112 + 2 * cell + 14, "steps, keys and values kept", size=11, fill="muted")
    fig.text(24, 334, "For token t that is t(t + 1)/2 cells against t, in every layer. Over n tokens: about n³/6 cells against n²/2.", size=11, fill="muted")
    fig.save("inference-work")


def inference_cache():
    mps = load("runs/bench/inference-mps.json")["length"]
    cpu = load("runs/bench/inference-cpu.json")["length"]
    panels = [
        ("GPU, 1 sequence", [r for r in mps if r["batch"] == 1]),
        ("GPU, 64 sequences", [r for r in mps if r["batch"] == 64]),
        ("CPU, 1 sequence", [r for r in cpu if r["batch"] == 1]),
    ]
    fig = Figure(720, 330, "When the cache pays", "Seconds to generate with the 14M model. Each panel has its own scale.")
    fig.legend(24, 76, [("s2", "without a cache", "line"), ("s1", "with a cache", "line")])
    for i, (title, rows) in enumerate(panels):
        xs = [r["new_tokens"] for r in rows]
        slow, fast = [r["uncached_seconds"] for r in rows], [r["cached_seconds"] for r in rows]
        top = max(slow + fast)
        step = next(s for s in [0.2, 0.5, 1, 2, 5, 10, 20] if top / s <= 4)
        ax = Axes(fig, 56 + i * 226, 124, 170, 130, (0, 262), (0, step * math.ceil(top / step)))
        ax.title(title)
        ax.yticks([step * k for k in range(math.ceil(top / step) + 1)], lambda t: f"{t:g}")
        ax.xticks([0, 64, 128, 255], str, "tokens generated")
        ax.line(xs, slow, "s2")
        ax.line(xs, fast, "s1")
        ax.dots(xs[-1:], slow[-1:], "s2")
        ax.dots(xs[-1:], fast[-1:], "s1")
        ratio = slow[-1] / fast[-1]
        note = f"{ratio:.0f}× faster" if ratio >= 1.5 else "no gain"
        ax.label(xs[-1], max(slow[-1], fast[-1]), note, dx=-10, dy=-10, anchor="end", fill="ink")
    fig.save("inference-cache")


def inference_batch():
    rows = load("runs/bench/inference-mps.json")["batch"]
    xs, ys = [r["batch"] for r in rows], [r["tokens_per_second"] for r in rows]
    fig = Figure(720, 300, "Generating many sequences at once", "Tokens per second on the GPU, with a cache, 200 tokens per sequence. Both scales are logarithmic.")
    ax = Axes(fig, 72, 84, 580, 150, (0.75, 1400), (300, 40000), xlog=True, ylog=True)
    ax.yticks([300, 1000, 3000, 10000, 30000], thousands)
    ax.xticks(xs, lambda t: f"{t:,}", "sequences in the batch")
    ax.line(xs, ys, "s1")
    ax.dots(xs, ys, "s1")
    for x, y in zip(xs, ys):
        ax.label(x, y, f"{y:,.0f}", dx=0, dy=-12, anchor="middle")
    fig.save("inference-batch")


# ----------------------------------------------------------------- 5 pretrained


def pretrained_files():
    fig = Figure(720, 372, "A published model is three files", "Qwen2.5-0.5B-Instruct, and what each file becomes in this repository.")
    rows = [
        ("config.json", "659 bytes", "Config", "24 layers · dim 896 · 14 heads on 2 key/value heads"),
        ("model.safetensors", "988 MB", "the weights of Transformer", "290 tensors · 494,032,768 numbers · names translated"),
        ("vocab.json, merges.txt", "4.4 MB", "Tokenizer", "151,643 tokens from 151,387 merges · 22 special tokens"),
    ]
    for i, (file, size, target, note) in enumerate(rows):
        y = 74 + i * 58
        fig.box(24, y, 220, 44, file, size)
        fig.arrow([(244, y + 22), (286, y + 22)])
        fig.box(288, y, 408, 44, target, note, fill="s1", opacity=0.1)

    y = 286
    fig.text(24, y - 14, "Inside model.safetensors", fill="ink", weight=600)
    for x, w, title, note, fill in [
        (24, 94, "8 bytes", "the number n", None),
        (120, 212, "n = 32,280 bytes of JSON", "name → dtype, shape, byte range", None),
        (334, 362, "988 MB of raw numbers", "the tensors, one after another, in bfloat16", "s1"),
    ]:
        fig.box(x, y, w - 2, 44, title, note, fill=fill or "surface", opacity=0.1 if fill else None)
    fig.text(24, y + 68, '"model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [896, 896], "data_offsets": [300258816, 301864448]}', size=10, fill="muted")
    fig.save("pretrained-files")


def pretrained_generation():
    bench = load("runs/bench/inference-qwen-mps.json")
    fig = Figure(720, 330, "Generating with the 0.5B model", "Qwen2.5-0.5B-Instruct on the GPU with 16-bit weights. All scales are logarithmic.")
    fig.legend(64, 78, [("s1", "1 sequence", "line"), ("s2", "16 sequences", "line")])

    left = Axes(fig, 64, 124, 270, 130, (22, 3000), (0.3, 40), xlog=True, ylog=True)
    left.title("Speedup from the cache, 48 new tokens")
    left.yticks([1, 3, 10, 30], lambda t: f"{t}×")
    left.xticks([32, 512, 2048], lambda t: f"{t:,}", "tokens in the prompt")
    for batch, color in [(1, "s1"), (16, "s2")]:
        rows = [r for r in bench["length"] if r["batch"] == batch]
        xs = [r["prompt_tokens"] for r in rows]
        ys = [r["uncached_seconds"] / r["cached_seconds"] for r in rows]
        left.line(xs, ys, color)
        left.dots(xs, ys, color)
        left.label(xs[0], ys[0], f"{ys[0]:.1f}×", dx=10, dy=4 if batch == 16 else 16)
        left.label(xs[-1], ys[-1], f"{ys[-1]:.0f}×", dx=0, dy=-12, anchor="middle", fill="ink")

    rows = bench["batch"]
    xs, ys = [r["batch"] for r in rows], [r["tokens_per_second"] for r in rows]
    right = Axes(fig, 420, 124, 270, 130, (0.75, 180), (50, 6000), xlog=True, ylog=True)
    right.title("Tokens per second, with the cache")
    right.yticks([100, 300, 1000, 3000], thousands)
    right.xticks(xs, str, "sequences in the batch")
    right.line(xs, ys, "s1")
    right.dots(xs, ys, "s1")
    for i, (x, y) in enumerate(zip(xs, ys)):
        below = i == len(xs) - 2  # the last two points sit close together
        right.label(x, y, f"{y:,.0f}", dx=0, dy=20 if below else -12, anchor="middle")
    fig.save("pretrained-generation")


# ---------------------------------------------------------------- 6 fine-tuning


def finetuning_targets():
    # One training example of the words task, as the tokenizer of stage 1 cuts it.
    context = ["<|im_start|>", "us", "er", "\\n", "W", "r", "ite", " a", " short", " story", " using", " these", " words", ":"]
    context += [" garden", ",", " kitten", ".", "<|im_end|>", "\\n", "<|im_start|>", "ass", "ist", "ant", "\\n"]
    reply = ["Lily", " found", " a", " kitten", " in", " the", " garden", ".", " They", " played", " all", " day", ".", "<|im_end|>"]
    tokens = [(t, False) for t in context] + [(t, True) for t in reply] + [("\\n", False)]

    fig = Figure(720, 262, "What a fine-tuning step learns from", "One conversation, token by token. The model reads all of it and is trained only on the filled tokens.")
    x, y, height, gap = 24, 78, 26, 4
    for token, learned in tokens:
        label = token.replace(" ", "·") if token.startswith(" ") else token
        w = max(18, text_width(label, 11) + 14)
        if x + w > 696:
            x, y = 24, y + height + 8
        if learned:
            fig.rect(x, y, w, height, fill="s1", rx=5, opacity=0.9)
            fig.text(x + w / 2, y + 17, label, size=11, fill="surface", anchor="middle", weight=600)
        else:
            fig.rect(x + 0.5, y + 0.5, w - 1, height - 1, stroke="axis", rx=5)
            fig.text(x + w / 2, y + 17, label, size=11, fill="ink2", anchor="middle")
        x += w + gap
    fig.rect(24.5, 196.5, 17, 17, stroke="axis", rx=4)
    fig.text(50, 210, "context: the instruction and the format around it", size=11)
    fig.rect(364, 196, 18, 18, fill="s1", rx=4, opacity=0.9)
    fig.text(390, 210, "target: the reply, and the <|im_end|> that ends it", size=11)
    fig.text(24, 242, "A dot marks a leading space. 14 of the 40 tokens are targets, and the output layer is applied to those alone.", size=11, fill="muted")
    fig.save("finetuning-targets")


def finetuning_lora():
    fig = Figure(720, 300, "LoRA trains a thin update next to a frozen matrix", "One query projection of the 0.5B model. Each of the 168 linear layers in its 24 blocks gets such a pair.")
    # the frozen matrix
    fig.rect(60, 86, 150, 150, fill="grid", stroke="axis", rx=4, opacity=0.6)
    fig.text(135, 156, "W", size=22, fill="ink", anchor="middle", weight=600)
    fig.text(135, 178, "896 × 896", size=11, anchor="middle")
    fig.text(135, 256, "frozen: 802,816 numbers", size=11, anchor="middle")
    fig.text(246, 168, "+", size=24, fill="ink", anchor="middle")
    # B: tall and thin, A: wide and thin
    fig.rect(292, 86, 14, 150, fill="s1", rx=3)
    fig.text(299, 256, "B", size=13, fill="ink", anchor="middle", weight=600)
    fig.text(299, 272, "896 × 16", size=11, anchor="middle")
    fig.text(332, 168, "·", size=24, fill="ink", anchor="middle")
    fig.rect(358, 86, 150, 14, fill="s1", rx=3)
    fig.text(433, 122, "A", size=13, fill="ink", anchor="middle", weight=600)
    fig.text(433, 138, "16 × 896", size=11, anchor="middle")
    fig.text(433, 256, "trained: 28,672 numbers, 3.6%", size=11, anchor="middle")
    # what it buys
    notes = ["B starts at zero, so training", "starts from exactly the", "pretrained model.", ""]
    notes += ["Switched off, the model is", "the pretrained one again.", "Merged into W, it is an", "ordinary model."]
    for i, line in enumerate(notes):
        fig.text(532, 104 + i * 17, line, size=11)
    fig.save("finetuning-lora")


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def bar(fig, x, y, w, height, color):
    """A horizontal bar, square at its baseline and rounded at its data end."""
    w = max(w, 5)
    fig.path(f"M{n(x)} {n(y)} H{n(x + w - 4)} a4 4 0 0 1 4 4 V{n(y + height - 4)} a4 4 0 0 1 -4 4 H{n(x)} Z", fill=color)


def finetuning_gsm8k():
    runs = [("LoRA, rank 16", "s1", "runs/sft-gsm8k-lora"), ("every weight", "s2", "runs/sft-gsm8k")]
    fig = Figure(720, 330, "Two ways to fine-tune the 0.5B model on GSM8K", "The same 900 steps of 8 problems. LoRA trains 1.8% of the weights.")
    fig.legend(64, 78, [(color, name, "line") for name, color, _ in runs])

    ax = Axes(fig, 64, 124, 280, 130, (0, 900), (0.5, 0.9))
    ax.title("Loss on held-out problems")
    ax.yticks([0.5, 0.6, 0.7, 0.8, 0.9], lambda t: f"{t:g}")
    ax.xticks([0, 300, 600, 900], str, "training step")
    metas = []
    for name, color, path in runs:
        log = [r for r in jsonl(f"{path}/log.jsonl") if "val_loss" in r]
        xs, ys = [r["step"] for r in log], [r["val_loss"] for r in log]
        ax.line(xs, ys, color)
        ax.dots(xs, ys, color)
        ax.label(xs[-1], ys[-1], f"{ys[-1]:.3f}", dx=0, dy=-12 if color == "s2" else 20, anchor="end", fill="ink")
        metas.append(load(f"{path}/meta.json"))

    # What each run costs, one small pair of bars per measure.
    measures = [
        ("Weights trained", [m["trainable_parameters"] / 1e6 for m in metas], lambda v: f"{v:,.1f}M" if v < 100 else f"{v:,.0f}M"),
        ("Minutes", [m["minutes"] for m in metas], lambda v: f"{v:.0f}"),
        ("Peak memory pool, GB", [m["memory_gb"] for m in metas], lambda v: f"{v:.0f}"),
        ("Saved file, MB", [35, 988], lambda v: f"{v:,.0f}"),
    ]
    left, width, top = 420, 190, 108
    for i, (title, values, fmt) in enumerate(measures):
        y = top + i * 50
        fig.text(left, y, title, fill="ink", weight=600)
        for j, (value, (_, color, _)) in enumerate(zip(values, runs)):
            bar(fig, left, y + 8 + j * 14, width * value / max(values), 10, color)
            fig.text(left + max(width * value / max(values), 5) + 8, y + 17 + j * 14, fmt(value), size=11)
    fig.save("finetuning-gsm8k")


# ---------------------------------------------------------------- 7 evaluation


def evaluation_scores():
    fig = Figure(720, 260, "What fine-tuning did, measured on the task", "Greedy replies to held-out prompts, with 95% confidence intervals. It helped on one task and hurt on the other.")
    panels = [
        ("Words: requested words used", [("14M base", "runs/eval/words-base.json"), ("fine-tuned", "runs/eval/words-sft-words.json")]),
        ("GSM8K: problems solved", [("0.5B as published", "runs/eval/gsm8k-qwen.json"), ("fine-tuned, LoRA", "runs/eval/gsm8k-sft-gsm8k-lora.json"), ("every weight", "runs/eval/gsm8k-sft-gsm8k.json")]),
    ]
    for i, (title, rows) in enumerate(panels):
        left, top, width = 150 + i * 350, 100, 150
        fig.text(left - 126, top - 22, title, fill="ink", weight=600)
        fig.line(left, top - 8, left, top + 40 * len(rows) - 8, stroke="axis")
        for j, (name, path) in enumerate(rows):
            r = load(path)
            y = top + j * 40
            scale = lambda v: width * v / 0.6
            fig.text(left - 10, y + 14, name, anchor="end", fill="ink")
            bar(fig, left, y, scale(r["mean"]), 18, "s1")
            # the confidence interval, as a thin line through the bar's end
            fig.line(left + scale(r["mean"] - r["error"]), y + 9, left + scale(r["mean"] + r["error"]), y + 9, stroke="ink", width=1.5)
            fig.text(left + scale(r["mean"] + r["error"]) + 8, y + 14, f"{r['mean']:.1%}", fill="ink")
            fig.text(left + scale(r["mean"] + r["error"]) + 8 + text_width(f"{r['mean']:.1%}", 12) + 6, y + 14, f"± {r['error']:.1%}", fill="muted", size=11)
        fig.text(left - 126, top + 40 * len(rows) + 18, f"{load(rows[0][1])['n']:,} examples", size=11, fill="muted")
    fig.save("evaluation-scores")


# --------------------------------------------------------- 6a tool calling


def tools_results():
    """GSM8K solved with and without the calculator, and when its calls fail."""
    summary = load("runs/eval/tools-summary.json")
    rows = [
        ("0.5B as published", load("runs/eval/gsm8k-qwen.json"), None),
        ("the same, told about the tool", load("runs/eval/gsm8k-tools-qwen-system-markers.json"), summary["runs"]["gsm8k-tools-qwen-system-markers"]["with_a_call"]),
        ("told, its markers spelled out", load("runs/eval/gsm8k-tools-qwen.json"), summary["runs"]["gsm8k-tools-qwen"]["with_a_call"]),
        ("fine-tuned on solutions (stage 6)", load("runs/eval/gsm8k-sft-gsm8k-lora.json"), None),
        ("fine-tuned on solutions with calls", load("runs/eval/gsm8k-tools-sft-gsm8k-tools-lora.json"), summary["runs"]["gsm8k-tools-sft-gsm8k-tools-lora"]["with_a_call"]),
        ("the same, every call fails", load("runs/eval/gsm8k-tools-sft-gsm8k-tools-lora-no-execute.json"), summary["runs"]["gsm8k-tools-sft-gsm8k-tools-lora-no-execute"]["with_a_call"]),
    ]
    fig = Figure(720, 356, "What the calculator is worth", "GSM8K's 1,319 test problems, with 95% intervals, and the share of problems with a call.")
    left, top, width, peak = 250, 84, 300, 0.6
    scored_bars(fig, left, top, width, [(label, r["mean"], r["error"]) for label, r, _ in rows], peak, lambda v: f"{v:.1%}")
    fig.text(640, top - 10, "calls", fill="ink", weight=600, anchor="middle")
    for j, (_, _, called) in enumerate(rows):
        if called is not None:
            fig.text(640, top + j * 36 + 14, f"{called:.1%}", fill="ink2", anchor="middle")
    fig.text(24, 322, "Both fine-tuned models trained on the same problems with LoRA, as in stage 6: one wrote the arithmetic, the other called for it.", size=11, fill="muted")
    fig.text(24, 338, "Told about the tool, its instructions' markers are single tokens, as Qwen's tokenizer writes them, or spelled out.", size=11, fill="muted")
    fig.save("tools-results")


# ---------------------------------------------------- 4c continuous batching


def batching_timeline():
    """Slots over decode steps, for batches that start and end together and for a running batch. The reply lengths are made up."""
    lengths = [5, 11, 4, 8, 6, 12, 3, 9, 7, 5, 10, 4, 6, 8, 5, 9]  # of the requests, in the order they arrive
    slots, unit, left = 4, 12, 24
    fig = Figure(720, 316, "Four slots, sixteen requests", "Rows are the slots of a batch, columns are decode steps. The reply lengths are made up for the drawing.")
    for k, (title, top) in enumerate([("Batches that start and end together", 92), ("A running batch", 196)]):
        fig.text(24, top - 6, title, fill="ink", weight=600)
        if k == 0:  # batch after batch: every row waits for the longest
            start, waiting = 0, list(lengths)
            while waiting:
                batch, waiting = waiting[:slots], waiting[slots:]
                longest = max(batch)
                for row, n in enumerate(batch):
                    y = top + 4 + row * 18
                    fig.rect(left + start * unit, y, n * unit - 1.5, 14, fill="s1", rx=2)
                    if n < longest:
                        fig.rect(left + (start + n) * unit, y, (longest - n) * unit - 1.5, 14, fill="grid", rx=2)
                start += longest
            end = start
        else:  # a request takes the slot that frees first
            free = [0] * slots
            for n in lengths:
                row = min(range(slots), key=lambda r: free[r])
                fig.rect(left + free[row] * unit, top + 4 + row * 18, n * unit - 1.5, 14, fill="s2", rx=2)
                free[row] += n
            end = max(free)
        fig.text(left + end * unit + 8, top + 4 + 2 * 18 + 2, f"{end} steps", fill="ink")
    fig.legend(24, 296, [("s1", "writing a reply", "dot"), ("grid", "done, waiting for the others", "dot"), ("s2", "writing a reply, in a running batch", "dot")])
    fig.save("batching-timeline")


def batching_results():
    """Batches that start and end together against a running batch, for both models of the benchmark."""
    models = [("runs/bench/batching-qwen-mps.json", "Qwen2.5-0.5B, 256 GSM8K problems"), ("runs/bench/batching-sft-words-mps.json", "The story model of stage 6, 512 requests")]
    measures = [("seconds", "All requests done"), ("latency_mean", "Until its reply ends, mean"), ("first_token_mean", "Until its first token, mean")]
    cases = [(16, "at once"), (16, "at random"), (64, "at once"), (64, "at random")]
    left, panel, gap, group = 150, 160, 24, 36
    section = 44 + group * len(cases)
    fig = Figure(720, 104 + section * len(models), "Batches that end together, and a running batch", "Seconds on an M5 Pro. The requests come all at once, or at random, 4 a second for Qwen and 40 for the story model.")
    fig.legend(24, 80, [("s1", "batches that start and end together", "dot"), ("s2", "a running batch", "dot")])
    top = 114
    for path, title in models:
        b = load(path)
        runs = {(r["batching"], r["slots"], r["arrival"]): r for r in b["runs"]}
        fig.text(24, top, title, fill="ink", weight=600)
        for j, (key, name) in enumerate(measures):
            x = left + j * (panel + gap)
            fig.text(x, top + 20, name, size=11, fill="muted")
            peak = max(r[key] for r in b["runs"])
            for i, (slots, arrival) in enumerate(cases):
                y = top + 30 + i * group
                if j == 0:
                    fig.text(x - 10, y + 17, f"{slots} slots, {arrival}", anchor="end", fill="ink")
                for k, kind in enumerate(("static", "continuous")):
                    value = runs[kind, slots, arrival][key]
                    w = (panel - 44) * value / peak
                    bar(fig, x, y + k * 14, w, 12, "s1" if kind == "static" else "s2")
                    fig.text(x + max(w, 5) + 6, y + k * 14 + 10, f"{value:.1f}" if value >= 0.1 else f"{value:.2f}", size=11, fill="ink")
        top += section
    fig.save("batching-results")


# ------------------------------------------------------ 4a structured output

FORMS = [
    ("thinking", "a JSON object: reasoning, then answer"),
    ("answer", "a JSON object: the answer alone"),
    ("answer-first", "a JSON object: answer, then reasoning"),
    ("reason-then-answer", "free text, then the answer as JSON"),
]


def structured_forms():
    """For every form: how often the reply has it, and how often it is right, asked in the prompt and held to it."""
    runs = {(form, how): load(f"runs/structured/{form}-{how}.json") for form, _ in FORMS for how in ("asked", "constrained")}
    free = load("runs/eval/gsm8k-qwen.json")
    fig = Figure(720, 330, "Asking for a form, and holding a reply to it", f"The 0.5B model on GSM8K's {free['n']:,} test problems. Left: replies in the asked form. Right: right answers, read from the form.")
    fig.legend(250, 80, [("axis", "asked in the prompt", "dot"), ("s2", "held to the form", "dot"), ("s1", "free text, as in stage 7", "line")])
    panels = [("In the form", "in_form", 250, 1.0), ("Right", "mean", 500, 0.6)]
    top = 112
    for title, key, left, peak in panels:
        fig.text(left, top - 8, title, fill="ink", weight=600)
        fig.line(left, top, left, top + 4 * 44, stroke="axis")
        for j, (form, label) in enumerate(FORMS):
            y = top + 8 + j * 44
            if left == 250:
                fig.text(left - 10, y + 15, label, anchor="end", fill="ink", size=11)
            for k, (how, color) in enumerate([("asked", "axis"), ("constrained", "s2")]):
                value = runs[(form, how)][key]
                width = 150 * value / peak
                bar(fig, left, y + k * 15, width, 12, color)
                fig.text(left + max(width, 5) + 6, y + k * 15 + 10, f"{value:.0%}" if key == "in_form" else f"{value:.1%}", size=11, fill="ink" if k else "ink2")
    x = 500 + 150 * free["mean"] / 0.6
    fig.line(x, top, x, top + 4 * 44, stroke="s1", width=1.5)
    fig.text(x + 5, top - 8, f"{free['mean']:.1%}", size=11, fill="s1")
    fig.text(24, 314, "A form leaves the model room to reason, or none. Where it leaves none, holding the reply to it cannot help.", size=11, fill="muted")
    fig.save("structured-forms")


# ------------------------------------------------------------- 7b the judge


def judge_gsm8k():
    """How often a judge approves right and wrong solutions, by its size and whether it is told the answer."""
    rows = []
    for label, name in [("0.5B", "gsm8k-qwen"), ("0.5B, told the answer", "gsm8k-qwen-reference"), ("1.5B", "gsm8k-qwen2.5-1.5b-instruct"), ("1.5B, told the answer", "gsm8k-qwen2.5-1.5b-instruct-reference")]:
        rows.append((label, load(f"runs/judge/{name}.json")))
    n = rows[0][1]["n"]
    fig = Figure(720, 336, "A judge of math solutions", f"The verdicts on the 0.5B model's {n:,} replies to GSM8K's test problems, of which {rows[0][1]['truly_right']:.1%} are right.")
    fig.legend(64, 80, [("s3", "approves a right solution", "dot"), ("s2", "approves a wrong one", "dot"), ("s1", "agrees with the truth", "dot")])
    left, top, width = 220, 104, 330
    fig.line(left, top - 6, left, top + 4 * 50 - 10, stroke="axis")
    for j, (label, r) in enumerate(rows):
        y = top + j * 50
        fig.text(left - 10, y + 18, label, anchor="end", fill="ink")
        for k, (key, color) in enumerate([("approves_right", "s3"), ("approves_wrong", "s2"), ("accuracy", "s1")]):
            value = r[key]
            bar(fig, left, y + k * 12, width * value, 10, color)
            fig.text(left + max(width * value, 5) + 6, y + k * 12 + 9, f"{value:.0%}", size=10, fill="ink2")
    fig.text(24, 320, "A judge that approved everything would agree with the truth as often as the replies are right.", size=11, fill="muted")
    fig.save("judge-gsm8k")


def judge_stories():
    """A judge of stories against the annotator of stage 8, and its preference for the story shown first."""
    reward = load("runs/rlhf/reward.json")
    rows = []
    for label, name in [("0.5B", "stories-qwen"), ("0.5B, told the taste", "stories-qwen-rubric"), ("1.5B", "stories-qwen2.5-1.5b-instruct"), ("1.5B, told the taste", "stories-qwen2.5-1.5b-instruct-rubric")]:
        rows.append((label, load(f"runs/judge/{name}.json")))
    n = rows[0][1]["n"]
    fig = Figure(720, 316, "A judge of stories, against the annotator", f"The {n} held-out comparisons of stage 8, each judged in both orders.")
    left, top, width, peak = 220, 92, 160, 1.0
    fig.text(left, top - 10, "Agrees with the annotator", fill="ink", weight=600)
    fig.text(left + 260, top - 10, "Picks the story shown first", fill="ink", weight=600)
    for j, (label, r) in enumerate(rows + [("reward model of stage 8", {"agrees_with_annotator": reward["agrees_with_annotator"]})]):
        y = top + j * 36
        fig.text(left - 10, y + 13, label, anchor="end", fill="ink")
        value = r["agrees_with_annotator"]
        bar(fig, left, y, width * value / peak, 16, "s1" if "picks_slot_a" in r else "axis")
        fig.text(left + width * value / peak + 6, y + 13, f"{value:.1%}", fill="ink", size=11)
        if "picks_slot_a" in r:
            first = r["picks_slot_a"]
            bar(fig, left + 260, y, width * first, 16, "s2")
            fig.text(left + 260 + width * first + 6, y + 13, f"{first:.0%}", fill="ink", size=11)
    for x, value, label in [(left + width * 0.5, 0.5, "chance"), (left + 260 + width * 0.5, 0.5, "no preference")]:
        fig.line(x, top - 2, x, top + 5 * 36 - 8, stroke="muted", width=1)
        fig.text(x, top + 5 * 36 + 6, label, size=11, fill="muted", anchor="middle")
    fig.text(24, 300, "The reward model learned from 3,500 comparisons. The judges were told the request, and in one case what the annotator likes.", size=11, fill="muted")
    fig.save("judge-stories")


# ---------------------------------------------------------------- 8a safety

SAFETY = [
    ("as published", "qwen", "gsm8k-qwen"),
    ("fine-tuned on GSM8K, every weight", "sft-gsm8k", "gsm8k-sft-gsm8k"),
    ("fine-tuned on GSM8K, LoRA", "sft-gsm8k-lora", "gsm8k-sft-gsm8k-lora"),
    ("every weight, 3% refusals mixed in", "sft-gsm8k-safe", "gsm8k-sft-gsm8k-safe"),
]


def safety_refusals():
    """How often each model refuses unsafe and safe requests, and what it does on GSM8K."""
    runs = [(label, load(f"runs/safety/xstest-{name}.json"), load(f"runs/eval/{task}.json")) for label, name, task in SAFETY]
    first = runs[0][1]
    fig = Figure(720, 316, "What fine-tuning on math does to refusals", f"The 0.5B model on XSTest's {sum(l == 'unsafe' for l in first['labels'])} unsafe requests and {sum(l == 'safe' for l in first['labels'])} safe ones that sound alike, and on GSM8K.")
    fig.legend(250, 80, [("s3", "refuses an unsafe request", "dot"), ("s2", "refuses a safe one", "dot")])
    left, top, width = 250, 104, 230
    fig.text(left + width + 70, top - 4, "GSM8K solved", fill="ink", weight=600)
    fig.line(left, top - 6, left, top + 4 * 44 - 6, stroke="axis")
    for j, (label, r, g) in enumerate(runs):
        y = top + j * 44
        fig.text(left - 10, y + 16, label, anchor="end", fill="ink", size=11)
        for k, (key, color) in enumerate([("refuses_unsafe", "s3"), ("refuses_safe", "s2")]):
            value = r[key]
            bar(fig, left, y + k * 15, width * value, 12, color)
            fig.text(left + max(width * value, 5) + 6, y + k * 15 + 10, f"{value:.1%}", size=11, fill="ink2")
        fig.text(left + width + 70, y + 18, f"{g['mean']:.1%}", fill="ink")
    fig.text(24, 300, "A refusal is read off the opening of a reply, as XSTest's authors did automatically. Replies to the unsafe requests are not kept.", size=11, fill="muted")
    fig.save("safety-refusals")


# -------------------------------------------------------- 8b DPO on GSM8K

DPO_RUNS = [
    ("lr 5e-5", "s2", "runs/dpo/dpo.jsonl"),
    ("lr 5e-5, with the right reply's own loss", "s1", "runs/dpo/dpo-nll.jsonl"),
    ("lr 1e-5", "s3", "runs/dpo/dpo-lr1e-5.jsonl"),
]


def dpo_curves():
    """Held-out score over the steps of three DPO runs, and how much likelier their pairs' right and wrong replies became."""
    pairs = load("runs/dpo/pairs.json")
    args = load("runs/dpo/dpo-meta.json")["args"]
    fig = Figure(720, 560, "DPO on the verifier's pairs", f"{pairs['pairs']} pairs of a right and a wrong reply, {args['epochs']} passes of {args['batch_size']} pairs a step, at two learning rates.")
    fig.legend(64, 78, [(color, label, "line") for label, color, _ in DPO_RUNS])
    logs = [(label, color, jsonl(path)) for label, color, path in DPO_RUNS]
    steps = max(r["step"] for r in logs[0][2])
    per_pass = pairs["pairs"] // args["batch_size"]
    held = [r["held_out"] for _, _, log in logs for r in log if "held_out" in r]
    lo, hi = math.floor(min(held) * 20) / 20, math.ceil(max(held) * 20) / 20
    ax = Axes(fig, 72, 124, 560, 100, (0, steps), (lo, hi))
    ax.title("Solved, of 200 held-out training problems: a signal of progress, ± 7 points")
    ax.yticks(nice_ticks(lo, hi, 3), lambda t: f"{t:.0%}")
    ax.xticks([], str)
    for label, color, log in logs:
        points = [(r["step"], r["held_out"]) for r in log if "held_out" in r]
        ax.line(*zip(*points), color)
        ax.dots(*zip(*points), color, r=3.5)

    # how much likelier each reply has become than under the starting model, a running mean over 5 steps
    smooth = lambda rows, key: [sum(q[key] for q in rows[max(0, j - 4) : j + 1]) / len(rows[max(0, j - 4) : j + 1]) for j in range(len(rows))]
    trained = [(label, color, [r for r in log if "gain_preferred" in r]) for label, color, log in logs]
    for i, (key, heading) in enumerate([("gain_preferred", "The right replies: log p − log p_start, in nats"), ("gain_other", "The wrong replies")]):
        values = [v for _, _, rows in trained for v in smooth(rows, key)]
        lo, hi = math.floor(min(values) / 5) * 5, max(5, math.ceil(max(values) / 5) * 5)
        low = Axes(fig, 72, 282 + i * 140, 560, 90, (0, steps), (lo, hi))
        low.title(heading)
        low.yticks(nice_ticks(lo, hi, 3), lambda t: f"{t:+g}" if t else "0")
        low.xticks(range(0, steps + 1, 10), str, "step" if i else "")
        fig.line(low.x(per_pass), low.top, low.x(per_pass), low.top + low.height, stroke="axis")
        if not i:
            fig.text(low.x(per_pass) + 6, low.top + 12, "second pass", size=11, fill="muted")
        for label, color, rows in trained:
            ys = smooth(rows, key)
            low.line([r["step"] for r in rows], ys, color)
            low.label(rows[-1]["step"], ys[-1], f"{ys[-1]:+.0f}", dx=8, dy=4)
    fig.save("dpo-curves")


def dpo_test():
    fig = Figure(720, 396, "GSM8K test problems solved", "1,319 problems, greedy, 95% intervals. Each model below the start learned from replies to 480 problems.")
    rows = []
    for label, name in [
        ("0.5B as published", "gsm8k-qwen"),
        ("fine-tuned on solutions, the start", "gsm8k-sft-gsm8k-lora"),
        ("self-training (9a), two rounds", "gsm8k-self-gsm8k"),
        ("DPO, lr 2e-6", "gsm8k-dpo-gsm8k-lr2e-6"),
        ("DPO, lr 1e-5", "gsm8k-dpo-gsm8k-lr1e-5"),
        ("DPO, lr 5e-5, one pass", "gsm8k-dpo-gsm8k-1"),
        ("DPO, lr 5e-5, two passes", "gsm8k-dpo-gsm8k"),
        ("DPO + the right reply's loss, 5e-5", "gsm8k-dpo-gsm8k-nll"),
        ("GRPO (stage 9)", "gsm8k-grpo-gsm8k"),
    ]:
        r = load(f"runs/eval/{name}.json")
        rows.append((label, r["mean"], r["error"]))
    scored_bars(fig, 250, 84, 300, rows, 0.6, lambda v: f"{v:.1%}")
    fig.save("dpo-gsm8k")


# -------------------------------------------------------- 7a multiple choice

MMLU = "runs/eval/mmlu-summary.json"  # written by scripts/analyze_choice.py


def guessing(fig, left, top, width, peak, rows):
    """A line through the bars at what guessing would score."""
    x = left + width * 0.25 / peak
    fig.line(x, top - 8, x, top + 36 * rows - 12, stroke="muted", width=1)
    fig.text(x, top + 36 * rows + 4, "guessing, 25%", size=11, fill="muted", anchor="middle")


def choice_methods():
    """One model and one set of questions, scored eight ways."""
    m = load(MMLU)
    runs, rules, written = m["runs"], m["cloze"]["rules"], m["written"]
    rows = [
        ("letter after a text prompt, 5 examples", runs["qwen-letter-plain-5shot"]),
        ("the same, no examples", runs["qwen-letter-plain-0shot"]),
        ("letter in a chat turn, no examples", runs["qwen-letter-chat-0shot"]),
        ("the same, 5 examples", runs["qwen-letter-chat-5shot"]),
        ("a written reply in a chat turn", written),
        ("an option's text, summed", rules["sum"]),
        ("an option's text, per token", rules["per_token"]),
        ("an option's text, less its own likelihood", rules["minus_unconditional"]),
    ]
    fig = Figure(720, 424, "One model, one benchmark, eight scores", f"The 0.5B model on the {m['n']:,} test questions of MMLU, with 95% confidence intervals.")
    left, top, width, peak = 280, 84, 300, 0.6
    scored_bars(fig, left, top, width, [(label, r["mean"], r["error"]) for label, r in rows], peak, lambda v: f"{v:.1%}")
    guessing(fig, left, top, width, peak, len(rows))
    fig.text(24, 388, "The first row is the protocol that the authors of MMLU published. The last three rows show no options to choose from,", size=11, fill="muted")
    fig.text(24, 404, f"and score each option's text as a continuation. The written replies are to {written['n']:,} of the questions, 20 per subject.", size=11, fill="muted")
    fig.save("choice-methods")


def choice_letters():
    """Which letter the model picks, and what moving the options round does."""
    m = load(MMLU)
    orders = m["orders"]
    fig = Figure(720, 330, "The letter matters, not only the answer", "The 0.5B model on MMLU, five examples. Right: the same questions with their options moved round.")
    # left: how often each letter is right, and how often it is picked
    left, top, height = 64, 96, 150
    fig.text(left, top - 14, "Share of the questions", fill="ink", weight=600)
    for t in (0, 0.1, 0.2, 0.3):
        y = top + height * (1 - t / 0.35)
        fig.line(left, y, left + 250, y)
        fig.text(left - 8, y + 4, f"{t:.0%}", size=11, fill="muted", anchor="end")
    for i, letter in enumerate("ABCD"):
        x = left + 22 + i * 58
        for j, (value, color) in enumerate([(m["right_letter"][i], "axis"), (orders["scores"][0]["picked_letter"][i], "s1")]):
            h = height * value / 0.35
            fig.rect(x + j * 20, top + height - h, 18, h, fill=color, rx=3)
            fig.text(x + j * 20 + 9 + (5 if j else -5), top + height - h - 5, f"{value:.0%}", size=10, anchor="middle", fill="ink" if j else "muted")
        fig.text(x + 19, top + height + 17, letter, anchor="middle", fill="ink", weight=600)
    fig.line(left, top + height, left + 250, top + height, stroke="axis")
    fig.legend(left, top + height + 44, [("axis", "the right answer", "dot"), ("s1", "the model's pick", "dot")])
    # right: the score in every order, and how many questions survive all four
    rows = [(f"options moved by {k}" if k else "as published", r["mean"], r["error"]) for k, r in enumerate(orders["scores"])]
    rows += [("right in all four orders", orders["right_in_all"], 0), ("the four orders added up", orders["added_up"], 0)]
    scored_bars(fig, 520, 92, 120, rows, 0.6, lambda v: f"{v:.1%}")
    fig.text(24, 314, f"A question answered right in no order: {orders['right_in_none']:.1%}. Guessing would be right in all four orders for 0.4% of the questions.", size=11, fill="muted")
    fig.save("choice-letters")


def choice_subjects():
    """The score of each of the 57 subjects, with what its number of questions lets one say."""
    m = load(MMLU)
    rows, overall = m["by_subject"], m["runs"]["qwen-letter-plain-5shot"]["mean"]
    fig = Figure(720, 330, "The 57 subjects of MMLU, from the worst to the best", f"The 0.5B model, five examples: {overall:.1%} over all {m['n']:,} questions. A subject has between {min(r['n'] for r in rows)} and {max(r['n'] for r in rows):,} of them.")
    ax = Axes(fig, 64, 84, 620, 190, (-1, len(rows)), (0, 1))
    ax.yticks([0, 0.25, 0.5, 0.75, 1], lambda t: f"{t:.0%}")
    fig.line(ax.left, ax.y(0.25), ax.left + ax.width, ax.y(0.25), stroke="muted", width=1)
    fig.line(ax.left, ax.y(overall), ax.left + ax.width, ax.y(overall), stroke="s2", width=1.5)
    fig.text(ax.left + ax.width, ax.y(overall) - 6, "all questions", size=11, fill="s2", anchor="end")
    fig.text(ax.left + ax.width, ax.y(0.25) + 14, "guessing", size=11, fill="muted", anchor="end")
    for i, r in enumerate(rows):
        fig.line(ax.x(i), ax.y(max(0, r["mean"] - r["error"])), ax.x(i), ax.y(min(1, r["mean"] + r["error"])), stroke="axis", width=1.5)
        fig.dot(ax.x(i), ax.y(r["mean"]), "s1", r=3)
    fig.line(ax.left, ax.top + ax.height, ax.left + ax.width, ax.top + ax.height, stroke="axis")
    name = lambda r: f"{r['subject'].replace('_', ' ')}, {r['mean']:.0%} of {r['n']}"
    for j, r in enumerate(rows[:2]):
        fig.text(ax.x(0) + 6, ax.y(0.12) + 14 * j, name(r), size=11, fill="ink")
    for j, r in enumerate(rows[:-3:-1]):
        fig.text(ax.x(len(rows) - 5), ax.y(0.98) + 14 * j, name(r), size=11, anchor="end", fill="ink")
    inside = sum(r["mean"] - r["error"] <= overall <= r["mean"] + r["error"] for r in rows)
    fig.text(24, 314, f"A line is a subject's 95% interval. For {inside} of the 57 subjects it includes the score over all questions.", size=11, fill="muted")
    fig.save("choice-subjects")


def choice_calibration():
    """Is the model as often right as it says it is sure?"""
    m = load(MMLU)
    groups = m["calibration"]["groups"]
    fig = Figure(720, 352, "How sure the model says it is, and how often it is right", "The 0.5B model on MMLU, five examples. Its confidence is the probability of the letter it picks, among the four letters.")
    ax = Axes(fig, 64, 92, 300, 180, (0.25, 1), (0, 1))
    ax.yticks([0, 0.25, 0.5, 0.75, 1], lambda t: f"{t:.0%}")
    ax.xticks([0.25, 0.5, 0.75, 1], lambda t: f"{t:.0%}", "confidence in its pick")
    ax.title("Share answered right")
    ax.line([0.25, 1], [0.25, 1], "axis", width=1.5)
    ax.line([g["confidence"] for g in groups], [g["right"] for g in groups], "s1")
    ax.dots([g["confidence"] for g in groups], [g["right"] for g in groups], "s1")
    ax.label(1, 1, "as sure as right", dx=-8, dy=18, anchor="end", fill="muted")
    # how the questions spread over the confidences
    ax2 = Axes(fig, 440, 92, 240, 180, (0.25, 1), (0, max(g["share"] for g in groups) * 1.15))
    ax2.title("Share of the questions")
    ax2.xticks([0.25, 0.5, 0.75, 1], lambda t: f"{t:.0%}", "confidence in its pick")
    for i, g in enumerate(groups):
        h = ax2.height * g["share"] / ax2.ylim[1]
        fig.rect(ax2.x(0.25 + i * 0.09375) + 1, ax2.top + ax2.height - h, ax2.width / 8 - 2, h, fill="s1", rx=3)
        fig.text(ax2.x(0.25 + (i + 0.5) * 0.09375), ax2.top + ax2.height - h - 5, f"{g['share']:.0%}", size=10, anchor="middle")
    fig.text(24, 336, f"On average the confidence is {m['calibration']['gap']:.1%} away from the share that is right.", size=11, fill="muted")
    fig.save("choice-calibration")


def choice_models():
    m = load(MMLU)
    runs = m["runs"]
    rows = [(label, runs[name]["mean"], runs[name]["error"]) for label, name in [("14M story model, no examples", "base-letter-plain-0shot"), ("0.5B as published", "qwen-letter-plain-5shot"), ("0.5B at 4 bits", "qwen-q4-letter-plain-5shot"), ("0.5B fine-tuned on GSM8K, LoRA", "sft-gsm8k-lora-letter-plain-5shot")]]
    fig = Figure(720, 250, "Four models on MMLU", f"The letter after a text prompt, with 95% confidence intervals. One model takes {runs['qwen-letter-plain-5shot']['seconds'] / 60:.0f} minutes for the {m['n']:,} questions.")
    left, top, width, peak = 250, 84, 320, 0.6
    scored_bars(fig, left, top, width, rows, peak, lambda v: f"{v:.1%}")
    guessing(fig, left, top, width, peak, len(rows))
    fig.save("choice-models")


# --------------------------------------------------------------------- 8 rlhf


def rlhf_pipeline():
    fig = Figure(720, 344, "From comparisons to a better model", "A reward model generalizes the annotator's choices, and two methods put it to use. DPO does without it.")
    fig.box(24, 78, 190, 50, "Two stories", "per prompt, from the model")
    fig.arrow([(214, 103), (244, 103)])
    fig.box(246, 78, 190, 50, "Annotator", "picks the one they prefer")
    fig.arrow([(436, 103), (466, 103)])
    fig.box(468, 78, 228, 50, "Comparisons", "4,000 of them")

    fig.arrow([(582, 128), (582, 160)])
    fig.box(468, 162, 228, 50, "Reward model", "a score for any story", fill="s1", opacity=0.1)
    fig.arrow([(601, 212), (601, 246)])
    fig.box(506, 248, 190, 50, "PPO", "train the model on rewards")
    fig.arrow([(500, 212), (500, 228), (375, 228), (375, 246)])
    fig.box(280, 248, 190, 50, "Rejection sampling", "write n stories, keep the best")

    fig.arrow([(490, 128), (490, 144), (119, 144), (119, 246)])
    fig.box(24, 248, 190, 50, "DPO", "train the model on comparisons", fill="s2", opacity=0.1)
    fig.text(131, 200, "without a", size=11, fill="muted")
    fig.text(131, 214, "reward model", size=11, fill="muted")
    fig.text(24, 326, "Both training methods hold the new model near the one that wrote the stories. That constraint is what keeps them honest.", size=11, fill="muted")
    fig.save("rlhf-pipeline")


def scored_bars(fig, left, top, width, rows, peak, fmt, color="s1"):
    """Horizontal bars with a 95% interval, one per (name, mean, error) row."""
    fig.line(left, top - 8, left, top + 36 * len(rows) - 12, stroke="axis")
    scale = lambda v: width * max(v, 0) / peak
    for j, (name, mean, error) in enumerate(rows):
        y = top + j * 36
        fig.text(left - 10, y + 14, name, anchor="end", fill="ink")
        bar(fig, left, y, scale(mean), 18, color)
        end = scale(mean)
        if error:
            fig.line(left + scale(mean - error), y + 9, left + scale(mean + error), y + 9, stroke="ink", width=1.5)
            end = scale(mean + error)
        fig.text(left + end + 8, y + 14, fmt(mean), fill="ink")
        if error:
            fig.text(left + end + 14 + text_width(fmt(mean), 12), y + 14, f"± {fmt(error).lstrip('+')}", fill="muted", size=11)


def spread(values):
    """Mean and the half-width of its 95% interval."""
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(variance / len(values))


def rlhf_reward():
    meta = load("runs/rlhf/reward.json")
    tastes, rewards = meta["held_out_tastes"], meta["held_out_rewards"]
    fig = Figure(720, 360, "The reward model against the taste it never saw", f"{len(tastes):,} held-out stories. Trained on {meta['pairs']:,} comparisons, it orders {meta['agrees_with_taste']:.0%} of held-out pairs as the annotator's taste does.")
    lo, hi = math.floor(min(tastes)), math.ceil(max(tastes))
    rlo, rhi = math.floor(min(rewards)), math.ceil(max(rewards))
    ax = Axes(fig, 64, 84, 330, 200, (lo, hi), (rlo, rhi))
    ax.title("Reward given, by true taste")
    ax.yticks(range(rlo, rhi + 1, max(1, (rhi - rlo) // 5)), lambda t: f"{t:+d}" if t else "0")
    ax.xticks(range(lo, hi + 1), lambda t: f"{t:+d}" if t else "0", "the annotator's taste, hidden from training")
    for t, r in zip(tastes, rewards):
        fig.point(ax.x(t), ax.y(r), "s1")

    curve = sorted((load(f"runs/rlhf/{name}") for name in os.listdir("runs/rlhf") if re.fullmatch(r"reward(-\d+)?\.json", name)), key=lambda m: m["pairs"])  # the runs by number of comparisons
    xs, ys = [m["pairs"] for m in curve], [m["agrees_with_taste"] for m in curve]
    right = Axes(fig, 470, 84, 220, 200, (min(xs) * 0.7, max(xs) * 1.4), (0.5, 1.0), xlog=True)
    right.title("Agreement with the taste")
    right.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda t: f"{t:.0%}")
    right.xticks(xs, lambda t: f"{t:,}", "comparisons trained on")
    right.line(xs, ys, "s1")
    right.dots(xs, ys, "s1")
    right.label(xs[-1], ys[-1], f"{ys[-1]:.0%}", dx=0, dy=-12, anchor="middle", fill="ink")
    right.label(xs[0], ys[0], f"{ys[0]:.0%}", dx=0, dy=-12, anchor="middle")
    fig.text(470, 340, "50% would be guessing.", size=11, fill="muted")
    fig.save("rlhf-reward")


def rlhf_best_of_n():
    r = load("runs/rlhf/best_of_n.json")
    xs = r["n"]
    fig = Figure(720, 300, "Rejection sampling: keep the best of n stories", f"Mean taste of the story that is kept, over {r['prompts']} held-out prompts. Taste is the annotator's hidden standard.")
    fig.legend(64, 78, [("s1", "the reward model picks", "line"), ("s2", "the annotator would pick", "line")])
    top = max(r["picked_by_taste"])
    ax = Axes(fig, 64, 104, 590, 140, (0.85, max(xs) * 1.18), (math.floor(r["any"] * 2) / 2, math.ceil(top * 2) / 2), xlog=True)
    ticks = [ax.ylim[0] + 0.5 * k for k in range(int((ax.ylim[1] - ax.ylim[0]) * 2) + 1)]
    ax.yticks(ticks, lambda t: f"{t:+.1f}" if t else "0")
    ax.xticks(xs, str, "stories written per prompt")
    for key, color in [("picked_by_taste", "s2"), ("picked_by_reward_model", "s1")]:
        ax.line(xs, r[key], color)
        ax.dots(xs, r[key], color)
        ax.label(xs[-1], r[key][-1], f"{r[key][-1]:+.2f}", dx=10, dy=-2 if color == "s2" else 10, fill="ink")
    ax.label(xs[0], r["any"], f"{r['any']:+.2f}, any one story", dx=10, dy=16)
    fig.save("rlhf-best-of-n")


def rlhf_ppo():
    runs = [("with the KL penalty", "s1", jsonl("runs/rlhf/ppo.jsonl")), ("without it", "s2", jsonl("runs/rlhf/ppo-no-kl.jsonl"))]
    fig = Figure(720, 470, "PPO against the reward model, with and without a leash", "Each panel has its own scale. The reward model's reward is what training sees. The taste is what the annotator wants.")
    fig.legend(64, 78, [(color, name, "line") for name, color, _ in runs])
    panels = [("reward", "Reward from the reward model"), ("taste", "The annotator's taste"), ("kl", "KL divergence from the starting model"), ("tokens", "Tokens per story")]
    steps = max(r["iteration"] for r in runs[0][2])
    for i, (key, title) in enumerate(panels):
        col, row = i % 2, i // 2
        values = [r[key] for _, _, log in runs for r in log]
        ticks = nice_ticks(min(values), max(values))
        step = ticks[1] - ticks[0]
        lo, hi = min(ticks[0], math.floor(min(values) / step) * step), max(ticks[-1], math.ceil(max(values) / step) * step)
        ax = Axes(fig, 64 + col * 340, 124 + row * 170, 280, 100, (0, steps), (lo, hi))
        ax.title(title)
        ax.yticks(nice_ticks(lo, hi), lambda t: f"{t:g}")
        ax.xticks(range(0, steps + 1, 20), str, "iteration" if row else "")
        for name, color, log in runs:
            ax.line([r["iteration"] for r in log], [r[key] for r in log], color)
    fig.save("rlhf-ppo")


def story_measures(name, title, subtitle, source, rows, panels, footnote):
    """Small bar charts side by side: one panel per measure, one bar per model."""
    stories = load(source)
    fig = Figure(720, 256 + 34 * (len(rows) - 3), title, subtitle)
    for i, (heading, key, fmt, note) in enumerate(panels):
        left, top, width = 196 + i * 178, 108, 96
        fig.text(left, top - 20, heading, fill="ink", weight=600)
        fig.line(left, top - 8, left, top + 34 * len(rows) - 12, stroke="axis")
        peak = max(abs(stories[model][key]) for _, model in rows)
        for j, (label, model) in enumerate(rows):
            y = top + j * 34
            if i == 0:
                fig.text(left - 10, y + 14, label, anchor="end", fill="ink")
            value = stories[model][key]
            bar(fig, left, y, width * max(value, 0) / peak, 18, "s1")
            fig.text(left + max(width * max(value, 0) / peak, 5) + 8, y + 14, fmt(value), fill="ink")
        fig.text(left, top + 34 * len(rows) + 8, note, size=11, fill="muted")
    fig.text(24, 238 + 34 * (len(rows) - 3), footnote, size=11, fill="muted")
    fig.save(name)


STRANGENESS = "Strangeness is the loss of the pretrained story model on the stories, in nats per token: how unlike a story they read."


def rlhf_comparison():
    story_measures(
        "rlhf-comparison",
        "What each method did to the stories",
        "1,000 sampled stories per model on held-out prompts. Training saw comparisons only, and none of these three measures.",
        "runs/rlhf/stories.json",
        [("fine-tuned, the start", "sft-words"), ("PPO", "ppo"), ("PPO without KL", "ppo-no-kl"), ("DPO", "dpo")],
        [
            ("The annotator's taste", "taste", lambda v: f"{v:+.2f}", "higher is better"),
            ("Requested words used", "words_used", lambda v: f"{v:.0%}", "higher is better"),
            ("Strangeness", "strangeness", lambda v: f"{v:.2f}", "lower is better"),
        ],
        STRANGENESS,
    )


# --------------------------------------------------------------------- 9 rlvr


def rlvr_loop():
    fig = Figure(720, 336, "One step of GRPO", "The model answers the same prompt several times, and its answers are compared with each other.")
    fig.box(24, 142, 120, 56, "Prompt", "a problem")
    fig.arrow([(144, 170), (176, 170)])
    # a group of replies, each with its reward
    rewards = [1, 0, 0, 1, 0, 0, 0, 1]
    fig.text(178, 76, "8 replies, sampled", size=11, fill="muted")
    for i, r in enumerate(rewards):
        y = 84 + i * 22
        fig.rect(178, y, 150, 18, fill="s1" if r else "surface", stroke=None if r else "axis", rx=4, opacity=0.9 if r else None)
        fig.text(188, y + 13, f"reply {i + 1}", size=11, fill="surface" if r else "ink2")
        fig.text(318, y + 13, "right" if r else "wrong", size=11, fill="surface" if r else "muted", anchor="end")
    fig.arrow([(328, 170), (366, 170)])
    fig.box(368, 142, 130, 56, "Verifier", "a program: 1 or 0")
    fig.arrow([(498, 170), (536, 170)])
    fig.rect(538, 112, 158, 116, fill="surface", stroke="axis", rx=6)
    fig.text(617, 134, "Advantage", fill="ink", anchor="middle", weight=600)
    for i, line in enumerate(["reward minus the group's", "mean, over its spread", "", "right:  +1.2", "wrong:  −0.7"]):
        fig.text(550, 156 + i * 15, line, size=11, fill="ink2" if i < 3 else "ink")
    fig.arrow([(617, 228), (617, 280), (84, 280), (84, 200)])
    fig.text(350, 298, "one policy-gradient step: the right replies become likelier", size=11, fill="muted", anchor="middle")
    fig.text(24, 320, "A group in which every reply earned the same reward has no advantages, and teaches nothing.", size=11, fill="muted")
    fig.save("rlvr-loop")


def rlvr_curves(name, title, subtitle, runs, unit, panels):
    """Held-out score over training for some GRPO runs, with further quantities from their logs below it."""
    fig = Figure(720, 300 + 126 * len(panels), title, subtitle)
    fig.legend(64, 78, [(color, label, "line") for label, color, _ in runs])
    logs = [(label, color, jsonl(path)) for label, color, path in runs]
    steps = max(r["iteration"] for r in logs[0][2])
    held = [r["held_out"] for _, _, log in logs for r in log if "held_out" in r]
    lo, hi = math.floor(min(held) * 10) / 10, math.ceil(max(held) * 10) / 10
    ax = Axes(fig, 64, 124, 590, 130, (0, steps), (lo, hi))
    ax.title(f"{unit}, held-out prompts, greedy")
    ax.yticks(nice_ticks(lo, hi, 4), lambda t: f"{t:.0%}")
    for label, color, log in logs:
        points = [(r["iteration"], r["held_out"]) for r in log if "held_out" in r]
        ax.line(*zip(*points), color)
        ax.dots(*zip(*points), color)
        ax.label(points[-1][0], points[-1][1], f"{points[-1][1]:.0%}", dx=10, dy=4, fill="ink")
    if len({log[0]["held_out"] for _, _, log in logs}) == 1:  # runs from one model share a start; others would collide there
        ax.label(0, logs[0][2][0]["held_out"], f"{logs[0][2][0]['held_out']:.0%}", dx=8, dy=16)
    for i, (key, heading, limits, fmt) in enumerate(panels):
        low = Axes(fig, 64, 310 + i * 126, 590, 60, (0, steps), limits)
        low.title(heading)
        low.yticks(nice_ticks(*limits, 2), fmt)
        last = i == len(panels) - 1
        low.xticks(range(0, steps + 1, 25 if steps > 100 else 10), str, "iteration" if last else "")
        for label, color, log in logs:
            rows = [r for r in log if key in r]
            if not rows:  # a quantity this run does not log
                continue
            xs = [r["iteration"] for r in rows]
            # a running mean over 5 iterations: single iterations are noisy
            ys = [sum(q[key] for q in rows[max(0, j - 4) : j + 1]) / len(rows[max(0, j - 4) : j + 1]) for j in range(len(rows))]
            low.line(xs, ys, color)
    fig.save(name)


def rlvr_words():
    runs = [
        ("GRPO", "s2", "runs/rlvr/grpo-words.jsonl"),
        ("reward only if the story ends", "s3", "runs/rlvr/grpo-words-finish.jsonl"),
        ("LoRA with a KL penalty", "s1", "runs/rlvr/grpo-words-kl.jsonl"),
    ]
    panels = [
        ("informative_groups", "Groups in which the replies differ in reward", (0, 1), lambda t: f"{t:.0%}"),
        ("tokens", "Tokens per story, of 220 allowed", (150, 220), lambda t: f"{t:g}"),
    ]
    rlvr_curves("rlvr-words", "The story model learns to use its words", "150 iterations of 32 prompts with 8 stories each. The verifier counts the requested words.", runs, "Requested words used", panels)
    story_measures(
        "rlvr-stories",
        "What the verifier got, and what it did not ask for",
        "1,000 sampled stories per model on held-out prompts.",
        "runs/rlvr/stories.json",
        [("fine-tuned, the start", "sft-words"), ("GRPO", "grpo-words"), ("reward only if it ends", "grpo-words-finish"), ("LoRA with a KL penalty", "grpo-words-kl")],
        [
            ("Requested words used", "words_used", lambda v: f"{v:.0%}", "what was rewarded"),
            ("Stories cut off", "cut_off", lambda v: f"{v:.0%}", "they never end"),
            ("Strangeness", "strangeness", lambda v: f"{v:.2f}", "lower is better"),
        ],
        STRANGENESS,
    )


def rlvr_gsm8k():
    runs = [("GRPO from the fine-tuned model", "s1", "runs/rlvr/grpo-gsm8k.jsonl")]
    panels = [
        ("reward", "Sampled replies that are right, during training", (0, 0.8), lambda t: f"{t:.0%}"),
        ("informative_groups", "Groups in which the replies differ in reward", (0, 1), lambda t: f"{t:.0%}"),
    ]
    rlvr_curves("rlvr-gsm8k", "The 0.5B model on GSM8K", "60 iterations of 8 problems with 8 replies each. The verifier compares the final number.", runs, "Problems solved", panels)

    fig = Figure(720, 232, "GSM8K test problems solved", "1,319 problems, greedy replies, with 95% confidence intervals. GRPO saw 480 training problems and no solutions.")
    rows = []
    for label, name in [("0.5B as published", "gsm8k-qwen"), ("fine-tuned on solutions", "gsm8k-sft-gsm8k-lora"), ("then 22 minutes of GRPO", "gsm8k-grpo-gsm8k")]:
        r = load(f"runs/eval/{name}.json")
        rows.append((label, r["mean"], r["error"]))
    scored_bars(fig, 220, 84, 330, rows, 0.6, lambda v: f"{v:.1%}")
    fig.save("rlvr-gsm8k-test")


def tool_rl_curves():
    runs = [("GRPO with the calculator, from 6a's model", "s2", "runs/rlvr/grpo-gsm8k-tools.jsonl"), ("stage 9: GRPO without it, from stage 6's", "s1", "runs/rlvr/grpo-gsm8k.jsonl")]
    panels = [
        ("reward", "Sampled replies that are right, during training", (0, 0.8), lambda t: f"{t:.0%}"),
        ("calls", "Calls per reply, with the calculator", (0, 6), lambda t: f"{t:g}"),
    ]
    rlvr_curves("tool-rl-curves", "GRPO with a calculator that runs during the rollouts", "60 iterations of 8 problems with 8 replies each, as in stage 9. The verifier compares the final number.", runs, "Problems solved", panels)


def tool_rl_test():
    """GSM8K's test set before and after GRPO, with the tool and without it."""
    summary = load("runs/eval/tools-summary.json")["runs"]
    rows = [
        ("fine-tuned on solutions (stage 6)", "gsm8k-sft-gsm8k-lora", None),
        ("then GRPO without a tool (stage 9)", "gsm8k-grpo-gsm8k", None),
        ("fine-tuned with calls (6a)", "gsm8k-tools-sft-gsm8k-tools-lora", "gsm8k-tools-sft-gsm8k-tools-lora"),
        ("then GRPO with the calculator", "gsm8k-tools-grpo-gsm8k-tools", "gsm8k-tools-grpo-gsm8k-tools"),
        ("the same, every call fails", "gsm8k-tools-grpo-gsm8k-tools-no-execute", "gsm8k-tools-grpo-gsm8k-tools-no-execute"),
    ]
    fig = Figure(720, 290, "GSM8K test problems solved, before and after GRPO", "1,319 problems, greedy replies, with 95% intervals, and calls per problem where the calculator was offered.")
    left, top, width, peak = 250, 84, 300, 0.6
    scored_bars(fig, left, top, width, [(label, load(f"runs/eval/{name}.json")["mean"], load(f"runs/eval/{name}.json")["error"]) for label, name, _ in rows], peak, lambda v: f"{v:.1%}")
    fig.text(650, top - 10, "calls", fill="ink", weight=600, anchor="middle")
    for j, (_, _, run) in enumerate(rows):
        if run:
            fig.text(650, top + j * 36 + 14, f"{summary[run]['calls_per_problem']:.2f}", fill="ink2", anchor="middle")
    fig.text(24, 274, "Both GRPO runs: 60 iterations of 8 training problems with 8 replies each, starting from the model above them.", size=11, fill="muted")
    fig.save("tool-rl-test")


def reasoning_curves():
    runs = [("from the published model, 512 tokens", "s2", "runs/rlvr/grpo-qwen.jsonl"), ("stage 9: from the fine-tuned model, 256", "s1", "runs/rlvr/grpo-gsm8k.jsonl")]
    panels = [
        ("tokens", "Tokens per sampled reply", (0, 512), lambda t: f"{t:g}"),
        ("finished", "Replies that end before the limit", (0, 1), lambda t: f"{t:.0%}"),
    ]
    rlvr_curves("reasoning-curves", "GRPO from the published model, with room to think", "Stage 9's exercise 7: LoRA, a KL penalty of 0.05 towards the published model, replies of up to 512 tokens.", runs, "Problems solved", panels)


# The models of 9c, in the order the note takes them: (file in runs/reasoning/, label, color).
REASONERS = [("test-time-qwen", "0.5B as published", "ink2"), ("test-time-grpo-qwen", "after GRPO with room to think", "s2"), ("test-time-sft-gsm8k-traces", "fine-tuned on a reasoning model's thinking", "s3")]


def reasoning_test_time():
    """Votes over samples, and thinking cut at budgets, for every model of 9c that has been measured."""
    models = [(load(f"runs/reasoning/{file}.json"), label, color) for file, label, color in REASONERS if os.path.exists(f"runs/reasoning/{file}.json")]
    n = models[0][0]["n"]
    fig = Figure(720, 420, "More computation when asked", f"GSM8K's first {n} test problems. Left: replies sampled at temperature 1. Right: the greedy reply cut at a budget.")
    peak = max(max(m["pass"]) for m, _, _ in models)
    top = math.ceil(peak * 10) / 10
    votes = Axes(fig, 64, 108, 270, 200, (1, 16), (0, top), xlog=True)
    votes.title("Several samples per problem")
    votes.yticks(nice_ticks(0, top, 4), lambda t: f"{t:.0%}")
    votes.xticks([1, 2, 4, 8, 16], str, "samples per problem, k")
    for m, label, color in models:
        votes.line(m["ks"], m["majority"], color)
        votes.dots(m["ks"], m["majority"], color, r=3)
        votes.line(m["ks"], m["pass"], color, dash="5 4")
    longest = max(m["max_new_tokens"] for m, _, _ in models)
    budget = Axes(fig, 418, 108, 270, 200, (16, longest), (0, top), xlog=True)
    budget.title("Thinking cut at a budget")
    budget.yticks(nice_ticks(0, top, 4), lambda t: f"{t:.0%}")
    budget.xticks([b for b in (16, 64, 256, 1024) if b <= longest], lambda t: f"{t:,}", "tokens of thinking allowed")
    for m, label, color in models:
        points = dict((int(b), v["right"]) for b, v in m["budgets"].items())
        points[m["max_new_tokens"]] = m["greedy"]["right"]  # the reply uncut, at its limit
        points = sorted(points.items())
        budget.line(*zip(*points), color)
        budget.dots(*zip(*points), color, r=3)
    x = 64
    for m, label, color in models:
        fig.line(x, 370, x + 18, 370, stroke=color, width=2)
        fig.text(x + 24, 374, label, size=11)
        x += 24 + text_width(label, 11) + 24
    fig.text(24, 404, "Solid: the most common answer of k samples is right. Dashed: at least one of k is right. Right: the last dot is the reply uncut.", size=11, fill="muted")
    fig.save("reasoning-test-time")


def reasoning_replies():
    """What became of every reply: right, wrong after it ended, or cut off at the limit, in a loop or not."""
    summary = load("runs/reasoning/summary.json")
    first = summary["first"]
    rows = [(summary["runs"][name]["label"], f"{summary['runs'][name]['n']:,} problems, up to {summary['runs'][name]['max_new_tokens']:,} tokens, greedy", summary["runs"][name]["replies"]) for name in ("gsm8k-qwen", "gsm8k-grpo-qwen", "gsm8k-sft-gsm8k-traces")]
    rows.append(("the teacher, 1.5B", f"the first {first['n']} problems, up to {first['teacher']['max_new_tokens']:,} tokens, sampled", first["teacher"]["replies"]))
    kinds = [("right", "right", "s1"), ("wrong_ended", "wrong, ended", "grid"), ("wrong_cut_off_otherwise", "wrong, cut off", "axis"), ("wrong_cut_off_looping", "wrong, cut off in a loop", "ink2")]
    fig = Figure(720, 330, "How the replies ended", "GSM8K's test problems. A reply cut off at the limit is in a loop when its last 200 characters were written before.")
    fig.legend(64, 78, [(color, label, "bar") for _, label, color in kinds])
    left, width, top = 300, 380, 112
    for j, (label, note, counts) in enumerate(rows):
        y = top + j * 48
        fig.text(left - 12, y + 12, label, anchor="end", fill="ink")
        fig.text(left - 12, y + 28, note, anchor="end", fill="muted", size=10)
        total, x = sum(counts.values()), left
        for key, _, color in kinds:
            w = width * counts[key] / total
            if w <= 0:
                continue
            fig.rect(x, y, max(w - 2, 1), 30, fill=color)  # a 2px gap of surface between segments
            if w >= 36:
                fig.text(x + w / 2 - 1, y + 20, f"{counts[key] / total:.0%}", anchor="middle", fill="surface" if color in ("s1", "ink2") else "ink", size=11)
            x += w
    fig.text(24, 314, "The teacher's replies are one sample each at temperature 0.6, as its authors advise; the others are greedy.", size=11, fill="muted")
    fig.save("reasoning-replies")


# ------------------------------------------------------- 2a mixture of experts

# One color per model in all the figures of this note.
DENSE, MOE, WIDE = "s1", "s2", "s3"


def moe_block():
    fig = Figure(720, 372, "A mixture of experts in place of the MLP", "One token in one block. The router scores eight experts, and only the two with the highest scores run.")
    fig.box(24, 165, 96, 56, "Token", "its vector x")
    fig.arrow([(120, 193), (150, 193)])
    fig.box(152, 165, 128, 56, "Router", "softmax(R x)")
    weights = {2: 0.7, 5: 0.3}  # the experts this token is routed to, and their weights
    fig.text(330, 76, "8 experts, each an MLP of its own", size=11, fill="muted")
    for i in range(8):
        y, on = 84 + i * 28, i in weights
        fig.rect(330, y, 170, 22, fill=MOE if on else "surface", stroke=None if on else "axis", rx=4, opacity=0.9 if on else None)
        fig.text(340, y + 15, f"expert {i + 1}", size=11, fill="surface" if on else "ink2")
        fig.text(490, y + 15, f"weight {weights[i]}" if on else "not run", size=11, fill="surface" if on else "muted", anchor="end")
        if on:
            fig.arrow([(280, 193), (304, 193), (304, y + 11), (328, y + 11)])
            fig.arrow([(500, y + 11), (528, y + 11), (528, 193), (552, 193)])
    fig.box(554, 165, 142, 56, "Weighted sum", "0.7 · e3 + 0.3 · e6")
    fig.arrow([(625, 221), (625, 252)])
    fig.text(625, 268, "added to the", size=11, anchor="middle")
    fig.text(625, 283, "residual stream", size=11, anchor="middle")
    fig.text(24, 334, "The block holds the parameters of all eight experts, and a token costs the compute of two.", size=11, fill="muted")
    fig.text(24, 350, "Here an expert has half the width of the dense MLP: the same compute per token, and four times the parameters.", size=11, fill="muted")
    fig.save("moe-block")


def moe_balance():
    runs = [("With the balance loss", "runs/moe/moe.jsonl"), ("Without it", "runs/moe/no-balance.jsonl")]
    logs = [[r for r in jsonl(path) if "expert_share" in r][1::2] for _, path in runs]  # every 100 steps
    experts = len(logs[0][0]["expert_share"][0])
    peak = max(v for log in logs for r in log[3:] for block in r["expert_share"] for v in block)
    top = math.ceil(peak * 10) / 10
    fig = Figure(720, 390, "The balance loss keeps every expert in work", "Each line is one expert of one block, 64 per panel: its share of the block's routed work in a training batch.")
    for i, ((title, _), log) in enumerate(zip(runs, logs)):
        steps = [r["step"] for r in log]
        ax = Axes(fig, 64 + i * 340, 100, 280, 200, (0, steps[-1]), (0, top))
        ax.title(title)
        ax.yticks(nice_ticks(0, top, 4), lambda t: f"{t:.0%}")
        ax.xticks(range(0, steps[-1] + 1, 2000), lambda t: f"{t:,}", "training step")
        for block in range(len(log[0]["expert_share"])):
            for e in range(experts):
                ax.line(steps, [min(r["expert_share"][block][e], top) for r in log], MOE, width=1, opacity=0.45)
        fig.line(ax.left, ax.y(1 / experts), ax.left + ax.width, ax.y(1 / experts), stroke="ink", width=1)
        ax.label(steps[-1], 1 / experts, "even", dx=6, fill="ink")
        idle = sum(v < 0.01 for block in log[-1]["expert_share"] for v in block)
        dropped = sum(r["dropped"] for r in log[len(log) // 2 :]) / len(log[len(log) // 2 :])
        fig.text(ax.left, 352, f"{idle} of {len(log[0]['expert_share']) * experts} experts end below 1% of their block's work.", size=11, fill="muted")
        fig.text(ax.left, 368, f"{dropped:.1%} of the routed work found no room.", size=11, fill="muted")
    fig.save("moe-balance")


def moe_loss():
    # (name, color, the run of 6,000 steps, the run planned to take 20 minutes)
    runs = [
        ("dense, 14M", DENSE, "runs/base/log.jsonl", "runs/base/log.jsonl"),
        ("mixture, 42M of which 14M active", MOE, "runs/moe/moe.jsonl", "runs/moe/moe-20min.jsonl"),
        ("dense, 42M", WIDE, "runs/moe/wide.jsonl", "runs/moe/wide-20min.jsonl"),
    ]
    evals = lambda path: [r for r in jsonl(path) if "val_loss" in r]
    fig = Figure(720, 344, "The mixture against two dense models", "Validation loss. Left: all three take the same 6,000 steps. Right: each is given the same 20 minutes.")
    fig.legend(64, 78, [(color, name, "line") for name, color, _, _ in runs])

    logs = [[r for r in evals(path) if r["step"] >= 2000] for _, _, path, _ in runs]
    losses = [r["val_loss"] for log in logs for r in log]
    lo, hi = math.floor(min(losses) * 20) / 20, math.ceil(max(losses) * 20) / 20
    ax = Axes(fig, 64, 124, 280, 150, (2000, 6000), (lo, hi))
    ax.title("The same steps, from step 2,000 on")
    ax.yticks(nice_ticks(lo, hi, 4), lambda t: f"{t:g}")
    ax.xticks(range(2000, 6001, 1000), lambda t: f"{t:,}", "training step")
    for (_, color, _, _), log in zip(runs, logs):
        xs, ys = [r["step"] for r in log], [r["val_loss"] for r in log]
        ax.line(xs, ys, color)
        ax.dots(xs[-1:], ys[-1:], color)
        ax.label(xs[-1], ys[-1], f"{ys[-1]:.3f}", dx=10, fill="ink")

    # The runs sized for 20 minutes: one row each, the final loss as a dot on a shared scale.
    finals = [evals(path)[-1] for _, _, _, path in runs]
    lo = math.floor(min(r["val_loss"] for r in finals) * 20) / 20
    hi = math.ceil(max(r["val_loss"] for r in finals) * 20) / 20
    time = Axes(fig, 452, 124, 220, 150, (lo, hi), (0, 1))
    time.title("The same time")
    time.xticks(nice_ticks(lo, hi, 3), lambda t: f"{t:g}", "validation loss at the end")
    for i, ((_, color, _, _), r) in enumerate(zip(runs, finals)):
        y = time.top + 28 + i * 46
        fig.line(time.left, y, time.left + time.width, y)
        fig.dot(time.x(r["val_loss"]), y, color, r=5)
        fig.text(time.x(r["val_loss"]), y - 11, f"{r['val_loss']:.3f}", anchor="middle", fill="ink")
        note = f"{r['step']:,} steps in {r['elapsed'] / 60:.1f} min"
        half = text_width(note, 11) / 2
        fig.text(min(max(time.x(r["val_loss"]), time.left + half), time.left + time.width - half), y + 20, note, size=11, anchor="middle")
    fig.save("moe-loss")


def moe_speed():
    bench = load("runs/bench/moe-mps.json")
    color = {"Dense, 14M": DENSE, "Dense, 42M": WIDE}
    label = {"Dense, 14M": "dense, 14M", "Dense, 42M": "dense, 42M"}
    label |= {"Experts take their own tokens": "experts take their own tokens", "Experts take every token": "experts take every token"}
    label |= {"Capacity 1.5": "capacity 1.5", "Capacity 1": "capacity 1"}
    sections = [
        ("A training step", [(r["name"], r["tokens_per_second"]) for r in bench["train"]]),
        ("Generating one sequence", [(r["name"], r["batch_1"]) for r in bench["generate"]]),
        ("Generating 64 sequences at once", [(r["name"], r["batch_64"]) for r in bench["generate"]]),
    ]
    row, left, width = 26, 250, 340
    fig = Figure(720, 96 + sum(48 + row * len(rows) for _, rows in sections), "What the mixture costs in time", "Tokens per second on an M5 Pro. Each section has its own scale.")
    fig.legend(24, 76, [(DENSE, "dense, 14M", "dot"), (MOE, "mixture, 42M of which 14M active", "dot"), (WIDE, "dense, 42M", "dot")])
    top = 118
    for title, rows in sections:
        fig.text(24, top - 12, title, fill="ink", weight=600)
        fig.line(left, top - 4, left, top + row * len(rows) - 6, stroke="axis")
        peak = max(v for _, v in rows)
        for i, (name, value) in enumerate(rows):
            y = top + i * row
            fig.text(left - 10, y + 13, label[name], anchor="end", fill="ink")
            bar(fig, left, y, width * value / peak, 16, color.get(name, MOE))
            text = f"{value / 1000:.0f}K" if value >= 10000 else f"{value / 1000:.1f}K" if value >= 1000 else f"{value:.0f}"
            fig.text(left + max(width * value / peak, 5) + 8, y + 13, text, fill="ink")
        top += 48 + row * len(rows)
    fig.save("moe-speed")


def moe_routing():
    data = load("runs/moe/routing.json")
    blocks, experts = data["blocks"], data["n_experts"]
    fig = Figure(720, 372, "What the routers do", f"The trained mixture on {data['tokens']:,} tokens of held-out stories.")
    # each expert's share of the work, block by block
    left, top, cell = 64, 124, 26
    fig.text(left, top - 26, "Share of the work", fill="ink", weight=600)
    fig.text(left, top - 10, "one row per block, one cell per expert", size=11)
    peak = max(v for b in blocks for v in b["share"])
    for i, b in enumerate(blocks):
        fig.text(left - 8, top + i * cell + 17, str(i + 1), size=11, fill="muted", anchor="end")
        for e, share in enumerate(b["share"]):
            fig.rect(left + e * cell, top + i * cell, cell - 2, cell - 2, fill="grid", opacity=0.6, rx=3)
            fig.rect(left + e * cell, top + i * cell, cell - 2, cell - 2, fill=MOE, opacity=round(share / peak, 2), rx=3)
    lo, hi = min(v for b in blocks for v in b["share"]), peak
    fig.text(left, top + len(blocks) * cell + 16, f"from {lo:.0%} to {hi:.0%}; even would be {1 / experts:.1%}", size=11, fill="muted")
    fig.text(left - 34, top + len(blocks) * cell / 2, "block", size=11, fill="muted", anchor="middle", rotate=-90)

    xs = list(range(1, len(blocks) + 1))
    ax = Axes(fig, 372, 124, 290, 150, (0.6, len(blocks) + 0.4), (0, 1))
    ax.title("How often a token's first expert can be guessed")
    ax.yticks([0, 0.25, 0.5, 0.75, 1], lambda t: f"{t:.0%}")
    ax.xticks(xs, str, "block")
    ax.line(xs, [b["by_token_baseline"] for b in blocks], "muted", width=1)
    for key, color in [("by_token", "s1"), ("by_previous", "s3")]:
        ax.line(xs, [b[key] for b in blocks], color)
        ax.dots(xs, [b[key] for b in blocks], color)
    ax.label(1, blocks[0]["by_token"], "from the token alone", dx=12, dy=-8, fill="ink")
    peak = max(xs, key=lambda x: blocks[x - 1]["by_previous"])
    ax.label(peak, blocks[peak - 1]["by_previous"], "it is the one of the token before", dx=-12, dy=4, anchor="end", fill="ink")
    fig.text(372, 326, "Gray: always guessing the block's most common first expert.", size=11, fill="muted")
    fig.save("moe-routing")


# ---------------------------------------------------------- 2c latent attention

ATTENTION = [  # (log, label, the model in the benchmark, is it latent attention, where its label goes: dx, dy, anchor)
    ("runs/attention/mha.jsonl", "6 key/value heads", "out/mha", False, (-10, 24, "end")),
    ("runs/base/log.jsonl", "2, stage 2's", "out/base", False, (8, -10, "start")),
    ("runs/attention/mqa.jsonl", "1 key/value head", "out/mqa", False, (-10, -10, "end")),
    ("runs/attention/mla-128.jsonl", "latent 128 + 32", "out/mla-128", True, (-20, 22, "middle")),
    ("runs/attention/mla-64.jsonl", "latent 64 + 32", "out/mla-64", True, (8, -10, "start")),
    ("runs/attention/mla-32.jsonl", "latent 32 + 16", "out/mla-32", True, (8, -10, "start")),
]


def latent_block():
    fig = Figure(720, 336, "Latent attention: what is cached", "One token in one layer, with a latent of 128. Only the shaded vectors are kept for later tokens.")
    fig.box(24, 150, 96, 56, "Token", "its vector x")
    fig.arrow([(120, 178), (146, 178)])
    fig.box(148, 150, 112, 56, "Down", "x to 128 + 32")
    fig.rect(282, 84, 154, 190, stroke="s2", rx=8)
    fig.text(359, 100, "the cache, per token", size=11, anchor="middle", fill="s2")
    fig.box(296, 112, 126, 56, "Latent c", "128, normalized", fill="s2", opacity=0.15, stroke="s2")
    fig.box(296, 200, 126, 56, "Rotary key", "32, rotated", fill="s2", opacity=0.15, stroke="s2")
    fig.arrow([(260, 170), (276, 170), (276, 140), (294, 140)])
    fig.arrow([(260, 186), (276, 186), (276, 228), (294, 228)])
    fig.arrow([(422, 140), (456, 140)])
    fig.box(458, 112, 112, 56, "Up, per head", "W_k and W_v")
    fig.arrow([(570, 140), (596, 140)])
    fig.box(598, 112, 98, 56, "6 keys", "6 values, of 64")
    fig.arrow([(422, 228), (596, 228)])
    fig.box(598, 200, 98, 56, "1 rotary key", "for all 6 heads")
    fig.text(24, 298, "Per token and layer the cache holds 160 numbers. Stage 2's grouped-query attention holds 2 keys and 2 values of 64: 256.", size=11, fill="muted")
    fig.text(24, 314, "Decoding never projects the cache up: the query is projected down into the latent instead, and W_v applied after the average.", size=11, fill="muted")
    fig.save("latent-block")


def last_val_loss(path):
    return [r["val_loss"] for r in jsonl(path) if "val_loss" in r][-1]


def attention_loss():
    bench = {r["model"]: r for r in load("runs/bench/attention-mps.json")["models"] if r["absorb"]}
    layers = 8
    numbers = lambda model: bench[model]["cache_bytes_per_token"] / 4 / layers  # 32-bit floats, per token and layer
    fig = Figure(720, 330, "Validation loss against the cache", "The 14M model of stage 3, trained for 6,000 steps with each kind of attention. The cache is counted per token and layer.")
    fig.legend(64, 78, [("s1", "key/value heads", "dot"), ("s2", "latent attention", "dot")])
    points = [(numbers(model), last_val_loss(path), label, latent, place) for path, label, model, latent, place in ATTENTION]
    seed = last_val_loss("runs/attention/base-seed1.jsonl")
    losses = [p[1] for p in points] + [seed]
    lo, hi = math.floor(min(losses) * 100 - 1) / 100, math.ceil(max(losses) * 100 + 1) / 100
    ax = Axes(fig, 72, 110, 560, 160, (30, 1000), (lo, hi), xlog=True)
    ax.yticks(nice_ticks(lo, hi, 4), lambda t: f"{t:.2f}")
    ax.xticks([32, 64, 128, 256, 512], str, "numbers cached per token and layer, log scale")
    base = numbers("out/base")
    fig.line(ax.x(base), ax.y(min(seed, points[1][1])), ax.x(base), ax.y(max(seed, points[1][1])), stroke="s1", width=2)
    ax.label(base, seed, f"another seed, {seed:.3f}", dx=10, dy=12, fill="muted")
    for x, y, label, latent, (dx, dy, anchor) in points:
        ax.dots([x], [y], "s2" if latent else "s1", r=5)
        ax.label(x, y, f"{label}, {y:.3f}", dx=dx, dy=dy, anchor=anchor, fill="ink")
    fig.save("attention-loss")


def attention_speed():
    bench = load("runs/bench/attention-mps.json")
    rows = [(r["name"], r) for r in bench["models"]]
    color = lambda r: "muted" if not r["absorb"] else "s2" if "mla" in r["model"] else "s1"
    sections = [("One sequence", "1"), ("1,024 sequences at once", "1024")]
    row, left, width = 26, 290, 300
    fig = Figure(720, 96 + sum(48 + row * len(rows) for _ in sections), "What each kind of attention costs in generation", f"Tokens per second, {bench['new_tokens']} new tokens each, with stage 4's sampler on an M5 Pro. Each section has its own scale.")
    fig.legend(24, 76, [("s1", "key/value heads", "dot"), ("s2", "latent, decoded against the latent", "dot"), ("muted", "latent, projected up at every step", "dot")])
    top = 118
    for title, batch in sections:
        fig.text(24, top - 12, title, fill="ink", weight=600)
        fig.line(left, top - 4, left, top + row * len(rows) - 6, stroke="axis")
        peak = max(r["tokens_per_second"][batch] for _, r in rows)
        for i, (name, r) in enumerate(rows):
            y, value = top + i * row, r["tokens_per_second"][batch]
            fig.text(left - 10, y + 13, name.replace(", projected up at every step", ", projected up"), anchor="end", fill="ink")
            bar(fig, left, y, width * value / peak, 16, color(r))
            fig.text(left + max(width * value / peak, 5) + 8, y + 13, f"{value:.0f}" if value < 1000 else f"{value / 1000:.1f}K" if value < 10000 else f"{value / 1000:.0f}K", fill="ink")
        top += 48 + row * len(rows)
    fig.save("attention-speed")


# ------------------------------------------------------- 3b data preparation


def data_funnel():
    """What every step of the preparation leaves of the crawl, in characters."""
    f = load("runs/data/funnel.json")["train"]
    steps = f["steps"]
    whole = steps[0]["characters"]
    names = ["8 files of the crawl", "pages tagged English", "their lines of prose", "pages that pass the rules", "without exact copies", "without near copies", "without lines that pages share"]
    fig = Figure(720, 372, "What the rules leave of a crawl", f"{steps[0]['pages']:,} pages of Common Crawl, {whole / 1e9:.2f} billion characters of extracted text. The bars are characters.")
    left, top, width = 228, 84, 330
    fig.line(left, top - 8, left, top + 36 * len(steps) - 12, stroke="axis")
    for j, (name, step) in enumerate(zip(names, steps)):
        y = top + j * 36
        share = step["characters"] / whole
        fig.text(left - 10, y + 14, name, anchor="end", fill="ink")
        bar(fig, left, y, width * share, 18, "s2" if j == len(steps) - 1 else "s1")
        fig.text(left + max(width * share, 5) + 8, y + 14, f"{step['characters'] / 1e6:,.0f}M", fill="ink")
        fig.text(left + max(width * share, 5) + 22 + text_width(f"{step['characters'] / 1e6:,.0f}M", 12), y + 14, f"{share:.1%}" + (f", {step['pages']:,} pages" if j else ""), fill="muted", size=11)
    dropped = f["dropped_by_rule"]
    fig.text(24, 352, "Pages dropped by a rule: " + ", ".join(f"{reason} {count:,}" for reason, count in dropped.items()) + ".", size=11, fill="muted")
    fig.save("data-funnel")


def data_loss():
    """Three models, trained on three versions of the text, on held-out text of three kinds."""
    c = load("runs/data/cross.json")["loss"]
    models = [("raw pages", "web-raw", "axis"), ("text by the rules", "web-rules", "s1"), ("the same without copies", "web-clean", "s2")]
    panels = [("raw", "Held-out pages as crawled", "menus and all"), ("clean", "Held-out prose", "the same pages, cleaned"), ("stories", "Children's stories", "which no model saw")]
    fig = Figure(720, 300, "What a model learns from each version of the text", "The 14M model, 3,000 steps on each. Loss per token on held-out text: lower is better, and comparable only within a panel.")
    fig.text(64, 80, "Trained on", fill="ink")
    fig.legend(140, 80, [(color, name, "dot") for name, _, color in models])
    for i, (corpus, title, note) in enumerate(panels):
        left, top, width = 64 + i * 220, 132, 130
        fig.text(left, top - 22, title, fill="ink", weight=600)
        fig.text(left, top - 7, note, size=11, fill="muted")
        fig.line(left, top + 2, left, top + 3 * 36 - 2, stroke="axis")
        for j, (_, model, color) in enumerate(models):
            value = c[model][corpus]
            bar(fig, left, top + 10 + j * 34, width * value / 4.5, 18, color)
            fig.text(left + width * value / 4.5 + 8, top + 24 + j * 34, f"{value:.2f}", fill="ink")
    fig.text(24, 280, "The raw model is best on raw pages, whose menus it has learned. On prose and on stories the cleaned text teaches more.", size=11, fill="muted")
    fig.save("data-loss")


# ----------------------------------------------------------- 3a distillation


def distill_targets():
    data = load("runs/distill/targets.json")
    stats = data["held_out"]
    fig = Figure(720, 330, "What a teacher says that a label does not", "The 42M model's prediction for the next token at three places of one sentence. The text itself names a single token.")
    fig.legend(24, 76, [("s2", "the token that came next in the text", "dot"), ("s1", "other tokens the teacher expects", "dot")])
    for i, position in enumerate(data["positions"]):
        left, top = 96 + i * 228, 132
        rows = [(t["token"], t["probability"]) for t in position["top"]]
        if position["label"] not in [token for token, _ in rows]:
            rows = rows[:-1] + [(position["label"], position["label_probability"])]
        words = position["context"].split()
        fig.text(left - 72, top - 22, "… " + " ".join(words[-3:]) + " ___", fill="ink", weight=600)
        fig.line(left, top - 8, left, top + 24 * len(rows) - 6, stroke="axis")
        for j, (token, probability) in enumerate(rows):
            y = top + j * 24
            fig.text(left - 8, y + 12, token.strip(), anchor="end", fill="ink")
            bar(fig, left, y, 104 * probability / 0.5, 14, "s2" if token == position["label"] else "s1")
            fig.text(left + max(104 * probability / 0.5, 5) + 6, y + 12, f"{probability:.0%}" if probability >= 0.01 else f"{probability:.1%}", size=11)
    fig.text(24, 296, f"Over {stats['tokens']:,} held-out tokens the teacher gives more than 1% to {stats['above_one_percent']:.0f} tokens on average,", size=11, fill="muted")
    fig.text(24, 312, f"and the token that came next is its first choice {stats['label_is_first_choice']:.0%} of the time.", size=11, fill="muted")
    fig.save("distill-targets")


def distill_loss():
    # (name, color, the run of 6,000 steps, the run planned to take the 20 minutes of the first)
    runs = [
        ("from the text alone", "s1", "runs/base/log.jsonl", "runs/base/log.jsonl"),
        ("half text, half teacher", "s2", "runs/distill/distilled.jsonl", "runs/distill/distilled-20min.jsonl"),
        ("from the teacher alone", "s3", "runs/distill/teacher-only.jsonl", None),
    ]
    evals = lambda path: [r for r in jsonl(path) if "val_loss" in r]
    teacher = evals("runs/moe/wide.jsonl")[-1]["val_loss"]
    fig = Figure(720, 344, "A 14M student of a 42M teacher", "Validation loss on the text. Left: the same 6,000 steps. Right: the same 20 minutes, the teacher's work included.")
    fig.legend(64, 78, [(color, name, "line") for name, color, _, _ in runs])

    logs = [[r for r in evals(path) if r["step"] >= 2000] for _, _, path, _ in runs]
    losses = [r["val_loss"] for log in logs for r in log] + [teacher]
    lo, hi = math.floor(min(losses) * 20) / 20, math.ceil(max(losses) * 20) / 20
    ax = Axes(fig, 64, 124, 280, 150, (2000, 6000), (lo, hi))
    ax.title("The same steps, from step 2,000 on")
    ax.yticks(nice_ticks(lo, hi, 4), lambda t: f"{t:g}")
    ax.xticks(range(2000, 6001, 1000), lambda t: f"{t:,}", "training step")
    fig.line(ax.left, ax.y(teacher), ax.left + ax.width, ax.y(teacher), stroke="ink", width=1)
    ax.label(2000, teacher, f"the teacher, {teacher:.3f}", dx=4, dy=-6, fill="ink")
    finals = []
    for (_, color, _, _), log in zip(runs, logs):
        xs, ys = [r["step"] for r in log], [r["val_loss"] for r in log]
        ax.line(xs, ys, color)
        ax.dots(xs[-1:], ys[-1:], color)
        finals.append((ys[-1], color))
    # final values from the top down, nudged apart where two runs end close together
    y = None
    for value, color in sorted(finals, reverse=True):
        y = ax.y(value) if y is None else max(ax.y(value), y + 13)
        fig.text(ax.left + ax.width + 10, y + 4, f"{value:.3f}", fill="ink")

    timed = [(color, evals(path)[-1]) for _, color, _, path in runs if path]
    lo = math.floor(min(r["val_loss"] for _, r in timed) * 50) / 50
    hi = math.ceil(max(r["val_loss"] for _, r in timed) * 50) / 50
    time = Axes(fig, 452, 124, 220, 150, (lo, hi), (0, 1))
    time.title("The same time")
    time.xticks(nice_ticks(lo, hi, 3), lambda t: f"{t:g}", "validation loss at the end")
    for i, (color, r) in enumerate(timed):
        y = time.top + 40 + i * 56
        fig.line(time.left, y, time.left + time.width, y)
        fig.dot(time.x(r["val_loss"]), y, color, r=5)
        fig.text(time.x(r["val_loss"]), y - 11, f"{r['val_loss']:.3f}", anchor="middle", fill="ink")
        note = f"{r['step']:,} steps in {r['elapsed'] / 60:.1f} min"
        half = text_width(note, 11) / 2
        fig.text(min(max(time.x(r["val_loss"]), time.left + half), time.left + time.width - half), y + 20, note, size=11, anchor="middle")
    fig.save("distill-loss")


# ---------------------------------------------------------- 9a self-training


def self_rounds():
    # (title, the self-training run, the GRPO run of stage 9, replies per GRPO iteration by the defaults of scripts/grpo.py)
    panels = [
        ("The 14M model: requested words used", "words", "runs/rlvr/grpo-words.jsonl", 32 * 8),
        ("The 0.5B model: GSM8K problems solved", "gsm8k", "runs/rlvr/grpo-gsm8k.jsonl", 8 * 8),
    ]
    fig = Figure(720, 330, "Self-training against GRPO, for the same replies", "Score on held-out prompts, greedy, against the replies sampled so far. Both methods start from the same model.")
    fig.legend(64, 78, [("s1", "self-training, one point per round", "line"), ("s2", "GRPO", "line")])
    for i, (title, name, grpo_path, per_iteration) in enumerate(panels):
        args = load(f"runs/self/{name}-meta.json")["args"]
        ours = [(r["round"] * args["prompts"] * args["samples"], r["held_out"]) for r in jsonl(f"runs/self/{name}.jsonl")]
        grpo = [(r["iteration"] * per_iteration, r["held_out"]) for r in jsonl(grpo_path) if "held_out" in r]
        most = max(ours[-1][0], grpo[-1][0])
        values = [v for _, v in ours + grpo]
        lo, hi = math.floor(min(values) * 10) / 10, math.ceil(max(values) * 10) / 10
        ax = Axes(fig, 72 + i * 332, 124, 270, 140, (0, most), (lo, hi))
        ax.title(title)
        ax.yticks(nice_ticks(lo, hi, 4), lambda t: f"{t:.0%}")
        ax.xticks(nice_ticks(0, most, 4), thousands, "replies sampled")
        for points, color in ((grpo, "s2"), (ours, "s1")):
            ax.line(*zip(*points), color)
            ax.dots(*zip(*points), color, r=3.5)
        ax.label(ours[-1][0], ours[-1][1], f"{ours[-1][1]:.0%}", dx=-2, dy=18 if ours[-1][1] < grpo[-1][1] else -10, anchor="end", fill="ink")
        ax.label(grpo[-1][0], grpo[-1][1], f"{grpo[-1][1]:.0%}", dx=-2, dy=-10 if ours[-1][1] < grpo[-1][1] else 18, anchor="end", fill="ink")
    fig.save("self-rounds")


def self_results():
    story_measures(
        "self-stories",
        "What self-training does to the stories",
        "1,000 sampled stories per model on held-out prompts, with the models of stage 9 for comparison.",
        "runs/self/stories.json",
        [("fine-tuned, the start", "sft-words"), ("GRPO", "grpo-words"), ("GRPO, LoRA with a KL penalty", "grpo-words-kl"), ("self-training", "self-words")],
        [
            ("Requested words used", "words_used", lambda v: f"{v:.0%}", "what was asked for"),
            ("Stories cut off", "cut_off", lambda v: f"{v:.0%}", "they never end"),
            ("Strangeness", "strangeness", lambda v: f"{v:.2f}", "lower is better"),
        ],
        STRANGENESS,
    )
    fig = Figure(720, 250, "GSM8K test problems solved", "1,319 problems, greedy replies, with 95% confidence intervals. Self-training sampled twice the replies of GRPO.")
    rows = []
    for label, name in [("0.5B as published", "gsm8k-qwen"), ("fine-tuned on solutions", "gsm8k-sft-gsm8k-lora"), ("then GRPO", "gsm8k-grpo-gsm8k"), ("or self-training", "gsm8k-self-gsm8k")]:
        r = load(f"runs/eval/{name}.json")
        rows.append((label, r["mean"], r["error"]))
    scored_bars(fig, 220, 84, 330, rows, 0.6, lambda v: f"{v:.1%}")
    fig.save("self-gsm8k")


# ----------------------------------------------------------- 10a quantization


def quant_levels():
    data = load("runs/quant/row.json")
    w, size = data["weights"], data["group"]
    fig = Figure(720, 318, "Rounding weights to 16 values", f"One row of a query projection of the 0.5B model, {len(w)} weights. Each orange line is a value that 4 bits can name.")
    lo, hi = min(w), max(w)
    left, width, tall = 64, 592, 46
    x = lambda v: left + width * (v - lo) / (hi - lo)

    def strip(top, values, title):
        a, step = min(values), (max(values) - min(values)) / 15
        fig.text(left, top - 12, title, fill="ink", weight=600)
        fig.line(left, top + tall, left + width, top + tall, stroke="axis")
        for k in range(16):
            fig.line(x(a + k * step), top, x(a + k * step), top + tall, stroke="s2", width=1.5)
        return step

    # the whole row, as a histogram of where its weights lie
    step = strip(96, w, "The whole row shares one scale")
    bins = 148
    counts = [0] * bins
    for v in w:
        counts[min(bins - 1, int((v - lo) / (hi - lo) * bins))] += 1
    for i, c in enumerate(counts):
        if c:
            fig.rect(left + i * width / bins, 96 + tall - 40 * c / max(counts), width / bins - 1, 40 * c / max(counts), fill="s1", opacity=0.6)
    fig.text(left, 96 + tall + 16, f"16 values from the row's smallest weight to its largest, {step:.3f} apart. Most weights share the few in the middle.", size=11, fill="muted")

    # its first group of 64 weights, each drawn as a tick
    first = w[:size]
    step = strip(206, first, f"A group of {size} weights has a scale of its own")
    for v in first:
        fig.line(x(v), 206 + 14, x(v), 206 + tall, stroke="s1", width=1.5)
    fig.text(left, 206 + tall + 16, f"16 values over the range of this group alone, {step:.3f} apart. Rounding moves a weight by at most half a step.", size=11, fill="muted")
    fig.text(left, 300, f"The row's range is {data['ratio']:.1f} times that of its average group. The price of groups is a scale and an offset for each.", size=11, fill="muted")
    fig.save("quant-levels")


def quant_bits():
    panels = [("The 14M model", "runs/quant/sweep-base.json"), ("The 0.5B model", "runs/quant/sweep-qwen.json")]
    kinds = [(0, "one scale per row", "s1"), (64, "per group of 64", "s2"), (16, "per group of 16", "s3")]
    floor, circled = 0.001, []
    fig = Figure(720, 392, "What a bit width costs", "Held-out loss above the unrounded model's, against the bits stored per weight: its code and its share of the scales.")
    fig.legend(64, 78, [(color, name, "line") for _, name, color in kinds])
    for i, (title, path) in enumerate(panels):
        sweep = load(path)
        ax = Axes(fig, 72 + i * 332, 124, 276, 170, (1.5, 10.5), (floor, 20), ylog=True)
        ax.title(f"{title}, loss {sweep['loss']:.3f}")
        ax.yticks([0.001, 0.01, 0.1, 1, 10], lambda t: f"+{t:g}")
        ax.xticks([2, 4, 6, 8, 10], str, "bits per weight")
        for group, _, color in kinds:
            points = sorted((r["bits_per_weight"], max(r["loss"] - sweep["loss"], floor), r["bits"]) for r in sweep["settings"] if r["group"] == group)
            ax.line([p[0] for p in points], [p[1] for p in points], color)
            ax.dots([p[0] for p in points], [p[1] for p in points], color, r=3.5)
        # the setting that is used most: 4-bit codes in groups of 64
        bpw, cost = next((r["bits_per_weight"], r["loss"] - sweep["loss"]) for r in sweep["settings"] if (r["bits"], r["group"]) == (4, 64))
        fig.parts.append(f'<circle class="s-ink" cx="{n(ax.x(bpw))}" cy="{n(ax.y(cost))}" r="8" fill="none" stroke-width="1.5"/>')
        circled.append(f"+{cost:.3f}")
    fig.text(64, 356, "Logarithmic scale. Each line has a point for 2, 3, 4, 5, 6 and 8 bits. Points on the bottom line are at +0.001 or below.", size=11, fill="muted")
    fig.text(64, 372, f"Circled: 4-bit codes in groups of 64, the setting used most. It costs {circled[0]} and {circled[1]}.", size=11, fill="muted")
    fig.save("quant-bits")


def quant_task():
    parameters = load("runs/quant/sweep-qwen.json")["parameters"]
    megabytes = lambda bits: parameters * bits / 8e6  # in groups of 64, a code's scales add half a bit
    rows = []
    for label, bits, name in [("16-bit floats", 16, "gsm8k-qwen"), ("8 bits", 8.5, "gsm8k-qwen-q8"), ("6 bits", 6.5, "gsm8k-qwen-6bit"), ("5 bits", 5.5, "gsm8k-qwen-5bit"), ("4 bits", 4.5, "gsm8k-qwen-q4")]:
        r = load(f"runs/eval/{name}.json")
        rows.append((f"{label}, {megabytes(bits):,.0f} MB", r["mean"], r["error"]))
    fig = Figure(720, 300, "GSM8K test problems solved by the 0.5B model at each bit width", "1,319 problems, greedy replies, with 95% confidence intervals. Codes in groups of 64, rounded to the nearest value.")
    scored_bars(fig, 220, 84, 330, rows, 0.6, lambda v: f"{v:.1%}")
    fig.text(24, 280, "The size is that of the weight file. At 5 and 6 bits the weights were rounded in memory and no file was written.", size=11, fill="muted")
    fig.save("quant-task")


def quant_speed():
    cases = load("runs/bench/quant-mps.json")["cases"]
    runtimes = [("this repository", "s1"), ("MLX", "s2")]
    widths = [(16, "16-bit floats"), (8, "8 bits"), (4, "4 bits")]
    value = lambda where, bits, key: next(c[key] for c in cases if (c["where"], c["bits"]) == (where, bits))
    panels = [("Tokens per second, one sequence", "batch_1", lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}"), ("Memory at its peak, GB", "peak_gb", lambda v: f"{v:.2f}")]
    fig = Figure(720, 262, "Where quantization pays", "Generating with the 0.5B model. This repository expands packed weights for every token, and MLX does not.")
    fig.legend(24, 76, [(color, name, "dot") for name, color in runtimes])
    for i, (title, key, fmt) in enumerate(panels):
        left, top, width = 150 + i * 330, 122, 150
        fig.text(left, top - 16, title, fill="ink", weight=600)
        fig.line(left, top - 6, left, top + 40 * len(widths) - 8, stroke="axis")
        peak = max(c[key] for c in cases)
        for j, (bits, label) in enumerate(widths):
            y = top + j * 40
            if i == 0:
                fig.text(left - 10, y + 17, label, anchor="end", fill="ink")
            for k, (where, color) in enumerate(runtimes):
                v = value(where, bits, key)
                bar(fig, left, y + k * 14, width * v / peak, 10, color)
                fig.text(left + max(width * v / peak, 5) + 8, y + 9 + k * 14, fmt(v), size=11)
    fig.save("quant-speed")


# ---------------------------------------------------------- 2b longer context


def rope_freqs(head_dim, theta, kind=None, s=1.0, original=256):
    """The turning speed of every pair of the rotary table, as minllm/model.py's rope_table computes it."""
    half = head_dim // 2
    base = [theta ** (-2 * i / head_dim) for i in range(half)]
    if kind == "linear":
        return [f / s for f in base]
    if kind == "ntk":
        t = theta * s ** (head_dim / (head_dim - 2))
        return [t ** (-2 * i / head_dim) for i in range(half)]
    if kind == "yarn":
        index = lambda turns: head_dim * math.log(original / (turns * 2 * math.pi)) / (2 * math.log(theta))
        low, high = max(math.floor(index(32)), 0), min(math.ceil(index(1)), head_dim - 1)
        ramp = [min(max((i - low) / max(high - low, 1e-3), 0), 1) for i in range(half)]
        return [f * (1 - r) + f / s * r for f, r in zip(base, ramp)]
    return base


def context_pairs():
    fig = Figure(720, 330, "Turns of every rotary pair within the window", "The model of 3b: 32 pairs per head, theta 10,000, trained on 256 tokens. Each method stretches it 4 times, for 1,024.")
    pairs = list(range(32))
    turns = lambda freqs, length: [length * f / (2 * math.pi) for f in freqs]
    ax = Axes(fig, 64, 108, 440, 180, (0, 31), (0.004, 200), ylog=True)
    ax.yticks([0.01, 0.1, 1, 10, 100], lambda t: f"{t:g}")
    ax.xticks([0, 8, 16, 24, 31], str, "pair, from the fastest to the slowest")
    fig.line(ax.left, ax.y(1), ax.left + ax.width, ax.y(1), stroke="axis", width=1.5)
    ax.label(31, 1, "one turn", dx=-4, dy=-6, anchor="end", fill="muted")
    series = [
        ("axis", "as trained, over 256 tokens", turns(rope_freqs(64, 1e4), 256)),
        ("ink2", "as it is, over 1,024", turns(rope_freqs(64, 1e4), 1024)),
        ("s1", "interpolated, over 1,024", turns(rope_freqs(64, 1e4, "linear", 4), 1024)),
        ("s2", "NTK, over 1,024", turns(rope_freqs(64, 1e4, "ntk", 4), 1024)),
        ("s3", "YaRN, over 1,024", turns(rope_freqs(64, 1e4, "yarn", 4), 1024)),
    ]
    for color, label, ys in series:
        ax.line(pairs, ys, color, width=8 if color == "axis" else 2)
    for i, (color, label, ys) in enumerate(series):
        y = 120 + i * 24 + (12 if i > 2 else 0)
        fig.line(530, y - 4, 548, y - 4, stroke=color, width=8 if color == "axis" else 2)
        fig.text(556, y, label, size=11)
    fig.text(556, 120 + 2 * 24 + 13, "on the trained band exactly", size=10, fill="muted")
    fig.text(530, 260, "Below one turn, a pair never", size=11, fill="muted")
    fig.text(530, 275, "came round in training: past the", size=11, fill="muted")
    fig.text(530, 290, "trained angle it reads new ones.", size=11, fill="muted")
    fig.save("context-pairs")


# The runs of 2b: (file in runs/context/, label, color, dashed). The hues are those of context_pairs.
CONTEXT_UNTRAINED = [("web-clean", "not stretched", "ink2", False), ("web-clean-linear4", "interpolated ×4", "s1", False), ("web-clean-ntk4", "NTK ×4", "s2", False),
                     ("web-clean-yarn4", "YaRN ×4", "s3", False), ("web-clean-yarn8", "YaRN ×8", "s3", True)]
CONTEXT_TRAINED = [("ctx-none", "not stretched", "ink2", False), ("ctx-linear", "interpolated ×4", "s1", False), ("ctx-ntk", "NTK ×4", "s2", False),
                   ("ctx-yarn", "YaRN ×4", "s3", False), ("web-clean-1024", "on 1,024 from the start", "ink", True)]


def context_panels(name, title, subtitle, measure, ylim, yticks, yfmt, xtitle, note, dots=False):
    """The runs without training on the left, and those trained on windows of 1,024 on the right."""
    fig = Figure(720, 424, title, subtitle)
    panels = [("Stretched, without training", CONTEXT_UNTRAINED, 64), ("Trained on windows of 1,024", CONTEXT_TRAINED, 418)]
    for heading, series, left in panels:
        ax = Axes(fig, left, 104, 268, 180, (0, 2048), ylim)
        ax.title(heading)
        ax.yticks(yticks, yfmt)
        ax.xticks([0, 512, 1024, 1536, 2048], lambda t: f"{t:,}", xtitle)
        for at, label in ((256, "256"), (1024, "1,024")):  # the trained window, and the stretched one
            fig.path(f"M{n(ax.x(at))} {n(ax.top)} L{n(ax.x(at))} {n(ax.top + ax.height)}", stroke="axis", width=1, dash="3 3")
        runs = [(load(f"runs/context/{file}.json"), label, color, dash) for file, label, color, dash in series if os.path.exists(f"runs/context/{file}.json")]
        for run, label, color, dash in runs:
            xs, ys = measure(run)
            ys = [min(max(y, ylim[0]), ylim[1]) for y in ys]
            ax.line(xs, ys, color, dash="5 4" if dash else None)
            if dots:
                ax.dots(xs, ys, color, r=3)
        for i, (run, label, color, dash) in enumerate(runs):
            x, y = left + (i % 2) * 140, 346 + (i // 2) * 20
            fig.path(f"M{x} {y - 4} L{x + 18} {y - 4}", stroke=color, width=2, dash="5 4" if dash else None)
            fig.text(x + 24, y, label, size=11)
    fig.text(24, 408, note, size=11, fill="muted")
    fig.save(name)


def context_loss():
    blocks = lambda run: ([32 * i + 16 for i in range(len(run["loss_by_32_positions"]))], run["loss_by_32_positions"])
    context_panels("context-loss", "Past the trained window, the loss rises unless the table is stretched far enough",
                   "Loss at every position of the first 2,048 tokens of 256 held-out web pages, in blocks of 32 positions.",
                   blocks, (3.0, 5.0), [3, 3.5, 4, 4.5, 5], lambda t: f"{t:g}", "position in the page",
                   "Dashed verticals: 256, the window the model was trained on, and 1,024, the window the stretch by 4 is meant for.")


def context_passage():
    gain = lambda run: ([int(d) for d in run["passage"]], [v["unseen"] - v["seen"] for v in run["passage"].values()])
    context_panels("context-passage", "A passage seen before, and how far back the model still finds it",
                   "How much lower the loss on a passage of 48 tokens is when the same passage came earlier, against its distance.",
                   gain, (0, 2.1), [0, 0.5, 1, 1.5, 2], lambda t: f"{t:g}", "tokens between the two copies",
                   "The loss on the second copy, subtracted from the loss on the same tokens after a different passage, over 64 pairs.", dots=True)


# ------------------------------------------------------- 4b speculative decoding


def speculative_round():
    # An illustration of one round; the tokens are made up for it.
    fig = Figure(720, 286, "One round of speculative decoding", "The draft guesses four tokens. The model reads them all in one pass, keeps those it agrees with, and adds one of its own.")
    prefix, drafts = "Natalia sold 48/2 =", [" 24", " clips", " in", " April"]
    x0, y0, bw, gap = 24, 96, 82, 8
    fig.rect(x0, y0, 168, 34, fill="surface", stroke="axis", rx=6)
    fig.text(x0 + 12, y0 + 22, prefix, fill="ink")
    xs = [x0 + 168 + gap + i * (bw + gap) for i in range(len(drafts))]
    for x, d in zip(xs, drafts):
        fig.rect(x, y0, bw, 34, fill="s2", stroke="s2", rx=6, opacity=0.12)
        fig.text(x + bw / 2, y0 + 22, d.strip(), fill="ink", anchor="middle")
    fig.text(xs[-1] + bw + 14, y0 + 15, "the draft:", size=11, fill="ink2")
    fig.text(xs[-1] + bw + 14, y0 + 30, "4 small passes", size=11, fill="ink2")

    # The model: one pass over the text and the four guesses, a distribution after each.
    y1 = 158
    fig.rect(x0, y1, xs[-1] + bw - x0, 30, fill="s1", stroke="s1", rx=6, opacity=0.12)
    fig.text(x0 + 12, y1 + 20, "the model: 1 pass, its next token after each of them", fill="ink")
    picks = [" 24", " clips", " in", " May"]

    # What is kept: every guess up to the first the model would not have made, then the model's own token.
    y2 = 222
    for i, (x, d) in enumerate(zip(xs, drafts)):
        agrees = picks[i] == d
        if agrees:
            fig.rect(x, y2, bw, 34, fill="s1", stroke="s1", rx=6, opacity=0.12)
            fig.text(x + bw / 2, y2 + 22, d.strip(), fill="ink", anchor="middle")
            fig.text(x + bw / 2, y2 - 8, "kept", size=11, fill="ink2", anchor="middle")
        else:
            fig.rect(x, y2, bw, 34, fill="surface", stroke="s1", rx=6, width=2)
            fig.text(x + bw / 2, y2 + 22, picks[i].strip(), fill="ink", anchor="middle", weight=600)
            fig.text(x + bw / 2, y2 - 8, f"not {d.strip()}: the model's", size=11, fill="ink2", anchor="middle")
            break
    fig.text(xs[-1] + bw + 14, y2 + 15, "3 kept, 1 added:", size=11, fill="ink2")
    fig.text(xs[-1] + bw + 14, y2 + 30, "4 tokens for 1 pass", size=11, fill="ink2")
    fig.save("speculative-round")


# The pairs measured, in the order of the figure, and one color per draft.
SPECULATIVE = ["speculative-wide-mps", "speculative-qwen2.5-1.5b-instruct-mps", "speculative-qwen2.5-7b-instruct-mps", "speculative-qwen2.5-7b-instruct-draft-1.5b-mps"]
DRAFTS = [("model", "a smaller model", "s1"), ("lookup", "prompt lookup", "s2")]


def speculative_speed():
    """Speed against the model alone, by draft length, for every pair of a model and its draft."""
    summary = load("runs/bench/speculative-summary.json")
    pairs = [summary[name] for name in SPECULATIVE if name in summary]
    greedy = [c for p in pairs for c in p["cases"] if c["temperature"] == 0]
    top = math.ceil(max(c["speedup"] for c in greedy) * 2) / 2
    rows = math.ceil(len(pairs) / 2)
    fig = Figure(720, 140 + rows * 214, "Speed with a draft", "One reply at a time, greedy, on this laptop's GPU: tokens per second against the model alone.")
    fig.legend(64, 84, [(color, label, "line") for _, label, color in DRAFTS] + [("ink2", "the model alone", "line")])
    for i, p in enumerate(pairs):
        left, top_px = 64 + (i % 2) * 340, 150 + (i // 2) * 214
        ks = sorted({c["k"] for c in p["cases"]})
        ax = Axes(fig, left, top_px, 270, 120, (1, 8), (0, top))  # one range of k for every pair
        fig.text(left, top_px - 30, p["label"], fill="ink", weight=600)
        fig.text(left, top_px - 12, f"alone {p['alone']:.0f} tokens a second", size=11, fill="muted")
        ax.yticks(nice_ticks(0, top, 3), lambda t: f"{t:g}×")
        ax.xticks(ks, str, "tokens drafted, k" if i // 2 == rows - 1 else "")
        ax.line([1, 8], [1, 1], "ink2", width=1.5, dash="4 3")
        for draft, _, color in DRAFTS:
            points = sorted((c["k"], c["speedup"]) for c in p["cases"] if c["draft"] == draft and c["temperature"] == 0)
            if points:
                ax.line(*zip(*points), color)
                ax.dots(*zip(*points), color, r=3)
    fig.save("speculative-speed")


def speculative_cost():
    """At k = 4: the tokens a round yields, and what it costs in tokens of the model alone. Their ratio is the speedup."""
    summary = load("runs/bench/speculative-summary.json")
    rows, seen = [], set()
    for name in SPECULATIVE:
        if name not in summary:
            continue
        p = summary[name]
        for draft, label, color in DRAFTS:
            if draft == "lookup" and p["target"] in seen:
                continue  # prompt lookup does not depend on the draft model, and was measured for each target once
            c = next((c for c in p["cases"] if c["draft"] == draft and c["k"] == 4 and c["temperature"] == 0), None)
            if c:
                rows.append((p["label"].split(",")[0], "drafted by " + p["label"].split("by ")[1] if draft == "model" else "prompt lookup", color, c))
        seen.add(p["target"])
    targets = [p["label"].split(",")[0] for p in summary.values()]
    rows.sort(key=lambda r: (targets.index(r[0]), r[1] == "prompt lookup"))  # each model's rows together, prompt lookup last
    right = math.ceil(max(max(c["round_cost"], c["tokens_per_pass"]) for *_, c in rows))
    fig = Figure(720, 150 + len(rows) * 32, "What a round yields and what it costs", "Four tokens drafted, greedy. Both in tokens of the model alone; the round is faster where it yields more than it costs.")
    fig.legend(64, 82, [("s1", "yields, drafted by a smaller model", "dot"), ("s2", "yields, prompt lookup", "dot"), ("ink2", "costs", "dot")])
    ax = Axes(fig, 300, 104, 300, len(rows) * 32, (0, right), (0, len(rows)))
    for i, (target, draft, color, c) in enumerate(rows):
        y = 104 + i * 32 + 16
        fig.text(288, y - 2, target, anchor="end", fill="ink", size=11)
        fig.text(288, y + 11, draft, anchor="end", fill="muted", size=10)
        fig.line(ax.x(min(c["round_cost"], c["tokens_per_pass"])), y, ax.x(max(c["round_cost"], c["tokens_per_pass"])), y, stroke="axis", width=2)
        fig.dot(ax.x(c["round_cost"]), y, "ink2")
        fig.dot(ax.x(c["tokens_per_pass"]), y, color)
        fig.text(ax.x(right) + 14, y + 4, f"{c['speedup']:.2f}×", fill="ink" if c["speedup"] >= 1 else "muted", size=11)
    ax.xticks(nice_ticks(0, right, 5), lambda t: f"{t:g}", "tokens of the model alone")
    fig.save("speculative-cost")


# ------------------------------------------------------------ 3c scaling laws


def budget_label(b):
    mantissa, exponent = f"{b:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def scaling_valleys():
    summary = load("runs/scaling/summary.json")
    valleys = sorted(summary["valleys"].items(), key=lambda kv: float(kv[0]))
    fig = Figure(720, 392, "At every budget there is a best size", "Final validation loss against size, one line per compute budget, darker for more.")
    fig.text(24, 68, "The orange dot is the floor of a parabola through the lowest three runs.", fill="ink2")
    runs = [r for _, v in valleys for r in v["runs"]]
    lo, hi = math.floor(min(r["loss"] for r in runs) * 10) / 10, math.ceil(min(max(r["loss"] for r in runs), 2.6) * 10) / 10
    ax = Axes(fig, 64, 108, 400, 200, (0.25e6, 40e6), (lo, hi), xlog=True)
    ax.title("Loss against parameters")
    ax.yticks(nice_ticks(lo, hi, 4), lambda t: f"{t:g}")
    ax.xticks([0.3e6, 1e6, 3e6, 10e6, 30e6], lambda t: f"{t / 1e6:g}M", "parameters (log scale)")
    for i, (budget, v) in enumerate(valleys):
        shade = 0.35 + 0.65 * i / max(1, len(valleys) - 1)
        rs = [r for r in v["runs"] if r["loss"] <= hi]
        ax.line([r["params"] for r in rs], [r["loss"] for r in rs], "s1", opacity=shade)
        for r in rs:
            if r["passes"] > 1.0:  # part of the text read twice: drawn hollow
                fig.parts.append(f'<circle class="f-surface s-s1" cx="{n(ax.x(r["params"]))}" cy="{n(ax.y(r["loss"]))}" r="3.5" stroke-width="1.5"/>')
            else:
                fig.dot(ax.x(r["params"]), ax.y(r["loss"]), "s1", r=3.5)
        fig.dot(ax.x(v["params"]), ax.y(v["loss"]), "s2", r=4.5)
        last = rs[-1]
        ax.label(last["params"], last["loss"], budget_label(float(budget)) + " FLOPs", dx=8, dy=4, fill="ink2")

    # Tokens per parameter at each floor, against the rule of thumb of Hoffmann et al.
    ratio = Axes(fig, 560, 108, 130, 200, (0.5, len(valleys) + 0.5), (0, 40))
    ratio.title("Tokens per parameter")
    ratio.yticks([0, 10, 20, 30, 40], lambda t: f"{t:g}")
    fig.line(ratio.left, ratio.y(20), ratio.left + ratio.width, ratio.y(20), stroke="s2", width=1.5)
    bottom = ratio.top + ratio.height
    fig.line(ratio.left, bottom, ratio.left + ratio.width, bottom, stroke="axis")
    for i, (budget, v) in enumerate(valleys, start=1):
        r = v["tokens"] / v["params"]
        x, shade = ratio.x(i), 0.35 + 0.65 * (i - 1) / max(1, len(valleys) - 1)
        if v["bracketed"]:
            fig.rect(x - 9, ratio.y(r), 18, bottom - ratio.y(r), fill="s1", rx=3, opacity=shade)
        else:  # the lowest run is the smallest: the floor is less sure, drawn as an outline
            fig.rect(x - 8.25, ratio.y(r) + 0.75, 16.5, bottom - ratio.y(r) - 0.75, fill="surface", stroke="s1", rx=3)
        fig.text(x, ratio.y(r) - 6, f"{r:.0f}", anchor="middle", fill="ink", size=11)
        fig.text(x, bottom + 17, budget_label(float(budget)), anchor="middle", fill="muted", size=10)
    fig.text(ratio.left + ratio.width / 2, bottom + 36, "orange: Chinchilla's 20", size=10, anchor="middle", fill="muted")
    fig.text(24, 374, "Hollow dot: part of TinyStories' 131M tokens read twice. Outlined bar: the budget's smallest run is its lowest.", size=11, fill="muted")
    fig.save("scaling-valleys")


def scaling_prediction():
    summary = load("runs/scaling/summary.json")
    predictions, fit = summary["predictions"], summary["fit_without_largest"]
    budget = predictions[0]["name"].split("-")[0][1:]
    fig = Figure(720, 316, "A law fitted on small budgets, asked about a larger one", f"L(N, D) fitted to every run below {budget} FLOPs, then asked for the runs at {budget}, which it has not seen.")
    xs = [p["params"] for p in predictions]
    ys = [p["loss"] for p in predictions] + [p["predicted"] for p in predictions]
    lo, hi = math.floor(min(ys) * 20) / 20, math.ceil(max(ys) * 20) / 20
    ax = Axes(fig, 64, 108, 360, 140, (min(xs) / 1.3, max(xs) * 1.3), (lo, hi), xlog=True)
    ax.title(f"The runs at {budget} FLOPs")
    ax.yticks(nice_ticks(lo, hi, 3), lambda t: f"{t:g}")
    ax.xticks(xs, lambda t: f"{t / 1e6:.1f}M", "parameters")
    ax.line(xs, [p["predicted"] for p in predictions], "s2")
    ax.dots(xs, [p["loss"] for p in predictions], "s1")
    fig.legend(470, 130, [("s1", "measured", "dot"), ("s2", "predicted", "line")])
    rows = [("E", f"{fit['E']:.2f}", "the loss no model reaches"), ("alpha", f"{fit['alpha']:.2f}", "how fast size helps"), ("beta", f"{fit['beta']:.2f}", "how fast tokens help")]
    for i, (name, value, what) in enumerate(rows):
        y = 172 + i * 22
        fig.text(470, y, name, fill="ink", weight=600)
        fig.text(520, y, value, fill="ink")
        fig.text(560, y, what, size=11)
    worst = max(abs(p["loss"] - p["predicted"]) for p in predictions)
    own = max(summary["misses"], key=lambda name: abs(summary["misses"][name]))
    run = next(r for v in summary["valleys"].values() for r in v["runs"] if r["name"] == own)
    fig.text(24, 300, f"The largest miss is {worst:.3f}. Fitted on every run, the law misses one of its own by {abs(summary['misses'][own]):.3f}: {run['params'] / 1e6:.1f}M parameters on {run['tokens'] / 1e6:.0f}M tokens.", size=11, fill="muted")
    fig.save("scaling-prediction")


# ---------------------------------------------------------------------- readme


def pipeline(done):
    fig = Figure(720, 420, "The pipeline", "Ten stages in order, from raw text to a model that other tools run. A side topic branches off the stage it changes.")
    rows = [
        ("From scratch: a 14M model, trained here", [(1, "Tokenizer"), (2, "Model"), (3, "Pretraining"), (4, "Inference")]),
        ("Post-training: that model, and an open 0.5B one", [(5, "Weights"), (6, "Fine-tuning"), (7, "Evaluation"), (8, "RLHF"), (9, "RLVR"), (10, "Export")]),
    ]
    leaves = {
        2: [("2a", "Experts"), ("2b", "Context"), ("2c", "Latent attention")],
        3: [("3a", "Distillation"), ("3b", "Data"), ("3c", "Scaling")],
        4: [("4a", "Structured output"), ("4b", "Speculative"), ("4c", "Batching")],
        6: [("6a", "Tool calling")],
        7: [("7a", "Multiple choice"), ("7b", "Judge")],
        8: [("8a", "Safety"), ("8b", "DPO")],
        9: [("9a", "Self-training"), ("9b", "Tool RL"), ("9c", "Reasoning")],
        10: [("10a", "Quantization")],
    }
    leaves = {stage: [(tag, topic) for tag, topic in topics if tag in done] for stage, topics in leaves.items()}  # only those that are out
    left, w, gap, tops, step = 20, 98, 16, [96, 268], 21
    for (label, stages), y in zip(rows, tops):
        fig.text(left + (w / 2 + 12 if y == tops[1] else 0), y - 12, label, size=11, fill="muted")
        for i, (number, name) in enumerate(stages):
            x = left + i * (w + gap)
            built = number in done
            fig.rect(x, y, w, 40, fill="s1" if built else "surface", stroke="s1" if built else "axis", rx=6, opacity=0.12 if built else None)
            fig.text(x + 12, y + 25, str(number), fill="muted", size=11)
            fig.text(x + 12 + (16 if number >= 10 else 12), y + 25, name, fill="ink", weight=600 if built else 400)
            if i:
                fig.arrow([(x - gap + 2, y + 20), (x - 2, y + 20)])
            # The side topics hang below their stage as a small tree, outside the chain.
            topics = leaves.get(number, [])
            if topics:
                fig.line(x + 14, y + 40, x + 14, y + 40 + step * len(topics) - 8, stroke="axis", width=1.5)
            for k, (tag, topic) in enumerate(topics):
                ly = y + 40 + step * (k + 1) - 4
                fig.line(x + 14, ly - 4, x + 22, ly - 4, stroke="axis", width=1.5)
                fig.text(x + 26, ly, tag, size=11, fill="muted")
                fig.text(x + 30 + text_width(tag, 11), ly, topic, size=11, fill="ink" if tag in done else "ink2")
    # The chain goes on from the end of the first row, past its side topics, down into the second.
    end = left + 4 * (w + gap) - gap
    past = max(left + 3 * (w + gap) + 30 + text_width(f"{tag} {topic}", 11) for tag, topic in leaves[4]) + 16
    fig.arrow([(end + 2, tops[0] + 20), (past, tops[0] + 20), (past, 236), (left + w / 2, 236), (left + w / 2, tops[1] - 2)])
    fig.text(left + 4, 402, "A side topic uses what its stage built, and the chain does not depend on it.", size=11, fill="muted")
    fig.save("pipeline")


tokenizer_vocab()
model_architecture()
model_attention()
pretraining_loss()
speed_bars("runs/bench/train-mps.json", "pretraining-speed", "Two switches make a training step about four times faster", "on an M5 Pro")
inference_work()
inference_cache()
inference_batch()
pretrained_files()
pretrained_generation()
speed_bars("runs/bench/train-qwen-mps.json", "pretrained-speed", "Training steps of the 0.5B model", "and memory in use")
finetuning_targets()
finetuning_lora()
rlhf_pipeline()
rlvr_loop()
if os.path.exists("runs/sft-gsm8k/meta.json"):
    finetuning_gsm8k()
if os.path.exists("runs/eval/gsm8k-sft-gsm8k.json"):
    evaluation_scores()
if os.path.exists("runs/eval/tools-summary.json"):
    tools_results()
if os.path.exists("runs/structured/reason-then-answer-constrained.json"):
    structured_forms()
if os.path.exists("runs/eval/gsm8k-sft-gsm8k-safe.json"):
    safety_refusals()
if os.path.exists("runs/judge/stories-qwen2.5-1.5b-instruct-rubric.json"):
    judge_gsm8k()
    judge_stories()
if os.path.exists(MMLU):
    choice_methods()
    choice_letters()
    choice_subjects()
    choice_calibration()
    choice_models()
if os.path.exists("runs/rlhf/dpo.jsonl"):
    rlhf_reward()
    rlhf_best_of_n()
    rlhf_ppo()
    rlhf_comparison()
if os.path.exists("runs/rlvr/grpo-words-finish.jsonl"):
    rlvr_words()
if os.path.exists("runs/rlvr/grpo-gsm8k.jsonl"):
    rlvr_gsm8k()
if os.path.exists("runs/data/funnel.json"):
    data_funnel()
if os.path.exists("runs/data/cross.json"):
    data_loss()
if os.path.exists("runs/distill/targets.json"):
    distill_targets()
if os.path.exists("runs/distill/distilled-20min.jsonl"):
    distill_loss()
if os.path.exists("runs/eval/gsm8k-self-gsm8k.json"):
    self_rounds()
    self_results()
if os.path.exists("runs/quant/sweep-qwen.json"):
    quant_levels()
    quant_bits()
if os.path.exists("runs/eval/gsm8k-qwen-5bit.json"):
    quant_task()
if os.path.exists("runs/bench/quant-mps.json"):
    quant_speed()
moe_block()
if os.path.exists("runs/bench/moe-mps.json"):
    moe_speed()
if os.path.exists("runs/moe/wide-20min.jsonl"):
    moe_balance()
    moe_loss()
if os.path.exists("runs/moe/routing.json"):
    moe_routing()
context_pairs()
if os.path.exists("runs/context/web-clean.json"):
    context_loss()
    context_passage()
speculative_round()
if os.path.exists("runs/bench/speculative-summary.json"):
    speculative_speed()
    speculative_cost()
if os.path.exists("runs/rlvr/grpo-gsm8k-tools.jsonl"):
    tool_rl_curves()
if os.path.exists("runs/eval/gsm8k-tools-grpo-gsm8k-tools-no-execute.json"):
    tool_rl_test()
if os.path.exists("runs/rlvr/grpo-qwen.jsonl"):
    reasoning_curves()
if os.path.exists("runs/reasoning/test-time-qwen.json"):
    reasoning_test_time()
if os.path.exists("runs/reasoning/summary.json"):
    reasoning_replies()
if os.path.exists("runs/scaling/summary.json"):
    scaling_valleys()
    scaling_prediction()
latent_block()
batching_timeline()
if os.path.exists("runs/bench/batching-qwen-mps.json"):
    batching_results()
if os.path.exists("runs/bench/attention-mps.json"):
    attention_loss()
    attention_speed()
if os.path.exists("runs/eval/gsm8k-dpo-gsm8k-nll.json"):
    dpo_curves()
    dpo_test()
pipeline(done={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "2a", "2b", "2c", "3a", "3b", "3c", "4a", "4b", "4c", "6a", "7a", "7b", "8a", "8b", "9a", "9b", "9c", "10a"})
