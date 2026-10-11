"""
Draw the figures of the notes into docs/figures/, from the logs in runs/.

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

    def path(self, d, stroke=None, fill=None, width=2, opacity=None):
        cls = (f"s-{stroke} " if stroke else "") + (f"f-{fill}" if fill else "")
        extra = "" if fill else ' fill="none"'
        extra += f' stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"' if stroke else ""
        extra += f' {"fill" if fill else "stroke"}-opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<path class="{cls}" d="{d}"{extra}/>')

    def polyline(self, points, stroke, width=2, opacity=None):
        self.path("M" + " L".join(f"{n(x)} {n(y)}" for x, y in points), stroke=stroke, width=width, opacity=opacity)

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

    def line(self, xs, ys, color, width=2, opacity=None):
        self.fig.polyline([(self.x(a), self.y(b)) for a, b in zip(xs, ys)], color, width, opacity)

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

    curve = sorted((load(f"runs/rlhf/{name}") for name in os.listdir("runs/rlhf") if name.startswith("reward") and name.endswith(".json")), key=lambda m: m["pairs"])
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
    ax.label(0, logs[0][2][0]["held_out"], f"{logs[0][2][0]['held_out']:.0%}", dx=8, dy=16)
    for i, (key, heading, limits, fmt) in enumerate(panels):
        low = Axes(fig, 64, 310 + i * 126, 590, 60, (0, steps), limits)
        low.title(heading)
        low.yticks(nice_ticks(*limits, 2), fmt)
        last = i == len(panels) - 1
        low.xticks(range(0, steps + 1, 25 if steps > 100 else 10), str, "iteration" if last else "")
        for label, color, log in logs:
            rows = [r for r in log if key in r]
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


# ---------------------------------------------------------------------- readme


def pipeline(done):
    fig = Figure(720, 352, "The pipeline", "Ten stages in order, from raw text to a model that other tools run. A side topic branches off the stage it changes.")
    rows = [
        ("From scratch: a 14M model, trained here", [(1, "Tokenizer"), (2, "Model"), (3, "Pretraining"), (4, "Inference")]),
        ("Post-training: that model, and an open 0.5B one", [(5, "Weights"), (6, "Fine-tuning"), (7, "Evaluation"), (8, "RLHF"), (9, "RLVR"), (10, "Export")]),
    ]
    leaves = {2: ("2a", "Experts"), 3: ("3a", "Distillation"), 10: ("10a", "Quantization")}
    left, w, gap, tops = 36, 98, 16, [96, 232]
    for (label, stages), y in zip(rows, tops):
        fig.text(left, y - 12, label, size=11, fill="muted")
        for i, (number, name) in enumerate(stages):
            x = left + i * (w + gap)
            built = number in done
            fig.rect(x, y, w, 40, fill="s1" if built else "surface", stroke="s1" if built else "axis", rx=6, opacity=0.12 if built else None)
            fig.text(x + 12, y + 25, str(number), fill="muted", size=11)
            fig.text(x + 12 + (16 if number >= 10 else 12), y + 25, name, fill="ink", weight=600 if built else 400)
            if i:
                fig.arrow([(x - gap + 2, y + 20), (x - 2, y + 20)])
            if number in leaves:
                # A side topic hangs below the stage it branches from, outside the chain.
                tag, topic = leaves[number]
                built = tag in done
                width = text_width(f"{tag} {topic}", 11) + 24
                lx = x if x + width < 700 else x + w - width
                fig.line(x + w / 2, y + 40, x + w / 2, y + 52, stroke="s1" if built else "axis", width=1.5)
                fig.rect(lx, y + 52, width, 28, fill="surface", stroke="s1" if built else "axis", rx=5)
                fig.text(lx + 10, y + 70, tag, fill="muted", size=11)
                fig.text(lx + 14 + text_width(tag, 11), y + 70, topic, fill="ink" if built else "ink2", size=11)
    # The chain goes on from the end of the first row to the start of the second.
    end = left + 4 * (w + gap) - gap
    fig.arrow([(end + 2, tops[0] + 20), (end + 22, tops[0] + 20), (end + 22, 198), (left - 18, 198), (left - 18, tops[1] + 20), (left - 2, tops[1] + 20)])
    fig.text(left, 336, "A side topic uses what its stage built, and the chain does not depend on it.", size=11, fill="muted")
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
if os.path.exists("runs/rlhf/dpo.jsonl"):
    rlhf_reward()
    rlhf_best_of_n()
    rlhf_ppo()
    rlhf_comparison()
if os.path.exists("runs/rlvr/grpo-words-finish.jsonl"):
    rlvr_words()
if os.path.exists("runs/rlvr/grpo-gsm8k.jsonl"):
    rlvr_gsm8k()
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
pipeline(done={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "2a", "10a"})
