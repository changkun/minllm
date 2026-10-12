"""
Turn pages of the web into three versions of a training corpus, each one
step cleaner than the last, and count what every step removes.

    uv run scripts/prepare_data.py              # 8 files of Common Crawl to train on, 1 to hold out
    uv run scripts/prepare_data.py --files 2    # a smaller sample

Writes, in the form that scripts/encode_data.py and scripts/train.py read:

    data/web/raw/     every page that the crawl took for English, as it was extracted
    data/web/rules/   only its lines of prose, and only the pages that pass the rules
    data/web/clean/   the same without copies: of pages, of near copies, of repeated lines

and the counts of every step in runs/data/funnel.json. See docs/03b-data.md.
"""

import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter

from minllm import prepare
from minllm.common import fetch
from minllm.data import EOT

parser = argparse.ArgumentParser()
parser.add_argument("--crawl", default="CC-MAIN-2024-33", help="the crawl of August 2024")
parser.add_argument("--files", type=int, default=8, help="files of the crawl to train on; the one after them is held out")
parser.add_argument("--dir", default="data/web")
parser.add_argument("--out", default="runs/data/funnel.json")
args = parser.parse_args()

BASE = "https://data.commoncrawl.org/"


def patiently(get):
    """Common Crawl answers 503 when it is busy, and wants to be asked again later."""
    for wait in (5, 15, 45, 120, 300, 600):
        try:
            return get()
        except urllib.error.HTTPError as error:
            if error.code != 503:
                raise
            print(f"busy, asking again in {wait}s")
            time.sleep(wait)
    return get()


paths = gzip.decompress(patiently(lambda: urllib.request.urlopen(f"{BASE}crawl-data/{args.crawl}/wet.paths.gz").read())).decode().split()
# A segment of the crawl is one stretch of time, and its files share sites. Taking files
# from far apart in the list makes the held-out pages come from another part of the crawl.
picked = [paths[i * (len(paths) // (args.files + 1))] for i in range(args.files + 1)]
splits = {"train": picked[: args.files], "valid": picked[args.files :]}


def write(path, texts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace(EOT, "") + EOT)
    return sum(map(len, texts))


funnel = {"crawl": args.crawl, "files_in_crawl": len(paths), "files": picked}
start = time.time()


def showable(text, lines):
    """A page to show what the line rule does: short, and with a menu to get through before its first sentence."""
    every = [line.strip() for line in text.split("\n") if line.strip()]
    first = next(i for i, line in enumerate(every) if line in lines)
    if 20 <= first <= 45 and len(lines) >= 6 and len(every) <= 120:
        return [[line[:100], line in lines] for line in every]


for split, files in splits.items():
    count = Counter()
    raw, ruled, why = [], [], Counter()
    for remote in files:
        local = f"{args.dir}/crawl/{os.path.basename(remote)}"
        patiently(lambda: fetch(BASE + remote, local))
        for url, language, text in prepare.pages(local):
            count["pages"] += 1
            count["characters"] += len(text)
            if language != "eng":
                continue
            # 1. the crawl's own guess of the language
            text = text.strip()
            raw.append(text)
            # 2. lines of prose, then the rules for a whole page
            lines = prepare.prose(text)
            count["prose characters"] += sum(map(len, lines))
            reason = prepare.verdict(lines)
            if reason:
                why[reason] += 1
                continue
            if "example" not in funnel and (shown := showable(text, lines)):
                funnel["example"] = {"url": url, "lines": shown}  # the first such page of the first file
            text, replaced = prepare.scrub("\n".join(lines))
            count["addresses replaced"] += replaced
            ruled.append(text)
    # 3. copies: the same page again, a page nearly the same, and lines that many pages share
    seen, unique = set(), []
    for text in ruled:
        if (d := prepare.digest(text)) not in seen:
            seen.add(d)
            unique.append(text)
    minhash = prepare.MinHash()
    near = prepare.near_duplicates([minhash.signature(text) for text in unique])
    distinct = [text for i, text in enumerate(unique) if i not in near]
    shared = prepare.repeated_lines(distinct)
    clean = []
    for text in distinct:
        kept = [line for line in text.split("\n") if line not in shared]
        if len(" ".join(kept).split()) >= 50:  # a page that was mostly shared lines is no longer a page
            clean.append("\n".join(kept))
    steps = [
        ("every page of the files", count["pages"], count["characters"]),
        ("tagged English by the crawl", len(raw), write(f"{args.dir}/raw/{split}.txt", raw)),
        ("lines of prose", len(raw), count["prose characters"]),
        ("pages that pass the rules", len(ruled), write(f"{args.dir}/rules/{split}.txt", ruled)),
        ("without exact copies", len(unique), sum(map(len, unique))),
        ("without near copies", len(distinct), sum(map(len, distinct))),
        ("without lines shared by pages", len(clean), write(f"{args.dir}/clean/{split}.txt", clean)),
    ]
    funnel[split] = {"steps": [{"step": name, "pages": pages, "characters": characters} for name, pages, characters in steps], "dropped_by_rule": dict(why.most_common()), "addresses_replaced": count["addresses replaced"], "shared_lines": len(shared), "most_shared_lines": [[line, pages] for line, pages in sorted(shared.items(), key=lambda item: -item[1])[:12]]}
    print(f"{split}: {len(files)} files, {time.time() - start:.0f}s")
    for name, pages, characters in steps:
        print(f"  {pages:>9,} pages {characters / 1e6:>8.1f}M characters  {name}")
    print("  dropped by rule:", ", ".join(f"{reason} {n:,}" for reason, n in why.most_common()))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(funnel, f, indent=1)
print(f"wrote {args.out}")
