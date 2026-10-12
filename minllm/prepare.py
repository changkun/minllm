"""
Data preparation: from pages of the web to text worth training on.

The corpus of stage 3 was written to be clean. Text collected from the web
is not: most of a page is menus and buttons, many pages are in another
language than the one wanted, some are spam, and the same text turns up
again and again. What a model is trained on is what a chain of plain rules
leaves over. The rules here are the ones that published pipelines describe:
the lines and sentences of C4, the page statistics of MassiveText, the
anonymizing of FineWeb, and MinHash for pages that are nearly the same.
See docs/03b-data.md.
"""

import gzip
import hashlib
import re
from collections import Counter

import numpy as np

# ------------------------------------------------------------------ reading


def pages(path):
    """
    Yield every page of a Common Crawl WET file as (url, language, text).
    WET is the crawl with the HTML already taken out: one record per page,
    each a block of headers, an empty line, and as many bytes of text as the
    header Content-Length says. language is the first language that the
    crawl itself identified on the page, or "" if it named none.
    """
    with gzip.open(path, "rb") as f:
        while line := f.readline():
            if line.strip() != b"WARC/1.0":
                continue
            headers = {}
            while (line := f.readline().strip()):
                name, _, value = line.partition(b": ")
                headers[name.decode()] = value.decode()
            body = f.read(int(headers["Content-Length"]))
            if headers.get("WARC-Type") == "conversion":  # the other records describe the file itself
                language = headers.get("WARC-Identified-Content-Language", "").split(",")[0]
                yield headers.get("WARC-Target-URI", ""), language, body.decode("utf-8", "replace")


# -------------------------------------------------------------------- lines

TERMINAL = (".", "!", "?", '"', "”", "’")


def prose(text, min_words=5):
    """
    The lines of a page that read as sentences: at least `min_words` words,
    ending in a period, a question mark, an exclamation mark or a closing
    quote. Menus, buttons, headings and lists of links have neither.
    """
    return [line for line in (raw.strip() for raw in text.split("\n")) if len(line.split()) >= min_words and line.endswith(TERMINAL)]


# -------------------------------------------------------------------- pages

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
WORD = re.compile(r"[a-z']+")


def verdict(lines):
    """
    Why a page is dropped, or None if it is kept. lines are its lines of
    prose. The rules are checked in this order, and the first that fails
    names the reason:

      too short       fewer than 50 words
      not English     under 2 of the 8 commonest function words, or more
                      than a tenth of its letters outside the Latin alphabet
      not words       under 80% of the words contain a letter, or a mean
                      word length outside 3 to 10 characters
      symbols         more than one # or ... per ten words
      repeats itself  over 30% of its lines occur more than once in it
    """
    text = "\n".join(lines)
    words = text.split()
    if len(words) < 50:
        return "too short"
    letters = [c for c in text if c.isalpha()]
    if len(STOP_WORDS & set(WORD.findall(text.lower()))) < 2 or sum(c.isascii() for c in letters) < 0.9 * len(letters):
        return "not English"
    if sum(any(c.isalpha() for c in w) for w in words) < 0.8 * len(words) or not 3 <= sum(map(len, words)) / len(words) <= 10:
        return "not words"
    if text.count("#") + text.count("...") + text.count("…") > 0.1 * len(words):
        return "symbols"
    if len(lines) - len(set(lines)) > 0.3 * len(lines):
        return "repeats itself"
    return None


EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def scrub(text):
    """E-mail and IP addresses replaced by placeholders. Returns the text and how many were replaced."""
    text, emails = EMAIL.subn("email@example.com", text)
    text, ips = IPV4.subn("0.0.0.0", text)
    return text, emails + ips


# --------------------------------------------------------------- duplicates


def digest(text):
    """A fingerprint of a text that ignores case and spacing: equal for exact copies."""
    return hashlib.blake2b(" ".join(text.lower().split()).encode(), digest_size=8).digest()


def shingles(text, k=5):
    """The set of all runs of k words in a text, each as a 64-bit number."""
    words = text.lower().split()
    runs = {" ".join(words[i : i + k]) for i in range(max(1, len(words) - k + 1))}
    return np.array([int.from_bytes(hashlib.blake2b(run.encode(), digest_size=8).digest(), "little") for run in runs], dtype=np.uint64)


class MinHash:
    """
    Signatures that tell how much two texts overlap without comparing
    them. A text is the set of its shingles. A random hash function puts
    all shingles in a random order, and the smallest hash of a text is its
    first shingle in that order. Two texts share that first shingle with a
    probability equal to the share of shingles they have in common, their
    Jaccard similarity. So with n hash functions, the share of the n
    smallest hashes on which two texts agree estimates their similarity.
    """

    def __init__(self, n=128, seed=0):
        rng = np.random.default_rng(seed)
        # h(x) = a * x + b, in 64-bit arithmetic that wraps around: a must be odd to keep it one to one
        self.a = rng.integers(1, 1 << 63, n, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self.b = rng.integers(0, 1 << 63, n, dtype=np.uint64)

    def signature(self, text):
        x = shingles(text)
        return (self.a[:, None] * x[None, :] + self.b[:, None]).min(axis=1)


def similarity(a, b):
    """The estimated Jaccard similarity of two texts, from their signatures."""
    return float((a == b).mean())


def near_duplicates(signatures, bands=16, threshold=0.8):
    """
    Which texts are a near copy of an earlier one, as a set of indices.
    Comparing all pairs is too many, so a signature is cut into bands, and
    two texts are compared only if they agree on a whole band: with 16
    bands of 8 numbers, texts that are 80% alike meet in some band 95% of
    the time, and texts 40% alike 1% of the time. A pair that meets is
    then checked against the threshold on its full signatures.
    """
    rows = len(signatures[0]) // bands
    seen = [{} for _ in range(bands)]
    copies = set()
    for i, signature in enumerate(signatures):
        keys = [signature[b * rows : (b + 1) * rows].tobytes() for b in range(bands)]
        candidates = {seen[b][key] for b, key in enumerate(keys) if key in seen[b]}
        if any(similarity(signature, signatures[j]) >= threshold for j in candidates):
            copies.add(i)
            continue
        for b, key in enumerate(keys):
            seen[b].setdefault(key, i)  # only texts that were kept are compared against
    return copies


def repeated_lines(texts, limit=2):
    """
    The lines that occur in more than `limit` texts, each with the number
    of texts it is in: notices about cookies, copyright lines and slogans
    that a site puts on every page. They pass as prose, and no page that
    holds one is a copy of another.
    """
    counts = Counter(line for text in texts for line in set(text.split("\n")))
    return {line: count for line, count in counts.items() if count > limit}
