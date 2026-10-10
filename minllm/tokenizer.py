"""
Byte-level BPE tokenizer.

A language model reads integers, not text. The tokenizer is the map between
the two:

    text --split--> chunks --utf-8--> bytes --merges--> token ids

Training starts from the 256 possible byte values and repeatedly glues the
most frequent adjacent pair of tokens into a new token, until the vocabulary
has the requested size. Encoding replays those merges in the order they were
learned. Decoding is a table lookup.

This is the algorithm behind the GPT, Llama 3 and Qwen tokenizers. See
docs/01-tokenizer.md.
"""

import json
from collections import Counter, defaultdict

import regex

# GPT-4's split pattern. Text is cut into words, numbers, punctuation runs and
# whitespace before any merging, so that no token spans two words. It also
# makes training cheap: a corpus has few distinct chunks, however long it is.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """
    Token ids are laid out as
        [0, 256)              raw bytes
        [256, 256 + merges)   learned merges, in the order they were learned
        the rest              special tokens such as <|endoftext|>
    """

    def __init__(self, merges=None, special_tokens=None, pattern=SPLIT_PATTERN):
        self.merges = dict(merges or {})  # (id, id) -> id
        self.pattern = pattern
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.special_tokens = {}  # str -> id
        for s in special_tokens or []:
            self.special_tokens[s] = len(self.vocab)
            self.vocab[len(self.vocab)] = s.encode("utf-8")

        self._split = regex.compile(pattern)
        # Longest first, so that a special token never loses to its own prefix.
        specials = sorted(self.special_tokens, key=len, reverse=True)
        self._special = regex.compile("(" + "|".join(map(regex.escape, specials)) + ")") if specials else None
        self._cache = {}  # chunk -> ids, since the same words recur constantly

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ------------------------------------------------------------------ train

    @classmethod
    def train(cls, texts, vocab_size, special_tokens=None, pattern=SPLIT_PATTERN, verbose=False):
        """
        Learn `vocab_size - 256 - len(special_tokens)` merges from `texts`,
        a string or an iterable of strings.
        """
        special_tokens = list(special_tokens or [])
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0, "vocab_size is too small to hold the bytes and the special tokens"
        if isinstance(texts, str):
            texts = [texts]

        # Count each distinct chunk once. Special tokens are cut out first:
        # they get their ids by decree, not by merging.
        base = cls(special_tokens=special_tokens, pattern=pattern)
        chunks = Counter()
        for text in texts:
            for part in base._split_special(text):
                if part not in base.special_tokens:
                    chunks.update(base._split.findall(part))
        seqs = [list(chunk.encode("utf-8")) for chunk in chunks]  # each chunk as token ids
        freqs = list(chunks.values())  # how often each chunk occurs

        # pairs[p] is how often the pair p occurs in the corpus, and where[p]
        # lists the chunks that contain it. Keeping both up to date means a
        # merge only has to touch the chunks it changes, instead of recounting
        # the whole corpus as the textbook version does.
        pairs = Counter()
        where = defaultdict(set)
        for i, seq in enumerate(seqs):
            for pair in zip(seq, seq[1:]):
                pairs[pair] += freqs[i]
                where[pair].add(i)

        merges = {}
        for new_id in range(256, 256 + n_merges):
            if not pairs:
                break  # every chunk is a single token already
            # The most frequent pair wins; ties go to the larger pair, so the
            # result does not depend on dictionary order.
            top = max(pairs.values())
            best = max(pair for pair, count in pairs.items() if count == top)
            for i in where.pop(best):
                seq = seqs[i]
                for pair in zip(seq, seq[1:]):  # withdraw this chunk's old pairs
                    pairs[pair] -= freqs[i]
                    if pairs[pair] == 0:
                        del pairs[pair]
                seq = seqs[i] = merge(seq, best, new_id)
                for pair in zip(seq, seq[1:]):  # and count its new ones
                    pairs[pair] += freqs[i]
                    where[pair].add(i)
            merges[best] = new_id
            if verbose and (new_id + 1) % 256 == 0:
                tok = cls(merges, pattern=pattern)
                print(f"merge {new_id - 255:5d}/{n_merges}: {tok.vocab[new_id]!r} ({top} occurrences)")

        return cls(merges, special_tokens, pattern)

    # ----------------------------------------------------------------- encode

    def _split_special(self, text):
        """Cut `text` into ordinary text and special tokens, kept in order."""
        return self._special.split(text) if self._special else [text]

    def _encode_chunk(self, chunk):
        ids = self._cache.get(chunk)
        if ids is None:
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                # Replay training: of all the pairs present, the one learned
                # earliest (the lowest new id) is merged first.
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special tokens as the plain characters they are spelled with."""
        ids = []
        for chunk in self._split.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text):
        """
        Encode text, mapping special tokens to their ids. Use encode_ordinary
        for text you do not control, so that it cannot smuggle in control tokens.
        """
        ids = []
        for part in self._split_special(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ----------------------------------------------------------------- decode

    def decode(self, ids):
        # A token can end in the middle of a multi-byte character, so an
        # arbitrary id sequence need not be valid UTF-8.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------- disk

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "pattern": self.pattern,
                    "merges": list(self.merges),  # the i-th pair became token 256 + i
                    "special_tokens": list(self.special_tokens),
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        merges = {tuple(pair): 256 + i for i, pair in enumerate(state["merges"])}
        return cls(merges, state["special_tokens"], state["pattern"])
