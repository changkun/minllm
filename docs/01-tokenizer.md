# 1. Tokenizer

Code: [`minllm/tokenizer.py`](../minllm/tokenizer.py) ·
Script: [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py) ·
Tests: [`tests/test_tokenizer.py`](../tests/test_tokenizer.py)

A language model computes with integers. The tokenizer decides which integers:
it maps text to a sequence of token ids and back. Everything downstream
inherits its choices. The vocabulary size fixes the shape of the model's first
and last layer, and the number of tokens per sentence fixes how much text fits
in the context and how much a training step costs.

## The trade-off

There are two obvious ways to cut text into pieces, and both are bad.

| Unit | Vocabulary | Sequence length | Problem |
|---|---|---|---|
| Bytes | 256 | 1 per byte | Sequences are long, and the model spends its capacity on spelling |
| Words | Unbounded | 1 per word | Any word not seen in training has no id |

Byte pair encoding (BPE) sits between them. Frequent words become one token,
rare words are spelled out in a few pieces, and since the pieces bottom out in
single bytes, every possible string has an encoding. There is no "unknown"
token.

## The algorithm

Start with the text as UTF-8 bytes, so the initial vocabulary is the 256 byte
values. Then repeat:

1. Count every pair of adjacent tokens.
2. Give the most frequent pair a new id.
3. Replace that pair with the new id everywhere.

Each round adds one token to the vocabulary and shortens the text. After
`vocab_size - 256` rounds, training is done, and the tokenizer is the ordered
list of merges. This is the whole of it:

```python
for new_id in range(256, vocab_size):
    pairs = Counter()
    for seq in seqs:                          # the corpus, as lists of ids
        pairs.update(zip(seq, seq[1:]))
    best = max(pairs, key=pairs.get)
    seqs = [merge(seq, best, new_id) for seq in seqs]
    merges[best] = new_id
```

Encoding new text replays the merges in the order they were learned. Decoding
concatenates the bytes of each token.

## Three things the real version adds

**Split first, then merge.** The text is cut by a regular expression into
words, numbers, punctuation runs and whitespace, and merges happen only inside
a piece. A token can then never be `"e. T"`, half of one sentence glued to the
start of the next. The pattern in `SPLIT_PATTERN` is the one GPT-4 uses. Note
that a word keeps its leading space: `" time"` and `"time"` are different
tokens, and the first is far more common.

**Count distinct chunks once.** The loop above rescans the corpus for every
merge. But after splitting, a corpus is a bag of chunks, and there are few
distinct ones: the first 64 MB of TinyStories contain 15.7 million chunks and
only 19,880 different ones. `Tokenizer.train` stores
each distinct chunk once with its count, keeps the pair counts up to date, and
on each merge touches only the chunks that contain the merged pair. It learns
3,837 merges from 64 MB in about 5 seconds of plain Python.
`test_train_matches_textbook` checks that it learns exactly the merges of the
loop above.

**Special tokens.** `<|endoftext|>` marks the boundary between documents. It
is not learned by merging. It is cut out of the text before training and given
an id after the merges. `encode` maps the string `<|endoftext|>` to that id,
and `encode_ordinary` does not, which is what you want for text that comes
from a user: otherwise a user could type a control token. The two chat markers
`<|im_start|>` and `<|im_end|>` are reserved now and used from the fine-tuning
stage on, so that the vocabulary never has to change.

## Run it

```sh
uv run scripts/download_data.py
uv run scripts/train_tokenizer.py
```

```
merge   256/3837: b' Sue' (30081 occurrences)
merge  1024/3837: b' draw' (3758 occurrences)
merge  2816/3837: b' spaghetti' (590 occurrences)
merge  3584/3837: b' Poppy' (406 occurrences)
learned 3837 merges in 4.9s
saved out/tokenizer.json, vocabulary size 4096
compression on held-out stories: 4.04 bytes per token
Once |  upon |  a |  time | , |  there |  was |  a |  little |  girl |  named |  Lily | .
To | ken | iz | ation |  of |  un | f | am | il | i | ar |  words | : |   | x | y | l | oph | one | , |   | 2 | 0 | 2 | 6 | .
```

What to notice:

- The first merges are letter pairs (`" t"`, `"he"`, `" a"`), and `" the"`
  arrives at merge 7. Late merges are whole words that occur a few hundred
  times.
- Text from the training distribution costs one token per word, 4.04 bytes
  per token. The model will see sequences four times shorter than the bytes.
- Words the corpus never used fall apart into small pieces. The tokenizer is
  fitted to its corpus as much as the model will be. This one is good at
  children's stories and poor at everything else.

`out/tokenizer.json` is readable: a list of id pairs, where entry `i` is the
pair that became token `256 + i`.

## Choosing the vocabulary size

A larger vocabulary gives shorter sequences, but each token is seen less
often in training, and the embedding table grows by one row of `dim` numbers
per token. TinyStories draws on a small set of words: with 4,096 tokens, 97%
of the words in held-out stories are a single token. With the model of the
next stage, the
embedding table then holds 1.6M of 14.2M parameters. With GPT-2's vocabulary
of 50,257 it would hold 19M and dwarf the rest of the model.

The vocabulary also fits in 16 bits per token, which is how the encoded corpus
is stored on disk.

## Exercises

1. Train with `--vocab-size 512` and `--vocab-size 16384`. How do the bytes
   per token change, and how many occurrences does the last merge have in each
   case? What does that say about how well the last tokens can be learned?
2. Replace `SPLIT_PATTERN` with `r"(?s).+"`, so that the whole text is one
   chunk, and train on a few hundred kilobytes. Look at the merges: which
   tokens now span a word boundary? Why is training so much slower?
3. Encode `"hello<|endoftext|>"` with `encode` and with `encode_ordinary`.
4. Find a string where `decode(encode(s)) == s` holds but decoding the two
   halves of the id list separately does not give the two halves of `s`.

## In practice

The algorithm is the same in production; the implementation is in Rust or C++
for speed. OpenAI's `tiktoken` and Hugging Face's `tokenizers` implement
byte-level BPE as described here. GPT-4, Llama 3 and Qwen use it with
vocabularies of 100k to 150k tokens trained on web-scale text in many
languages. Llama 2 and Gemma use SentencePiece, which merges Unicode
characters rather than bytes and needs a fallback for characters it has not
seen.

A model and its tokenizer are inseparable. Weights trained with one
vocabulary are meaningless with another, which is why loading a published
model later starts with loading its merges into this class.
