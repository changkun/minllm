"""
Constrained decoding: a reply that cannot leave its form.

A model asked for JSON usually writes JSON. A program that reads the reply
needs it always. The model gives every token a score at every step, so
"always" is cheap to have: before a token is picked, every token that
would take the text out of the wanted form is ruled out. The model still
chooses, among what is left.

The form is an automaton over bytes (minllm/automaton.py), and a token is
several bytes. So for a state of the automaton, a token is allowed if all
of its bytes can be read from there, and it leads to the state after the
last of them. Index works that out for the whole vocabulary at once, for
every state. See docs/04a-structured-output.md.
"""

import time

import numpy as np
import torch

from minllm import chat, inference
from minllm import schema as schemas
from minllm.automaton import Automaton


class Index:
    """For every state of an automaton: where each token of a vocabulary leads from there, or that it leads nowhere."""

    def __init__(self, automaton, tok, size=None):
        """size: the number of logits the model has, which can be more than the tokens of the vocabulary."""
        start = time.time()
        self.automaton = automaton
        special = set(tok.special_tokens.values())  # control tokens are no text, whatever they spell
        pieces = [b"" if token in special else tok.vocab.get(token, b"") for token in range(size or tok.vocab_size)]
        self.lengths = np.array([len(piece) for piece in pieces])
        self.bytes = np.zeros((len(pieces), self.lengths.max()), dtype=np.uint8)
        for token, piece in enumerate(pieces):
            self.bytes[token, : len(piece)] = np.frombuffer(piece, dtype=np.uint8)
        self.next = [self._walk(state) for state in range(len(automaton.table))]
        # How many tokens a text needs at least to reach a match from each state. A
        # distance in bytes would not do: a token can be one byte or ten.
        never = 1 << 30  # more than any budget
        self.tokens_to_end = np.where(automaton.accepting, 0, never)
        for _ in range(len(self.next)):
            ahead = np.array([1 + self.tokens_to_end[n[n >= 0]].min(initial=never - 1) for n in self.next])
            self.tokens_to_end = np.minimum(self.tokens_to_end, ahead)
        # Per state: how many tokens each token leaves still to go, never for a token that cannot be read.
        self.left_after = [np.where(n >= 0, self.tokens_to_end[np.maximum(n, 0)], never) for n in self.next]
        # The most tokens that any readable token leaves to go: with more than that
        # left in a row's budget, the budget rules out nothing.
        self.most_left = [int(a[a < never].max(initial=0)) for a in self.left_after]
        self.masks = {}  # (state, stop, device) -> which tokens may come next, when the budget rules out nothing
        self.seconds = time.time() - start  # what building it took

    def _walk(self, state):
        states = np.full(len(self.lengths), state, dtype=np.int32)
        states[self.lengths == 0] = -1
        # All tokens walk through the table together, one byte at a time.
        for j in range(self.bytes.shape[1]):
            live = (self.lengths > j) & (states >= 0)
            if not live.any():
                break
            states[live] = self.automaton.table[states[live], self.bytes[live, j]]
        return states

    def after(self, state):
        """The state after each token when read from `state`, (vocab,), with -1 for a token that cannot be read."""
        return self.next[state]

    def mask(self, state, stop, device, left=None):
        """
        Which tokens may come next from `state`, as a boolean tensor on the
        device: those that can be read, and the stop token where the text is
        complete. With `left` tokens to go, also only those after which the
        rest still reach the end. Without it the mask is made once per state.
        """
        key = (state, stop, str(device))
        if left is None and key in self.masks:
            return self.masks[key]
        allowed = self.left_after[state] <= (left - 2 if left is not None else self.most_left[state])  # after it, the stop needs a token
        allowed[stop] = bool(self.automaton.accepting[state])
        allowed = torch.from_numpy(allowed).to(device)
        if left is None:
            self.masks[key] = allowed
        return allowed


class Constraint:
    """
    What inference.stream asks before every step: which tokens
    may come next, for every row of a batch. It follows each row through
    the automaton. A row may stop where its text is complete. It may not
    pick a token after which its remaining tokens would no longer reach
    the end of the form, since a text that fits so far and is cut off does
    not fit: near the end of its budget, a row is led to the nearest end.
    """

    def __init__(self, index, rows, stop, budget, device="cpu"):
        assert index.tokens_to_end[0] < budget, f"the form needs {index.tokens_to_end[0]} tokens at least, and {budget} were given"
        self.index, self.stop, self.budget, self.device = index, stop, budget, device
        self.states = [0] * rows  # -1 once a row has stopped
        self.steps = 0
        self.done = torch.zeros(len(index.lengths), dtype=torch.bool, device=device)
        self.done[stop] = True  # a row that has stopped only stops again

    def __call__(self, previous):
        if previous is not None:
            self.steps += 1
            for row, token in enumerate(previous.tolist()):
                if self.states[row] >= 0:
                    self.states[row] = -1 if token == self.stop else int(self.index.after(self.states[row])[token])
        left = self.budget - self.steps  # tokens that can still be generated, this one included
        rows = []
        for state in self.states:
            if state < 0:
                rows.append(self.done)
            elif left - 2 >= self.index.most_left[state]:
                rows.append(self.index.mask(state, self.stop, self.device))  # the same for every step, made once
            else:
                rows.append(self.index.mask(state, self.stop, self.device, left))  # near the end of the budget
        return torch.stack(rows)


@torch.no_grad()
def replies(model, tok, examples, schema, max_new_tokens, batch_size=64, temperature=0.0, progress=None, index=None, timing=None, pattern=None):
    """
    The model's reply to every example, as text that fits the schema. Like
    evaluate.replies, with the one difference that every step may only pick
    a token that the schema allows. A regular expression can be given in
    place of the schema as `pattern`. timing, a dict if given, receives the
    seconds that building the index took and the size of the automaton.
    """
    index = index or Index(Automaton(pattern or schemas.regex(schema)), tok, model.cfg.vocab_size)
    stop = tok.special_tokens[chat.IM_END]
    prompts = [chat.encode(tok, e["messages"]) for e in examples]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    device_type = next(model.parameters()).device.type
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        steps = min(max_new_tokens, model.cfg.max_seq_len - max(len(prompts[i]) for i in chunk))  # as many as inference.stream will take
        allow = Constraint(index, len(chunk), stop, steps, next(model.parameters()).device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type != "cpu"):
            ids = inference.generate(model, [prompts[i] for i in chunk], max_new_tokens, temperature=temperature, stop=[stop], allow=allow)
        for i, row in zip(chunk, ids):
            out[i] = tok.decode(row)
        if progress:
            progress(min(start + batch_size, len(order)), len(order))
    if timing is not None:
        a = index.automaton
        timing |= {"index": index.seconds, "states": len(index.next), "bytes_to_end": int(a.distance[0]), "tokens_to_end": int(index.tokens_to_end[0])}
        timing["allowed_at_start"] = int((index.left_after[0] < index.most_left[0] + 1).sum())
    return out
