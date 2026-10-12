"""
A regular expression as an automaton over bytes.

Constrained decoding needs to know, after every token, whether the text so
far can still become a text of the wanted form, and which bytes may come
next. A regular expression answers that once it is compiled into a
deterministic automaton: a table that says, for every state and every byte,
which state follows, or that none does. Python's re module compiles
patterns too, and can only say whether a finished text matches.

The expressions here are a small dialect, enough to write JSON in: literal
characters, classes like [0-9a-f] and [^"\\\\], groups, |, and the
repetitions *, + and ?. See docs/04a-structured-output.md.
"""

import numpy as np

# ------------------------------------------------------------------ parsing
#
# A parsed expression is a tree of tuples:
#   ("bytes", set of byte values)        one byte out of a set
#   ("cat", [parts]), ("alt", [parts])   one after another, or one of several
#   ("star", part)                       any number of times, also none


def parse(pattern):
    """The tree of a pattern. Characters outside ASCII stand for their UTF-8 bytes, one after another."""
    data = pattern.encode("utf-8")
    tree, end = _alternatives(data, 0)
    assert end == len(data), f"unexpected {chr(data[end])!r} at {end} in {pattern!r}"
    return tree


def _alternatives(data, i):
    options = []
    while True:
        parts = []
        while i < len(data) and data[i] not in b"|)":
            part, i = _repeated(data, i)
            parts.append(part)
        options.append(("cat", parts))
        if i == len(data) or data[i] != ord("|"):
            return (options[0] if len(options) == 1 else ("alt", options)), i
        i += 1


def _repeated(data, i):
    part, i = _atom(data, i)
    while i < len(data) and data[i] in b"*+?":
        if data[i] == ord("*"):
            part = ("star", part)
        elif data[i] == ord("+"):
            part = ("cat", [part, ("star", part)])
        else:
            part = ("alt", [part, ("cat", [])])
        i += 1
    return part, i


ESCAPES = {ord("n"): 0x0A, ord("r"): 0x0D, ord("t"): 0x09}


def _char(data, i):
    """One byte of a pattern, which may be written with a backslash: \\n, \\x1f, or a character that otherwise means something."""
    if data[i] != ord("\\"):
        return data[i], i + 1
    if data[i + 1] == ord("x"):
        return int(data[i + 2 : i + 4], 16), i + 4
    return ESCAPES.get(data[i + 1], data[i + 1]), i + 2


def _atom(data, i):
    if data[i] == ord("("):
        tree, i = _alternatives(data, i + 1)
        assert data[i] == ord(")"), "a group is not closed"
        return tree, i + 1
    if data[i] == ord("["):
        negated = data[i + 1] == ord("^")
        i += 2 if negated else 1
        chosen = set()
        while data[i] != ord("]"):
            low, i = _char(data, i)
            high = low
            if data[i] == ord("-") and data[i + 1] != ord("]"):
                high, i = _char(data, i + 1)
            chosen.update(range(low, high + 1))
        return ("bytes", set(range(256)) - chosen if negated else chosen), i + 1
    byte, i = _char(data, i)
    return ("bytes", {byte}), i


# ------------------------------------------------------- the automaton itself


class Automaton:
    """
    table[state, byte] is the state after reading the byte, or -1 if no
    text of the pattern goes on that way. The automaton starts in state 0,
    and a text matches if it ends in a state of `accepting`.
    """

    def __init__(self, pattern):
        # First an automaton that may be in several states at once (Thompson's
        # construction): one small piece per node of the tree, joined by moves
        # that read nothing.
        moves, free = [], []  # per state: {byte: [states]}, and states reachable without reading

        def state():
            moves.append({})
            free.append([])
            return len(moves) - 1

        def build(tree, start):
            """Add the piece for a tree that begins in `start`. Returns the state it ends in."""
            kind, body = tree
            if kind == "bytes":
                end = state()
                for byte in body:
                    moves[start].setdefault(byte, []).append(end)
                return end
            if kind == "cat":
                for part in body:
                    start = build(part, start)
                return start
            if kind == "alt":
                end = state()
                for option in body:
                    first = state()
                    free[start].append(first)
                    free[build(option, first)].append(end)
                return end
            inner, end = state(), state()  # star: skip it, or go through it and come back
            free[start] += [inner, end]
            free[build(body, inner)] += [inner, end]
            return end

        first = state()
        last = build(parse(pattern), first)

        def closure(states):
            """Everything reachable from a set of states without reading a byte."""
            seen, todo = set(states), list(states)
            while todo:
                for nxt in free[todo.pop()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        todo.append(nxt)
            return frozenset(seen)

        # Then the deterministic one: each of its states is a set of states of the first.
        sets = {closure([first]): 0}
        todo, rows = [closure([first])], []
        while todo:
            current = todo.pop(0)
            row = np.full(256, -1, dtype=np.int32)
            reach = {}
            for s in current:
                for byte, targets in moves[s].items():
                    reach.setdefault(byte, set()).update(targets)
            for byte, targets in reach.items():
                target = closure(targets)
                if target not in sets:
                    sets[target] = len(sets)
                    todo.append(target)
                row[byte] = sets[target]
            rows.append(row)
        self.table = np.stack(rows)
        self.accepting = np.array([last in s for s in sets])
        # How many bytes a text needs at least to reach a match from each state.
        self.distance = np.where(self.accepting, 0, len(rows) + 1)
        for _ in range(len(rows)):
            ahead = np.where(self.table >= 0, self.distance[self.table], len(rows) + 1).min(axis=1) + 1
            self.distance = np.minimum(self.distance, ahead)

    def run(self, data, state=0):
        """The state after reading bytes from `state`, or -1."""
        for byte in data:
            if state < 0:
                break
            state = self.table[state, byte]
        return int(state)

    def matches(self, text):
        state = self.run(text.encode("utf-8"))
        return state >= 0 and bool(self.accepting[state])
