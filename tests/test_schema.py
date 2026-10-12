import json
import random

from minllm import schema
from minllm.automaton import Automaton

CALL = {
    "type": "object",
    "properties": {
        "name": {"enum": ["add", "search"]},
        "arguments": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"}, "weights": {"type": "array", "items": {"type": "number"}}}},
    },
}


def dumps(value):
    return json.dumps(value, ensure_ascii=False)  # one space after colons and commas, as the chat template writes JSON


def test_accepts_what_the_schema_describes():
    automaton = Automaton(schema.regex(CALL))
    values = [
        {"name": "search", "arguments": {"query": "a \"quoted\" word\nand a new line, a tab\t and é", "limit": 10, "exact": True, "weights": [1, -2.5, 0.125]}},
        {"name": "add", "arguments": {"query": "", "limit": -3, "exact": False, "weights": []}},
        {"name": "add", "arguments": {"query": "\\u00e9 {\"not\": \"an object\"}", "limit": 0, "exact": True, "weights": [7]}},
    ]
    for value in values:
        assert automaton.matches(dumps(value)), dumps(value)
        assert automaton.matches(json.dumps(value))  # non-ASCII written as \uXXXX fits as well


def test_rejects_what_it_does_not():
    automaton = Automaton(schema.regex(CALL))
    good = {"name": "add", "arguments": {"query": "x", "limit": 1, "exact": True, "weights": [1]}}
    assert automaton.matches(dumps(good))
    wrong = [
        dumps(good | {"name": "delete"}),  # not one of the names
        dumps({"arguments": good["arguments"], "name": "add"}),  # the keys in another order
        dumps(good | {"arguments": good["arguments"] | {"limit": 1.5}}),  # a fraction where an integer belongs
        dumps(good | {"arguments": good["arguments"] | {"limit": "1"}}),
        dumps(good | {"extra": 1}),
        dumps(good)[:-1],  # not closed
        json.dumps(good, separators=(",", ":")),  # without the spaces
        dumps(good).replace('"x"', '"a\nb"'),  # a raw line break inside a string
        dumps(good).replace(": 1,", ": 01,"),  # a number with a leading zero
    ]
    broken = dumps(good).encode().replace(b'"x"', b'"\xff"')  # a byte that starts no character
    assert automaton.run(broken) == -1 or not automaton.accepting[automaton.run(broken)]
    for text in wrong:
        assert not automaton.matches(text), text


def test_everything_that_matches_is_json_of_the_schema():
    # Walk the automaton at random to a match, and let Python's parser judge the result.
    form = {"type": "object", "properties": {"reasoning": {"type": "string"}, "answer": {"type": "number"}, "sure": {"type": "boolean"}}}
    automaton = Automaton(schema.regex(form))
    random.seed(0)
    for _ in range(300):
        state, out = 0, bytearray()
        while not (automaton.accepting[state] and len(out) > 4):
            options = [b for b in range(256) if automaton.table[state, b] >= 0]
            # after a while, head for the end by the shortest way
            if len(out) > 60:
                options = [b for b in options if automaton.distance[automaton.table[state, b]] < automaton.distance[state]]
            byte = random.choice(options)
            out.append(byte)
            state = automaton.table[state, byte]
        value = json.loads(out.decode("utf-8", "replace"))
        assert list(value) == ["reasoning", "answer", "sure"]
        assert isinstance(value["reasoning"], str) and isinstance(value["answer"], (int, float)) and isinstance(value["sure"], bool)


def test_strings_are_well_formed_utf8_and_nothing_else():
    # every character of Unicode that a string can hold unescaped, at the edges of the ranges of the table
    automaton = Automaton(schema.STRING)
    for code in [0x20, 0x7E, 0x7F, 0x80, 0x7FF, 0x800, 0xFFF, 0x1000, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x3FFFF, 0x40000, 0x10FFFF]:
        assert automaton.matches(json.dumps(chr(code), ensure_ascii=False)), hex(code)
    # and no byte sequence that Python does not decode
    random.seed(1)
    for _ in range(3000):
        body = bytes(random.choice([random.randrange(0x80, 0x100), random.randrange(0x20, 0x80)]) for _ in range(random.randint(1, 4)))
        state = automaton.run(b'"' + body + b'"')
        accepted = state >= 0 and bool(automaton.accepting[state])
        try:
            json.loads(b'"' + body + b'"')
            valid = True
        except ValueError:
            valid = False
        assert accepted == valid, body
