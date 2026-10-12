"""
A JSON schema as a regular expression.

A schema says what a JSON value has to look like: an object with these
keys, a string here, a number there. For schemas without recursion, the
texts that fit are a regular language, so one expression describes them
all, and minllm/automaton.py turns it into a table that constrained
decoding can follow byte by byte.

Only what a tool call needs is covered: objects with all their keys in the
order given, strings, numbers, integers, booleans, null, a choice among
constants, and arrays. The text is laid out the way Qwen2.5's chat template
writes JSON, with one space after every colon and comma, because that is
the JSON the model saw in training. See docs/04a-structured-output.md.
"""

import json

HEX = "[0-9a-fA-F]"
TAIL = r"[\x80-\xbf]"  # a byte that continues a character
# The characters of more than one byte, as Unicode allows them to be written
# (the table of well-formed UTF-8): no character written longer than it needs,
# none of the halves of UTF-16's surrogate pairs, and nothing above U+10FFFF.
MULTIBYTE = "|".join([
    rf"[\xc2-\xdf]{TAIL}",
    rf"\xe0[\xa0-\xbf]{TAIL}",
    rf"[\xe1-\xec\xee-\xef]{TAIL}{TAIL}",
    rf"\xed[\x80-\x9f]{TAIL}",
    rf"\xf0[\x90-\xbf]{TAIL}{TAIL}",
    rf"[\xf1-\xf3]{TAIL}{TAIL}{TAIL}",
    rf"\xf4[\x80-\x8f]{TAIL}{TAIL}",
])
# A character of a string: a printable ASCII character but a quote or a
# backslash, a character of several bytes, or an escape. The automaton reads
# bytes, and without the second part it would accept bytes that are no text.
CHARACTER = rf'([\x20-\x21\x23-\x5b\x5d-\x7f]|{MULTIBYTE}|\\["\\/bfnrt]|\\u{HEX}{HEX}{HEX}{HEX})'
STRING = f'"{CHARACTER}*"'
# Any text of printable characters and line breaks, which a reply can reason in before its answer.
TEXT = rf"([\x20-\x7e\n]|{MULTIBYTE})*"
INTEGER = r"-?(0|[1-9][0-9]*)"
NUMBER = INTEGER + r"(\.[0-9]+)?"


def literal(text):
    """A text as an expression that matches only itself."""
    return "".join(f"\\{c}" if c in r"\()[]|*+?.^-" else c for c in text)


def regex(schema):
    """The expression for all JSON texts that fit a schema."""
    if "enum" in schema:
        return "(" + "|".join(literal(json.dumps(value)) for value in schema["enum"]) + ")"
    if "const" in schema:
        return literal(json.dumps(schema["const"]))
    kind = schema["type"]
    if kind == "string":
        return STRING
    if kind == "integer":
        return INTEGER
    if kind == "number":
        return NUMBER
    if kind == "boolean":
        return "(true|false)"
    if kind == "null":
        return "null"
    if kind == "array":
        item = regex(schema["items"])
        return rf"\[({item}(, {item})*)?\]"
    if kind == "object":
        members = [f"{literal(json.dumps(key))}: {regex(value)}" for key, value in schema.get("properties", {}).items()]
        return r"\{" + ", ".join(members) + r"\}"
    raise ValueError(f"a schema of type {kind!r} is not handled")
