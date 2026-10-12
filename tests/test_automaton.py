import random
import re

from minllm.automaton import Automaton


def test_matches_what_python_matches():
    # The dialect is a part of Python's: on it the two must agree.
    random.seed(0)
    patterns = [r"ab*c", r"(a|bc)+d?", r"[0-9]+(\.[0-9]+)?", r'"([^"\\]|\\["\\n])*"', r"-?(0|[1-9][0-9]*)", r"(true|false|null)", r"\[(x(, x)*)?\]", r"a\+b\*\(c\)"]
    alphabet = 'abcdx0159.-+*()[], "\\ntruefalsenull'
    for pattern in patterns:
        automaton, compiled = Automaton(pattern), re.compile(pattern)
        for _ in range(600):
            text = "".join(random.choice(alphabet) for _ in range(random.randint(0, 9)))
            assert automaton.matches(text) == bool(compiled.fullmatch(text)), (pattern, text)


def test_a_dead_end_is_known_at_once():
    automaton = Automaton(r"(yes|no)!")
    assert automaton.run(b"ye") >= 0  # can still become "yes!"
    assert automaton.run(b"yo") == -1  # can not become anything
    assert automaton.matches("no!") and not automaton.matches("no")


def test_characters_outside_ascii_are_their_bytes():
    automaton = Automaton("(é|日本)+")
    assert automaton.matches("é日本é")
    assert automaton.run("日".encode()[:2]) >= 0  # halfway through a character is still on a path
    assert not automaton.matches("e")


def test_distance_is_the_shortest_way_to_a_match():
    automaton = Automaton(r'\{"a": [0-9]+\}')
    assert automaton.distance[0] == len('{"a": 0}')
    assert automaton.distance[automaton.run(b'{"a": 12')] == 1  # one closing brace away
    assert automaton.distance[automaton.run(b'{"a": 12}')] == 0
