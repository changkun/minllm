import json
import random

import numpy as np
import torch
from minllm.model import Config, Transformer
from minllm.tokenizer import Tokenizer

from minllm import constrain, schema
from minllm.automaton import Automaton

FORM = {"type": "object", "properties": {"reasoning": {"type": "string"}, "answer": {"type": "number"}}}
TEXT = '{"reasoning": "two and two make four, and one more", "answer": 5} {"reasoning": "nothing", "answer": -12.5} true false null [1, 2]'


def tokenizer():
    return Tokenizer.train(TEXT * 30, 420, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])


def test_tokens_to_end_is_the_fewest_tokens():
    tok = tokenizer()
    automaton = Automaton(schema.regex(FORM))
    index = constrain.Index(automaton, tok)
    # never more than the bytes, since every byte is a token
    assert (index.tokens_to_end <= automaton.distance).all()
    shortest = '{"reasoning": "", "answer": 0}'
    assert automaton.matches(shortest) and index.tokens_to_end[0] <= len(tok.encode_ordinary(shortest))
    assert index.tokens_to_end[automaton.run(shortest.encode())] == 0


def test_a_token_is_allowed_if_all_its_bytes_can_be_read():
    tok = tokenizer()
    automaton = Automaton(schema.regex(FORM))
    index = constrain.Index(automaton, tok)
    random.seed(0)
    prefixes = ['{"reasoning": "two', '{"', '{"reasoning": "x", "answer": 1', '{"reasoning": "x", "answer": -', ""]
    for prefix in prefixes:
        state = automaton.run(prefix.encode())
        after = index.after(state)
        for token in random.sample(range(tok.vocab_size), 120) + list(tok.special_tokens.values()):
            expected = -1 if token in tok.special_tokens.values() else automaton.run(tok.vocab[token], state)
            assert after[token] == expected, (prefix, tok.vocab.get(token))
    assert (index.after(automaton.run(b'{"reasoning": "x", "answer": 1')) >= 0).sum() < (index.after(automaton.run(b'{"reasoning": "x')) >= 0).sum()


def test_replies_fit_the_schema_whatever_the_model():
    # A model that was never trained picks tokens almost at random. Its replies fit all the same.
    tok = tokenizer()
    torch.manual_seed(0)
    # an output layer wider than the vocabulary, as published models have
    model = Transformer(Config(vocab_size=tok.vocab_size + 18, max_seq_len=160, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_dim=64)).eval()
    examples = [{"messages": [{"role": "user", "content": text}]} for text in ["two and two", "nothing", "one more, and one more"]]
    index = constrain.Index(Automaton(schema.regex(FORM)), tok, model.cfg.vocab_size)
    shortest = index.tokens_to_end[0]
    for budget, temperature in [(130, 1.0), (60, 0.0), (24, 1.0), (shortest + 1, 1.0)]:  # far more tokens than the form has states, and just enough
        replies = constrain.replies(model, tok, examples, FORM, budget, temperature=temperature, index=index)
        for reply in replies:
            value = json.loads(reply)
            assert list(value) == ["reasoning", "answer"] and isinstance(value["reasoning"], str) and isinstance(value["answer"], (int, float))
            assert "\ufffd" not in reply  # every byte of it is part of a character


def test_a_row_that_runs_out_of_tokens_is_led_to_the_end():
    tok = tokenizer()
    automaton = Automaton(schema.regex(FORM))
    index = constrain.Index(automaton, tok)
    stop = tok.special_tokens["<|im_end|>"]
    budget = index.tokens_to_end[0] + 3
    allow = constrain.Constraint(index, 1, stop, budget=budget)
    rng = np.random.default_rng(0)
    previous, text = None, b""
    for _ in range(budget):
        allowed = allow(previous)[0].numpy()
        token = int(rng.choice(np.flatnonzero(allowed)))
        if token == stop:
            break
        text += tok.vocab[token]
        previous = torch.tensor([token])
    assert token == stop and automaton.matches(text.decode())
