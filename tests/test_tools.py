import json
import os

import pytest
import torch

from minllm import chat, pretrained, tools

QWEN = "data/qwen2.5-0.5b-instruct"
needs_tokenizer = pytest.mark.skipif(not os.path.exists(f"{QWEN}/vocab.json"), reason="run scripts/download_qwen.py")


def test_calculate():
    assert tools.calculate("48/2") == "24"
    assert tools.calculate("16 - 3 - 4") == "9"
    assert tools.calculate("(48 + 24) / 5") == "14.4"
    assert tools.calculate("1,200 * 0.15") == "180"
    assert tools.calculate("2**10") == "1024"
    assert tools.calculate("1/3") == "0.333333"
    assert tools.calculate("1/0").startswith("error")
    assert tools.calculate("__import__('os')").startswith("error")  # nothing but arithmetic is evaluated
    assert tools.calculate("9**9**9").startswith("error")


@needs_tokenizer
def test_system_prompt_is_the_templates():
    # The published chat template, as Qwen wrote it, must give the same text.
    jinja = json.load(open(f"{QWEN}/tokenizer_config.json"))["chat_template"]
    expected = jinja.split("{{- '<|im_start|>system\\n' }}")[1]
    assert "# Tools\\n\\nYou may call one or more functions" in expected
    text = tools.system_prompt(pretrained.SYSTEM_PROMPT, [tools.CALCULATE])
    assert text.startswith(pretrained.SYSTEM_PROMPT + "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.")
    assert '\n<tools>\n{"type": "function", "function": {"name": "calculate"' in text
    assert text.endswith('<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')


def test_gsm8k_solution_becomes_calls():
    examples = tools.gsm8k_conversations("train", "system")
    first = examples[0]
    roles = [m["role"] for m in first["messages"]]
    assert roles[:2] == ["system", "user"] and roles[-1] == "assistant"
    assert roles.count("assistant") == first["marks"] + 1
    # Natalia sold 48/2 = <<48/2=24>>24 clips in May. ...
    assert first["messages"][2]["content"] == 'Natalia sold 48/2 =\n<tool_call>\n{"name": "calculate", "arguments": {"expression": "48/2"}}\n</tool_call>'
    assert first["messages"][3]["content"] == "<tool_response>\n24\n</tool_response>"
    assert first["messages"][4]["content"].startswith("24 clips in May.")
    assert first["messages"][-1]["content"].endswith("#### 72")
    # every mark of every solution is a call that the calculator can do
    results = [m["content"] for e in examples for m in e["messages"] if m["role"] == "user" and m["content"].startswith("<tool_response>")]
    assert sum(r.startswith("<tool_response>\nerror") for r in results) < 0.01 * len(results)


@needs_tokenizer
def test_calls_are_control_tokens_in_the_assistants_turn_only():
    tok = pretrained.load_tokenizer(QWEN)
    call = tools.call_text("calculate", {"expression": "1+1"})
    messages = [{"role": "user", "content": call}, {"role": "assistant", "content": "x\n" + call}]
    ids, labels = tools.training_example(tok, messages)
    start, end = tok.special_tokens["<tool_call>"], tok.special_tokens["</tool_call>"]
    assert ids.count(start) == 1 and ids.count(end) == 1  # the user's copy stays characters
    assert all(label == token for label, token in zip(labels, ids) if label != -100)
    learned = [token for label, token in zip(labels, ids) if label != -100]
    assert tok.decode(learned) == "x\n" + call + chat.IM_END
    # without calls, the same as stage 6's encoding
    plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert tools.training_example(tok, plain) == chat.training_example(tok, plain)
    assert tools.encode(tok, plain)[0] == chat.encode(tok, plain)


@needs_tokenizer
def test_system_markers_tokenize_the_tool_instructions_as_qwen_does():
    tok = pretrained.load_tokenizer(QWEN)
    system = {"role": "system", "content": tools.system_prompt(pretrained.SYSTEM_PROMPT, [tools.CALCULATE])}
    question = {"role": "user", "content": "Say <tool_call> back."}
    start, end = tok.special_tokens["<tool_call>"], tok.special_tokens["</tool_call>"]
    spelled = tools.encode(tok, [system, question])[0]
    marked = tools.encode(tok, [system, question], system_markers=True)[0]
    assert start not in spelled and end not in spelled
    assert marked.count(start) == 2 and marked.count(end) == 2  # the instructions' two pairs; the user's copy stays characters
    assert tok.decode(marked) == tok.decode(spelled)  # the same text, in fewer tokens
    assert len(marked) < len(spelled)


def test_calls_are_read_from_a_reply():
    reply = "First\n" + tools.call_text("calculate", {"expression": "2*3"}) + "\n<tool_call>\nnot json\n</tool_call>"
    assert tools.calls(reply) == [("calculate", {"expression": "2*3"}), (None, "\nnot json\n")]
    assert tools.calls("no calls here") == []


def test_written_calculations():
    found = tools.written_calculations("She has 16 - 3 - 4 = 9 eggs, sells them for 9 * 2 = $18, and 18 + 5 = 22 is wrong.")
    assert found == [("16 - 3 - 4", "9", True), ("9 * 2", "18", True), ("18 + 5", "22", False)]
    # a result that goes on is not a result
    assert tools.written_calculations("so 1 - 2/3 = 1/3 of it") == []
    assert tools.written_calculations("then 3x + 9 - 1 = 87, so") == []  # part of an equation
    assert tools.written_calculations("he walks 80/7 = 11.43 km") == [("80/7", "11.43", True)]  # rounded
    assert tools.written_calculations("he walks 80/7 = 11.5 km") == [("80/7", "11.5", False)]


def scripted(turns):
    """A stand-in for inference.stream that writes the given turns, one per call, in every row, each ended by the model."""
    turns = iter(turns)

    def stream(model, prompts, max_new_tokens, stop=(), **sampling):
        for token in next(turns):
            yield torch.tensor([token] * len(prompts))
        yield torch.tensor([-1] * len(prompts))  # the stop token, as stream reports it

    return stream


@needs_tokenizer
def test_rollout_keeps_the_sampled_ids(monkeypatch):
    tok = pretrained.load_tokenizer(QWEN)
    conversation = [{"role": "system", "content": tools.system_prompt("system", [tools.CALCULATE])}, {"role": "user", "content": "48/2?"}]
    first = "Natalia sold 48/2 =\n" + tools.call_text("calculate", {"expression": "48/2"})
    turns = [tools._with_calls(tok, first), tok.encode_ordinary("24 clips.\n#### 24")]
    monkeypatch.setattr(tools.inference, "stream", scripted(turns))
    prompts, added, written, messages, counts = tools.rollout(None, tok, [conversation])
    assert counts == [{"calls": 1, "bad_calls": 0, "turns": 2}]
    assert [m["role"] for m in messages[0][2:]] == ["assistant", "user", "assistant"]
    assert messages[0][2]["content"] == first and messages[0][3]["content"] == "<tool_response>\n24\n</tool_response>"
    # The ids are the conversation as encode writes it, and the model's are the ones fine-tuning learns.
    ids, labels = tools.encode(tok, messages[0], add_generation_prompt=False)
    newline = tok.encode_ordinary("\n")
    assert prompts[0] + added[0] == ids[: -len(newline)]
    learned = [label != -100 for label in labels[len(prompts[0]) : len(ids) - len(newline)]]
    assert written[0] == [int(x) for x in learned]
    assert tools.answer_text(messages[0], 2) == "Natalia sold 48/2 =\n24 clips.\n#### 24"
