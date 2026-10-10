"""
The chat format.

A chat model is the same next-token predictor as before. What makes it a chat
model is that it was trained on conversations written in one fixed format, so
that after the right prefix, the likeliest continuation is a reply. The format
here is ChatML, which Qwen uses:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is 2 + 2?<|im_end|>
    <|im_start|>assistant
    4<|im_end|>

<|im_start|> and <|im_end|> are special tokens. The model ends its turn by
writing <|im_end|>, which is where generation stops.
"""

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def render(messages, add_generation_prompt=True):
    """
    Write a conversation, a list of {"role": ..., "content": ...}, as text.
    With add_generation_prompt the text ends where the assistant's next reply
    begins, which is the prompt to generate from.
    """
    text = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text


def encode(tok, messages, add_generation_prompt=True):
    """
    The token ids of a conversation. Only the markers this function writes
    become control tokens. A message that contains the characters <|im_end|>
    is encoded as those characters, so the content of a message cannot end
    its own turn or open another.
    """
    start, end = tok.special_tokens[IM_START], tok.special_tokens[IM_END]
    ids = []
    for m in messages:
        ids += [start] + tok.encode_ordinary(f"{m['role']}\n{m['content']}") + [end] + tok.encode_ordinary("\n")
    if add_generation_prompt:
        ids += [start] + tok.encode_ordinary("assistant\n")
    return ids
