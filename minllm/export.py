"""
Exporting a model to the layout that the rest of the ecosystem reads.

A directory in "Hugging Face format" is what stage 5 took apart: a
config.json, the weights in safetensors under the reference names, and the
tokenizer's files. Writing one makes a model of this repository loadable by
transformers, vLLM, MLX and llama.cpp's converter. See docs/10-stack.md.
"""

import json
import os
import shutil

import torch

from minllm import chat, lora, pretrained

# ChatML as a Jinja template, the form in which tokenizer_config.json carries a chat format.
CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def to_reference(name):
    """Translate a weight name of minllm/model.py to the reference name. The inverse of pretrained.to_minllm."""
    for ours, theirs in [("attn_norm", "input_layernorm"), ("mlp_norm", "post_attention_layernorm"), (".attn.", ".self_attn.")]:
        name = name.replace(ours, theirs)
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    return name if name.startswith("lm_head") else "model." + name


def config_json(cfg, tok):
    """
    The model's shape in the reference vocabulary. A model without biases is
    a Llama, one with biases on q, k and v is a Qwen2: the two reference
    classes differ in nothing else that matters here.
    """
    return {
        "architectures": ["Qwen2ForCausalLM" if cfg.qkv_bias else "LlamaForCausalLM"],
        "model_type": "qwen2" if cfg.qkv_bias else "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.mlp_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "max_position_embeddings": cfg.max_seq_len,
        "hidden_act": "silu",
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False,
        "bos_token_id": tok.special_tokens["<|endoftext|>"],
        "eos_token_id": tok.special_tokens[chat.IM_END],
        "pad_token_id": tok.special_tokens["<|endoftext|>"],
        "torch_dtype": "float32",
    }


def tokenizer_json(tok):
    """A tokenizer trained here, in the format of the Hugging Face tokenizers library."""
    spell = {byte: char for char, byte in pretrained.gpt2_alphabet().items()}
    text = lambda i: "".join(spell[b] for b in tok.vocab[i])
    special = set(tok.special_tokens.values())
    vocab = {text(i): i for i in tok.vocab if i not in special}
    added = [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True} for s, i in tok.special_tokens.items()]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    split = {"type": "Split", "pattern": {"Regex": tok.pattern}, "behavior": "Isolated", "invert": False}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [split, byte_level]},
        "post_processor": None,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [f"{text(a)} {text(b)}" for a, b in tok.merges],
        },
    }


def export(model, tok, meta, dir, source=pretrained.DIR):
    """Write model and tokenizer to dir in Hugging Face format. LoRA adapters are merged in first."""
    os.makedirs(dir, exist_ok=True)
    model = lora.merge(model)
    state = {to_reference(name): t.detach().float().cpu() for name, t in model.state_dict().items()}
    if model.cfg.tie_embeddings:
        del state["lm_head.weight"]  # stored once, like the published files do
    pretrained.write_safetensors(f"{dir}/model.safetensors", state)

    if meta["family"] == "qwen":
        # The published tokenizer and configuration are unchanged by fine-tuning.
        for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
            if os.path.exists(f"{source}/{name}"):
                shutil.copy(f"{source}/{name}", f"{dir}/{name}")
        with open(f"{dir}/config.json") as f:
            config = json.load(f) | {"torch_dtype": "float32"}
    else:
        config = config_json(model.cfg, tok)
        with open(f"{dir}/tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_json(tok), f, ensure_ascii=False)
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|endoftext|>",
            "eos_token": chat.IM_END,
            "pad_token": "<|endoftext|>",
            "chat_template": CHAT_TEMPLATE,
            "clean_up_tokenization_spaces": False,
            "model_max_length": model.cfg.max_seq_len,
        }
        with open(f"{dir}/tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=1)
    with open(f"{dir}/config.json", "w") as f:
        json.dump(config, f, indent=1)
