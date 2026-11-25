import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer, LlamaConfig

from .config import ModelConfig

TOKENIZER = "hf-internal-testing/llama-tokenizer"
EOS_TOKEN_ID = 2

LAYER_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def load_training_weights(path: str | Path):
    path = Path(path)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    return load_file(str(path)), metadata


def convert_to_hf(weights_path: str | Path, out_dir: str | Path, tokenizer_id: str = TOKENIZER):
    arrays, metadata = load_training_weights(weights_path)
    cfg = ModelConfig.from_dict(metadata["model_config"])
    tensors: dict[str, np.ndarray] = {}

    for key, value in arrays.items():
        if not key.startswith("p:"):
            continue
        name = key[2:]
        if name.startswith("layers."):
            continue
        tensors["model." + name] = value

    for layer in range(cfg.num_layers):
        for suffix in LAYER_SUFFIXES:
            stacked_name = "p:layers." + suffix
            if stacked_name not in arrays:
                raise KeyError(f"missing {stacked_name}")
            tensors[f"model.layers.{layer}.{suffix}"] = arrays[stacked_name][layer]

    expected = {"model.embed_tokens.weight", "model.norm.weight"} | {
        f"model.layers.{layer}.{suffix}"
        for layer in range(cfg.num_layers)
        for suffix in LAYER_SUFFIXES
    }
    missing = expected - set(tensors)
    extra = set(tensors) - expected
    if missing or extra:
        raise KeyError(f"tensor mismatch; missing={sorted(missing)[:4]} extra={sorted(extra)[:4]}")

    config = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.seq_len,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        hidden_act="silu",
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=EOS_TOKEN_ID,
        architectures=["LlamaForCausalLM"],
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(tokenizer_id).save_pretrained(out)
    export_metadata = {
        "training_step": metadata.get("step"),
        "parameter_count": cfg.parameter_count(),
        "model_config": cfg.to_dict(),
    }
    (out / "parth-training-metadata.json").write_text(json.dumps(export_metadata, indent=2))
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Parth TPU checkpoint export to HF Llama format")
    parser.add_argument("weights", help="parth_weights_step_*.safetensors or training checkpoint")
    parser.add_argument("out", help="output HF model directory")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    return parser.parse_args()


def main():
    args = parse_args()
    config = convert_to_hf(args.weights, args.out, args.tokenizer)
    print(f"wrote {args.out}: {config.num_hidden_layers} layers, hidden={config.hidden_size}")


if __name__ == "__main__":
    main()
