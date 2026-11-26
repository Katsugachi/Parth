import numpy as np

from parth.config import ModelConfig
from parth.data import WeightedStream
from parth.model import causal_lm_loss, forward, initialize_parameters
from parth.qwenputer_curriculum import arithmetic_program, TinyCPU, cpu_trace_document
from parth.ttc import extract_answer, self_consistent_answer


def test_tiny_forward_loss():
    import jax.numpy as jnp

    cfg = ModelConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64, seq_len=16)
    params = {k: jnp.asarray(v) for k, v in initialize_parameters(cfg).items()}
    ids = jnp.asarray(np.random.randint(0, 64, (2, 17)))
    logits = forward(params, ids[:, :-1], cfg, jnp.float32)
    assert logits.shape == (2, 16, 64)
    assert np.isfinite(float(causal_lm_loss(params, ids, cfg, jnp.float32)))


def test_cpu_traces_execute_correctly():
    for index in range(20):
        instructions, expected = arithmetic_program(index)
        assert TinyCPU().run(instructions) == expected
        assert f"R0 = {expected}" in cpu_trace_document(index)


def test_ttc_voting():
    candidates = ["Final answer: 42", "Therefore final answer is 42", "Final answer: 7"]
    answer, extracted, counts = self_consistent_answer(candidates)
    assert answer == "42"
    assert extract_answer("x Final answer: 123") == "123"


def test_weighted_stream_includes_documents():
    stream = WeightedStream(shuffle_buffer=1)
    assert len(stream.next_document()) > 20


def test_default_model_is_under_150m():
    from parth.config import DEFAULT_MODEL

    assert DEFAULT_MODEL.parameter_count() == 146_428_800
    assert DEFAULT_MODEL.parameter_count() < 150_000_000
    assert DEFAULT_MODEL.head_dim == 40


def test_hf_export_parity_with_head_dim_40(tmp_path):
    import json

    import jax.numpy as jnp
    import torch
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM

    from parth.export_hf import convert_to_hf

    cfg = ModelConfig(vocab_size=64, hidden_size=80, num_layers=2, num_heads=2, num_kv_heads=1, intermediate_size=128, seq_len=16)
    params = initialize_parameters(cfg)
    weights = tmp_path / "w.safetensors"
    save_file({"p:" + k: v for k, v in params.items()}, str(weights))
    (tmp_path / "w.json").write_text(json.dumps({"step": 0, "model_config": cfg.to_dict()}))
    convert_to_hf(weights, tmp_path / "hf")
    hf = AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    ids = np.random.RandomState(0).randint(0, 64, (2, 15))
    jax_logits = np.asarray(forward({k: jnp.asarray(v) for k, v in params.items()}, jnp.asarray(ids), cfg, jnp.float32, remat=False))
    with torch.no_grad():
        hf_logits = hf(input_ids=torch.tensor(ids)).logits.numpy()
    assert np.max(np.abs(jax_logits - hf_logits)) < 2e-6
