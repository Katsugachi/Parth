import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig


def parameter_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    d = cfg.hidden_size
    kv_width = cfg.num_kv_heads * cfg.head_dim
    return {
        "embed_tokens.weight": (cfg.vocab_size, d),
        "norm.weight": (d,),
        "layers.input_layernorm.weight": (cfg.num_layers, d),
        "layers.post_attention_layernorm.weight": (cfg.num_layers, d),
        "layers.self_attn.q_proj.weight": (cfg.num_layers, d, d),
        "layers.self_attn.k_proj.weight": (cfg.num_layers, kv_width, d),
        "layers.self_attn.v_proj.weight": (cfg.num_layers, kv_width, d),
        "layers.self_attn.o_proj.weight": (cfg.num_layers, d, d),
        "layers.mlp.gate_proj.weight": (cfg.num_layers, cfg.intermediate_size, d),
        "layers.mlp.up_proj.weight": (cfg.num_layers, cfg.intermediate_size, d),
        "layers.mlp.down_proj.weight": (cfg.num_layers, d, cfg.intermediate_size),
    }


def initialize_parameters(cfg: ModelConfig, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    output_std = 0.02 / math.sqrt(2 * cfg.num_layers)
    for name, shape in parameter_shapes(cfg).items():
        if name.endswith("layernorm.weight") or name == "norm.weight":
            value = np.ones(shape, dtype=np.float32)
        else:
            std = output_std if name.endswith(("o_proj.weight", "down_proj.weight")) else 0.02
            value = rng.standard_normal(shape, dtype=np.float32) * std
        params[name] = value.astype(np.float32)
    return params


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    xf = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return xf * inv * weight.astype(jnp.float32)


def matmul(x: jax.Array, weight: jax.Array, dtype: jnp.dtype, out_dtype: jnp.dtype | None = None) -> jax.Array:
    return jnp.einsum(
        "...i,oi->...o",
        x.astype(dtype),
        weight.astype(dtype),
        preferred_element_type=out_dtype or dtype,
    )


def rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[jax.Array, jax.Array]:
    frequencies = theta ** (-jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    angles = positions * frequencies[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    first = x[..., :half].astype(jnp.float32)
    second = x[..., half:].astype(jnp.float32)
    return jnp.concatenate([first * cos - second * sin, first * sin + second * cos], axis=-1)


def transformer_block(
    layer: dict[str, jax.Array],
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    cfg: ModelConfig,
    dtype: jnp.dtype,
) -> jax.Array:
    batch, seq_len, _ = x.shape
    head_dim = cfg.head_dim

    h = rms_norm(x, layer["input_layernorm.weight"], cfg.rms_norm_eps)
    q = matmul(h, layer["self_attn.q_proj.weight"], dtype)
    k = matmul(h, layer["self_attn.k_proj.weight"], dtype)
    v = matmul(h, layer["self_attn.v_proj.weight"], dtype)

    q = q.reshape(batch, seq_len, cfg.num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, cfg.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, cfg.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    q = apply_rope(q, cos, sin).astype(dtype)
    k = apply_rope(k, cos, sin).astype(dtype)
    if cfg.num_kv_heads != cfg.num_heads:
        repeats = cfg.num_heads // cfg.num_kv_heads
        k = jnp.repeat(k, repeats, axis=1)
        v = jnp.repeat(v, repeats, axis=1)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32)
    scores = scores / math.sqrt(head_dim)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    scores = jnp.where(causal[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1).astype(dtype)
    attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v, preferred_element_type=dtype)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, cfg.hidden_size)
    x = x + matmul(attended, layer["self_attn.o_proj.weight"], dtype, jnp.float32)

    h = rms_norm(x, layer["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    gate = matmul(h, layer["mlp.gate_proj.weight"], dtype)
    up = matmul(h, layer["mlp.up_proj.weight"], dtype)
    hidden = jax.nn.silu(gate) * up
    return x + matmul(hidden, layer["mlp.down_proj.weight"], dtype, jnp.float32)


def forward(
    params: dict[str, jax.Array],
    token_ids: jax.Array,
    cfg: ModelConfig,
    dtype: jnp.dtype,
    remat: bool = True,
) -> jax.Array:
    cos, sin = rope_tables(token_ids.shape[1], cfg.head_dim, cfg.rope_theta)
    x = params["embed_tokens.weight"][token_ids].astype(jnp.float32)
    layers = {
        key[len("layers."):]: value
        for key, value in params.items()
        if key.startswith("layers.")
    }

    def run_layer(carry, layer):
        return transformer_block(layer, carry, cos, sin, cfg, dtype), None

    if remat:
        run_layer = jax.checkpoint(
            run_layer,
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        )
    x, _ = jax.lax.scan(run_layer, x, layers)
    x = rms_norm(x, params["norm.weight"], cfg.rms_norm_eps)
    return matmul(x, params["embed_tokens.weight"], dtype, jnp.float32)


def causal_lm_loss(
    params: dict[str, jax.Array],
    sequences: jax.Array,
    cfg: ModelConfig,
    dtype: jnp.dtype,
    remat: bool = True,
    loss_mask: jax.Array | None = None,
) -> jax.Array:
    logits = forward(params, sequences[:, :-1], cfg, dtype, remat)
    targets = sequences[:, 1:]
    logZ = jax.nn.logsumexp(logits, axis=-1)
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    nll = logZ - target_logits
    if loss_mask is None:
        return jnp.mean(nll)
    mask = loss_mask[:, 1:].astype(jnp.float32)
    return jnp.sum(nll * mask) / jnp.maximum(jnp.sum(mask), 1.0)
