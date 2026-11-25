from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 2688
    seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def parameter_count(self) -> int:
        d = self.hidden_size
        kv = self.num_kv_heads * self.head_dim
        per_layer = (
            2 * d
            + d * d
            + 2 * kv * d
            + d * d
            + 3 * self.intermediate_size * d
        )
        return self.vocab_size * d + d + self.num_layers * per_layer

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**data)


@dataclass(frozen=True)
class TrainConfig:
    micro_batch: int = 4
    grad_accum: int = 16
    total_tokens: int = 8_000_000_000
    learning_rate: float = 4e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_sequences: int = 128
    eval_documents: int = 1200
    shuffle_buffer: int = 5000
    save_every: int = 250
    print_every: int = 50
    stats_every: int = 250
    eval_every: int = 500
    max_hours: float = 8.25
    sample_length: int = 256

    def tokens_per_step(self, model: ModelConfig, devices: int) -> int:
        return self.micro_batch * self.grad_accum * model.seq_len * devices


DEFAULT_MODEL = ModelConfig()
DEFAULT_TRAIN = TrainConfig()
