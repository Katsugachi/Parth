import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from .config import DEFAULT_MODEL, DEFAULT_TRAIN, TrainConfig
from .data import PRETRAIN_STREAMS, WeightedStream, download_training_files, json_safe
from .train_tpu import OUTPUT_DIR, TOKENIZER, cosine_lr, prune_checkpoints


def hf_config(model_cfg):
    return LlamaConfig(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_hidden_layers=model_cfg.num_layers,
        num_attention_heads=model_cfg.num_heads,
        num_key_value_heads=model_cfg.num_kv_heads,
        max_position_embeddings=model_cfg.seq_len,
        rms_norm_eps=model_cfg.rms_norm_eps,
        rope_theta=model_cfg.rope_theta,
        hidden_act="silu",
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="sdpa",
    )


class TorchTokenLoader:
    def __init__(self, tokenizer, model_cfg, train_cfg, world_size, rank):
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.world_size = world_size
        self.rank = rank
        self.stream = WeightedStream(PRETRAIN_STREAMS, shuffle_buffer=train_cfg.shuffle_buffer, seed=0)
        self.buffer = np.empty(0, dtype=np.int32)

    def next_batch(self):
        cfg = self.train_cfg
        required = self.world_size * cfg.grad_accum * cfg.micro_batch * (self.model_cfg.seq_len + 1)
        while self.buffer.size < required:
            docs = [self.stream.next_document() for _ in range(32)]
            encoded = self.tokenizer(docs, add_special_tokens=False)["input_ids"]
            pieces = [np.asarray(ids + [self.tokenizer.eos_token_id], dtype=np.int32) for ids in encoded]
            self.buffer = np.concatenate([self.buffer] + pieces)
        batch = self.buffer[:required].reshape(
            self.world_size, cfg.grad_accum, cfg.micro_batch, self.model_cfg.seq_len + 1
        )[self.rank].copy()
        self.buffer = self.buffer[required:]
        return torch.from_numpy(batch.astype(np.int64)), json_safe(self.stream.state_dict())


def save_torch_checkpoint(model, optimizer, scaler, step, out_dir, model_cfg, history, stream_state):
    base = os.path.join(out_dir, f"parth_step_{step:08d}")
    os.makedirs(out_dir, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    state = {"p:" + k: v.detach().cpu().contiguous().clone() for k, v in raw.state_dict().items()}
    save_file(state, base + ".model.safetensors", metadata={"format": "pt"})
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, base + ".optimizer.pt")
    with open(base + ".json", "w") as handle:
        json.dump({
            "step": step,
            "model_config": model_cfg.to_dict(),
            "history": history,
            "stream_state": stream_state,
            "format": "torch",
        }, handle)
    print(f"[checkpoint] step {step} -> {base}.model.safetensors", flush=True)


def upload_checkpoint(repo_id, base, token):
    if not repo_id or not token:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
        for suffix in (".model.safetensors", ".json"):
            path = base + suffix
            if os.path.exists(path):
                api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path), repo_id=repo_id, repo_type="model")
        print(f"[hf] uploaded checkpoint metadata/weights for {os.path.basename(base)}", flush=True)
    except Exception as exc:
        print(f"[hf] upload failed: {type(exc).__name__}: {exc}", flush=True)


def train(args):
    model_cfg = DEFAULT_MODEL
    train_cfg = TrainConfig(
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        total_tokens=args.total_tokens,
        max_hours=args.max_hours,
        save_every=args.save_every,
        print_every=args.print_every,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = hf_config(model_cfg)
    model = LlamaForCausalLM(config).to(device)
    if args.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if cuda else None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.adam_eps,
        weight_decay=train_cfg.weight_decay,
        fused=cuda,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and args.dtype == "float16")
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    if args.download_data:
        cache_dir = os.environ.get("PARTH_DATA_CACHE", "/tmp/parth_data_cache")
        os.environ["PARTH_DATA_CACHE"] = cache_dir
        if rank == 0:
            download_training_files(cache_dir, workers=args.download_workers)
        if world_size > 1:
            dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.model_max_length = 10**9
    loader = TorchTokenLoader(tokenizer, model_cfg, train_cfg, world_size, rank)
    tokens_per_step = train_cfg.tokens_per_step(model_cfg, world_size)
    total_steps = math.ceil(train_cfg.total_tokens / tokens_per_step)
    if rank == 0:
        print(f"torch {torch.__version__}: {world_size} x {device}, {model_cfg.parameter_count()/1e6:.1f}M params")
        print(f"{total_steps} steps planned, {tokens_per_step:,} tokens/step", flush=True)

    history = {"train": [], "eval": []}
    started = time.time()
    timer = started
    loss_sum = 0.0
    loss_count = 0
    stream_state = None
    for step in range(1, total_steps + 1):
        timed_out = torch.tensor([int(time.time() - started > train_cfg.max_hours * 3600)], device=device)
        if world_size > 1:
            dist.all_reduce(timed_out, op=dist.ReduceOp.MAX)
        if int(timed_out.item()):
            break
        batch, stream_state = loader.next_batch()
        batch = batch.to(device, non_blocking=True)
        lr = cosine_lr(step - 1, train_cfg, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for micro in range(train_cfg.grad_accum):
            tokens = batch[micro]
            inputs = tokens[:, :-1]
            sync = micro == train_cfg.grad_accum - 1
            context = model.no_sync() if world_size > 1 and not sync else torch.enable_grad()
            with context:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cuda and amp_dtype != torch.float32):
                    output = model(input_ids=inputs, labels=inputs)
                    loss = output.loss / train_cfg.grad_accum
                scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach().float().cpu()) * train_cfg.grad_accum
        loss_sum += value
        loss_count += 1
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if rank == 0 and step % train_cfg.print_every == 0:
            mean = loss_sum / max(loss_count, 1)
            elapsed = max(time.time() - timer, 1e-9)
            rate = tokens_per_step * loss_count / elapsed
            eta = (total_steps - step) * elapsed / max(loss_count, 1) / 3600
            print(f"step {step}/{total_steps} loss={mean:.4f} gnorm={float(grad_norm):.2f} lr={lr:.2e} {rate:,.0f} tok/s eta={eta:.1f}h", flush=True)
            history["train"].append([step, mean])
            loss_sum = 0.0
            loss_count = 0
            timer = time.time()
        if rank == 0 and step % train_cfg.save_every == 0:
            base = os.path.join(OUTPUT_DIR, f"parth_step_{step:08d}")
            save_torch_checkpoint(model, optimizer, scaler, step, OUTPUT_DIR, model_cfg, history, stream_state)
            upload_checkpoint(args.hf_checkpoint_repo, base, os.environ.get("HF_TOKEN"))
            prune_checkpoints(OUTPUT_DIR)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        step_done = step if "step" in locals() else 0
        base = os.path.join(OUTPUT_DIR, f"parth_step_{step_done:08d}")
        save_torch_checkpoint(model, optimizer, scaler, step_done, OUTPUT_DIR, model_cfg, history, stream_state)
        upload_checkpoint(args.hf_checkpoint_repo, base, os.environ.get("HF_TOKEN"))
        raw = model.module if hasattr(model, "module") else model
        final_dir = Path(OUTPUT_DIR) / "hf-final"
        raw.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
        print(f"[export] wrote {final_dir}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch", type=int, default=DEFAULT_TRAIN.micro_batch)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_TRAIN.grad_accum)
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TRAIN.total_tokens)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_TRAIN.max_hours)
    parser.add_argument("--save-every", type=int, default=DEFAULT_TRAIN.save_every)
    parser.add_argument("--print-every", type=int, default=DEFAULT_TRAIN.print_every)
    parser.add_argument("--download-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hf-checkpoint-repo", default=os.environ.get("HF_CHECKPOINT_REPO", ""))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
