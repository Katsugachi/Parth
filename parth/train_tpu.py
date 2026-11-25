import argparse
import glob
import json
import math
import os
import queue
import threading
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer

from .config import DEFAULT_MODEL, DEFAULT_TRAIN, ModelConfig, TrainConfig
from .data import PRETRAIN_STREAMS, WeightedStream, download_training_files, json_safe
from .model import causal_lm_loss, forward, initialize_parameters, parameter_shapes

TOKENIZER = "hf-internal-testing/llama-tokenizer"
OUTPUT_DIR = os.environ.get("PARTH_OUT", "/kaggle/working/parth")
RESUME_GLOB = "/kaggle/input/**/*parth_step_*.json"


def cosine_lr(step: int, cfg: TrainConfig, total_steps: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
    floor = cfg.learning_rate * cfg.min_lr_ratio
    return floor + (cfg.learning_rate - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def adamw_step(
    params: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    opt: dict[str, Any],
    lr: jax.Array,
    cfg: TrainConfig,
):
    step = opt["step"] + 1
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads.values()))
    clip = jnp.minimum(1.0, cfg.grad_clip / (grad_norm + 1e-6))
    t = step.astype(jnp.float32)
    correction1 = 1.0 - cfg.beta1**t
    correction2 = 1.0 - cfg.beta2**t
    new_params: dict[str, jax.Array] = {}
    new_m: dict[str, jax.Array] = {}
    new_v: dict[str, jax.Array] = {}
    for name, value in params.items():
        grad = grads[name] * clip
        m = cfg.beta1 * opt["m"][name] + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * opt["v"][name] + (1.0 - cfg.beta2) * jnp.square(grad)
        update = (m / correction1) / (jnp.sqrt(v / correction2) + cfg.adam_eps)
        if not name.endswith("layernorm.weight") and name != "norm.weight":
            update = update + cfg.weight_decay * value
        new_params[name] = value - lr * update
        new_m[name] = m
        new_v[name] = v
    return new_params, {"step": step, "m": new_m, "v": new_v}, grad_norm


def make_train_step(mesh, model_cfg: ModelConfig, train_cfg: TrainConfig, dtype, loss_scale: float = 1.0):
    replicated = jax.sharding.PartitionSpec()
    data_sharded = jax.sharding.PartitionSpec("data")

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_step(params, opt, batch, lr):
        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(replicated, replicated, data_sharded, replicated),
            out_specs=(replicated, replicated, replicated, replicated),
        )
        def parallel_step(params, opt, batch, lr):
            microbatches = batch[0]

            def accumulate(acc_grads, microbatch):
                def scaled_loss_fn(params):
                    return causal_lm_loss(params, microbatch, model_cfg, dtype) * loss_scale

                loss, scaled_grads = jax.value_and_grad(scaled_loss_fn)(params)
                grads = jax.tree_util.tree_map(lambda g: g / loss_scale, scaled_grads)
                return jax.tree_util.tree_map(jnp.add, acc_grads, grads), loss / loss_scale

            initial = jax.tree_util.tree_map(jnp.zeros_like, params)
            grads, losses = jax.lax.scan(accumulate, initial, microbatches)
            count = microbatches.shape[0]
            grads = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x / count, "data"), grads)
            loss = jax.lax.pmean(jnp.mean(losses), "data")
            new_params, new_opt, grad_norm = adamw_step(params, grads, opt, lr, train_cfg)
            return new_params, new_opt, loss, grad_norm

        return parallel_step(params, opt, batch, lr)

    return train_step


def make_eval_step(model_cfg: ModelConfig, dtype):
    @jax.jit
    def eval_step(params, batch):
        return causal_lm_loss(params, batch, model_cfg, dtype, remat=False)

    return eval_step


class ResourceMonitor:
    def __init__(self):
        self.phase = "setup"
        self.low_memory = False
        self.peak_rss = 0.0
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        process = psutil.Process()
        while True:
            self.peak_rss = max(self.peak_rss, process.memory_info().rss / 2**30)
            self.low_memory = psutil.virtual_memory().available / 2**30 < 2.0
            time.sleep(1)


class TokenBatchLoader:
    def __init__(self, tokenizer, model_cfg: ModelConfig, train_cfg: TrainConfig, devices: int, state=None):
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.devices = devices
        self.initial_state = state
        self.latest_state = state
        self.queue = queue.Queue(maxsize=24)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_weighted_stream(self):
        return WeightedStream(
            PRETRAIN_STREAMS,
            shuffle_buffer=self.train_cfg.shuffle_buffer,
            seed=0,
            states=self.latest_state,
        )

    def _run(self):
        required = self.devices * self.train_cfg.grad_accum * self.train_cfg.micro_batch * (self.model_cfg.seq_len + 1)
        failures = 0
        stream = None
        while not self.stop_event.is_set():
            try:
                if stream is None:
                    stream = self._open_weighted_stream()
                buffer = np.empty(0, dtype=np.int32)
                while not self.stop_event.is_set():
                    while buffer.size < required:
                        docs = [stream.next_document() for _ in range(32)]
                        encoded = self.tokenizer(docs, add_special_tokens=False)["input_ids"]
                        pieces = [np.asarray(ids + [self.tokenizer.eos_token_id], dtype=np.int32) for ids in encoded]
                        buffer = np.concatenate([buffer] + pieces)
                    self.latest_state = stream.state_dict()
                    batch = buffer[:required].reshape(
                        self.devices,
                        self.train_cfg.grad_accum,
                        self.train_cfg.micro_batch,
                        self.model_cfg.seq_len + 1,
                    ).copy()
                    buffer = buffer[required:]
                    failures = 0
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put((batch, self.latest_state), timeout=1)
                            break
                        except queue.Full:
                            pass
            except Exception as exc:
                failures += 1
                if failures >= 6:
                    self.queue.put(exc)
                    return
                print(f"[data] {type(exc).__name__}: {exc}; reopening in {20 * failures}s", flush=True)
                stream = None
                self.stop_event.wait(20 * failures)

    def get(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise RuntimeError("data loader failed") from item
        return item

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)


def checkpoint_base(step: int, out_dir: str) -> str:
    return os.path.join(out_dir, f"parth_step_{step:08d}")


def find_checkpoint(out_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(out_dir, "parth_step_*.json"))
    candidates += glob.glob(RESUME_GLOB, recursive=True)
    if not candidates:
        return None
    newest = max(candidates, key=os.path.basename)
    return newest[:-5]


def flatten_for_save(params, opt) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for prefix, tree in (("p:", params), ("m:", opt["m"]), ("v:", opt["v"])):
        for key, value in tree.items():
            arrays[prefix + key] = np.asarray(value)
    return arrays


def unflatten_tree(flat: dict[str, np.ndarray], prefix: str, sharding=None):
    tree = {}
    for key, value in flat.items():
        if key.startswith(prefix):
            arr = value[len(prefix):]
            tree[arr] = jax.device_put(value, sharding) if sharding is not None else value
    return tree


def save_checkpoint(path_base: str, params, opt, step: int, stream_state, history, model_cfg: ModelConfig):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    arrays = flatten_for_save(params, opt)
    tmp = path_base + ".safetensors.tmp"
    save_file(arrays, tmp)
    os.replace(tmp, path_base + ".safetensors")
    metadata = {
        "step": int(step),
        "optimizer_step": int(opt["step"]),
        "stream_state": json_safe(stream_state),
        "history": history,
        "model_config": model_cfg.to_dict(),
        "datasets_version": datasets.__version__,
    }
    with open(path_base + ".json", "w") as handle:
        json.dump(metadata, handle)
    for old in glob.glob(os.path.join(os.path.dirname(path_base), "parth_step_*")):
        if not old.startswith(path_base):
            os.remove(old)
    print(f"[checkpoint] step {step} -> {path_base}.safetensors", flush=True)


def load_checkpoint(path_base: str, sharding=None):
    with open(path_base + ".json") as handle:
        metadata = json.load(handle)
    flat = load_file(path_base + ".safetensors")
    params = unflatten_tree(flat, "p:", sharding)
    opt = {
        "step": jax.device_put(jnp.asarray(metadata["optimizer_step"], dtype=jnp.int32), sharding),
        "m": unflatten_tree(flat, "m:", sharding),
        "v": unflatten_tree(flat, "v:", sharding),
    }
    return params, opt, metadata


def build_validation(tokenizer, model_cfg: ModelConfig, train_cfg: TrainConfig):
    needed = train_cfg.eval_sequences * (model_cfg.seq_len + 1)
    from .data import PRETRAIN_STREAMS, open_stream

    spec = next(x for x in PRETRAIN_STREAMS if x.name == "fineweb_edu")
    ds = open_stream(spec, 0, 123)
    ids: list[int] = []
    read = 0
    while len(ids) < needed and read < train_cfg.eval_documents:
        example = next(ds)
        read += 1
        ids.extend(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id)
    if len(ids) < needed:
        raise RuntimeError(f"validation stream ended at {len(ids)} tokens")
    return np.asarray(ids[:needed], dtype=np.int32).reshape(train_cfg.eval_sequences, model_cfg.seq_len + 1)


def evaluate(params, validation, eval_step, micro_batch: int):
    losses = []
    for start in range(0, len(validation), micro_batch):
        batch = jnp.asarray(validation[start:start + micro_batch])
        losses.append(float(eval_step(params, batch)))
    return float(np.mean(losses))


@jax.jit
def static_logits(params, tokens, cfg: ModelConfig, dtype):
    return forward(params, tokens, cfg, dtype, remat=False)[0, 0]


def sample_completion(params, tokenizer, model_cfg: ModelConfig, dtype, prompt: str, length: int = 100):
    ids = [tokenizer.eos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = ids[-(model_cfg.seq_len - length - 1):]
    buf = np.zeros((1, model_cfg.seq_len), dtype=np.int32)
    buf[0, :len(ids)] = ids
    end = len(ids)
    while end < min(len(ids) + length, model_cfg.seq_len):
        # Evaluate the whole fixed-width window; causal attention prevents padding from affecting earlier positions.
        logits = forward(params, jnp.asarray(buf), model_cfg, dtype, remat=False)[0, end - 1]
        next_id = int(jnp.argmax(logits))
        if next_id == tokenizer.eos_token_id:
            break
        buf[0, end] = next_id
        end += 1
    return tokenizer.decode(buf[0, 1:end])


def export_weights_only(out_dir: str, model_cfg: ModelConfig):
    base = find_checkpoint(out_dir)
    if base is None:
        raise RuntimeError("no checkpoint found")
    with open(base + ".json") as handle:
        metadata = json.load(handle)
    arrays = {k: v for k, v in load_file(base + ".safetensors").items() if k.startswith("p:")}
    target = os.path.join(out_dir, f"parth_weights_step_{metadata['step']:08d}")
    save_file(arrays, target + ".safetensors")
    with open(target + ".json", "w") as handle:
        json.dump({"step": metadata["step"], "model_config": model_cfg.to_dict()}, handle)
    print(f"[export] {target}.safetensors", flush=True)


def prune_checkpoints(out_dir: str, keep: int = 2):
    checkpoints = sorted(glob.glob(os.path.join(out_dir, "parth_step_*.json")), key=os.path.basename)
    for metadata_path in checkpoints[:-keep]:
        base = metadata_path[:-5]
        for suffix in (".json", ".safetensors"):
            try:
                os.remove(base + suffix)
            except FileNotFoundError:
                pass


def train(args):
    model_cfg = DEFAULT_MODEL
    train_cfg = TrainConfig(
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        total_tokens=args.total_tokens,
        max_hours=args.max_hours,
    )
    started = time.time()
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    if args.dtype == "auto":
        dtype = jnp.bfloat16 if devices[0].platform == "tpu" else jnp.float16 if devices[0].platform == "gpu" else jnp.float32
    else:
        dtype = getattr(jnp, args.dtype)
    loss_scale = args.loss_scale if dtype == jnp.float16 else 1.0
    print(f"jax {jax.__version__}: {devices[0].platform} {devices[0].device_kind}, {len(devices)} device(s), dtype={dtype}, loss_scale={loss_scale}")
    print(f"model: {model_cfg.parameter_count() / 1e6:.1f}M parameters, config={model_cfg.to_dict()}")

    tokens_per_step = train_cfg.tokens_per_step(model_cfg, len(devices))
    total_steps = math.ceil(train_cfg.total_tokens / tokens_per_step)
    print(f"{total_steps} steps planned, {tokens_per_step:,} tokens/step, {train_cfg.total_tokens / 1e9:.1f}B tokens")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.model_max_length = 10**9
    if args.download_data:
        cache_dir = os.environ.get("PARTH_DATA_CACHE", "/tmp/parth_data_cache")
        os.environ["PARTH_DATA_CACHE"] = cache_dir
        monitor = ResourceMonitor()
        monitor.phase = "data download"
        download_training_files(cache_dir, workers=args.download_workers)
    if not args.download_data:
        monitor = ResourceMonitor()
    monitor.phase = "validation data"
    validation = build_validation(tokenizer, model_cfg, train_cfg)
    eval_step = make_eval_step(model_cfg, dtype)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint = find_checkpoint(OUTPUT_DIR)
    if checkpoint:
        params, opt, metadata = load_checkpoint(checkpoint, replicated)
        step = int(metadata["step"])
        stream_state = metadata.get("stream_state")
        history = metadata.get("history", {"train": [], "eval": []})
        print(f"resuming from {checkpoint} at step {step}")
    else:
        params = {k: jax.device_put(v, replicated) for k, v in initialize_parameters(model_cfg).items()}
        opt = {
            "step": jax.device_put(jnp.asarray(0, dtype=jnp.int32), replicated),
            "m": {k: jax.device_put(jnp.zeros_like(v), replicated) for k, v in params.items()},
            "v": {k: jax.device_put(jnp.zeros_like(v), replicated) for k, v in params.items()},
        }
        step = 0
        stream_state = None
        history = {"train": [], "eval": []}
        print("initializing from scratch")

    loader = TokenBatchLoader(tokenizer, model_cfg, train_cfg, len(devices), stream_state)
    train_step = make_train_step(mesh, model_cfg, train_cfg, dtype, loss_scale)
    monitor.phase = "compile"
    loss_sum = 0.0
    loss_count = 0
    timed_out = False
    try:
        while step < total_steps:
            if monitor.low_memory:
                raise MemoryError("host memory nearly exhausted")
            if time.time() - started > train_cfg.max_hours * 3600:
                timed_out = True
                break
            batch, stream_state = loader.get()
            device_batch = jax.device_put(jnp.asarray(batch), sharded)
            lr = jnp.asarray(cosine_lr(step, train_cfg, total_steps), dtype=jnp.float32)
            params, opt, loss, grad_norm = train_step(params, opt, device_batch, lr)
            step += 1
            if monitor.phase == "compile":
                loss.block_until_ready()
                print(f"compiled; host RAM peak {monitor.peak_rss:.1f} GB", flush=True)
                monitor.phase = "train"
                timer = time.time()
            else:
                loss_sum += float(loss)
                loss_count += 1

            if loss_count and (step % train_cfg.print_every == 0 or step == total_steps):
                mean_loss = loss_sum / loss_count
                elapsed = max(time.time() - timer, 1e-9)
                per_step = elapsed / loss_count
                rate = tokens_per_step / per_step
                history["train"].append([step, mean_loss])
                if step % train_cfg.stats_every == 0 or step == total_steps:
                    eta = (total_steps - step) * per_step / 3600
                    print(
                        f"step {step}/{total_steps} loss={mean_loss:.4f} gnorm={float(grad_norm):.2f} "
                        f"lr={cosine_lr(step - 1, train_cfg, total_steps):.2e} {rate:,.0f} tok/s eta={eta:.1f}h q={loader.queue.qsize()}",
                        flush=True,
                    )
                else:
                    print(f"step {step}/{total_steps} loss={mean_loss:.4f}", flush=True)
                if not math.isfinite(mean_loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                loss_sum = 0.0
                loss_count = 0
                timer = time.time()

            if step % train_cfg.eval_every == 0 or step == total_steps:
                monitor.phase = "eval"
                value = evaluate(params, validation, eval_step, train_cfg.micro_batch)
                history["eval"].append([step, value])
                print(f"[eval] step {step} heldout_loss={value:.4f} ppl={math.exp(value):.1f}", flush=True)
                monitor.phase = "train"

            if step % train_cfg.save_every == 0 or step == total_steps:
                save_checkpoint(checkpoint_base(step, OUTPUT_DIR), params, opt, step, stream_state, history, model_cfg)
                prune_checkpoints(OUTPUT_DIR)
    except FloatingPointError:
        raise
    except BaseException as exc:
        if not any(x.is_deleted() for x in jax.tree_util.tree_leaves((params, opt))):
            save_checkpoint(checkpoint_base(step, OUTPUT_DIR), params, opt, step, stream_state, history, model_cfg)
            print(f"{type(exc).__name__} at step {step}; checkpoint saved", flush=True)
        raise
    finally:
        loader.close()

    if timed_out:
        save_checkpoint(checkpoint_base(step, OUTPUT_DIR), params, opt, step, stream_state, history, model_cfg)
        print(f"out of time at step {step}/{total_steps}; resume this kernel output in the next version", flush=True)

    for prompt in [
        "The water cycle begins when",
        "User: What causes rain?\n\nAssistant:",
        "Compute the result.\n148 + 265 =",
    ]:
        print(sample_completion(params, tokenizer, model_cfg, dtype, prompt, 100), "\n---", flush=True)
    export_weights_only(OUTPUT_DIR, model_cfg)
    return step, total_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch", type=int, default=DEFAULT_TRAIN.micro_batch)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_TRAIN.grad_accum)
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TRAIN.total_tokens)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_TRAIN.max_hours)
    parser.add_argument("--download-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--loss-scale", type=float, default=256.0)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
