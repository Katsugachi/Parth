import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow.parquet as pq
from datasets import IterableDataset, load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download

from .qwenputer_curriculum import cpu_trace_document


@dataclass(frozen=True)
class StreamSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    weight: int
    text_field: str | None = "content"
    kind: str = "text"
    data_files: tuple[str, ...] | None = None


def spread_paths(template: str, indexes: list[int]) -> tuple[str, ...]:
    return tuple(template.format(i) for i in indexes)


ULTRA_FINEWEB_FILES = spread_paths(
    "data/ultrafineweb_en/ultrafineweb-en-part-{:04d}-of-2048.parquet",
    [1, 97, 193, 289, 385, 481, 577, 673, 769, 864],
)
ULTRADATA_MATH_FILES = spread_paths(
    "data/UltraData-Math-L2-preview/UltraData-Math-L2-part-{:05d}-of-00138.parquet",
    [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121, 133],
)
FINEWEB_EDU_FILES = spread_paths("sample/10BT/{:03d}_00000.parquet", list(range(8)))
SMOLTALK_FILES = spread_paths("data/all/train-{:05d}-of-00009.parquet", list(range(9)))


def cache_path(cache_dir: str, dataset: str, filename: str) -> str:
    return os.path.join(cache_dir, dataset, filename)


def download_training_files(cache_dir: str, workers: int = 8) -> list[str]:
    """Download the bounded parquet subset once so training is not network-bound."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    os.makedirs(cache_dir, exist_ok=True)
    jobs = []
    for spec in PRETRAIN_STREAMS:
        for filename in spec.data_files or ():
            jobs.append((spec.dataset, filename))

    def fetch(job):
        dataset, filename = job
        target = cache_path(cache_dir, dataset, filename)
        if os.path.exists(target) and os.path.getsize(target) > 1024:
            return target
        path = hf_hub_download(
            repo_id=dataset,
            filename=filename,
            repo_type="dataset",
            local_dir=os.path.join(cache_dir, dataset),
        )
        return path

    done = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            done.append(future.result())
            if index % 5 == 0 or index == len(futures):
                print(f"[data] cached {index}/{len(futures)} parquet files", flush=True)
    return done

PRETRAIN_STREAMS = [
    StreamSpec("ultrafineweb_en", "openbmb/Ultra-FineWeb", None, "en", 42, "content", "text", ULTRA_FINEWEB_FILES),
    StreamSpec("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 20, "text", "text", FINEWEB_EDU_FILES),
    StreamSpec("ultradata_math_l2", "openbmb/UltraData-Math", "UltraData-Math-L2-preview", "train", 16, "content", "text", ULTRADATA_MATH_FILES),
    StreamSpec("smoltalk", "HuggingFaceTB/smoltalk", "all", "train", 12, None, "chat", SMOLTALK_FILES),
    StreamSpec("arithmetic_words", "", None, "", 6, None, "synthetic_arithmetic"),
    StreamSpec("tiny_cpu_traces", "", None, "", 4, None, "synthetic_cpu"),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def render_chat(messages: list[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content:
            return None
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    if len(parts) < 2 or not any(p.startswith("Assistant:") for p in parts):
        return None
    return "\n\n".join(parts)


class ParquetExampleStream:
    """Small-memory deterministic stream over explicitly selected parquet shards.

    HF streaming materializes very large parquet row groups for Ultra-FineWeb. This
    reader uses PyArrow row batches directly and shuffles shard/row-group order,
    keeping host memory low enough for Kaggle's 32 GB workers.
    """

    def __init__(self, spec: StreamSpec, seed: int = 0, state: dict[str, Any] | None = None, batch_rows: int = 64):
        if spec.data_files is None:
            raise ValueError(f"{spec.name} requires explicit data_files")
        self.spec = spec
        self.seed = seed
        self.batch_rows = batch_rows
        self.fs = HfFileSystem()
        self.epoch = 0
        self.file_pos = 0
        self.group_pos = 0
        self.batch_pos = 0
        self.file_order: list[str] = []
        self.group_order: list[int] = []
        self.file = None
        self.parquet = None
        self.columns = ["messages"] if spec.kind == "chat" else [spec.text_field or "text"]
        self.batch: list[dict[str, Any]] = []
        self.batch_iter = None
        self._new_epoch()
        if state:
            self.epoch = int(state.get("epoch", 0))
            self.file_pos = int(state.get("file_pos", 0))
            target_group_pos = int(state.get("group_pos", 0))
            target_batch_pos = int(state.get("batch_pos", 0))
            self._new_epoch(replay=True)
            try:
                self._open_file()
                while self.group_pos < target_group_pos:
                    self._read_batch()
                    self.batch = []
                self._read_batch()
                for _ in range(min(target_batch_pos, len(self.batch))):
                    self.batch.pop(0)
                self.batch_pos = target_batch_pos
            except Exception:
                self.batch = []

    def _new_epoch(self, replay: bool = False):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        self.file_order = list(self.spec.data_files or ())
        rng.shuffle(self.file_order)
        if not replay:
            self.file_pos = 0
            self.group_pos = 0
            self.batch_pos = 0

    def _open_file(self):
        self._close_file()
        filename = self.file_order[self.file_pos % len(self.file_order)]
        local = os.path.join(os.environ.get("PARTH_DATA_CACHE", ""), self.spec.dataset, filename)
        if os.path.exists(local):
            self.file = open(local, "rb")
        else:
            path = "datasets/" + self.spec.dataset + "/" + filename
            self.file = self.fs.open(path, "rb")
        self.parquet = pq.ParquetFile(self.file)
        rng = random.Random(self.seed + 1000003 * self.epoch + 7919 * (self.file_pos + 1))
        self.group_order = list(range(self.parquet.metadata.num_row_groups))
        rng.shuffle(self.group_order)
        self.group_pos = 0
        self.batch_pos = 0
        self.batch = []
        self.batch_iter = None

    def _close_file(self):
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass
        self.file = None
        self.parquet = None

    def _read_batch(self):
        while True:
            if self.parquet is not None and self.batch_iter is not None:
                try:
                    table = next(self.batch_iter)
                    rows = table.to_pylist()
                    rng = random.Random(
                        self.seed + 1000003 * self.epoch + 7919 * (self.file_pos + 1) + 104729 * self.group_pos
                    )
                    rng.shuffle(rows)
                    self.batch = rows
                    self.batch_pos = 0
                    return
                except StopIteration:
                    self.batch_iter = None
            if self.parquet is not None and self.group_pos < len(self.group_order):
                group = self.group_order[self.group_pos]
                self.group_pos += 1
                self.batch_iter = self.parquet.iter_batches(
                    batch_size=self.batch_rows,
                    row_groups=[group],
                    columns=self.columns,
                    use_threads=True,
                )
                continue
            self.file_pos += 1
            if self.file_pos >= len(self.file_order):
                self.epoch += 1
                self._new_epoch()
            self._open_file()

    def __iter__(self):
        return self

    def __next__(self):
        while not self.batch:
            self._read_batch()
        row = self.batch.pop(0)
        self.batch_pos += 1
        return row

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "file_pos": self.file_pos,
            "group_pos": self.group_pos,
            "batch_pos": self.batch_pos,
        }

    def load_state_dict(self, state):
        # State is applied by the constructor in WeightedStream; this method exists
        # for interface compatibility with HF IterableDataset.
        return None


def open_stream(spec: StreamSpec, shuffle_buffer: int, seed: int, state: dict[str, Any] | None = None):
    if spec.data_files is not None:
        return ParquetExampleStream(spec, seed=seed, state=state)
    ds = load_dataset(spec.dataset, spec.config, split=spec.split, streaming=True)
    return ds.shuffle(buffer_size=shuffle_buffer, seed=seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


NAMES = [
    "Asha", "Bruno", "Carla", "Dev", "Elena", "Farid", "Gina", "Hugo", "Ivy", "Jasper",
    "Kira", "Liam", "Mina", "Nikhil", "Olive", "Pavel", "Quinn", "Rosa", "Sana", "Theo",
    "Uma", "Victor", "Wren", "Xavier", "Yara", "Zane", "Adrian", "Mei", "Noor", "Felix",
]
OBJECTS = [
    ("pencil", "pencils"), ("shell", "shells"), ("ticket", "tickets"), ("coin", "coins"),
    ("sticker", "stickers"), ("card", "cards"), ("book", "books"), ("apple", "apples"),
    ("orange", "oranges"), ("bead", "beads"), ("block", "blocks"), ("seed", "seeds"),
    ("ribbon", "ribbons"), ("bottle cap", "bottle caps"), ("crayon", "crayons"), ("stamp", "stamps"),
]
CONTAINERS = [("box", "boxes"), ("bag", "bags"), ("jar", "jars"), ("packet", "packets"), ("crate", "crates"), ("album", "albums")]


def arithmetic_document(index: int) -> str:
    rng = random.Random(0x51A7 + index)
    name = rng.choice(NAMES)
    singular, plural = rng.choice(OBJECTS)
    style = index % 10

    if style in (0, 1, 2):
        a = rng.randint(7, 489)
        b = rng.randint(4, 389)
        result = a + b
        templates = [
            f"{name} collected {a} {plural}. Then {name} found {b} more. Now {name} has {result} {plural}.",
            f"{name} started with {a} {plural} and bought {b} additional {plural}. The new total is {result} {plural}.",
            f"There were {a} {plural} in a pile. Another {b} were added. The pile then contained {result} {plural}.",
        ]
        return rng.choice(templates) + f"\nArithmetic check: {a} + {b} = {result}"

    if style in (3, 4):
        result = rng.randint(2, 240)
        b = rng.randint(3, 280)
        a = result + b
        templates = [
            f"{name} had {a} {plural} before giving away {b}. Afterward, {result} {plural} remained.",
            f"From a group of {a} {plural}, {b} were removed. This left {result} {plural}.",
        ]
        return rng.choice(templates) + f"\nArithmetic check: {a} - {b} = {result}"

    if style in (5, 6):
        groups = rng.randint(2, 24)
        each = rng.randint(2, 48)
        result = groups * each
        container_s, container_p = rng.choice(CONTAINERS)
        templates = [
            f"{name} packed {each} {plural} into each of {groups} {container_p}. Altogether there were {result} {plural}.",
            f"Each of {groups} {container_p} holds {each} {plural}. The total count is {result} {plural}.",
        ]
        return rng.choice(templates) + f"\nArithmetic check: {groups} × {each} = {result}"

    if style == 7:
        divisor = rng.randint(2, 24)
        quotient = rng.randint(2, 64)
        total = divisor * quotient
        return (
            f"{name} shared {total} {plural} equally among {divisor} friends. "
            f"Each friend received {quotient} {plural}.\nArithmetic check: {total} ÷ {divisor} = {quotient}"
        )

    if style == 8:
        a = rng.randint(12, 899)
        b = rng.randint(11, 899)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        result = a + b if op == "+" else a - b
        return f"Compute the result and continue the pattern.\n{a} {op} {b} = {result}"

    a = rng.randint(11, 99)
    b = rng.randint(11, 99)
    c = rng.randint(3, 29)
    result = a + b - c
    return (
        f"{name} had {a} {plural}, received {b} more, and then used {c}. "
        f"The final number was {result} {plural}.\nArithmetic check: {a} + {b} - {c} = {result}"
    )


def document_from_example(spec: StreamSpec, example: dict[str, Any], synthetic_index: int) -> str | None:
    if spec.kind == "synthetic_arithmetic":
        return arithmetic_document(synthetic_index)
    if spec.kind == "synthetic_cpu":
        return cpu_trace_document(synthetic_index)
    if spec.kind == "chat":
        messages = example.get("messages")
        return render_chat(messages) if isinstance(messages, list) else None
    text = example.get(spec.text_field or "text")
    if not isinstance(text, str):
        return None
    return text if text.strip() else None


class WeightedStream:
    """Deterministic weighted interleaving over independently resumable HF streams."""

    def __init__(
        self,
        specs: list[StreamSpec] = PRETRAIN_STREAMS,
        shuffle_buffer: int = 5000,
        seed: int = 0,
        states: dict[str, Any] | None = None,
    ):
        self.specs = specs
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.synthetic_index = 0 if not states else int(states.get("__synthetic_index", 0))
        self.position = 0 if not states else int(states.get("__position", 0))
        self.schedule = self._make_schedule()
        self.states: dict[str, Any] = states or {}
        self.datasets: dict[str, IterableDataset] = {}
        self.iterators: dict[str, Any] = {}
        for spec in specs:
            if spec.kind in {"synthetic_arithmetic", "synthetic_cpu"}:
                continue
            state = self.states.get(spec.name)
            ds = open_stream(spec, shuffle_buffer, seed, state=state)
            if state and spec.data_files is None:
                ds.load_state_dict(state)
            self.datasets[spec.name] = ds
            self.iterators[spec.name] = iter(ds)

    @staticmethod
    def _make_schedule_specs(specs: list[StreamSpec]) -> list[str]:
        schedule = []
        for spec in specs:
            schedule.extend([spec.name] * spec.weight)
        return schedule

    def _make_schedule(self) -> list[str]:
        schedule = self._make_schedule_specs(self.specs)
        random.Random(9137 + self.seed).shuffle(schedule)
        return schedule

    def state_dict(self) -> dict[str, Any]:
        state = {
            "__position": self.position,
            "__synthetic_index": self.synthetic_index,
        }
        for name, ds in self.datasets.items():
            try:
                state[name] = json_safe(ds.state_dict())
            except Exception as exc:
                state[name] = {"unserializable": repr(exc)}
        return state

    def next_document(self) -> str:
        by_name = {spec.name: spec for spec in self.specs}
        attempts = 0
        while True:
            attempts += 1
            if attempts > 100:
                raise RuntimeError("could not draw a usable document from the weighted stream")
            name = self.schedule[self.position % len(self.schedule)]
            self.position += 1
            spec = by_name[name]
            if spec.kind in {"synthetic_arithmetic", "synthetic_cpu"}:
                self.synthetic_index += 1
                if spec.kind == "synthetic_cpu":
                    return cpu_trace_document(self.synthetic_index)
                return arithmetic_document(self.synthetic_index)
            try:
                example = next(self.iterators[name])
            except StopIteration:
                ds = open_stream(spec, self.shuffle_buffer, self.seed + 1)
                self.datasets[name] = ds
                self.iterators[name] = iter(ds)
                example = next(self.iterators[name])
            text = document_from_example(spec, example, self.synthetic_index)
            if text is not None:
                return text
