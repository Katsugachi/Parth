"""Queue a batch TPU pretraining job on Kaggle.

The working API path is a script kernel with ``machine_shape=TpuV5E8`` and no
custom docker image. Interactive sessions and docker-pinned TPU pushes can boot
CPU-only; batch TPU sessions normally enter Kaggle's scarce-resource queue.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "kaggle_tpu_runner.py"


def runner_source(commit: str, total_tokens: int, max_hours: float, micro_batch: int, grad_accum: int) -> str:
    source = TEMPLATE.read_text()
    replacements = {
        "__PARTH_COMMIT__": commit,
        "__PARTH_TOTAL_TOKENS__": str(total_tokens),
        "__PARTH_MAX_HOURS__": str(max_hours),
        "__PARTH_MICRO_BATCH__": str(micro_batch),
        "__PARTH_GRAD_ACCUM__": str(grad_accum),
        "__PARTH_FALLBACK_GRAD_ACCUM__": str(grad_accum * 2 if micro_batch > 1 else grad_accum),
    }
    for key, value in replacements.items():
        if key not in source:
            raise RuntimeError(f"template placeholder missing: {key}")
        source = source.replace(key, value)
    return source


def write_kernel(directory: Path, args, commit: str):
    (directory / "parth_tpu_train.py").write_text(
        runner_source(commit, args.total_tokens, args.max_hours, args.micro_batch, args.grad_accum)
    )
    metadata = {
        "id": args.slug,
        "title": args.title,
        "code_file": "parth_tpu_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if args.private else "false",
        "enable_gpu": "false",
        "enable_internet": "true",
        "machine_shape": "TpuV5E8",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": args.kernel_source,
        "model_sources": [],
    }
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default="katsugachi/parth-pretrain-tpu-v1")
    parser.add_argument("--title", default="Parth pretrain TPU v1")
    parser.add_argument("--commit", default=None, help="full commit or ref pushed to GitHub; defaults to HEAD")
    parser.add_argument("--total-tokens", type=int, default=8_000_000_000)
    parser.add_argument("--max-hours", type=float, default=7.5)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--kernel-source", action="append", default=[], help="prior kernel output to attach for checkpoint resume")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.dry_run:
        directory = Path(tempfile.mkdtemp(prefix="parth-kaggle-dry-"))
        write_kernel(directory, args, commit)
        print(directory)
        return
    with tempfile.TemporaryDirectory(prefix="parth-kaggle-") as tmp:
        directory = Path(tmp)
        write_kernel(directory, args, commit)
        kaggle = os.environ.get("KAGGLE_CLI", "kaggle")
        subprocess.run([kaggle, "kernels", "push", "-p", str(directory)], check=True)
        print(f"queued https://www.kaggle.com/code/{args.slug}")


if __name__ == "__main__":
    main()
