import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ARITHMARK3_URL = "https://huggingface.co/datasets/AxiomicLabs/Arithmark-3.0/resolve/main/arithmark-3.jsonl"
ARITHMARK3_SCRIPT_URL = "https://huggingface.co/datasets/AxiomicLabs/Arithmark-3.0/raw/main/bencharithmark-3.py"
ARITHMARK2_URL = "https://huggingface.co/datasets/AxiomicLabs/ArithMark-2.0/resolve/main/arithmark_2.0.jsonl"
ARITHMARK2_SCRIPT_URL = "https://huggingface.co/datasets/AxiomicLabs/ArithMark-2.0/raw/main/benchmark_arithmark-2.0.py"

LEADERBOARD_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa"]


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)


def normalized(score: float, chance: float) -> float:
    return 100.0 * (score - chance) / (100.0 - chance)


def intelligence_index(scores: dict[str, float]) -> float:
    hellaswag = normalized(scores["hellaswag"], 25.0)
    arc_combined = normalized((scores["arc_easy"] + scores["arc_challenge"]) / 2.0, 25.0)
    piqa = normalized(scores["piqa"], 50.0)
    arithmark3 = normalized(scores["arithmark3"], 25.0)
    return (hellaswag + arc_combined + piqa + 0.65 * arithmark3) / 3.65


def metric_result(task_result: dict[str, Any]) -> float:
    for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if key in task_result:
            return float(task_result[key]) * 100.0
    raise KeyError(f"no accuracy metric in {task_result}")


def run_lm_eval(model_path: str, output_dir: Path, device: str, batch_size: str, limit: int | None = None) -> dict[str, float]:
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=float32" if device == "cpu" else f"pretrained={model_path},dtype=bfloat16",
        tasks=LEADERBOARD_TASKS,
        num_fewshot=0,
        batch_size=batch_size,
        device=device,
        log_samples=False,
        limit=limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lm_eval_raw.json").write_text(json.dumps(results, indent=2, default=str))
    return {task: metric_result(results["results"][task]) for task in LEADERBOARD_TASKS}


def run_arithmark3(model_path: str, work_dir: Path, device: str, batch_size: int, limit: int | None = None) -> tuple[float, dict[str, Any]]:
    data = work_dir / "arithmark-3.jsonl"
    script = work_dir / "bencharithmark-3.py"
    download(ARITHMARK3_URL, data)
    download(ARITHMARK3_SCRIPT_URL, script)
    if limit is not None:
        limited = work_dir / f"arithmark-3-limit-{limit}.jsonl"
        lines = data.read_text().splitlines()[:limit]
        limited.write_text("\n".join(lines) + "\n")
        data = limited
    cmd = [
        sys.executable,
        str(script),
        "--model",
        model_path,
        "--data-path",
        str(data),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--dtype",
        "float32" if device == "cpu" else "bfloat16",
        "--results-dir",
        str(work_dir / "arithmark3"),
    ]
    subprocess.run(cmd, check=True)
    tag = model_path.replace("/", "_")
    result_file = work_dir / "arithmark3" / f"{tag}_arithmark-3_results.json"
    result = json.loads(result_file.read_text())
    model_result = result["results"]["arithmark-3"]
    return float(model_result.get("acc_norm", model_result.get("acc"))), result


def run_evaluation(model_path: str, output_dir: str, device: str = "cuda", batch_size: str = "auto", skip_lm_eval: bool = False, limit: int | None = None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}
    if not skip_lm_eval:
        scores.update(run_lm_eval(model_path, out, device, batch_size, limit))
    arith_score, arith_details = run_arithmark3(
        model_path,
        out,
        device if device in ("cpu", "cuda") else "cuda",
        8 if device == "cpu" else 32,
        limit,
    )
    scores["arithmark3"] = arith_score
    details["arithmark3"] = arith_details
    ii = intelligence_index(scores) if all(k in scores for k in ["arc_easy", "arc_challenge", "hellaswag", "piqa"]) else None
    report = {
        "model": model_path,
        "evaluation_mode": "limited-smoke" if limit is not None else "full",
        "limit": limit,
        "scores": scores,
        "intelligence_index": ii,
        "method": {
            "lm_eval": "zero-shot acc_norm when available, acc otherwise",
            "arithmark3": "official script acc_norm",
            "formula": "(N(HellaSwag,25)+N(mean(ARC),25)+N(PIQA,50)+0.65*N(ArithMark3,25))/3.65",
        },
        "details": details,
    }
    report_name = "open_slm_report.json" if limit is None else f"open_slm_report_limit_{limit}.json"
    (out / report_name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["scores"], indent=2))
    print(f"Intelligence Index estimate: {ii:.2f}" if ii is not None else "incomplete evaluation")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--skip-lm-eval", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test only: evaluate at most this many examples per task")
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args.model, args.output_dir, args.device, args.batch_size, args.skip_lm_eval, args.limit)


if __name__ == "__main__":
    main()
