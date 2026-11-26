"""Publish an evaluated Parth HF export with an auditable model card."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def write_model_card(model_dir: str | Path, report_path: str | Path, repo_id: str):
    model_dir = Path(model_dir)
    report = json.loads(Path(report_path).read_text())
    if report.get("evaluation_mode", "full") != "full" or report.get("intelligence_index") is None:
        raise ValueError("publish requires a complete full evaluation report, not a limited smoke-test report")
    scores = report.get("scores", {})
    table = "\n".join(f"| {name} | {value:.2f} |" for name, value in sorted(scores.items()))
    card = f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- from-scratch
- small-language-model
- open-slm
---

# Parth

Parth is a 337M-parameter decoder-only language model pretrained from scratch as a light general chat assistant. It uses a Llama-style architecture with RoPE, RMSNorm, SwiGLU, grouped-query attention, and tied embeddings.

## Training data

The pretraining mixture uses open datasets including OpenBMB Ultra-FineWeb, OpenBMB UltraData-Math, FineWeb-Edu, SmolTalk, fresh synthetic arithmetic, and Qwenputer-inspired tiny-CPU execution traces. It does not train on Open SLM benchmark examples.

## Open SLM local estimate

| Benchmark | Score |
|---|---:|
{table}

Intelligence Index estimate: **{report.get('intelligence_index'):.2f}**

The estimate was produced locally with zero-shot LM-eval tasks and the official ArithMark-3 continuation script. These are continuation/log-likelihood scores; chain-of-thought prompting and test-time compute do not alter the official metric.

## Intended use

Parth is a small general assistant for experimentation. It is not benchmaxxed and should not be used for high-stakes factual, medical, legal, or safety-critical decisions.
"""
    (model_dir / "README.md").write_text(card)


def publish(model_dir: str, report: str, repo_id: str, private: bool = False):
    write_model_card(model_dir, report, repo_id)
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("report")
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    print(publish(args.model_dir, args.report, args.repo_id, args.private))


if __name__ == "__main__":
    main()
