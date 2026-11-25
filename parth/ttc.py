"""Test-time-compute decoding utilities for Parth.

Implements self-consistency voting and verifier-style reranking without changing
the stock architecture used by the Open SLM continuation benchmarks.
"""

from collections import Counter
import re
from typing import Callable

import torch


def extract_answer(text: str) -> str:
    patterns = [
        r"(?:final answer|answer|result)\s*(?:is|=|:)?\s*(-?\d[\d,]*)",
        r"=\s*(-?\d[\d,]*)\s*(?:\.|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", text)
    return numbers[-1].replace(",", "") if numbers else text.strip().splitlines()[-1]


def answer_quality(text: str) -> float:
    answer = extract_answer(text)
    score = 0.0
    if re.fullmatch(r"-?\d+", answer):
        score += 2.0
    lowered = text.lower()
    if "final answer" in lowered or "therefore" in lowered:
        score += 0.5
    if "i don't know" in lowered or "cannot" in lowered:
        score -= 2.0
    repetitive = max((len(list(g)) for _, g in __import__("itertools").groupby(text.split())), default=0)
    score -= max(0, repetitive - 4) * 0.2
    score -= max(0, len(text) - 1600) / 1600
    return score


@torch.no_grad()
def sample_candidates(model, tokenizer, prompt: str, samples: int = 8, max_new_tokens: int = 160, temperature: float = 0.8, device: str | None = None) -> list[str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=samples,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in outputs]


def self_consistent_answer(candidates: list[str], quality_weight: float = 0.25) -> tuple[str, list[str], dict[str, int]]:
    extracted = [extract_answer(x) for x in candidates]
    counts = Counter(extracted)
    best = max(counts, key=lambda answer: counts[answer] + quality_weight * sum(answer_quality(x) for x, a in zip(candidates, extracted) if a == answer))
    return best, extracted, dict(counts)


def test_time_compute_answer(model, tokenizer, prompt: str, samples: int = 8, max_new_tokens: int = 160, device: str | None = None) -> dict:
    candidates = sample_candidates(model, tokenizer, prompt, samples, max_new_tokens, device=device)
    answer, extracted, counts = self_consistent_answer(candidates)
    return {"answer": answer, "extracted": extracted, "vote_counts": counts, "candidates": candidates}
