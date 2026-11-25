"""Small GRPO-style post-training stage with exact-answer rewards.

This is deliberately narrow and auditable: arithmetic/CPU-trace prompts are grouped,
sampled, scored by a deterministic verifier, and optimized with a group-relative
advantage. It is RL rather than benchmark distillation: rewards come from executing
or solving fresh synthetic problems, not from leaderboard examples.
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .qwenputer_curriculum import arithmetic_program, cpu_trace_document, TinyCPU
from .ttc import extract_answer


def arithmetic_prompt(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(12, 480)
    b = rng.randint(7, 240)
    op = rng.choice(["+", "-", "*"])
    answer = str({"+" : a + b, "-": a - b, "*": a * b}[op])
    prompt = (
        "Solve step by step, then end with 'Final answer: <number>'.\n"
        f"User: Compute {a} {op} {b}.\nAssistant:"
    )
    return prompt, answer


def reward_completion(completion: str, expected: str) -> float:
    extracted = extract_answer(completion)
    reward = 0.0
    if extracted == expected:
        reward += 1.0
    elif extracted.lstrip("-") == expected.lstrip("-"):
        reward += 0.5
    if "final answer" in completion.lower():
        reward += 0.1
    if len(completion) > 1200:
        reward -= 0.2
    if not completion.strip():
        reward -= 1.0
    return reward


def completion_loss(model, tokenizer, prompts: list[str], completions: list[str], device: str) -> torch.Tensor:
    losses = []
    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([prompt_ids + completion_ids], device=device)
        labels = ids.clone()
        labels[0, : len(prompt_ids)] = -100
        out = model(input_ids=ids, labels=labels)
        losses.append(out.loss)
    return torch.stack(losses)


def grpo_epoch(model, tokenizer, step: int, args, device: str) -> dict:
    rng = random.Random(args.seed + step)
    all_prompts: list[str] = []
    all_completions: list[str] = []
    all_advantages: list[float] = []
    rewards = []
    for _ in range(args.problems_per_step):
        prompt, expected = arithmetic_prompt(rng)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                num_return_sequences=args.group_size,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        group_rewards = []
        group_completions = []
        for row in output:
            completion = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            r = reward_completion(completion, expected)
            group_rewards.append(r)
            group_completions.append(completion)
        mean = sum(group_rewards) / len(group_rewards)
        variance = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)
        std = max(variance ** 0.5, 1e-4)
        rewards.extend(group_rewards)
        for completion, r in zip(group_completions, group_rewards):
            all_prompts.append(prompt)
            all_completions.append(completion)
            all_advantages.append((r - mean) / std)
    losses = completion_loss(model, tokenizer, all_prompts, all_completions, device)
    advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device).clamp(-3, 3)
    objective = -(advantages.detach() * losses).mean() + args.kl_coef * losses.mean()
    return {
        "loss": objective,
        "mean_reward": sum(rewards) / len(rewards),
        "exact_rate": sum(r >= 1.0 for r in rewards) / len(rewards),
        "mean_nll": float(losses.detach().mean().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--problems-per-step", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    history = []
    for step in range(1, args.steps + 1):
        result = grpo_epoch(model, tokenizer, step, args, device)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"step": step, "mean_reward": result["mean_reward"], "exact_rate": result["exact_rate"], "mean_nll": result["mean_nll"]}
        history.append(row)
        print(json.dumps(row), flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "grpo_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
