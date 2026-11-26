"""Apply validation-guarded direct weight edits to a Parth HF checkpoint."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .weight_editing import interpolate, repair_nonfinite, rescale_to_rms, sparsify_smallest

DEFAULT_VALIDATION = [
    "The water cycle begins when sunlight warms water in oceans, lakes, and rivers. The water evaporates, cools, condenses into clouds, and later falls as rain or snow.",
    "User: Give two practical tips for debugging Python code.\nAssistant: First, reproduce the failure with a small input. Second, inspect the exact traceback and add narrow assertions or logging around the failing state.",
    "Compute the result.\n148 + 265 = 413",
]


@torch.no_grad()
def validation_loss(model, tokenizer, texts: list[str], device: str, max_tokens: int = 512) -> float:
    losses = []
    weights = []
    for text in texts:
        ids = tokenizer(text + tokenizer.eos_token, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
        if len(ids) < 2:
            continue
        batch = torch.tensor([ids], device=device)
        out = model(input_ids=batch, labels=batch)
        count = len(ids) - 1
        losses.append(out.loss.float() * count)
        weights.append(count)
    return float(torch.stack(losses).sum() / sum(weights))


def candidate_edits(state: dict[str, torch.Tensor], other: dict[str, torch.Tensor] | None):
    repaired, repair_count = repair_nonfinite(state)
    yield "repair_nonfinite", repaired, {"changed_values": repair_count}
    yield "rms_rescale", rescale_to_rms(state), {}
    sparse, sparse_count = sparsify_smallest(state, 0.0005)
    yield "sparsify_smallest_0.0005", sparse, {"changed_values": sparse_count}
    if other is not None:
        shared = set(state) & set(other)
        if len(shared) == len(state):
            for alpha in (0.25, 0.5, 0.75):
                yield f"interpolate_posttrain_{alpha}", interpolate(state, other, alpha), {"alpha": alpha}


def accepts_edit(loss: float, baseline: float, tolerance: float) -> bool:
    if not math.isfinite(loss):
        return False
    if not math.isfinite(baseline):
        return True
    return loss <= baseline + tolerance


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def run_edits(model_path: str, output: str, validation_file: str | None = None, other_model: str | None = None, tolerance: float = 0.0) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()
    texts = Path(validation_file).read_text().split("\n\n") if validation_file else DEFAULT_VALIDATION
    texts = [x for x in texts if x.strip()]
    baseline = validation_loss(model, tokenizer, texts, device)
    original = clone_state(model.state_dict())
    other = None
    if other_model:
        donor = AutoModelForCausalLM.from_pretrained(other_model, torch_dtype=dtype)
        other = clone_state(donor.state_dict())
        del donor
    best_loss = baseline
    best_name = "original"
    best_state = original
    trials = []
    for name, candidate_cpu, metadata in candidate_edits(original, other):
        candidate = {k: v.to(device=device, dtype=dtype) for k, v in candidate_cpu.items()}
        model.load_state_dict(candidate, strict=False)
        loss = validation_loss(model, tokenizer, texts, device)
        accepted = accepts_edit(loss, baseline, tolerance)
        trials.append({"name": name, "loss": loss, "accepted": accepted, **metadata})
        print(json.dumps(trials[-1]), flush=True)
        if accepted and (not math.isfinite(best_loss) or loss < best_loss):
            best_loss = loss
            best_name = name
            best_state = clone_state(candidate)
        model.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in original.items()}, strict=False)
        del candidate
    if best_state is not original:
        model.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in best_state.items()}, strict=False)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    report = {"model": model_path, "output": output, "baseline_loss": baseline, "best_loss": best_loss, "selected_edit": best_name, "trials": trials}
    (out / "weight_edit_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--validation-file")
    parser.add_argument("--other-model", help="optional post-trained checkpoint for validated weight interpolation")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run_edits(args.model, args.output, args.validation_file, args.other_model, args.tolerance), indent=2))


if __name__ == "__main__":
    main()
