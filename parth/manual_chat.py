"""Manual fluency test harness for Parth."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "User: Explain photosynthesis to a curious ten-year-old in two short paragraphs.\nAssistant:",
    "User: Give three practical tips for debugging a Python program.\nAssistant:",
    "User: Write a friendly one-paragraph note inviting a colleague to lunch.\nAssistant:",
    "User: What is the difference between a virus and a bacterium?\nAssistant:",
    "User: Continue this story naturally: The lighthouse keeper found a blue envelope under the door.\nAssistant:",
]


def run_manual_test(model_path: str, output_dir: str, max_new_tokens: int = 180) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    rows = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.75,
                top_p=0.92,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        rows.append({"prompt": prompt, "completion": completion})
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"model": model_path, "device": device, "samples": rows}
    (outdir / "manual_chat_samples.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_dir")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()
    run_manual_test(args.model, args.output_dir, args.max_new_tokens)


if __name__ == "__main__":
    main()
