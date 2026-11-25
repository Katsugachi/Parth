# Parth

Parth is a from-scratch 337M-parameter small language model pipeline targeting the Open SLM leaderboard while remaining a useful lightweight chat assistant.

## Components

- JAX/TPU pretraining (`parth/train_tpu.py`) with a Llama-style architecture, weighted open-data mixture, checkpoint resume, validation loss, and sample generations.
- Bounded local parquet caching (`parth/data.py`) for OpenBMB Ultra-FineWeb, OpenBMB UltraData-Math, FineWeb-Edu, and SmolTalk, plus fresh synthetic arithmetic and Qwenputer-inspired tiny-CPU execution traces.
- Stock Hugging Face Llama export (`parth/export_hf.py`) so leaderboard evaluation uses standard continuation log-likelihood scoring.
- Open SLM evaluation (`parth/evaluate.py`): zero-shot ARC-Easy/ARC-Challenge/HellaSwag/PIQA through LM-eval and official ArithMark-3 script.
- GRPO-style exact-reward post-training (`parth/rl_posttrain.py`), self-consistency test-time compute (`parth/ttc.py`), and validation-guarded direct weight editing (`parth/weight_editing.py`).
- Manual fluency sample generation (`parth/manual_chat.py`) and auditable Hugging Face publishing (`parth/publish_hf.py`).

## Important scope note

Open SLM scores are zero-shot continuation/log-likelihood results. Chain-of-thought and test-time compute can improve interactive use but cannot change the official continuation benchmark score. Qwenputer's nonstandard 4B artifact cannot be submitted directly because the leaderboard requires a from-scratch model below 400M parameters; Parth uses an adapted CPU-trace curriculum instead of copying that artifact or architecture.

## Reproduce

```bash
pip install -r requirements.txt -r requirements-eval.txt
python -m pytest tests
python -m parth.train_tpu --total-tokens 8000000000 --max-hours 8.25
python -m parth.export_hf /path/to/parth_weights_step_XXXXXXXX.safetensors exported/parth
python -m parth.evaluate exported/parth results/open-slm --device cuda
python -m parth.manual_chat exported/parth results/manual
```

Kaggle TPU pushes are automated by `scripts/kaggle_push.py`; the training commit must be pushed to GitHub first.
