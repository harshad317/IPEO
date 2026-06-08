# IPEO

Runnable MVP for Invariant Prompt-Edit Optimization across black-box model
environments. The default dry run is deterministic and requires no API keys.

## Smoke Test

```bash
python -m pytest
python -m ipeo.runners.run_dry \
  --tasks gsm8k \
  --models mock_openai_a mock_openai_b mock_openai_c mock_openai_d \
  --num_prompts 8 \
  --num_examples 8 \
  --fold_target mock_openai_d
```

## Full Offline Dry Run

```bash
python -m ipeo.runners.run_dry \
  --tasks gsm8k bbh classification extraction_qa \
  --models mock_openai_a mock_openai_b mock_openai_c mock_openai_d \
  --num_prompts 20 \
  --num_examples 24 \
  --fold_target mock_openai_d \
  --cache_dir artifacts/cache \
  --cost_log artifacts/costs/dry_run.jsonl
```

Artifacts are written under `artifacts/` as JSONL and CSV files.

## Live OpenAI Benchmark

Set `OPENAI_API_KEY` first, then run:

```bash
python -m ipeo.runners.run_openai \
  --tasks gsm8k bbh classification extraction_qa \
  --model gpt-4.1-mini \
  --num_prompts 20 \
  --num_examples 24 \
  --methods all \
  --workers 8 \
  --progress both \
  --artifact_dir artifacts/gpt41mini_benchmark \
  --cache_dir artifacts/gpt41mini_benchmark/cache \
  --cost_log artifacts/gpt41mini_benchmark/costs/run.jsonl
```

`--methods all` runs the implemented fixed-pool methods and records requested
official optimizer status for GEPA, MIPROv2, and CAPO.
