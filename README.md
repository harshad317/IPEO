# IPEO

Runnable MVP for Invariant Prompt-Edit Optimization across black-box model
environments. The default dry run is deterministic and requires no API keys.

## Smoke Test

```bash
python -m pytest
python -m ipeo.runners.run_dry \
  --tasks gsm8k \
  --models mock_openai mock_anthropic mock_google mock_llama \
  --num_prompts 8 \
  --num_examples 8 \
  --fold_target mock_llama
```

## Full Offline Dry Run

```bash
python -m ipeo.runners.run_dry \
  --tasks gsm8k bbh classification extraction_qa \
  --models mock_openai mock_anthropic mock_google mock_llama \
  --num_prompts 20 \
  --num_examples 24 \
  --fold_target mock_llama \
  --cache_dir artifacts/cache \
  --cost_log artifacts/costs/dry_run.jsonl
```

Artifacts are written under `artifacts/` as JSONL and CSV files.
