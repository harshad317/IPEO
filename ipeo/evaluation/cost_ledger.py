"""Cost ledger logging and aggregation."""

from __future__ import annotations

from pathlib import Path

from ipeo.core.io import read_jsonl, write_jsonl
from ipeo.core.schemas import CostLog, GenerationConfig, ModelResponse, PromptCandidate
from ipeo.models.base import ModelAdapter


class CostLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: list[CostLog] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(
        self,
        *,
        run_id: str,
        task_id: str,
        model: ModelAdapter,
        method: str,
        fold_id: str,
        seed: int,
        phase: str,
        prompt: PromptCandidate | None,
        edit_id: str | None,
        example_id: str | None,
        response: ModelResponse,
        config: GenerationConfig,
        cache_hit: bool,
    ) -> CostLog:
        dollar_cost = (
            response.input_tokens / 1000 * model.price_input_per_1k
            + response.output_tokens / 1000 * model.price_output_per_1k
        )
        if cache_hit:
            dollar_cost = 0.0
        row = CostLog(
            run_id=run_id,
            task_id=task_id,
            model_id=model.model_id,
            provider=model.provider,
            api_model_version=model.version,
            method=method,
            fold_id=fold_id,
            seed=seed,
            phase=phase,  # type: ignore[arg-type]
            prompt_id=prompt.prompt_id if prompt else None,
            edit_id=edit_id,
            example_id=example_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.input_tokens + response.output_tokens,
            api_price_input=model.price_input_per_1k,
            api_price_output=model.price_output_per_1k,
            dollar_cost=dollar_cost,
            latency_ms=response.latency_ms,
            timestamp=response.timestamp,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cache_hit=cache_hit,
        )
        self.rows.append(row)
        write_jsonl(self.path, [row], append=True)
        return row

    def aggregate(self) -> dict[str, float]:
        rows = read_jsonl(self.path)
        return {
            "calls": float(sum(1 for row in rows if not row.get("cache_hit", False))),
            "dollars": float(sum(row.get("dollar_cost", 0.0) for row in rows)),
            "input_tokens": float(sum(row.get("input_tokens", 0) for row in rows)),
            "output_tokens": float(sum(row.get("output_tokens", 0) for row in rows)),
        }

    def method_cost(self, method: str, phase: str | None = None, model_ids: set[str] | None = None) -> float:
        total = 0.0
        for row in read_jsonl(self.path):
            if row.get("method") != method:
                continue
            if phase is not None and row.get("phase") != phase:
                continue
            if model_ids is not None and row.get("model_id") not in model_ids:
                continue
            total += float(row.get("dollar_cost", 0.0))
        return total

    def method_calls(self, method: str, model_ids: set[str] | None = None) -> int:
        total = 0
        for row in read_jsonl(self.path):
            if row.get("method") != method:
                continue
            if row.get("cache_hit"):
                continue
            if model_ids is not None and row.get("model_id") not in model_ids:
                continue
            total += 1
        return total
