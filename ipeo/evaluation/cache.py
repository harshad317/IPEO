"""File-backed response cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ipeo.core.ids import stable_hash
from ipeo.core.io import ensure_parent
from ipeo.core.schemas import Example, GenerationConfig, ModelResponse, PromptCandidate
from ipeo.models.base import ModelAdapter


def make_cache_key(
    model: ModelAdapter,
    prompt: PromptCandidate,
    example: Example,
    generation_config: GenerationConfig,
) -> str:
    payload = {
        "provider": model.provider,
        "model_id": model.model_id,
        "model_version": model.version,
        "prompt_text": prompt.text,
        "input_text": example.input,
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
        "max_tokens": generation_config.max_tokens,
        "stop": generation_config.stop,
        "system_prompt": generation_config.system_prompt,
    }
    return stable_hash(payload, length=64)


class ResponseCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> ModelResponse:
        data = json.loads(self._path(key).read_text(encoding="utf-8"))
        return ModelResponse(**data)

    def save(self, key: str, response: ModelResponse) -> None:
        path = ensure_parent(self._path(key))
        path.write_text(json.dumps(response.__dict__, sort_keys=True), encoding="utf-8")

    def raw_output_path(self, key: str) -> str:
        return str(self._path(key))


def response_to_dict(response: ModelResponse) -> dict[str, Any]:
    return response.__dict__.copy()
