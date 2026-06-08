"""File-backed response cache."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock, get_ident
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
        self._lock = RLock()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._path(key).exists()

    def load(self, key: str) -> ModelResponse:
        response = self.load_or_none(key)
        if response is None:
            raise FileNotFoundError(f"No valid cache entry for key {key}")
        return response

    def load_or_none(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        with self._lock:
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return ModelResponse(**data)
            except (json.JSONDecodeError, OSError, TypeError):
                try:
                    path.unlink()
                except OSError:
                    pass
                return None

    def save(self, key: str, response: ModelResponse) -> None:
        path = ensure_parent(self._path(key))
        tmp_path = path.with_name(f".{path.name}.{get_ident()}.tmp")
        payload = json.dumps(response.__dict__, sort_keys=True)
        with self._lock:
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(path)

    def raw_output_path(self, key: str) -> str:
        return str(self._path(key))


def response_to_dict(response: ModelResponse) -> dict[str, Any]:
    return response.__dict__.copy()
