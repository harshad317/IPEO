"""OpenAI Responses API model adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ipeo.core.schemas import GenerationConfig, ModelResponse
from ipeo.models.base import count_tokens

GPT_41_MINI_INPUT_PER_1K = 0.0004
GPT_41_MINI_OUTPUT_PER_1K = 0.0016


def _pricing_for_model(model: str) -> tuple[float, float]:
    if model == "gpt-4.1-mini":
        return GPT_41_MINI_INPUT_PER_1K, GPT_41_MINI_OUTPUT_PER_1K
    return GPT_41_MINI_INPUT_PER_1K, GPT_41_MINI_OUTPUT_PER_1K


def _extract_output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    texts: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts).strip()


@dataclass
class OpenAIResponsesAdapter:
    model_id: str
    api_model: str
    provider: str = "openai_family"
    version: str = "gpt-4.1-mini"
    price_input_per_1k: float = GPT_41_MINI_INPUT_PER_1K
    price_output_per_1k: float = GPT_41_MINI_OUTPUT_PER_1K
    timeout_seconds: int = 120
    max_retries: int = 3

    @classmethod
    def from_model(cls, api_model: str, env_id: str) -> "OpenAIResponsesAdapter":
        price_in, price_out = _pricing_for_model(api_model)
        return cls(
            model_id=env_id,
            api_model=api_model,
            version=api_model,
            price_input_per_1k=price_in,
            price_output_per_1k=price_out,
        )

    def generate(self, prompt: str, input: str, config: GenerationConfig) -> ModelResponse:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for ipeo.runners.run_openai")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": self.api_model,
            "input": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input},
            ],
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }
        if config.top_p != 1.0:
            payload["top_p"] = config.top_p
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                latency_ms = int((time.perf_counter() - start) * 1000)
                usage = data.get("usage", {}) or {}
                raw_text = _extract_output_text(data)
                input_tokens = int(usage.get("input_tokens") or count_tokens(prompt) + count_tokens(input))
                output_tokens = int(usage.get("output_tokens") or count_tokens(raw_text))
                return ModelResponse(
                    raw_text=raw_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    provider_request_id=data.get("id"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_version=self.api_model,
                    finish_reason=data.get("status"),
                )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"OpenAI API request failed with HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI API request failed after {self.max_retries} attempts: {last_error}")


def build_openai_environments(api_model: str, count: int = 4) -> list[OpenAIResponsesAdapter]:
    if count < 1:
        raise ValueError("count must be at least 1")
    suffixes = ["source_a", "source_b", "source_c", "target"]
    while len(suffixes) < count:
        suffixes.append(f"env_{len(suffixes) + 1}")
    return [OpenAIResponsesAdapter.from_model(api_model, f"{api_model}:{suffixes[idx]}") for idx in range(count)]
