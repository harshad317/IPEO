"""Model adapter protocol."""

from __future__ import annotations

from typing import Protocol

from ipeo.core.schemas import GenerationConfig, ModelResponse


class ModelAdapter(Protocol):
    model_id: str
    provider: str
    version: str
    price_input_per_1k: float
    price_output_per_1k: float

    def generate(self, prompt: str, input: str, config: GenerationConfig) -> ModelResponse:
        ...


def count_tokens(text: str) -> int:
    return max(1, len(text.split()))
