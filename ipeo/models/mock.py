"""Deterministic mock model families for offline IPEO runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ipeo.core.schemas import GenerationConfig, ModelResponse
from ipeo.models.base import count_tokens


FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "mock_openai_a": {
        "output_format": 0.13,
        "reasoning_strategy": 0.12,
        "verification": 0.09,
        "evidence_use": 0.07,
        "label_mapping": 0.06,
        "extraction_boundary": 0.05,
        "cost_reduction": -0.02,
        "generic_hygiene": 0.03,
        "placebo": 0.0,
    },
    "mock_openai_b": {
        "output_format": 0.10,
        "reasoning_strategy": 0.08,
        "verification": 0.11,
        "evidence_use": 0.10,
        "label_mapping": 0.05,
        "extraction_boundary": 0.08,
        "cost_reduction": 0.01,
        "generic_hygiene": 0.03,
        "placebo": 0.0,
    },
    "mock_openai_c": {
        "output_format": 0.11,
        "reasoning_strategy": 0.10,
        "verification": 0.08,
        "evidence_use": 0.09,
        "label_mapping": 0.09,
        "extraction_boundary": 0.06,
        "cost_reduction": -0.01,
        "generic_hygiene": 0.02,
        "placebo": 0.0,
    },
    "mock_openai_d": {
        "output_format": 0.09,
        "reasoning_strategy": 0.11,
        "verification": 0.10,
        "evidence_use": 0.06,
        "label_mapping": 0.07,
        "extraction_boundary": 0.09,
        "cost_reduction": 0.02,
        "generic_hygiene": 0.02,
        "placebo": 0.0,
    },
}


TASK_BASE = {
    "gsm8k": 0.48,
    "bbh": 0.50,
    "classification": 0.58,
    "extraction_qa": 0.54,
    "ifbench": 0.46,
}


TASK_EDIT_HINTS = {
    "gsm8k": {
        "reasoning_strategy": 0.08,
        "verification": 0.07,
        "output_format": 0.04,
    },
    "bbh": {
        "decomposition": 0.08,
        "verification": 0.06,
        "output_format": 0.04,
    },
    "classification": {
        "label_mapping": 0.11,
        "output_format": 0.05,
        "verbosity_control": 0.03,
    },
    "extraction_qa": {
        "evidence_use": 0.10,
        "extraction_boundary": 0.10,
        "output_format": 0.04,
    },
    "ifbench": {
        "output_format": 0.10,
        "verification": 0.12,
        "verbosity_control": 0.08,
    },
}


def _hash_unit(*parts: str) -> float:
    payload = "|".join(parts).encode()
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)


def _infer_task(input_text: str) -> str:
    if "Constraint:" in input_text:
        return "ifbench"
    if "stickers" in input_text:
        return "gsm8k"
    if "what day" in input_text.lower():
        return "bbh"
    if "Classify" in input_text or "team won" in input_text or "company reported" in input_text:
        return "classification"
    return "extraction_qa"


def _gold_from_input(task_id: str, input_text: str) -> str:
    if task_id == "gsm8k":
        nums = [int(n) for n in re.findall(r"\d+", input_text)]
        return str(nums[0] * nums[1] + nums[2])
    if task_id == "bbh":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        today = next(day for day in days if day in input_text)
        offset = int(re.search(r"(\d+) days later", input_text).group(1))
        return days[(days.index(today) + offset) % 7]
    if task_id == "classification":
        lowered = input_text.lower()
        if "team" in lowered or "goal" in lowered:
            return "sports"
        if "company" in lowered or "revenue" in lowered:
            return "business"
        if "researchers" in lowered or "experiment" in lowered:
            return "science"
        return "world"
    if task_id == "ifbench":
        if "exactly 3 words" in input_text:
            return "Careful science wins"
        if "coral exactly 2 times" in input_text:
            return "coral reefs protect coral life."
        if "exactly 3 non-empty lines" in input_text:
            return "red\nblue\ngreen"
        if "all alphabetic letters must be uppercase" in input_text:
            return "FOCUS BUILDS MOMENTUM"
        if "exact token <END>" in input_text:
            return "Plans become action <END>"
        if "keys answer and confidence" in input_text:
            return '{"answer":"yes","confidence":1}'
        return "compliant"
    match = re.search(r"in ([A-Z][a-z]+) during", input_text)
    return match.group(1) if match else "unknown"


def _wrong_answer(task_id: str, gold: str) -> str:
    if task_id == "gsm8k":
        return str(int(float(gold)) + 1)
    if task_id == "bbh":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[(days.index(gold) + 1) % 7]
    if task_id == "classification":
        labels = ["sports", "business", "science", "world"]
        return labels[(labels.index(gold) + 1) % len(labels)]
    if task_id == "ifbench":
        return "This response ignores the constraint."
    return "unknown"


def _present_edit_types(prompt: str) -> set[str]:
    return set(re.findall(r"\[EDIT:([a-z_]+):[a-z0-9-]+\]", prompt))


@dataclass
class MockModelAdapter:
    model_id: str
    provider: str
    version: str
    price_input_per_1k: float = 0.0001
    price_output_per_1k: float = 0.0002

    def generate(self, prompt: str, input: str, config: GenerationConfig) -> ModelResponse:
        task_id = _infer_task(input)
        gold = _gold_from_input(task_id, input)
        edit_types = _present_edit_types(prompt)
        score_prob = TASK_BASE[task_id]
        for edit_type in edit_types:
            score_prob += FAMILY_WEIGHTS[self.model_id].get(edit_type, 0.0)
            score_prob += TASK_EDIT_HINTS.get(task_id, {}).get(edit_type, 0.0)
        score_prob -= max(0, count_tokens(prompt) - 120) * 0.001
        score_prob = min(0.96, max(0.05, score_prob))
        correct = _hash_unit(self.model_id, prompt, input) <= score_prob
        answer = gold if correct else _wrong_answer(task_id, gold)
        if "Respond with only the final answer" in prompt or "OUTPUT_ONLY" in prompt:
            raw = answer
        else:
            raw = f"Answer: {answer}"
        input_tokens = count_tokens(prompt) + count_tokens(input)
        output_tokens = count_tokens(raw)
        latency_ms = int(20 + 2 * output_tokens + 10 * _hash_unit(self.model_id, input, prompt))
        return ModelResponse(
            raw_text=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            provider_request_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=self.version,
            finish_reason="stop",
        )


def get_mock_model(model_id: str) -> MockModelAdapter:
    providers = {
        "mock_openai_a": "openai_family",
        "mock_openai_b": "openai_family",
        "mock_openai_c": "openai_family",
        "mock_openai_d": "openai_family",
    }
    if model_id not in providers:
        raise ValueError(f"Unknown OpenAI mock model: {model_id}")
    return MockModelAdapter(model_id=model_id, provider=providers[model_id], version=f"{model_id}-2026-06-08")


def get_models(model_ids: list[str]) -> list[MockModelAdapter]:
    return [get_mock_model(model_id) for model_id in model_ids]
