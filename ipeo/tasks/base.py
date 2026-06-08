"""Task adapter protocol and common metric helpers."""

from __future__ import annotations

import re
import string
from typing import Any, Protocol

from ipeo.core.schemas import Example


class TaskAdapter(Protocol):
    task_id: str
    metric_name: str
    max_tokens: int

    def load_split(self, split: str, limit: int | None = None) -> list[Example]:
        ...

    def format_input(self, example: Example) -> str:
        ...

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        ...

    def score(self, parsed: dict[str, Any], gold: Any) -> float:
        ...


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = 0
    remaining = gold_tokens.copy()
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_first_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else None
