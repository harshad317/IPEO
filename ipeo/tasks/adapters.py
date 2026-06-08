"""Offline task adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ipeo.core.schemas import Example
from ipeo.tasks.base import exact_match, parse_first_number, token_f1
from ipeo.tasks.fixtures import (
    bbh_examples,
    classification_examples,
    extraction_qa_examples,
    gsm8k_examples,
    ifbench_examples,
    ifbench_hard_examples,
)
from ipeo.tasks.ifbench_official import (
    OFFICIAL_TASK_ID,
    ensure_official_ifbench_evaluator_available,
    load_official_ifbench_examples,
    score_local_ifbench_constraint,
    score_official_ifbench_response,
)


@dataclass
class SimpleTask:
    task_id: str
    metric_name: str
    max_tokens: int
    _examples: list[Example]

    def load_split(self, split: str, limit: int | None = None) -> list[Example]:
        rows = [ex for ex in self._examples if ex.split == split]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def format_input(self, example: Example) -> str:
        return example.input

    def parse_output(self, raw_output: str) -> dict[str, Any]:
        if self.task_id == "gsm8k":
            answer = parse_first_number(raw_output)
            return {"answer": answer, "raw": raw_output}
        if self.task_id == "classification":
            lowered = raw_output.lower()
            labels = ["sports", "business", "science", "world"]
            for label in labels:
                if re.search(rf"\b{re.escape(label)}\b", lowered):
                    return {"answer": label, "raw": raw_output}
            return {"answer": lowered.strip().split()[0] if lowered.strip() else "", "raw": raw_output}
        if self.task_id in {"ifbench", "ifbench_hard", OFFICIAL_TASK_ID}:
            return {"answer": raw_output.strip(), "raw": raw_output}
        answer = raw_output.strip().splitlines()[0].strip()
        answer = re.sub(r"^(answer|final)\s*:\s*", "", answer, flags=re.I)
        return {"answer": answer, "raw": raw_output}

    def score(self, parsed: dict[str, Any], gold: Any) -> float:
        answer = "" if parsed.get("answer") is None else str(parsed.get("answer"))
        if self.task_id == "extraction_qa":
            return token_f1(answer, str(gold))
        if self.task_id in {"ifbench", "ifbench_hard"}:
            return score_local_ifbench_constraint(answer, gold)
        if self.task_id == OFFICIAL_TASK_ID:
            return score_official_ifbench_response(answer, gold)
        return exact_match(answer, str(gold))


def get_task(task_id: str) -> SimpleTask:
    if task_id == "gsm8k":
        return SimpleTask(task_id, "exact_match", 32, gsm8k_examples())
    if task_id == "bbh":
        return SimpleTask(task_id, "exact_match", 16, bbh_examples())
    if task_id == "classification":
        return SimpleTask(task_id, "accuracy", 12, classification_examples())
    if task_id == "extraction_qa":
        return SimpleTask(task_id, "token_f1", 16, extraction_qa_examples())
    if task_id == "ifbench":
        return SimpleTask(task_id, "constraint_accuracy", 96, ifbench_examples())
    if task_id == "ifbench_hard":
        return SimpleTask(task_id, "constraint_accuracy", 160, ifbench_hard_examples())
    if task_id == OFFICIAL_TASK_ID:
        examples = load_official_ifbench_examples()
        ensure_official_ifbench_evaluator_available(examples[0].gold if examples else None)
        return SimpleTask(task_id, "official_ifbench_loose_accuracy", 512, examples)
    raise ValueError(f"Unknown task_id: {task_id}")


def get_tasks(task_ids: list[str]) -> list[SimpleTask]:
    return [get_task(task_id) for task_id in task_ids]
