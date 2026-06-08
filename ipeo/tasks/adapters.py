"""Offline task adapters."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any

from ipeo.core.schemas import Example
from ipeo.tasks.base import exact_match, parse_first_number, token_f1
from ipeo.tasks.fixtures import bbh_examples, classification_examples, extraction_qa_examples, gsm8k_examples, ifbench_examples


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
        if self.task_id == "ifbench":
            return {"answer": raw_output.strip(), "raw": raw_output}
        answer = raw_output.strip().splitlines()[0].strip()
        answer = re.sub(r"^(answer|final)\s*:\s*", "", answer, flags=re.I)
        return {"answer": answer, "raw": raw_output}

    def score(self, parsed: dict[str, Any], gold: Any) -> float:
        answer = "" if parsed.get("answer") is None else str(parsed.get("answer"))
        if self.task_id == "extraction_qa":
            return token_f1(answer, str(gold))
        if self.task_id == "ifbench":
            return score_ifbench_response(answer, gold)
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
    raise ValueError(f"Unknown task_id: {task_id}")


def get_tasks(task_ids: list[str]) -> list[SimpleTask]:
    return [get_task(task_id) for task_id in task_ids]


def score_ifbench_response(answer: str, gold: Any) -> float:
    if not isinstance(gold, dict):
        return 0.0
    kind = gold.get("kind")
    stripped = answer.strip()
    if kind == "word_count":
        words = re.findall(r"\b[\w'<>\-]+\b", stripped)
        return float(len(words) == int(gold["n"]))
    if kind == "keyword_exact":
        keyword = str(gold["keyword"]).lower()
        count = len(re.findall(rf"\b{re.escape(keyword)}\b", stripped.lower()))
        return float(count == int(gold["n"]))
    if kind == "line_count":
        lines = [line for line in stripped.splitlines() if line.strip()]
        return float(len(lines) == int(gold["n"]))
    if kind == "uppercase":
        letters = [ch for ch in stripped if ch.isalpha()]
        return float(bool(letters) and all(ch.upper() == ch for ch in letters))
    if kind == "suffix":
        return float(stripped.endswith(str(gold["suffix"])))
    if kind == "json_keys":
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return 0.0
        return float(isinstance(obj, dict) and sorted(obj.keys()) == sorted(gold["keys"]))
    return 0.0
