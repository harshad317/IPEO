"""Optional adapter for the official AllenAI IFBench evaluator."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from ipeo.core.schemas import Example


OFFICIAL_TASK_ID = "ifbench_official"


def _split_for_index(idx: int) -> str:
    cycle = idx % 3
    if cycle == 0:
        return "opt"
    if cycle == 1:
        return "val"
    return "test"


def _repo_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("IFBENCH_REPO"):
        candidates.append(Path(os.environ["IFBENCH_REPO"]))
    for path in [
        Path("external/IFBench"),
        Path("IFBench"),
        Path("../IFBench"),
    ]:
        candidates.append(path)
    return candidates


def _data_path_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("IFBENCH_DATA_PATH"):
        candidates.append(Path(os.environ["IFBENCH_DATA_PATH"]))
    for repo in _repo_candidates():
        candidates.append(repo / "data" / "IFBench_test.jsonl")
    candidates.extend([Path("data/IFBench_test.jsonl"), Path("IFBench_test.jsonl")])
    return candidates


def _find_official_data_path() -> Path:
    for candidate in _data_path_candidates():
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _data_path_candidates())
    raise RuntimeError(
        "Official IFBench data was not found. Clone the AllenAI IFBench repo and set "
        "IFBENCH_REPO=/path/to/IFBench, or set IFBENCH_DATA_PATH=/path/to/IFBench_test.jsonl. "
        f"Searched: {searched}"
    )


def _infer_repo_from_data_path(data_path: Path) -> Path | None:
    parent = data_path.parent
    if parent.name == "data" and (parent.parent / "evaluation_lib.py").exists():
        return parent.parent.resolve()
    return None


def _find_official_repo(repo_hint: str | None = None) -> Path:
    if repo_hint:
        candidate = Path(repo_hint)
        if (candidate / "evaluation_lib.py").exists():
            return candidate.resolve()
    for candidate in _repo_candidates():
        if (candidate / "evaluation_lib.py").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Official IFBench evaluator was not found. Clone https://github.com/allenai/IFBench "
        "and set IFBENCH_REPO=/path/to/IFBench. If dependencies are missing, install them with "
        "python -m pip install -r $IFBENCH_REPO/requirements.txt."
    )


@contextlib.contextmanager
def _prepended_syspath(path: Path) -> Iterator[None]:
    text = str(path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(text)


@lru_cache(maxsize=4)
def _load_official_eval_lib(repo_text: str) -> ModuleType:
    repo = Path(repo_text).resolve()
    eval_path = repo / "evaluation_lib.py"
    spec = importlib.util.spec_from_file_location(f"_ipeo_ifbench_eval_{abs(hash(str(repo)))}", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official IFBench evaluator from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        with _prepended_syspath(repo):
            spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Official IFBench evaluator dependencies are missing. Install them with "
            "python -m pip install -r $IFBENCH_REPO/requirements.txt."
        ) from exc
    return module


def load_official_ifbench_examples(limit: int | None = None) -> list[Example]:
    """Load official IFBench examples from a local AllenAI data file.

    The full official dataset is intentionally not vendored. Use IFBENCH_REPO
    or IFBENCH_DATA_PATH to point this adapter at the upstream JSONL file.
    """

    data_path = _find_official_data_path()
    repo = _infer_repo_from_data_path(data_path)
    rows: list[Example] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("key", idx))
            prompt = str(item["prompt"])
            instruction_ids = list(item["instruction_id_list"])
            kwargs = [dict(value) for value in item["kwargs"]]
            gold = {
                "kind": OFFICIAL_TASK_ID,
                "key": key,
                "prompt": prompt,
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs,
                "eval_mode": os.environ.get("IFBENCH_EVAL_MODE", "loose").lower(),
                "ifbench_repo": str(repo) if repo is not None else None,
            }
            rows.append(
                Example(
                    example_id=f"ifbench-official-{key}",
                    task_id=OFFICIAL_TASK_ID,
                    split=_split_for_index(idx),  # type: ignore[arg-type]
                    input=prompt,
                    gold=gold,
                    meta={
                        "source_path": str(data_path),
                        "instruction_id_list": instruction_ids,
                        "official_ifbench": True,
                    },
                )
            )
    return rows


def score_official_ifbench_response(answer: str, gold: Any) -> float:
    if not isinstance(gold, dict) or gold.get("kind") != OFFICIAL_TASK_ID:
        return 0.0
    repo = _find_official_repo(gold.get("ifbench_repo"))
    eval_lib = _load_official_eval_lib(str(repo))
    inp = eval_lib.InputExample(
        key=gold.get("key", "0"),
        instruction_id_list=list(gold["instruction_id_list"]),
        prompt=str(gold["prompt"]),
        kwargs=[dict(value) for value in gold["kwargs"]],
    )
    prompt_to_response = {inp.prompt: answer}
    mode = str(gold.get("eval_mode", "loose")).lower()
    if mode == "strict":
        output = eval_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        output = eval_lib.test_instruction_following_loose(inp, prompt_to_response)
    return float(bool(output.follow_all_instructions))


def ensure_official_ifbench_evaluator_available(gold: Any | None = None) -> None:
    repo_hint = None
    if isinstance(gold, dict):
        repo_hint = gold.get("ifbench_repo")
    repo = _find_official_repo(repo_hint)
    _load_official_eval_lib(str(repo))


def score_local_ifbench_constraint(answer: str, gold: Any) -> float:
    """Score local MVP and hard-fixture IFBench constraints."""

    if not isinstance(gold, dict):
        return 0.0
    stripped = answer.strip()
    kind = gold.get("kind")
    if kind == "all":
        constraints = list(gold.get("constraints", []))
        if not constraints:
            return 0.0
        scores = [score_local_ifbench_constraint(stripped, constraint) for constraint in constraints]
        return float(all(score == 1.0 for score in scores))
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
    if kind == "paragraph_count":
        paragraphs = [part for part in re.split(r"\n\s*\n", stripped) if part.strip()]
        return float(len(paragraphs) == int(gold["n"]))
    if kind == "uppercase":
        letters = [ch for ch in stripped if ch.isalpha()]
        return float(bool(letters) and all(ch.upper() == ch for ch in letters))
    if kind == "suffix":
        return float(stripped.endswith(str(gold["suffix"])))
    if kind == "starts_with":
        return float(stripped.startswith(str(gold["prefix"])))
    if kind == "ends_with_word":
        words = re.findall(r"\b[\w'<>\-]+\b", stripped)
        return float(bool(words) and words[-1].lower() == str(gold["word"]).lower())
    if kind == "forbidden_substrings":
        return float(not any(str(value) in stripped for value in gold.get("substrings", [])))
    if kind == "forbidden_words":
        lowered_words = [word.lower() for word in re.findall(r"\b[\w'<>\-]+\b", stripped)]
        forbidden = {str(word).lower() for word in gold.get("words", [])}
        return float(not any(word in forbidden for word in lowered_words))
    if kind == "json_keys":
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return 0.0
        return float(isinstance(obj, dict) and sorted(obj.keys()) == sorted(gold["keys"]))
    if kind == "json_value":
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return 0.0
        return float(isinstance(obj, dict) and obj.get(str(gold["key"])) == gold["value"])
    if kind == "csv_field_count":
        rows = list(csv.reader([stripped]))
        return float(len(rows) == 1 and len(rows[0]) == int(gold["n"]))
    if kind == "csv_field_value":
        rows = list(csv.reader([stripped]))
        index = int(gold["index"])
        return float(len(rows) == 1 and len(rows[0]) > index and rows[0][index].strip() == str(gold["value"]))
    if kind == "numbered_lines":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        n = int(gold["n"])
        if len(lines) != n:
            return 0.0
        if gold.get("style") == "paren":
            return float(all(line.startswith(f"{idx})") for idx, line in enumerate(lines, start=1)))
        return float(all(line.startswith(f"{idx}.") for idx, line in enumerate(lines, start=1)))
    if kind == "sentence_count":
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
        return float(len(sentences) == int(gold["n"]))
    if kind == "terminal_question_count":
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
        return float(sum(sentence.endswith("?") for sentence in sentences) == int(gold["n"]))
    return 0.0
