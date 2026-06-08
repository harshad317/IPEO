"""Frozen prompt/edit pool construction."""

from __future__ import annotations

import itertools
import random
from dataclasses import replace
from pathlib import Path

from ipeo.core.ids import stable_hash
from ipeo.core.io import write_jsonl
from ipeo.core.schemas import AtomicEdit, PromptCandidate
from ipeo.models.base import count_tokens
from ipeo.prompts.composer import compose_text
from ipeo.prompts.seed_prompts import seed_prompt_texts


BASE_EDIT_SPECS = [
    ("output_format", "Respond with only the final answer and no explanation.", "instruction", 9, False, False),
    ("reasoning_strategy", "Work through the reasoning internally before selecting the final answer.", "instruction", 11, False, False),
    ("verification", "Check the answer against the input once before finalizing it.", "rubric", 10, False, False),
    ("generic_hygiene", "Be accurate, concise, and clear.", "instruction", 5, True, False),
    ("cost_reduction", "Use the shortest valid response that satisfies the task.", "instruction", -4, False, False),
    ("placebo", "Remember that blue notebooks can be useful.", "instruction", 7, False, True),
]

TASK_EDIT_SPECS = {
    "gsm8k": [
        ("reasoning_strategy", "Translate the word problem into arithmetic before calculating.", "instruction", 10, False, False),
        ("verification", "Recompute the arithmetic using a second pass.", "rubric", 8, False, False),
    ],
    "bbh": [
        ("decomposition", "Break the symbolic rule into ordered intermediate steps.", "instruction", 9, False, False),
        ("output_format", "For date tasks, output exactly one weekday name.", "output_schema", 8, False, False),
    ],
    "classification": [
        ("label_mapping", "Choose exactly one label from sports, business, science, world.", "output_schema", 10, False, False),
        ("verbosity_control", "Do not include rationale or extra words after the label.", "output_schema", 8, False, False),
    ],
    "extraction_qa": [
        ("evidence_use", "Use only words supported by the provided context.", "rubric", 9, False, False),
        ("extraction_boundary", "Return the minimal answer span, not a full sentence.", "output_schema", 9, False, False),
    ],
    "ifbench": [
        ("output_format", "Satisfy all visible formatting constraints exactly.", "output_schema", 8, False, False),
        ("verification", "Check counts, casing, required tokens, and JSON keys before finalizing.", "rubric", 11, False, False),
        ("verbosity_control", "Do not add explanations, apologies, markdown fences, or extra text.", "output_schema", 10, False, False),
    ],
}


def _make_edit(task_id: str, spec: tuple[str, str, str, int, bool, bool], parent_prompt_ids: list[str]) -> AtomicEdit:
    edit_type, delta, location, token_delta, is_generic, is_placebo = spec
    edit_id = stable_hash({"task": task_id, "type": edit_type, "delta": delta}, prefix="e-")
    return AtomicEdit(
        edit_id=edit_id,
        task_id=task_id,
        edit_type=edit_type,  # type: ignore[arg-type]
        natural_language_delta=delta,
        insertion_location=location,  # type: ignore[arg-type]
        estimated_token_delta=token_delta,
        parent_prompt_ids=parent_prompt_ids,
        parser_source="rule",
        is_generic=is_generic,
        is_placebo=is_placebo,
    )


def build_edits(task_id: str, seed_prompt_ids: list[str]) -> list[AtomicEdit]:
    specs = BASE_EDIT_SPECS + TASK_EDIT_SPECS[task_id]
    edits = [_make_edit(task_id, spec, seed_prompt_ids) for spec in specs]
    seen: set[str] = set()
    unique: list[AtomicEdit] = []
    for edit in edits:
        if edit.edit_id not in seen:
            unique.append(edit)
            seen.add(edit.edit_id)
    return unique


def _candidate(
    *,
    task_id: str,
    seed_text: str,
    all_edits: list[AtomicEdit],
    selected_edits: list[AtomicEdit],
    source_generator: str,
    parent_prompt_ids: list[str],
) -> PromptCandidate:
    text = compose_text(seed_text, selected_edits)
    prompt_id = stable_hash(
        {"task": task_id, "text": text, "edits": [edit.edit_id for edit in selected_edits]},
        prefix="p-",
    )
    edit_ids = [edit.edit_id for edit in selected_edits]
    vector = [1 if edit.edit_id in edit_ids else 0 for edit in all_edits]
    return PromptCandidate(
        prompt_id=prompt_id,
        task_id=task_id,
        text=text,
        edit_ids=edit_ids,
        edit_vector=vector,
        source_generator=source_generator,  # type: ignore[arg-type]
        parent_prompt_ids=parent_prompt_ids,
        prompt_tokens_by_model={"mock": count_tokens(text)},
        estimated_deployment_cost={"mock": count_tokens(text) * 0.0001 / 1000},
        coherence_repaired=False,
        frozen_pool_version="mvp-v1",
    )


def build_frozen_pool(
    task_id: str,
    *,
    num_prompts: int,
    seed: int = 0,
    artifact_dir: str | Path = "artifacts",
) -> tuple[list[PromptCandidate], list[AtomicEdit]]:
    rng = random.Random(seed)
    seed_texts = seed_prompt_texts(task_id)
    seed_ids = [stable_hash({"task": task_id, "seed": text}, prefix="p-") for text in seed_texts]
    placeholder_edits = build_edits(task_id, seed_ids)
    pool: list[PromptCandidate] = []
    for text, prompt_id in zip(seed_texts, seed_ids):
        candidate = _candidate(
            task_id=task_id,
            seed_text=text,
            all_edits=placeholder_edits,
            selected_edits=[],
            source_generator="seed",
            parent_prompt_ids=[],
        )
        pool.append(replace(candidate, prompt_id=prompt_id))

    edits = build_edits(task_id, [p.prompt_id for p in pool])
    pool = [replace(prompt, edit_vector=[0] * len(edits)) for prompt in pool]
    primary_seed = seed_texts[0]
    for edit in edits:
        generator = "capo_like" if edit.edit_type == "cost_reduction" else "gepa_like"
        if edit.edit_type in {"output_format", "label_mapping"}:
            generator = "mipro_like"
        if edit.is_placebo:
            generator = "random"
        pool.append(
            _candidate(
                task_id=task_id,
                seed_text=primary_seed,
                all_edits=edits,
                selected_edits=[edit],
                source_generator=generator,
                parent_prompt_ids=[pool[0].prompt_id],
            )
        )

    non_placebo = [edit for edit in edits if not edit.is_placebo]
    for combo_size in (2, 3):
        combos = list(itertools.combinations(non_placebo, combo_size))
        rng.shuffle(combos)
        for combo in combos[: max(2, num_prompts // 4)]:
            pool.append(
                _candidate(
                    task_id=task_id,
                    seed_text=primary_seed,
                    all_edits=edits,
                    selected_edits=list(combo),
                    source_generator="random",
                    parent_prompt_ids=[pool[0].prompt_id],
                )
            )

    short_edits = sorted(non_placebo, key=lambda e: e.estimated_token_delta)[:3]
    pool.append(
        _candidate(
            task_id=task_id,
            seed_text=seed_texts[2],
            all_edits=edits,
            selected_edits=short_edits,
            source_generator="capo_like",
            parent_prompt_ids=[pool[2].prompt_id],
        )
    )

    seen: set[str] = set()
    unique_pool: list[PromptCandidate] = []
    for prompt in pool:
        if prompt.prompt_id in seen:
            continue
        unique_pool.append(prompt)
        seen.add(prompt.prompt_id)
        if len(unique_pool) >= num_prompts:
            break

    artifact_root = Path(artifact_dir)
    write_jsonl(artifact_root / "prompts" / f"{task_id}_pool.jsonl", unique_pool)
    write_jsonl(artifact_root / "edits" / f"{task_id}_edits.jsonl", edits)
    return unique_pool, edits
