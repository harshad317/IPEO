"""Deterministic offline task fixtures."""

from __future__ import annotations

from ipeo.core.schemas import Example


def _split_for_index(idx: int) -> str:
    cycle = idx % 3
    if cycle == 0:
        return "opt"
    if cycle == 1:
        return "val"
    return "test"


def gsm8k_examples(total: int = 180) -> list[Example]:
    rows: list[Example] = []
    for i in range(total):
        a = 2 + (i % 17)
        b = 3 + ((i * 5) % 19)
        c = 1 + ((i * 7) % 11)
        answer = a * b + c
        rows.append(
            Example(
                example_id=f"gsm8k-{i:04d}",
                task_id="gsm8k",
                split=_split_for_index(i),
                input=f"Maya buys {a} packs with {b} stickers each, then finds {c} more. How many stickers does she have?",
                gold=str(answer),
                meta={"operation": "multiply_add", "answer": answer},
            )
        )
    return rows


def bbh_examples(total: int = 180) -> list[Example]:
    rows: list[Example] = []
    for i in range(total):
        day = i % 7
        offset = (i * 3 + 2) % 7
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        answer = names[(day + offset) % 7]
        rows.append(
            Example(
                example_id=f"bbh-{i:04d}",
                task_id="bbh",
                split=_split_for_index(i),
                input=f"If today is {names[day]} and an event is {offset} days later, what day is the event?",
                gold=answer,
                meta={"subtask": "date_understanding"},
            )
        )
    return rows


def classification_examples(total: int = 180) -> list[Example]:
    labels = ["sports", "business", "science", "world"]
    templates = {
        "sports": "The team won the final after a late goal and a strong defense.",
        "business": "The company reported revenue growth after a new market launch.",
        "science": "Researchers measured a new signal in the lab experiment.",
        "world": "Leaders met to discuss the treaty after regional talks.",
    }
    rows: list[Example] = []
    for i in range(total):
        label = labels[i % len(labels)]
        rows.append(
            Example(
                example_id=f"classification-{i:04d}",
                task_id="classification",
                split=_split_for_index(i),
                input=templates[label],
                gold=label,
                meta={"labels": labels},
            )
        )
    return rows


def extraction_qa_examples(total: int = 180) -> list[Example]:
    people = ["Ada", "Grace", "Katherine", "Alan", "Barbara", "Edsger"]
    places = ["London", "Paris", "Delhi", "Tokyo", "Nairobi", "Toronto"]
    rows: list[Example] = []
    for i in range(total):
        person = people[i % len(people)]
        place = places[(i * 2 + 1) % len(places)]
        rows.append(
            Example(
                example_id=f"extraction-qa-{i:04d}",
                task_id="extraction_qa",
                split=_split_for_index(i),
                input=f"Context: {person} presented the prototype in {place} during the annual summit.\nQuestion: Where was the prototype presented?",
                gold=place,
                meta={"answer_type": "location"},
            )
        )
    return rows


def ifbench_examples(total: int = 180) -> list[Example]:
    """Small verifiable instruction-following fixture inspired by IFBench.

    The official IFBench dataset ships many more constraints and verifier
    functions. This local fixture keeps the same spirit for MVP benchmarking:
    every example is scored by a deterministic verifier, not an LLM judge.
    """

    constraints = [
        {
            "kind": "word_count",
            "input": "Write a response about careful science. Constraint: use exactly 3 words.",
            "gold": {"kind": "word_count", "n": 3},
        },
        {
            "kind": "keyword_exact",
            "input": "Write one sentence about coral reefs. Constraint: include the word coral exactly 2 times.",
            "gold": {"kind": "keyword_exact", "keyword": "coral", "n": 2},
        },
        {
            "kind": "line_count",
            "input": "List colors. Constraint: output exactly 3 non-empty lines.",
            "gold": {"kind": "line_count", "n": 3},
        },
        {
            "kind": "uppercase",
            "input": "Write a short motto about focus. Constraint: all alphabetic letters must be uppercase.",
            "gold": {"kind": "uppercase"},
        },
        {
            "kind": "suffix",
            "input": "Write a short answer about planning. Constraint: end the response with the exact token <END>.",
            "gold": {"kind": "suffix", "suffix": "<END>"},
        },
        {
            "kind": "json_keys",
            "input": "Answer with a JSON object. Constraint: use exactly the keys answer and confidence.",
            "gold": {"kind": "json_keys", "keys": ["answer", "confidence"]},
        },
    ]
    rows: list[Example] = []
    for i in range(total):
        item = constraints[i % len(constraints)]
        rows.append(
            Example(
                example_id=f"ifbench-{i:04d}",
                task_id="ifbench",
                split=_split_for_index(i),
                input=item["input"],
                gold=item["gold"],
                meta={"constraint_kind": item["kind"]},
            )
        )
    return rows
