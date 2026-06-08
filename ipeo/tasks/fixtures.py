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


def ifbench_hard_examples(total: int = 180) -> list[Example]:
    """Harder local instruction-following stress fixture.

    Each item has multiple independently verifiable constraints. This is meant
    to expose cases where picking one whole source-best prompt can overfit a
    narrow formatting habit, while composing invariant edits can still help.
    """

    constraints = [
        {
            "kind": "compound_lines_keywords_suffix",
            "input": (
                "Hard IFBench: Write exactly 3 non-empty lines about climate adaptation. "
                "Include the keyword tide exactly 2 times across the entire response. "
                "End the final line with the exact token <DONE>. Do not use bullet symbols."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "line_count", "n": 3},
                    {"kind": "keyword_exact", "keyword": "tide", "n": 2},
                    {"kind": "suffix", "suffix": "<DONE>"},
                    {"kind": "forbidden_substrings", "substrings": ["- ", "* ", "1.", "1)"]},
                ],
            },
        },
        {
            "kind": "json_exact_fields",
            "input": (
                "Hard IFBench: Return only a JSON object with exactly the keys summary, risk, and action. "
                "The value for risk must be the string low. Do not wrap the JSON in markdown."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "json_keys", "keys": ["summary", "risk", "action"]},
                    {"kind": "json_value", "key": "risk", "value": "low"},
                    {"kind": "forbidden_substrings", "substrings": ["```", "json\n"]},
                ],
            },
        },
        {
            "kind": "word_count_uppercase_keyword",
            "input": (
                "Hard IFBench: Write exactly 6 words. Every alphabetic letter must be uppercase. "
                "Include the token FOCUS exactly once."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "word_count", "n": 6},
                    {"kind": "uppercase"},
                    {"kind": "keyword_exact", "keyword": "FOCUS", "n": 1},
                ],
            },
        },
        {
            "kind": "csv_row",
            "input": (
                "Hard IFBench: Return one CSV row with exactly 4 comma-separated fields and no header. "
                "The fields should be city, country, river, code. The code field must be ZX-7."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "csv_field_count", "n": 4},
                    {"kind": "csv_field_value", "index": 3, "value": "ZX-7"},
                    {"kind": "line_count", "n": 1},
                ],
            },
        },
        {
            "kind": "numbered_lines_keyword_counts",
            "input": (
                "Hard IFBench: Output exactly 4 numbered lines using the pattern '1)' through '4)'. "
                "Include apple exactly once and banana exactly twice across the entire response."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "numbered_lines", "n": 4, "style": "paren"},
                    {"kind": "keyword_exact", "keyword": "apple", "n": 1},
                    {"kind": "keyword_exact", "keyword": "banana", "n": 2},
                ],
            },
        },
        {
            "kind": "paragraph_suffix_forbidden",
            "input": (
                "Hard IFBench: Write exactly 2 paragraphs about release planning. "
                "The final paragraph must end with ENDCAP. Do not use the words maybe or soon."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "paragraph_count", "n": 2},
                    {"kind": "suffix", "suffix": "ENDCAP"},
                    {"kind": "forbidden_words", "words": ["maybe", "soon"]},
                ],
            },
        },
        {
            "kind": "first_last_word_count",
            "input": (
                "Hard IFBench: Write exactly 5 words. The first word must be Atlas and the last word must be omega."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "word_count", "n": 5},
                    {"kind": "starts_with", "prefix": "Atlas"},
                    {"kind": "ends_with_word", "word": "omega"},
                ],
            },
        },
        {
            "kind": "sentence_and_punctuation",
            "input": (
                "Hard IFBench: Write exactly 2 sentences about verification. "
                "Exactly one sentence must end with a question mark."
            ),
            "gold": {
                "kind": "all",
                "constraints": [
                    {"kind": "sentence_count", "n": 2},
                    {"kind": "terminal_question_count", "n": 1},
                ],
            },
        },
    ]
    rows: list[Example] = []
    for i in range(total):
        item = constraints[i % len(constraints)]
        rows.append(
            Example(
                example_id=f"ifbench-hard-{i:04d}",
                task_id="ifbench_hard",
                split=_split_for_index(i),
                input=item["input"],
                gold=item["gold"],
                meta={"constraint_kind": item["kind"], "stress_fixture": True},
            )
        )
    return rows
