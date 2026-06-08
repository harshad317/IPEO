"""Seed prompts for offline MVP tasks."""

from __future__ import annotations


def seed_prompt_texts(task_id: str) -> list[str]:
    common = [
        "Solve the task carefully. Answer the user request.",
        "Read the input, identify the requested answer, and respond with only the final answer. OUTPUT_ONLY",
        "Follow the instruction exactly and keep the response concise.",
    ]
    task_specific = {
        "gsm8k": [
            "Solve the math word problem step by step internally. Respond with only the final number. OUTPUT_ONLY",
            "Compute the arithmetic carefully and verify the final numeric answer before responding.",
        ],
        "bbh": [
            "Track the date offset carefully. Respond with only the weekday name. OUTPUT_ONLY",
            "Decompose the symbolic reasoning problem and verify the final answer.",
        ],
        "classification": [
            "Classify the text into exactly one label: sports, business, science, or world. OUTPUT_ONLY",
            "Map evidence words to the allowed label set and answer with only the label.",
        ],
        "extraction_qa": [
            "Answer the question using only the provided context. Respond with the shortest supported span. OUTPUT_ONLY",
            "Find the exact evidence span in the context and output only that answer.",
        ],
        "ifbench": [
            "Follow every explicit constraint exactly. Output only the requested response, with no explanation.",
            "Before answering, check formatting, counts, required words, and ending tokens internally. Then provide only the final compliant answer.",
        ],
        "ifbench_hard": [
            "Follow every explicit constraint as a hard requirement. Build the answer internally, verify each constraint, then output only the final response.",
            "Treat counts, line boundaries, JSON/CSV structure, forbidden words, and suffix tokens as exact checks. Do not include any explanation.",
        ],
        "ifbench_official": [
            "Follow every explicit instruction and constraint exactly. Output only the requested response, with no explanation.",
            "Before answering, verify keyword counts, formatting, lengths, ordering, and forbidden content internally. Then provide only the final compliant answer.",
        ],
    }
    return common + task_specific[task_id]
