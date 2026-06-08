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
    }
    return common + task_specific[task_id]
