"""Prompt composition utilities."""

from __future__ import annotations

from ipeo.core.schemas import AtomicEdit, PromptCandidate
from ipeo.models.base import count_tokens


CONFLICTING_EDIT_TYPES = {
    frozenset({"verbosity_control", "reasoning_strategy"}),
}


def edit_marker(edit: AtomicEdit) -> str:
    return f"[EDIT:{edit.edit_type}:{edit.edit_id}]"


def render_edit(edit: AtomicEdit) -> str:
    return f"{edit_marker(edit)} {edit.natural_language_delta}"


def compose_text(seed_text: str, edits: list[AtomicEdit]) -> str:
    if not edits:
        return seed_text
    instruction_lines = [seed_text.rstrip(), "", "Additional constraints:"]
    for edit in edits:
        instruction_lines.append(f"- {render_edit(edit)}")
    return "\n".join(instruction_lines).strip()


def has_conflict(candidate: AtomicEdit, selected: list[AtomicEdit]) -> bool:
    candidate_type = candidate.edit_type
    for edit in selected:
        if frozenset({candidate_type, edit.edit_type}) in CONFLICTING_EDIT_TYPES:
            return True
    return False


def prompt_token_count(prompt: PromptCandidate) -> int:
    return count_tokens(prompt.text)
