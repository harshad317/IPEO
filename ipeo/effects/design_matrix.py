"""Edit design matrix helpers."""

from __future__ import annotations

import numpy as np

from ipeo.core.schemas import AtomicEdit, PromptCandidate


def build_edit_matrix(pool: list[PromptCandidate], edits: list[AtomicEdit]) -> np.ndarray:
    edit_index = {edit.edit_id: idx for idx, edit in enumerate(edits)}
    matrix = np.zeros((len(pool), len(edits)), dtype=float)
    for row_idx, prompt in enumerate(pool):
        for edit_id in prompt.edit_ids:
            if edit_id in edit_index:
                matrix[row_idx, edit_index[edit_id]] = 1.0
    return matrix


def prompt_index(pool: list[PromptCandidate]) -> dict[str, int]:
    return {prompt.prompt_id: idx for idx, prompt in enumerate(pool)}
