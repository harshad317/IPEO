"""Public data structures for IPEO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EditType = Literal[
    "output_format",
    "reasoning_strategy",
    "evidence_use",
    "abstention_or_uncertainty",
    "label_mapping",
    "extraction_boundary",
    "verbosity_control",
    "decomposition",
    "verification",
    "cost_reduction",
    "generic_hygiene",
    "placebo",
    "other",
]

InsertionLocation = Literal[
    "system",
    "instruction",
    "examples",
    "output_schema",
    "rubric",
    "postprocessing_hint",
]


@dataclass(frozen=True)
class AtomicEdit:
    edit_id: str
    task_id: str
    edit_type: EditType
    natural_language_delta: str
    insertion_location: InsertionLocation
    estimated_token_delta: int
    parent_prompt_ids: list[str] = field(default_factory=list)
    parser_source: Literal["rule", "llm", "human_audit"] = "rule"
    is_generic: bool = False
    is_placebo: bool = False


@dataclass(frozen=True)
class PromptCandidate:
    prompt_id: str
    task_id: str
    text: str
    edit_ids: list[str]
    edit_vector: list[int]
    source_generator: Literal[
        "seed",
        "gepa_like",
        "mipro_like",
        "capo_like",
        "bo",
        "random",
        "manual",
        "promptbridge_emulation",
        "bedrock_emulation",
        "ipeo_composed",
    ]
    parent_prompt_ids: list[str] = field(default_factory=list)
    prompt_tokens_by_model: dict[str, int] = field(default_factory=dict)
    estimated_deployment_cost: dict[str, float] = field(default_factory=dict)
    coherence_repaired: bool = False
    frozen_pool_version: str = "mvp-v1"


@dataclass(frozen=True)
class Example:
    example_id: str
    task_id: str
    split: Literal["opt", "val", "test", "calibration"]
    input: str
    gold: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 64
    stop: list[str] | None = None
    system_prompt: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    provider_request_id: str | None
    timestamp: str
    model_version: str
    finish_reason: str | None = "stop"


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    task_id: str
    model_id: str
    prompt_id: str
    example_id: str
    split: Literal["opt", "val", "test", "calibration"]
    raw_output_path: str
    parsed_output: dict[str, Any]
    score: float
    parse_success: bool
    error_type: str | None = None


@dataclass(frozen=True)
class CostLog:
    run_id: str
    task_id: str
    model_id: str
    provider: str
    api_model_version: str
    method: str
    fold_id: str
    seed: int
    phase: Literal[
        "proposal",
        "reflection",
        "evaluation",
        "calibration",
        "repair",
        "final_test",
        "baseline_optimization",
    ]
    prompt_id: str | None
    edit_id: str | None
    example_id: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    api_price_input: float
    api_price_output: float
    dollar_cost: float
    latency_ms: int
    timestamp: str
    temperature: float
    max_tokens: int
    retry_count: int = 0
    cache_hit: bool = False


@dataclass(frozen=True)
class InvariantEditStats:
    task_id: str
    edit_id: str
    edit_type: str
    token_delta: int
    mean_effect: float
    effect_variance: float
    sign_agreement: float
    rank_stability: float
    lcb_mean_effect: float
    ipeo_score: float
    is_generic: bool
    is_placebo: bool
    per_model_effects: dict[str, float]


@dataclass(frozen=True)
class MethodSelection:
    method: str
    task_id: str
    fold_id: str
    target_model: str
    source_models: list[str]
    prompt_id: str
    prompt_text: str
    selected_edit_ids: list[str]
    target_calls: int = 0
    source_calls: int = 0
    total_dollars: float = 0.0
