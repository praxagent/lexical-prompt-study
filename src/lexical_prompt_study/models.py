from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactFile(StrictModel):
    path: str
    size_bytes: int
    git_blob_oid: str | None = None
    sha256: str | None = None


class Artifact(StrictModel):
    role: Literal["target_model", "tokenizer", "jacobian_lens", "sae", "evaluator", "dataset", "attack"]
    repository: str
    revision: str
    files: list[ArtifactFile]
    license: str
    source_url: str
    expected_model: str | None = None
    notes: str | None = None


class BehaviorRef(StrictModel):
    behavior_id: str
    category: str
    split: Literal["discovery", "confirmatory", "reserve", "utility"]
    content_sha256: str
    source: str


class Arm(StrictModel):
    arm_id: Literal["base", "full", "structural_sham", "inert_length"]
    scaffold_sha256: str
    token_count_rule: str


class StudyPlan(StrictModel):
    schema_version: Literal["1.0"]
    study_id: Literal["lexical-scaffold-llama33-70b-v1"]
    created_utc: str
    outcome_status: Literal["outcome-free"]
    split_seed: str
    followup_text_sha256: str
    primary_turn: Literal[2]
    decoding: dict
    evaluator: dict
    probes: dict
    thresholds: dict
    artifacts_manifest_sha256: str
    behaviors: list[BehaviorRef]
    arms: list[Arm]
    expected_trials: int = Field(gt=0)


class TrialReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: str
    plan_sha256: str
    run_id: str
    trial_id: str
    attempt: int = Field(ge=1)
    behavior_id: str
    category: str
    split: str
    arm: str
    turn: int
    seed: int
    prompt_sha256: str
    prompt_token_ids_sha256: str
    generated_token_ids: list[int]
    generated_text_sha256: str
    restricted_text_path: str | None
    restricted_artifact_sha256: str | None
    finish_reason: str
    generated_token_count: int
    refusal_detected: bool
    divider_detected: bool
    post_divider_present: bool
    truncated: bool
    evaluator_score: float | None
    evaluator_parse_ok: bool
    elapsed_seconds: float
    peak_memory_bytes: int | None
    model_revision: str
    tokenizer_revision: str
    source_commit: str
    software: dict
    runtime: dict
    error: str | None = None


class MechanismMargin(StrictModel):
    vocabulary_logit_mean: float
    vocabulary_logit_std: float = Field(gt=0)
    refusal_probe_mean_z: float
    compliance_probe_mean_z: float
    refusal_minus_compliance_margin: float


class MechanismReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: str
    public_plan_sha256: str
    source_commit: str
    run_id: str
    split: Literal["discovery", "confirmatory"]
    behavior_id: str
    arm: Literal["base", "full", "structural_sham", "inert_length"]
    turn: Literal[2]
    position: Literal["assistant_boundary", "generated"]
    position_token_index: int | None
    position_available: bool
    missing_position_reason: str | None
    prompt_token_ids_sha256: str
    prefix_token_ids_sha256: str
    transport: Literal["jacobian_lens", "identity", "random_gaussian"]
    layer: int = Field(ge=0)
    random_seed: int | None = None
    fitted_frobenius_norm: float | None = None
    realized_frobenius_norm: float | None = None
    refusal_probe_token_ids: list[int]
    compliance_probe_token_ids: list[int]
    margin: MechanismMargin | None
    model_revision: str
    tokenizer_revision: str
    lens_sha256: str
    sae_sha256: str
    observation_sha256: str
    runtime: dict
