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
    observation_id: str
    behavior_id: str
    category: str
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
    runtime: dict


class InterventionStep(StrictModel):
    generated_token_index: int = Field(ge=0)
    requested_delta_norm: float = Field(ge=0)
    realized_delta_norm: float = Field(ge=0)
    pre_residual_norm: float = Field(gt=0)
    post_residual_norm: float = Field(gt=0)
    requested_realized_relative_error: float = Field(ge=0)
    delta_to_pre_residual_norm: float = Field(ge=0)
    clipped: bool


class InterventionReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: str
    intervention_plan_sha256: str
    public_plan_sha256: str
    source_commit: str
    run_id: str
    phase: Literal["discovery_calibration", "confirmatory_attack", "utility"]
    trial_id: str
    attempt: int = Field(ge=1)
    behavior_id: str
    category: str
    split: Literal["discovery", "confirmatory", "utility"]
    arm: Literal["full", "base"]
    turn: Literal[1, 2]
    condition_id: str
    direction_kind: Literal["zero", "primary", "matched_sae", "isotropic"]
    feature_id: int | None = Field(default=None, ge=0)
    isotropic_seed: int | None = None
    direction_sha256: str | None = None
    requested_sign: Literal[-1, 0, 1]
    requested_alpha: float = Field(ge=0)
    calibration_rho: float | None = Field(default=None, gt=0)
    calibration_reference_norm: float | None = Field(default=None, gt=0)
    layer: Literal[50]
    hook: Literal["residual_post"]
    positions: Literal["current_final_token_on_every_decoding_step"]
    prompt_sha256: str
    prompt_token_ids_sha256: str
    parent_generation_receipt_sha256: str | None = None
    generated_token_ids: list[int]
    generated_text_sha256: str
    restricted_text_path: str
    restricted_artifact_sha256: str
    finish_reason: Literal["eos", "length", "error"]
    generated_token_count: int = Field(ge=0)
    refusal_detected: bool
    truncated: bool
    evaluator_score: float | None = Field(default=None, ge=0, le=1)
    evaluator_parse_ok: bool
    intervention_steps: list[InterventionStep]
    model_revision: str
    tokenizer_revision: str
    sae_sha256: str
    elapsed_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    software: dict
    runtime: dict
    error: str | None = None


class FollowupQualificationReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-followup-v2"]
    plan_sha256: str
    run_id: str
    source_commit: str
    qualification_kind: Literal["synthetic_safe"]
    status: Literal["complete"]
    model_revision: str
    tokenizer_revision: str
    lens_sha256: str
    sae_sha256: str
    render_validation: dict
    placements: list[dict]
    pipeline_checks: dict
    model_topology: dict
    elapsed_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    private_bundle_path: str
    private_bundle_sha256: str
    software: dict


class FollowupTrialReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-followup-v2"]
    plan_sha256: str
    private_plan_sha256: str
    source_commit: str
    run_id: str
    trial_id: str
    partition: Literal[
        "discovery",
        "calibration",
        "confirmatory",
        "adaptive_stress",
        "utility_calibration",
        "utility_confirmatory",
    ]
    behavior_id: str
    category: str
    arm: Literal[
        "base",
        "inert_length",
        "structural_sham",
        "full",
        "ordinary_benign",
        "structured_benign",
    ]
    placement: Literal["ep_before_request", "ep_after_request"] | None
    stratum: str
    shared_reference: bool
    request_sha256: str
    prompt_sha256: str
    prompt_token_ids_sha256: str
    render_validation_sha256: str | None
    generated_text_sha256: str
    generated_token_ids_sha256: str
    generated_token_count: int = Field(ge=0)
    finish_reason: Literal["eos", "length"]
    truncated: bool
    refusal_detected: bool
    divider_detected: bool
    post_divider_present: bool
    restricted_artifact_path: str
    restricted_artifact_sha256: str
    state_bundle_path: str
    state_bundle_sha256: str
    capture_layers: list[int]
    state_shape: list[int]
    state_dtype: Literal["torch.bfloat16"]
    generation_elapsed_seconds: float = Field(ge=0)
    capture_elapsed_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    model_revision: str
    tokenizer_revision: str
    lens_sha256: str
    sae_sha256: str
    software: dict
