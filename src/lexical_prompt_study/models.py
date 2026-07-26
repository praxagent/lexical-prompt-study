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


class FollowupPatchReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-followup-v2"]
    public_plan_sha256: str
    patch_private_plan_sha256: str
    source_commit: str
    run_id: str
    stage: Literal["coarse_discovery", "coarse_calibration"]
    partition: Literal["discovery", "calibration"]
    trial_id: str
    behavior_id: str
    category: str
    placement: Literal["ep_before_request", "ep_after_request"]
    candidate_layer: int = Field(ge=0, le=31)
    applied_layer: int = Field(ge=0, le=31)
    condition: Literal[
        "sham_into_full",
        "full_into_sham",
        "full_into_full_identity",
        "sham_into_sham_identity",
        "no_op_hook",
        "same_site_magnitude_matched_seeded_random_delta",
        "irrelevant_layer",
        "irrelevant_token_position",
        "cross_behavior_category_and_length_matched_donor",
    ]
    recipient_arm: Literal["full", "structural_sham"]
    donor_arm: Literal["full", "structural_sham"] | None
    donor_behavior_id: str | None
    baseline_arm: Literal["full", "structural_sham"]
    token_offset: Literal[-2, -1]
    recipient_generation_receipt_sha256: str
    donor_state_bundle_sha256: str | None
    prompt_token_ids_sha256: str
    recipient_pre_patch_sha256: str
    realized_delta_sha256: str
    recipient_pre_patch_norm: float = Field(gt=0)
    realized_delta_norm: float = Field(ge=0)
    tensor_shape: list[int]
    tensor_dtype: Literal["torch.bfloat16"]
    replay_bundle_path: str
    replay_bundle_sha256: str
    generated_text_sha256: str
    generated_token_ids_sha256: str
    generated_token_count: int = Field(ge=0)
    restricted_artifact_path: str
    restricted_artifact_sha256: str
    finish_reason: Literal["eos", "length"]
    truncated: bool
    elapsed_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    model_revision: str
    tokenizer_revision: str
    software: dict


class FactorialPrivatePlanReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-8b-factorial-v1"]
    status: Literal["factorial_private_plan_complete_no_target_outcomes"]
    source_commit: str
    public_plan_sha256: str
    material_source_sha256: str
    private_plan_sha256: str
    tokenizer_revision: str
    tokenizer_chat_template_sha256: str
    request_panel_counts: dict[str, int]
    request_panel_sha256: dict[str, str]
    material_block_counts: dict[str, int]
    material_canonical_sha256: dict[str, str]
    realized_doses: list[dict]
    canonical_observation_count: Literal[422]
    additional_dose_observation_count: int = Field(ge=0, le=540)
    total_observation_count: int = Field(ge=422, le=962)
    placement_levels: list[
        Literal["ep_before_request", "ep_after_request"]
    ]
    exact_size_matching_passed: Literal[True]
    raw_prompt_or_token_ids_public: Literal[False]
    target_generation_performed: Literal[False]
    target_outcome_exists: Literal[False]


class FactorialAssayReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-8b-factorial-v1"]
    status: Literal["passed", "failed"]
    qualification_kind: Literal["noninferential_legacy_canary"]
    public_plan_sha256: str
    private_plan_sha256: str
    source_commit: str
    run_id: str
    model_revision: str
    tokenizer_revision: str
    lens_sha256: str
    sae_sha256: str
    selected_feature_id: Literal[6779]
    frozen_subspace_feature_ids: list[int]
    planned_canary_conditions: Literal[8]
    completed_canary_conditions: int = Field(ge=0, le=8)
    exact_identity_checks: dict[str, bool]
    final_render_checks: dict[str, bool]
    deterministic_rerun_passed: bool
    reconstruction_metric: str
    reconstruction_absolute_tolerance: float = Field(gt=0)
    reconstruction_cross_condition_tolerance: float = Field(gt=0)
    maximum_reconstruction_error: float = Field(ge=0)
    reconstruction_error_range: float = Field(ge=0)
    qualitative_ordering_passed_by_placement: dict[str, bool]
    private_bundle_path: str
    private_bundle_sha256: str
    elapsed_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    target_factorial_outcome_generated: Literal[False]
    software: dict


class FactorialTrialReceipt(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: Literal["lexical-scaffold-8b-factorial-v1"]
    status: Literal["complete"]
    public_plan_sha256: str
    private_plan_sha256: str
    assay_receipt_sha256: str
    source_commit: str
    run_id: str
    trial_id: str
    attempt: int = Field(ge=1)
    request_class: Literal[
        "minimal_neutral_carrier",
        "ordinary_benign_request",
        "harmful_request",
        "literal_sentinel",
    ]
    request_id: str
    prompt_family_id: str
    request_sha256: str
    material: Literal[
        "no_scaffold",
        "inert_length",
        "structural_sham",
        "full_scaffold",
    ]
    placement: Literal["ep_before_request", "ep_after_request"] | None
    size_id: str
    injected_token_count: int = Field(ge=0)
    shared_reference: bool
    render_group_sha256: str | None
    prompt_sha256: str
    prompt_token_ids_sha256: str
    generated_text_sha256: str
    generated_token_ids_sha256: str
    generated_token_count: int = Field(ge=0, le=1024)
    finish_reason: Literal["eos", "length"]
    truncated: bool
    feature_6779_magnitude: float = Field(ge=0)
    feature_6779_positive: bool
    frozen_subspace_score: float
    sae_reconstruction_relative_error: float = Field(ge=0)
    assistant_boundary_jlens_margin: float
    refusal_detected: bool | None
    task_completion: bool | None
    utility_score: float | None
    phase_parser_result: dict
    restricted_artifact_path: str
    restricted_artifact_sha256: str
    generation_elapsed_seconds: float = Field(ge=0)
    readout_elapsed_seconds: float = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    model_revision: str
    tokenizer_revision: str
    lens_sha256: str
    sae_sha256: str
    software: dict
