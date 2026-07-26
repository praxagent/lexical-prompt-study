from __future__ import annotations

import hashlib

import numpy as np

from .hashing import canonical_json_bytes
from .models import InterventionReceipt

REQUESTED_REALIZED_TOLERANCE = 1e-3
MAXIMUM_DELTA_TO_RESIDUAL_NORM = 0.05


def unit_direction(values: np.ndarray, *, expected_dimension: int | None = None) -> np.ndarray:
    direction = np.asarray(values, dtype=np.float32)
    if direction.ndim != 1:
        raise ValueError("intervention direction must be one-dimensional")
    if expected_dimension is not None and direction.shape != (expected_dimension,):
        raise ValueError("intervention direction dimension mismatch")
    if not np.all(np.isfinite(direction)):
        raise ValueError("intervention direction must be finite")
    norm = float(np.linalg.norm(direction.astype(np.float64)))
    if norm <= 0:
        raise ValueError("intervention direction must have positive norm")
    normalized = direction / np.float32(norm)
    realized = float(np.linalg.norm(normalized.astype(np.float64)))
    if not np.isclose(realized, 1.0, rtol=REQUESTED_REALIZED_TOLERANCE, atol=0):
        raise ValueError("normalized direction norm mismatch")
    return normalized


def deterministic_isotropic_direction(seed: int, dimension: int) -> np.ndarray:
    if seed < 0 or dimension <= 0:
        raise ValueError("invalid isotropic direction parameters")
    generator = np.random.default_rng(seed)
    return unit_direction(generator.standard_normal(dimension))


def direction_sha256(direction: np.ndarray) -> str:
    normalized = unit_direction(direction)
    return hashlib.sha256(normalized.astype("<f4", copy=False).tobytes()).hexdigest()


def intervention_delta(
    residual: np.ndarray,
    direction: np.ndarray,
    *,
    sign: int,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    hidden = np.asarray(residual, dtype=np.float32)
    if hidden.ndim != 1 or not np.all(np.isfinite(hidden)):
        raise ValueError("residual must be a finite vector")
    pre_norm = float(np.linalg.norm(hidden.astype(np.float64)))
    if pre_norm <= 0:
        raise ValueError("residual norm must be positive")
    if sign not in {-1, 0, 1}:
        raise ValueError("intervention sign must be -1, 0, or 1")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("intervention alpha must be finite and nonnegative")
    if (sign == 0) != (alpha == 0):
        raise ValueError("zero sign and zero alpha must occur together")
    unit = unit_direction(direction, expected_dimension=hidden.size)
    delta = np.float32(sign * alpha) * unit
    realized_delta_norm = float(np.linalg.norm(delta.astype(np.float64)))
    relative_error = (
        0.0
        if alpha == 0
        else abs(realized_delta_norm - alpha) / float(alpha)
    )
    ratio = realized_delta_norm / pre_norm
    if relative_error > REQUESTED_REALIZED_TOLERANCE:
        raise ValueError("requested/realized intervention norm mismatch")
    if ratio > MAXIMUM_DELTA_TO_RESIDUAL_NORM:
        raise ValueError("intervention exceeds residual norm budget")
    post = hidden + delta
    post_norm = float(np.linalg.norm(post.astype(np.float64)))
    if not np.all(np.isfinite(post)) or post_norm <= 0:
        raise ValueError("intervention produced invalid residual")
    return post, {
        "requested_delta_norm": float(alpha),
        "realized_delta_norm": realized_delta_norm,
        "pre_residual_norm": pre_norm,
        "post_residual_norm": post_norm,
        "requested_realized_relative_error": relative_error,
        "delta_to_pre_residual_norm": ratio,
        "clipped": False,
    }


def validate_intervention_receipt(payload: dict) -> InterventionReceipt:
    receipt = InterventionReceipt.model_validate(payload)
    zero = receipt.direction_kind == "zero"
    if zero:
        if (
            receipt.feature_id is not None
            or receipt.isotropic_seed is not None
            or receipt.direction_sha256 is not None
            or receipt.requested_sign != 0
            or receipt.requested_alpha != 0
        ):
            raise ValueError("zero intervention metadata mismatch")
    else:
        if receipt.requested_sign not in {-1, 1} or receipt.requested_alpha <= 0:
            raise ValueError("nonzero intervention metadata mismatch")
        if receipt.direction_sha256 is None:
            raise ValueError("nonzero intervention requires direction hash")
    if receipt.direction_kind in {"primary", "matched_sae"}:
        if receipt.feature_id is None or receipt.isotropic_seed is not None:
            raise ValueError("SAE intervention identity mismatch")
    if receipt.direction_kind == "isotropic":
        if receipt.feature_id is not None or receipt.isotropic_seed is None:
            raise ValueError("isotropic intervention identity mismatch")
    if receipt.generated_token_count != len(receipt.generated_token_ids):
        raise ValueError("generated token count mismatch")
    if receipt.error is None and len(receipt.intervention_steps) != len(
        receipt.generated_token_ids
    ):
        raise ValueError("intervention step topology mismatch")
    for index, step in enumerate(receipt.intervention_steps):
        if step.generated_token_index != index:
            raise ValueError("intervention step index mismatch")
        if step.clipped:
            raise ValueError("clipping is forbidden")
        if (
            step.requested_realized_relative_error
            > REQUESTED_REALIZED_TOLERANCE
        ):
            raise ValueError("receipt requested/realized norm mismatch")
        if step.delta_to_pre_residual_norm > MAXIMUM_DELTA_TO_RESIDUAL_NORM:
            raise ValueError("receipt residual norm budget exceeded")
    return receipt


def stable_intervention_trial_id(
    study_id: str,
    run_id: str,
    behavior_id: str,
    condition_id: str,
) -> str:
    payload = [study_id, run_id, behavior_id, condition_id]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]
