from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes
from .models import MechanismReceipt


MECHANISM_SCHEMA_VERSION = "1.0"
PRIMARY_POSITION = "assistant_boundary"
SECONDARY_GENERATED_POSITIONS = (0, 1, 2, 4, 8, 16)
RANDOM_TRANSPORT_SEED = 20260725
SAE_HOOK_LAYER = 50


def probe_margin(
    logits: Sequence[float] | np.ndarray,
    refusal_token_ids: Sequence[int],
    compliance_token_ids: Sequence[int],
) -> dict[str, float]:
    """Compute the frozen vocabulary-z-scored refusal-minus-compliance margin."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("logits must be a one-dimensional vocabulary vector")
    if not np.isfinite(values).all():
        raise ValueError("logits contain non-finite values")
    refusal = tuple(int(item) for item in refusal_token_ids)
    compliance = tuple(int(item) for item in compliance_token_ids)
    if not refusal or not compliance:
        raise ValueError("both probe sets must be non-empty")
    if set(refusal) & set(compliance):
        raise ValueError("refusal and compliance probes must be disjoint")
    all_ids = refusal + compliance
    if min(all_ids) < 0 or max(all_ids) >= values.size:
        raise ValueError("probe token ID outside vocabulary")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        raise ValueError("vocabulary logit standard deviation is zero")
    z = (values - mean) / std
    refusal_mean = float(z[list(refusal)].mean())
    compliance_mean = float(z[list(compliance)].mean())
    return {
        "vocabulary_logit_mean": mean,
        "vocabulary_logit_std": std,
        "refusal_probe_mean_z": refusal_mean,
        "compliance_probe_mean_z": compliance_mean,
        "refusal_minus_compliance_margin": refusal_mean - compliance_mean,
    }


def probe_margin_from_embedding_moments(
    vector: Sequence[float] | np.ndarray,
    effective_unembedding: np.ndarray,
    refusal_token_ids: Sequence[int],
    compliance_token_ids: Sequence[int],
    *,
    rms_norm_epsilon: float,
) -> dict[str, float]:
    """Compute the exact Llama RMSNorm/unembedding margin without materializing logits."""
    hidden = np.asarray(vector, dtype=np.float64)
    embedding = np.asarray(effective_unembedding, dtype=np.float64)
    if hidden.ndim != 1 or embedding.ndim != 2 or embedding.shape[1] != hidden.size:
        raise ValueError("hidden vector and effective unembedding shapes do not align")
    if rms_norm_epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
    normalized = hidden / np.sqrt(np.mean(np.square(hidden)) + rms_norm_epsilon)
    embedding_mean = embedding.mean(axis=0)
    embedding_second_moment = embedding.T @ embedding / embedding.shape[0]
    mean = float(normalized @ embedding_mean)
    second = float(normalized @ embedding_second_moment @ normalized)
    std = float(np.sqrt(max(second - mean * mean, 0.0)))
    if std <= np.finfo(np.float64).eps:
        raise ValueError("vocabulary logit standard deviation is zero")
    refusal = tuple(int(item) for item in refusal_token_ids)
    compliance = tuple(int(item) for item in compliance_token_ids)
    if not refusal or not compliance or set(refusal) & set(compliance):
        raise ValueError("probe sets must be non-empty and disjoint")
    probe_ids = refusal + compliance
    if min(probe_ids) < 0 or max(probe_ids) >= embedding.shape[0]:
        raise ValueError("probe token ID outside vocabulary")
    probe_logits = embedding[list(probe_ids)] @ normalized
    probe_z = (probe_logits - mean) / std
    refusal_mean = float(probe_z[: len(refusal)].mean())
    compliance_mean = float(probe_z[len(refusal) :].mean())
    return {
        "vocabulary_logit_mean": mean,
        "vocabulary_logit_std": std,
        "refusal_probe_mean_z": refusal_mean,
        "compliance_probe_mean_z": compliance_mean,
        "refusal_minus_compliance_margin": refusal_mean - compliance_mean,
    }


def deterministic_transport_seed(base_seed: int, layer: int) -> int:
    """Derive a stable per-layer seed without depending on Python's hash salt."""
    payload = canonical_json_bytes(
        {"base_seed": int(base_seed), "layer": int(layer), "transport": "gaussian"}
    )
    return int.from_bytes(bytes.fromhex(sha256_bytes(payload))[:8], "big") % (2**63 - 1)


def validate_transport_metadata(
    *,
    transport: str,
    layer: int,
    fitted_frobenius_norm: float | None,
    realized_frobenius_norm: float | None,
    random_seed: int | None,
    relative_tolerance: float = 1e-5,
) -> None:
    if transport not in {"jacobian_lens", "identity", "random_gaussian"}:
        raise ValueError(f"unknown transport: {transport}")
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if transport == "random_gaussian":
        if random_seed is None:
            raise ValueError("random Gaussian transport requires a seed")
        if fitted_frobenius_norm is None or realized_frobenius_norm is None:
            raise ValueError("random Gaussian transport requires both Frobenius norms")
        if fitted_frobenius_norm <= 0 or realized_frobenius_norm <= 0:
            raise ValueError("transport Frobenius norms must be positive")
        relative_error = abs(realized_frobenius_norm - fitted_frobenius_norm) / (
            fitted_frobenius_norm
        )
        if relative_error > relative_tolerance:
            raise ValueError(
                f"random transport norm mismatch: relative_error={relative_error}"
            )
    elif random_seed is not None:
        raise ValueError(f"{transport} transport must not record a random seed")


@dataclass(frozen=True)
class SAEFeatureDiagnostic:
    feature_id: int
    paired_mean_delta: float
    paired_standardized_delta: float
    full_prevalence: float
    sham_prevalence: float
    all_prevalence: float
    decoder_norm: float


def sae_feature_diagnostics(
    full: np.ndarray,
    sham: np.ndarray,
    decoder_norms: Sequence[float] | np.ndarray,
) -> list[SAEFeatureDiagnostic]:
    """Return discovery-only paired diagnostics for every SAE feature."""
    full_values = np.asarray(full, dtype=np.float64)
    sham_values = np.asarray(sham, dtype=np.float64)
    norms = np.asarray(decoder_norms, dtype=np.float64)
    if full_values.ndim != 2 or sham_values.shape != full_values.shape:
        raise ValueError("full and sham activations must have identical [behavior, feature] shape")
    if norms.shape != (full_values.shape[1],):
        raise ValueError("decoder norms must have one value per feature")
    if full_values.shape[0] < 2:
        raise ValueError("at least two paired discovery behaviors are required")
    if not (
        np.isfinite(full_values).all()
        and np.isfinite(sham_values).all()
        and np.isfinite(norms).all()
    ):
        raise ValueError("SAE diagnostics contain non-finite values")
    if (full_values < 0).any() or (sham_values < 0).any() or (norms < 0).any():
        raise ValueError("SAE activations and decoder norms must be non-negative")
    delta = full_values - sham_values
    mean_delta = delta.mean(axis=0)
    # RMS standardization is finite for perfectly consistent non-zero paired
    # deltas and bounded in [-1, 1], avoiding an arbitrary epsilon in Cohen's dz.
    scale = np.sqrt(np.mean(np.square(delta), axis=0))
    standardized = np.divide(
        mean_delta,
        scale,
        out=np.zeros_like(mean_delta),
        where=scale > np.finfo(np.float64).eps,
    )
    all_values = np.concatenate([full_values, sham_values], axis=0)
    return [
        SAEFeatureDiagnostic(
            feature_id=feature_id,
            paired_mean_delta=float(mean_delta[feature_id]),
            paired_standardized_delta=float(standardized[feature_id]),
            full_prevalence=float(np.mean(full_values[:, feature_id] > 0)),
            sham_prevalence=float(np.mean(sham_values[:, feature_id] > 0)),
            all_prevalence=float(np.mean(all_values[:, feature_id] > 0)),
            decoder_norm=float(norms[feature_id]),
        )
        for feature_id in range(full_values.shape[1])
    ]


def select_sae_candidates(
    diagnostics: Iterable[SAEFeatureDiagnostic],
    *,
    minimum_full_prevalence: float = 0.10,
    maximum_candidates: int = 4,
) -> list[SAEFeatureDiagnostic]:
    """Freeze a primary plus at most three secondary discovery candidates."""
    if not 0 <= minimum_full_prevalence <= 1:
        raise ValueError("minimum prevalence must be in [0, 1]")
    if not 1 <= maximum_candidates <= 4:
        raise ValueError("maximum_candidates must be in [1, 4]")
    eligible = [
        item
        for item in diagnostics
        if item.full_prevalence >= minimum_full_prevalence
        and item.paired_mean_delta > 0
        and item.paired_standardized_delta > 0
        and item.decoder_norm > 0
    ]
    return sorted(
        eligible,
        key=lambda item: (
            -item.paired_standardized_delta,
            -item.paired_mean_delta,
            item.feature_id,
        ),
    )[:maximum_candidates]


def validate_mechanism_receipt(receipt: dict[str, Any]) -> None:
    parsed = MechanismReceipt.model_validate(receipt)
    if parsed.schema_version != MECHANISM_SCHEMA_VERSION:
        raise ValueError("mechanism receipt schema version mismatch")
    if parsed.position == PRIMARY_POSITION:
        if parsed.position_token_index is not None:
            raise ValueError("assistant boundary must not have a generated-token index")
    elif parsed.position_token_index not in SECONDARY_GENERATED_POSITIONS:
        raise ValueError("generated position is outside the frozen secondary set")
    if parsed.position_available:
        if parsed.margin is None or parsed.missing_position_reason is not None:
            raise ValueError("available position must have a margin and no missing reason")
    elif parsed.margin is not None or not parsed.missing_position_reason:
        raise ValueError("missing position must have a reason and no margin")
    validate_transport_metadata(
        transport=parsed.transport,
        layer=parsed.layer,
        fitted_frobenius_norm=parsed.fitted_frobenius_norm,
        realized_frobenius_norm=parsed.realized_frobenius_norm,
        random_seed=parsed.random_seed,
    )
    if parsed.margin is not None and not all(
        np.isfinite(float(value)) for value in parsed.margin.model_dump().values()
    ):
        raise ValueError("mechanism margin contains non-finite values")
