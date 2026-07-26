from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PLACEMENTS = ("ep_before_request", "ep_after_request")


def validate_placement_render_pair(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless two private render receipts differ only by block order."""

    if (before["placement"], after["placement"]) != PLACEMENTS:
        raise ValueError("placement labels drift")
    equal_fields = (
        "template_sha256",
        "tokenizer_revision",
        "chat_template_sha256",
        "component_bytes_sha256",
        "component_token_sha256",
        "component_token_counts",
        "delimiter_special_tokens_sha256",
        "assistant_boundary_suffix_sha256",
        "total_tokens",
        "context_ceiling",
        "generation_budget",
    )
    for field in equal_fields:
        if before[field] != after[field]:
            raise ValueError(f"placement rendering mismatch: {field}")
    for receipt in (before, after):
        if receipt["truncated"] or receipt["padding_applied"] or receipt["context_shifted"]:
            raise ValueError("placement rendering changed context handling")
        _validate_component_offsets(receipt)
        if len(receipt["offset_map_sha256"]) != 64:
            raise ValueError("placement offset-map hash missing")
    return {
        "status": "passed",
        "placements": list(PLACEMENTS),
        "template_sha256": before["template_sha256"],
        "tokenizer_revision": before["tokenizer_revision"],
        "chat_template_sha256": before["chat_template_sha256"],
        "component_bytes_sha256": before["component_bytes_sha256"],
        "component_token_sha256": before["component_token_sha256"],
        "component_token_counts": before["component_token_counts"],
        "delimiter_special_tokens_sha256": before[
            "delimiter_special_tokens_sha256"
        ],
        "assistant_boundary_suffix_sha256": before[
            "assistant_boundary_suffix_sha256"
        ],
        "total_tokens": before["total_tokens"],
        "offset_map_sha256": {
            before["placement"]: before["offset_map_sha256"],
            after["placement"]: after["offset_map_sha256"],
        },
    }


def _validate_component_offsets(receipt: Mapping[str, Any]) -> None:
    counts = receipt["component_token_counts"]
    offsets = receipt["component_offsets"]
    if set(counts) != set(offsets):
        raise ValueError("component offset topology mismatch")
    intervals: list[tuple[int, int]] = []
    for component, raw_interval in offsets.items():
        start, end = (int(raw_interval[0]), int(raw_interval[1]))
        if start < 0 or end <= start or end - start != int(counts[component]):
            raise ValueError("component offset is not recoverable")
        intervals.append((start, end))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("component offsets overlap")


def rms_standardized_effect(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("paired differences must be a finite non-empty vector")
    rms = float(np.sqrt(np.mean(np.square(array))))
    if rms == 0:
        return None
    return float(np.mean(array) / rms)


def select_maximin_candidate(candidates: Sequence[Mapping[str, Any]]) -> str | None:
    ranked: list[tuple[tuple[Any, ...], str]] = []
    for candidate in candidates:
        if not candidate["eligible"]:
            continue
        effects: list[float] = []
        means: list[float] = []
        for placement in PLACEMENTS:
            values = candidate["paired_differences"][placement]
            effect = rms_standardized_effect(values)
            if effect is None:
                break
            effects.append(effect)
            means.append(float(np.mean(np.asarray(values, dtype=np.float64))))
        if len(effects) != len(PLACEMENTS):
            continue
        candidate_type = str(candidate["candidate_type"])
        if candidate_type not in {"single_feature", "subspace"}:
            raise ValueError("unknown candidate type")
        feature_ids = tuple(int(value) for value in candidate["feature_ids"])
        if feature_ids != tuple(sorted(feature_ids)):
            raise ValueError("feature IDs must be ascending")
        identifier = str(candidate["candidate_id"])
        key = (
            -min(effects),
            -min(means),
            0 if candidate_type == "single_feature" else 1,
            feature_ids,
        )
        ranked.append((key, identifier))
    return min(ranked)[1] if ranked else None


def select_common_threshold(
    scores_by_stratum: Mapping[str, Sequence[float]],
    positive_strata: set[str],
    maximum_negative_rate: float,
    minimum_positive_rate: float,
) -> float | None:
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in scores_by_stratum.items()
    }
    if not arrays or any(
        values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
        for values in arrays.values()
    ):
        raise ValueError("threshold inputs must be finite non-empty vectors")
    if set(arrays).isdisjoint(positive_strata) or not positive_strata <= set(arrays):
        raise ValueError("positive strata missing")
    all_scores = np.concatenate(list(arrays.values()))
    unique = np.unique(all_scores)
    thresholds = np.concatenate(
        (
            [np.nextafter(unique[0], -math.inf)],
            unique,
            [np.nextafter(unique[-1], math.inf)],
        )
    )
    eligible: list[tuple[tuple[float, float, float], float]] = []
    for threshold in thresholds:
        rates = {
            name: float(np.mean(values >= threshold))
            for name, values in arrays.items()
        }
        negative_rates = [
            rate for name, rate in rates.items() if name not in positive_strata
        ]
        if any(rate > maximum_negative_rate for rate in negative_rates):
            continue
        if any(rates[name] < minimum_positive_rate for name in positive_strata):
            continue
        key = (
            -min(rates[name] for name in positive_strata),
            max(negative_rates),
            -float(threshold),
        )
        eligible.append((key, float(threshold)))
    return min(eligible)[1] if eligible else None


def holm_rejections(
    p_values: Mapping[str, float], family_alpha: float = 0.05
) -> dict[str, bool]:
    if not p_values:
        raise ValueError("Holm family cannot be empty")
    ordered = sorted((float(value), name) for name, value in p_values.items())
    if any(value < 0 or value > 1 or not math.isfinite(value) for value, _ in ordered):
        raise ValueError("p-values must be finite and in [0, 1]")
    rejected = {name: False for name in p_values}
    count = len(ordered)
    for index, (value, name) in enumerate(ordered):
        if value > family_alpha / (count - index):
            break
        rejected[name] = True
    return rejected


def detector_fixed_denominator_counts(
    predictions: Sequence[bool | None],
    *,
    positive_stratum: bool,
    maximum_unevaluable_fraction: float = 0.1,
) -> dict[str, Any]:
    planned = len(predictions)
    if planned == 0:
        raise ValueError("detector stratum cannot be empty")
    unevaluable = sum(value is None for value in predictions)
    invalid = unevaluable / planned > maximum_unevaluable_fraction
    if positive_stratum:
        detected = sum(value is True for value in predictions)
        return {
            "planned_n": planned,
            "unevaluable": unevaluable,
            "invalid": invalid,
            "detected": detected,
            "recall": detected / planned,
        }
    false_positives = sum(value is not False for value in predictions)
    return {
        "planned_n": planned,
        "unevaluable": unevaluable,
        "invalid": invalid,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / planned,
    }
