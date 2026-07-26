from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .followup_design import PLACEMENTS
from .hashing import canonical_json_bytes


PRIMARY = "sham_into_full"
RECIPROCAL = "full_into_sham"
IDENTITY_CONDITIONS = {
    "full_into_full_identity",
    "sham_into_sham_identity",
    "no_op_hook",
}
NEGATIVE_CONTROL_CONDITIONS = {
    "same_site_magnitude_matched_seeded_random_delta",
    "irrelevant_layer",
    "irrelevant_token_position",
    "cross_behavior_category_and_length_matched_donor",
}


@dataclass(frozen=True)
class PatchUnit:
    behavior_id: str
    category: str
    prompt_token_count: int


def stable_patch_seed(
    *,
    base_seed: int,
    partition: str,
    placement: str,
    layer: int,
    condition: str,
    behavior_id: str,
) -> int:
    payload = {
        "base_seed": int(base_seed),
        "partition": partition,
        "placement": placement,
        "layer": int(layer),
        "condition": condition,
        "behavior_id": behavior_id,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def select_cross_behavior_donors(
    units: Iterable[PatchUnit],
) -> dict[str, str]:
    rows = list(units)
    if len(rows) < 2 or len({row.behavior_id for row in rows}) != len(rows):
        raise ValueError("cross-behavior units must have unique behavior IDs")
    hashes = {
        row.behavior_id: hashlib.sha256(row.behavior_id.encode()).hexdigest()
        for row in rows
    }
    hash_order = sorted(rows, key=lambda row: (hashes[row.behavior_id], row.behavior_id))
    rank = {row.behavior_id: index for index, row in enumerate(hash_order)}
    result = {}
    for recipient in rows:
        candidates = [
            row
            for row in rows
            if row.behavior_id != recipient.behavior_id
            and row.category == recipient.category
        ]
        if not candidates:
            raise ValueError(
                f"{recipient.behavior_id}: no cross-behavior donor in category"
            )
        nearest = min(
            abs(row.prompt_token_count - recipient.prompt_token_count)
            for row in candidates
        )
        candidates = [
            row
            for row in candidates
            if abs(row.prompt_token_count - recipient.prompt_token_count) == nearest
        ]
        count = len(hash_order)
        candidates.sort(
            key=lambda row: (
                (rank[row.behavior_id] - rank[recipient.behavior_id]) % count,
                hashes[row.behavior_id],
                row.behavior_id,
            )
        )
        result[recipient.behavior_id] = candidates[0].behavior_id
    return result


class ResidualStatePatch:
    """Apply one batched residual-post edit during prefill and never during decode."""

    def __init__(
        self,
        torch,
        *,
        replacement=None,
        delta=None,
        token_offset: int = -1,
    ):
        if (replacement is None) == (delta is None):
            raise ValueError("patch requires exactly one of replacement or delta")
        if token_offset not in {-1, -2}:
            raise ValueError("patch token offset must be -1 or -2")
        tensor = replacement if replacement is not None else delta
        if tensor.ndim != 2 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("patch tensor must be finite [batch, hidden]")
        self.torch = torch
        self.replacement = replacement
        self.delta = delta
        self.token_offset = token_offset
        self.applied = False
        self.replay: dict[str, Any] | None = None

    def __call__(self, _module, _inputs, output):
        if self.applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise ValueError("patch must first run on a multi-token prefill")
        edit = self.replacement if self.replacement is not None else self.delta
        if hidden.shape[0] != edit.shape[0] or hidden.shape[2] != edit.shape[1]:
            raise ValueError("patch batch or hidden width mismatch")
        position = hidden.shape[1] + self.token_offset
        recipient = hidden[:, position, :].detach()
        modified = hidden.clone()
        if self.replacement is not None:
            proposed = self.replacement.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            proposed = recipient + self.delta.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        realized = proposed.float() - recipient.float()
        modified[:, position, :] = proposed
        self.replay = {
            "recipient": recipient.to("cpu", dtype=self.torch.bfloat16),
            "realized_delta": realized.to("cpu", dtype=self.torch.bfloat16),
        }
        self.applied = True
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified


class NoOpResidualHook:
    def __init__(self, torch, *, batch_size: int):
        if batch_size < 1:
            raise ValueError("no-op batch size must be positive")
        self.torch = torch
        self.batch_size = batch_size
        self.applied = False
        self.replay: dict[str, Any] | None = None

    def __call__(self, _module, _inputs, output):
        if self.applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != self.batch_size or hidden.shape[1] < 2:
            raise ValueError("no-op hook must first run on the expected prefill")
        recipient = hidden[:, -1, :].detach()
        zero = self.torch.zeros_like(recipient)
        self.replay = {
            "recipient": recipient.to("cpu", dtype=self.torch.bfloat16),
            "realized_delta": zero.to("cpu", dtype=self.torch.bfloat16),
        }
        self.applied = True
        modified = hidden.clone()
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified


def magnitude_matched_random_deltas(
    torch,
    reference_delta,
    *,
    seeds: list[int],
):
    if reference_delta.ndim != 2 or len(seeds) != reference_delta.shape[0]:
        raise ValueError("random delta topology mismatch")
    rows = []
    for row, seed in zip(reference_delta.float(), seeds, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        random = torch.randn(row.shape, generator=generator, dtype=torch.float32)
        random = random / random.norm()
        rows.append(random * row.norm())
    return torch.stack(rows)


def paired_interval(
    values: np.ndarray,
    *,
    level: float,
    replicates: int,
    seed: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("paired interval requires finite one-dimensional values")
    if not 0 < level < 1 or replicates < 1:
        raise ValueError("invalid interval configuration")
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, array.size, size=(replicates, array.size))]
    means = sampled.mean(axis=1)
    alpha = (1 - level) / 2
    return [float(value) for value in np.quantile(means, [alpha, 1 - alpha])]


def _summarize_effects(
    rows: list[Mapping[str, Any]],
    *,
    level: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["behavior_id"]))
    if len(ordered) < 2 or len({row["behavior_id"] for row in ordered}) != len(ordered):
        raise ValueError("patch effects require unique behavior IDs")
    values = np.asarray(
        [float(row["patched_score"]) - float(row["baseline_score"]) for row in ordered],
        dtype=np.float64,
    )
    rms = float(np.sqrt(np.mean(np.square(values))))
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "rms": rms,
        "standardized": float(values.mean() / rms) if rms > 0 else 0.0,
        "negative_concordance": float(np.mean(values < 0)),
        "positive_concordance": float(np.mean(values > 0)),
        "interval_level": level,
        "interval": paired_interval(
            values,
            level=level,
            replicates=replicates,
            seed=seed,
        ),
        "behavior_ids": [str(row["behavior_id"]) for row in ordered],
        "paired_effects": [float(value) for value in values],
    }


def analyze_coarse_patch_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    partition: str,
) -> dict[str, Any]:
    execution = plan["causal_localization"]["execution"]
    analysis = execution["analysis"]
    layers = plan["causal_localization"]["coarse_residual_post_layers"]
    conditions = execution["condition_kinds"]
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["partition"] != partition:
            raise ValueError("patch analysis partition drift")
        key = (str(row["placement"]), int(row["candidate_layer"]), str(row["condition"]))
        grouped[key].append(row)
    expected = {
        (placement, layer, condition)
        for placement in PLACEMENTS
        for layer in layers
        for condition in conditions
    }
    if set(grouped) != expected:
        raise ValueError("patch analysis topology drift")

    summaries: dict[str, dict[int, dict[str, Any]]] = {
        placement: {} for placement in PLACEMENTS
    }
    base_seed = int(analysis["bootstrap_seed"])
    replicates = int(analysis["bootstrap_replicates"])
    for placement in PLACEMENTS:
        for layer in layers:
            layer_result: dict[str, Any] = {}
            for condition in conditions:
                level = 0.9 if condition in IDENTITY_CONDITIONS else 0.95
                seed = stable_patch_seed(
                    base_seed=base_seed,
                    partition=partition,
                    placement=placement,
                    layer=layer,
                    condition=condition,
                    behavior_id="aggregate",
                )
                layer_result[condition] = _summarize_effects(
                    grouped[(placement, layer, condition)],
                    level=level,
                    replicates=replicates,
                    seed=seed,
                )
            primary = layer_result[PRIMARY]
            reciprocal = layer_result[RECIPROCAL]
            minimum = float(
                analysis["primary_and_reciprocal_require_absolute_mean_at_least"]
            )
            concordance = float(
                analysis[
                    "primary_and_reciprocal_minimum_directional_concordance"
                ]
            )
            primary_pass = (
                primary["mean"] <= -minimum
                and primary["negative_concordance"] >= concordance
                and primary["interval"][1] < 0
            )
            reciprocal_pass = (
                reciprocal["mean"] >= minimum
                and reciprocal["positive_concordance"] >= concordance
                and reciprocal["interval"][0] > 0
            )
            identity_pass = all(
                abs(layer_result[condition]["mean"]) <= 0.02
                and layer_result[condition]["interval"][0] >= -0.05
                and layer_result[condition]["interval"][1] <= 0.05
                for condition in IDENTITY_CONDITIONS
            )
            negative_control_pass = all(
                abs(layer_result[condition]["mean"]) <= 0.05
                and layer_result[condition]["interval"][0] <= 0
                <= layer_result[condition]["interval"][1]
                for condition in NEGATIVE_CONTROL_CONDITIONS
            )
            layer_result["eligibility"] = {
                "primary_pass": primary_pass,
                "reciprocal_pass": reciprocal_pass,
                "identity_and_noop_pass": identity_pass,
                "negative_controls_pass": negative_control_pass,
                "eligible": (
                    primary_pass
                    and reciprocal_pass
                    and identity_pass
                    and negative_control_pass
                ),
            }
            summaries[placement][layer] = layer_result

    eligible = [
        layer
        for layer in layers
        if all(
            summaries[placement][layer]["eligibility"]["eligible"]
            for placement in PLACEMENTS
        )
    ]
    ranking = sorted(
        eligible,
        key=lambda layer: (
            -min(
                abs(summaries[placement][layer][PRIMARY]["standardized"])
                for placement in PLACEMENTS
            ),
            layer,
        ),
    )
    return {
        "schema_version": "1.0",
        "partition": partition,
        "pooled_placement_estimate_reported": False,
        "ordering_results": summaries,
        "eligible_common_layers": eligible,
        "selected_common_layer": ranking[0] if ranking else None,
        "status": "selected_common_layer" if ranking else "stopped_no_eligible_layer",
    }
