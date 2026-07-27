from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .behavior import _as_token_ids
from .factorial_plan import validate_factorial_plan
from .followup_design import PLACEMENTS
from .followup_runner import (
    _assert_no_raw_public_fields,
    build_placement_render_pair,
)
from .hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from .models import FactorialPrivatePlanReceipt

REQUEST_LEVELS = (
    "minimal_neutral_carrier",
    "ordinary_benign_request",
    "harmful_request",
)
MATERIALS = ("inert_length", "structural_sham", "full_scaffold")
NOMINAL_DOSES = (0.25, 0.5, 0.75, 1.0)


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return _as_token_ids(tokenizer.encode(text, add_special_tokens=False))


def _atomic_private_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_bytes(payload)


def _atomic_public_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o644)
    temporary.replace(path)
    return sha256_bytes(payload)


def _validate_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label}: expected lowercase SHA-256")
    return text


def _validate_request_panels(
    source: Mapping[str, Any], *, expected_count: int
) -> dict[str, list[dict[str, Any]]]:
    panels = source["request_panels"]
    if set(panels) != set(REQUEST_LEVELS):
        raise ValueError("factorial request-panel topology drift")
    seen_request_hashes: set[str] = set()
    validated: dict[str, list[dict[str, Any]]] = {}
    for level in REQUEST_LEVELS:
        rows = panels[level]
        if len(rows) != expected_count:
            raise ValueError(f"{level}: expected {expected_count} requests")
        ids: set[str] = set()
        family_ids: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for raw in rows:
            row = dict(raw)
            request_id = str(row["request_id"])
            family_id = str(row["prompt_family_id"])
            request = str(row["request"])
            request_sha256 = _validate_sha256(
                row["request_sha256"], f"{request_id}: request"
            )
            _validate_sha256(row["source_sha256"], f"{request_id}: source")
            if not request_id or request_id in ids:
                raise ValueError(f"{level}: duplicate or empty request ID")
            if not family_id or family_id in family_ids:
                raise ValueError(f"{level}: prompt-family pseudoreplication")
            if not request or sha256_text(request) != request_sha256:
                raise ValueError(f"{request_id}: request hash drift")
            if request_sha256 in seen_request_hashes:
                raise ValueError("request bytes reused across request classes")
            ids.add(request_id)
            family_ids.add(family_id)
            seen_request_hashes.add(request_sha256)
            normalized.append(row)
        validated[level] = sorted(normalized, key=lambda item: item["request_id"])
    return validated


def _validate_material_blocks(
    source: Mapping[str, Any], tokenizer: Any
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    materials = source["scaffold_materials"]
    if set(materials) != set(MATERIALS):
        raise ValueError("factorial scaffold-material topology drift")
    joiner = str(source["material_block_joiner"])
    if sha256_text(joiner) != _validate_sha256(
        source["material_block_joiner_sha256"], "material block joiner"
    ):
        raise ValueError("material block joiner hash drift")
    normalized: dict[str, list[dict[str, Any]]] = {}
    block_counts: set[int] = set()
    for material in MATERIALS:
        blocks = materials[material]["blocks"]
        if not blocks:
            raise ValueError(f"{material}: block list cannot be empty")
        block_ids: set[str] = set()
        normalized_blocks: list[dict[str, Any]] = []
        for raw in blocks:
            block = dict(raw)
            block_id = str(block["block_id"])
            text = str(block["text"])
            if not block_id or block_id in block_ids:
                raise ValueError(f"{material}: duplicate or empty block ID")
            if not text or sha256_text(text) != _validate_sha256(
                block["text_sha256"], f"{material}:{block_id}"
            ):
                raise ValueError(f"{material}:{block_id}: block hash drift")
            block_ids.add(block_id)
            normalized_blocks.append(block)
        normalized[material] = normalized_blocks
        block_counts.add(len(normalized_blocks))
    if len(block_counts) != 1:
        raise ValueError("full, sham, and inert materials must have equal block counts")

    cumulative_counts: dict[str, list[int]] = {}
    for material, blocks in normalized.items():
        cumulative_counts[material] = [
            len(_token_ids(tokenizer, joiner.join(block["text"] for block in blocks[:index])))
            for index in range(1, len(blocks) + 1)
        ]
    reference = cumulative_counts["full_scaffold"]
    for material in MATERIALS:
        if cumulative_counts[material] != reference:
            raise ValueError(
                "full, sham, and inert cumulative injected-token counts must match exactly"
            )
    return normalized, joiner


def _resolve_doses(
    materials: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    joiner: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    full_blocks = materials["full_scaffold"]
    cumulative = [
        len(_token_ids(tokenizer, joiner.join(block["text"] for block in full_blocks[:index])))
        for index in range(1, len(full_blocks) + 1)
    ]
    canonical_count = cumulative[-1]
    by_prefix: dict[int, list[float]] = {}
    for fraction in NOMINAL_DOSES:
        target = fraction * canonical_count
        prefix_count = min(
            range(1, len(cumulative) + 1),
            key=lambda count: (abs(cumulative[count - 1] - target), count),
        )
        by_prefix.setdefault(prefix_count, []).append(fraction)
    realized = []
    for prefix_count, nominal_fractions in sorted(by_prefix.items()):
        token_count = cumulative[prefix_count - 1]
        material_hashes = {}
        for material in MATERIALS:
            text = joiner.join(
                block["text"] for block in materials[material][:prefix_count]
            )
            if len(_token_ids(tokenizer, text)) != token_count:
                raise ValueError("realized material prefix token-count drift")
            material_hashes[material] = sha256_text(text)
        realized.append(
            {
                "size_id": f"blocks-{prefix_count:03d}",
                "prefix_block_count": prefix_count,
                "nominal_fractions": nominal_fractions,
                "representative_nominal_fraction": max(nominal_fractions),
                "injected_token_count": token_count,
                "realized_token_fraction": token_count / canonical_count,
                "material_prefix_sha256": material_hashes,
                "canonical": prefix_count == len(full_blocks),
            }
        )
    if sum(item["canonical"] for item in realized) != 1:
        raise ValueError("dose resolution must contain exactly one canonical prefix")
    return realized


def _base_render(
    *, tokenizer: Any, request: str, context_ceiling: int, generation_budget: int
) -> dict[str, Any]:
    messages = [{"role": "user", "content": request}]
    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if len(prompt_ids) + generation_budget > context_ceiling:
        raise ValueError("natural no-scaffold render exceeds context ceiling")
    return {
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_ids,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_token_ids_sha256": sha256_bytes(canonical_json_bytes(prompt_ids)),
        "total_tokens": len(prompt_ids),
        "padding_applied": False,
        "request_trimmed": False,
    }


def _matched_render_group(
    *,
    tokenizer: Any,
    request: str,
    scaffold_by_material: Mapping[str, str],
    tokenizer_revision: str,
    context_ceiling: int,
    generation_budget: int,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    private: dict[str, dict[str, dict[str, Any]]] = {}
    public: dict[str, Any] = {}
    for material in MATERIALS:
        validation, renders = build_placement_render_pair(
            tokenizer=tokenizer,
            request=request,
            scaffold=scaffold_by_material[material],
            tokenizer_revision=tokenizer_revision,
            context_ceiling=context_ceiling,
            generation_budget=generation_budget,
        )
        private[material] = renders
        public[material] = validation
    for placement in PLACEMENTS:
        receipts = [
            private[material][placement]["render_receipt"] for material in MATERIALS
        ]
        reference = receipts[0]
        for receipt in receipts[1:]:
            for field in (
                "total_tokens",
                "tokenizer_revision",
                "chat_template_sha256",
                "delimiter_special_tokens_sha256",
                "assistant_boundary_suffix_sha256",
                "context_ceiling",
                "generation_budget",
            ):
                if receipt[field] != reference[field]:
                    raise ValueError(f"matched material rendering mismatch: {field}")
            if (
                receipt["component_offsets"]["request"]
                != reference["component_offsets"]["request"]
                or receipt["component_token_counts"]["request"]
                != reference["component_token_counts"]["request"]
                or receipt["component_token_counts"]["scaffold"]
                != reference["component_token_counts"]["scaffold"]
            ):
                raise ValueError("matched material request/scaffold offset drift")
    summary = {
        "status": "passed",
        "materials": list(MATERIALS),
        "placements": list(PLACEMENTS),
        "injected_token_count": public["full_scaffold"]["component_token_counts"][
            "scaffold"
        ],
        "material_render_validation_sha256": {
            material: sha256_bytes(canonical_json_bytes(public[material]))
            for material in MATERIALS
        },
    }
    return summary, private


def _trial_id(
    *,
    study_id: str,
    request_class: str,
    request_id: str,
    material: str,
    placement: str | None,
    size_id: str,
) -> str:
    payload = {
        "study_id": study_id,
        "request_class": request_class,
        "request_id": request_id,
        "material": material,
        "placement": placement,
        "size_id": size_id,
    }
    return sha256_bytes(canonical_json_bytes(payload))[:24]


def build_factorial_private_plan(
    *,
    public_plan_path: Path,
    material_source_path: Path,
    tokenizer: Any,
    tokenizer_revision: str,
    private_output_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    public_plan = json.loads(public_plan_path.read_text())
    validate_factorial_plan(public_plan)
    source = json.loads(material_source_path.read_text())
    if source["schema_version"] != "1.0":
        raise ValueError("factorial material-source schema drift")
    if source["study_id"] != public_plan["study_id"]:
        raise ValueError("factorial material-source study mismatch")
    if source["public_plan_sha256"] != sha256_file(public_plan_path):
        raise ValueError("factorial material-source public-plan hash drift")
    provenance = source["provenance"]
    if (
        provenance["human_reviewed_restricted_scaffold"] is not True
        or provenance["new_harmful_panel_disjointness_attested"] is not True
        or provenance["outcome_independent_material_construction"] is not True
    ):
        raise ValueError("factorial material-source provenance gate failed")

    expected_count = int(
        public_plan["request_factor"]["canonical_matrix_items_per_level"]
    )
    panels = _validate_request_panels(source, expected_count=expected_count)
    materials, joiner = _validate_material_blocks(source, tokenizer)
    doses = _resolve_doses(materials, joiner=joiner, tokenizer=tokenizer)
    sentinel = source["literal_sentinel"]
    sentinel_request = str(sentinel["minimal_carrier"])
    if (
        not sentinel_request
        or sha256_text(sentinel_request)
        != _validate_sha256(
            sentinel["minimal_carrier_sha256"], "literal sentinel carrier"
        )
    ):
        raise ValueError("literal sentinel carrier hash drift")

    decoding = public_plan["execution_and_missingness"]["decoding"]
    context_ceiling = 8192
    generation_budget = 1024
    if (
        "8192-token context ceiling" not in decoding
        or "maximum 1024 generated tokens" not in decoding
    ):
        raise ValueError("factorial decoding contract drift")

    observations: list[dict[str, Any]] = []
    render_groups: list[dict[str, Any]] = []
    study_id = public_plan["study_id"]
    for request_class in REQUEST_LEVELS:
        rows = panels[request_class]
        dose_ids = {
            row["request_id"]
            for row in sorted(rows, key=lambda item: item["request_sha256"])[:10]
        }
        for row in rows:
            base = _base_render(
                tokenizer=tokenizer,
                request=row["request"],
                context_ceiling=context_ceiling,
                generation_budget=generation_budget,
            )
            observations.append(
                {
                    "trial_id": _trial_id(
                        study_id=study_id,
                        request_class=request_class,
                        request_id=row["request_id"],
                        material="no_scaffold",
                        placement=None,
                        size_id="natural-base",
                    ),
                    "request_class": request_class,
                    "request_id": row["request_id"],
                    "prompt_family_id": row["prompt_family_id"],
                    "request_sha256": row["request_sha256"],
                    "material": "no_scaffold",
                    "placement": None,
                    "size_id": "natural-base",
                    "nominal_fractions": [0.0],
                    "injected_token_count": 0,
                    "shared_reference": True,
                    **base,
                }
            )
            for dose in doses:
                if not dose["canonical"] and row["request_id"] not in dose_ids:
                    continue
                scaffold_by_material = {
                    material: joiner.join(
                        block["text"]
                        for block in materials[material][
                            : dose["prefix_block_count"]
                        ]
                    )
                    for material in MATERIALS
                }
                validation, renders = _matched_render_group(
                    tokenizer=tokenizer,
                    request=row["request"],
                    scaffold_by_material=scaffold_by_material,
                    tokenizer_revision=tokenizer_revision,
                    context_ceiling=context_ceiling,
                    generation_budget=generation_budget,
                )
                render_group_sha256 = sha256_bytes(
                    canonical_json_bytes(
                        {
                            "request_sha256": row["request_sha256"],
                            "size_id": dose["size_id"],
                            **validation,
                        }
                    )
                )
                render_groups.append(
                    {
                        "request_class": request_class,
                        "request_id": row["request_id"],
                        "size_id": dose["size_id"],
                        "render_group_sha256": render_group_sha256,
                        **validation,
                    }
                )
                for material in MATERIALS:
                    for placement in PLACEMENTS:
                        render = renders[material][placement]
                        observations.append(
                            {
                                "trial_id": _trial_id(
                                    study_id=study_id,
                                    request_class=request_class,
                                    request_id=row["request_id"],
                                    material=material,
                                    placement=placement,
                                    size_id=dose["size_id"],
                                ),
                                "request_class": request_class,
                                "request_id": row["request_id"],
                                "prompt_family_id": row["prompt_family_id"],
                                "request_sha256": row["request_sha256"],
                                "material": material,
                                "placement": placement,
                                "size_id": dose["size_id"],
                                "nominal_fractions": dose["nominal_fractions"],
                                "realized_token_fraction": dose[
                                    "realized_token_fraction"
                                ],
                                "injected_token_count": dose[
                                    "injected_token_count"
                                ],
                                "shared_reference": False,
                                "render_group_sha256": render_group_sha256,
                                "prompt_text": render["prompt_text"],
                                "prompt_token_ids": render["prompt_token_ids"],
                                "prompt_sha256": sha256_text(render["prompt_text"]),
                                "prompt_token_ids_sha256": sha256_bytes(
                                    canonical_json_bytes(
                                        render["prompt_token_ids"]
                                    )
                                ),
                                "render_receipt": render["render_receipt"],
                            }
                        )

    canonical_dose = next(dose for dose in doses if dose["canonical"])
    sentinel_scaffold = joiner.join(
        block["text"] for block in materials["full_scaffold"]
    )
    sentinel_validation, sentinel_renders = build_placement_render_pair(
        tokenizer=tokenizer,
        request=sentinel_request,
        scaffold=sentinel_scaffold,
        tokenizer_revision=tokenizer_revision,
        context_ceiling=context_ceiling,
        generation_budget=generation_budget,
    )
    for placement in PLACEMENTS:
        render = sentinel_renders[placement]
        observations.append(
            {
                "trial_id": _trial_id(
                    study_id=study_id,
                    request_class="literal_sentinel",
                    request_id="literal-sentinel",
                    material="full_scaffold",
                    placement=placement,
                    size_id=canonical_dose["size_id"],
                ),
                "request_class": "literal_sentinel",
                "request_id": "literal-sentinel",
                "prompt_family_id": "literal-sentinel-descriptive-n1",
                "request_sha256": sha256_text(sentinel_request),
                "material": "full_scaffold",
                "placement": placement,
                "size_id": canonical_dose["size_id"],
                "nominal_fractions": [1.0],
                "realized_token_fraction": 1.0,
                "injected_token_count": canonical_dose["injected_token_count"],
                "shared_reference": False,
                "descriptive_n_equals_1": True,
                "prompt_text": render["prompt_text"],
                "prompt_token_ids": render["prompt_token_ids"],
                "prompt_sha256": sha256_text(render["prompt_text"]),
                "prompt_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(render["prompt_token_ids"])
                ),
                "render_receipt": render["render_receipt"],
            }
        )

    trial_ids = [row["trial_id"] for row in observations]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("factorial trial IDs are not unique")
    canonical_count = sum(
        row["material"] == "no_scaffold"
        or row["size_id"] == canonical_dose["size_id"]
        for row in observations
    )
    expected_canonical = int(
        public_plan["staged_bill_of_materials"]["stage_C_canonical_factorial"][
            "unique_conditions_total"
        ]
    )
    if canonical_count != expected_canonical:
        raise ValueError(
            f"canonical factorial topology drift: {canonical_count} != {expected_canonical}"
        )
    expected_maximum = int(
        public_plan["staged_bill_of_materials"]["combined_factorial_and_dose"][
            "unique_target_conditions"
        ]
    )
    if len(observations) > expected_maximum:
        raise ValueError("factorial observation count exceeds frozen maximum")

    private = {
        "schema_version": "1.0",
        "study_id": study_id,
        "source_commit": _source_commit(),
        "public_plan_sha256": sha256_file(public_plan_path),
        "material_source_sha256": sha256_file(material_source_path),
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_chat_template_sha256": sha256_text(
            getattr(tokenizer, "chat_template", None) or ""
        ),
        "doses": doses,
        "request_panels": panels,
        "material_blocks": materials,
        "material_block_joiner": joiner,
        "literal_sentinel": sentinel,
        "render_groups": render_groups,
        "observations": observations,
    }
    private_sha256 = _atomic_private_json(private_output_path, private)
    receipt = {
        "schema_version": "1.0",
        "study_id": study_id,
        "status": "factorial_private_plan_complete_no_target_outcomes",
        "source_commit": private["source_commit"],
        "public_plan_sha256": private["public_plan_sha256"],
        "material_source_sha256": private["material_source_sha256"],
        "private_plan_sha256": private_sha256,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_chat_template_sha256": private[
            "tokenizer_chat_template_sha256"
        ],
        "request_panel_counts": {
            level: len(panels[level]) for level in REQUEST_LEVELS
        },
        "request_panel_sha256": {
            level: sha256_bytes(
                canonical_json_bytes(
                    [
                        {
                            "request_id": row["request_id"],
                            "prompt_family_id": row["prompt_family_id"],
                            "request_sha256": row["request_sha256"],
                            "source_sha256": row["source_sha256"],
                        }
                        for row in panels[level]
                    ]
                )
            )
            for level in REQUEST_LEVELS
        },
        "material_block_counts": {
            material: len(materials[material]) for material in MATERIALS
        },
        "material_canonical_sha256": {
            material: sha256_text(
                joiner.join(block["text"] for block in materials[material])
            )
            for material in MATERIALS
        },
        "realized_doses": doses,
        "canonical_observation_count": canonical_count,
        "additional_dose_observation_count": len(observations) - canonical_count,
        "total_observation_count": len(observations),
        "placement_levels": list(PLACEMENTS),
        "exact_size_matching_passed": True,
        "raw_prompt_or_token_ids_public": False,
        "target_generation_performed": False,
        "target_outcome_exists": False,
    }
    validated_receipt = FactorialPrivatePlanReceipt.model_validate(receipt)
    public_payload = validated_receipt.model_dump(mode="json")
    _assert_no_raw_public_fields(public_payload)
    receipt_sha256 = _atomic_public_json(public_receipt_path, public_payload)
    return {
        "status": receipt["status"],
        "private_plan_path": str(private_output_path),
        "private_plan_sha256": private_sha256,
        "public_receipt_path": str(public_receipt_path),
        "public_receipt_sha256": receipt_sha256,
        "canonical_observations": canonical_count,
        "additional_dose_observations": len(observations) - canonical_count,
        "total_observations": len(observations),
    }
