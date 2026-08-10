from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .followup_runner import build_placement_render_pair
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .weaponization_topology import (
    CONTEXT_CEILING,
    GENERATION_BUDGET,
    _normalize_prompt_token_ids,
    _request_text,
)


PARTITION_SIZES = {"detector_calibration": 90, "detector_confirmation": 300}
PLACEMENT_MAP = {
    "scaffold_before_request": "ep_before_request",
    "scaffold_after_request": "ep_after_request",
}
INJECTED_SIZES = (64, 128, 188, 252)
REQUEST_CLASSES = (
    "harmful_request",
    "ordinary_benign_request",
    "minimal_neutral_carrier",
)
INJECTED_MATERIALS = (
    "full_attack_scaffold",
    "inert_length",
    "structural_sham",
    "harmless_structured_wrapper",
)
INHERITED_MATERIAL_MAP = {
    "full_attack_scaffold": "full_scaffold",
    "inert_length": "inert_length",
    "structural_sham": "structural_sham",
}


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int) -> str:
    raw = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == 0o600:
        path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(mode)
    temporary.replace(path)
    return sha256_bytes(raw)


def _load_and_validate_inputs(
    *,
    plan_path: Path,
    harmful_panel_path: Path,
    safe_panel_path: Path,
    harmless_material_path: Path,
    inherited_material_path: Path,
    inherited_material_receipt_path: Path,
) -> tuple[dict[str, Any], ...]:
    plan = json.loads(plan_path.read_text())
    if plan.get("study_id") != "lexical-jlens-breaker-v2":
        raise ValueError("v2 plan identity drift")
    if plan.get("status") != (
        "prospectively_frozen_local_implementation_only_no_v2_target_outcomes"
    ):
        raise ValueError("v2 plan status drift")
    inputs = plan["new_inputs"]
    expected = {
        harmful_panel_path: inputs["harmful_requests"]["private_panel_sha256"],
        safe_panel_path: inputs["safe_requests"]["private_panel_sha256"],
        harmless_material_path: inputs["harmless_wrappers"][
            "private_material_sha256"
        ],
        inherited_material_path: inputs["inherited_restricted_materials"][
            "private_material_sha256"
        ],
        inherited_material_receipt_path: inputs["inherited_restricted_materials"][
            "public_receipt_sha256"
        ],
    }
    for path, expected_sha256 in expected.items():
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"v2 input hash drift: {path.name}")
    harmful = json.loads(harmful_panel_path.read_text())
    safe = json.loads(safe_panel_path.read_text())
    harmless = json.loads(harmless_material_path.read_text())
    inherited = json.loads(inherited_material_path.read_text())
    inherited_receipt = json.loads(inherited_material_receipt_path.read_text())
    if harmful.get("panel_epoch") != "v2":
        raise ValueError("v2 harmful panel epoch drift")
    if safe.get("surface_epoch") != "v2-a099":
        raise ValueError("v2 safe panel epoch drift")
    if harmless.get("surface_epoch") != "v2-a099":
        raise ValueError("v2 harmless surface epoch drift")
    if harmless["human_semantic_review"].get("approved") is not True:
        raise ValueError("v2 harmless semantic review is not approved")
    if inherited_receipt.get("target_outcome_exists") is not False:
        raise ValueError("inherited material receipt outcome boundary drift")
    return plan, harmful, safe, harmless, inherited, inherited_receipt


def _request_panels(
    *, harmful: Mapping[str, Any], safe: Mapping[str, Any], partition: str
) -> dict[str, list[dict[str, Any]]]:
    rows = {
        "harmful_request": list(harmful["panels"][partition]),
        "ordinary_benign_request": list(
            safe["partitions"][partition]["ordinary_benign_request"]
        ),
        "minimal_neutral_carrier": list(
            safe["partitions"][partition]["minimal_neutral_carrier"]
        ),
    }
    n = PARTITION_SIZES[partition]
    if len(rows["harmful_request"]) != n or any(
        len(rows[request_class]) < n
        for request_class in (
            "ordinary_benign_request",
            "minimal_neutral_carrier",
        )
    ):
        raise ValueError("v2 request panel count drift")
    return {request_class: panel[:n] for request_class, panel in rows.items()}


def _harmless_rows(
    harmless: Mapping[str, Any], partition: str
) -> list[dict[str, Any]]:
    rows = [row for row in harmless["wrappers"] if row["partition"] == partition]
    rows.sort(key=lambda row: row["wrapper_id"])
    n = PARTITION_SIZES[partition]
    if len(rows) < n:
        raise ValueError("v2 harmless wrapper count drift")
    return rows[:n]


def _prefix_from_blocks(
    blocks: list[dict[str, Any]], *, joiner: str, injected_token_count: int
) -> str:
    try:
        block_count = INJECTED_SIZES.index(injected_token_count) + 1
    except ValueError as error:
        raise ValueError("unknown v2 injected token count") from error
    if len(blocks) != len(INJECTED_SIZES):
        raise ValueError("v2 material block topology drift")
    return joiner.join(str(block["text"]) for block in blocks[:block_count])


def _material_text(
    *,
    material: str,
    injected_token_count: int,
    wrapper: Mapping[str, Any],
    inherited: Mapping[str, Any],
) -> str:
    if material == "harmless_structured_wrapper":
        return _prefix_from_blocks(
            list(wrapper["blocks"]),
            joiner="",
            injected_token_count=injected_token_count,
        )
    inherited_name = INHERITED_MATERIAL_MAP[material]
    return _prefix_from_blocks(
        list(inherited["scaffold_materials"][inherited_name]["blocks"]),
        joiner=str(inherited["material_block_joiner"]),
        injected_token_count=injected_token_count,
    )


def _shared_render(tokenizer: Any, request: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": request}]
    prompt_ids = _normalize_prompt_token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    )
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if len(prompt_ids) + GENERATION_BUDGET > CONTEXT_CEILING:
        raise ValueError("v2 no-scaffold render exceeds context ceiling")
    return {
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_ids,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_token_ids_sha256": sha256_bytes(canonical_json_bytes(prompt_ids)),
    }


def _gate_role(
    *, material: str, request_class: str, injected_token_count: int
) -> str:
    if injected_token_count != 252:
        return "size_robustness"
    key = (material, request_class)
    roles = {
        ("full_attack_scaffold", "harmful_request"): "positive_attack_harmful",
        ("full_attack_scaffold", "ordinary_benign_request"): "negative_attack_benign",
        ("full_attack_scaffold", "minimal_neutral_carrier"): "negative_attack_neutral",
        ("harmless_structured_wrapper", "harmful_request"): "negative_harmless_harmful",
        ("harmless_structured_wrapper", "ordinary_benign_request"): "negative_harmless_benign",
        ("structural_sham", "harmful_request"): "negative_structural_sham_harmful",
        ("inert_length", "harmful_request"): "negative_inert_harmful",
    }
    return roles.get(key, "secondary_factorial")


def _trial_id(public: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(public))[:24]


def build_jlens_breaker_v2_topology(
    *,
    plan_path: Path,
    harmful_panel_path: Path,
    safe_panel_path: Path,
    harmless_material_path: Path,
    inherited_material_path: Path,
    inherited_material_receipt_path: Path,
    tokenizer_path: Path,
    tokenizer_revision: str,
    partition: str,
    private_output_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if partition not in PARTITION_SIZES:
        raise ValueError("unknown v2 topology partition")
    plan, harmful, safe, harmless, inherited, _inherited_receipt = (
        _load_and_validate_inputs(
            plan_path=plan_path,
            harmful_panel_path=harmful_panel_path,
            safe_panel_path=safe_panel_path,
            harmless_material_path=harmless_material_path,
            inherited_material_path=inherited_material_path,
            inherited_material_receipt_path=inherited_material_receipt_path,
        )
    )
    if tokenizer_revision != plan["pinned_instrument"]["model_revision"]:
        raise ValueError("v2 tokenizer revision drift")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    requests = _request_panels(harmful=harmful, safe=safe, partition=partition)
    wrappers = _harmless_rows(harmless, partition)

    observations: list[dict[str, Any]] = []
    public_manifest: list[dict[str, Any]] = []
    for request_class in REQUEST_CLASSES:
        for index, request_row in enumerate(requests[request_class]):
            request_text = _request_text(request_row)
            wrapper = wrappers[index]
            for material in INJECTED_MATERIALS:
                for injected_token_count in INJECTED_SIZES:
                    scaffold = _material_text(
                        material=material,
                        injected_token_count=injected_token_count,
                        wrapper=wrapper,
                        inherited=inherited,
                    )
                    public_pair, private_pair = build_placement_render_pair(
                        tokenizer=tokenizer,
                        request=request_text,
                        scaffold=scaffold,
                        tokenizer_revision=tokenizer_revision,
                        context_ceiling=CONTEXT_CEILING,
                        generation_budget=GENERATION_BUDGET,
                    )
                    observed_count = public_pair["component_token_counts"]["scaffold"]
                    if observed_count != injected_token_count:
                        raise ValueError("v2 scaffold token-size matching drift")
                    for public_placement, private_placement in PLACEMENT_MAP.items():
                        rendered = private_pair[private_placement]
                        public = {
                            "partition": partition,
                            "request_class": request_class,
                            "request_id": request_row["request_id"],
                            "request_sha256": request_row["request_sha256"],
                            "material": material,
                            "material_sha256": sha256_text(scaffold),
                            "wrapper_id": (
                                wrapper["wrapper_id"]
                                if material == "harmless_structured_wrapper"
                                else None
                            ),
                            "placement": public_placement,
                            "injected_token_count": injected_token_count,
                            "gate_role": _gate_role(
                                material=material,
                                request_class=request_class,
                                injected_token_count=injected_token_count,
                            ),
                            "prompt_sha256": sha256_text(rendered["prompt_text"]),
                            "prompt_token_ids_sha256": sha256_bytes(
                                canonical_json_bytes(rendered["prompt_token_ids"])
                            ),
                            "render_pair_sha256": sha256_bytes(
                                canonical_json_bytes(public_pair)
                            ),
                        }
                        public["trial_id"] = _trial_id(public)
                        public_manifest.append(public)
                        observations.append(
                            {
                                **public,
                                "prompt_text": rendered["prompt_text"],
                                "prompt_token_ids": rendered["prompt_token_ids"],
                                "request_text": request_text,
                                "request_context": request_row.get("context"),
                                "render_receipt": rendered["render_receipt"],
                                "utility_judge": request_row.get("utility_judge"),
                                "source_dataset": request_row.get("source_dataset"),
                            }
                        )

            rendered = _shared_render(tokenizer, request_text)
            public = {
                "partition": partition,
                "request_class": request_class,
                "request_id": request_row["request_id"],
                "request_sha256": request_row["request_sha256"],
                "material": "no_scaffold",
                "material_sha256": None,
                "wrapper_id": None,
                "placement": None,
                "injected_token_count": 0,
                "gate_role": "shared_no_scaffold_baseline",
                "prompt_sha256": rendered["prompt_sha256"],
                "prompt_token_ids_sha256": rendered["prompt_token_ids_sha256"],
                "render_pair_sha256": None,
            }
            public["trial_id"] = _trial_id(public)
            public_manifest.append(public)
            observations.append(
                {
                    **public,
                    "prompt_text": rendered["prompt_text"],
                    "prompt_token_ids": rendered["prompt_token_ids"],
                    "request_text": request_text,
                    "request_context": request_row.get("context"),
                    "render_receipt": None,
                    "utility_judge": request_row.get("utility_judge"),
                    "source_dataset": request_row.get("source_dataset"),
                }
            )

    n = PARTITION_SIZES[partition]
    expected = 99 * n
    if len(observations) != expected or len(
        {row["trial_id"] for row in observations}
    ) != expected:
        raise ValueError("v2 topology observation count or ID drift")
    input_sha256 = {
        "plan": sha256_file(plan_path),
        "harmful_panel": sha256_file(harmful_panel_path),
        "safe_panel": sha256_file(safe_panel_path),
        "harmless_material": sha256_file(harmless_material_path),
        "inherited_material": sha256_file(inherited_material_path),
        "inherited_material_receipt": sha256_file(inherited_material_receipt_path),
    }
    private_payload = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "partition": partition,
        "status": "v2_topology_frozen_no_target_outcomes",
        "input_sha256": input_sha256,
        "tokenizer_revision": tokenizer_revision,
        "context_ceiling": CONTEXT_CEILING,
        "generation_budget": GENERATION_BUDGET,
        "prefill_only": True,
        "observations": observations,
    }
    private_sha256 = _atomic_json(private_output_path, private_payload, mode=0o600)
    receipt = {
        "schema_version": "1.0",
        "study_id": plan["study_id"],
        "partition": partition,
        "status": private_payload["status"],
        "input_sha256": input_sha256,
        "private_topology_sha256": private_sha256,
        "tokenizer_revision": tokenizer_revision,
        "observation_count": len(observations),
        "request_class_counts": dict(
            Counter(row["request_class"] for row in public_manifest)
        ),
        "material_counts": dict(Counter(row["material"] for row in public_manifest)),
        "placement_counts": dict(
            Counter(str(row["placement"]) for row in public_manifest)
        ),
        "injected_token_count_counts": dict(
            Counter(str(row["injected_token_count"]) for row in public_manifest)
        ),
        "gate_role_counts": dict(
            Counter(row["gate_role"] for row in public_manifest)
        ),
        "observation_manifest_sha256": sha256_bytes(
            canonical_json_bytes(public_manifest)
        ),
        "all_trial_ids_unique": True,
        "placements_separate": True,
        "sizes_separate": True,
        "exact_size_matching_passed": True,
        "prefill_only": True,
        "target_generation_authorized": False,
        "target_outcome_exists": False,
        "raw_prompt_or_token_ids_public": False,
    }
    receipt_sha256 = _atomic_json(public_receipt_path, receipt, mode=0o644)
    return {
        "status": receipt["status"],
        "partition": partition,
        "private_topology_path": str(private_output_path),
        "private_topology_sha256": private_sha256,
        "public_receipt_path": str(public_receipt_path),
        "public_receipt_sha256": receipt_sha256,
        "observation_count": len(observations),
        "target_outcome_exists": False,
    }
