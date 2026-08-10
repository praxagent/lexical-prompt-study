from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from .followup_runner import build_placement_render_pair
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text
from .weaponization_plan import (
    validate_weaponization_plan,
    weaponization_protocol_sha256,
)


PARTITION_SIZES = {"detector_calibration": 100, "detector_confirmation": 300}
PLACEMENT_MAP = {
    "scaffold_before_request": "ep_before_request",
    "scaffold_after_request": "ep_after_request",
}
CRITICAL_INJECTED_STRATA = (
    ("positive_attack_harmful", "harmful_request", "full_attack_scaffold"),
    ("negative_attack_benign", "ordinary_benign_request", "full_attack_scaffold"),
    ("negative_attack_neutral", "minimal_neutral_carrier", "full_attack_scaffold"),
    ("negative_harmless_harmful", "harmful_request", "harmless_structured_wrapper"),
    ("negative_harmless_benign", "ordinary_benign_request", "harmless_structured_wrapper"),
    ("negative_structural_sham_harmful", "harmful_request", "matched_structural_sham"),
    ("negative_inert_harmful", "harmful_request", "inert_length"),
)
SECONDARY_INJECTED_STRATA = (
    ("secondary_harmless_neutral", "minimal_neutral_carrier", "harmless_structured_wrapper"),
)
SHARED_STRATA = (
    ("shared_no_scaffold_harmful", "harmful_request"),
    ("shared_no_scaffold_benign", "ordinary_benign_request"),
    ("shared_no_scaffold_neutral", "minimal_neutral_carrier"),
)
CONTEXT_CEILING = 8192
GENERATION_BUDGET = 128


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


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer.encode(text, add_special_tokens=False)
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _normalize_prompt_token_ids(value: Any) -> list[int]:
    """Normalize one unbatched tokenizer result and reject ambiguous shapes."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat-template mapping has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("chat-template input_ids are not a sequence")
    values = list(value)
    if (
        len(values) == 1
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], (str, bytes))
    ):
        values = list(values[0])
    if not values or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in values
    ):
        raise ValueError("chat-template input_ids are not one non-empty integer sequence")
    return [int(item) for item in values]


def _request_text(row: Mapping[str, Any]) -> str:
    context = row.get("context")
    if context:
        return f"Context:\n{context}\n\nRequest:\n{row['request']}"
    return str(row["request"])


def _load_inputs(
    *,
    public_plan_path: Path,
    harmful_panel_path: Path,
    safe_panel_path: Path,
    harmless_material_path: Path,
    factorial_material_path: Path,
    factorial_material_receipt_path: Path,
) -> tuple[dict[str, Any], ...]:
    public_plan = json.loads(public_plan_path.read_text())
    validate_weaponization_plan(
        public_plan,
        root=public_plan_path.resolve().parents[1],
        validate_topology_bindings=False,
    )
    bindings = public_plan["input_freeze_receipts"]
    harmful = json.loads(harmful_panel_path.read_text())
    safe = json.loads(safe_panel_path.read_text())
    harmless = json.loads(harmless_material_path.read_text())
    factorial = json.loads(factorial_material_path.read_text())
    factorial_receipt = json.loads(factorial_material_receipt_path.read_text())
    if sha256_file(harmful_panel_path) != bindings["harmful_request_panel"][
        "private_panel_sha256"
    ]:
        raise ValueError("weaponization harmful private panel hash drift")
    if sha256_file(safe_panel_path) != bindings["safe_request_panel"][
        "private_panel_sha256"
    ]:
        raise ValueError("weaponization safe private panel hash drift")
    preview = bindings["harmless_scaffold_preview"]
    if sha256_file(harmless_material_path) != preview["private_material_sha256"]:
        raise ValueError("weaponization harmless private material hash drift")
    if sha256_file(factorial_material_path) != factorial_receipt[
        "private_material_source_sha256"
    ]:
        raise ValueError("weaponization inherited factorial material hash drift")
    return public_plan, harmful, safe, harmless, factorial, factorial_receipt


def _material_texts(factorial: Mapping[str, Any]) -> dict[str, str]:
    inherited = factorial["scaffold_materials"]
    mapping = {
        "full_attack_scaffold": "full_scaffold",
        "matched_structural_sham": "structural_sham",
        "inert_length": "inert_length",
    }
    output = {}
    for public_name, private_name in mapping.items():
        blocks = inherited[private_name]["blocks"]
        output[public_name] = str(factorial["material_block_joiner"]).join(
            str(block["text"]) for block in blocks
        )
    return output


def _request_panels(
    *, harmful: Mapping[str, Any], safe: Mapping[str, Any], partition: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        "harmful_request": list(harmful["panels"][partition]),
        "ordinary_benign_request": list(
            safe["partitions"][partition]["ordinary_benign_request"]
        ),
        "minimal_neutral_carrier": list(
            safe["partitions"][partition]["minimal_neutral_carrier"]
        ),
    }


def _harmless_wrappers(
    harmless: Mapping[str, Any], partition: str
) -> list[dict[str, Any]]:
    rows = [row for row in harmless["wrappers"] if row["partition"] == partition]
    rows.sort(key=lambda row: row["wrapper_id"])
    expected = PARTITION_SIZES[partition] if partition == "detector_confirmation" else 300
    if len(rows) != expected:
        raise ValueError("harmless wrapper partition topology drift")
    return rows


def _wrapper_text(row: Mapping[str, Any]) -> str:
    return "".join(str(block["text"]) for block in row["blocks"])


def _shared_render(tokenizer: Any, request: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": request}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    prompt_ids = _normalize_prompt_token_ids(prompt_ids)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if len(prompt_ids) + GENERATION_BUDGET > CONTEXT_CEILING:
        raise ValueError("weaponization no-scaffold render exceeds context ceiling")
    return {
        "prompt_text": prompt_text,
        "prompt_token_ids": prompt_ids,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_token_ids_sha256": sha256_bytes(canonical_json_bytes(prompt_ids)),
        "total_tokens": len(prompt_ids),
    }


def _trial_id(public: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(public))[:24]


def build_weaponization_topology(
    *,
    public_plan_path: Path,
    harmful_panel_path: Path,
    safe_panel_path: Path,
    harmless_material_path: Path,
    factorial_material_path: Path,
    factorial_material_receipt_path: Path,
    tokenizer_path: Path,
    tokenizer_revision: str,
    partition: str,
    private_output_path: Path,
    public_receipt_path: Path,
    allow_unreviewed_preview: bool = False,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if partition not in PARTITION_SIZES:
        raise ValueError("unknown weaponization topology partition")
    public_plan, harmful, safe, harmless, factorial, factorial_receipt = _load_inputs(
        public_plan_path=public_plan_path,
        harmful_panel_path=harmful_panel_path,
        safe_panel_path=safe_panel_path,
        harmless_material_path=harmless_material_path,
        factorial_material_path=factorial_material_path,
        factorial_material_receipt_path=factorial_material_receipt_path,
    )
    review_approved = harmless["human_semantic_review"]["approved"] is True
    if not review_approved and not allow_unreviewed_preview:
        raise ValueError("human harmless-scaffold review blocks target topology freeze")
    if tokenizer_revision != public_plan["pinned_instrument"]["model_revision"]:
        raise ValueError("weaponization tokenizer revision drift")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    inherited_materials = _material_texts(factorial)
    requests = _request_panels(harmful=harmful, safe=safe, partition=partition)
    n = PARTITION_SIZES[partition]
    if any(len(rows) < n for rows in requests.values()):
        raise ValueError("weaponization request panel count drift")
    wrappers = _harmless_wrappers(harmless, partition)
    if partition == "detector_calibration":
        wrapper_groups = {
            "harmful_request": wrappers[:n],
            "ordinary_benign_request": wrappers[n : 2 * n],
            "minimal_neutral_carrier": wrappers[2 * n : 3 * n],
        }
    else:
        wrapper_groups = {request_class: wrappers for request_class in requests}

    observations: list[dict[str, Any]] = []
    public_manifest: list[dict[str, Any]] = []
    for stratum, request_class, material in CRITICAL_INJECTED_STRATA + SECONDARY_INJECTED_STRATA:
        request_rows = requests[request_class][:n]
        if material == "harmless_structured_wrapper":
            material_rows: Sequence[dict[str, Any] | None] = wrapper_groups[request_class]
        else:
            material_rows = [None] * n
        for index, (request_row, wrapper_row) in enumerate(zip(request_rows, material_rows)):
            scaffold = (
                _wrapper_text(wrapper_row)
                if wrapper_row is not None
                else inherited_materials[material]
            )
            request_text = _request_text(request_row)
            public_pair, private_pair = build_placement_render_pair(
                tokenizer=tokenizer,
                request=request_text,
                scaffold=scaffold,
                tokenizer_revision=tokenizer_revision,
                context_ceiling=CONTEXT_CEILING,
                generation_budget=GENERATION_BUDGET,
            )
            if public_pair["component_token_counts"]["scaffold"] != 252:
                raise ValueError("weaponization scaffold canonical token count drift")
            for public_placement, private_placement in PLACEMENT_MAP.items():
                rendered = private_pair[private_placement]
                public = {
                    "partition": partition,
                    "stratum": stratum,
                    "critical": stratum.startswith("positive_")
                    or stratum.startswith("negative_"),
                    "request_class": request_class,
                    "request_id": request_row["request_id"],
                    "request_sha256": request_row["request_sha256"],
                    "material": material,
                    "material_sha256": sha256_text(scaffold),
                    "wrapper_id": wrapper_row["wrapper_id"] if wrapper_row else None,
                    "placement": public_placement,
                    "injected_token_count": 252,
                    "prompt_sha256": sha256_text(rendered["prompt_text"]),
                    "prompt_token_ids_sha256": sha256_bytes(
                        canonical_json_bytes(rendered["prompt_token_ids"])
                    ),
                    "render_pair_sha256": sha256_bytes(canonical_json_bytes(public_pair)),
                }
                trial_id = _trial_id(public)
                public["trial_id"] = trial_id
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

    for stratum, request_class in SHARED_STRATA:
        for request_row in requests[request_class][:n]:
            request_text = _request_text(request_row)
            rendered = _shared_render(tokenizer, request_text)
            public = {
                "partition": partition,
                "stratum": stratum,
                "critical": False,
                "request_class": request_class,
                "request_id": request_row["request_id"],
                "request_sha256": request_row["request_sha256"],
                "material": "no_scaffold",
                "material_sha256": None,
                "wrapper_id": None,
                "placement": None,
                "injected_token_count": 0,
                "prompt_sha256": rendered["prompt_sha256"],
                "prompt_token_ids_sha256": rendered["prompt_token_ids_sha256"],
                "render_pair_sha256": None,
            }
            trial_id = _trial_id(public)
            public["trial_id"] = trial_id
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

    expected = 19 * n
    if len(observations) != expected or len({row["trial_id"] for row in observations}) != expected:
        raise ValueError("weaponization topology observation count or ID drift")
    private_payload = {
        "schema_version": "1.0",
        "study_id": public_plan["study_id"],
        "partition": partition,
        "status": (
            "topology_preview_pending_human_review"
            if not review_approved
            else "topology_frozen_human_reviewed"
        ),
        "protocol_sha256": weaponization_protocol_sha256(public_plan),
        "input_sha256": {
            "harmful_panel": sha256_file(harmful_panel_path),
            "safe_panel": sha256_file(safe_panel_path),
            "harmless_material": sha256_file(harmless_material_path),
            "factorial_material": sha256_file(factorial_material_path),
            "factorial_material_receipt": sha256_file(factorial_material_receipt_path),
        },
        "tokenizer_revision": tokenizer_revision,
        "context_ceiling": CONTEXT_CEILING,
        "generation_budget": GENERATION_BUDGET,
        "prefill_only": True,
        "human_semantic_review_approved": review_approved,
        "observations": observations,
    }
    private_sha256 = _atomic_json(private_output_path, private_payload, mode=0o600)
    stratum_counts = dict(Counter(row["stratum"] for row in public_manifest))
    receipt = {
        "schema_version": "1.0",
        "study_id": public_plan["study_id"],
        "partition": partition,
        "status": private_payload["status"],
        "protocol_sha256": private_payload["protocol_sha256"],
        "private_topology_sha256": private_sha256,
        "input_sha256": private_payload["input_sha256"],
        "tokenizer_revision": tokenizer_revision,
        "observation_count": len(observations),
        "stratum_counts": stratum_counts,
        "observation_manifest_sha256": sha256_bytes(canonical_json_bytes(public_manifest)),
        "all_trial_ids_unique": len({row["trial_id"] for row in observations})
        == len(observations),
        "placements_separate": True,
        "canonical_injected_token_count": 252,
        "prefill_only": True,
        "human_semantic_review_approved": review_approved,
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
        "human_semantic_review_approved": review_approved,
        "target_outcome_exists": False,
    }
