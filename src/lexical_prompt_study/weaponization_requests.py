from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_text


SOURCE_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
SOURCE_FILE_SHA256 = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"
SECONDARY_SOURCE_FILE_SHA256 = (
    "0e635254df9065ae6f8d074534b660fa7a361d1ad269e2c8e1b7377961802a86"
)
HARMBENCH_PARTITION_COUNTS = {
    "detector_calibration": {"standard": 47, "contextual": 23},
    "detector_confirmation": {"standard": 144, "contextual": 70},
}
ADVBENCH_PARTITION_COUNTS = {
    "detector_calibration": 30,
    "detector_confirmation": 86,
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


def _collect_request_hashes(value: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "request_sha256" and isinstance(child, str) and len(child) == 64:
                hashes.add(child)
            else:
                hashes.update(_collect_request_hashes(child))
    elif isinstance(value, list):
        for child in value:
            hashes.update(_collect_request_hashes(child))
    return hashes


def _largest_remainder_quotas(counts: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError("invalid stratified target")
    raw = {key: counts[key] * target / total for key in counts}
    quotas = {key: math.floor(raw[key]) for key in counts}
    remaining = target - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    if any(quotas[key] > counts[key] for key in counts):
        raise ValueError("stratified quota exceeds available rows")
    return quotas


def _row_hash(row: Mapping[str, str]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "Behavior": row["Behavior"],
                "ContextString": row["ContextString"],
                "BehaviorID": row["BehaviorID"],
                "FunctionalCategory": row["FunctionalCategory"],
                "SemanticCategory": row["SemanticCategory"],
            }
        )
    )


def _select_partition(
    available: dict[str, list[dict[str, str]]],
    *,
    targets: Mapping[str, int],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for functional, target in targets.items():
        candidates = available[functional]
        by_semantic: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            by_semantic[row["SemanticCategory"]].append(row)
        for rows in by_semantic.values():
            rows.sort(key=lambda row: (_row_hash(row), row["BehaviorID"]))
        quotas = _largest_remainder_quotas(
            {key: len(rows) for key, rows in by_semantic.items()}, target
        )
        functional_selected: list[dict[str, str]] = []
        for semantic in sorted(by_semantic):
            take = quotas[semantic]
            functional_selected.extend(by_semantic[semantic][:take])
            del by_semantic[semantic][:take]
        selected.extend(functional_selected)
        selected_hashes = {_row_hash(row) for row in functional_selected}
        available[functional] = [
            row for row in candidates if _row_hash(row) not in selected_hashes
        ]
    return sorted(selected, key=lambda row: (_row_hash(row), row["BehaviorID"]))


def build_harmful_request_panels(
    *,
    source_csv_path: Path,
    secondary_source_csv_path: Path,
    predecessor_private_paths: Sequence[Path],
    private_output_path: Path,
    public_receipt_path: Path,
) -> dict[str, Any]:
    if sha256_file(source_csv_path) != SOURCE_FILE_SHA256:
        raise ValueError("pinned HarmBench source hash drift")
    if sha256_file(secondary_source_csv_path) != SECONDARY_SOURCE_FILE_SHA256:
        raise ValueError("pinned AdvBench source hash drift")
    excluded_hashes: set[str] = set()
    predecessor_hashes: dict[str, str] = {}
    for path in predecessor_private_paths:
        payload = json.loads(path.read_text())
        excluded_hashes.update(_collect_request_hashes(payload))
        predecessor_hashes[str(path)] = sha256_file(path)

    with source_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = {
            "Behavior",
            "FunctionalCategory",
            "SemanticCategory",
            "Tags",
            "ContextString",
            "BehaviorID",
        }
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError("HarmBench source schema drift")
        source_rows = [dict(row) for row in reader]
    if len(source_rows) != 400 or len({row["BehaviorID"] for row in source_rows}) != 400:
        raise ValueError("HarmBench source row topology drift")

    available: dict[str, list[dict[str, str]]] = {"standard": [], "contextual": []}
    eligible_by_request_hash: dict[str, dict[str, str]] = {}
    overlap_count = 0
    for row in source_rows:
        functional = row["FunctionalCategory"]
        if functional not in available:
            continue
        request_hash = sha256_text(row["Behavior"])
        if request_hash in excluded_hashes:
            overlap_count += 1
            continue
        current = eligible_by_request_hash.get(request_hash)
        if current is None or (_row_hash(row), row["BehaviorID"]) < (
            _row_hash(current),
            current["BehaviorID"],
        ):
            eligible_by_request_hash[request_hash] = row
    duplicate_source_rows_excluded = (
        sum(
            row["FunctionalCategory"] in available
            and sha256_text(row["Behavior"]) not in excluded_hashes
            for row in source_rows
        )
        - len(eligible_by_request_hash)
    )
    for row in eligible_by_request_hash.values():
        available[row["FunctionalCategory"]].append(row)

    panels: dict[str, list[dict[str, Any]]] = {}
    selected_request_hashes: set[str] = set()
    for partition in ("detector_calibration", "detector_confirmation"):
        selected = _select_partition(
            available, targets=HARMBENCH_PARTITION_COUNTS[partition]
        )
        expected = sum(HARMBENCH_PARTITION_COUNTS[partition].values())
        if len(selected) != expected:
            raise ValueError(f"{partition}: insufficient disjoint harmful rows")
        panels[partition] = [
            {
                "request_id": row["BehaviorID"],
                "request": row["Behavior"],
                "request_sha256": sha256_text(row["Behavior"]),
                "context": row["ContextString"] or None,
                "context_sha256": sha256_text(row["ContextString"])
                if row["ContextString"]
                else None,
                "functional_category": row["FunctionalCategory"],
                "semantic_category": row["SemanticCategory"],
                "source_dataset": "harmbench",
                "source_row_sha256": _row_hash(row),
            }
            for row in selected
        ]
        selected_request_hashes.update(row["request_sha256"] for row in panels[partition])

    with secondary_source_csv_path.open(newline="") as handle:
        secondary_reader = csv.DictReader(handle)
        expected_secondary_fields = {
            "Behavior",
            "Category",
            "Tags",
            "ContextString",
            "BehaviorID",
        }
        if set(secondary_reader.fieldnames or ()) != expected_secondary_fields:
            raise ValueError("AdvBench source schema drift")
        secondary_raw = [dict(row) for row in secondary_reader]
    if len(secondary_raw) != 520 or len({row["BehaviorID"] for row in secondary_raw}) != 520:
        raise ValueError("AdvBench source row topology drift")
    secondary_rows: list[dict[str, str]] = []
    secondary_overlap_count = 0
    secondary_seen: set[str] = set()
    for raw in secondary_raw:
        request_hash = sha256_text(raw["Behavior"])
        if request_hash in excluded_hashes or request_hash in selected_request_hashes:
            secondary_overlap_count += 1
            continue
        if request_hash in secondary_seen:
            raise ValueError("duplicate AdvBench request bytes")
        secondary_seen.add(request_hash)
        secondary_rows.append(
            {
                "Behavior": raw["Behavior"],
                "FunctionalCategory": "advbench",
                "SemanticCategory": raw["Category"] or "advbench",
                "Tags": raw["Tags"],
                "ContextString": raw["ContextString"],
                "BehaviorID": raw["BehaviorID"],
            }
        )
    secondary_rows.sort(key=lambda row: (_row_hash(row), row["BehaviorID"]))
    offset = 0
    for partition in ("detector_calibration", "detector_confirmation"):
        count = ADVBENCH_PARTITION_COUNTS[partition]
        selected = secondary_rows[offset : offset + count]
        if len(selected) != count:
            raise ValueError(f"{partition}: insufficient disjoint AdvBench rows")
        offset += count
        panels[partition].extend(
            {
                "request_id": row["BehaviorID"],
                "request": row["Behavior"],
                "request_sha256": sha256_text(row["Behavior"]),
                "context": None,
                "context_sha256": None,
                "functional_category": row["FunctionalCategory"],
                "semantic_category": row["SemanticCategory"],
                "source_dataset": "advbench",
                "source_row_sha256": _row_hash(row),
            }
            for row in selected
        )
        panels[partition].sort(
            key=lambda row: (row["source_row_sha256"], row["request_id"])
        )

    all_selected = [row for rows in panels.values() for row in rows]
    if len(all_selected) != 400 or len({row["request_sha256"] for row in all_selected}) != 400:
        raise ValueError("harmful panel request uniqueness drift")
    if {row["request_sha256"] for row in all_selected} & excluded_hashes:
        raise ValueError("harmful panel overlaps a protected predecessor request")

    private_payload = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-weaponization-breaker-v1",
        "sources": [
            {
                "repository": "https://github.com/centerforaisafety/HarmBench.git",
                "revision": SOURCE_REVISION,
                "path": "data/behavior_datasets/harmbench_behaviors_text_all.csv",
                "sha256": SOURCE_FILE_SHA256,
            },
            {
                "repository": "https://github.com/centerforaisafety/HarmBench.git",
                "revision": SOURCE_REVISION,
                "path": "data/behavior_datasets/extra_behavior_datasets/advbench_behaviors.csv",
                "sha256": SECONDARY_SOURCE_FILE_SHA256,
            },
        ],
        "predecessor_private_sha256": predecessor_hashes,
        "selection_rule": "largest-remainder semantic-category quotas within frozen functional strata, then ascending source-row SHA-256 and behavior ID",
        "panels": panels,
    }
    private_sha256 = _atomic_json(private_output_path, private_payload, mode=0o600)
    public_manifest = {
        partition: [
            {
                "request_id": row["request_id"],
                "request_sha256": row["request_sha256"],
                "context_sha256": row["context_sha256"],
                "functional_category": row["functional_category"],
                "semantic_category": row["semantic_category"],
                "source_dataset": row["source_dataset"],
                "source_row_sha256": row["source_row_sha256"],
            }
            for row in rows
        ]
        for partition, rows in panels.items()
    }
    receipt = {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-weaponization-breaker-v1",
        "status": "harmful_request_panels_frozen_no_target_outcomes",
        "source_revision": SOURCE_REVISION,
        "source_file_sha256": {
            "harmbench": SOURCE_FILE_SHA256,
            "advbench": SECONDARY_SOURCE_FILE_SHA256,
        },
        "predecessor_private_sha256": predecessor_hashes,
        "excluded_predecessor_request_hash_count": len(excluded_hashes),
        "source_overlap_rows_excluded": overlap_count,
        "duplicate_source_rows_excluded": duplicate_source_rows_excluded,
        "secondary_source_overlap_rows_excluded": secondary_overlap_count,
        "partition_counts": {key: len(rows) for key, rows in panels.items()},
        "source_dataset_counts": {
            partition: dict(Counter(row["source_dataset"] for row in rows))
            for partition, rows in panels.items()
        },
        "functional_category_counts": {
            partition: dict(Counter(row["functional_category"] for row in rows))
            for partition, rows in panels.items()
        },
        "semantic_category_counts": {
            partition: dict(Counter(row["semantic_category"] for row in rows))
            for partition, rows in panels.items()
        },
        "panel_manifest_sha256": {
            partition: sha256_bytes(canonical_json_bytes(rows))
            for partition, rows in public_manifest.items()
        },
        "private_panel_sha256": private_sha256,
        "raw_request_text_public": False,
        "target_generation_performed": False,
        "target_outcome_exists": False,
    }
    receipt_sha256 = _atomic_json(public_receipt_path, receipt, mode=0o644)
    return {
        "status": receipt["status"],
        "private_panel_path": str(private_output_path),
        "private_panel_sha256": private_sha256,
        "public_receipt_path": str(public_receipt_path),
        "public_receipt_sha256": receipt_sha256,
        "partition_counts": receipt["partition_counts"],
        "target_outcome_exists": False,
    }
