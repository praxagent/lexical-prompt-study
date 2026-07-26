from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .followup_plan import validate_followup_plan
from .hashing import canonical_json_bytes, sha256_file, sha256_text, write_json_atomic
from .plans import CATEGORIES, _rank


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _id_list_hash(ids: list[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(sorted(ids))).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _partition_harmful(source_private: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    partitions: dict[str, list[dict[str, Any]]] = {
        "discovery": [],
        "calibration": [],
        "confirmatory": [],
        "adaptive_stress": [],
    }
    reserve: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_private["harmful_behaviors"]:
        reduced = {
            "behavior_id": row["behavior_id"],
            "category": row["category"],
            "request": row["goal"],
            "request_sha256": row["goal_sha256"],
            "source": row["source"],
        }
        if sha256_text(reduced["request"]) != reduced["request_sha256"]:
            raise ValueError(f"{row['behavior_id']}: harmful request hash drift")
        if row["split"] in {"discovery", "confirmatory"}:
            partitions[row["split"]].append(reduced)
        elif row["split"] == "reserve":
            reserve[row["category"]].append(reduced)
        else:
            raise ValueError(f"{row['behavior_id']}: unknown source split")
    if set(reserve) != set(CATEGORIES):
        raise ValueError("reserve category topology drift")
    for category in CATEGORIES:
        rows = sorted(reserve[category], key=lambda item: item["behavior_id"])
        if len(rows) != 4:
            raise ValueError(f"{category}: expected four reserve behaviors")
        partitions["calibration"].extend(rows[:2])
        partitions["adaptive_stress"].extend(rows[2:])
    return partitions


def _partition_benign(benign_csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(benign_csv_path):
        by_category[row["Category"]].append(row)
    if set(by_category) != set(CATEGORIES):
        raise ValueError("benign category topology drift")
    partitions = {"utility_calibration": [], "utility_confirmatory": []}
    for category in CATEGORIES:
        ranked = sorted(by_category[category], key=_rank)
        if len(ranked) != 10:
            raise ValueError(f"{category}: expected ten benign behaviors")
        for partition, rows in (
            ("utility_calibration", ranked[:5]),
            ("utility_confirmatory", ranked[5:]),
        ):
            partitions[partition].extend(
                {
                    "behavior_id": f"JBB-B-{int(row['Index']):03d}",
                    "category": row["Category"],
                    "request": row["Goal"],
                    "request_sha256": sha256_text(row["Goal"]),
                    "source": row["Source"],
                }
                for row in rows
            )
    return partitions


def validate_followup_private_plan(
    private: dict[str, Any],
    *,
    public_plan_path: Path,
    source_private_path: Path,
    source_public_path: Path,
    benign_csv_path: Path,
) -> None:
    public = json.loads(public_plan_path.read_text())
    validate_followup_plan(public)
    if private["schema_version"] != "1.0":
        raise ValueError("follow-up private-plan schema drift")
    if private["study_id"] != public["study_id"]:
        raise ValueError("follow-up private/public study mismatch")
    bindings = private["source_bindings"]
    expected_bindings = {
        "public_plan_sha256": sha256_file(public_plan_path),
        "source_private_plan_sha256": sha256_file(source_private_path),
        "source_public_plan_sha256": sha256_file(source_public_path),
        "benign_csv_sha256": sha256_file(benign_csv_path),
    }
    if bindings != expected_bindings:
        raise ValueError("follow-up private source-binding drift")
    expected_partitions = public["partitions"]
    all_ids: list[str] = []
    for name in (
        "discovery",
        "calibration",
        "confirmatory",
        "adaptive_stress",
        "utility_calibration",
        "utility_confirmatory",
    ):
        rows = private["partitions"][name]
        ids = [row["behavior_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate behavior ID")
        if len(ids) != expected_partitions[name]["count"]:
            raise ValueError(f"{name}: behavior count drift")
        if _id_list_hash(ids) != expected_partitions[name]["sorted_id_list_sha256"]:
            raise ValueError(f"{name}: frozen ID-list hash drift")
        for row in rows:
            if sha256_text(row["request"]) != row["request_sha256"]:
                raise ValueError(f"{row['behavior_id']}: request hash drift")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("behavior ID reused across follow-up partitions")


def build_followup_private_plan(
    *,
    public_plan_path: Path,
    source_private_path: Path,
    source_public_path: Path,
    benign_csv_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    public = json.loads(public_plan_path.read_text())
    validate_followup_plan(public)
    source_private = json.loads(source_private_path.read_text())
    if source_private["public_plan_sha256"] != sha256_file(source_public_path):
        raise ValueError("source study private/public hash drift")
    harmful = _partition_harmful(source_private)
    benign = _partition_benign(benign_csv_path)
    payload = {
        "schema_version": "1.0",
        "study_id": public["study_id"],
        "created_from_source_commit": _source_commit(),
        "source_bindings": {
            "public_plan_sha256": sha256_file(public_plan_path),
            "source_private_plan_sha256": sha256_file(source_private_path),
            "source_public_plan_sha256": sha256_file(source_public_path),
            "benign_csv_sha256": sha256_file(benign_csv_path),
        },
        "partitions": {**harmful, **benign},
    }
    validate_followup_private_plan(
        payload,
        public_plan_path=public_plan_path,
        source_private_path=source_private_path,
        source_public_path=source_public_path,
        benign_csv_path=benign_csv_path,
    )
    digest = write_json_atomic(output_path, payload)
    output_path.chmod(0o600)
    return {
        "status": "complete",
        "private_plan_path": str(output_path),
        "private_plan_sha256": digest,
        "source_commit": payload["created_from_source_commit"],
        "partition_counts": {
            name: len(rows) for name, rows in payload["partitions"].items()
        },
    }
