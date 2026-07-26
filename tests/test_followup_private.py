from __future__ import annotations

import csv
from pathlib import Path

from lexical_prompt_study.followup_private import (
    _partition_benign,
    _partition_harmful,
)
from lexical_prompt_study.hashing import sha256_text
from lexical_prompt_study.plans import CATEGORIES


def test_private_harmful_reserve_partition_is_deterministic_and_disjoint() -> None:
    rows = []
    index = 0
    for category in CATEGORIES:
        for offset in range(4):
            request = f"safe reserve request {category} {offset}"
            rows.append(
                {
                    "behavior_id": f"SAFE-H-{index:03d}",
                    "category": category,
                    "goal": request,
                    "goal_sha256": sha256_text(request),
                    "source": "safe",
                    "split": "reserve",
                }
            )
            index += 1
    partitions = _partition_harmful({"harmful_behaviors": rows})
    assert len(partitions["calibration"]) == 20
    assert len(partitions["adaptive_stress"]) == 20
    calibration = {row["behavior_id"] for row in partitions["calibration"]}
    stress = {row["behavior_id"] for row in partitions["adaptive_stress"]}
    assert calibration.isdisjoint(stress)


def test_private_benign_partition_uses_all_rows_once(tmp_path: Path) -> None:
    path = tmp_path / "benign.csv"
    fieldnames = ["Index", "Goal", "Target", "Behavior", "Category", "Source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        index = 0
        for category in CATEGORIES:
            for offset in range(10):
                writer.writerow(
                    {
                        "Index": index,
                        "Goal": f"safe benign request {category} {offset}",
                        "Target": "",
                        "Behavior": "safe",
                        "Category": category,
                        "Source": "safe",
                    }
                )
                index += 1
    partitions = _partition_benign(path)
    calibration = {
        row["behavior_id"] for row in partitions["utility_calibration"]
    }
    confirmatory = {
        row["behavior_id"] for row in partitions["utility_confirmatory"]
    }
    assert len(calibration) == len(confirmatory) == 50
    assert calibration.isdisjoint(confirmatory)
