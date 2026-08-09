from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from lexical_prompt_study.hashing import sha256_text
from lexical_prompt_study.weaponization_requests import build_harmful_request_panels


FIELDS = [
    "Behavior",
    "FunctionalCategory",
    "SemanticCategory",
    "Tags",
    "ContextString",
    "BehaviorID",
]


def _write_source(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts = {"standard": 201, "contextual": 100, "copyright": 99}
    index = 0
    for functional, count in counts.items():
        for local in range(count):
            rows.append(
                {
                    "Behavior": f"Synthetic safe request {index}",
                    "FunctionalCategory": functional,
                    "SemanticCategory": f"category-{local % 7}",
                    "Tags": "synthetic",
                    "ContextString": f"Synthetic context {index}"
                    if functional == "contextual"
                    else "",
                    "BehaviorID": f"synthetic-{index:03d}",
                }
            )
            index += 1
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_weaponization_request_builder_is_disjoint_and_text_private(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "behaviors.csv"
    rows = _write_source(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "lexical_prompt_study.weaponization_requests.SOURCE_FILE_SHA256", source_hash
    )
    predecessor = tmp_path / "prior.private.json"
    predecessor.write_text(json.dumps({"request_sha256": sha256_text(rows[0]["Behavior"])}))
    private_out = tmp_path / "panels.private.json"
    receipt = tmp_path / "receipt.json"

    result = build_harmful_request_panels(
        source_csv_path=source,
        predecessor_private_paths=[predecessor],
        private_output_path=private_out,
        public_receipt_path=receipt,
    )
    assert result["partition_counts"] == {
        "detector_calibration": 100,
        "detector_confirmation": 150,
    }
    private = json.loads(private_out.read_text())
    selected = [row for panel in private["panels"].values() for row in panel]
    assert len(selected) == 250
    assert len({row["request_sha256"] for row in selected}) == 250
    assert sha256_text(rows[0]["Behavior"]) not in {
        row["request_sha256"] for row in selected
    }
    public_text = receipt.read_text()
    assert "Synthetic safe request" not in public_text
    assert json.loads(public_text)["source_overlap_rows_excluded"] == 1


def test_weaponization_request_builder_rejects_source_hash_drift(tmp_path) -> None:
    source = tmp_path / "behaviors.csv"
    _write_source(source)
    predecessor = tmp_path / "prior.private.json"
    predecessor.write_text("{}")
    with pytest.raises(ValueError, match="source hash drift"):
        build_harmful_request_panels(
            source_csv_path=source,
            predecessor_private_paths=[predecessor],
            private_output_path=tmp_path / "private.json",
            public_receipt_path=tmp_path / "receipt.json",
        )
