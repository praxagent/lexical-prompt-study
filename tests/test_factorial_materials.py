from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lexical_prompt_study.factorial_materials import (
    BENIGN_SOURCE_PARTITION,
    HARMFUL_SOURCE_PARTITION,
    NEUTRAL_CARRIERS,
    assemble_factorial_material_source,
)
from lexical_prompt_study.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PLAN = ROOT / "plans" / "factorial_8b_v1.public.json"


def _row(prefix: str, index: int) -> dict[str, str]:
    request = f"safe {prefix} request {index:03d}"
    return {
        "behavior_id": f"{prefix}-{index:03d}",
        "request": request,
        "request_sha256": sha256_text(request),
        "category": "safe",
        "source": "synthetic",
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    followup = {
        "partitions": {
            "discovery": [_row("discovery", index) for index in range(20)],
            "calibration": [_row("calibration", index) for index in range(20)],
            "confirmatory": [_row("confirmatory", index) for index in range(40)],
            HARMFUL_SOURCE_PARTITION: [
                _row("stress", index) for index in range(20)
            ],
            BENIGN_SOURCE_PARTITION: [_row("benign", index) for index in range(50)],
        }
    }
    followup_path = tmp_path / "followup.private.json"
    followup_path.write_bytes(canonical_json_bytes(followup))

    materials = {}
    for material, prefix in (
        ("inert_length", "I"),
        ("structural_sham", "S"),
        ("full_scaffold", "F"),
    ):
        blocks = []
        for index in range(4):
            text = f"{prefix}{index:03d}"
            blocks.append(
                {
                    "block_id": f"block-{index + 1:03d}",
                    "text": text,
                    "text_sha256": sha256_text(text),
                }
            )
        materials[material] = {"blocks": blocks}
    blocks_path = tmp_path / "blocks.private.json"
    blocks_path.write_bytes(
        canonical_json_bytes({"schema_version": "1.0", "scaffold_materials": materials})
    )

    public = json.loads(PUBLIC_PLAN.read_text())
    public["size_factor"]["boundary_feasibility"][
        "compiled_private_blocks_sha256"
    ] = sha256_file(blocks_path)
    public_path = tmp_path / "factorial.public.json"
    public_path.write_bytes(canonical_json_bytes(public))
    return public_path, followup_path, blocks_path


def _assemble(tmp_path: Path) -> dict:
    public, followup, blocks = _write_fixture(tmp_path)
    return assemble_factorial_material_source(
        public_plan_path=public,
        followup_private_path=followup,
        compiled_blocks_path=blocks,
        private_output_path=tmp_path / "factorial-materials.private.json",
        public_receipt_path=tmp_path / "factorial-materials.public.json",
    )


def test_assemble_factorial_material_source_is_result_free_and_private(
    tmp_path: Path,
) -> None:
    result = _assemble(tmp_path)
    assert result["request_panel_counts"] == {
        "minimal_neutral_carrier": 20,
        "ordinary_benign_request": 20,
        "harmful_request": 20,
    }
    assert result["harmful_disjointness_passed"] is True
    assert result["target_outcome_exists"] is False
    private_path = Path(result["private_material_source_path"])
    public_path = Path(result["public_receipt_path"])
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert public_path.stat().st_mode & 0o777 == 0o644
    private = json.loads(private_path.read_text())
    receipt_text = public_path.read_text()
    assert len(private["request_panels"]["minimal_neutral_carrier"]) == len(
        NEUTRAL_CARRIERS
    )
    assert {
        row["request_id"] for row in private["request_panels"]["harmful_request"]
    } == {f"stress-{index:03d}" for index in range(20)}
    assert '"request":' not in receipt_text
    assert "safe benign request" not in receipt_text
    assert "safe stress request" not in receipt_text


def test_benign_selection_is_lowest_request_hash_before_outcomes(
    tmp_path: Path,
) -> None:
    result = _assemble(tmp_path)
    private = json.loads(Path(result["private_material_source_path"]).read_text())
    selected = private["request_panels"]["ordinary_benign_request"]
    expected = sorted(
        (_row("benign", index) for index in range(50)),
        key=lambda row: (row["request_sha256"], row["behavior_id"]),
    )[:20]
    assert [row["request_id"] for row in selected] == [
        row["behavior_id"] for row in expected
    ]


def test_harmful_overlap_with_protected_partition_fails_closed(tmp_path: Path) -> None:
    public, followup_path, blocks = _write_fixture(tmp_path)
    followup = json.loads(followup_path.read_text())
    overlap = copy.deepcopy(followup["partitions"][HARMFUL_SOURCE_PARTITION][0])
    overlap["behavior_id"] = followup["partitions"]["discovery"][0]["behavior_id"]
    followup["partitions"][HARMFUL_SOURCE_PARTITION][0] = overlap
    followup_path.write_bytes(canonical_json_bytes(followup))
    with pytest.raises(ValueError, match="overlaps a protected prior partition"):
        assemble_factorial_material_source(
            public_plan_path=public,
            followup_private_path=followup_path,
            compiled_blocks_path=blocks,
            private_output_path=tmp_path / "factorial-materials.private.json",
            public_receipt_path=tmp_path / "factorial-materials.public.json",
        )


def test_compiled_block_hash_drift_fails_closed(tmp_path: Path) -> None:
    public, followup, blocks = _write_fixture(tmp_path)
    payload = json.loads(blocks.read_text())
    payload["extra"] = sha256_bytes(b"drift")
    blocks.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="do not match the public-plan hash"):
        assemble_factorial_material_source(
            public_plan_path=public,
            followup_private_path=followup,
            compiled_blocks_path=blocks,
            private_output_path=tmp_path / "factorial-materials.private.json",
            public_receipt_path=tmp_path / "factorial-materials.public.json",
        )
