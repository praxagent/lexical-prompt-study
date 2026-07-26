from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.hashing import sha256_file
from lexical_prompt_study.intervention_figures import (
    generate_intervention_figures,
    verify_intervention_figures,
)

ROOT = Path(__file__).parents[1]
ANALYSIS = ROOT / "results" / "gate4.calibration.discovery.json"
PLAN = ROOT / "plans" / "gate4_intervention_v1.public.json"


def test_gate4_stop_figure_is_receipt_backed_and_byte_verifiable(
    tmp_path: Path,
) -> None:
    result = generate_intervention_figures(ANALYSIS, PLAN, tmp_path)
    assert result["figures"] == 1

    receipt_path = tmp_path / "E05a-discovery-calibration-stop.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["figure_id"] == "E05a"
    assert receipt["derived_data"]["status"] == "stopped_no_eligible_alpha"
    assert receipt["derived_data"]["selection"] is None
    assert receipt["derived_data"]["confirmatory_outcomes_opened"] is False
    assert len(receipt["derived_data"]["candidates"]) == 4
    assert receipt["source_receipts"] == [
        {"path": str(ANALYSIS), "sha256": sha256_file(ANALYSIS)},
        {"path": str(PLAN), "sha256": sha256_file(PLAN)},
    ]
    assert all(
        Path(item["path"]).exists() and sha256_file(Path(item["path"])) == item["sha256"]
        for item in receipt["outputs"].values()
    )

    verification = verify_intervention_figures(ANALYSIS, PLAN, tmp_path)
    assert verification["status"] == "verified"
    assert all(item["byte_identical"] for item in verification["comparisons"])
