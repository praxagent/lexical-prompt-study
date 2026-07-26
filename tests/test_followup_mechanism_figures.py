import json
from pathlib import Path

from lexical_prompt_study.followup_mechanism_figures import (
    generate_followup_mechanism_figures,
    verify_followup_mechanism_figures,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "g3.followup-mechanism.public.json"


def test_followup_mechanism_figures_are_complete_and_byte_verifiable(
    tmp_path: Path,
) -> None:
    generated = generate_followup_mechanism_figures(RESULT, tmp_path)
    assert generated["figures"] == 3
    arm_receipt = json.loads(
        (tmp_path / "E09-followup-selected-sae-arm-matrix.receipt.json").read_text()
    )
    assert arm_receipt["counts"]["placement_levels"] == 2
    assert arm_receipt["non_claims"][0] == "no common detector threshold"
    trajectory_receipt = json.loads(
        (tmp_path / "E11-followup-jlens-trajectories.receipt.json").read_text()
    )
    assert trajectory_receipt["counts"]["rows"] == 372
    verified = verify_followup_mechanism_figures(RESULT, tmp_path)
    assert verified["status"] == "verified"
    assert len(verified["comparisons"]) == 9
    assert all(row["byte_identical"] for row in verified["comparisons"])
