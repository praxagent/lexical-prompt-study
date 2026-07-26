import json
from pathlib import Path

from lexical_prompt_study.followup_patch_figures import (
    generate_followup_patch_figures,
    verify_followup_patch_figures,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "g4.followup-patch-discovery.public.json"
PLAN = ROOT / "plans" / "followup_v2.public.json"


def test_followup_patch_stop_figure_is_complete_and_byte_verifiable(
    tmp_path: Path,
) -> None:
    generated = generate_followup_patch_figures(RESULT, PLAN, tmp_path)
    assert generated["figures"] == 1
    receipt = json.loads(
        (tmp_path / "E12-followup-causal-localization-stop.receipt.json").read_text()
    )
    assert receipt["counts"]["patch_receipts"] == 1800
    assert receipt["counts"]["score_receipts"] == 1800
    assert receipt["counts"]["placements"] == 2
    assert receipt["derived_data"]["eligible_common_layers"] == []
    assert receipt["derived_data"]["selected_common_layer"] is None
    assert len(receipt["derived_data"]["rows"]) == 10
    verified = verify_followup_patch_figures(RESULT, PLAN, tmp_path)
    assert verified["status"] == "verified"
    assert len(verified["comparisons"]) == 3
    assert all(row["byte_identical"] for row in verified["comparisons"])
