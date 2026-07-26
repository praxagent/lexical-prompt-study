import json
from pathlib import Path

from lexical_prompt_study.replay_figures import (
    generate_replay_figure,
    verify_replay_figure,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "gate3.sae-four-arm-replay.discovery.json"


def test_replay_figure_is_receipt_backed_and_byte_verifiable(tmp_path: Path) -> None:
    generated = generate_replay_figure(RESULT, tmp_path)
    assert generated["figures"] == 1
    receipt_path = tmp_path / "E05b-feature-10146-four-arm-replay.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["derived_data"]["candidate_gate"]["passed"] is False
    assert receipt["counts"]["realized_primary_observations"] == 80
    verified = verify_replay_figure(RESULT, tmp_path)
    assert verified["status"] == "verified"
    assert all(row["byte_identical"] for row in verified["comparisons"])
