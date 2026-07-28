from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.factorial_dose_figures import (
    generate_factorial_dose_figures,
    verify_factorial_dose_figures,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "factorial-8b-secondary-dose.public.json"


def test_secondary_dose_figures_have_receipts_and_verify(tmp_path: Path) -> None:
    generated = generate_factorial_dose_figures(RESULT, tmp_path)
    assert generated["figures"] == 5
    verified = verify_factorial_dose_figures(RESULT, tmp_path)
    assert verified["status"] == "verified"
    assert len(verified["comparisons"]) == 15
    for receipt_path in tmp_path.glob("*.receipt.json"):
        receipt = json.loads(receipt_path.read_text())
        assert receipt["verification"]["status"] == "verified"
        assert receipt["source_receipts"][0]["sha256"]
