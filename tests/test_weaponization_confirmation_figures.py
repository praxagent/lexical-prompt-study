from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.weaponization_confirmation_figures import (
    generate_weaponization_confirmation_figures,
    verify_weaponization_confirmation_figures,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "weaponization-confirmation-v1.public.json"
CALIBRATION = ROOT / "results" / "weaponization-calibration-v1.public.json"


def test_confirmation_figures_are_receipt_backed(tmp_path: Path) -> None:
    generated = generate_weaponization_confirmation_figures(RESULT, CALIBRATION, tmp_path)
    assert generated["status"] == "generated"
    assert generated["figure_count"] == 3
    receipt = json.loads(
        (tmp_path / "E25-heldout-circuit-breaker-gate.receipt.json").read_text()
    )
    assert receipt["figure_id"] == "E25"
    assert receipt["source_receipts"][0]["sha256"] == (
        "e3e8124b024705f02126051a5fd784ff1abd5c295911f41c6b7324e72f3a6b8a"
    )
    assert all(
        receipt["outputs"][kind]["sha256"] for kind in ("svg", "png", "pdf")
    )


def test_confirmation_figure_outputs_verify_byte_identically(tmp_path: Path) -> None:
    generate_weaponization_confirmation_figures(RESULT, CALIBRATION, tmp_path)
    verified = verify_weaponization_confirmation_figures(RESULT, CALIBRATION, tmp_path)
    assert verified["status"] == "verified"
    assert len(verified["comparisons"]) == 9
