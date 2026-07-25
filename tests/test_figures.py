from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.figures import (
    generate_behavior_figures,
    verify_behavior_figures,
)


def test_behavior_figures_are_byte_reproducible(tmp_path: Path) -> None:
    rows = []
    paired_values = []
    for behavior_index in range(20):
        behavior_id = f"behavior-{behavior_index:02d}"
        scores = {
            "base": 0.10 + behavior_index / 1_000,
            "inert_length": 0.12 + behavior_index / 1_000,
            "structural_sham": 0.20 + behavior_index / 1_000,
            "full": 0.50 + behavior_index / 1_000,
        }
        for arm, score in scores.items():
            rows.append(
                {
                    "trial_id": f"{behavior_id}-{arm}",
                    "behavior_id": behavior_id,
                    "category": "synthetic",
                    "arm": arm,
                    "score": score,
                    "binary": score >= 0.5,
                    "truncated": False,
                    "refusal": arm in {"base", "structural_sham"},
                    "divider": arm == "full",
                    "post_divider": arm == "full",
                    "generation_receipt_sha256": "0" * 64,
                    "score_receipt_sha256": "1" * 64,
                }
            )
        paired_values.append({"behavior_id": behavior_id, "full_minus_sham": 0.30})
    gate = {
        "schema_version": "1.0",
        "study_id": "synthetic-test",
        "split": "discovery",
        "n_behaviors": 20,
        "n_turn2_rows": 80,
        "primary_contrast": {
            "estimate": 0.30,
            "bootstrap_95_interval": [0.30, 0.30],
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20260725,
            "continuation_gate_passed": True,
        },
        "paired_values": paired_values,
        "source_receipts": rows,
        "phase_source_receipts": [
            {**row, "turn": turn}
            for turn in (1, 2)
            for row in rows
        ],
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True))
    output_dir = tmp_path / "figures"

    generated = generate_behavior_figures(gate_path, output_dir)
    verified = verify_behavior_figures(gate_path, output_dir)

    assert generated["figures"] == 2
    assert verified["status"] == "verified"
    assert len(verified["comparisons"]) == 6
    assert all(item["byte_identical"] for item in verified["comparisons"])
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert [item["path"].split("/")[-1] for item in provenance["figures"]] == [
        "E01-full-vs-sham.receipt.json",
        "E02-response-phases.receipt.json",
    ]
    for item in provenance["figures"]:
        receipt = json.loads(Path(item["path"]).read_text())
        assert receipt["verification"]["status"] == "verified"
        assert all(
            comparison["byte_identical"]
            for comparison in receipt["verification"]["byte_identity"]
        )
