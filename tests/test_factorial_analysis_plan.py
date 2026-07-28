from __future__ import annotations

import json
from pathlib import Path

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "factorial_8b_v1.analysis.json"


def test_factorial_analysis_plan_is_exact_and_bound_before_outcomes() -> None:
    payload = json.loads(PLAN.read_text())
    assert payload["status"] == "prospective_exact_analysis_freeze"
    assert payload["execution_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "factorial_8b_v1.execution-receipt.json"
    )
    assert payload["canonical_matrix_receipt_count"] == 420
    assert payload["descriptive_sentinel_receipt_count"] == 2
    assert payload["primary_readout"] == "feature_6779_magnitude"
    assert payload["uncertainty"] == {
        "resampling_unit": "prompt_family_id",
        "class_resampling": (
            "independent_within_request_class_with_all_arms_and_both_placements_preserved"
        ),
        "bootstrap_replicates": 100000,
        "seed": 20260802,
        "simultaneous_vector": (
            "all_request_class_by_placement_paired_components_then_placement_"
            "specific_interactions_in_the_orders_above"
        ),
        "statistic": (
            "maximum_absolute_centered_deviation_from_the_observed_complete-vector_"
            "estimate"
        ),
        "quantile": 0.95,
        "numpy_quantile_method": "higher",
        "interval": "estimate_plus_or_minus_one_common_familywise_critical_value",
        "interval_role": "fixed_panel_stability_not_population_inference",
    }
    assert payload["literal_sentinels"]["inferential_pooling"] is False
    assert payload["literal_sentinels"]["threshold_fit"] is False
    assert "placement_pooling" in payload["forbidden"]
    assert "held_out_confirmation_opening" in payload["forbidden"]
