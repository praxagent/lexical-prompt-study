from __future__ import annotations

import json
import math
from pathlib import Path

from lexical_prompt_study.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "compute-reconciliation.weaponization.json"


def test_weaponization_reconciliation_binds_receipts_and_arithmetic() -> None:
    payload = json.loads(RESULT.read_text())
    sources = payload["source_receipts"]
    assert sources["calibration_execution_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "weaponization_calibration_a090.execution-receipt.json"
    )
    assert sources["confirmation_execution_receipt_sha256"] == sha256_file(
        ROOT / "validation" / "weaponization_confirmation_a096.execution-receipt.json"
    )
    ingested = payload["task_owned_provider_ingestion"]["provider_ingested_amount_usd"]
    adjustments = sum(
        row["additional_reserved_amount_usd"]
        for row in payload["conservative_recent_run_adjustments"]
    )
    gpu = payload["conservative_gpu_compute_ceiling_usd"]
    volume = payload["retained_network_volume"]["estimated_accrual_through_query_usd"]
    assert math.isclose(gpu, ingested + adjustments, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        payload["conservative_estimated_task_infrastructure_ceiling_usd"],
        gpu + volume,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert payload["active_pod_inventory"] == []
    assert payload["budget"]["hard_ceiling_respected"] is True
