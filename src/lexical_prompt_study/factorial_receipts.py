from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .followup_design import PLACEMENTS
from .hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from .models import FactorialTrialReceipt


def validate_factorial_trial_receipt(
    value: FactorialTrialReceipt | dict[str, Any],
) -> FactorialTrialReceipt:
    receipt = (
        value
        if isinstance(value, FactorialTrialReceipt)
        else FactorialTrialReceipt.model_validate(value)
    )
    if receipt.material == "no_scaffold":
        if (
            receipt.placement is not None
            or receipt.shared_reference is not True
            or receipt.injected_token_count != 0
            or receipt.render_group_sha256 is not None
            or receipt.size_id != "natural-base"
        ):
            raise ValueError("natural no-scaffold receipt topology drift")
    else:
        literal_sentinel = receipt.request_class == "literal_sentinel"
        if (
            receipt.placement not in PLACEMENTS
            or receipt.shared_reference is not False
            or receipt.injected_token_count <= 0
            or (
                literal_sentinel
                and receipt.render_group_sha256 is not None
            )
            or (
                not literal_sentinel
                and receipt.render_group_sha256 is None
            )
        ):
            raise ValueError("injected factorial receipt topology drift")
    if (
        receipt.request_class == "literal_sentinel"
        and receipt.material != "full_scaffold"
    ):
        raise ValueError("literal sentinel must use the full scaffold")
    if receipt.feature_6779_positive != (receipt.feature_6779_magnitude > 0):
        raise ValueError("feature-6779 magnitude/prevalence inconsistency")
    if receipt.task_completion is not None or receipt.utility_score is not None:
        raise ValueError(
            "generation receipt cannot contain semantic task-completion or utility judgments"
        )
    if receipt.request_class == "harmful_request" and receipt.utility_score is not None:
        raise ValueError("harmful receipt cannot carry benign utility score")
    restricted_path = Path(receipt.restricted_artifact_path)
    if (
        not restricted_path.exists()
        or sha256_file(restricted_path) != receipt.restricted_artifact_sha256
    ):
        raise ValueError("restricted factorial artifact hash drift")
    restricted = __import__("json").loads(restricted_path.read_text())
    if (
        restricted.get("trial_id") != receipt.trial_id
        or restricted.get("attempt") != receipt.attempt
        or restricted.get("request_class") != receipt.request_class
        or restricted.get("request_id") != receipt.request_id
        or restricted.get("prompt_sha256") != receipt.prompt_sha256
        or restricted.get("prompt_token_ids_sha256")
        != receipt.prompt_token_ids_sha256
        or sha256_text(str(restricted.get("generated_text", "")))
        != receipt.generated_text_sha256
        or sha256_bytes(
            canonical_json_bytes(restricted.get("generated_token_ids", []))
        )
        != receipt.generated_token_ids_sha256
    ):
        raise ValueError("restricted factorial content provenance drift")
    return receipt


class FactorialReceiptStore:
    def __init__(self, root: Path):
        self.root = root
        self.trials = root / "trials"
        self.attempts = root / "attempts.jsonl"
        self.trials.mkdir(parents=True, exist_ok=True)

    def write(self, value: FactorialTrialReceipt | dict[str, Any]) -> str:
        receipt = validate_factorial_trial_receipt(value)
        payload = canonical_json_bytes(receipt.model_dump(mode="json"))
        path = self.trials / f"{receipt.trial_id}.json"
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"{receipt.trial_id}: refusing completed receipt overwrite")
            return sha256_bytes(payload)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        with self.attempts.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return sha256_bytes(payload)

    def load_validated(
        self,
        trial_id: str,
        *,
        public_plan_sha256: str,
        private_plan_sha256: str,
        assay_receipt_sha256: str,
        source_commit: str,
        run_id: str,
    ) -> FactorialTrialReceipt | None:
        path = self.trials / f"{trial_id}.json"
        if not path.exists():
            return None
        receipt = validate_factorial_trial_receipt(
            FactorialTrialReceipt.model_validate_json(path.read_text())
        )
        expected = {
            "trial_id": trial_id,
            "public_plan_sha256": public_plan_sha256,
            "private_plan_sha256": private_plan_sha256,
            "assay_receipt_sha256": assay_receipt_sha256,
            "source_commit": source_commit,
            "run_id": run_id,
        }
        for field, value in expected.items():
            if getattr(receipt, field) != value:
                raise ValueError(f"{trial_id}: receipt provenance drift for {field}")
        return receipt
