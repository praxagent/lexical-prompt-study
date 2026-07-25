from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .hashing import canonical_json_bytes, sha256_bytes
from .models import TrialReceipt


def stable_trial_id(study_id: str, behavior_id: str, arm: str, turn: int, seed: int) -> str:
    raw = f"{study_id}|{behavior_id}|{arm}|{turn}|{seed}".encode()
    return sha256_bytes(raw)[:24]


class ReceiptStore:
    """One atomic JSON file per trial plus an append-only attempt ledger."""

    def __init__(self, root: Path):
        self.root = root
        self.trials = root / "trials"
        self.attempts = root / "attempts.jsonl"
        self.trials.mkdir(parents=True, exist_ok=True)

    def completed_ids(self) -> set[str]:
        completed: set[str] = set()
        for path in self.trials.glob("*.json"):
            try:
                receipt = TrialReceipt.model_validate_json(path.read_text())
            except Exception:
                continue
            if receipt.error is None:
                completed.add(receipt.trial_id)
        return completed

    def write(self, receipt: TrialReceipt) -> str:
        payload = canonical_json_bytes(receipt.model_dump(mode="json"))
        final = self.trials / f"{receipt.trial_id}.json"
        temporary = self.trials / f".{receipt.trial_id}.{os.getpid()}.tmp"
        temporary.write_bytes(payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(final)
        with self.attempts.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return sha256_bytes(payload)

    def validate_expected(self, expected: Iterable[str]) -> None:
        expected_set = set(expected)
        completed = self.completed_ids()
        missing = sorted(expected_set - completed)
        unexpected = sorted(completed - expected_set)
        if missing or unexpected:
            raise ValueError(
                f"receipt structure mismatch: missing={missing[:5]} ({len(missing)}), "
                f"unexpected={unexpected[:5]} ({len(unexpected)})"
            )
