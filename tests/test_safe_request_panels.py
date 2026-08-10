from __future__ import annotations

import json

import pytest

from lexical_prompt_study.safe_request_panels import (
    build_safe_request_panels,
    score_deterministic_utility,
    write_safe_request_panels,
)


def test_safe_request_panels_are_disjoint_and_balanced() -> None:
    payload = build_safe_request_panels()
    rows = []
    for classes in payload["partitions"].values():
        assert {key: len(value) for key, value in classes.items()} == {
            "minimal_neutral_carrier": 300,
            "ordinary_benign_request": 300,
        }
        rows.extend(row for class_rows in classes.values() for row in class_rows)
    assert len(rows) == 1200
    assert len({row["request_sha256"] for row in rows}) == 1200
    calibration = {
        row["request_sha256"]
        for class_rows in payload["partitions"]["detector_calibration"].values()
        for row in class_rows
    }
    confirmation = {
        row["request_sha256"]
        for class_rows in payload["partitions"]["detector_confirmation"].values()
        for row in class_rows
    }
    assert calibration.isdisjoint(confirmation)


def test_v2_safe_requests_are_byte_disjoint_from_v1() -> None:
    first = build_safe_request_panels()
    second = build_safe_request_panels(surface_epoch="v2-a099")
    first_hashes = {
        row["request_sha256"]
        for classes in first["partitions"].values()
        for rows in classes.values()
        for row in rows
    }
    second_hashes = {
        row["request_sha256"]
        for classes in second["partitions"].values()
        for rows in classes.values()
        for row in rows
    }
    assert first_hashes.isdisjoint(second_hashes)
    assert second["surface_epoch"] == "v2-a099"


@pytest.mark.parametrize(
    ("text", "judge", "expected"),
    [
        ("READY", {"kind": "exact_text", "expected": "READY"}, True),
        (" 42 ", {"kind": "integer", "expected": 42}, True),
        ("a, b", {"kind": "comma_list", "expected": ["a", "b"]}, True),
        ('{"item":"amber","count":2}', {"kind": "json", "expected": {"item": "amber", "count": 2}}, True),
        ("extra READY", {"kind": "exact_text", "expected": "READY"}, False),
        ("forty two", {"kind": "integer", "expected": 42}, False),
    ],
)
def test_deterministic_utility_judges(text: str, judge: dict, expected: bool) -> None:
    assert score_deterministic_utility(text, judge) is expected


def test_safe_panel_receipt_contains_no_request_text(tmp_path) -> None:
    private_out = tmp_path / "safe.private.json"
    receipt = tmp_path / "safe.public.json"
    result = write_safe_request_panels(
        private_output_path=private_out,
        public_receipt_path=receipt,
    )
    assert result["target_outcome_exists"] is False
    public_text = receipt.read_text()
    assert "Reply with the exact code" not in public_text
    public = json.loads(public_text)
    assert public["all_requests_unique"] is True
    assert public["all_utility_judges_deterministic"] is True
