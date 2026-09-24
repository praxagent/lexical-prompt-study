"""A192 fresh answer paths and fixed-cohort analysis; pure standard library.

The coordinator authenticates native template/token geometry and full FP32
readout bytes. Hashes supplied here declare those already checked readouts.
No function executes a tokenizer/model or interprets free-generation evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

NAMESPACE = "a192-answer-path-ranking-v1"
SCHEMA = "a192-answer-path-ranking-tasks-v1"
CORE_COUNT = 8
ROW_COUNT = 16
CALL_COUNT = 64
CONDITIONS = ("computed", "supplied")
CANDIDATES = ("correct", "first_sum_plus_one")
MAX_PROMPT_TOKENS = 256
MAX_CANDIDATE_TOKENS = 16
EOT_TOKEN = "<|eot_id|>"
SYSTEM_INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "If the answer field is ?, add the two integers; otherwise copy the supplied answer. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
IDENTITY_KEYS = (
    "observation_id",
    "core_index",
    "core_id",
    "group_id",
    "condition",
    "sequence_index",
)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def operand(core_index, item_index, side):
    _require(type(core_index) is int and 0 <= core_index < CORE_COUNT, "core_index")
    _require(type(item_index) is int and item_index in (0, 1), "item_index")
    _require(type(side) is int and side in (0, 1), "operand_side")
    raw = hashlib.sha256(f"{NAMESPACE}|{core_index}|{item_index}|{side}".encode("ascii")).digest()
    magnitude = 10 + int.from_bytes(raw, "big") % 90
    return -magnitude if raw[0] & 1 else magnitude


def core_operands(core_index):
    return [[operand(core_index, item, side) for side in (0, 1)] for item in (0, 1)]


def _strata(pairs):
    return {
        key: str(value)
        for key, value in {
            "positive_positive_items": sum(a > 0 and b > 0 for a, b in pairs),
            "negative_negative_items": sum(a < 0 and b < 0 for a, b in pairs),
            "opposite_sign_items": sum(a * b < 0 for a, b in pairs),
            "units_carry_items": sum(
                a * b > 0 and abs(a) % 10 + abs(b) % 10 >= 10 for a, b in pairs
            ),
            "units_borrow_items": sum(
                a * b < 0 and max(abs(a), abs(b)) % 10 < min(abs(a), abs(b)) % 10 for a, b in pairs
            ),
            "zero_sum_items": sum(a + b == 0 for a, b in pairs),
        }.items()
    }


def build_roster():
    """Sixteen input-only contexts; canonical candidate text excludes native EOT."""
    core_order = sorted(
        (hashlib.sha256(f"{NAMESPACE}|schedule|{core}".encode("ascii")).hexdigest(), core)
        for core in range(CORE_COUNT)
    )
    rows = []
    for pair_index, (schedule_sha, core) in enumerate(core_order):
        order = CONDITIONS if pair_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for within_pair_index, condition in enumerate(order):
            pairs = core_operands(core)
            answers = [a + b for a, b in pairs]
            group = f"core_{core:02d}"
            rows.append(
                {
                    "observation_id": f"{group}_{condition}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "condition": condition,
                    "operands": pairs,
                    "expected_sums": answers,
                    "strata": _strata(pairs),
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {
                            "role": "user",
                            "content": "\n".join(
                                f"{a} {b} {answer if condition == 'supplied' else '?'}"
                                for (a, b), answer in zip(pairs, answers, strict=True)
                            ),
                        },
                    ],
                    "candidates": {
                        "correct": f"{answers[0]},{answers[1]}",
                        "first_sum_plus_one": f"{answers[0] + 1},{answers[1]}",
                    },
                    "core_schedule_sha256": schedule_sha,
                    "pair_index": pair_index,
                    "within_pair_index": within_pair_index,
                    "sequence_index": len(rows),
                }
            )
    return rows


compile_roster = build_roster


def validate_roster(rows):
    _require(type(rows) is list, "roster_type")
    expected = build_roster()
    _require(_canonical(rows) == _canonical(expected), "roster_mismatch")
    return expected


def validate_row(row):
    _require(type(row) is dict and type(row.get("sequence_index")) is int, "row_type")
    index = row["sequence_index"]
    _require(0 <= index < ROW_COUNT, "row_index")
    expected = build_roster()[index]
    _require(_canonical(row) == _canonical(expected), "row_mismatch")
    return expected


def build_schedule(rows=None):
    rows = build_roster() if rows is None else validate_roster(rows)
    schedule = []
    for row in rows:
        order = CANDIDATES if row["sequence_index"] % 2 == 0 else tuple(reversed(CANDIDATES))
        counts = dict.fromkeys(CANDIDATES, 0)
        for candidate in (*order, *reversed(order)):
            repeat = counts[candidate]
            counts[candidate] += 1
            schedule.append(
                {
                    **{key: row[key] for key in IDENTITY_KEYS},
                    "candidate": candidate,
                    "repeat_index": repeat,
                    "evaluation_index": len(schedule),
                    "evaluation_id": f"{row['observation_id']}_{candidate}_repeat_{repeat}",
                }
            )
    return schedule


def validate_schedule(schedule, rows=None):
    _require(type(schedule) is list, "schedule_type")
    expected = build_schedule(rows)
    _require(_canonical(schedule) == _canonical(expected), "schedule_mismatch")
    return expected


def _measurement(value):
    if value is None:
        return None
    _require(type(value) is dict and set(value) == {"score", "logits_sha256"}, "measurement_keys")
    _require(
        type(value["score"]) is float and math.isfinite(value["score"]) and value["score"] <= 0,
        "measurement_score",
    )
    _require(
        type(value["logits_sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", value["logits_sha256"]) is not None,
        "measurement_hash",
    )
    return value


def _point(values, planned):
    _require(len(values) == planned, "fixed_cohort")
    resolved = sum(value is not None for value in values)
    point = math.fsum(value / planned for value in values) if resolved == planned else None
    _require(point is None or math.isfinite(point), "nonfinite_aggregate")
    return {
        "planned_contexts": planned,
        "resolved_contexts": resolved,
        "point": point,
        "lower": point,
        "upper": point,
        "unbounded": point is None,
    }


def _condition_summary(records):
    return {
        **_point([record["margin_nats"] for record in records], len(records)),
        "complete_contexts": sum(record["complete"] for record in records),
        "numerical_valid": {
            key: sum(record["numerical_valid"] is value for record in records)
            for key, value in (("true", True), ("false", False), ("unknown", None))
        },
        "margin_signs": {
            key: sum(record["margin_sign"] == key for record in records)
            for key in ("positive", "zero", "negative", "unknown")
        },
    }


def analyze_measurements(rows, schedule, measurements):
    """Readout hashes assert prior byte authentication; repetitions are not averaged."""
    rows = validate_roster(rows)
    schedule = validate_schedule(schedule, rows)
    _require(type(measurements) is list and len(measurements) == CALL_COUNT, "measurement_slots")
    measurements = [_measurement(value) for value in measurements]
    records = []
    for row in rows:
        start = row["sequence_index"] * 4
        slots = {candidate: [None, None] for candidate in CANDIDATES}
        for evaluation, value in zip(
            schedule[start : start + 4], measurements[start : start + 4], strict=True
        ):
            slots[evaluation["candidate"]][evaluation["repeat_index"]] = value
        repeated, scores = {}, {}
        for candidate, (first, second) in slots.items():
            equal = (
                None
                if first is None or second is None
                else first["logits_sha256"] == second["logits_sha256"]
            )
            if equal:
                _require(
                    first["score"].hex() == second["score"].hex(), "equal_readout_score_mismatch"
                )
            repeated[candidate] = equal
            scores[candidate] = first["score"] if equal else None
        numeric = (
            False
            if False in repeated.values()
            else True
            if all(value is True for value in repeated.values())
            else None
        )
        margin = scores["correct"] - scores["first_sum_plus_one"] if numeric is True else None
        _require(margin is None or math.isfinite(margin), "nonfinite_margin")
        count = sum(value is not None for value in measurements[start : start + 4])
        records.append(
            {
                **{key: row[key] for key in IDENTITY_KEYS},
                "measurement_count": count,
                "complete": count == 4,
                "repeat_equal": repeated,
                "candidate_scores": scores,
                "numerical_valid": numeric,
                "margin_nats": margin,
                "margin_sign": "unknown"
                if margin is None
                else "positive"
                if margin > 0
                else "negative"
                if margin < 0
                else "zero",
                "correct_preferred": None if margin is None else margin > 0,
            }
        )
    by_condition = {
        condition: [record for record in records if record["condition"] == condition]
        for condition in CONDITIONS
    }
    pairs = {}
    for record in records:
        pairs.setdefault(record["core_index"], {})[record["condition"]] = record["margin_nats"]
    differences = [
        values["supplied"] - values["computed"]
        if all(value is not None for value in values.values())
        else None
        for values in pairs.values()
    ]
    _require(all(value is None or math.isfinite(value) for value in differences), "nonfinite_pair")
    primary = _point(differences, CORE_COUNT)
    primary["planned_contexts"] = ROW_COUNT
    primary["resolved_contexts"] = sum(record["margin_nats"] is not None for record in records)
    primary["planned_pairs"] = CORE_COUNT
    primary["resolved_pairs"] = sum(value is not None for value in differences)
    supplied = [record["margin_nats"] for record in by_condition["supplied"]]
    qualification = (
        False
        if any(value is not None and value <= 0 for value in supplied)
        else True
        if all(value is not None for value in supplied)
        else None
    )
    analysis = {
        "schema_version": "a192-answer-path-ranking-analysis-v1",
        "planned_measurements": CALL_COUNT,
        "resolved_measurements": sum(value is not None for value in measurements),
        "overall": _condition_summary(records),
        "by_condition": {key: _condition_summary(value) for key, value in by_condition.items()},
        "contrasts": {"margin_nats": {"supplied_minus_computed": primary}},
        "supplied_qualification": qualification,
    }
    return {"records": records, "analysis": analysis}
