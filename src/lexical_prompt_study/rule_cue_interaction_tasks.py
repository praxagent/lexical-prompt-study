"""A197 fresh rule-by-cue paths and fixed-cohort analysis; pure standard library.

The coordinator authenticates native template/token geometry and full FP32
readout bytes. Hashes supplied here declare those already checked readouts.
No function executes a tokenizer/model or interprets free-generation evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

NAMESPACE = "a197-rule-cue-interaction-v1"
SCHEMA = "a197-rule-cue-interaction-tasks-v1"
CORE_COUNT = 8
ROW_COUNT = 32
CALL_COUNT = 128
RULES = ("copy", "recompute")
CONDITIONS = ("true_cue", "foil_cue")
CELLS = tuple((rule, condition) for rule in RULES for condition in CONDITIONS)
CANDIDATES = ("arithmetic_truth", "arithmetic_foil")
MAX_PROMPT_TOKENS = 256
MAX_CANDIDATE_TOKENS = 16
EOT_TOKEN = "<|eot_id|>"
COPY_SYSTEM_INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "If the answer field is ?, add the two integers; otherwise copy the supplied answer. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
RECOMPUTE_SYSTEM_INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "Ignore the supplied answer field and add the two integers. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
SYSTEM_INSTRUCTIONS = {"copy": COPY_SYSTEM_INSTRUCTION, "recompute": RECOMPUTE_SYSTEM_INSTRUCTION}
IDENTITY_KEYS = (
    "observation_id",
    "core_index",
    "core_id",
    "group_id",
    "rule",
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


def foil_delta(core_index):
    _require(type(core_index) is int and 0 <= core_index < CORE_COUNT, "core_index")
    return 1 if core_index % 2 == 0 else -1


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
    """Thirty-two input-only contexts; canonical candidate text excludes native EOT."""
    core_order = sorted(
        (hashlib.sha256(f"{NAMESPACE}|schedule|{core}".encode("ascii")).hexdigest(), core)
        for core in range(CORE_COUNT)
    )
    rows = []
    for core_order_index, (schedule_sha, core) in enumerate(core_order):
        rotation = core_order_index % 4
        order = CELLS[rotation:] + CELLS[:rotation]
        for within_core_index, (rule, condition) in enumerate(order):
            pairs = core_operands(core)
            answers = [a + b for a, b in pairs]
            delta = foil_delta(core)
            foil = [answers[0] + delta, answers[1]]
            supplied = answers.copy() if condition == "true_cue" else foil.copy()
            group = f"core_{core:02d}"
            rows.append(
                {
                    "observation_id": f"{group}_{rule}_{condition}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "rule": rule,
                    "condition": condition,
                    "operands": pairs,
                    "expected_sums": answers,
                    "foil_delta": delta,
                    "supplied_answers": supplied,
                    "instructed_candidate": "arithmetic_truth"
                    if rule == "recompute" or condition == "true_cue"
                    else "arithmetic_foil",
                    "strata": _strata(pairs),
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS[rule]},
                        {
                            "role": "user",
                            "content": "\n".join(
                                f"{a} {b} {answer}"
                                for (a, b), answer in zip(pairs, supplied, strict=True)
                            ),
                        },
                    ],
                    "candidates": {
                        "arithmetic_truth": f"{answers[0]},{answers[1]}",
                        "arithmetic_foil": f"{foil[0]},{foil[1]}",
                    },
                    "core_schedule_sha256": schedule_sha,
                    "core_order_index": core_order_index,
                    "within_core_index": within_core_index,
                    "rotation": rotation,
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
        parity = (row["core_order_index"] // 4 + row["within_core_index"]) % 2
        order = CANDIDATES if parity == 0 else tuple(reversed(CANDIDATES))
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


def _contrast(values, records):
    result = _point(values, CORE_COUNT)
    result["planned_contexts"] = len(records)
    result["resolved_contexts"] = sum(record["margin_nats"] is not None for record in records)
    result["planned_cores"] = CORE_COUNT
    result["resolved_cores"] = sum(value is not None for value in values)
    return result


def _strict_pattern(margins, signs):
    if any(
        value is not None and value * sign <= 0 for value, sign in zip(margins, signs, strict=True)
    ):
        return False
    return True if all(value is not None for value in margins) else None


def _pattern_summary(values):
    _require(len(values) == CORE_COUNT, "fixed_pattern_cohort")
    counts = {
        key: sum(value is state for value in values)
        for key, state in (("true", True), ("false", False), ("unknown", None))
    }
    lower = counts["true"] / CORE_COUNT
    return {
        "planned_cores": CORE_COUNT,
        "resolved_cores": CORE_COUNT - counts["unknown"],
        "counts": counts,
        "rate": {
            "point": lower if counts["unknown"] == 0 else None,
            "lower": lower,
            "upper": (counts["true"] + counts["unknown"]) / CORE_COUNT,
        },
        "qualification": False
        if counts["false"]
        else True
        if counts["true"] == CORE_COUNT
        else None,
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
        margin = scores["arithmetic_truth"] - scores["arithmetic_foil"] if numeric is True else None
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
                "arithmetic_truth_preferred": None if margin is None else margin > 0,
                "instruction_compliant_preferred": None
                if margin is None
                else margin > 0
                if row["instructed_candidate"] == "arithmetic_truth"
                else margin < 0,
            }
        )
    cells = {
        rule: {
            cue: [
                record
                for record in records
                if record["rule"] == rule and record["condition"] == cue
            ]
            for cue in CONDITIONS
        }
        for rule in RULES
    }
    core_margins = {core: {} for core in range(CORE_COUNT)}
    for record in records:
        core_margins[record["core_index"]][(record["rule"], record["condition"])] = record[
            "margin_nats"
        ]
    rule_effects = {rule: [] for rule in RULES}
    interactions = []
    patterns = {name: [] for name in ("copying_switch", "recompute_truth", "joint")}
    for margins in core_margins.values():
        for rule in RULES:
            true, foil = (margins[(rule, cue)] for cue in CONDITIONS)
            effect = None if true is None or foil is None else true - foil
            _require(effect is None or math.isfinite(effect), "nonfinite_rule_effect")
            rule_effects[rule].append(effect)
        copy_effect, recompute_effect = (rule_effects[rule][-1] for rule in RULES)
        interaction = (
            None
            if copy_effect is None or recompute_effect is None
            else copy_effect - recompute_effect
        )
        _require(interaction is None or math.isfinite(interaction), "nonfinite_interaction")
        interactions.append(interaction)
        copy_margins = [margins[("copy", cue)] for cue in CONDITIONS]
        recompute_margins = [margins[("recompute", cue)] for cue in CONDITIONS]
        patterns["copying_switch"].append(_strict_pattern(copy_margins, (1, -1)))
        patterns["recompute_truth"].append(_strict_pattern(recompute_margins, (1, 1)))
        patterns["joint"].append(_strict_pattern(copy_margins + recompute_margins, (1, -1, 1, 1)))
    analysis = {
        "schema_version": "a197-rule-cue-interaction-analysis-v1",
        "planned_measurements": CALL_COUNT,
        "resolved_measurements": sum(value is not None for value in measurements),
        "overall": _condition_summary(records),
        "by_rule_cue": {
            rule: {cue: _condition_summary(cell) for cue, cell in groups.items()}
            for rule, groups in cells.items()
        },
        "contrasts": {
            "margin_nats": {
                "by_rule": {
                    rule: _contrast(
                        values, [record for record in records if record["rule"] == rule]
                    )
                    for rule, values in rule_effects.items()
                },
                "copy_minus_recompute_cue_effect": _contrast(interactions, records),
            }
        },
        "patterns": {name: _pattern_summary(values) for name, values in patterns.items()},
    }
    return {"records": records, "analysis": analysis}
