"""Independent stdlib A197 construction and fixed-cohort analysis.

No producer imports or file I/O. The main auditor authenticates native inputs,
full FP32 matrices, literal scores, normal returns and exact repeated bytes.
Hashes here declare that prior authentication. Only audit_outcomes returns an
aggregate packet; reconstruct_analysis includes private in-memory context data.
"""

import hashlib
import json
import math
import re

if not __debug__:
    raise RuntimeError("independent_a197_assertions_required")

NAMESPACE = "a197-rule-cue-interaction-v1"
COPY_INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "If the answer field is ?, add the two integers; otherwise copy the supplied answer. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
RECOMPUTE_INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "Ignore the supplied answer field and add the two integers. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
IDENTITY = (
    "observation_id",
    "core_index",
    "core_id",
    "group_id",
    "rule",
    "condition",
    "sequence_index",
)
PATHS = ("arithmetic_truth", "arithmetic_foil")
RULES = ("copy", "recompute")
CUES = ("true_cue", "foil_cue")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def same(actual, expected):
    assert canonical(actual) == canonical(expected), "independent_typed_equality"


def reconstruct_roster():
    ordering = sorted(
        (hashlib.sha256(f"{NAMESPACE}|schedule|{core}".encode("ascii")).hexdigest(), core)
        for core in range(8)
    )
    result = []
    for order_index, (order_hash, core) in enumerate(ordering):
        for position in range(4):
            cell = (order_index + position) % 4
            rule, cue = RULES[cell // 2], CUES[cell % 2]
            pairs = []
            for item in range(2):
                pair = []
                for side in range(2):
                    raw = hashlib.sha256(
                        f"{NAMESPACE}|{core}|{item}|{side}".encode("ascii")
                    ).digest()
                    remainder = 0
                    for byte in raw:
                        remainder = (256 * remainder + byte) % 90
                    pair.append((10 + remainder) * (-1 if raw[0] % 2 else 1))
                pairs.append(pair)
            sums = [left + right for left, right in pairs]
            delta = (-1, 1)[core % 2 == 0]
            supplied = [sums[0] + (delta if cue == "foil_cue" else 0), sums[1]]
            counts = dict.fromkeys(
                (
                    "positive_positive_items",
                    "negative_negative_items",
                    "opposite_sign_items",
                    "units_carry_items",
                    "units_borrow_items",
                    "zero_sum_items",
                ),
                0,
            )
            for left, right in pairs:
                if left > 0 and right > 0:
                    counts["positive_positive_items"] += 1
                elif left < 0 and right < 0:
                    counts["negative_negative_items"] += 1
                else:
                    counts["opposite_sign_items"] += 1
                same_sign = (left > 0) == (right > 0)
                if same_sign and abs(left) % 10 + abs(right) % 10 > 9:
                    counts["units_carry_items"] += 1
                if (
                    not same_sign
                    and max(abs(left), abs(right)) % 10 < min(abs(left), abs(right)) % 10
                ):
                    counts["units_borrow_items"] += 1
                if left + right == 0:
                    counts["zero_sum_items"] += 1
            group = f"core_{core:02d}"
            lines = [f"{pairs[item][0]} {pairs[item][1]} {supplied[item]}" for item in range(2)]
            result.append(
                {
                    "observation_id": f"{group}_{rule}_{cue}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "rule": rule,
                    "condition": cue,
                    "operands": pairs,
                    "expected_sums": sums,
                    "foil_delta": delta,
                    "supplied_answers": supplied,
                    "instructed_candidate": "arithmetic_foil"
                    if rule == "copy" and cue == "foil_cue"
                    else "arithmetic_truth",
                    "strata": {key: str(value) for key, value in counts.items()},
                    "messages": [
                        {
                            "role": "system",
                            "content": COPY_INSTRUCTION
                            if rule == "copy"
                            else RECOMPUTE_INSTRUCTION,
                        },
                        {"role": "user", "content": "\n".join(lines)},
                    ],
                    "candidates": {
                        "arithmetic_truth": f"{sums[0]},{sums[1]}",
                        "arithmetic_foil": f"{sums[0] + delta},{sums[1]}",
                    },
                    "core_schedule_sha256": order_hash,
                    "core_order_index": order_index,
                    "within_core_index": position,
                    "rotation": order_index % 4,
                    "sequence_index": len(result),
                }
            )
    return result


def reconstruct_schedule():
    schedule = []
    for row in reconstruct_roster():
        first_index = (row["core_order_index"] >= 4) ^ bool(row["within_core_index"] % 2)
        first, other = PATHS[first_index], PATHS[not first_index]
        for candidate, repeat in ((first, 0), (other, 0), (other, 1), (first, 1)):
            schedule.append(
                {
                    **{key: row[key] for key in IDENTITY},
                    "candidate": candidate,
                    "repeat_index": repeat,
                    "evaluation_index": len(schedule),
                    "evaluation_id": f"{row['observation_id']}_{candidate}_repeat_{repeat}",
                }
            )
    return schedule


def _checked_measurements(values):
    assert type(values) is list and len(values) == 128, "independent_fixed_measurement_slots"
    for value in values:
        if value is None:
            continue
        assert type(value) is dict and set(value) == {"score", "logits_sha256"}
        assert (
            type(value["score"]) is float and math.isfinite(value["score"]) and value["score"] <= 0
        )
        assert type(value["logits_sha256"]) is str
        assert re.fullmatch(r"[0123456789abcdef]{64}", value["logits_sha256"]) is not None
    return values


def _summarize(contexts):
    total, complete, known = len(contexts), 0, 0
    numeric = {"true": 0, "false": 0, "unknown": 0}
    signs = {"positive": 0, "zero": 0, "negative": 0, "unknown": 0}
    weighted = []
    for context in contexts:
        complete += int(context["complete"])
        valid = context["numerical_valid"]
        numeric["true" if valid is True else "false" if valid is False else "unknown"] += 1
        value = context["margin_nats"]
        if value is None:
            signs["unknown"] += 1
        else:
            known += 1
            weighted.append(value / total)
            signs["positive" if value > 0 else "negative" if value < 0 else "zero"] += 1
    point = math.fsum(weighted) if known == total else None
    assert point is None or math.isfinite(point)
    return {
        "planned_contexts": total,
        "resolved_contexts": known,
        "point": point,
        "lower": point,
        "upper": point,
        "unbounded": known != total,
        "complete_contexts": complete,
        "numerical_valid": numeric,
        "margin_signs": signs,
    }


def _contrast(contexts, differences):
    assert len(differences) == 8
    available = [value for value in differences if value is not None]
    point = math.fsum(value / 8 for value in available) if len(available) == 8 else None
    assert point is None or math.isfinite(point)
    return {
        "planned_contexts": len(contexts),
        "resolved_contexts": sum(c["margin_nats"] is not None for c in contexts),
        "point": point,
        "lower": point,
        "upper": point,
        "unbounded": point is None,
        "planned_cores": 8,
        "resolved_cores": len(available),
    }


def _pattern_summary(states):
    assert len(states) == 8 and all(value is None or type(value) is bool for value in states)
    counts = {
        name: sum(value is flag for value in states)
        for name, flag in (("true", True), ("false", False), ("unknown", None))
    }
    lower, upper = counts["true"] / 8, 1 - counts["false"] / 8
    return {
        "planned_cores": 8,
        "resolved_cores": 8 - counts["unknown"],
        "counts": counts,
        "rate": {
            "point": lower if counts["unknown"] == 0 else None,
            "lower": lower,
            "upper": upper,
        },
        "qualification": False if counts["false"] else True if counts["true"] == 8 else None,
    }


def reconstruct_analysis(plan, measurements):
    """Private context reconstruction plus aggregates; caller controls disclosure."""
    assert type(plan) is dict
    rows, schedule = reconstruct_roster(), reconstruct_schedule()
    same(plan["rows"], rows)
    same(plan["schedule"], schedule)
    measurements = _checked_measurements(measurements)
    grouped = {}
    for slot, value in zip(schedule, measurements, strict=True):
        key = (slot["observation_id"], slot["candidate"], slot["repeat_index"])
        assert key not in grouped
        grouped[key] = value
    contexts = []
    for row in rows:
        scores, repeats, count = {}, {}, 0
        for path in PATHS:
            first = grouped[(row["observation_id"], path, 0)]
            again = grouped[(row["observation_id"], path, 1)]
            count += int(first is not None) + int(again is not None)
            if first is None or again is None:
                repeats[path], scores[path] = None, None
            elif first["logits_sha256"] != again["logits_sha256"]:
                repeats[path], scores[path] = False, None
            else:
                assert first["score"].hex() == again["score"].hex(), (
                    "independent_same_readout_score"
                )
                repeats[path], scores[path] = True, first["score"]
        valid = (
            False
            if any(v is False for v in repeats.values())
            else True
            if all(v is True for v in repeats.values())
            else None
        )
        margin = scores["arithmetic_truth"] - scores["arithmetic_foil"] if valid is True else None
        assert margin is None or math.isfinite(margin)
        sign = (
            "unknown"
            if margin is None
            else "positive"
            if margin > 0
            else "negative"
            if margin < 0
            else "zero"
        )
        contexts.append(
            {
                **{name: row[name] for name in IDENTITY},
                "measurement_count": count,
                "complete": count == 4,
                "repeat_equal": repeats,
                "candidate_scores": scores,
                "numerical_valid": valid,
                "margin_nats": margin,
                "margin_sign": sign,
                "arithmetic_truth_preferred": None if margin is None else margin > 0,
                "instruction_compliant_preferred": None
                if margin is None
                else (
                    margin < 0
                    if row["rule"] == "copy" and row["condition"] == "foil_cue"
                    else margin > 0
                ),
            }
        )
    cells = {(rule, cue): [] for rule in RULES for cue in CUES}
    cores = {core: {} for core in range(8)}
    for context in contexts:
        key = (context["rule"], context["condition"])
        cells[key].append(context)
        cores[context["core_index"]][key] = context["margin_nats"]
    differences = {rule: [] for rule in RULES}
    interactions = []
    patterns = {name: [] for name in ("copying_switch", "recompute_truth", "joint")}
    for core in range(8):
        assert set(cores[core]) == set(cells)
        margins = [cores[core][key] for key in cells]
        good = [
            None if value is None else (value < 0 if index == 1 else value > 0)
            for index, value in enumerate(margins)
        ]
        for name, selected in (
            ("copying_switch", good[:2]),
            ("recompute_truth", good[2:]),
            ("joint", good),
        ):
            state = (
                False
                if any(v is False for v in selected)
                else True
                if all(v is True for v in selected)
                else None
            )
            patterns[name].append(state)
        for index, rule in enumerate(RULES):
            true, foil = margins[index * 2 : index * 2 + 2]
            value = true - foil if true is not None and foil is not None else None
            assert value is None or math.isfinite(value)
            differences[rule].append(value)
        copy, recompute = differences["copy"][-1], differences["recompute"][-1]
        value = copy - recompute if copy is not None and recompute is not None else None
        assert value is None or math.isfinite(value)
        interactions.append(value)
    analysis = {
        "schema_version": "a197-rule-cue-interaction-analysis-v1",
        "planned_measurements": 128,
        "resolved_measurements": sum(value is not None for value in measurements),
        "overall": _summarize(contexts),
        "by_rule_cue": {
            rule: {cue: _summarize(cells[(rule, cue)]) for cue in CUES} for rule in RULES
        },
        "contrasts": {
            "margin_nats": {
                "by_rule": {
                    rule: _contrast([c for c in contexts if c["rule"] == rule], differences[rule])
                    for rule in RULES
                },
                "copy_minus_recompute_cue_effect": _contrast(contexts, interactions),
            }
        },
        "patterns": {name: _pattern_summary(states) for name, states in patterns.items()},
    }
    return {"records": contexts, "analysis": analysis}


def audit_outcomes(plan, records):
    """Aggregate only; the main auditor has already authenticated every input."""
    assert type(records) is list and len(records) == 128
    measurements = []
    for slot, record in zip(reconstruct_schedule(), records, strict=True):
        assert type(record) is dict and set(record) == {
            "evaluation_id",
            "evaluation_index",
            "status",
            "measurement",
        }
        same(record["evaluation_id"], slot["evaluation_id"])
        same(record["evaluation_index"], slot["evaluation_index"])
        assert record["status"] in ("completed", "failed", "unattempted")
        assert (record["status"] == "completed") == (record["measurement"] is not None)
        measurements.append(record["measurement"])
    return {
        "record_slots": 128,
        "context_slots": 32,
        "analysis": reconstruct_analysis(plan, measurements)["analysis"],
    }
