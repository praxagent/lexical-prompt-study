"""Independent stdlib A196 construction and finite-cohort analysis.

No producer imports or file reads. The caller must independently authenticate
native inputs, full FP32 readouts, score arithmetic, returns and exact repeats.
Readout hashes here are declarations from that prior authentication. Only
audit_outcomes returns a public aggregate packet; reconstruct_analysis also
returns private context metadata for in-memory comparison by the main auditor.
"""

import hashlib
import json
import math
import re

if not __debug__:
    raise RuntimeError("independent_a196_assertions_required")

NAMESPACE = "a196-cue-validity-ranking-v1"
INSTRUCTION = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "If the answer field is ?, add the two integers; otherwise copy the supplied answer. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
IDENTITY = ("observation_id", "core_index", "core_id", "group_id", "condition", "sequence_index")
PATHS = ("arithmetic_truth", "arithmetic_foil")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def same(actual, expected):
    assert canonical(actual) == canonical(expected), "independent_typed_equality"


def reconstruct_roster():
    ordering = []
    for core in range(8):
        key = hashlib.sha256(f"{NAMESPACE}|schedule|{core}".encode("ascii")).hexdigest()
        ordering.append((key, core))
    result = []
    for pair_index, (key, core) in enumerate(sorted(ordering)):
        for position in range(2):
            condition = ("true_cue", "foil_cue")[(pair_index % 2) ^ position]
            pairs = []
            for item in range(2):
                operands = []
                for side in range(2):
                    raw = hashlib.sha256(
                        f"{NAMESPACE}|{core}|{item}|{side}".encode("ascii")
                    ).digest()
                    modulus = 0
                    for byte in raw:
                        modulus = (modulus * 256 + byte) % 90
                    operands.append((10 + modulus) * (-1 if raw[0] % 2 else 1))
                pairs.append(operands)
            sums = [pair[0] + pair[1] for pair in pairs]
            delta = (-1, 1)[core % 2 == 0]
            supplied = [sums[0], sums[1]]
            if condition == "foil_cue":
                supplied[0] += delta
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
            lines = []
            for item in range(2):
                answer = str(supplied[item])
                lines.append(f"{pairs[item][0]} {pairs[item][1]} {answer}")
            result.append(
                {
                    "observation_id": f"{group}_{condition}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "condition": condition,
                    "operands": pairs,
                    "expected_sums": sums,
                    "foil_delta": delta,
                    "supplied_answers": supplied,
                    "instructed_candidate": "arithmetic_foil"
                    if condition == "foil_cue"
                    else "arithmetic_truth",
                    "strata": {name: str(value) for name, value in counts.items()},
                    "messages": [
                        {"role": "system", "content": INSTRUCTION},
                        {"role": "user", "content": "\n".join(lines)},
                    ],
                    "candidates": {
                        "arithmetic_truth": f"{sums[0]},{sums[1]}",
                        "arithmetic_foil": f"{sums[0] + delta},{sums[1]}",
                    },
                    "core_schedule_sha256": key,
                    "pair_index": pair_index,
                    "within_pair_index": position,
                    "sequence_index": len(result),
                }
            )
    return result


def reconstruct_schedule():
    schedule = []
    for row in reconstruct_roster():
        first, second = PATHS[row["sequence_index"] % 2], PATHS[1 - row["sequence_index"] % 2]
        for candidate, repeat in ((first, 0), (second, 0), (second, 1), (first, 1)):
            schedule.append(
                {
                    **{name: row[name] for name in IDENTITY},
                    "candidate": candidate,
                    "repeat_index": repeat,
                    "evaluation_index": len(schedule),
                    "evaluation_id": f"{row['observation_id']}_{candidate}_repeat_{repeat}",
                }
            )
    return schedule


def _checked_measurements(values):
    assert type(values) is list and len(values) == 64, "independent_fixed_measurement_slots"
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


def reconstruct_analysis(plan, measurements):
    """Return private context reconstruction plus aggregates; caller controls disclosure."""
    assert type(plan) is dict
    rows, schedule = reconstruct_roster(), reconstruct_schedule()
    same(plan["rows"], rows)
    same(plan["schedule"], schedule)
    measurements = _checked_measurements(measurements)
    grouped = {}
    observed = 0
    for slot, measurement in zip(schedule, measurements, strict=True):
        key = (slot["observation_id"], slot["candidate"], slot["repeat_index"])
        assert key not in grouped
        grouped[key] = measurement
        observed += int(measurement is not None)
    contexts = []
    for row in rows:
        score_by_path, repeats = {}, {}
        count = 0
        for path in PATHS:
            first = grouped[(row["observation_id"], path, 0)]
            again = grouped[(row["observation_id"], path, 1)]
            count += int(first is not None) + int(again is not None)
            if first is None or again is None:
                repeats[path], score_by_path[path] = None, None
            elif first["logits_sha256"] != again["logits_sha256"]:
                repeats[path], score_by_path[path] = False, None
            else:
                assert first["score"].hex() == again["score"].hex(), (
                    "independent_same_readout_score"
                )
                repeats[path], score_by_path[path] = True, first["score"]
        guard_values = tuple(repeats.values())
        valid = (
            False
            if any(value is False for value in guard_values)
            else True
            if all(value is True for value in guard_values)
            else None
        )
        if valid is True:
            margin = score_by_path["arithmetic_truth"] - score_by_path["arithmetic_foil"]
            assert math.isfinite(margin)
            sign, preferred = (
                ("positive", True)
                if margin > 0
                else ("negative", False)
                if margin < 0
                else ("zero", False)
            )
        else:
            margin, sign, preferred = None, "unknown", None
        contexts.append(
            {
                **{name: row[name] for name in IDENTITY},
                "measurement_count": count,
                "complete": count == 4,
                "repeat_equal": repeats,
                "candidate_scores": score_by_path,
                "numerical_valid": valid,
                "margin_nats": margin,
                "margin_sign": sign,
                "arithmetic_truth_preferred": preferred,
                "instruction_compliant_preferred": None
                if margin is None
                else (margin > 0 if row["condition"] == "true_cue" else margin < 0),
            }
        )
    arms = {name: [] for name in ("true_cue", "foil_cue")}
    pairs = {core: {} for core in range(8)}
    for context in contexts:
        arms[context["condition"]].append(context)
        pairs[context["core_index"]][context["condition"]] = context["margin_nats"]
    paired_weighted, paired_known, known_contexts = [], 0, 0
    switch_counts = {"true": 0, "false": 0, "unknown": 0}
    for core in range(8):
        assert set(pairs[core]) == {"true_cue", "foil_cue"}
        true_cue, foil_cue = pairs[core]["true_cue"], pairs[core]["foil_cue"]
        available = int(true_cue is not None) + int(foil_cue is not None)
        known_contexts += available
        failed_signs = 0
        if true_cue is not None:
            failed_signs += int(true_cue <= 0)
        if foil_cue is not None:
            failed_signs += int(foil_cue >= 0)
        if failed_signs:
            switch_counts["false"] += 1
        elif available == 2:
            switch_counts["true"] += 1
        else:
            switch_counts["unknown"] += 1
        if available == 2:
            difference = true_cue - foil_cue
            assert math.isfinite(difference)
            paired_weighted.append(difference / 8)
            paired_known += 1
    primary = math.fsum(paired_weighted) if paired_known == 8 else None
    assert primary is None or math.isfinite(primary)
    switch_lower = switch_counts["true"] / 8
    switch_upper = 1 - switch_counts["false"] / 8
    switch_point = switch_lower if switch_counts["unknown"] == 0 else None
    qualification = (
        False if switch_counts["false"] > 0 else True if switch_counts["true"] == 8 else None
    )
    analysis = {
        "schema_version": "a196-cue-validity-ranking-analysis-v1",
        "planned_measurements": 64,
        "resolved_measurements": observed,
        "overall": _summarize(contexts),
        "by_condition": {name: _summarize(arms[name]) for name in ("true_cue", "foil_cue")},
        "contrasts": {
            "margin_nats": {
                "true_cue_minus_foil_cue": {
                    "planned_contexts": 16,
                    "resolved_contexts": known_contexts,
                    "point": primary,
                    "lower": primary,
                    "upper": primary,
                    "unbounded": primary is None,
                    "planned_pairs": 8,
                    "resolved_pairs": paired_known,
                }
            }
        },
        "paired_switch": {
            "planned_pairs": 8,
            "resolved_pairs": switch_counts["true"] + switch_counts["false"],
            "counts": switch_counts,
            "rate": {"point": switch_point, "lower": switch_lower, "upper": switch_upper},
            "qualification": qualification,
        },
    }
    return {"records": contexts, "analysis": analysis}


def audit_outcomes(plan, records):
    """Aggregate-only result after the main auditor has authenticated every input."""
    assert type(records) is list and len(records) == 64
    schedule = reconstruct_schedule()
    measurements = []
    for slot, record in zip(schedule, records, strict=True):
        assert type(record) is dict
        assert set(record) == {"evaluation_id", "evaluation_index", "status", "measurement"}
        same(record["evaluation_id"], slot["evaluation_id"])
        same(record["evaluation_index"], slot["evaluation_index"])
        assert record["status"] in ("completed", "failed", "unattempted")
        assert (record["status"] == "completed") == (record["measurement"] is not None)
        measurements.append(record["measurement"])
    return {
        "record_slots": 64,
        "context_slots": 16,
        "analysis": reconstruct_analysis(plan, measurements)["analysis"],
    }
