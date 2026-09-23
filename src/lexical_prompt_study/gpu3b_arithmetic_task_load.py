"""A189 fixed task-load construction, finite oracle and aggregate-only analysis.

Pure standard-library functions: no model, tokenizer, capture or data access.
Acquisition must authenticate sampled IDs, native decoding and forward receipts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from fractions import Fraction

NAMESPACE = "gpu3b-arithmetic-load-v1"
SCHEMA = "a189-arithmetic-task-load-v1"
CORE_COUNT = 8
ITEM_COUNTS = (2, 4)
MAGNITUDES = ("one_digit", "two_digit")
ROW_COUNT = 32
CAP = 64
LANDMARK = 4
MAX_PROMPT_TOKENS = 256
CHAT_TEMPLATE_DATE = "23 Sep 2026"
TECHNICAL_FORWARDS = 2
MAX_TARGET_ENTRIES = TECHNICAL_FORWARDS + ROW_COUNT * CAP
TECHNICAL_MESSAGES = (
    ("system", "Follow the user's instruction."),
    ("user", "Reply with the digit 0."),
)
ASCII_WHITESPACE = " \t\r\n\v\f"
SYSTEMS = {
    count: (
        f"For each of the {word} data rows, add its two integers. Preserve their row order. "
        f"Output exactly {word} comma-separated base-10 integer sums, with no other text."
    )
    for count, word in ((2, "two"), (4, "four"))
}
_INTEGER = re.compile(r"[+-]?[0-9]+\Z", flags=re.ASCII)
SCORE_KEYS = (
    "outcome_status",
    "label",
    "terminal_eos",
    "capped",
    "censored",
    "prefix_reached",
    "completed_fields",
    "known_error",
    "remaining_fields",
    "unflagged_remaining_ge2",
)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _core(core):
    _require(type(core) is int and 0 <= core < CORE_COUNT, "core_index")


def _condition(item_count, magnitude):
    _require(type(item_count) is int and item_count in ITEM_COUNTS, "item_count")
    _require(type(magnitude) is str and magnitude in MAGNITUDES, "magnitude")


def operand(core_index, item_index, side, magnitude):
    _core(core_index)
    _require(type(item_index) is int and 0 <= item_index < 4, "item_index")
    _require(type(side) is int and side in (0, 1), "side")
    _condition(2, magnitude)
    digest = hashlib.sha256(
        f"{NAMESPACE}|{core_index}|{item_index}|{side}".encode("ascii")
    ).digest()
    number = int.from_bytes(digest, "big")
    absolute = 1 + number % 9 if magnitude == "one_digit" else 10 + number % 90
    return -absolute if digest[0] & 1 else absolute


def core_operands(core_index, item_count, magnitude):
    _core(core_index)
    _condition(item_count, magnitude)
    return [
        [operand(core_index, item, side, magnitude) for side in (0, 1)]
        for item in range(item_count)
    ]


def pair_stratum(a, b):
    """Input-only units-column definitions; these are not measured difficulty."""
    _require(type(a) is int and type(b) is int and a != 0 and b != 0, "stratum_operands")
    same = (a > 0) == (b > 0)
    larger, smaller = max(abs(a), abs(b)), min(abs(a), abs(b))
    return {
        "sign": "positive_positive"
        if a > 0 and b > 0
        else "negative_negative"
        if a < 0 and b < 0
        else "opposite_sign",
        "units_carry": same and abs(a) % 10 + abs(b) % 10 >= 10,
        "units_borrow": not same and larger % 10 < smaller % 10,
        "zero_sum": a + b == 0,
    }


def core_strata(core_index, item_count, magnitude):
    items = [pair_stratum(a, b) for a, b in core_operands(core_index, item_count, magnitude)]
    counts = {
        name + "_items": sum(x["sign"] == name for x in items)
        for name in ("positive_positive", "negative_negative", "opposite_sign")
    }
    counts.update(
        {
            name + "_items": sum(x[name] for x in items)
            for name in ("units_carry", "units_borrow", "zero_sum")
        }
    )
    return {key: str(value) for key, value in counts.items()}


def user_text(core_index, item_count, magnitude):
    return "\n".join(f"{a} {b}" for a, b in core_operands(core_index, item_count, magnitude))


def technical_messages():
    """Return fresh standard messages for the two identical technical forwards."""
    return [{"role": role, "content": content} for role, content in TECHNICAL_MESSAGES]


def build_roster():
    rows = []
    for core in range(CORE_COUNT):
        group = f"core_{core:02d}"
        for count in ITEM_COUNTS:
            for magnitude in MAGNITUDES:
                pairs = core_operands(core, count, magnitude)
                rows.append(
                    {
                        "observation_id": f"{group}_n{count}_{magnitude}",
                        "core_index": core,
                        "core_id": group,
                        "group_id": group,
                        "item_count": count,
                        "magnitude": magnitude,
                        "operands": pairs,
                        "expected_sums": [a + b for a, b in pairs],
                        "strata": core_strata(core, count, magnitude),
                        "messages": [
                            {"role": "system", "content": SYSTEMS[count]},
                            {"role": "user", "content": user_text(core, count, magnitude)},
                        ],
                        "schedule_sha256": hashlib.sha256(
                            f"{NAMESPACE}|schedule|{core}|{count}|{magnitude}".encode("ascii")
                        ).hexdigest(),
                    }
                )
    rows.sort(
        key=lambda row: (
            row["schedule_sha256"],
            row["core_index"],
            row["item_count"],
            row["magnitude"],
        )
    )
    for index, row in enumerate(rows):
        row["sequence_index"] = index
    return rows


compile_roster = build_roster


def validate_roster(rows):
    _require(type(rows) is list, "roster_type")
    expected = build_roster()
    _require(_canonical(rows) == _canonical(expected), "roster_mismatch")
    return expected


def validate_row(row):
    _require(type(row) is dict, "row_type")
    _core(row.get("core_index"))
    _condition(row.get("item_count"), row.get("magnitude"))
    expected = next(
        x
        for x in build_roster()
        if x["core_index"] == row["core_index"]
        and x["item_count"] == row["item_count"]
        and x["magnitude"] == row["magnitude"]
    )
    _require(_canonical(row) == _canonical(expected), "row_mismatch")
    return expected


def _expected(expected_sums):
    _require(
        type(expected_sums) is list
        and len(expected_sums) in ITEM_COUNTS
        and all(type(x) is int for x in expected_sums),
        "expected_sums",
    )


def _normalized_integer(field):
    field = field.strip(ASCII_WHITESPACE)
    if _INTEGER.fullmatch(field) is None:
        return None
    negative = field.startswith("-")
    digits = field.lstrip("+-").lstrip("0") or "0"
    return ("-" if negative and digits != "0" else "") + digits


def oracle_response(text, expected_sums):
    """Exact ordered ASCII integer fields; native termination is checked separately."""
    _require(type(text) is str, "response_text")
    _expected(expected_sums)
    fields = text.split(",")
    return len(fields) == len(expected_sums) and all(
        _normalized_integer(field) == str(expected)
        for field, expected in zip(fields, expected_sums, strict=True)
    )


def prefix_check(text, expected_sums):
    """Only comma-closed fields can be adjudicated; the open tail is ignored."""
    _require(type(text) is str, "prefix_text")
    _expected(expected_sums)
    completed = text.split(",")[:-1]
    count, required = len(completed), len(expected_sums)
    error = count >= required or any(
        _normalized_integer(field) != str(expected_sums[index])
        for index, field in enumerate(completed[:required])
    )
    remaining = max(required - count, 0)
    return {
        "completed_fields": count,
        "known_error": error,
        "remaining_fields": remaining,
        "unflagged_remaining_ge2": not error and remaining >= 2,
    }


def outcome_for_generation(row, generation, decoded_text, eos_token_ids):
    """Completed native EOS and exact text are required for success; partial is unknown."""
    row = validate_row(row)
    _require(type(generation) is dict, "generation_type")
    _require(
        type(eos_token_ids) is list
        and bool(eos_token_ids)
        and all(type(x) is int and x >= 0 for x in eos_token_ids)
        and len(eos_token_ids) == len(set(eos_token_ids)),
        "eos_token_ids",
    )
    tokens = generation.get("token_ids")
    _require(
        type(tokens) is list
        and len(tokens) <= CAP
        and all(type(x) is int and x >= 0 for x in tokens),
        "generation_tokens",
    )
    _require(
        type(generation.get("observed_tokens")) is int
        and generation["observed_tokens"] == len(tokens),
        "observed_tokens",
    )
    _require(not any(x in eos_token_ids for x in tokens[:-1]), "nonterminal_eos")
    terminal = bool(tokens and tokens[-1] in eos_token_ids)
    status, stop = generation.get("status"), generation.get("stop_reason")
    _require(
        type(status) is str
        and status in ("completed", "interrupted", "infrastructure_failed", "missing"),
        "generation_status",
    )
    if status == "completed":
        _require(
            (stop == "eos" and terminal) or (stop == "cap" and not terminal and len(tokens) == CAP),
            "generation_stop",
        )
        _require(type(decoded_text) is str, "decoded_text")
        label = int(terminal and oracle_response(decoded_text, row["expected_sums"]))
    else:
        _require(stop == status, "generation_stop")
        _require(status != "missing" or not tokens, "missing_tokens")
        _require(decoded_text is None or type(decoded_text) is str, "decoded_text")
        label = None
    return {
        "status": "known" if label is not None else "missing",
        "label": label,
        "reason": None if label is not None else "not_measured",
        "terminal_eos": None if status == "missing" else terminal,
        "capped": None if status == "missing" else stop == "cap",
        "censored": None if status == "missing" else not terminal and len(tokens) < CAP,
    }


def score_generation(row, generation, decoded_text, prefix_text, eos_token_ids):
    outcome = outcome_for_generation(row, generation, decoded_text, eos_token_ids)
    count = len(generation["token_ids"]) - int(outcome["terminal_eos"] is True)
    reached = True if count >= LANDMARK else False if outcome["terminal_eos"] is True else None
    _require(prefix_text is None or type(prefix_text) is str, "prefix_text")
    _require(prefix_text is None or reached is True, "prefix_without_landmark")
    prefix = (
        {
            key: None
            for key in (
                "completed_fields",
                "known_error",
                "remaining_fields",
                "unflagged_remaining_ge2",
            )
        }
        if prefix_text is None
        else prefix_check(prefix_text, row["expected_sums"])
    )
    return {
        "outcome_status": outcome["status"],
        "label": outcome["label"],
        "terminal_eos": outcome["terminal_eos"],
        "capped": outcome["capped"],
        "censored": outcome["censored"],
        "prefix_reached": reached,
        **prefix,
    }


def _validate_record(row, record):
    _require(type(record) is dict, "record_type")
    for key in ("observation_id", "core_index", "item_count", "magnitude", "sequence_index"):
        _require(
            key in record and _canonical(record[key]) == _canonical(row[key]), "record_identity"
        )
    _require(all(key in record for key in SCORE_KEYS), "record_score_keys")
    label = record["label"]
    _require(label is None or type(label) is int and label in (0, 1), "record_label")
    _require(
        record["outcome_status"] == ("missing" if label is None else "known"), "record_outcome"
    )
    stop_flags = [record[key] for key in ("terminal_eos", "capped", "censored")]
    no_stop_evidence = all(value is None for value in stop_flags)
    _require(no_stop_evidence or all(type(value) is bool for value in stop_flags), "record_boolean")
    _require(not no_stop_evidence or label is None, "record_unobserved_label")
    _require(
        not (record["terminal_eos"] and (record["capped"] or record["censored"])), "record_stop"
    )
    _require(not (record["capped"] and record["censored"]), "record_stop")
    _require(label != 1 or record["terminal_eos"], "record_success")
    _require(label is None or record["terminal_eos"] or record["capped"], "record_completion")
    reached = record["prefix_reached"]
    _require(reached is None or type(reached) is bool, "record_reached")
    _require(not no_stop_evidence or reached is None, "record_unobserved_prefix")
    _require(label is None or reached is not None, "record_completed_prefix")
    _require(not record["capped"] or reached is True, "record_cap_prefix")
    _require(reached is not False or record["terminal_eos"], "record_unreached")
    fields = [
        record[k]
        for k in ("completed_fields", "known_error", "remaining_fields", "unflagged_remaining_ge2")
    ]
    if all(x is None for x in fields):
        return
    completed, error, remaining, eligible = fields
    _require(
        reached is True
        and type(completed) is int
        and completed >= 0
        and type(error) is bool
        and type(remaining) is int
        and type(eligible) is bool,
        "record_prefix_types",
    )
    _require(remaining == max(row["item_count"] - completed, 0), "record_remaining")
    _require(completed < row["item_count"] or error, "record_extra_fields")
    _require(eligible == (not error and remaining >= 2), "record_prefix_eligibility")


def _tri(values):
    return {
        "true": sum(x is True for x in values),
        "false": sum(x is False for x in values),
        "unknown": sum(x is None for x in values),
    }


def _binary_mean(records):
    planned = len(records)
    known = [r["label"] for r in records if r["label"] is not None]
    successes, missing = sum(known), planned - len(known)
    return {
        "point": successes / planned if not missing else None,
        "lower": successes / planned,
        "upper": (successes + missing) / planned,
        "planned_outcomes": planned,
        "resolved_outcomes": len(known),
    }


def _summary(records):
    labels = [r["label"] for r in records]
    membership = [
        False if r["prefix_reached"] is False else r["unflagged_remaining_ge2"] for r in records
    ]
    eligible = [r for r, member in zip(records, membership, strict=True) if member is True]
    return {
        "planned_rows": len(records),
        "success_rate": _binary_mean(records),
        "labels": {
            "success": labels.count(1),
            "failure": labels.count(0),
            "unknown": labels.count(None),
        },
        "core_support": {
            name: len({r["core_index"] for r in records if r["label"] == value})
            for name, value in (("success", 1), ("failure", 0), ("unknown", None))
        },
        "terminal_eos": sum(r["terminal_eos"] is True for r in records),
        "terminal_eos_unknown": sum(r["terminal_eos"] is None for r in records),
        "capped": sum(r["capped"] is True for r in records),
        "capped_unknown": sum(r["capped"] is None for r in records),
        "censored": _tri([r["censored"] for r in records]),
        "prefix": {
            "planned_rows": len(records),
            "reached": _tri([r["prefix_reached"] for r in records]),
            "known_error": _tri([r["known_error"] for r in records]),
            "completed_fields_counts": dict(
                sorted(
                    Counter(
                        str(r["completed_fields"])
                        for r in records
                        if r["completed_fields"] is not None
                    ).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "completed_fields_unknown": sum(r["completed_fields"] is None for r in records),
            "remaining_fields_counts": dict(
                sorted(
                    Counter(
                        str(r["remaining_fields"])
                        for r in records
                        if r["remaining_fields"] is not None
                    ).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "remaining_fields_unknown": sum(r["remaining_fields"] is None for r in records),
            "reached_unflagged_remaining_ge2": {
                "planned_rows": len(records),
                "membership": _tri(membership),
                "eligible_labels": {
                    "success": sum(r["label"] == 1 for r in eligible),
                    "failure": sum(r["label"] == 0 for r in eligible),
                    "unknown": sum(r["label"] is None for r in eligible),
                },
                "eligible_cores": len({r["core_index"] for r in eligible}),
                "eligible_core_support": {
                    name: len({r["core_index"] for r in eligible if r["label"] == value})
                    for name, value in (("success", 1), ("failure", 0), ("unknown", None))
                },
            },
        },
    }


def _contrast(records, axis, positive, negative):
    other = "magnitude" if axis == "item_count" else "item_count"
    pairs = {}
    for record in records:
        pairs.setdefault((record["core_index"], record[other]), {})[record[axis]] = record["label"]
    _require(
        len(pairs) == 16 and all(set(p) == {positive, negative} for p in pairs.values()),
        "contrast_pairs",
    )
    lower = upper = Fraction(0)
    resolved_outcomes = resolved_pairs = 0
    for pair in pairs.values():
        resolved_pairs += all(x is not None for x in pair.values())
        for level, coefficient in ((positive, Fraction(1, 16)), (negative, Fraction(-1, 16))):
            label = pair[level]
            if label is None:
                lower += min(coefficient, 0)
                upper += max(coefficient, 0)
            else:
                lower += coefficient * label
                upper += coefficient * label
                resolved_outcomes += 1
    return {
        "point": float(lower) if resolved_outcomes == 32 else None,
        "lower": float(lower),
        "upper": float(upper),
        "planned_outcomes": 32,
        "resolved_outcomes": resolved_outcomes,
        "planned_pairs": 16,
        "resolved_pairs": resolved_pairs,
    }


def aggregate_records(rows, records):
    """Analyze all fixed slots, without emitting identifiers or observed-subset rates."""
    rows = validate_roster(rows)
    _require(type(records) is list and len(records) == ROW_COUNT, "record_slots")
    for row, record in zip(rows, records, strict=True):
        _validate_record(row, record)
    return {
        "schema_version": "a189-task-load-analysis-v1",
        "overall": _summary(records),
        "by_condition": {
            str(count): {
                magnitude: _summary(
                    [r for r in records if r["item_count"] == count and r["magnitude"] == magnitude]
                )
                for magnitude in MAGNITUDES
            }
            for count in ITEM_COUNTS
        },
        "contrasts": {
            "success_rate": {"shorter_minus_longer": _contrast(records, "item_count", 2, 4)}
        },
        "secondary_contrasts": {
            "success_rate": {
                "one_digit_minus_two_digit": _contrast(
                    records, "magnitude", "one_digit", "two_digit"
                )
            }
        },
    }


analyze = aggregate_records
