"""A186 fixed public arithmetic tasks and common-time, model-free scoring.

This module performs no tokenizer/model calls and reads no experimental data.
Native rendering, generation provenance, and A183 evidence belong to acquisition.
"""

from __future__ import annotations

import hashlib
import json
import re

NAMESPACE = "matched-prefix-arithmetic-v1"
SCHEMA = "a186-matched-prefix-tasks-v1"
CORE_COUNT = 48
FIT_CORE_COUNT = 32
ITEM_COUNT = 12
ROW_COUNT = 112
CAP = 64
LANDMARK = 8
MAX_PROMPT_TOKENS = 256
FORMATS = ("plain", "labeled", "csv")
ASCII_WHITESPACE = " \t\r\n\v\f"
SYSTEM = (
    "For each of the twelve data rows, add its two integers. Preserve their row order. "
    "Output exactly twelve comma-separated base-10 integer sums, with no other text."
)
_INTEGER = re.compile(r"[+-]?[0-9]+\Z", flags=re.ASCII)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _core_index(core_index):
    _require(type(core_index) is int and 0 <= core_index < CORE_COUNT, "core_index")


def operand(core_index, item_index, side):
    """Return one exactly specified signed two-digit operand."""
    _core_index(core_index)
    _require(type(item_index) is int and 0 <= item_index < ITEM_COUNT, "item_index")
    _require(type(side) is int and side in (0, 1), "side")
    source = f"{NAMESPACE}|{core_index}|{item_index}|{side}".encode("ascii")
    residue = int.from_bytes(hashlib.sha256(source).digest(), "big") % 180
    return residue - 99 if residue < 90 else residue - 80


def core_operands(core_index):
    _core_index(core_index)
    return [[operand(core_index, item, side) for side in (0, 1)] for item in range(ITEM_COUNT)]


def pair_stratum(a, b):
    """Input-only units-column carry/borrow definitions, not measured difficulty."""
    _require(type(a) is int and type(b) is int and a != 0 and b != 0, "stratum_operands")
    if a > 0 and b > 0:
        sign = "positive_positive"
    elif a < 0 and b < 0:
        sign = "negative_negative"
    else:
        sign = "opposite_sign"
    same_sign = (a > 0) == (b > 0)
    larger, smaller = max(abs(a), abs(b)), min(abs(a), abs(b))
    return {
        "sign": sign,
        "units_carry": same_sign and abs(a) % 10 + abs(b) % 10 >= 10,
        "units_borrow": not same_sign and larger % 10 < smaller % 10,
        "zero_sum": a + b == 0,
    }


def core_strata(core_index):
    """String-valued metadata compatible with the unchanged A183 manifest."""
    items = [pair_stratum(a, b) for a, b in core_operands(core_index)]
    counts = {
        name + "_items": sum(item["sign"] == name for item in items)
        for name in ("positive_positive", "negative_negative", "opposite_sign")
    }
    counts.update(
        {
            name + "_items": sum(item[name] for item in items)
            for name in ("units_carry", "units_borrow", "zero_sum")
        }
    )
    return {name: str(count) for name, count in counts.items()}


def user_text(core_index, format):
    _require(type(format) is str and format in FORMATS, "format")
    pairs = core_operands(core_index)
    if format == "plain":
        return "\n".join(f"{a} {b}" for a, b in pairs)
    if format == "labeled":
        return "\n".join(f"a={a}; b={b}" for a, b in pairs)
    return "a,b\n" + "\n".join(f"{a},{b}" for a, b in pairs)


def build_roster():
    """Return new containers for all 112 rows in the fixed hash execution order."""
    rows = []
    for core in range(CORE_COUNT):
        group = f"core_{core:02d}"
        split = "fit" if core < FIT_CORE_COUNT else "evaluation"
        for format in FORMATS[:2] if split == "fit" else FORMATS:
            pairs = core_operands(core)
            rows.append(
                {
                    "observation_id": f"{group}_{format}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "split": split,
                    "format": format,
                    "operands": pairs,
                    "expected_sums": [a + b for a, b in pairs],
                    "strata": core_strata(core),
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_text(core, format)},
                    ],
                    "schedule_sha256": hashlib.sha256(
                        f"{NAMESPACE}|schedule|{core}|{format}".encode("ascii")
                    ).hexdigest(),
                }
            )
    rows.sort(key=lambda row: (row["schedule_sha256"], row["core_index"], row["format"]))
    for index, row in enumerate(rows):
        row["sequence_index"] = index
    return rows


compile_roster = build_roster


def validate_roster(rows):
    """Reject any changed field, type, slot, ordering, or additional data."""
    _require(type(rows) is list, "roster_type")
    expected = build_roster()
    _require(_canonical(rows) == _canonical(expected), "roster_mismatch")
    return expected


def validate_row(row):
    _require(type(row) is dict, "row_type")
    _core_index(row.get("core_index"))
    _require(type(row.get("format")) is str and row["format"] in FORMATS, "row_format")
    expected = next(
        (
            item
            for item in build_roster()
            if item["core_index"] == row["core_index"] and item["format"] == row["format"]
        ),
        None,
    )
    _require(expected is not None and _canonical(row) == _canonical(expected), "row_mismatch")
    return expected


def prediction_roster(rows):
    keys = ("observation_id", "core_index", "format", "split")
    return [{key: row[key] for key in keys} for row in validate_roster(rows)]


def _expected(expected_sums):
    _require(
        type(expected_sums) is list
        and len(expected_sums) == ITEM_COUNT
        and all(type(value) is int for value in expected_sums),
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
    """Exact twelve-field grammar and arithmetic; termination is checked separately."""
    _require(type(text) is str, "response_text")
    _expected(expected_sums)
    fields = text.split(",")
    return len(fields) == ITEM_COUNT and all(
        _normalized_integer(field) == str(expected)
        for field, expected in zip(fields, expected_sums, strict=True)
    )


def prefix_check(text, expected_sums):
    """Use only observed comma-closed fields; never adjudicate the open tail."""
    _require(type(text) is str, "prefix_text")
    _expected(expected_sums)
    completed = text.split(",")[:-1]
    count = len(completed)
    error = count >= ITEM_COUNT or any(
        _normalized_integer(field) != str(expected_sums[index])
        for index, field in enumerate(completed[:ITEM_COUNT])
    )
    return {
        "completed_fields": count,
        "completed_fields_normalized": count / ITEM_COUNT,
        "prefix_known_error": error,
    }


def prefix_features(core_index, prefix_text):
    expected = [a + b for a, b in core_operands(core_index)]
    return prefix_check(prefix_text, expected)


def outcome_for_generation(row, generation, decoded_text, eos_token_ids):
    """Finite endpoint policy, separate from decoding and A183 provenance validation.

    ``decoded_text`` is the exact decode of the retained non-EOS content, not a
    terminal marker or cleaned string. Acquisition must verify that binding.
    Incomplete generation statuses never gain a label from their partial text.
    """
    row = validate_row(row)
    _require(type(generation) is dict, "generation_type")
    _require(
        type(eos_token_ids) is list
        and bool(eos_token_ids)
        and all(type(token) is int and token >= 0 for token in eos_token_ids)
        and len(eos_token_ids) == len(set(eos_token_ids)),
        "eos_token_ids",
    )
    tokens = generation.get("token_ids")
    _require(
        type(tokens) is list
        and len(tokens) <= CAP
        and all(type(token) is int and token >= 0 for token in tokens),
        "generation_tokens",
    )
    _require(
        type(generation.get("observed_tokens")) is int
        and generation["observed_tokens"] == len(tokens),
        "observed_tokens",
    )
    _require(not any(token in eos_token_ids for token in tokens[:-1]), "nonterminal_eos")
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
        _require(stop == status and not terminal, "generation_stop")
        _require(status != "missing" or not tokens, "missing_tokens")
        _require(decoded_text is None or type(decoded_text) is str, "decoded_text")
        label = None
    return {
        "status": "known" if label is not None else "missing",
        "label": label,
        "reason": None if label is not None else "not_measured",
        "terminal_eos": terminal,
        "capped": stop == "cap",
        "censored": not terminal and len(tokens) < CAP,
    }
