"""Independent stdlib A189 arithmetic audit; only audit_outcomes reads evidence.

This checks endpoint arithmetic after the main verifier authenticates native
receipts and decoding provenance. It neither retokenizes nor proves native
decoding. Its return value contains aggregates only.
"""

from pathlib import Path
import hashlib
import json
import re

if not __debug__:
    raise RuntimeError("independent_audit_requires_assertions")

NAMESPACE = "gpu3b-arithmetic-load-v1"
ACQUISITION_SCHEMA = "a189-bf16-arithmetic-acquisition-v1"
FIELDS = (
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
SPACE = " \t\r\n\v\f"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def same(a, b):
    assert canonical(a) == canonical(b), "independent_typed_mismatch"


def reconstruct_roster():
    """Reconstruct all public inputs without importing the producing module."""
    rows = []
    for core in range(8):
        for magnitude, start, modulus in (("one_digit", 1, 9), ("two_digit", 10, 90)):
            pairs = []
            for item in range(4):
                pair = []
                for side in range(2):
                    raw = hashlib.sha256(
                        f"{NAMESPACE}|{core}|{item}|{side}".encode("ascii")
                    ).digest()
                    absolute = start + int.from_bytes(raw, "big") % modulus
                    pair.append(-absolute if raw[0] % 2 else absolute)
                pairs.append(pair)
            for count, word in ((2, "two"), (4, "four")):
                selected = [list(pair) for pair in pairs[:count]]
                strata = dict.fromkeys(
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
                for a, b in selected:
                    sign = (
                        "positive_positive"
                        if a > 0 and b > 0
                        else "negative_negative"
                        if a < 0 and b < 0
                        else "opposite_sign"
                    )
                    strata[sign + "_items"] += 1
                    strata["units_carry_items"] += int(
                        a * b > 0 and (abs(a) % 10 + abs(b) % 10 >= 10)
                    )
                    strata["units_borrow_items"] += int(
                        a * b < 0 and max(abs(a), abs(b)) % 10 < min(abs(a), abs(b)) % 10
                    )
                    strata["zero_sum_items"] += int(a + b == 0)
                group = f"core_{core:02d}"
                rows.append(
                    {
                        "observation_id": f"{group}_n{count}_{magnitude}",
                        "core_index": core,
                        "core_id": group,
                        "group_id": group,
                        "item_count": count,
                        "magnitude": magnitude,
                        "operands": selected,
                        "expected_sums": [sum(pair) for pair in selected],
                        "strata": {key: str(value) for key, value in strata.items()},
                        "messages": [
                            {
                                "role": "system",
                                "content": f"For each of the {word} data rows, add its two integers. Preserve their row order. Output exactly {word} comma-separated base-10 integer sums, with no other text.",
                            },
                            {"role": "user", "content": "\n".join(f"{a} {b}" for a, b in selected)},
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


def integer_text(field):
    field = field.strip(SPACE)
    if re.fullmatch(r"[+-]?[0123456789]+", field, flags=re.ASCII) is None:
        return None
    sign = "-" if field[0] == "-" else ""
    digits = field[1:] if field[0] in "+-" else field
    digits = digits.lstrip("0") or "0"
    return sign + digits if digits != "0" else "0"


def whole_content(text, answers):
    assert type(text) is str and type(answers) is list and len(answers) in (2, 4)
    assert all(type(x) is int for x in answers)
    pieces = text.split(",")
    return len(pieces) == len(answers) and [integer_text(x) for x in pieces] == [
        str(x) for x in answers
    ]


def rebuild_score(answers, state, tokens, text, prefix, eos):
    assert state in ("completed", "infrastructure_failed", "unattempted")
    assert (
        type(tokens) is list
        and len(tokens) <= 64
        and all(type(x) is int and x >= 0 for x in tokens)
    )
    assert (
        type(eos) is list
        and eos
        and all(type(x) is int and x >= 0 for x in eos)
        and len(set(eos)) == len(eos)
    )
    assert not any(t in eos for t in tokens[:-1])
    assert type(answers) is list and len(answers) in (2, 4) and all(type(x) is int for x in answers)
    terminal = bool(tokens and tokens[-1] in eos)
    if state == "unattempted":
        assert not tokens and text is None and prefix is None
        return {key: "missing" if key == "outcome_status" else None for key in FIELDS}
    complete = state == "completed"
    if complete:
        assert terminal or len(tokens) == 64
        assert type(text) is str
        label = int(terminal and whole_content(text, answers))
    else:
        assert text is None
        label = None
    content_len = len(tokens) - int(terminal)
    reached = True if content_len >= 4 else False if terminal else None
    result = dict(
        outcome_status="known" if complete else "missing",
        label=label,
        terminal_eos=terminal,
        capped=complete and not terminal,
        censored=not terminal and len(tokens) < 64,
        prefix_reached=reached,
        completed_fields=None,
        known_error=None,
        remaining_fields=None,
        unflagged_remaining_ge2=None,
    )
    if prefix is not None:
        assert type(prefix) is str and reached is True
        closed = prefix.split(",")[:-1]
        count = len(closed)
        wrong = count >= len(answers) or any(
            integer_text(piece) != str(answer) for piece, answer in zip(closed, answers)
        )
        remaining = max(0, len(answers) - count)
        result.update(
            completed_fields=count,
            known_error=wrong,
            remaining_fields=remaining,
            unflagged_remaining_ge2=not wrong and remaining >= 2,
        )
    return result


def rate(values):
    assert values and all(x is None or type(x) is int and x in (0, 1) for x in values)
    n = len(values)
    known = sum(x is not None for x in values)
    success = sum(x == 1 for x in values)
    return {
        "point": success / n if known == n else None,
        "lower": success / n,
        "upper": (success + n - known) / n,
        "planned_outcomes": n,
        "resolved_outcomes": known,
    }


def contrast(rows, labels, axis, positive, negative):
    assert len(rows) == len(labels) == 32 and axis in ("item_count", "magnitude")
    other = "magnitude" if axis == "item_count" else "item_count"
    paired = {}
    for row, label in zip(rows, labels):
        assert type(row["core_index"]) is int and 0 <= row["core_index"] < 8
        assert type(row["item_count"]) is int and row["item_count"] in (2, 4)
        assert row["magnitude"] in ("one_digit", "two_digit")
        pair = paired.setdefault((row["core_index"], row[other]), {})
        assert row[axis] not in pair and (label is None or type(label) is int and label in (0, 1))
        pair[row[axis]] = label
    assert len(paired) == 16 and all(set(pair) == {positive, negative} for pair in paired.values())
    lower = upper = resolved = pairs = 0
    for pair in paired.values():
        plus, minus = pair[positive], pair[negative]
        lower += (0 if plus is None else plus) - (1 if minus is None else minus)
        upper += (1 if plus is None else plus) - (0 if minus is None else minus)
        resolved += int(plus is not None) + int(minus is not None)
        pairs += int(plus is not None and minus is not None)
    return {
        "point": lower / 16 if resolved == 32 else None,
        "lower": lower / 16,
        "upper": upper / 16,
        "planned_outcomes": 32,
        "resolved_outcomes": resolved,
        "planned_pairs": 16,
        "resolved_pairs": pairs,
    }


def tri(values):
    assert all(v is None or type(v) is bool for v in values)
    return {
        "true": sum(v is True for v in values),
        "false": sum(v is False for v in values),
        "unknown": sum(v is None for v in values),
    }


def labels_count(values):
    assert all(v is None or type(v) is int and v in (0, 1) for v in values)
    return {
        "success": sum(v == 1 for v in values),
        "failure": sum(v == 0 for v in values),
        "unknown": sum(v is None for v in values),
    }


def summary(scores, rows):
    assert len(scores) == len(rows) and scores
    values = [r["label"] for r in scores]
    memberships = [
        False if r["prefix_reached"] is False else r["unflagged_remaining_ge2"] for r in scores
    ]
    eligible = [
        (row, score) for row, score, member in zip(rows, scores, memberships) if member is True
    ]

    def core_counts(items):
        return {
            name: len({row["core_index"] for row, score in items if score["label"] == value})
            for name, value in (("success", 1), ("failure", 0), ("unknown", None))
        }

    def histogram(field):
        observed = [r[field] for r in scores if r[field] is not None]
        assert all(type(x) is int and x >= 0 for x in observed)
        return {str(x): observed.count(x) for x in sorted(set(observed))}

    return {
        "planned_rows": len(scores),
        "success_rate": rate(values),
        "labels": labels_count(values),
        "core_support": core_counts(list(zip(rows, scores))),
        "terminal_eos": sum(r["terminal_eos"] is True for r in scores),
        "terminal_eos_unknown": sum(r["terminal_eos"] is None for r in scores),
        "capped": sum(r["capped"] is True for r in scores),
        "capped_unknown": sum(r["capped"] is None for r in scores),
        "censored": tri([r["censored"] for r in scores]),
        "prefix": {
            "planned_rows": len(scores),
            "reached": tri([r["prefix_reached"] for r in scores]),
            "known_error": tri([r["known_error"] for r in scores]),
            "completed_fields_counts": histogram("completed_fields"),
            "completed_fields_unknown": sum(r["completed_fields"] is None for r in scores),
            "remaining_fields_counts": histogram("remaining_fields"),
            "remaining_fields_unknown": sum(r["remaining_fields"] is None for r in scores),
            "reached_unflagged_remaining_ge2": {
                "planned_rows": len(scores),
                "membership": tri(memberships),
                "eligible_labels": labels_count([score["label"] for _, score in eligible]),
                "eligible_cores": len({row["core_index"] for row, _ in eligible}),
                "eligible_core_support": core_counts(eligible),
            },
        },
    }


def check_summary(scores, producer, rows):
    rebuilt = summary(scores, rows)
    same(rebuilt, producer)
    prefix = rebuilt["prefix"]
    eligible = prefix["reached_unflagged_remaining_ge2"]
    return {
        **{key: value for key, value in rebuilt.items() if key != "prefix"},
        "prefix_reached": prefix["reached"],
        "prefix_known_error": prefix["known_error"],
        "completed_fields_counts": prefix["completed_fields_counts"],
        "completed_fields_unknown": prefix["completed_fields_unknown"],
        "remaining_fields_counts": prefix["remaining_fields_counts"],
        "remaining_fields_unknown": prefix["remaining_fields_unknown"],
        "unflagged_membership": eligible["membership"],
        "unflagged_eligible_labels": eligible["eligible_labels"],
        "unflagged_eligible_cores": eligible["eligible_cores"],
        "unflagged_eligible_core_support": eligible["eligible_core_support"],
    }


def read_bytes(path):
    assert path.is_file() and not path.is_symlink()
    return path.read_bytes()


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value, "duplicate_json_key"
            value[key] = item
        return value

    def nonfinite(_):
        raise AssertionError("nonfinite_json")

    value = json.loads(read_bytes(path), object_pairs_hook=unique, parse_constant=nonfinite)
    canonical(value)  # Reject overflow-to-infinity and preserve JSON types.
    return value


def audit_outcomes(plan, acquired, acquisition_root):
    """Call after authenticated A189 replay; emit only seven aggregate fields."""
    root = Path(acquisition_root)
    rows, records = plan["rows"], acquired["records"]
    same(rows, reconstruct_roster())
    assert type(records) is list and len(records) == 32
    rebuilt = []
    oracle_count = content_correct = 0
    for index, (row, record) in enumerate(zip(rows, records)):
        for key in ("observation_id", "core_index", "item_count", "magnitude", "sequence_index"):
            same(record[key], row[key])
        answers = row["expected_sums"]
        base = root / "rows" / f"{index:03d}" / "generation"
        state = record["generation_status"]
        if state == "missing":
            assert record["status"] == "unattempted"
            state = "unattempted"
        if state == "unattempted":
            assert not base.exists()
            tokens, text, prefix = [], None, None
        else:
            assert state in ("completed", "infrastructure_failed")
            assert record["status"] in ("completed", "failed")
            assert record["status"] != "completed" or state == "completed"
            tokens = []
            steps = sorted(base.glob("step_*"))
            assert len(steps) <= 64 and [p.name for p in steps] == [
                f"step_{j:02d}" for j in range(len(steps))
            ]
            for position, step in enumerate(steps):
                assert step.is_dir() and not step.is_symlink()
                if (step / "result.json").exists() and not (step / "failure.json").exists():
                    token = read_json(step / "result.json")["chosen_token_id"]
                    assert type(token) is int and token >= 0
                    tokens.append(token)
                else:
                    assert state != "completed" and position == len(steps) - 1
            text = None
            if state == "completed":
                result = read_json(base / "result.json")
                same(result["token_ids"], tokens)
                same(result["observed_tokens"], len(tokens))
                terminal = bool(tokens and tokens[-1] in plan["eos_token_ids"])
                same(result["stop_reason"], "eos" if terminal else "cap")
                same(result["status"], "completed")
                text = read_bytes(base / "response.utf8").decode("utf-8")
                oracle_count += 1
                content_correct += int(whole_content(text, answers))
            prefix = None
            metadata, returned, payload = (
                base / "prefix.json",
                base / "prefix-return.json",
                base / "prefix.utf8",
            )
            content = tokens[:-1] if tokens and tokens[-1] in plan["eos_token_ids"] else tokens
            available = len(content) >= 4
            if (
                state == "completed"
                or metadata.exists()
                and returned.exists()
                and (not available or payload.exists())
            ):
                meta = read_json(metadata)
                same(
                    read_json(returned),
                    {
                        "schema_version": ACQUISITION_SCHEMA,
                        "event": "prefix_writer_returned_normally",
                        "prefix_sha256": hashlib.sha256(canonical(meta)).hexdigest(),
                    },
                )
                raw = read_bytes(payload) if available else None
                same(
                    meta,
                    {
                        "schema_version": ACQUISITION_SCHEMA,
                        "non_eos_tokens": 4,
                        "available": available,
                        "token_ids": content[:4] if available else None,
                        "decoded_utf8_sha256": hashlib.sha256(raw).hexdigest()
                        if raw is not None
                        else None,
                    },
                )
                prefix = raw.decode("utf-8") if raw is not None else None
        score = rebuild_score(answers, state, tokens, text, prefix, plan["eos_token_ids"])
        same({key: record[key] for key in FIELDS}, score)
        rebuilt.append(score)
    analysis = acquired["analysis"]
    same(analysis["schema_version"], "a189-task-load-analysis-v1")
    overall = check_summary(rebuilt, analysis["overall"], rows)
    cells = {}
    for count in (2, 4):
        cells[str(count)] = {}
        for magnitude in ("one_digit", "two_digit"):
            selected = [
                (row, score)
                for row, score in zip(rows, rebuilt)
                if row["item_count"] == count and row["magnitude"] == magnitude
            ]
            assert len(selected) == 8
            cells[str(count)][magnitude] = check_summary(
                [score for _, score in selected],
                analysis["by_condition"][str(count)][magnitude],
                [row for row, _ in selected],
            )
    labels = [r["label"] for r in rebuilt]
    primary = contrast(rows, labels, "item_count", 2, 4)
    secondary = contrast(rows, labels, "magnitude", "one_digit", "two_digit")
    same(primary, analysis["contrasts"]["success_rate"]["shorter_minus_longer"])
    same(secondary, analysis["secondary_contrasts"]["success_rate"]["one_digit_minus_two_digit"])
    return {
        "independent_response_oracle_evaluations": oracle_count,
        "independently_correct_whole_contents": content_correct,
        "overall": overall,
        "by_condition": cells,
        "primary_contrast": primary,
        "secondary_contrast": secondary,
        "record_slots": 32,
    }
