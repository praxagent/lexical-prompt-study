"""Independent stdlib A191 roster/oracle/analysis; no producing-module imports.

Only audit_outcomes reads evidence, after the main verifier authenticates native
call chains, decoding provenance and inventory. This helper neither tokenizes
nor executes a model. Its public audit result is aggregate-only.
"""

from pathlib import Path
import hashlib
import json
import re

if not __debug__:
    raise RuntimeError("independent_a191_assertions_required")

NAMESPACE = "a191-answer-availability-v1"
ACQUISITION_SCHEMA = "a191-answer-availability-acquisition-v1"
SYSTEM = (
    "Each of the two data rows contains a left integer, a right integer, and an answer field. "
    "If the answer field is ?, add the two integers; otherwise copy the supplied answer. "
    "Preserve row order. Output exactly two comma-separated base-10 integer answers, "
    "with no other text."
)
IDENTITY = ("observation_id", "core_index", "core_id", "group_id", "condition", "sequence_index")
FIELDS = ("outcome_status", "label", "terminal_eos", "capped", "censored")
SPACE = " \t\r\n\v\f"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def same(actual, expected):
    assert canonical(actual) == canonical(expected), "independent_typed_mismatch"


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def reconstruct_roster():
    cores = []
    for core in range(8):
        key = hashlib.sha256(f"{NAMESPACE}|schedule|{core}".encode("ascii")).hexdigest()
        cores.append((key, core))
    rows = []
    for pair_index, (key, core) in enumerate(sorted(cores)):
        for position in range(2):
            condition = "computed" if (pair_index + position) % 2 == 0 else "supplied"
            operands = []
            for item in range(2):
                values = []
                for side in range(2):
                    raw = hashlib.sha256(
                        f"{NAMESPACE}|{core}|{item}|{side}".encode("ascii")
                    ).digest()
                    # Streaming modular arithmetic, independent of the producer's
                    # whole-digest integer conversion.
                    remainder = 0
                    for byte in raw:
                        remainder = (256 * remainder + byte) % 90
                    values.append((10 + remainder) * (-1 if raw[0] % 2 else 1))
                operands.append(values)
            answers = [sum(values) for values in operands]
            strata = {
                "positive_positive_items": sum(a > 0 and b > 0 for a, b in operands),
                "negative_negative_items": sum(a < 0 and b < 0 for a, b in operands),
                "opposite_sign_items": sum(a * b < 0 for a, b in operands),
                "units_carry_items": sum(
                    a * b > 0 and abs(a) % 10 + abs(b) % 10 >= 10 for a, b in operands
                ),
                "units_borrow_items": sum(
                    a * b < 0 and max(abs(a), abs(b)) % 10 < min(abs(a), abs(b)) % 10
                    for a, b in operands
                ),
                "zero_sum_items": sum(a == -b for a, b in operands),
            }
            group = f"core_{core:02d}"
            lines = [
                f"{a} {b} {str(answer) if condition == 'supplied' else '?'}"
                for (a, b), answer in zip(operands, answers)
            ]
            rows.append(
                {
                    "observation_id": f"{group}_{condition}",
                    "core_index": core,
                    "core_id": group,
                    "group_id": group,
                    "condition": condition,
                    "operands": operands,
                    "expected_sums": answers,
                    "strata": {name: str(value) for name, value in strata.items()},
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "\n".join(lines)},
                    ],
                    "core_schedule_sha256": key,
                    "pair_index": pair_index,
                    "within_pair_index": position,
                    "sequence_index": len(rows),
                }
            )
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
    assert type(text) is str
    assert type(answers) is list and len(answers) == 2 and all(type(x) is int for x in answers)
    fields = text.split(",")
    return len(fields) == 2 and [integer_text(x) for x in fields] == [str(x) for x in answers]


def rebuild_score(answers, state, tokens, text, eos):
    assert state in ("completed", "infrastructure_failed", "unattempted")
    assert type(answers) is list and len(answers) == 2 and all(type(x) is int for x in answers)
    assert (
        type(tokens) is list
        and len(tokens) <= 64
        and all(type(x) is int and x >= 0 for x in tokens)
    )
    assert type(eos) is list and eos and all(type(x) is int and x >= 0 for x in eos)
    assert len(set(eos)) == len(eos) and not any(token in eos for token in tokens[:-1])
    if state == "unattempted":
        assert tokens == [] and text is None
        return {name: "missing" if name == "outcome_status" else None for name in FIELDS}
    terminal = bool(tokens and tokens[-1] in eos)
    complete = state == "completed"
    if complete:
        assert terminal or len(tokens) == 64
        assert type(text) is str
        label = int(terminal and whole_content(text, answers))
    else:
        assert text is None
        label = None
    return {
        "outcome_status": "known" if complete else "missing",
        "label": label,
        "terminal_eos": terminal,
        "capped": complete and not terminal,
        "censored": not terminal and len(tokens) < 64,
    }


def summary(scores):
    assert type(scores) is list and scores
    labels = [score["label"] for score in scores]
    assert all(x is None or type(x) is int and x in (0, 1) for x in labels)
    for score in scores:
        assert all(
            score[key] is None or type(score[key]) is bool
            for key in ("terminal_eos", "capped", "censored")
        )
    planned, missing = len(scores), sum(x is None for x in labels)
    successes = sum(x == 1 for x in labels)
    return {
        "planned_rows": planned,
        "success_rate": {
            "point": successes / planned if not missing else None,
            "lower": successes / planned,
            "upper": (successes + missing) / planned,
            "planned_outcomes": planned,
            "resolved_outcomes": planned - missing,
        },
        "labels": {
            "success": successes,
            "failure": planned - missing - successes,
            "unknown": missing,
        },
        "terminal_eos": sum(score["terminal_eos"] is True for score in scores),
        "terminal_eos_unknown": sum(score["terminal_eos"] is None for score in scores),
        "capped": sum(score["capped"] is True for score in scores),
        "capped_unknown": sum(score["capped"] is None for score in scores),
        "censored": {
            name: sum(score["censored"] is value for score in scores)
            for name, value in (("true", True), ("false", False), ("unknown", None))
        },
    }


def reconstruct_analysis(rows, scores):
    same(rows, reconstruct_roster())
    assert type(scores) is list and len(scores) == 16
    per_condition = {name: [] for name in ("computed", "supplied")}
    paired = {}
    for row, score in zip(rows, scores):
        value = score["label"]
        assert value is None or type(value) is int and value in (0, 1)
        per_condition[row["condition"]].append(score)
        paired.setdefault(row["core_index"], {})[row["condition"]] = value
    assert len(paired) == 8 and all(len(values) == 2 for values in paired.values())
    lower = upper = known = resolved_pairs = 0
    counts = dict.fromkeys(
        ("both_success", "both_failure", "supplied_only", "computed_only", "unknown"), 0
    )
    for values in paired.values():
        supplied, computed = values["supplied"], values["computed"]
        possible = [
            s - c
            for s in ((0, 1) if supplied is None else (supplied,))
            for c in ((0, 1) if computed is None else (computed,))
        ]
        lower += min(possible)
        upper += max(possible)
        known += int(supplied is not None) + int(computed is not None)
        if supplied is None or computed is None:
            category = "unknown"
        else:
            resolved_pairs += 1
            category = (
                "both_success"
                if supplied == computed == 1
                else "both_failure"
                if supplied == computed == 0
                else "supplied_only"
                if supplied
                else "computed_only"
            )
        counts[category] += 1
    supplied_labels = [x["label"] for x in per_condition["supplied"]]
    qualification = (
        False
        if any(x == 0 for x in supplied_labels)
        else None
        if any(x is None for x in supplied_labels)
        else True
    )
    contrast = {
        "point": lower / 8 if known == 16 else None,
        "lower": lower / 8,
        "upper": upper / 8,
        "planned_outcomes": 16,
        "resolved_outcomes": known,
        "planned_pairs": 8,
        "resolved_pairs": resolved_pairs,
    }
    return {
        "schema_version": "a191-answer-availability-analysis-v1",
        "overall": summary(scores),
        "by_condition": {name: summary(values) for name, values in per_condition.items()},
        "contrasts": {"success_rate": {"supplied_minus_computed": contrast}},
        "supplied_qualification": qualification,
        "paired_outcomes": {"planned_pairs": 8, "resolved_pairs": resolved_pairs, "counts": counts},
    }


def read_bytes(path):
    assert path.is_file() and not path.is_symlink(), "independent_regular_file"
    return path.read_bytes()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, "independent_duplicate_json_key"
            result[key] = value
        return result

    def nonfinite(_):
        raise AssertionError("independent_nonfinite_json")

    result = json.loads(read_bytes(path), object_pairs_hook=unique, parse_constant=nonfinite)
    canonical(result)
    return result


def _retained_tokens(root, completed):
    if not root.exists():
        assert not completed
        return [], []
    assert root.is_dir() and not root.is_symlink()
    steps = sorted(root.glob("step_*"))
    assert len(steps) <= 64 and [p.name for p in steps] == [
        f"step_{i:02d}" for i in range(len(steps))
    ]
    tokens, hashes = [], []
    for index, step in enumerate(steps):
        assert step.is_dir() and not step.is_symlink()
        if (step / "failure.json").exists() or not (step / "result.json").exists():
            assert not completed and index == len(steps) - 1
            break
        result = read_json(step / "result.json")
        token = result["chosen_token_id"]
        assert type(token) is int and token >= 0
        tokens.append(token)
        hashes.append(digest(result))
    return tokens, hashes


def audit_outcomes(plan, records, acquisition_root):
    """Read only after native-chain authentication; return no per-item material."""
    root = Path(acquisition_root)
    rows = plan["rows"]
    same(rows, reconstruct_roster())
    assert type(records) is list and len(records) == 16
    rebuilt, oracle_count, content_correct = [], 0, 0
    run_sha = None
    for index, (row, record) in enumerate(zip(rows, records)):
        for key in IDENTITY:
            same(record[key], row[key])
        assert record["status"] in ("completed", "failed", "unattempted")
        state = {
            "completed": "completed",
            "failed": "infrastructure_failed",
            "unattempted": "unattempted",
        }[record["status"]]
        same(record["generation_status"], "missing" if state == "unattempted" else state)
        base = root / "rows" / f"{index:03d}"
        if state == "unattempted":
            assert not base.exists()
            tokens, step_hashes = [], []
        else:
            tokens, step_hashes = _retained_tokens(base, state == "completed")
        text = None
        if state == "completed":
            receipt = read_json(base / "result.json")
            same(
                read_json(base / "return.json"),
                {
                    "event": "generation_returned_normally",
                    "result_sha256": digest(receipt),
                },
            )
            terminal = bool(tokens and tokens[-1] in plan["eos_token_ids"])
            generation = {
                "status": "completed",
                "stop_reason": "eos" if terminal else "cap",
                "token_ids": tokens,
                "observed_tokens": len(tokens),
            }
            raw = read_bytes(base / "response.utf8")
            text = raw.decode("utf-8")
            if run_sha is None:
                run_sha = digest(read_json(root / "run.json"))
            same(receipt["schema_version"], ACQUISITION_SCHEMA)
            same(receipt["run_sha256"], run_sha)
            same(receipt["row_sha256"], digest(row))
            same(receipt["generation"], generation)
            same(receipt["step_result_sha256"], step_hashes)
            same(receipt["response_sha256"], hashlib.sha256(raw).hexdigest())
            oracle_count += 1
            content_correct += int(whole_content(text, row["expected_sums"]))
        else:
            assert not (base / "return.json").exists()
        score = rebuild_score(row["expected_sums"], state, tokens, text, plan["eos_token_ids"])
        same({key: record[key] for key in FIELDS}, score)
        if state == "completed":
            same(receipt["score"], score)
        rebuilt.append(score)
    return {
        "independent_response_oracle_evaluations": oracle_count,
        "independently_correct_whole_contents": content_correct,
        "record_slots": 16,
        "analysis": reconstruct_analysis(rows, rebuilt),
    }
