"""Invented, model-free A187 constructor, oracle and finite-cohort controls."""

import ast
import copy
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import pytest

from lexical_prompt_study import arithmetic_task_load as task


@pytest.fixture
def rows():
    return task.build_roster()


def generation(tokens, status="completed", stop="eos"):
    return {
        "token_ids": tokens,
        "observed_tokens": len(tokens),
        "status": status,
        "stop_reason": stop,
    }


def response(row):
    return ",".join(map(str, row["expected_sums"]))


def record(row, label=None, prefix=""):
    if label is None:
        result = task.score_generation(row, generation([], "missing", "missing"), None, None, [99])
    else:
        result = task.score_generation(
            row, generation([1] * 8 + [99]), response(row) if label else "wrong", prefix, [99]
        )
    return {
        key: row[key]
        for key in ("observation_id", "core_index", "item_count", "magnitude", "sequence_index")
    } | result


def contrast(rows, records, secondary=False):
    result = task.aggregate_records(rows, records)
    return (
        result["secondary_contrasts"]["success_rate"]["one_digit_minus_two_digit"]
        if secondary
        else result["contrasts"]["success_rate"]["shorter_minus_longer"]
    )


def test_constructor_independent_complete_factorial(rows):
    assert len(rows) == 32
    assert {(r["core_index"], r["item_count"], r["magnitude"]) for r in rows} == set(
        itertools.product(range(8), (4, 8), ("one_digit", "two_digit"))
    )
    assert len({r["observation_id"] for r in rows}) == 32
    assert len({r["group_id"] for r in rows}) == 8
    for row in rows:
        expected = []
        for item in range(row["item_count"]):
            pair = []
            for side in (0, 1):
                digest = hashlib.sha256(
                    f"arithmetic-load-feasibility-v1|{row['core_index']}|{item}|{side}".encode()
                ).digest()
                n = int(digest.hex(), 16)
                magnitude = (n % 9 + 1) if row["magnitude"] == "one_digit" else (n % 90 + 10)
                pair.append(magnitude * (-1 if digest[0] % 2 else 1))
            expected.append(pair)
        assert row["operands"] == expected
        assert row["expected_sums"] == [sum(pair) for pair in expected]
        assert row["messages"][1] == {
            "role": "user",
            "content": "\n".join(" ".join(map(str, pair)) for pair in expected),
        }


def test_schedule_independently_hashed_and_typed(rows):
    identities = list(itertools.product(range(8), (4, 8), ("one_digit", "two_digit")))
    ordered = sorted(
        identities,
        key=lambda x: (
            hashlib.sha256(
                f"arithmetic-load-feasibility-v1|schedule|{x[0]}|{x[1]}|{x[2]}".encode()
            ).hexdigest(),
            *x,
        ),
    )
    assert [(r["core_index"], r["item_count"], r["magnitude"]) for r in rows] == ordered
    assert [r["sequence_index"] for r in rows] == list(range(32))


def test_matching_subset_signs_and_fresh_containers(rows):
    lookup = {(r["core_index"], r["item_count"], r["magnitude"]): r for r in rows}
    for core in range(8):
        for magnitude in task.MAGNITUDES:
            assert (
                lookup[core, 4, magnitude]["operands"] == lookup[core, 8, magnitude]["operands"][:4]
            )
        assert [[x > 0 for x in pair] for pair in lookup[core, 8, "one_digit"]["operands"]] == [
            [x > 0 for x in pair] for pair in lookup[core, 8, "two_digit"]["operands"]
        ]
    before = task.build_roster()
    rows[0]["operands"][0][0] = 9999
    rows[1]["messages"][0]["content"] = "changed"
    assert task.build_roster() == before


@pytest.mark.parametrize(
    "a,b,sign,carry,borrow,zero",
    [
        (8, 7, "positive_positive", True, False, False),
        (-18, -12, "negative_negative", True, False, False),
        (21, -19, "opposite_sign", False, True, False),
        (-21, 19, "opposite_sign", False, True, False),
        (7, -7, "opposite_sign", False, False, True),
        (9, -1, "opposite_sign", False, False, False),
    ],
)
def test_input_strata(a, b, sign, carry, borrow, zero):
    assert task.pair_stratum(a, b) == {
        "sign": sign,
        "units_carry": carry,
        "units_borrow": borrow,
        "zero_sum": zero,
    }


def test_strata_and_instruction_counts(rows):
    for row in rows:
        strata = row["strata"]
        assert (
            sum(
                int(strata[k + "_items"])
                for k in ("positive_positive", "negative_negative", "opposite_sign")
            )
            == row["item_count"]
        )
        assert all(type(x) is str for x in strata.values())
        word = "four" if row["item_count"] == 4 else "eight"
        assert row["messages"][0]["content"] == (
            f"For each of the {word} data rows, add its two integers. Preserve their row order. "
            f"Output exactly {word} comma-separated base-10 integer sums, with no other text."
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("core_index", True),
        ("item_count", 4.0),
        ("magnitude", "easy"),
        ("sequence_index", False),
        ("observation_id", "other"),
        ("expected_sums", [0] * 4),
        ("messages", []),
        ("strata", {}),
        ("schedule_sha256", "0" * 64),
    ],
)
def test_changed_row_rejected(rows, field, value):
    row = next(r for r in rows if r["item_count"] == 4)
    row[field] = value
    with pytest.raises(ValueError):
        task.validate_row(row)
    with pytest.raises(ValueError):
        task.validate_roster(rows)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reverse", "extra"])
def test_roster_complete_slots(rows, change):
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "reverse":
        rows.reverse()
    else:
        rows[0]["label"] = 1
    with pytest.raises(ValueError):
        task.validate_roster(rows)


@pytest.mark.parametrize(
    "text",
    [
        "1,-2,0,4",
        " +0001, -0002, -0, +4\n",
        "\t1\v,\f-2\r,0,4 ",
    ],
)
def test_oracle_ascii_allowed(text):
    assert task.oracle_response(text, [1, -2, 0, 4])


@pytest.mark.parametrize(
    "text",
    [
        "1,-2,0",
        "1,-2,0,4,",
        "1,-2,0,4,5",
        "[1,-2,0,4]",
        "1,-2,0,4 done",
        "1,-2,0,4.0",
        "1,-2,0,4e0",
        "1,-2,0,٤",
        "1,-2,0,４",
        "1,−2,0,4",
        "1,-2,0,\u00a04",
        "1,-2,0,4\u200b",
        "1,-2,0,+ 4",
        "1,-2,0,++4",
        "1,-2,0,5",
        "1,0,-2,4",
        "1,-2,0,4\x00",
        "```1,-2,0,4```",
    ],
)
def test_oracle_rejects_changed_grammar_or_value(text):
    assert not task.oracle_response(text, [1, -2, 0, 4])


def test_eight_and_huge_leading_zeros_without_int_conversion():
    assert task.oracle_response("0," * 7 + "0", [0] * 8)
    assert not task.oracle_response("0," * 7 + "0", [0] * 4)
    assert task.oracle_response("+" + "0" * 10000 + "1,-2,0,4", [1, -2, 0, 4])


@pytest.mark.parametrize(
    "text,closed,error,remaining,eligible",
    [
        ("", 0, False, 4, True),
        ("99", 0, False, 4, True),
        ("bad unfinished", 0, False, 4, True),
        ("1,", 1, False, 3, True),
        ("1,-2,", 2, False, 2, True),
        ("1,-2,0,", 3, False, 1, False),
        ("1,-2,0,4", 3, False, 1, False),
        ("1,-2,0,4,", 4, True, 0, False),
        ("1,-2,0,4,9,", 5, True, 0, False),
        ("2,", 1, True, 3, False),
        (",", 1, True, 3, False),
        ("1,garbage,", 2, True, 2, False),
    ],
)
def test_prefix_closed_only(text, closed, error, remaining, eligible):
    assert task.prefix_check(text, [1, -2, 0, 4]) == {
        "completed_fields": closed,
        "known_error": error,
        "remaining_fields": remaining,
        "unflagged_remaining_ge2": eligible,
    }


def test_flag_is_irreversible_but_unflagged_is_not_forecast_proof():
    for suffix in ("", "correct", "1,-2,0,4", ",,,,", "\n"):
        assert task.prefix_check("9," + suffix, [1, -2, 0, 4])["known_error"]
    assert not task.prefix_check("wrong unclosed", [1, -2, 0, 4])["known_error"]


@pytest.mark.parametrize("token_count", [1, 8, 9, 64])
def test_native_eos_inclusive_cap_and_landmark(rows, token_count):
    row = rows[0]
    gen = generation([1] * (token_count - 1) + [99])
    reached = token_count - 1 >= 8
    score = task.score_generation(row, gen, response(row), "" if reached else None, [99])
    assert score["label"] == 1 and score["terminal_eos"] and not score["capped"]
    assert score["prefix_reached"] is reached


def test_cap_never_rescued_by_correct_text(rows):
    score = task.score_generation(
        rows[0], generation([1] * 64, stop="cap"), response(rows[0]), "", [99]
    )
    assert score["label"] == 0 and score["capped"] and not score["terminal_eos"]
    assert not score["censored"] and score["prefix_reached"] is True


@pytest.mark.parametrize(
    "status,n",
    [
        ("missing", 0),
        ("interrupted", 0),
        ("interrupted", 7),
        ("infrastructure_failed", 8),
        ("infrastructure_failed", 64),
    ],
)
def test_partial_text_never_gets_outcome_label(rows, status, n):
    score = task.score_generation(
        rows[0], generation([1] * n, status, status), response(rows[0]), None, [99]
    )
    assert score["label"] is None and score["outcome_status"] == "missing"
    assert score["prefix_reached"] is (True if n >= 8 else None)
    assert score["known_error"] is None


@pytest.mark.parametrize(
    "gen,eos",
    [
        (generation([99, 1, 99]), [99]),
        (generation([99, 99]), [99]),
        (generation([1] * 63, stop="cap"), [99]),
        (generation([1], stop="eos"), [99]),
        (generation([1], "missing", "missing"), [99]),
        (generation([True, 99]), [99]),
        (generation([1, 99]), [True]),
        (generation([1, 99]), [99, 99]),
        (generation([1, 99]), []),
        (generation([1] * 65, stop="cap"), [99]),
    ],
)
def test_invalid_generation_binding_rejected(rows, gen, eos):
    with pytest.raises(ValueError):
        task.score_generation(rows[0], gen, response(rows[0]), None, eos)


def test_prefix_not_permitted_before_landmark(rows):
    with pytest.raises(ValueError, match="prefix_without_landmark"):
        task.score_generation(rows[0], generation([1, 99]), response(rows[0]), "", [99])


@pytest.mark.parametrize("n", [0, 7, 8, 63])
def test_retained_eos_with_failed_parent_does_not_impute_label(rows, n):
    score = task.score_generation(
        rows[0],
        generation([1] * n + [99], "infrastructure_failed", "infrastructure_failed"),
        None,
        None,
        [99],
    )
    assert score["terminal_eos"] is True and score["label"] is None
    assert score["prefix_reached"] is (n >= 8)
    records = [record(r) for r in rows]
    records[0].update(score)
    result = task.aggregate_records(rows, records)
    assert result["overall"]["labels"]["unknown"] == 32
    assert result["overall"]["terminal_eos"] == 1
    assert result["overall"]["terminal_eos_unknown"] == 31


def test_unattempted_stop_facts_are_null(rows):
    records = [record(r) for r in rows]
    assert all(records[0][key] is None for key in ("terminal_eos", "capped", "censored"))
    result = task.aggregate_records(rows, records)
    assert result["overall"]["terminal_eos_unknown"] == 32
    assert result["overall"]["capped_unknown"] == 32
    assert result["overall"]["censored"] == {"true": 0, "false": 0, "unknown": 32}


def test_all_missing_has_full_binary_bounds_and_fixed_denominators(rows):
    records = [record(r) for r in rows]
    result = task.aggregate_records(rows, records)
    assert result["overall"]["success_rate"] == {
        "point": None,
        "lower": 0.0,
        "upper": 1.0,
        "planned_outcomes": 32,
        "resolved_outcomes": 0,
    }
    for secondary in (False, True):
        assert contrast(rows, records, secondary) == {
            "point": None,
            "lower": -1.0,
            "upper": 1.0,
            "planned_outcomes": 32,
            "resolved_outcomes": 0,
            "planned_pairs": 16,
            "resolved_pairs": 0,
        }
    for count in ("4", "8"):
        for magnitude in task.MAGNITUDES:
            assert result["by_condition"][count][magnitude]["planned_rows"] == 8


@pytest.mark.parametrize(
    "mode,primary,secondary",
    [
        ("floor", 0, 0),
        ("ceiling", 0, 0),
        ("short", 1, 0),
        ("long", -1, 0),
        ("small", 0, 1),
        ("large", 0, -1),
    ],
)
def test_contrast_signs_complete_finite_cells(rows, mode, primary, secondary):
    labels = {
        "floor": lambda r: 0,
        "ceiling": lambda r: 1,
        "short": lambda r: int(r["item_count"] == 4),
        "long": lambda r: int(r["item_count"] == 8),
        "small": lambda r: int(r["magnitude"] == "one_digit"),
        "large": lambda r: int(r["magnitude"] == "two_digit"),
    }
    records = [record(r, labels[mode](r)) for r in rows]
    for expected, is_secondary in ((primary, False), (secondary, True)):
        result = contrast(rows, records, is_secondary)
        assert result["point"] == result["lower"] == result["upper"] == expected
        assert result["resolved_outcomes"] == 32 and result["resolved_pairs"] == 16


def test_signed_bounds_exhaustive_missing_assignments(rows):
    records = [record(r, int(r["core_index"] % 2 == 0)) for r in rows]
    missing_indices = [0, 3, 10, 21]
    for index in missing_indices:
        records[index] = record(rows[index])
    for secondary in (False, True):
        observed = contrast(rows, records, secondary)
        exhaustive = []
        for replacements in itertools.product((0, 1), repeat=4):
            completed = copy.deepcopy(records)
            for index, label in zip(missing_indices, replacements, strict=True):
                completed[index] = record(rows[index], label)
            signed = sum(
                Fraction(
                    r["label"]
                    * (
                        1
                        if (r["magnitude"] == "one_digit" if secondary else r["item_count"] == 4)
                        else -1
                    ),
                    16,
                )
                for r in completed
            )
            exhaustive.append(float(signed))
        assert observed["point"] is None
        assert (observed["lower"], observed["upper"]) == (min(exhaustive), max(exhaustive))
        assert observed["resolved_outcomes"] == 28


def test_single_unknown_weight_and_no_observed_subset_estimate(rows):
    records = [record(r, 0) for r in rows]
    index = next(i for i, r in enumerate(rows) if r["item_count"] == 4)
    records[index] = record(rows[index])
    result = contrast(rows, records)
    assert result["point"] is None
    assert result["lower"] == 0 and result["upper"] == 1 / 16
    assert result["resolved_pairs"] == 15
    assert task.aggregate_records(rows, records)["overall"]["success_rate"]["point"] is None


def test_prefix_diagnostics_do_not_filter_primary(rows):
    records = [record(r, 1, "bad,") for r in rows]
    result = task.aggregate_records(rows, records)
    assert result["overall"]["success_rate"]["point"] == 1
    assert contrast(rows, records)["resolved_outcomes"] == 32
    assert result["overall"]["prefix"]["known_error"] == {"true": 32, "false": 0, "unknown": 0}
    subgroup = result["overall"]["prefix"]["reached_unflagged_remaining_ge2"]
    assert subgroup["membership"] == {"true": 0, "false": 32, "unknown": 0}
    assert subgroup["eligible_cores"] == 0 and "point" not in subgroup


def test_early_eos_versus_unknown_prefix_membership(rows):
    records = [record(r) for r in rows]
    row = rows[0]
    early = task.score_generation(row, generation([1, 99]), response(row), None, [99])
    records[0].update(early)
    reached_but_unavailable = task.score_generation(
        rows[1], generation([1] * 8 + [99]), "bad", None, [99]
    )
    records[1].update(reached_but_unavailable)
    prefix = task.aggregate_records(rows, records)["overall"]["prefix"]
    assert prefix["reached"] == {"true": 1, "false": 1, "unknown": 30}
    assert prefix["reached_unflagged_remaining_ge2"]["membership"] == {
        "true": 0,
        "false": 1,
        "unknown": 31,
    }
    assert prefix["completed_fields_unknown"] == 32


def test_subgroup_reports_core_support_without_row_replication_claim(rows):
    records = [record(r, int(r["item_count"] == 4)) for r in rows]
    subgroup = task.aggregate_records(rows, records)["overall"]["prefix"][
        "reached_unflagged_remaining_ge2"
    ]
    assert subgroup["eligible_labels"] == {"success": 16, "failure": 16, "unknown": 0}
    assert subgroup["eligible_cores"] == 8
    assert subgroup["eligible_core_support"] == {"success": 8, "failure": 8, "unknown": 0}


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", True),
        ("core_index", True),
        ("terminal_eos", 1),
        ("prefix_reached", 1),
        ("known_error", 0),
        ("completed_fields", True),
        ("remaining_fields", -1),
        ("unflagged_remaining_ge2", False),
        ("outcome_status", "missing"),
        ("capped", True),
    ],
)
def test_analysis_rejects_typed_or_inconsistent_records(rows, field, value):
    records = [record(r, 1) for r in rows]
    records[0][field] = value
    with pytest.raises(ValueError):
        task.aggregate_records(rows, records)


def test_analysis_rejects_drops_duplicates_order_and_emits_aggregate_only(rows):
    records = [record(r, 1) for r in rows]
    for invalid in (records[:-1], list(reversed(records)), records[:-1] + [records[0]]):
        with pytest.raises(ValueError):
            task.aggregate_records(rows, invalid)
    records[0]["result_sha256"] = "restricted-per-item-hash"
    serialized = json.dumps(task.aggregate_records(rows, records))
    assert "restricted-per-item-hash" not in serialized
    assert not any(row["observation_id"] in serialized for row in rows)
    assert "messages" not in serialized and "operands" not in serialized


def test_source_has_only_pure_stdlib_imports():
    source = Path(task.__file__).read_text()
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports <= {"__future__", "hashlib", "json", "re", "collections", "fractions"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"open", "eval", "exec", "__import__"}
        for node in ast.walk(tree)
    )
