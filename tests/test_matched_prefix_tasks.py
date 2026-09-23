"""Model-free qualification of the fixed A186 constructor and boundary rules."""

import copy
import hashlib
from collections import Counter

import pytest

from lexical_prompt_study import matched_prefix_tasks as tasks


def independent_operand(core, item, side):
    value = (
        int(
            hashlib.sha256(
                f"matched-prefix-arithmetic-v1|{core}|{item}|{side}".encode()
            ).hexdigest(),
            16,
        )
        % 180
    )
    signed_two_digit_values = list(range(-99, -9)) + list(range(10, 100))
    return signed_two_digit_values[value]


def test_fixed_roster_independent_enumeration_and_schedule():
    rows = tasks.build_roster()
    expected = {
        (core, format)
        for core in range(48)
        for format in (("plain", "labeled") if core < 32 else ("plain", "labeled", "csv"))
    }
    assert {(r["core_index"], r["format"]) for r in rows} == expected
    assert len(rows) == len({r["observation_id"] for r in rows}) == 112
    assert Counter(r["split"] for r in rows) == {"fit": 64, "evaluation": 48}
    assert Counter(r["format"] for r in rows) == {"plain": 48, "labeled": 48, "csv": 16}
    independent_order = sorted(
        expected,
        key=lambda key: (
            hashlib.sha256(
                f"matched-prefix-arithmetic-v1|schedule|{key[0]}|{key[1]}".encode()
            ).digest(),
            key[0],
            key[1],
        ),
    )
    assert [(r["core_index"], r["format"]) for r in rows] == independent_order
    for index, row in enumerate(rows):
        core = row["core_index"]
        pairs = [[independent_operand(core, item, side) for side in (0, 1)] for item in range(12)]
        assert row["operands"] == pairs
        assert row["expected_sums"] == [sum(pair) for pair in pairs]
        assert row["sequence_index"] == index
        assert row["core_id"] == row["group_id"] == f"core_{core:02d}"
        assert row["observation_id"] == f"core_{core:02d}_{row['format']}"
        assert row["split"] == ("fit" if core < 32 else "evaluation")
    assert tasks.compile_roster() == rows


def test_format_rendering_and_group_separation():
    assert tasks.SYSTEM == (
        "For each of the twelve data rows, add its two integers. Preserve their row order. "
        "Output exactly twelve comma-separated base-10 integer sums, with no other text."
    )
    rows = tasks.build_roster()
    for row in rows:
        a, b = row["operands"][0]
        text = row["messages"][1]["content"]
        assert row["messages"][0] == {"role": "system", "content": tasks.SYSTEM}
        assert not text.endswith("\n")
        if row["format"] == "plain":
            assert text.splitlines()[0] == f"{a} {b}"
            assert len(text.splitlines()) == 12
        elif row["format"] == "labeled":
            assert text.splitlines()[0] == f"a={a}; b={b}"
            assert len(text.splitlines()) == 12
        else:
            assert text.splitlines()[:2] == ["a,b", f"{a},{b}"]
            assert len(text.splitlines()) == 13
        assert all(
            other["split"] == row["split"] for other in rows if other["core_id"] == row["core_id"]
        )
    assert len({r["core_id"] for r in rows if r["split"] == "evaluation"}) == 16


@pytest.mark.parametrize(
    "a,b,sign,carry,borrow,zero",
    [
        (18, 27, "positive_positive", True, False, False),
        (-18, -27, "negative_negative", True, False, False),
        (12, 23, "positive_positive", False, False, False),
        (31, -18, "opposite_sign", False, True, False),
        (-31, 18, "opposite_sign", False, True, False),
        (18, -31, "opposite_sign", False, True, False),
        (-31, 11, "opposite_sign", False, False, False),
        (-19, 19, "opposite_sign", False, False, True),
    ],
)
def test_input_only_strata(a, b, sign, carry, borrow, zero):
    assert tasks.pair_stratum(a, b) == {
        "sign": sign,
        "units_carry": carry,
        "units_borrow": borrow,
        "zero_sum": zero,
    }


def test_strata_are_preoutcome_string_counts_and_returned_containers_are_independent():
    rows = tasks.build_roster()
    for row in rows:
        strata = row["strata"]
        assert all(type(v) is str for v in strata.values())
        assert (
            sum(
                int(strata[k])
                for k in (
                    "positive_positive_items",
                    "negative_negative_items",
                    "opposite_sign_items",
                )
            )
            == 12
        )
        assert all(0 <= int(v) <= 12 for v in strata.values())
    rows[0]["messages"][0]["content"] = "tampered"
    rows[0]["operands"][0][0] = 0
    assert tasks.build_roster()[0]["messages"][0]["content"] == tasks.SYSTEM
    assert tasks.build_roster()[0]["operands"][0][0] != 0


@pytest.mark.parametrize(
    "mutation",
    [
        "delete",
        "duplicate",
        "reorder",
        "extra",
        "bool",
        "float",
        "sum",
        "split",
        "message",
        "strata",
    ],
)
def test_exact_roster_rejects_tamper(mutation):
    rows = tasks.build_roster()
    if mutation == "delete":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "extra":
        rows[0]["label"] = 1
    elif mutation == "bool":
        rows[0]["sequence_index"] = False
    elif mutation == "float":
        rows[0]["expected_sums"][0] = float(rows[0]["expected_sums"][0])
    elif mutation == "sum":
        rows[0]["expected_sums"][0] += 1
    elif mutation == "split":
        rows[0]["split"] = "evaluation" if rows[0]["split"] == "fit" else "fit"
    elif mutation == "message":
        rows[0]["messages"][1]["content"] += "\n"
    else:
        rows[0]["strata"]["units_carry_items"] = "12"
        rows[0]["strata"]["units_borrow_items"] = "12"
    with pytest.raises(ValueError):
        tasks.validate_roster(rows)


def test_prediction_roster_allowlist_and_revalidation():
    rows = tasks.build_roster()
    slim = tasks.prediction_roster(rows)
    assert len(slim) == 112
    assert all(set(row) == {"observation_id", "core_index", "format", "split"} for row in slim)
    rows[0]["label"] = 1
    with pytest.raises(ValueError):
        tasks.prediction_roster(rows)


@pytest.mark.parametrize("value", [-1, 48, True, 1.0, "0", None])
def test_constructor_indices_are_strict(value):
    with pytest.raises(ValueError):
        tasks.core_operands(value)


@pytest.mark.parametrize("whitespace", list(" \t\r\n\v\f") + [" \t\r\n\v\f", ""])
def test_oracle_accepts_only_declared_surrounding_ascii_whitespace(whitespace):
    expected = [-2, 0, 3] * 4
    response = ",".join(whitespace + str(x) + whitespace for x in expected)
    assert tasks.oracle_response(response, expected)


def test_oracle_zero_plus_leading_zeros_and_unbounded_string_normalization():
    expected = [-2, 0, 3] * 4
    assert tasks.oracle_response(",".join(["-0002", "+000", "+0003"] * 4), expected)
    assert tasks.oracle_response(",".join(["-0002", "-0", "+0003"] * 4), expected)
    assert tasks.oracle_response(",".join(["0" * 5000 + "3"] * 12), [3] * 12)
    assert not tasks.oracle_response(",".join(["9" * 5000] * 12), [3] * 12)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "1,",
        ",1",
        "1,,1",
        "1.0",
        "1e0",
        "١",
        "１",
        "−1",
        "++1",
        "+-1",
        "1 0",
        "1\x00",
        "\u00a01",
        "1\u2003",
        "1\nprose",
        "```1",
        "1;",
    ],
)
def test_oracle_rejects_malformed_or_nonascii_integer(bad):
    assert not tasks.oracle_response(",".join([bad] + ["1"] * 11), [1] * 12)


@pytest.mark.parametrize("count", [0, 1, 11, 13, 20])
def test_oracle_exact_cardinality(count):
    assert not tasks.oracle_response(",".join(["1"] * count), [1] * 12)


def test_prefix_open_tail_never_peeks_at_later_delimiter_or_final_outcome():
    expected = [1] * 12
    for tail in ("", "-", "wrong", "2", "١", "```", "1\u00a0"):
        result = tasks.prefix_check("1," + tail, expected)
        assert result == {
            "completed_fields": 1,
            "completed_fields_normalized": 1 / 12,
            "prefix_known_error": False,
        }
    assert tasks.prefix_check("1,2,", expected)["prefix_known_error"]
    assert tasks.prefix_check("1,wrong,", expected)["prefix_known_error"]


def test_prefix_twelfth_comma_and_extra_count():
    expected = [1] * 12
    assert not tasks.prefix_check(",".join(["1"] * 12), expected)["prefix_known_error"]
    at_extra = tasks.prefix_check(",".join(["1"] * 12) + ",", expected)
    assert at_extra == {
        "completed_fields": 12,
        "completed_fields_normalized": 1.0,
        "prefix_known_error": True,
    }
    more = tasks.prefix_check("1," * 15, expected)
    assert more["completed_fields"] == 15 and more["completed_fields_normalized"] == 1.25
    assert more["prefix_known_error"]


def test_prefix_error_is_irreversible_under_append_only_continuation():
    expected = [1] * 12
    for prefix in ("2,", "x,", ",", "1," * 12):
        assert tasks.prefix_check(prefix, expected)["prefix_known_error"]
        for suffix in ("", "1", "1," * 11 + "1", "\n"):
            assert not tasks.oracle_response(prefix + suffix, expected)


def test_prefix_features_reconstructs_public_input_without_future_label():
    for core in range(48):
        expected = [sum(pair) for pair in tasks.core_operands(core)]
        prefix = f"{expected[0]}, {expected[1]},wrong_unclosed"
        assert tasks.prefix_features(core, prefix) == {
            "completed_fields": 2,
            "completed_fields_normalized": 2 / 12,
            "prefix_known_error": False,
        }


def generation(status="completed", stop="eos", tokens=None):
    tokens = [1, 99] if tokens is None else tokens
    return {
        "status": status,
        "stop_reason": stop,
        "token_ids": tokens,
        "observed_tokens": len(tokens),
    }


def test_endpoint_native_eos_last_cap_position_and_correct_content():
    row = tasks.build_roster()[0]
    response = ",".join(map(str, row["expected_sums"]))
    for tokens in ([99], [1] * 63 + [99]):
        result = tasks.outcome_for_generation(row, generation(tokens=tokens), response, [99, 98])
        assert result == {
            "status": "known",
            "label": 1,
            "reason": None,
            "terminal_eos": True,
            "capped": False,
            "censored": False,
        }
    assert (
        tasks.outcome_for_generation(row, generation(tokens=[98]), response, [99, 98])["label"] == 1
    )


def test_endpoint_completed_cap_is_zero_even_if_text_correct():
    row = tasks.build_roster()[0]
    response = ",".join(map(str, row["expected_sums"]))
    result = tasks.outcome_for_generation(
        row, generation(stop="cap", tokens=[1] * 64), response, [99]
    )
    assert result == {
        "status": "known",
        "label": 0,
        "reason": None,
        "terminal_eos": False,
        "capped": True,
        "censored": False,
    }


@pytest.mark.parametrize("status", ["interrupted", "infrastructure_failed", "missing"])
def test_endpoint_partial_generation_never_gains_label(status):
    row = tasks.build_roster()[0]
    response = ",".join(map(str, row["expected_sums"]))
    result = tasks.outcome_for_generation(row, generation(status, status, []), response, [99])
    assert result["label"] is None and result["status"] == "missing"
    assert result["censored"] and result["reason"] == "not_measured"


@pytest.mark.parametrize("text", ["", "wrong", "0", "1," * 12])
def test_early_eos_or_malformed_complete_response_is_zero(text):
    assert (
        tasks.outcome_for_generation(tasks.build_roster()[0], generation(), text, [99])["label"]
        == 0
    )


@pytest.mark.parametrize(
    "bad",
    [
        generation(stop="cap", tokens=[1] * 63),
        generation(stop="eos", tokens=[1]),
        generation(tokens=[99, 1]),
        generation(tokens=[99, 99]),
        generation(tokens=[1] * 64 + [99]),
        generation(tokens=[True, 99]),
        generation(tokens=[-1, 99]),
        generation(status="missing", stop="missing", tokens=[1]),
        generation(status="interrupted", stop="interrupted", tokens=[99]),
        generation(status="invented", stop="invented", tokens=[]),
    ],
)
def test_invalid_stream_rejected_before_outcome(bad):
    with pytest.raises(ValueError):
        tasks.outcome_for_generation(tasks.build_roster()[0], bad, "", [99])


def test_outcome_and_expected_types_are_not_coerced():
    row = tasks.build_roster()[0]
    row["expected_sums"][0] = True
    with pytest.raises(ValueError):
        tasks.outcome_for_generation(row, generation(), "", [99])
    for expected in ([True] * 12, [1] * 11, tuple([1] * 12)):
        with pytest.raises(ValueError):
            tasks.oracle_response("", expected)
    with pytest.raises(ValueError):
        tasks.prefix_features(0, b"1,")


def test_pure_task_source_has_no_model_or_data_access():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tasks))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {"hashlib", "json", "re"}
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"open", "eval", "exec", "__import__"}
        for node in ast.walk(tree)
    )


def test_endpoint_does_not_infer_t8_availability_from_its_known_label():
    row = tasks.build_roster()[0]
    response = ",".join(map(str, row["expected_sums"]))
    before = tasks.outcome_for_generation(row, generation(tokens=[1] * 7 + [99]), response, [99])
    after = tasks.outcome_for_generation(row, generation(tokens=[1] * 8 + [99]), response, [99])
    assert before["label"] == after["label"] == 1
    assert "prefix" not in before and "prefix" not in after
