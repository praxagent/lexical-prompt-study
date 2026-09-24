"""Pure invented A191 task, oracle and partial-identification controls."""

import copy
import hashlib
import itertools

import pytest

from lexical_prompt_study import answer_availability_tasks as tasks


def records_for(rows, labels):
    records = []
    for row, label in zip(rows, labels, strict=True):
        records.append(
            {
                **{key: row[key] for key in tasks.IDENTITY_KEYS},
                "outcome_status": "missing" if label is None else "known",
                "label": label,
                "terminal_eos": None if label is None else True,
                "capped": None if label is None else False,
                "censored": None if label is None else False,
            }
        )
    return records


def generation(tokens, status="completed", stop=None):
    return {
        "token_ids": tokens,
        "observed_tokens": len(tokens),
        "status": status,
        "stop_reason": stop if stop is not None else "eos" if status == "completed" else None,
    }


def test_exact_fresh_roster_and_pair_answer_identity():
    rows = tasks.build_roster()
    assert len(rows) == 16
    assert tasks.validate_roster(rows) == rows
    assert {r["core_index"] for r in rows} == set(range(8))
    assert len({r["observation_id"] for r in rows}) == 16
    assert len({r["messages"][0]["content"] for r in rows}) == 1
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert first["core_index"] == second["core_index"]
        assert first["operands"] == second["operands"]
        assert first["expected_sums"] == second["expected_sums"]
        assert first["strata"] == second["strata"]
        assert set((first["condition"], second["condition"])) == {"computed", "supplied"}
        for row in (first, second):
            assert row["expected_sums"] == [a + b for a, b in row["operands"]]
            fields = [line.split(" ") for line in row["messages"][1]["content"].split("\n")]
            assert len(fields) == 2 and all(len(f) == 3 for f in fields)
            assert [[int(f[0]), int(f[1])] for f in fields] == row["operands"]
            assert [f[2] for f in fields] == (
                ["?", "?"]
                if row["condition"] == "computed"
                else list(map(str, row["expected_sums"]))
            )
            assert all(10 <= abs(x) <= 99 for pair in row["operands"] for x in pair)


def test_hash_core_schedule_and_balanced_order():
    rows = tasks.build_roster()
    expected = sorted(
        (
            hashlib.sha256(
                f"a191-answer-availability-v1|schedule|{core}".encode("ascii")
            ).hexdigest(),
            core,
        )
        for core in range(8)
    )
    assert [(r["core_schedule_sha256"], r["core_index"]) for r in rows[::2]] == expected
    assert [r["sequence_index"] for r in rows] == list(range(16))
    assert [r["pair_index"] for r in rows] == [i // 2 for i in range(16)]
    assert [r["within_pair_index"] for r in rows] == [i % 2 for i in range(16)]
    assert [r["condition"] for r in rows[::2]].count("computed") == 4
    assert [r["condition"] for r in rows[::2]].count("supplied") == 4


def test_operand_mapping_matches_independent_byte_modulus():
    for core, item, side in itertools.product(range(8), range(2), range(2)):
        raw = hashlib.sha256(
            f"a191-answer-availability-v1|{core}|{item}|{side}".encode("ascii")
        ).digest()
        residue = 0
        for byte in raw:
            residue = (residue * 256 + byte) % 90
        assert tasks.operand(core, item, side) == (10 + residue) * (-1 if raw[0] % 2 else 1)


@pytest.mark.parametrize(
    "field", ["expected_sums", "messages", "strata", "core_schedule_sha256", "sequence_index"]
)
def test_full_roster_binding_rejects_mutations(field):
    rows = tasks.build_roster()
    rows[0][field] = None
    with pytest.raises(ValueError, match="roster_mismatch"):
        tasks.validate_roster(rows)


def test_pairs_and_calls_have_no_mutable_alias():
    rows = tasks.build_roster()
    before = copy.deepcopy(rows[1])
    rows[0]["operands"][0][0] = 0
    rows[0]["messages"][0]["content"] = "changed"
    assert rows[1] == before
    assert tasks.build_roster()[1] == before


@pytest.mark.parametrize(
    "arguments", [(True, 0, 0), (8, 0, 0), (0, 2, 0), (0, 0, False), (0, 0, 2)]
)
def test_operand_types_and_bounds(arguments):
    with pytest.raises(ValueError):
        tasks.operand(*arguments)


def test_exact_ascii_oracle_and_unbounded_digit_normalization():
    assert tasks.oracle_response("\t+0003 , -0004\r\n", [3, -4])
    assert tasks.oracle_response("-000,+0", [0, 0])
    assert tasks.oracle_response("0" * 6000 + "3,-4", [3, -4])
    assert not tasks.oracle_response("9" * 6000 + ",-4", [3, -4])


@pytest.mark.parametrize(
    "text",
    [
        "3",
        "3,-4,0",
        "3,-4,",
        "[3,-4]",
        "3.0,-4",
        "３,-4",
        "3,−4",
        "3,-4\u00a0",
        "Answers:3,-4",
        "-4,3",
        "3;-4",
        "3,,",
    ],
)
def test_exact_oracle_rejects_other_grammars(text):
    assert not tasks.oracle_response(text, [3, -4])


def test_eos_success_cap_failure_and_terminal_eos_at_cap():
    row = tasks.build_roster()[0]
    text = ",".join(map(str, row["expected_sums"]))
    exact = tasks.score_generation(row, generation([7, 99]), text, [99])
    assert exact == {
        "outcome_status": "known",
        "label": 1,
        "terminal_eos": True,
        "capped": False,
        "censored": False,
    }
    cap = tasks.score_generation(row, generation([7] * 64, stop="cap"), text, [99])
    assert cap["label"] == 0 and cap["capped"] is True
    assert tasks.score_generation(row, generation([7] * 63 + [99]), text, [99])["label"] == 1
    assert tasks.score_generation(row, generation([99]), "", [99])["label"] == 0


def test_incomplete_generation_never_gets_known_label_even_after_eos():
    row = tasks.build_roster()[0]
    text = ",".join(map(str, row["expected_sums"]))
    for tokens in ([], [7] * 3, [7, 99], [7] * 64):
        score = tasks.score_generation(
            row, generation(tokens, status="infrastructure_failed"), text, [99]
        )
        assert score["label"] is None and score["outcome_status"] == "missing"
        assert score["terminal_eos"] is bool(tokens and tokens[-1] == 99)
        assert score["capped"] is False
        assert score["censored"] is (99 not in tokens and len(tokens) < 64)


def test_unattempted_preserves_unknown_stopping_facts():
    score = tasks.score_generation(
        tasks.build_roster()[0], generation([], status="unattempted"), None, [99]
    )
    assert score == {
        "outcome_status": "missing",
        "label": None,
        "terminal_eos": None,
        "capped": None,
        "censored": None,
    }


@pytest.mark.parametrize(
    "gen",
    [
        generation([99, 7, 99]),
        generation([7]),
        generation([7] * 65, stop="cap"),
        generation([True, 99]),
        generation([7], status="unattempted"),
        generation([7, 99], stop="cap"),
    ],
)
def test_invalid_termination_or_token_sequences_rejected(gen):
    with pytest.raises(ValueError):
        tasks.score_generation(tasks.build_roster()[0], gen, "0,0", [99])


def test_invalid_eos_and_observed_count_types_rejected():
    row = tasks.build_roster()[0]
    for eos in ([], [True], [99, 99]):
        with pytest.raises(ValueError):
            tasks.score_generation(row, generation([], status="unattempted"), None, eos)
    gen = generation([99])
    gen["observed_tokens"] = True
    with pytest.raises(ValueError, match="observed_tokens"):
        tasks.score_generation(row, gen, "0,0", [99])


def test_known_rate_contrast_and_pair_categories():
    rows = tasks.build_roster()
    outcomes = [(1, 1), (0, 0), (1, 0), (0, 1)] * 2
    labels = [outcomes[r["pair_index"]][0 if r["condition"] == "supplied" else 1] for r in rows]
    result = tasks.aggregate_records(rows, records_for(rows, labels))
    assert result["overall"]["success_rate"]["point"] == 0.5
    assert result["contrasts"]["success_rate"]["supplied_minus_computed"]["point"] == 0
    assert result["paired_outcomes"] == {
        "planned_pairs": 8,
        "resolved_pairs": 8,
        "counts": {
            "both_success": 2,
            "both_failure": 2,
            "supplied_only": 2,
            "computed_only": 2,
            "unknown": 0,
        },
    }


def test_supplied_minus_computed_direction():
    rows = tasks.build_roster()
    for condition, expected in (("supplied", 1), ("computed", -1)):
        labels = [int(r["condition"] == condition) for r in rows]
        result = tasks.aggregate_records(rows, records_for(rows, labels))
        assert result["contrasts"]["success_rate"]["supplied_minus_computed"]["point"] == expected


def test_all_unknown_and_one_missing_fixed_denominators():
    rows = tasks.build_roster()
    result = tasks.aggregate_records(rows, records_for(rows, [None] * 16))
    contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
    assert contrast == {
        "point": None,
        "lower": -1.0,
        "upper": 1.0,
        "planned_outcomes": 16,
        "resolved_outcomes": 0,
        "planned_pairs": 8,
        "resolved_pairs": 0,
    }
    assert result["paired_outcomes"]["counts"]["unknown"] == 8
    labels = [0] * 16
    index = next(i for i, r in enumerate(rows) if r["condition"] == "supplied")
    labels[index] = None
    result = tasks.aggregate_records(rows, records_for(rows, labels))
    contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
    assert (contrast["point"], contrast["lower"], contrast["upper"]) == (None, 0, 0.125)
    assert (contrast["resolved_outcomes"], contrast["resolved_pairs"]) == (15, 7)
    assert result["by_condition"]["computed"]["success_rate"]["point"] == 0
    assert result["by_condition"]["supplied"]["success_rate"]["point"] is None
    assert result["overall"]["success_rate"]["upper"] == 1 / 16


def test_missing_bounds_exhaustive_assignments():
    rows = tasks.build_roster()
    for first in itertools.product((None, 0, 1), repeat=4):
        labels = list(first) + [0] * 12
        result = tasks.aggregate_records(rows, records_for(rows, labels))
        contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
        absent = [i for i, x in enumerate(labels) if x is None]
        complete_points = []
        for assignment in itertools.product((0, 1), repeat=len(absent)):
            full = list(labels)
            for index, value in zip(absent, assignment, strict=True):
                full[index] = value
            complete_points.append(
                sum(
                    (1 if r["condition"] == "supplied" else -1) * label
                    for r, label in zip(rows, full, strict=True)
                )
                / 8
            )
        assert (contrast["lower"], contrast["upper"]) == (
            min(complete_points),
            max(complete_points),
        )
        assert contrast["point"] == (None if absent else complete_points[0])
        expected_unknown = sum(
            any(labels[i] is None for i in (2 * pair, 2 * pair + 1)) for pair in range(8)
        )
        assert result["paired_outcomes"]["counts"]["unknown"] == expected_unknown
        assert sum(result["paired_outcomes"]["counts"].values()) == 8


def test_record_identity_types_coverage_and_outcome_consistency():
    rows = tasks.build_roster()
    records = records_for(rows, [None] * 16)
    with pytest.raises(ValueError, match="record_slots"):
        tasks.aggregate_records(rows, records[:-1])
    mutations = [
        {"core_index": float(records[0]["core_index"])},
        {"label": True},
        {"label": 1, "outcome_status": "known"},
        {"terminal_eos": False},
        {"terminal_eos": True, "capped": True, "censored": False},
    ]
    for changes in mutations:
        bad = copy.deepcopy(records)
        bad[0].update(changes)
        with pytest.raises(ValueError):
            tasks.aggregate_records(rows, bad)


def test_aggregate_contains_no_text_tokens_or_row_ids():
    rows = tasks.build_roster()
    result = tasks.aggregate_records(rows, records_for(rows, [0] * 16))
    serialized = repr(result)
    assert "observation_id" not in serialized and "messages" not in serialized
    assert "token_ids" not in serialized and "expected_sums" not in serialized


def test_coordinator_minimal_generation_api_and_extra_record_metadata():
    rows = tasks.build_roster()
    text = ",".join(map(str, rows[0]["expected_sums"]))
    result = tasks.score_generation(
        rows[0], {"status": "completed", "token_ids": [99], "stop_reason": "eos"}, text, [99]
    )
    assert result["label"] == 1 and set(result) == set(tasks.SCORE_KEYS)
    records = records_for(rows, [None] * 16)
    for record in records:
        record.update(status="unattempted", generation_status="unattempted")
    assert tasks.aggregate_records(rows, records)["overall"]["labels"]["unknown"] == 16
    with pytest.raises(ValueError, match="generation_fields"):
        tasks.score_generation(rows[0], {"status": "unattempted", "token_ids": []}, None, [99])


def test_supplied_qualification_known_failure_dominates_unknown_and_never_filters():
    rows = tasks.build_roster()
    labels = [None] * 16
    assert (
        tasks.aggregate_records(rows, records_for(rows, labels))["supplied_qualification"] is None
    )
    supplied_indices = [i for i, row in enumerate(rows) if row["condition"] == "supplied"]
    labels[supplied_indices[0]] = 0
    result = tasks.aggregate_records(rows, records_for(rows, labels))
    assert result["supplied_qualification"] is False
    contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
    assert contrast["resolved_outcomes"] == 1 and contrast["point"] is None
    assert (contrast["lower"], contrast["upper"]) == (-1, 7 / 8)
    for index in supplied_indices:
        labels[index] = 1
    result = tasks.aggregate_records(rows, records_for(rows, labels))
    assert result["supplied_qualification"] is True
    assert result["contrasts"]["success_rate"]["supplied_minus_computed"]["point"] is None
    assert result["paired_outcomes"]["counts"]["unknown"] == 8


@pytest.mark.parametrize("status", ["missing", "interrupted", "other"])
def test_only_selected_generation_states_accepted(status):
    with pytest.raises(ValueError, match="generation_status"):
        tasks.score_generation(tasks.build_roster()[0], generation([], status=status), None, [99])
