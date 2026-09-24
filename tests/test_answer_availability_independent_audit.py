"""Invented stdlib controls only; no producer import or native evidence access."""

import copy
import hashlib
import itertools
from pathlib import Path
import tempfile
import unittest

from lexical_prompt_study import answer_availability_independent_audit as m


def scores_for(labels):
    return [
        dict(
            outcome_status="missing" if label is None else "known",
            label=label,
            terminal_eos=None if label is None else True,
            capped=None if label is None else False,
            censored=None if label is None else False,
        )
        for label in labels
    ]


class ConstructorAndOracleTests(unittest.TestCase):
    def test_exact_factorial_balanced_pair_order_and_shared_answer(self):
        rows = m.reconstruct_roster()
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {(r["core_index"], r["condition"]) for r in rows},
            set(itertools.product(range(8), ("computed", "supplied"))),
        )
        self.assertEqual([r["sequence_index"] for r in rows], list(range(16)))
        self.assertEqual([r["condition"] for r in rows[::2]], ["computed", "supplied"] * 4)
        order = sorted(
            (
                hashlib.sha256(f"a191-answer-availability-v1|schedule|{core}".encode()).hexdigest(),
                core,
            )
            for core in range(8)
        )
        self.assertEqual([(r["core_schedule_sha256"], r["core_index"]) for r in rows[::2]], order)
        for pair_index, (first, second) in enumerate(zip(rows[::2], rows[1::2])):
            self.assertEqual(first["operands"], second["operands"])
            self.assertEqual(first["expected_sums"], second["expected_sums"])
            self.assertEqual(first["strata"], second["strata"])
            self.assertEqual(first["messages"][0], second["messages"][0])
            self.assertEqual(first["pair_index"], pair_index)
            for row in (first, second):
                fields = [line.split() for line in row["messages"][1]["content"].split("\n")]
                self.assertEqual([[int(x[0]), int(x[1])] for x in fields], row["operands"])
                self.assertEqual(
                    [x[2] for x in fields],
                    ["?", "?"]
                    if row["condition"] == "computed"
                    else list(map(str, row["expected_sums"])),
                )

    def test_operand_and_strata_direct_arithmetic(self):
        for row in m.reconstruct_roster():
            for item, pair in enumerate(row["operands"]):
                for side, value in enumerate(pair):
                    raw = hashlib.sha256(
                        f"a191-answer-availability-v1|{row['core_index']}|{item}|{side}".encode()
                    ).digest()
                    self.assertEqual(
                        value, (10 + int.from_bytes(raw, "big") % 90) * (-1 if raw[0] & 1 else 1)
                    )
                    self.assertTrue(10 <= abs(value) <= 99)
            self.assertEqual(row["expected_sums"], [a + b for a, b in row["operands"]])
            self.assertEqual(
                row["strata"]["zero_sum_items"], str(sum(a + b == 0 for a, b in row["operands"]))
            )
            self.assertEqual(
                sum(
                    int(row["strata"][name])
                    for name in (
                        "positive_positive_items",
                        "negative_negative_items",
                        "opposite_sign_items",
                    )
                ),
                2,
            )

    def test_fresh_reconstruction_has_no_shared_mutable_values(self):
        rows = m.reconstruct_roster()
        snapshot = m.canonical(rows)
        second = copy.deepcopy(rows[1])
        rows[0]["operands"][0][0] = 0
        self.assertEqual(rows[1], second)
        self.assertEqual(m.canonical(m.reconstruct_roster()), snapshot)

    def test_ascii_integer_normalization_and_large_fields(self):
        self.assertTrue(m.whole_content("\t+0003 , -0004\r\n", [3, -4]))
        self.assertTrue(m.whole_content("-000,+00", [0, 0]))
        self.assertTrue(m.whole_content("0" * 6000 + "3,-4", [3, -4]))
        self.assertFalse(m.whole_content("9" * 6000 + ",-4", [3, -4]))

    def test_no_wrapper_recovery_or_prose_extraction(self):
        for text in (
            "[3,-4]",
            "```\n3,-4\n```",
            "Answers:3,-4",
            "3",
            "3,-4,",
            "3,-4,0",
            "-4,3",
            "3.0,-4",
            "３,-4",
            "3,−4",
            "3,-4\u00a0",
            "3,,",
        ):
            with self.subTest(text=text):
                self.assertFalse(m.whole_content(text, [3, -4]))

    def test_eos_success_and_cap_failure(self):
        eos = m.rebuild_score([3, -4], "completed", [7, 99], "3,-4", [99])
        cap = m.rebuild_score([3, -4], "completed", [7] * 64, "3,-4", [99])
        self.assertEqual(
            eos,
            dict(outcome_status="known", label=1, terminal_eos=True, capped=False, censored=False),
        )
        self.assertEqual(
            cap,
            dict(outcome_status="known", label=0, terminal_eos=False, capped=True, censored=False),
        )
        self.assertEqual(
            m.rebuild_score([3, -4], "completed", [7] * 63 + [99], "3,-4", [99])["label"], 1
        )

    def test_partial_eos_and_partial_cap_never_known(self):
        for ids in ([], [7], [7, 99], [7] * 64):
            with self.subTest(length=len(ids)):
                value = m.rebuild_score([3, -4], "infrastructure_failed", ids, None, [99])
                self.assertIsNone(value["label"])
                self.assertFalse(value["capped"])
                self.assertEqual(value["terminal_eos"], bool(ids and ids[-1] == 99))
                self.assertEqual(value["censored"], 99 not in ids and len(ids) < 64)

    def test_unattempted_stopping_facts_unknown(self):
        self.assertEqual(
            m.rebuild_score([3, -4], "unattempted", [], None, [99]),
            {key: "missing" if key == "outcome_status" else None for key in m.FIELDS},
        )

    def test_invalid_tokens_and_termination_rejected(self):
        for ids in ([99, 7, 99], [7], [7] * 65, [True], [-1]):
            with self.assertRaises(AssertionError):
                m.rebuild_score([3, -4], "completed", ids, "3,-4", [99])
        for eos in ([], [True], [99, 99]):
            with self.assertRaises(AssertionError):
                m.rebuild_score([3, -4], "unattempted", [], None, eos)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = m.reconstruct_roster()

    def analyze(self, labels):
        return m.reconstruct_analysis(self.rows, scores_for(labels))

    def test_all_unknown_full_cohort_bounds(self):
        result = self.analyze([None] * 16)
        self.assertIsNone(result["supplied_qualification"])
        self.assertEqual(
            result["overall"]["success_rate"],
            dict(point=None, lower=0.0, upper=1.0, planned_outcomes=16, resolved_outcomes=0),
        )
        self.assertEqual(
            result["contrasts"]["success_rate"]["supplied_minus_computed"],
            dict(
                point=None,
                lower=-1.0,
                upper=1.0,
                planned_outcomes=16,
                resolved_outcomes=0,
                planned_pairs=8,
                resolved_pairs=0,
            ),
        )
        self.assertEqual(result["paired_outcomes"]["counts"]["unknown"], 8)

    def test_all_four_resolved_pair_categories(self):
        pairs = [(1, 1), (0, 0), (1, 0), (0, 1)] * 2
        labels = [
            pairs[r["pair_index"]][0 if r["condition"] == "supplied" else 1] for r in self.rows
        ]
        result = self.analyze(labels)
        self.assertEqual(
            result["paired_outcomes"]["counts"],
            dict(both_success=2, both_failure=2, supplied_only=2, computed_only=2, unknown=0),
        )
        self.assertEqual(result["contrasts"]["success_rate"]["supplied_minus_computed"]["point"], 0)
        self.assertEqual(result["overall"]["success_rate"]["point"], 0.5)
        self.assertFalse(result["supplied_qualification"])

    def test_contrast_direction_and_gate_independence(self):
        for success_arm, direction in (("supplied", 1), ("computed", -1)):
            labels = [int(r["condition"] == success_arm) for r in self.rows]
            result = self.analyze(labels)
            self.assertEqual(
                result["contrasts"]["success_rate"]["supplied_minus_computed"]["point"], direction
            )
            self.assertEqual(result["supplied_qualification"], success_arm == "supplied")

    def test_supplied_known_failure_dominates_missing(self):
        labels = [None] * 16
        supplied = [i for i, r in enumerate(self.rows) if r["condition"] == "supplied"]
        labels[supplied[0]] = 0
        result = self.analyze(labels)
        self.assertFalse(result["supplied_qualification"])
        contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
        self.assertEqual(
            (contrast["point"], contrast["lower"], contrast["upper"]), (None, -1, 7 / 8)
        )
        for index in supplied:
            labels[index] = 1
        result = self.analyze(labels)
        self.assertTrue(result["supplied_qualification"])
        self.assertIsNone(result["contrasts"]["success_rate"]["supplied_minus_computed"]["point"])
        self.assertEqual(result["paired_outcomes"]["counts"]["unknown"], 8)

    def test_observed_supplied_successes_do_not_qualify_unknown_supplied_rows(self):
        labels = [1 if row["condition"] == "supplied" else 0 for row in self.rows]
        missing_index = next(i for i, row in enumerate(self.rows) if row["condition"] == "supplied")
        labels[missing_index] = None
        result = self.analyze(labels)
        self.assertIsNone(result["supplied_qualification"])
        contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
        self.assertEqual(
            (contrast["point"], contrast["lower"], contrast["upper"]), (None, 7 / 8, 1)
        )

    def test_exhaustive_missing_bounds_and_fixed_denominators(self):
        for first in itertools.product((None, 0, 1), repeat=4):
            labels = list(first) + [0] * 12
            result = self.analyze(labels)
            absent = [i for i, x in enumerate(labels) if x is None]
            points = []
            for filling in itertools.product((0, 1), repeat=len(absent)):
                full = list(labels)
                for index, value in zip(absent, filling):
                    full[index] = value
                points.append(
                    sum(
                        (1 if row["condition"] == "supplied" else -1) * value
                        for row, value in zip(self.rows, full)
                    )
                    / 8
                )
            contrast = result["contrasts"]["success_rate"]["supplied_minus_computed"]
            self.assertEqual((contrast["lower"], contrast["upper"]), (min(points), max(points)))
            self.assertEqual(contrast["point"], None if absent else points[0])
            self.assertEqual(contrast["planned_outcomes"], 16)
            self.assertEqual(
                result["by_condition"]["supplied"]["success_rate"]["planned_outcomes"], 8
            )
            unknown_pairs = sum(any(labels[2 * p + i] is None for i in range(2)) for p in range(8))
            self.assertEqual(result["paired_outcomes"]["counts"]["unknown"], unknown_pairs)

    def test_boolean_label_and_roster_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            self.analyze([True] + [None] * 15)
        self.rows[0]["condition"] = "supplied"
        with self.assertRaises(AssertionError):
            self.analyze([None] * 16)


class InventedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a191-independent-invented-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = m.reconstruct_roster()
        self.plan = {"rows": self.rows, "eos_token_ids": [99]}
        self.header = {"synthetic_fixture": True}
        self.put(self.root / "run.json", self.header)
        self.records = [
            {
                **{key: row[key] for key in m.IDENTITY},
                "status": "unattempted",
                "generation_status": "missing",
                **m.rebuild_score(row["expected_sums"], "unattempted", [], None, [99]),
            }
            for row in self.rows
        ]

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(m.canonical(value))

    def install(self, index, *, mode="correct", returned=True):
        row = self.rows[index]
        base = self.root / "rows" / f"{index:03d}"
        tokens = [7, 99] if mode != "cap" else [7] * 64
        text = (
            ",".join(map(str, row["expected_sums"]))
            if mode != "wrong"
            else "invented wrong response"
        )
        step_hashes = []
        for step, token in enumerate(tokens):
            result = {"chosen_token_id": token, "invented": True}
            self.put(base / f"step_{step:02d}" / "result.json", result)
            step_hashes.append(m.digest(result))
        score = m.rebuild_score(row["expected_sums"], "completed", tokens, text, [99])
        raw = text.encode()
        receipt = {
            "schema_version": m.ACQUISITION_SCHEMA,
            "run_sha256": m.digest(self.header),
            "row_sha256": m.digest(row),
            "generation": {
                "status": "completed",
                "stop_reason": "cap" if mode == "cap" else "eos",
                "token_ids": tokens,
                "observed_tokens": len(tokens),
            },
            "step_result_sha256": step_hashes,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "score": score,
        }
        self.put(base / "result.json", receipt)
        (base / "response.utf8").write_bytes(raw)
        if returned:
            self.put(
                base / "return.json",
                {"event": "generation_returned_normally", "result_sha256": m.digest(receipt)},
            )
            self.records[index].update(status="completed", generation_status="completed", **score)
        else:
            self.records[index].update(
                status="failed",
                generation_status="infrastructure_failed",
                **m.rebuild_score(
                    row["expected_sums"], "infrastructure_failed", tokens, None, [99]
                ),
            )
        return base

    def audit(self):
        return m.audit_outcomes(self.plan, self.records, self.root)

    def test_all_unattempted_retained(self):
        result = self.audit()
        self.assertEqual(result["record_slots"], 16)
        self.assertEqual(result["independent_response_oracle_evaluations"], 0)
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)

    def test_full16_mixed_eos_fixture_and_aggregate_privacy(self):
        for i, row in enumerate(self.rows):
            self.install(i, mode="correct" if row["condition"] == "supplied" else "wrong")
        result = self.audit()
        self.assertEqual(
            set(result),
            {
                "independent_response_oracle_evaluations",
                "independently_correct_whole_contents",
                "record_slots",
                "analysis",
            },
        )
        self.assertEqual(result["independent_response_oracle_evaluations"], 16)
        self.assertEqual(result["independently_correct_whole_contents"], 8)
        self.assertEqual(
            result["analysis"]["contrasts"]["success_rate"]["supplied_minus_computed"]["point"], 1
        )
        self.assertEqual(result["analysis"]["paired_outcomes"]["counts"]["supplied_only"], 8)
        for forbidden in ("observation_id", "token_ids", "response.utf8", "core_index"):
            self.assertNotIn(forbidden, repr(result))

    def test_correct_content_under_cap_stays_failure(self):
        self.install(0, mode="cap")
        result = self.audit()
        self.assertEqual(result["independently_correct_whole_contents"], 1)
        self.assertEqual(
            result["analysis"]["overall"]["labels"], {"success": 0, "failure": 1, "unknown": 15}
        )

    def test_orphan_eos_response_not_read_or_promoted(self):
        base = self.install(0, returned=False)
        (base / "response.utf8").write_bytes(b"\xff")
        result = self.audit()
        self.assertEqual(result["independent_response_oracle_evaluations"], 0)
        self.assertEqual(result["analysis"]["overall"]["terminal_eos"], 1)
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)

    def test_failed_final_dispatch_result_is_not_usable(self):
        base = self.install(0, returned=False)
        self.put(base / "step_01" / "failure.json", {"usable": False})
        self.records[0].update(
            m.rebuild_score(self.rows[0]["expected_sums"], "infrastructure_failed", [7], None, [99])
        )
        result = self.audit()
        self.assertEqual(result["analysis"]["overall"]["terminal_eos"], 0)
        self.assertEqual(result["analysis"]["overall"]["censored"]["true"], 1)

    def test_full_response_hash_and_return_hash_tampering_rejected(self):
        base = self.install(0)
        raw = (base / "response.utf8").read_bytes()
        (base / "response.utf8").write_bytes(b"0,0")
        with self.assertRaises(AssertionError):
            self.audit()
        (base / "response.utf8").write_bytes(raw)
        self.put(
            base / "return.json",
            {"event": "generation_returned_normally", "result_sha256": "0" * 64},
        )
        with self.assertRaises(AssertionError):
            self.audit()

    def test_completed_without_return_rejected(self):
        base = self.install(0)
        (base / "return.json").unlink()
        with self.assertRaises(AssertionError):
            self.audit()

    def test_record_identity_and_score_type_tamper_rejected(self):
        original = copy.deepcopy(self.records[0])
        self.records[0]["core_index"] = float(self.records[0]["core_index"])
        with self.assertRaises(AssertionError):
            self.audit()
        self.records[0] = original
        self.install(0)
        self.records[0]["label"] = True
        with self.assertRaises(AssertionError):
            self.audit()

    def test_constructor_does_not_trust_supplied_plan_answers(self):
        self.rows[0]["operands"][0][0] += 1
        self.rows[0]["expected_sums"][0] += 1
        with self.assertRaises(AssertionError):
            self.audit()

    def test_step_chain_hashes_and_order_rejected(self):
        base = self.install(0)
        step = base / "step_00" / "result.json"
        self.put(step, {"chosen_token_id": 8, "invented": True})
        with self.assertRaises(AssertionError):
            self.audit()
        (base / "step_00").rename(base / "step_02")
        with self.assertRaises(AssertionError):
            self.audit()

    def test_json_duplicate_and_nonfinite_rejected(self):
        path = self.root / "invented.json"
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            path.write_bytes(raw)
            with self.assertRaises((AssertionError, ValueError)):
                m.read_json(path)

    def test_typed_canonical_equality(self):
        for a, b in ((True, 1), (False, 0), (1, 1.0)):
            with self.assertRaises(AssertionError):
                m.same(a, b)


if __name__ == "__main__":
    unittest.main()
