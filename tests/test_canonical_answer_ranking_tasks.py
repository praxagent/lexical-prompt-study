"""Invented stdlib A195 controls; no native tokenizer, model or study evidence."""

import copy
import hashlib
import itertools
import math
import unittest

from lexical_prompt_study import canonical_answer_ranking_tasks as m


def measured(correct=-2.0, foil=-3.0):
    schedule = m.build_schedule()
    return [
        {
            "score": float(correct if slot["candidate"] == "correct" else foil),
            "logits_sha256": hashlib.sha256(
                f"invented|{slot['observation_id']}|{slot['candidate']}".encode()
            ).hexdigest(),
        }
        for slot in schedule
    ]


class ConstructorTests(unittest.TestCase):
    def test_fixed_fresh_namespace_and_factorial(self):
        rows = m.build_roster()
        self.assertEqual(m.NAMESPACE, "a195-canonical-answer-ranking-v1")
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {(r["core_index"], r["condition"]) for r in rows},
            set(itertools.product(range(8), ("computed", "supplied"))),
        )
        self.assertEqual([r["sequence_index"] for r in rows], list(range(16)))

    def test_independent_hash_arithmetic(self):
        for core, item, side in itertools.product(range(8), range(2), range(2)):
            raw = hashlib.sha256(
                f"a195-canonical-answer-ranking-v1|{core}|{item}|{side}".encode()
            ).digest()
            remainder = 0
            for byte in raw:
                remainder = (remainder * 256 + byte) % 90
            expected = (remainder + 10) * (-1 if raw[0] % 2 else 1)
            self.assertEqual(m.operand(core, item, side), expected)
            self.assertTrue(10 <= abs(expected) <= 99)

    def test_schedule_hash_order_and_balanced_arm_order(self):
        rows = m.build_roster()
        expected = sorted(
            (hashlib.sha256(f"{m.NAMESPACE}|schedule|{core}".encode()).hexdigest(), core)
            for core in range(8)
        )
        self.assertEqual(
            [(r["core_schedule_sha256"], r["core_index"]) for r in rows[::2]], expected
        )
        self.assertEqual([r["condition"] for r in rows[::2]], ["computed", "supplied"] * 4)

    def test_shared_answers_and_only_answer_column_changes(self):
        for first, second in zip(m.build_roster()[::2], m.build_roster()[1::2], strict=True):
            self.assertEqual(first["operands"], second["operands"])
            self.assertEqual(first["expected_sums"], second["expected_sums"])
            self.assertEqual(first["messages"][0], second["messages"][0])
            self.assertEqual(first["candidates"], second["candidates"])
            for row in (first, second):
                fields = [line.split(" ") for line in row["messages"][1]["content"].split("\n")]
                self.assertEqual([[int(a), int(b)] for a, b, _ in fields], row["operands"])
                expected = (
                    ["?", "?"]
                    if row["condition"] == "computed"
                    else list(map(str, row["expected_sums"]))
                )
                self.assertEqual([answer for _, _, answer in fields], expected)

    def test_exact_foil_and_canonical_text_no_embedded_eot(self):
        for row in m.build_roster():
            first, second = row["expected_sums"]
            self.assertEqual(
                row["candidates"],
                {"correct": f"{first},{second}", "first_sum_plus_one": f"{first + 1},{second}"},
            )
            self.assertNotEqual(*row["candidates"].values())
            self.assertTrue(
                all(text.isascii() and "<" not in text for text in row["candidates"].values())
            )
        self.assertEqual((m.MAX_PROMPT_TOKENS, m.MAX_CANDIDATE_TOKENS), (256, 16))
        self.assertEqual(m.EOT_TOKEN, "<|eot_id|>")

    def test_input_strata_consistent(self):
        for row in m.build_roster():
            pairs = row["operands"]
            self.assertEqual(row["expected_sums"], [a + b for a, b in pairs])
            strata = row["strata"]
            self.assertTrue(all(type(value) is str for value in strata.values()))
            self.assertEqual(
                sum(
                    int(strata[key])
                    for key in (
                        "positive_positive_items",
                        "negative_negative_items",
                        "opposite_sign_items",
                    )
                ),
                2,
            )
            self.assertEqual(int(strata["zero_sum_items"]), sum(a + b == 0 for a, b in pairs))

    def test_exact_64_slots_and_repeat_occurrences(self):
        slots = m.build_schedule()
        self.assertEqual(len(slots), 64)
        self.assertEqual([s["evaluation_index"] for s in slots], list(range(64)))
        self.assertEqual(len({s["evaluation_id"] for s in slots}), 64)
        for index in range(16):
            block = slots[index * 4 : index * 4 + 4]
            order = list(m.CANDIDATES if index % 2 == 0 else reversed(m.CANDIDATES))
            self.assertEqual([s["candidate"] for s in block], order + order[::-1])
            self.assertEqual([s["repeat_index"] for s in block], [0, 0, 1, 1])
            self.assertTrue(all(s["sequence_index"] == index for s in block))
        for condition in m.CONDITIONS:
            starts = [s["candidate"] for s in slots[::4] if s["condition"] == condition]
            self.assertEqual(starts.count("correct"), 4)

    def test_exact_validation_rejects_mutation_extra_fields_and_boolean(self):
        for key, value in (("sequence_index", False), ("expected_sums", [0, 0]), ("extra", 1)):
            rows = m.build_roster()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                m.validate_roster(rows)
        slots = m.build_schedule()
        slots[0]["repeat_index"] = False
        with self.assertRaises(ValueError):
            m.validate_schedule(slots)

    def test_fresh_values_no_aliasing(self):
        rows = m.build_roster()
        rows[0]["candidates"]["correct"] = "mutated"
        self.assertNotEqual(rows[1]["candidates"]["correct"], "mutated")
        self.assertNotEqual(m.build_roster()[0]["candidates"]["correct"], "mutated")
        self.assertEqual(m.validate_row(m.build_roster()[3]), m.build_roster()[3])

    def test_invalid_constructor_indices(self):
        for args in ((True, 0, 0), (8, 0, 0), (0, 2, 0), (0, 0, False), (0, 0, 2)):
            with self.assertRaises(ValueError):
                m.operand(*args)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = m.build_roster()
        self.schedule = m.build_schedule()

    def analyze(self, values):
        return m.analyze_measurements(self.rows, self.schedule, values)

    def test_all_unknown_has_unbounded_fixed_cohort(self):
        result = self.analyze([None] * 64)
        a = result["analysis"]
        self.assertIsNone(a["supplied_qualification"])
        p = a["contrasts"]["margin_nats"]["supplied_minus_computed"]
        self.assertEqual(
            p,
            dict(
                planned_contexts=16,
                resolved_contexts=0,
                planned_pairs=8,
                resolved_pairs=0,
                point=None,
                lower=None,
                upper=None,
                unbounded=True,
            ),
        )
        self.assertEqual(len(result["records"]), 16)

    def test_first_copy_scores_and_exact_finite_zero_contrast(self):
        result = self.analyze(measured())
        for r in result["records"]:
            self.assertEqual(r["candidate_scores"], {"correct": -2.0, "first_sum_plus_one": -3.0})
            self.assertEqual(r["margin_nats"], 1.0)
            self.assertIs(r["correct_preferred"], True)
        a = result["analysis"]
        self.assertTrue(a["supplied_qualification"])
        self.assertEqual(a["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"], 0.0)

    def test_equal_core_direction_and_no_length_normalization(self):
        values = measured()
        expected = []
        for row in self.rows:
            margin = (row["core_index"] + 1) * (2 if row["condition"] == "supplied" else 1)
            for i, slot in enumerate(self.schedule):
                if slot["sequence_index"] == row["sequence_index"]:
                    values[i]["score"] = -20.0 + margin if slot["candidate"] == "correct" else -20.0
            if row["condition"] == "supplied":
                expected.append(row["core_index"] + 1)
        a = self.analyze(values)["analysis"]
        self.assertEqual(
            a["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"], sum(expected) / 8
        )

    def test_negative_availability_shift_can_coexist_with_positive_supplied_control(self):
        values = measured()
        for index, slot in enumerate(self.schedule):
            if slot["condition"] == "computed" and slot["candidate"] == "correct":
                values[index]["score"] = -1.0
        analysis = self.analyze(values)["analysis"]
        self.assertEqual(analysis["by_condition"]["computed"]["point"], 2.0)
        self.assertEqual(analysis["by_condition"]["supplied"]["point"], 1.0)
        self.assertEqual(
            analysis["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"], -1.0
        )
        self.assertIs(analysis["supplied_qualification"], True)

    def test_exact_tie_fails_and_positive_tiny_margin_passes(self):
        for correct, expected, sign in (
            (-3.0, False, "zero"),
            (-4.0, False, "negative"),
            (math.nextafter(-3.0, 0.0), True, "positive"),
        ):
            result = self.analyze(measured(correct=correct))
            self.assertIs(result["analysis"]["supplied_qualification"], expected)
            self.assertTrue(all(r["margin_sign"] == sign for r in result["records"]))

    def test_known_supplied_failure_dominates_unknown(self):
        values = [None] * 64
        row = next(row for row in self.rows if row["condition"] == "supplied")
        full = measured(correct=-4.0)
        values[row["sequence_index"] * 4 : row["sequence_index"] * 4 + 4] = full[
            row["sequence_index"] * 4 : row["sequence_index"] * 4 + 4
        ]
        self.assertFalse(self.analyze(values)["analysis"]["supplied_qualification"])

    def test_known_supplied_positive_does_not_fill_missing(self):
        values = measured()
        missing = next(i for i, s in enumerate(self.schedule) if s["condition"] == "supplied")
        values[missing] = None
        a = self.analyze(values)["analysis"]
        self.assertIsNone(a["supplied_qualification"])
        self.assertIsNone(a["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"])
        self.assertEqual(a["by_condition"]["computed"]["point"], 1.0)

    def test_computed_missing_does_not_erase_supplied_qualification(self):
        values = measured()
        for i, s in enumerate(self.schedule):
            if s["condition"] == "computed":
                values[i] = None
        a = self.analyze(values)["analysis"]
        self.assertTrue(a["supplied_qualification"])
        self.assertIsNone(a["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"])
        self.assertEqual(a["by_condition"]["supplied"]["resolved_contexts"], 8)

    def test_different_raw_hash_fails_even_when_scores_equal(self):
        values = measured()
        values[0]["logits_sha256"] = "0" * 64
        result = self.analyze(values)
        record = result["records"][0]
        self.assertTrue(record["complete"])
        self.assertFalse(record["numerical_valid"])
        self.assertIsNone(record["margin_nats"])
        self.assertIsNone(
            result["analysis"]["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"]
        )

    def test_same_raw_hash_with_different_scalar_rejected(self):
        for score in (-2.1, math.nextafter(-2.0, -math.inf)):
            values = measured()
            values[0]["score"] = score
            with self.assertRaisesRegex(ValueError, "equal_readout_score_mismatch"):
                self.analyze(values)

    def test_exhaustive_local_missingness_retains_other_fifteen(self):
        for observed in itertools.product((False, True), repeat=4):
            values = measured()
            for i, present in enumerate(observed):
                if not present:
                    values[i] = None
            result = self.analyze(values)
            self.assertEqual(result["records"][0]["measurement_count"], sum(observed))
            self.assertEqual(
                result["records"][0]["numerical_valid"], True if all(observed) else None
            )
            p = result["analysis"]["contrasts"]["margin_nats"]["supplied_minus_computed"]
            self.assertEqual(p["resolved_contexts"], 16 if all(observed) else 15)
            self.assertEqual(p["resolved_pairs"], 8 if all(observed) else 7)
            self.assertEqual(p["point"], 0.0 if all(observed) else None)
            self.assertEqual((p["lower"], p["upper"]), (p["point"], p["point"]))

    def test_known_repeat_failure_dominates_local_missing_repeat(self):
        values = measured()
        values[0]["logits_sha256"] = "0" * 64
        values[1] = None
        r = self.analyze(values)["records"][0]
        self.assertFalse(r["numerical_valid"])
        self.assertIsNone(r["margin_nats"])

    def test_supplied_repeat_failure_alone_leaves_qualification_unknown(self):
        values = measured()
        i = next(i for i, s in enumerate(self.schedule) if s["condition"] == "supplied")
        values[i]["logits_sha256"] = "0" * 64
        self.assertIsNone(self.analyze(values)["analysis"]["supplied_qualification"])

    def test_invalid_measurement_shapes_and_values(self):
        invalid = [
            True,
            {},
            {"score": True, "logits_sha256": "a" * 64},
            {"score": 1, "logits_sha256": "a" * 64},
            {"score": float("nan"), "logits_sha256": "a" * 64},
            {"score": float("inf"), "logits_sha256": "a" * 64},
            {"score": math.nextafter(0.0, 1.0), "logits_sha256": "a" * 64},
            {"score": -1.0, "logits_sha256": "A" * 64},
            {"score": -1.0, "logits_sha256": "a" * 64, "label": 1},
        ]
        for value in invalid:
            values = measured()
            values[0] = value
            with self.assertRaises(ValueError):
                self.analyze(values)
        with self.assertRaises(ValueError):
            self.analyze([None] * 63)

    def test_boolean_identity_and_shuffled_schedule_rejected(self):
        schedule = copy.deepcopy(self.schedule)
        schedule[0]["evaluation_index"] = False
        with self.assertRaises(ValueError):
            m.analyze_measurements(self.rows, schedule, measured())
        with self.assertRaises(ValueError):
            m.analyze_measurements(self.rows[::-1], self.schedule, measured())

    def test_analysis_contains_only_aggregate_counts_and_scores(self):
        text = m._canonical(self.analyze(measured())["analysis"]).decode()
        for excluded in (
            "observation_id",
            "expected_sums",
            "messages",
            "logits_sha256",
            "evaluation_id",
        ):
            self.assertNotIn(excluded, text)


if __name__ == "__main__":
    unittest.main()
