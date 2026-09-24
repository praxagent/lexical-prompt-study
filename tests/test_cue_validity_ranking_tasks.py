"""Invented A196 pure controls; no native tokenizer, model or evidence reads."""

import copy
import hashlib
import itertools
import math
import unittest

from lexical_prompt_study import cue_validity_ranking_tasks as m


def measured(true_margin=2.0, foil_margin=-1.0):
    values = []
    for slot in m.build_schedule():
        margin = true_margin if slot["condition"] == "true_cue" else foil_margin
        margin = margin(slot) if callable(margin) else margin
        values.append(
            None
            if margin is None
            else {
                "score": float(-100.0 + margin)
                if slot["candidate"] == "arithmetic_truth"
                else -100.0,
                "logits_sha256": hashlib.sha256(
                    f"invented|{slot['observation_id']}|{slot['candidate']}".encode()
                ).hexdigest(),
            }
        )
    return values


class ConstructorTests(unittest.TestCase):
    def test_fresh_full_factorial_and_identity(self):
        rows = m.build_roster()
        self.assertEqual(m.NAMESPACE, "a196-cue-validity-ranking-v1")
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {(r["core_index"], r["condition"]) for r in rows},
            set(itertools.product(range(8), ("true_cue", "foil_cue"))),
        )
        self.assertEqual([r["sequence_index"] for r in rows], list(range(16)))
        self.assertEqual(len({r["observation_id"] for r in rows}), 16)

    def test_independent_digest_modulus_and_operand_range(self):
        for core, item, side in itertools.product(range(8), range(2), range(2)):
            raw = hashlib.sha256(
                f"a196-cue-validity-ranking-v1|{core}|{item}|{side}".encode()
            ).digest()
            remainder = 0
            for byte in raw:
                remainder = (remainder * 256 + byte) % 90
            expected = (remainder + 10) * (-1 if raw[0] % 2 else 1)
            self.assertEqual(m.operand(core, item, side), expected)
            self.assertTrue(10 <= abs(expected) <= 99)

    def test_core_parity_foil_is_balanced_not_schedule_parity(self):
        self.assertEqual([m.foil_delta(i) for i in range(8)], [1, -1] * 4)
        for row in m.build_roster():
            a, b = row["expected_sums"]
            delta = 1 if row["core_index"] % 2 == 0 else -1
            self.assertEqual(row["foil_delta"], delta)
            self.assertEqual(
                row["candidates"],
                {"arithmetic_truth": f"{a},{b}", "arithmetic_foil": f"{a + delta},{b}"},
            )
            self.assertNotEqual(*row["candidates"].values())

    def test_identical_instruction_and_only_first_answer_field_changes(self):
        rows = m.build_roster()
        for one, two in zip(rows[::2], rows[1::2], strict=True):
            self.assertEqual(one["operands"], two["operands"])
            self.assertEqual(one["expected_sums"], two["expected_sums"])
            self.assertEqual(one["candidates"], two["candidates"])
            self.assertEqual(one["messages"][0], two["messages"][0])
            fields = []
            for row in (one, two):
                tokens = [line.split(" ") for line in row["messages"][1]["content"].split("\n")]
                self.assertEqual([[int(a), int(b)] for a, b, _ in tokens], row["operands"])
                self.assertEqual([int(answer) for _, _, answer in tokens], row["supplied_answers"])
                self.assertNotIn("?", row["messages"][1]["content"])
                expected = row["expected_sums"].copy()
                if row["condition"] == "foil_cue":
                    expected[0] += row["foil_delta"]
                self.assertEqual(row["supplied_answers"], expected)
                self.assertEqual(
                    row["instructed_candidate"],
                    "arithmetic_truth" if row["condition"] == "true_cue" else "arithmetic_foil",
                )
                fields.append(tokens)
            self.assertEqual(fields[0][1], fields[1][1])
            self.assertEqual(fields[0][0][:2], fields[1][0][:2])
            self.assertNotEqual(fields[0][0][2], fields[1][0][2])

    def test_hash_core_order_balanced_cue_and_candidate_orders(self):
        rows = m.build_roster()
        order = sorted(
            (hashlib.sha256(f"{m.NAMESPACE}|schedule|{core}".encode()).hexdigest(), core)
            for core in range(8)
        )
        self.assertEqual([(r["core_schedule_sha256"], r["core_index"]) for r in rows[::2]], order)
        self.assertEqual([r["condition"] for r in rows[::2]], ["true_cue", "foil_cue"] * 4)
        slots = m.build_schedule()
        self.assertEqual(len(slots), 64)
        self.assertEqual([s["evaluation_index"] for s in slots], list(range(64)))
        self.assertEqual(len({s["evaluation_id"] for s in slots}), 64)
        for row in rows:
            index = row["sequence_index"]
            block = slots[4 * index : 4 * index + 4]
            expected = list(m.CANDIDATES if index % 2 == 0 else reversed(m.CANDIDATES))
            self.assertEqual([s["candidate"] for s in block], expected + expected[::-1])
            self.assertEqual([s["repeat_index"] for s in block], [0, 0, 1, 1])
            self.assertTrue(all(s["sequence_index"] == index for s in block))
        for condition in m.CONDITIONS:
            first = [s["candidate"] for s in slots[::4] if s["condition"] == condition]
            self.assertEqual(first.count("arithmetic_truth"), 4)

    def test_strata_and_limits_are_input_only(self):
        for row in m.build_roster():
            self.assertEqual(row["expected_sums"], [a + b for a, b in row["operands"]])
            self.assertTrue(all(type(x) is str for x in row["strata"].values()))
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
            self.assertTrue(all(t.isascii() and "<" not in t for t in row["candidates"].values()))
        self.assertEqual((m.MAX_PROMPT_TOKENS, m.MAX_CANDIDATE_TOKENS), (256, 16))
        self.assertEqual(m.EOT_TOKEN, "<|eot_id|>")

    def test_exact_validation_and_no_aliasing(self):
        for key, value in (("sequence_index", False), ("foil_delta", 0), ("extra", 1)):
            rows = m.build_roster()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                m.validate_roster(rows)
        rows = m.build_roster()
        rows[0]["supplied_answers"][0] += 500
        self.assertNotEqual(rows[0]["supplied_answers"], rows[1]["supplied_answers"])
        self.assertEqual(m.validate_row(m.build_roster()[3]), m.build_roster()[3])
        slots = m.build_schedule()
        slots[0]["repeat_index"] = False
        with self.assertRaises(ValueError):
            m.validate_schedule(slots)

    def test_invalid_indices(self):
        for index in (True, -1, 8, 0.0):
            with self.assertRaises(ValueError):
                m.foil_delta(index)
        for args in ((True, 0, 0), (8, 0, 0), (0, 2, 0), (0, 0, False)):
            with self.assertRaises(ValueError):
                m.operand(*args)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.schedule = m.build_roster(), m.build_schedule()

    def analyze(self, values):
        return m.analyze_measurements(self.rows, self.schedule, values)

    def primary(self, result):
        return result["analysis"]["contrasts"]["margin_nats"]["true_cue_minus_foil_cue"]

    def test_all_unknown_preserves_full_cohort_and_binary_bounds(self):
        result = self.analyze([None] * 64)
        self.assertEqual(
            self.primary(result),
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
        switch = result["analysis"]["paired_switch"]
        self.assertEqual(switch["counts"], {"true": 0, "false": 0, "unknown": 8})
        self.assertEqual(switch["rate"], {"point": None, "lower": 0.0, "upper": 1.0})
        self.assertIsNone(switch["qualification"])

    def test_complete_reciprocal_switch_and_instruction_compliance(self):
        result = self.analyze(measured())
        self.assertEqual(self.primary(result)["point"], 3.0)
        switch = result["analysis"]["paired_switch"]
        self.assertEqual(switch["counts"], {"true": 8, "false": 0, "unknown": 0})
        self.assertEqual(switch["rate"], {"point": 1.0, "lower": 1.0, "upper": 1.0})
        self.assertTrue(switch["qualification"])
        for row in result["records"]:
            self.assertTrue(row["instruction_compliant_preferred"])
            self.assertIs(row["arithmetic_truth_preferred"], row["condition"] == "true_cue")
            self.assertEqual(row["candidate_scores"]["arithmetic_foil"], -100.0)

    def test_positive_shift_is_not_reciprocal_switch(self):
        result = self.analyze(measured(3.0, 1.0))
        self.assertEqual(self.primary(result)["point"], 2.0)
        self.assertEqual(result["analysis"]["paired_switch"]["counts"]["false"], 8)
        self.assertFalse(result["analysis"]["paired_switch"]["qualification"])

    def test_zero_and_negative_primary_do_not_change_sign_orientation(self):
        for true, foil, expected in ((1.0, 1.0, 0.0), (1.0, 2.0, -1.0), (-2.0, 1.0, -3.0)):
            result = self.analyze(measured(true, foil))
            self.assertEqual(self.primary(result)["point"], expected)
            self.assertEqual(result["analysis"]["by_condition"]["true_cue"]["point"], true)
            self.assertEqual(result["analysis"]["by_condition"]["foil_cue"]["point"], foil)

    def test_equal_core_mean_includes_all_differences(self):
        result = self.analyze(
            measured(lambda s: s["core_index"] + 1, lambda s: -(s["core_index"] + 1))
        )
        self.assertEqual(self.primary(result)["point"], 9.0)
        self.assertEqual(self.primary(result)["resolved_pairs"], 8)

    def test_pair_switch_truth_table_with_known_failure_dominance(self):
        for true, foil in itertools.product((None, -1.0, 0.0, 1.0), repeat=2):
            result = self.analyze(measured(true, foil))
            expected = (
                False
                if (true is not None and true <= 0) or (foil is not None and foil >= 0)
                else True
                if true is not None and foil is not None
                else None
            )
            key = "true" if expected is True else "false" if expected is False else "unknown"
            switch = result["analysis"]["paired_switch"]
            self.assertEqual(switch["counts"][key], 8)
            self.assertIs(switch["qualification"], expected)
            self.assertEqual(switch["resolved_pairs"], 0 if expected is None else 8)
            self.assertIs(self.primary(result)["point"] is None, true is None or foil is None)

    def test_tiny_strict_signs_satisfy_switch_without_tolerance(self):
        values = measured(0.0, 0.0)
        for index, slot in enumerate(self.schedule):
            if slot["candidate"] == "arithmetic_truth":
                values[index]["score"] = math.nextafter(
                    -100.0, 0.0 if slot["condition"] == "true_cue" else -math.inf
                )
        self.assertTrue(self.analyze(values)["analysis"]["paired_switch"]["qualification"])

    def test_mixed_switch_counts_and_sharp_binary_missing_bounds(self):
        def true(slot):
            return 1.0 if slot["core_index"] < 2 else 0.0 if slot["core_index"] < 5 else None

        result = self.analyze(measured(true, -1.0))
        switch = result["analysis"]["paired_switch"]
        self.assertEqual(switch["counts"], {"true": 2, "false": 3, "unknown": 3})
        self.assertEqual(switch["rate"], {"point": None, "lower": 0.25, "upper": 0.625})
        self.assertFalse(switch["qualification"])
        possible = [(2 + sum(bits)) / 8 for bits in itertools.product((0, 1), repeat=3)]
        self.assertEqual(
            (min(possible), max(possible)), (switch["rate"]["lower"], switch["rate"]["upper"])
        )
        self.assertIsNone(self.primary(result)["point"])

    def test_each_single_missing_measurement_leaves_primary_unbounded(self):
        for index in range(64):
            values = measured()
            values[index] = None
            result = self.analyze(values)
            primary = self.primary(result)
            self.assertEqual((primary["resolved_contexts"], primary["resolved_pairs"]), (15, 7))
            self.assertIsNone(primary["point"])
            self.assertIsNone(primary["lower"])
            self.assertIsNone(primary["upper"])
            self.assertTrue(primary["unbounded"])
            self.assertEqual(result["analysis"]["paired_switch"]["counts"]["unknown"], 1)

    def test_all_local_missingness_patterns_preserve_other_contexts(self):
        for present in itertools.product((False, True), repeat=4):
            values = measured()
            for index, exists in enumerate(present):
                if not exists:
                    values[index] = None
            result = self.analyze(values)
            self.assertEqual(result["records"][0]["measurement_count"], sum(present))
            self.assertIs(result["records"][0]["numerical_valid"], True if all(present) else None)
            self.assertEqual(self.primary(result)["resolved_contexts"], 16 if all(present) else 15)

    def test_different_readout_even_same_score_invalidates_margin(self):
        values = measured()
        values[0]["logits_sha256"] = "f" * 64
        result = self.analyze(values)
        self.assertTrue(result["records"][0]["complete"])
        self.assertFalse(result["records"][0]["numerical_valid"])
        self.assertIsNone(result["records"][0]["margin_nats"])
        self.assertEqual(result["analysis"]["paired_switch"]["counts"]["unknown"], 1)

    def test_equal_readout_with_different_scalar_rejected(self):
        values = measured()
        values[0]["score"] = math.nextafter(values[0]["score"], -math.inf)
        with self.assertRaisesRegex(ValueError, "equal_readout_score_mismatch"):
            self.analyze(values)

    def test_known_repeat_failure_dominates_other_missing_repeat(self):
        values = measured()
        values[0]["logits_sha256"] = "f" * 64
        values[1] = None
        self.assertFalse(self.analyze(values)["records"][0]["numerical_valid"])

    def test_invalid_measurement_shape_types_values_and_hash(self):
        for value in (
            True,
            {},
            {"score": True, "logits_sha256": "a" * 64},
            {"score": 1, "logits_sha256": "a" * 64},
            {"score": math.nan, "logits_sha256": "a" * 64},
            {"score": math.inf, "logits_sha256": "a" * 64},
            {"score": math.nextafter(0.0, 1.0), "logits_sha256": "a" * 64},
            {"score": -1.0, "logits_sha256": "A" * 64},
            {"score": -1.0, "logits_sha256": "a" * 64, "extra": 1},
        ):
            values = measured()
            values[0] = value
            with self.assertRaises(ValueError):
                self.analyze(values)
        with self.assertRaises(ValueError):
            self.analyze([None] * 63)

    def test_shuffled_or_boolean_identity_rejected(self):
        slots = copy.deepcopy(self.schedule)
        slots[0]["evaluation_index"] = False
        with self.assertRaises(ValueError):
            m.analyze_measurements(self.rows, slots, measured())
        with self.assertRaises(ValueError):
            m.analyze_measurements(self.rows[::-1], self.schedule, measured())

    def test_aggregate_has_no_private_item_payloads_or_obsolete_gate(self):
        analysis = self.analyze(measured())["analysis"]
        self.assertNotIn("supplied_qualification", analysis)
        text = m._canonical(analysis).decode()
        for excluded in (
            "observation_id",
            "expected_sums",
            "messages",
            "logits_sha256",
            "evaluation_id",
            "supplied_answers",
        ):
            self.assertNotIn(excluded, text)


if __name__ == "__main__":
    unittest.main()
