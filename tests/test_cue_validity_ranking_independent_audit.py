"""Invented A196 audit controls; no producer imports, file or native evidence reads."""

import copy
import hashlib
import itertools
import math
import unittest

from lexical_prompt_study import cue_validity_ranking_independent_audit as a


def plan():
    return {"rows": a.reconstruct_roster(), "schedule": a.reconstruct_schedule()}


def measurements(true=2.0, foil=-1.0):
    result = []
    for slot in a.reconstruct_schedule():
        margin = true if slot["condition"] == "true_cue" else foil
        margin = margin(slot) if callable(margin) else margin
        result.append(
            None
            if margin is None
            else {
                "score": float(-100 + margin)
                if slot["candidate"] == "arithmetic_truth"
                else -100.0,
                "logits_sha256": hashlib.sha256(
                    f"invented|{slot['observation_id']}|{slot['candidate']}".encode()
                ).hexdigest(),
            }
        )
    return result


def records(values):
    return [
        {
            "evaluation_id": slot["evaluation_id"],
            "evaluation_index": slot["evaluation_index"],
            "status": "unattempted" if value is None else "completed",
            "measurement": value,
        }
        for slot, value in zip(a.reconstruct_schedule(), values, strict=True)
    ]


class ConstructionTests(unittest.TestCase):
    def test_exact_namespace_factorial_and_hash_order(self):
        rows = a.reconstruct_roster()
        self.assertEqual(a.NAMESPACE, "a196-cue-validity-ranking-v1")
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {(r["core_index"], r["condition"]) for r in rows},
            set(itertools.product(range(8), ("true_cue", "foil_cue"))),
        )
        keys = sorted(
            (hashlib.sha256(f"a196-cue-validity-ranking-v1|schedule|{i}".encode()).hexdigest(), i)
            for i in range(8)
        )
        self.assertEqual([(r["core_schedule_sha256"], r["core_index"]) for r in rows[::2]], keys)
        self.assertEqual([r["condition"] for r in rows[::2]], ["true_cue", "foil_cue"] * 4)

    def test_independent_whole_digest_operand_check(self):
        for row in a.reconstruct_roster():
            for item, pair in enumerate(row["operands"]):
                for side, value in enumerate(pair):
                    raw = hashlib.sha256(
                        f"a196-cue-validity-ranking-v1|{row['core_index']}|{item}|{side}".encode()
                    ).digest()
                    self.assertEqual(
                        value, (10 + int.from_bytes(raw, "big") % 90) * (-1 if raw[0] & 1 else 1)
                    )
            self.assertEqual(row["expected_sums"], [sum(pair) for pair in row["operands"]])

    def test_core_parity_foil_and_explicit_instruction_target(self):
        deltas = {}
        for row in a.reconstruct_roster():
            x, y = row["expected_sums"]
            delta = 1 if row["core_index"] % 2 == 0 else -1
            deltas[row["core_index"]] = row["foil_delta"]
            self.assertEqual(row["foil_delta"], delta)
            self.assertEqual(
                row["candidates"],
                {"arithmetic_truth": f"{x},{y}", "arithmetic_foil": f"{x + delta},{y}"},
            )
            self.assertEqual(
                row["supplied_answers"],
                [x, y] if row["condition"] == "true_cue" else [x + delta, y],
            )
            self.assertEqual(
                row["instructed_candidate"],
                "arithmetic_truth" if row["condition"] == "true_cue" else "arithmetic_foil",
            )
        self.assertEqual([deltas[i] for i in range(8)], [1, -1] * 4)

    def test_paired_payload_only_first_answer_differs(self):
        rows = a.reconstruct_roster()
        for one, two in zip(rows[::2], rows[1::2], strict=True):
            for key in ("operands", "expected_sums", "candidates", "strata", "foil_delta"):
                self.assertEqual(one[key], two[key])
            self.assertEqual(one["messages"][0], two["messages"][0])
            field_rows = []
            for row in (one, two):
                self.assertNotIn("?", row["messages"][1]["content"])
                fields = [
                    [int(x) for x in line.split(" ")]
                    for line in row["messages"][1]["content"].split("\n")
                ]
                self.assertEqual([f[:2] for f in fields], row["operands"])
                self.assertEqual([f[2] for f in fields], row["supplied_answers"])
                field_rows.append(fields)
            self.assertEqual(field_rows[0][1], field_rows[1][1])
            self.assertEqual(field_rows[0][0][:2], field_rows[1][0][:2])
            self.assertNotEqual(field_rows[0][0][2], field_rows[1][0][2])

    def test_exact_schedule_and_repeat_ids(self):
        schedule = a.reconstruct_schedule()
        self.assertEqual(len(schedule), 64)
        self.assertEqual(len({s["evaluation_id"] for s in schedule}), 64)
        self.assertEqual([s["evaluation_index"] for s in schedule], list(range(64)))
        for index in range(16):
            first = "arithmetic_truth" if index % 2 == 0 else "arithmetic_foil"
            other = "arithmetic_foil" if first == "arithmetic_truth" else "arithmetic_truth"
            block = schedule[index * 4 : index * 4 + 4]
            self.assertEqual(
                [(s["candidate"], s["repeat_index"]) for s in block],
                [(first, 0), (other, 0), (other, 1), (first, 1)],
            )
        for condition in ("true_cue", "foil_cue"):
            self.assertEqual(
                sum(
                    s["candidate"] == "arithmetic_truth" and s["condition"] == condition
                    for s in schedule[::4]
                ),
                4,
            )

    def test_input_strata_and_no_aliasing(self):
        rows = a.reconstruct_roster()
        for row in rows:
            pairs = row["operands"]
            self.assertEqual(
                int(row["strata"]["units_carry_items"]),
                sum(x * y > 0 and abs(x) % 10 + abs(y) % 10 >= 10 for x, y in pairs),
            )
            self.assertEqual(
                int(row["strata"]["zero_sum_items"]), sum(x + y == 0 for x, y in pairs)
            )
        original = copy.deepcopy(rows[1])
        rows[0]["operands"][0][0] = 999
        rows[0]["supplied_answers"][0] = 999
        self.assertEqual(rows[1], original)

    def test_plan_tampering_rejected(self):
        for key, value in (
            ("foil_delta", 0),
            ("instructed_candidate", "other"),
            ("sequence_index", False),
            ("supplied_answers", [0, 0]),
        ):
            p = plan()
            p["rows"][0][key] = value
            with self.assertRaises(AssertionError):
                a.reconstruct_analysis(p, [None] * 64)
        p = plan()
        p["schedule"][0]["repeat_index"] = True
        with self.assertRaises(AssertionError):
            a.reconstruct_analysis(p, [None] * 64)


class AnalysisTests(unittest.TestCase):
    def analyze(self, values):
        return a.reconstruct_analysis(plan(), values)

    def primary(self, result):
        return result["analysis"]["contrasts"]["margin_nats"]["true_cue_minus_foil_cue"]

    def test_all_unknown_fixed_continuous_and_binary_cohorts(self):
        result = self.analyze([None] * 64)
        self.assertEqual(
            self.primary(result),
            dict(
                planned_contexts=16,
                resolved_contexts=0,
                point=None,
                lower=None,
                upper=None,
                unbounded=True,
                planned_pairs=8,
                resolved_pairs=0,
            ),
        )
        switch = result["analysis"]["paired_switch"]
        self.assertEqual(switch["counts"], {"true": 0, "false": 0, "unknown": 8})
        self.assertEqual(switch["rate"], {"point": None, "lower": 0.0, "upper": 1.0})
        self.assertIsNone(switch["qualification"])

    def test_complete_switch_is_separate_from_arithmetic_truth(self):
        result = self.analyze(measurements())
        self.assertEqual(self.primary(result)["point"], 3.0)
        self.assertTrue(result["analysis"]["paired_switch"]["qualification"])
        for row in result["records"]:
            self.assertTrue(row["instruction_compliant_preferred"])
            self.assertIs(row["arithmetic_truth_preferred"], row["condition"] == "true_cue")
            self.assertEqual(row["candidate_scores"]["arithmetic_foil"], -100.0)

    def test_all_sixteen_pair_sign_missingness_states(self):
        for true, foil in itertools.product((None, -1.0, 0.0, 1.0), repeat=2):
            result = self.analyze(measurements(true, foil))
            failure = (true is not None and true <= 0) or (foil is not None and foil >= 0)
            known = true is not None and foil is not None
            expected = False if failure else True if known else None
            switch = result["analysis"]["paired_switch"]
            self.assertIs(switch["qualification"], expected)
            self.assertEqual(switch["counts"]["false"], 8 if failure else 0)
            self.assertEqual(switch["counts"]["true"], 8 if expected is True else 0)
            self.assertEqual(switch["resolved_pairs"], 8 if expected is not None else 0)
            self.assertEqual(self.primary(result)["resolved_pairs"], 8 if known else 0)

    def test_known_all_failed_switch_can_have_no_continuous_pair(self):
        for true, foil in ((0.0, None), (-1.0, None), (None, 0.0), (None, 1.0)):
            result = self.analyze(measurements(true, foil))
            self.assertEqual(
                result["analysis"]["paired_switch"]["rate"],
                {"point": 0.0, "lower": 0.0, "upper": 0.0},
            )
            self.assertIsNone(self.primary(result)["point"])
            self.assertTrue(self.primary(result)["unbounded"])
            self.assertEqual(self.primary(result)["resolved_pairs"], 0)

    def test_positive_shift_without_switch_and_negative_or_zero_primary(self):
        for true, foil, point in ((3.0, 1.0, 2.0), (1.0, 1.0, 0.0), (1.0, 2.0, -1.0)):
            result = self.analyze(measurements(true, foil))
            self.assertEqual(self.primary(result)["point"], point)
            self.assertFalse(result["analysis"]["paired_switch"]["qualification"])

    def test_equal_core_weight_with_all_differences(self):
        result = self.analyze(
            measurements(
                lambda s: float(s["core_index"] + 1), lambda s: -float(s["core_index"] + 1)
            )
        )
        self.assertEqual(self.primary(result)["point"], 9.0)
        self.assertEqual(result["analysis"]["by_condition"]["true_cue"]["point"], 4.5)
        self.assertEqual(result["analysis"]["by_condition"]["foil_cue"]["point"], -4.5)

    def test_mixed_switch_bounds_are_sharp_with_fixed_eight_denominator(self):
        def true(slot):
            return 1.0 if slot["core_index"] < 2 else 0.0 if slot["core_index"] < 5 else None

        result = self.analyze(measurements(true, -1.0))
        switch = result["analysis"]["paired_switch"]
        self.assertEqual(switch["counts"], {"true": 2, "false": 3, "unknown": 3})
        values = [(2 + sum(bits)) / 8 for bits in itertools.product((0, 1), repeat=3)]
        self.assertEqual(
            switch["rate"], {"point": None, "lower": min(values), "upper": max(values)}
        )
        self.assertFalse(switch["qualification"])

    def test_each_missing_slot_and_all_local_missing_patterns(self):
        for missing in range(64):
            values = measurements()
            values[missing] = None
            result = self.analyze(values)
            self.assertEqual(self.primary(result)["resolved_pairs"], 7)
            self.assertIsNone(self.primary(result)["point"])
            self.assertEqual(result["analysis"]["paired_switch"]["counts"]["unknown"], 1)
        for pattern in itertools.product((False, True), repeat=4):
            values = measurements()
            for index, keep in enumerate(pattern):
                if not keep:
                    values[index] = None
            self.assertEqual(self.analyze(values)["records"][0]["measurement_count"], sum(pattern))

    def test_all_completed_readouts_can_still_be_numerically_invalid(self):
        values = measurements()
        for index, slot in enumerate(a.reconstruct_schedule()):
            if slot["repeat_index"] == 1:
                values[index]["logits_sha256"] = hashlib.sha256(
                    f"different|{index}".encode()
                ).hexdigest()
        result = self.analyze(values)
        self.assertEqual(result["analysis"]["resolved_measurements"], 64)
        self.assertEqual(result["analysis"]["overall"]["numerical_valid"]["false"], 16)
        self.assertIsNone(self.primary(result)["point"])
        self.assertIsNone(result["analysis"]["paired_switch"]["qualification"])

    def test_known_repeat_failure_dominates_missing_and_does_not_create_sign(self):
        values = measurements()
        values[0]["logits_sha256"] = "f" * 64
        values[1] = None
        record = self.analyze(values)["records"][0]
        self.assertFalse(record["numerical_valid"])
        self.assertIsNone(record["margin_nats"])
        self.assertIsNone(record["instruction_compliant_preferred"])

    def test_no_tolerance_or_scalar_override(self):
        values = measurements()
        values[0]["score"] = math.nextafter(values[0]["score"], -math.inf)
        with self.assertRaises(AssertionError):
            self.analyze(values)
        values = measurements(0.0, 0.0)
        for index, slot in enumerate(a.reconstruct_schedule()):
            if slot["candidate"] == "arithmetic_truth":
                values[index]["score"] = math.nextafter(
                    -100.0, 0.0 if slot["condition"] == "true_cue" else -math.inf
                )
        self.assertTrue(self.analyze(values)["analysis"]["paired_switch"]["qualification"])

    def test_measurement_type_value_hash_guards(self):
        for value in (
            True,
            {},
            {"score": 0, "logits_sha256": "a" * 64},
            {"score": True, "logits_sha256": "a" * 64},
            {"score": math.nan, "logits_sha256": "a" * 64},
            {"score": math.inf, "logits_sha256": "a" * 64},
            {"score": math.nextafter(0.0, 1.0), "logits_sha256": "a" * 64},
            {"score": -1.0, "logits_sha256": "A" * 64},
            {"score": -1.0, "logits_sha256": "a" * 64, "extra": 0},
        ):
            values = measurements()
            values[0] = value
            with self.assertRaises(AssertionError):
                self.analyze(values)
        with self.assertRaises(AssertionError):
            self.analyze([None] * 63)


class PacketTests(unittest.TestCase):
    def test_exact_aggregate_packet_and_no_item_payload(self):
        packet = a.audit_outcomes(plan(), records(measurements()))
        self.assertEqual(set(packet), {"record_slots", "context_slots", "analysis"})
        self.assertEqual((packet["record_slots"], packet["context_slots"]), (64, 16))
        text = a.canonical(packet).decode()
        for private in (
            "observation_id",
            "evaluation_id",
            "supplied_answers",
            "instructed_candidate",
            "candidate_scores",
            "logits_sha256",
            "messages",
            "expected_sums",
            "supplied_qualification",
        ):
            self.assertNotIn(private, text)

    def test_partial_slots_preserved_without_orphan_promotion(self):
        rows = records([None] * 64)
        rows[0]["status"] = "failed"
        self.assertEqual(a.audit_outcomes(plan(), rows)["analysis"]["resolved_measurements"], 0)
        for rows in (records(measurements()), records([None] * 64)):
            rows[0]["status"] = "failed" if rows[0]["measurement"] is not None else "completed"
            with self.assertRaises(AssertionError):
                a.audit_outcomes(plan(), rows)

    def test_identity_type_order_and_extra_fields_rejected(self):
        for key, value in (
            ("evaluation_index", False),
            ("evaluation_id", "other"),
            ("status", "other"),
            ("private_extra", "text"),
        ):
            rows = records(measurements())
            rows[0][key] = value
            with self.assertRaises(AssertionError):
                a.audit_outcomes(plan(), rows)
        with self.assertRaises(AssertionError):
            a.audit_outcomes(plan(), records(measurements())[::-1])


if __name__ == "__main__":
    unittest.main()
