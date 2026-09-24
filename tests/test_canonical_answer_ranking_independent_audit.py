"""Invented A195 independent-auditor fixtures; no producer or runtime imports."""

import copy
import hashlib
import itertools
import math
import unittest

from lexical_prompt_study import canonical_answer_ranking_independent_audit as a


def plan():
    return {"rows": a.reconstruct_roster(), "schedule": a.reconstruct_schedule()}


def measurements(margin=1.0):
    values = []
    for slot in a.reconstruct_schedule():
        value = margin(slot) if callable(margin) else margin
        score = -100.0 + value if slot["candidate"] == "correct" else -100.0
        digest = hashlib.sha256(
            f"invented|{slot['observation_id']}|{slot['candidate']}".encode()
        ).hexdigest()
        values.append({"score": float(score), "logits_sha256": digest})
    return values


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
    def test_roster_full_factorial_hash_order_and_balance(self):
        rows = a.reconstruct_roster()
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {(row["core_index"], row["condition"]) for row in rows},
            set(itertools.product(range(8), ("computed", "supplied"))),
        )
        expected = sorted(
            (
                hashlib.sha256(
                    f"a195-canonical-answer-ranking-v1|schedule|{core}".encode()
                ).hexdigest(),
                core,
            )
            for core in range(8)
        )
        self.assertEqual(
            [(row["core_schedule_sha256"], row["core_index"]) for row in rows[::2]], expected
        )
        self.assertEqual([row["condition"] for row in rows[::2]], ["computed", "supplied"] * 4)
        self.assertEqual([row["sequence_index"] for row in rows], list(range(16)))

    def test_streamed_modulus_matches_whole_digest_integer_and_strata(self):
        for row in a.reconstruct_roster():
            for item, (left, right) in enumerate(row["operands"]):
                for side, actual in enumerate((left, right)):
                    raw = hashlib.sha256(
                        f"a195-canonical-answer-ranking-v1|{row['core_index']}|{item}|{side}".encode()
                    ).digest()
                    self.assertEqual(
                        actual, (10 + int.from_bytes(raw, "big") % 90) * (-1 if raw[0] & 1 else 1)
                    )
                    self.assertTrue(10 <= abs(actual) <= 99)
            pairs = row["operands"]
            self.assertEqual(row["expected_sums"], [left + right for left, right in pairs])
            self.assertEqual(
                int(row["strata"]["units_carry_items"]),
                sum(
                    left * right > 0 and abs(left) % 10 + abs(right) % 10 >= 10
                    for left, right in pairs
                ),
            )
            self.assertEqual(
                int(row["strata"]["units_borrow_items"]),
                sum(
                    left * right < 0
                    and max(abs(left), abs(right)) % 10 < min(abs(left), abs(right)) % 10
                    for left, right in pairs
                ),
            )

    def test_pair_payload_and_fixed_first_sum_plus_one(self):
        rows = a.reconstruct_roster()
        for one, two in zip(rows[::2], rows[1::2], strict=True):
            self.assertEqual(one["operands"], two["operands"])
            self.assertEqual(one["candidates"], two["candidates"])
            self.assertEqual(one["messages"][0], two["messages"][0])
            for row in (one, two):
                x, y = row["expected_sums"]
                self.assertEqual(
                    row["candidates"], {"correct": f"{x},{y}", "first_sum_plus_one": f"{x + 1},{y}"}
                )
                fields = [line.split(" ") for line in row["messages"][1]["content"].split("\n")]
                self.assertEqual([[int(f[0]), int(f[1])] for f in fields], row["operands"])
                self.assertEqual(
                    [f[2] for f in fields],
                    [str(x), str(y)] if row["condition"] == "supplied" else ["?", "?"],
                )

    def test_schedule_exact_occurrences_ids_and_candidate_order(self):
        slots = a.reconstruct_schedule()
        self.assertEqual(len(slots), 64)
        self.assertEqual(len({slot["evaluation_id"] for slot in slots}), 64)
        self.assertEqual([slot["evaluation_index"] for slot in slots], list(range(64)))
        for row in range(16):
            block = slots[4 * row : 4 * row + 4]
            first = "correct" if row % 2 == 0 else "first_sum_plus_one"
            other = "first_sum_plus_one" if first == "correct" else "correct"
            self.assertEqual(
                [(slot["candidate"], slot["repeat_index"]) for slot in block],
                [(first, 0), (other, 0), (other, 1), (first, 1)],
            )
            self.assertTrue(all(slot["sequence_index"] == row for slot in block))

    def test_plan_answer_condition_schedule_tampering_rejected(self):
        for field, replacement in (
            ("expected_sums", [0, 0]),
            ("condition", "supplied"),
            ("sequence_index", False),
        ):
            p = plan()
            p["rows"][0][field] = replacement
            with self.assertRaises(AssertionError):
                a.reconstruct_analysis(p, [None] * 64)
        p = plan()
        p["schedule"][0]["repeat_index"] = True
        with self.assertRaises(AssertionError):
            a.reconstruct_analysis(p, [None] * 64)

    def test_independent_objects_and_no_tokenizer_fields_needed(self):
        p = plan()
        old = copy.deepcopy(p["rows"][1])
        p["rows"][0]["operands"][0][0] = 999
        self.assertEqual(p["rows"][1], old)
        self.assertNotEqual(a.reconstruct_roster()[0]["operands"][0][0], 999)
        self.assertEqual(set(plan()), {"rows", "schedule"})


class AnalysisTests(unittest.TestCase):
    def analyze(self, values):
        return a.reconstruct_analysis(plan(), values)

    def test_all_unknown_is_full_unbounded_cohort(self):
        result = self.analyze([None] * 64)
        analysis = result["analysis"]
        self.assertEqual(analysis["resolved_measurements"], 0)
        self.assertIsNone(analysis["supplied_qualification"])
        self.assertEqual(
            analysis["contrasts"]["margin_nats"]["supplied_minus_computed"],
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
        self.assertEqual(
            analysis["overall"]["numerical_valid"], {"true": 0, "false": 0, "unknown": 16}
        )
        self.assertEqual(analysis["overall"]["margin_signs"]["unknown"], 16)

    def test_exact_positive_zero_negative_and_first_score_records(self):
        for margin, sign, qualified in (
            (2.0, "positive", True),
            (0.0, "zero", False),
            (-2.0, "negative", False),
        ):
            result = self.analyze(measurements(margin))
            for row in result["records"]:
                self.assertEqual(
                    row["candidate_scores"],
                    {"correct": -100.0 + margin, "first_sum_plus_one": -100.0},
                )
                self.assertEqual(row["margin_nats"], margin)
                self.assertEqual(row["margin_sign"], sign)
                self.assertEqual(row["measurement_count"], 4)
            self.assertIs(result["analysis"]["supplied_qualification"], qualified)
            self.assertEqual(
                result["analysis"]["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"],
                0.0,
            )

    def test_equal_core_primary_sign_weight_and_fixed_arm_means(self):
        def margin(slot):
            return float(slot["core_index"] + 1) * (2 if slot["condition"] == "supplied" else 1)

        result = self.analyze(measurements(margin))["analysis"]
        self.assertEqual(result["by_condition"]["computed"]["point"], 4.5)
        self.assertEqual(result["by_condition"]["supplied"]["point"], 9.0)
        contrast = result["contrasts"]["margin_nats"]["supplied_minus_computed"]
        self.assertEqual((contrast["point"], contrast["lower"], contrast["upper"]), (4.5, 4.5, 4.5))
        self.assertFalse(contrast["unbounded"])
        self.assertEqual((contrast["resolved_contexts"], contrast["resolved_pairs"]), (16, 8))

    def test_cancellation_uses_full_pairs_without_sign_selection(self):
        def margin(slot):
            return ((-1.0) ** slot["core_index"]) if slot["condition"] == "supplied" else 0.0

        result = self.analyze(measurements(margin))["analysis"]
        self.assertEqual(
            result["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"], 0.0
        )
        self.assertEqual(
            result["by_condition"]["supplied"]["margin_signs"],
            {"positive": 4, "negative": 4, "zero": 0, "unknown": 0},
        )
        self.assertFalse(result["supplied_qualification"])

    def test_every_single_missing_measurement_keeps_denominators(self):
        for index in range(64):
            values = measurements()
            values[index] = None
            result = self.analyze(values)["analysis"]
            contrast = result["contrasts"]["margin_nats"]["supplied_minus_computed"]
            self.assertEqual((contrast["resolved_contexts"], contrast["resolved_pairs"]), (15, 7))
            self.assertTrue(contrast["unbounded"])
            self.assertIsNone(contrast["point"])
            self.assertEqual(result["resolved_measurements"], 63)

    def test_all_sixteen_local_missingness_patterns(self):
        for pattern in itertools.product((False, True), repeat=4):
            values = measurements()
            for index, present in enumerate(pattern):
                if not present:
                    values[index] = None
            result = self.analyze(values)
            first = result["records"][0]
            self.assertEqual(first["measurement_count"], sum(pattern))
            self.assertIs(first["numerical_valid"], True if all(pattern) else None)
            self.assertEqual(
                result["analysis"]["overall"]["resolved_contexts"], 16 if all(pattern) else 15
            )

    def test_mismatch_is_false_not_known_nonpositive_margin(self):
        values = measurements()
        slot = next(
            index
            for index, item in enumerate(a.reconstruct_schedule())
            if item["condition"] == "supplied"
        )
        values[slot]["logits_sha256"] = "f" * 64
        result = self.analyze(values)
        row = result["records"][slot // 4]
        self.assertTrue(row["complete"])
        self.assertFalse(row["numerical_valid"])
        self.assertIsNone(row["margin_nats"])
        self.assertIsNone(result["analysis"]["supplied_qualification"])

    def test_all_completed_calls_do_not_establish_repeat_or_scientific_validity(self):
        values = measurements()
        for index, slot in enumerate(a.reconstruct_schedule()):
            if slot["repeat_index"] == 1:
                values[index]["logits_sha256"] = hashlib.sha256(
                    f"different invented bytes|{slot['evaluation_id']}".encode()
                ).hexdigest()
        packet = a.audit_outcomes(plan(), records(values))
        analysis = packet["analysis"]
        self.assertEqual(analysis["resolved_measurements"], 64)
        self.assertEqual(analysis["overall"]["complete_contexts"], 16)
        self.assertEqual(
            analysis["overall"]["numerical_valid"], {"true": 0, "false": 16, "unknown": 0}
        )
        self.assertEqual(analysis["overall"]["resolved_contexts"], 0)
        primary = analysis["contrasts"]["margin_nats"]["supplied_minus_computed"]
        self.assertIsNone(primary["point"])
        self.assertIsNone(primary["lower"])
        self.assertIsNone(primary["upper"])
        self.assertTrue(primary["unbounded"])
        self.assertIsNone(analysis["supplied_qualification"])

    def test_known_repeat_failure_dominates_other_unknown_repeat(self):
        values = measurements()
        values[0]["logits_sha256"] = "f" * 64
        values[1] = None
        self.assertFalse(self.analyze(values)["records"][0]["numerical_valid"])

    def test_known_supplied_nonpositive_dominates_missing(self):
        for margin in (0.0, -1.0):
            values = [None] * 64
            full = measurements(margin)
            index = (
                next(
                    row["sequence_index"]
                    for row in a.reconstruct_roster()
                    if row["condition"] == "supplied"
                )
                * 4
            )
            values[index : index + 4] = full[index : index + 4]
            self.assertFalse(self.analyze(values)["analysis"]["supplied_qualification"])

    def test_supplied_known_positive_all_required_and_computed_is_separate(self):
        values = measurements()
        for index, slot in enumerate(a.reconstruct_schedule()):
            if slot["condition"] == "computed":
                values[index] = None
        result = self.analyze(values)["analysis"]
        self.assertTrue(result["supplied_qualification"])
        self.assertEqual(result["by_condition"]["supplied"]["point"], 1.0)
        self.assertIsNone(result["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"])
        values[next(i for i, v in enumerate(values) if v is not None)] = None
        self.assertIsNone(self.analyze(values)["analysis"]["supplied_qualification"])

    def test_no_repeat_tolerance_and_no_score_override(self):
        values = measurements()
        values[0]["score"] = math.nextafter(values[0]["score"], -math.inf)
        with self.assertRaises(AssertionError):
            self.analyze(values)
        values = measurements()
        values[0]["score"] = values[3]["score"] = math.nextafter(-100.0, 0.0)
        self.assertGreater(self.analyze(values)["records"][0]["margin_nats"], 0)

    def test_invalid_measurement_types_hashes_and_impossible_scores(self):
        bad = [
            True,
            {},
            {"score": 0, "logits_sha256": "a" * 64},
            {"score": True, "logits_sha256": "a" * 64},
            {"score": math.nan, "logits_sha256": "a" * 64},
            {"score": math.inf, "logits_sha256": "a" * 64},
            {"score": math.nextafter(0.0, 1.0), "logits_sha256": "a" * 64},
            {"score": -1.0, "logits_sha256": "A" * 64},
            {"score": -1.0, "logits_sha256": "a" * 64, "extra": 1},
        ]
        for value in bad:
            values = measurements()
            values[0] = value
            with self.assertRaises(AssertionError):
                self.analyze(values)
        with self.assertRaises(AssertionError):
            self.analyze([None] * 63)


class AggregatePacketTests(unittest.TestCase):
    def test_only_three_aggregate_keys_and_no_item_payloads(self):
        packet = a.audit_outcomes(plan(), records(measurements()))
        self.assertEqual(set(packet), {"record_slots", "context_slots", "analysis"})
        self.assertEqual((packet["record_slots"], packet["context_slots"]), (64, 16))
        serialized = a.canonical(packet).decode()
        for private in (
            "evaluation_id",
            "observation_id",
            "expected_sums",
            "messages",
            "candidate_scores",
            "logits_sha256",
        ):
            self.assertNotIn(private, serialized)

    def test_partial_slot_states_preserve_unknowns(self):
        slots = records([None] * 64)
        slots[0]["status"] = "failed"
        result = a.audit_outcomes(plan(), slots)
        self.assertEqual(result["analysis"]["resolved_measurements"], 0)
        self.assertIsNone(result["analysis"]["supplied_qualification"])

    def test_orphan_measurement_cannot_be_promoted(self):
        slots = records(measurements())
        slots[0]["status"] = "failed"
        with self.assertRaises(AssertionError):
            a.audit_outcomes(plan(), slots)
        slots = records([None] * 64)
        slots[0]["status"] = "completed"
        with self.assertRaises(AssertionError):
            a.audit_outcomes(plan(), slots)

    def test_record_identity_types_order_and_extra_fields_rejected(self):
        for key, value in (
            ("evaluation_index", False),
            ("evaluation_id", "other"),
            ("status", "missing"),
            ("private_extra", "text"),
        ):
            slots = records(measurements())
            slots[0][key] = value
            with self.assertRaises(AssertionError):
                a.audit_outcomes(plan(), slots)
        with self.assertRaises(AssertionError):
            a.audit_outcomes(plan(), records(measurements())[::-1])


if __name__ == "__main__":
    unittest.main()
