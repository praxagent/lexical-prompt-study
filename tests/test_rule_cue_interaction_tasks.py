"""Invented A197 pure task controls; no native calls or study evidence reads."""

import copy
import hashlib
import itertools
import math
import unittest

from lexical_prompt_study import rule_cue_interaction_tasks as m


def measurements(margins=(3.0, -1.0, 2.0, 2.0)):
    values = []
    for slot in m.build_schedule():
        local = margins(slot) if callable(margins) else margins
        margin = local[m.CELLS.index((slot["rule"], slot["condition"]))]
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
    def test_exact_namespace_full_factorial_and_rule_identity(self):
        rows = m.build_roster()
        self.assertEqual(m.NAMESPACE, "a197-rule-cue-interaction-v1")
        self.assertEqual((m.CORE_COUNT, m.ROW_COUNT, m.CALL_COUNT), (8, 32, 128))
        self.assertEqual(
            {(r["core_index"], r["rule"], r["condition"]) for r in rows},
            set(itertools.product(range(8), ("copy", "recompute"), ("true_cue", "foil_cue"))),
        )
        self.assertEqual(len({r["observation_id"] for r in rows}), 32)
        self.assertEqual([r["sequence_index"] for r in rows], list(range(32)))
        self.assertIn("rule", m.IDENTITY_KEYS)

    def test_independent_hash_arithmetic_and_core_parity_foil(self):
        for core, item, side in itertools.product(range(8), range(2), range(2)):
            raw = hashlib.sha256(
                f"a197-rule-cue-interaction-v1|{core}|{item}|{side}".encode()
            ).digest()
            value = 0
            for byte in raw:
                value = (256 * value + byte) % 90
            self.assertEqual(m.operand(core, item, side), (10 + value) * (-1 if raw[0] & 1 else 1))
        self.assertEqual([m.foil_delta(i) for i in range(8)], [1, -1] * 4)

    def test_exact_rule_strings_and_no_missing_answer_fields(self):
        self.assertEqual(
            m.COPY_SYSTEM_INSTRUCTION,
            "Each of the two data rows contains a left integer, a right integer, and an answer field. If the answer field is ?, add the two integers; otherwise copy the supplied answer. Preserve row order. Output exactly two comma-separated base-10 integer answers, with no other text.",
        )
        self.assertEqual(
            m.RECOMPUTE_SYSTEM_INSTRUCTION,
            "Each of the two data rows contains a left integer, a right integer, and an answer field. Ignore the supplied answer field and add the two integers. Preserve row order. Output exactly two comma-separated base-10 integer answers, with no other text.",
        )
        for row in m.build_roster():
            self.assertEqual(row["messages"][0]["content"], m.SYSTEM_INSTRUCTIONS[row["rule"]])
            self.assertNotIn("?", row["messages"][1]["content"])

    def test_same_user_by_cue_and_candidates_across_all_four_contexts(self):
        rows = m.build_roster()
        for core in range(8):
            local = {(r["rule"], r["condition"]): r for r in rows if r["core_index"] == core}
            first = next(iter(local.values()))
            for row in local.values():
                for field in ("operands", "expected_sums", "candidates", "foil_delta", "strata"):
                    self.assertEqual(row[field], first[field])
                x, y = row["expected_sums"]
                delta = 1 if core % 2 == 0 else -1
                self.assertEqual(
                    row["candidates"],
                    {"arithmetic_truth": f"{x},{y}", "arithmetic_foil": f"{x + delta},{y}"},
                )
                expected = [x, y] if row["condition"] == "true_cue" else [x + delta, y]
                self.assertEqual(row["supplied_answers"], expected)
                fields = [
                    [int(v) for v in line.split(" ")]
                    for line in row["messages"][1]["content"].split("\n")
                ]
                self.assertEqual([field[:2] for field in fields], row["operands"])
                self.assertEqual([field[2] for field in fields], expected)
                target = (
                    "arithmetic_foil"
                    if row["rule"] == "copy" and row["condition"] == "foil_cue"
                    else "arithmetic_truth"
                )
                self.assertEqual(row["instructed_candidate"], target)
            for cue in m.CONDITIONS:
                self.assertEqual(
                    local[("copy", cue)]["messages"][1], local[("recompute", cue)]["messages"][1]
                )

    def test_hash_core_order_and_all_rotations_twice(self):
        rows = m.build_roster()
        order = sorted(
            (hashlib.sha256(f"{m.NAMESPACE}|schedule|{core}".encode()).hexdigest(), core)
            for core in range(8)
        )
        self.assertEqual([(r["core_schedule_sha256"], r["core_index"]) for r in rows[::4]], order)
        self.assertEqual([r["rotation"] for r in rows[::4]], [0, 1, 2, 3] * 2)
        for index in range(8):
            local = rows[index * 4 : index * 4 + 4]
            rotation = index % 4
            expected = m.CELLS[rotation:] + m.CELLS[:rotation]
            self.assertEqual(tuple((r["rule"], r["condition"]) for r in local), expected)
            self.assertEqual([r["within_core_index"] for r in local], list(range(4)))
        for cell in m.CELLS:
            positions = [
                r["within_core_index"] for r in rows if (r["rule"], r["condition"]) == cell
            ]
            self.assertEqual([positions.count(i) for i in range(4)], [2] * 4)

    def test_first_candidate_balanced_each_cell_and_each_core(self):
        slots = m.build_schedule()
        self.assertEqual(len(slots), 128)
        self.assertEqual(len({s["evaluation_id"] for s in slots}), 128)
        self.assertEqual([s["evaluation_index"] for s in slots], list(range(128)))
        for row in m.build_roster():
            start = 4 * row["sequence_index"]
            block = slots[start : start + 4]
            parity = (row["core_order_index"] // 4 + row["within_core_index"]) % 2
            first, other = m.CANDIDATES[parity], m.CANDIDATES[1 - parity]
            self.assertEqual(
                [(s["candidate"], s["repeat_index"]) for s in block],
                [(first, 0), (other, 0), (other, 1), (first, 1)],
            )
            self.assertTrue(all(s["rule"] == row["rule"] for s in block))
        for rule, cue in m.CELLS:
            starts = [s for s in slots[::4] if s["rule"] == rule and s["condition"] == cue]
            self.assertEqual(sum(s["candidate"] == "arithmetic_truth" for s in starts), 4)
        for core in range(8):
            self.assertEqual(
                sum(
                    s["candidate"] == "arithmetic_truth" and s["core_index"] == core
                    for s in slots[::4]
                ),
                2,
            )

    def test_strata_limits_no_aliasing(self):
        rows = m.build_roster()
        for row in rows:
            self.assertEqual(row["expected_sums"], [sum(p) for p in row["operands"]])
            self.assertTrue(all(type(v) is str for v in row["strata"].values()))
            self.assertEqual(
                int(row["strata"]["zero_sum_items"]), sum(x + y == 0 for x, y in row["operands"])
            )
        original = copy.deepcopy(rows[1])
        rows[0]["supplied_answers"][0] = 500
        rows[0]["operands"][0][0] = 500
        self.assertEqual(rows[1], original)
        self.assertEqual(
            (m.MAX_PROMPT_TOKENS, m.MAX_CANDIDATE_TOKENS, m.EOT_TOKEN), (256, 16, "<|eot_id|>")
        )

    def test_exact_validation_indices_and_rule_tampering(self):
        for key, value in (
            ("rule", "other"),
            ("condition", "other"),
            ("core_order_index", False),
            ("rotation", 99),
            ("extra", 0),
        ):
            rows = m.build_roster()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                m.validate_roster(rows)
        slots = m.build_schedule()
        del slots[0]["rule"]
        with self.assertRaises(ValueError):
            m.validate_schedule(slots)
        self.assertEqual(m.validate_row(m.build_roster()[31]), m.build_roster()[31])
        for args in ((True, 0, 0), (8, 0, 0), (0, 2, 0), (0, 0, False)):
            with self.assertRaises(ValueError):
                m.operand(*args)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.schedule = m.build_roster(), m.build_schedule()

    def analyze(self, values):
        return m.analyze_measurements(self.rows, self.schedule, values)

    def primary(self, result):
        return result["analysis"]["contrasts"]["margin_nats"]["copy_minus_recompute_cue_effect"]

    def test_all_unknown_has_fixed_continuous_and_pattern_bounds(self):
        result = self.analyze([None] * 128)
        self.assertEqual(
            self.primary(result),
            dict(
                planned_contexts=32,
                resolved_contexts=0,
                planned_cores=8,
                resolved_cores=0,
                point=None,
                lower=None,
                upper=None,
                unbounded=True,
            ),
        )
        for pattern in result["analysis"]["patterns"].values():
            self.assertEqual(pattern["counts"], {"true": 0, "false": 0, "unknown": 8})
            self.assertEqual(pattern["rate"], {"point": None, "lower": 0.0, "upper": 1.0})
            self.assertIsNone(pattern["qualification"])

    def test_positive_interaction_and_all_strict_patterns(self):
        result = self.analyze(measurements())
        self.assertEqual(self.primary(result)["point"], 4.0)
        by_rule = result["analysis"]["contrasts"]["margin_nats"]["by_rule"]
        self.assertEqual((by_rule["copy"]["point"], by_rule["recompute"]["point"]), (4.0, 0.0))
        for pattern in result["analysis"]["patterns"].values():
            self.assertEqual(pattern["counts"], {"true": 8, "false": 0, "unknown": 0})
            self.assertTrue(pattern["qualification"])
        for record in result["records"]:
            self.assertTrue(record["instruction_compliant_preferred"])
            self.assertIs(
                record["arithmetic_truth_preferred"],
                not (record["rule"] == "copy" and record["condition"] == "foil_cue"),
            )

    def test_zero_negative_interaction_can_have_all_patterns_true(self):
        for margins, expected in (((3.0, -1.0, 6.0, 2.0), 0.0), ((2.0, -1.0, 6.0, 1.0), -2.0)):
            result = self.analyze(measurements(margins))
            self.assertEqual(self.primary(result)["point"], expected)
            self.assertTrue(result["analysis"]["patterns"]["joint"]["qualification"])

    def test_positive_interaction_does_not_imply_strict_copy_pattern(self):
        result = self.analyze(measurements((3.0, 1.0, 1.0, 1.0)))
        self.assertEqual(self.primary(result)["point"], 2.0)
        patterns = result["analysis"]["patterns"]
        self.assertFalse(patterns["copying_switch"]["qualification"])
        self.assertTrue(patterns["recompute_truth"]["qualification"])
        self.assertFalse(patterns["joint"]["qualification"])

    def test_equal_core_weight_and_four_cell_summaries(self):
        result = self.analyze(measurements(lambda s: (float(s["core_index"] + 1), -1.0, 2.0, 2.0)))
        self.assertEqual(self.primary(result)["point"], 5.5)
        cells = result["analysis"]["by_rule_cue"]
        self.assertEqual(cells["copy"]["true_cue"]["point"], 4.5)
        self.assertEqual(cells["copy"]["foil_cue"]["point"], -1.0)
        self.assertEqual(cells["recompute"]["true_cue"]["planned_contexts"], 8)

    def test_all_256_four_context_sign_and_missing_states(self):
        for margins in itertools.product((None, -1.0, 0.0, 1.0), repeat=4):
            result = self.analyze(measurements(margins))
            expected_signs = {
                "copying_switch": ((0, 1), (1, -1)),
                "recompute_truth": ((2, 3), (1, 1)),
                "joint": ((0, 1, 2, 3), (1, -1, 1, 1)),
            }
            for name, (indices, signs) in expected_signs.items():
                selected = [margins[i] for i in indices]
                disqualified = any(
                    v is not None and v * sign <= 0 for v, sign in zip(selected, signs, strict=True)
                )
                known = all(v is not None for v in selected)
                expected = False if disqualified else True if known else None
                pattern = result["analysis"]["patterns"][name]
                self.assertIs(pattern["qualification"], expected)
                self.assertEqual(pattern["resolved_cores"], 0 if expected is None else 8)
            self.assertEqual(
                self.primary(result)["resolved_cores"],
                8 if all(v is not None for v in margins) else 0,
            )

    def test_known_bad_partial_resolves_patterns_without_continuous_primary(self):
        result = self.analyze(measurements((0.0, None, None, None)))
        self.assertIsNone(self.primary(result)["point"])
        self.assertTrue(self.primary(result)["unbounded"])
        for name in ("copying_switch", "joint"):
            self.assertEqual(
                result["analysis"]["patterns"][name]["rate"],
                {"point": 0.0, "lower": 0.0, "upper": 0.0},
            )
        self.assertIsNone(result["analysis"]["patterns"]["recompute_truth"]["qualification"])

    def test_missing_other_rule_does_not_erase_complete_rule_contrast(self):
        result = self.analyze(measurements((None, None, 3.0, 1.0)))
        effects = result["analysis"]["contrasts"]["margin_nats"]
        self.assertIsNone(effects["copy_minus_recompute_cue_effect"]["point"])
        self.assertIsNone(effects["by_rule"]["copy"]["point"])
        self.assertEqual(effects["by_rule"]["recompute"]["point"], 2.0)
        self.assertTrue(result["analysis"]["patterns"]["recompute_truth"]["qualification"])

    def test_mixed_binary_bounds_keep_all_eight_cores(self):
        def margins(slot):
            return (
                (3.0, -1.0, 2.0, 2.0)
                if slot["core_index"] < 2
                else (0.0, -1.0, 2.0, 2.0)
                if slot["core_index"] < 5
                else (None, -1.0, 2.0, 2.0)
            )

        result = self.analyze(measurements(margins))
        for name in ("copying_switch", "joint"):
            pattern = result["analysis"]["patterns"][name]
            self.assertEqual(pattern["counts"], {"true": 2, "false": 3, "unknown": 3})
            possible = [(2 + sum(bits)) / 8 for bits in itertools.product((0, 1), repeat=3)]
            self.assertEqual(
                pattern["rate"], {"point": None, "lower": min(possible), "upper": max(possible)}
            )
        self.assertIsNone(self.primary(result)["point"])

    def test_every_single_missing_measurement_keeps_primary_cohort(self):
        for index in range(128):
            values = measurements()
            values[index] = None
            result = self.analyze(values)
            primary = self.primary(result)
            self.assertEqual((primary["resolved_contexts"], primary["resolved_cores"]), (31, 7))
            self.assertEqual(
                (primary["point"], primary["lower"], primary["upper"]), (None, None, None)
            )
            self.assertTrue(primary["unbounded"])
            self.assertEqual(result["analysis"]["patterns"]["joint"]["counts"]["unknown"], 1)

    def test_all_local_repeat_missing_patterns(self):
        for present in itertools.product((False, True), repeat=4):
            values = measurements()
            for index, keep in enumerate(present):
                if not keep:
                    values[index] = None
            result = self.analyze(values)
            self.assertEqual(result["records"][0]["measurement_count"], sum(present))
            self.assertIs(result["records"][0]["numerical_valid"], True if all(present) else None)

    def test_complete_measurements_can_have_all_invalid_margins(self):
        values = measurements()
        for index, slot in enumerate(self.schedule):
            if slot["repeat_index"] == 1:
                values[index]["logits_sha256"] = hashlib.sha256(
                    f"different|{index}".encode()
                ).hexdigest()
        result = self.analyze(values)
        self.assertEqual(result["analysis"]["resolved_measurements"], 128)
        self.assertEqual(result["analysis"]["overall"]["numerical_valid"]["false"], 32)
        self.assertIsNone(self.primary(result)["point"])
        self.assertIsNone(result["analysis"]["patterns"]["joint"]["qualification"])

    def test_first_copy_no_repeat_tolerance_and_known_bad_repeat_dominance(self):
        values = measurements()
        values[0]["score"] = math.nextafter(values[0]["score"], -math.inf)
        with self.assertRaisesRegex(ValueError, "equal_readout_score_mismatch"):
            self.analyze(values)
        values = measurements()
        values[0]["logits_sha256"] = "f" * 64
        values[1] = None
        record = self.analyze(values)["records"][0]
        self.assertFalse(record["numerical_valid"])
        self.assertIsNone(record["margin_nats"])

    def test_strict_tiny_signs_and_ties(self):
        values = measurements((0.0, 0.0, 0.0, 0.0))
        for index, slot in enumerate(self.schedule):
            if slot["candidate"] == "arithmetic_truth":
                target = (
                    -math.inf if slot["rule"] == "copy" and slot["condition"] == "foil_cue" else 0.0
                )
                values[index]["score"] = math.nextafter(-100.0, target)
        self.assertTrue(self.analyze(values)["analysis"]["patterns"]["joint"]["qualification"])
        self.assertFalse(
            self.analyze(measurements((0.0, 0.0, 0.0, 0.0)))["analysis"]["patterns"]["joint"][
                "qualification"
            ]
        )

    def test_invalid_measurements_and_rule_schedule_identity(self):
        for value in (
            True,
            {},
            {"score": True, "logits_sha256": "a" * 64},
            {"score": 0, "logits_sha256": "a" * 64},
            {"score": math.nan, "logits_sha256": "a" * 64},
            {"score": math.inf, "logits_sha256": "a" * 64},
            {"score": math.nextafter(0.0, 1.0), "logits_sha256": "a" * 64},
            {"score": -1.0, "logits_sha256": "A" * 64},
            {"score": -1.0, "logits_sha256": "a" * 64, "extra": 0},
        ):
            values = measurements()
            values[0] = value
            with self.assertRaises(ValueError):
                self.analyze(values)
        with self.assertRaises(ValueError):
            self.analyze([None] * 127)
        slots = copy.deepcopy(self.schedule)
        slots[0]["rule"] = "recompute" if slots[0]["rule"] == "copy" else "copy"
        with self.assertRaises(ValueError):
            m.analyze_measurements(self.rows, slots, measurements())

    def test_aggregate_contains_no_item_payload_or_old_gate(self):
        analysis = self.analyze(measurements())["analysis"]
        self.assertEqual(set(analysis["patterns"]), {"copying_switch", "recompute_truth", "joint"})
        text = m._canonical(analysis).decode()
        for excluded in (
            "observation_id",
            "evaluation_id",
            "expected_sums",
            "messages",
            "logits_sha256",
            "supplied_answers",
            "paired_switch",
            "supplied_qualification",
        ):
            self.assertNotIn(excluded, text)


if __name__ == "__main__":
    unittest.main()
