"""Invented A197 audit fixtures; no producer import or native/evidence reads."""

import copy
import hashlib
import itertools
import math
import unittest

from lexical_prompt_study import rule_cue_interaction_independent_audit as a

CELLS = (
    ("copy", "true_cue"),
    ("copy", "foil_cue"),
    ("recompute", "true_cue"),
    ("recompute", "foil_cue"),
)


def plan():
    return {"rows": a.reconstruct_roster(), "schedule": a.reconstruct_schedule()}


def measurements(margins=(3.0, -1.0, 2.0, 2.0)):
    values = []
    for slot in a.reconstruct_schedule():
        local = margins(slot["core_index"]) if callable(margins) else margins
        margin = local[CELLS.index((slot["rule"], slot["condition"]))]
        values.append(
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
    def test_namespace_factorial_and_independent_digest_arithmetic(self):
        rows = a.reconstruct_roster()
        self.assertEqual(a.NAMESPACE, "a197-rule-cue-interaction-v1")
        self.assertEqual(len(rows), 32)
        self.assertEqual(
            {(r["core_index"], r["rule"], r["condition"]) for r in rows},
            set(itertools.product(range(8), ("copy", "recompute"), ("true_cue", "foil_cue"))),
        )
        for row in rows:
            for item, pair in enumerate(row["operands"]):
                for side, value in enumerate(pair):
                    raw = hashlib.sha256(
                        f"a197-rule-cue-interaction-v1|{row['core_index']}|{item}|{side}".encode()
                    ).digest()
                    self.assertEqual(
                        value, (10 + int.from_bytes(raw, "big") % 90) * (-1 if raw[0] & 1 else 1)
                    )
            self.assertEqual(row["expected_sums"], list(map(sum, row["operands"])))

    def test_rules_are_exact_selected_literals(self):
        self.assertEqual(
            a.COPY_INSTRUCTION,
            "Each of the two data rows contains a left integer, a right integer, and an answer field. If the answer field is ?, add the two integers; otherwise copy the supplied answer. Preserve row order. Output exactly two comma-separated base-10 integer answers, with no other text.",
        )
        self.assertEqual(
            a.RECOMPUTE_INSTRUCTION,
            "Each of the two data rows contains a left integer, a right integer, and an answer field. Ignore the supplied answer field and add the two integers. Preserve row order. Output exactly two comma-separated base-10 integer answers, with no other text.",
        )
        for row in a.reconstruct_roster():
            self.assertEqual(
                row["messages"][0]["content"],
                a.COPY_INSTRUCTION if row["rule"] == "copy" else a.RECOMPUTE_INSTRUCTION,
            )

    def test_four_contexts_common_candidates_and_cue_payload(self):
        rows = a.reconstruct_roster()
        deltas = {}
        for core in range(8):
            cells = {(r["rule"], r["condition"]): r for r in rows if r["core_index"] == core}
            base = cells[("copy", "true_cue")]
            x, y = base["expected_sums"]
            delta = 1 if core % 2 == 0 else -1
            deltas[core] = base["foil_delta"]
            for (rule, cue), row in cells.items():
                for key in ("operands", "expected_sums", "strata", "candidates", "foil_delta"):
                    self.assertEqual(row[key], base[key])
                self.assertEqual(
                    row["candidates"],
                    {"arithmetic_truth": f"{x},{y}", "arithmetic_foil": f"{x + delta},{y}"},
                )
                self.assertEqual(
                    row["supplied_answers"], [x + (delta if cue == "foil_cue" else 0), y]
                )
                self.assertEqual(
                    row["instructed_candidate"],
                    "arithmetic_foil"
                    if (rule, cue) == ("copy", "foil_cue")
                    else "arithmetic_truth",
                )
                parsed = [
                    [int(v) for v in line.split(" ")]
                    for line in row["messages"][1]["content"].split("\n")
                ]
                self.assertEqual([v[:2] for v in parsed], row["operands"])
                self.assertEqual([v[2] for v in parsed], row["supplied_answers"])
                self.assertNotIn("?", row["messages"][1]["content"])
            for cue in ("true_cue", "foil_cue"):
                self.assertEqual(
                    cells[("copy", cue)]["messages"][1], cells[("recompute", cue)]["messages"][1]
                )
        self.assertEqual([deltas[i] for i in range(8)], [1, -1] * 4)

    def test_hash_order_rotations_position_and_candidate_balance(self):
        rows, slots = a.reconstruct_roster(), a.reconstruct_schedule()
        wanted = sorted(
            (hashlib.sha256(f"{a.NAMESPACE}|schedule|{i}".encode()).hexdigest(), i)
            for i in range(8)
        )
        self.assertEqual([(r["core_schedule_sha256"], r["core_index"]) for r in rows[::4]], wanted)
        self.assertEqual([r["rotation"] for r in rows[::4]], [0, 1, 2, 3] * 2)
        self.assertEqual([s["evaluation_index"] for s in slots], list(range(128)))
        self.assertEqual(len({s["evaluation_id"] for s in slots}), 128)
        for row in rows:
            index = row["sequence_index"]
            self.assertEqual((row["rule"], row["condition"]), CELLS[(index // 4 + index % 4) % 4])
            first = a.PATHS[((index // 16) + index % 4) % 2]
            other = next(p for p in a.PATHS if p != first)
            self.assertEqual(
                [(s["candidate"], s["repeat_index"]) for s in slots[index * 4 : index * 4 + 4]],
                [(first, 0), (other, 0), (other, 1), (first, 1)],
            )
        for cell in CELLS:
            local = [r for r in rows if (r["rule"], r["condition"]) == cell]
            self.assertEqual(
                [sum(r["within_core_index"] == i for r in local) for i in range(4)], [2] * 4
            )
            self.assertEqual(
                sum(
                    slots[4 * r["sequence_index"]]["candidate"] == "arithmetic_truth" for r in local
                ),
                4,
            )
        for core in range(8):
            self.assertEqual(
                sum(
                    s["core_index"] == core and s["candidate"] == "arithmetic_truth"
                    for s in slots[::4]
                ),
                2,
            )

    def test_strata_and_defensive_construction(self):
        rows = a.reconstruct_roster()
        for row in rows:
            pairs = row["operands"]
            self.assertEqual(
                int(row["strata"]["units_carry_items"]),
                sum(x * y > 0 and abs(x) % 10 + abs(y) % 10 >= 10 for x, y in pairs),
            )
            self.assertEqual(
                int(row["strata"]["units_borrow_items"]),
                sum(
                    x * y < 0 and max(abs(x), abs(y)) % 10 < min(abs(x), abs(y)) % 10
                    for x, y in pairs
                ),
            )
            self.assertEqual(
                int(row["strata"]["zero_sum_items"]), sum(x + y == 0 for x, y in pairs)
            )
        saved = copy.deepcopy(rows[1])
        rows[0]["operands"][0][0] = 900
        rows[0]["supplied_answers"][0] = 900
        self.assertEqual(rows[1], saved)

    def test_plan_type_identity_and_rule_tampering(self):
        for key, value in (
            ("rule", "other"),
            ("sequence_index", False),
            ("core_order_index", False),
            ("within_core_index", 9),
            ("foil_delta", 0),
            ("supplied_answers", [0, 0]),
        ):
            p = plan()
            p["rows"][0][key] = value
            with self.assertRaises(AssertionError):
                a.reconstruct_analysis(p, [None] * 128)
        for key, value in (("rule", "other"), ("evaluation_index", False), ("repeat_index", True)):
            p = plan()
            p["schedule"][0][key] = value
            with self.assertRaises(AssertionError):
                a.reconstruct_analysis(p, [None] * 128)


class AnalysisTests(unittest.TestCase):
    def run_values(self, values):
        return a.reconstruct_analysis(plan(), values)

    def primary(self, result):
        return result["analysis"]["contrasts"]["margin_nats"]["copy_minus_recompute_cue_effect"]

    def test_all_unknown_has_no_continuous_estimate_and_full_binary_bounds(self):
        result = self.run_values([None] * 128)
        self.assertEqual(
            self.primary(result),
            {
                "planned_contexts": 32,
                "resolved_contexts": 0,
                "point": None,
                "lower": None,
                "upper": None,
                "unbounded": True,
                "planned_cores": 8,
                "resolved_cores": 0,
            },
        )
        for value in result["analysis"]["patterns"].values():
            self.assertEqual(value["counts"], {"true": 0, "false": 0, "unknown": 8})
            self.assertEqual(value["rate"], {"point": None, "lower": 0.0, "upper": 1.0})
            self.assertIsNone(value["qualification"])

    def test_hand_calculated_primary_components_and_pattern(self):
        result = self.run_values(measurements())
        self.assertEqual(self.primary(result)["point"], 4.0)
        effects = result["analysis"]["contrasts"]["margin_nats"]["by_rule"]
        self.assertEqual((effects["copy"]["point"], effects["recompute"]["point"]), (4.0, 0.0))
        for item in result["analysis"]["patterns"].values():
            self.assertEqual(item["counts"], {"true": 8, "false": 0, "unknown": 0})
            self.assertTrue(item["qualification"])
        for record in result["records"]:
            self.assertTrue(record["instruction_compliant_preferred"])
            self.assertIs(
                record["arithmetic_truth_preferred"],
                (record["rule"], record["condition"]) != ("copy", "foil_cue"),
            )

    def test_signed_continuous_effect_and_behavioral_patterns_are_separate(self):
        cases = [
            ((3.0, 1.0, 2.0, 2.0), 2.0, False),
            ((3.0, -1.0, 6.0, 2.0), 0.0, True),
            ((2.0, -1.0, 6.0, 1.0), -2.0, True),
        ]
        for margins, expected, joint in cases:
            result = self.run_values(measurements(margins))
            self.assertEqual(self.primary(result)["point"], expected)
            self.assertIs(result["analysis"]["patterns"]["joint"]["qualification"], joint)

    def test_equal_core_weights_and_cell_means(self):
        result = self.run_values(measurements(lambda core: (float(core + 1), -1.0, 2.0, 2.0)))
        self.assertEqual(self.primary(result)["point"], 5.5)
        cells = result["analysis"]["by_rule_cue"]
        self.assertEqual(cells["copy"]["true_cue"]["point"], 4.5)
        self.assertEqual(cells["copy"]["foil_cue"]["point"], -1.0)
        self.assertTrue(
            all(v["planned_contexts"] == 8 for group in cells.values() for v in group.values())
        )

    def test_all_256_four_sign_and_missing_states(self):
        for margins in itertools.product((None, -1.0, 0.0, 1.0), repeat=4):
            result = self.run_values(measurements(margins))
            for name, indices, signs in (
                ("copying_switch", (0, 1), (1, -1)),
                ("recompute_truth", (2, 3), (1, 1)),
                ("joint", (0, 1, 2, 3), (1, -1, 1, 1)),
            ):
                chosen = [margins[i] for i in indices]
                bad = any(v is not None and v * s <= 0 for v, s in zip(chosen, signs, strict=True))
                good = all(v is not None for v in chosen)
                expected = False if bad else True if good else None
                got = result["analysis"]["patterns"][name]
                self.assertIs(got["qualification"], expected)
                self.assertEqual(got["resolved_cores"], 8 if expected is not None else 0)
            self.assertEqual(
                self.primary(result)["resolved_cores"], 8 if None not in margins else 0
            )

    def test_known_failure_can_resolve_binary_with_continuous_unknown(self):
        result = self.run_values(measurements((0.0, None, None, None)))
        self.assertIsNone(self.primary(result)["point"])
        for name in ("copying_switch", "joint"):
            self.assertEqual(
                result["analysis"]["patterns"][name]["rate"],
                {"point": 0.0, "lower": 0.0, "upper": 0.0},
            )
        self.assertIsNone(result["analysis"]["patterns"]["recompute_truth"]["qualification"])

    def test_missing_rule_preserves_other_rule_component(self):
        result = self.run_values(measurements((None, None, 3.0, 1.0)))
        effects = result["analysis"]["contrasts"]["margin_nats"]["by_rule"]
        self.assertEqual(effects["recompute"]["point"], 2.0)
        self.assertIsNone(effects["copy"]["point"])
        self.assertIsNone(self.primary(result)["point"])

    def test_mixed_pattern_bounds_enumerate_all_unknown_completions(self):
        def margins(core):
            return (
                (3.0, -1.0, 2.0, 2.0)
                if core < 2
                else (0.0, -1.0, 2.0, 2.0)
                if core < 5
                else (None, -1.0, 2.0, 2.0)
            )

        result = self.run_values(measurements(margins))
        possibilities = [(2 + sum(bits)) / 8 for bits in itertools.product((0, 1), repeat=3)]
        for name in ("copying_switch", "joint"):
            got = result["analysis"]["patterns"][name]
            self.assertEqual(got["counts"], {"true": 2, "false": 3, "unknown": 3})
            self.assertEqual(
                got["rate"],
                {"point": None, "lower": min(possibilities), "upper": max(possibilities)},
            )
            self.assertFalse(got["qualification"])
        self.assertIsNone(self.primary(result)["point"])

    def test_every_single_missing_slot_preserves_fixed_cohort(self):
        for index in range(128):
            values = measurements()
            values[index] = None
            result = self.run_values(values)
            primary = self.primary(result)
            self.assertEqual((primary["resolved_contexts"], primary["resolved_cores"]), (31, 7))
            self.assertEqual(
                (primary["point"], primary["lower"], primary["upper"]), (None, None, None)
            )
            self.assertTrue(primary["unbounded"])

    def test_all_local_repeat_presence_patterns(self):
        for present in itertools.product((False, True), repeat=4):
            values = measurements()
            for index, exists in enumerate(present):
                if not exists:
                    values[index] = None
            got = self.run_values(values)["records"][0]
            self.assertEqual(got["measurement_count"], sum(present))
            self.assertIs(got["numerical_valid"], True if all(present) else None)

    def test_complete_all_calls_can_have_no_valid_margin(self):
        values = measurements()
        for index, slot in enumerate(a.reconstruct_schedule()):
            if slot["repeat_index"] == 1:
                values[index]["logits_sha256"] = hashlib.sha256(
                    f"changed|{index}".encode()
                ).hexdigest()
        result = self.run_values(values)
        self.assertEqual(result["analysis"]["resolved_measurements"], 128)
        self.assertEqual(result["analysis"]["overall"]["numerical_valid"]["false"], 32)
        self.assertIsNone(self.primary(result)["point"])
        self.assertIsNone(result["analysis"]["patterns"]["joint"]["qualification"])

    def test_equal_hash_different_score_is_rejected_without_tolerance(self):
        values = measurements()
        values[0]["score"] = math.nextafter(values[0]["score"], -math.inf)
        with self.assertRaises(AssertionError):
            self.run_values(values)
        values = measurements((0.0, 0.0, 0.0, 0.0))
        values[0]["score"], values[3]["score"] = -0.0, 0.0
        with self.assertRaises(AssertionError):
            self.run_values(values)

    def test_known_repeat_failure_dominates_another_missing_repeat(self):
        values = measurements()
        values[0]["logits_sha256"] = "f" * 64
        values[1] = None
        got = self.run_values(values)["records"][0]
        self.assertFalse(got["numerical_valid"])
        self.assertIsNone(got["margin_nats"])

    def test_literal_small_signs_and_exact_ties(self):
        values = measurements((0.0, 0.0, 0.0, 0.0))
        for index, slot in enumerate(a.reconstruct_schedule()):
            if slot["candidate"] == "arithmetic_truth":
                destination = (
                    -math.inf if (slot["rule"], slot["condition"]) == ("copy", "foil_cue") else 0.0
                )
                values[index]["score"] = math.nextafter(-100.0, destination)
        self.assertTrue(self.run_values(values)["analysis"]["patterns"]["joint"]["qualification"])
        self.assertFalse(
            self.run_values(measurements((0.0, 0.0, 0.0, 0.0)))["analysis"]["patterns"]["joint"][
                "qualification"
            ]
        )

    def test_invalid_measurement_types_finiteness_and_cardinality(self):
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
            with self.assertRaises(AssertionError):
                self.run_values(values)
        for count in (64, 127, 129):
            with self.assertRaises(AssertionError):
                self.run_values([None] * count)

    def test_nonfinite_interaction_is_rejected(self):
        values = measurements()
        for index, slot in enumerate(a.reconstruct_schedule()):
            positive = (slot["rule"], slot["condition"]) in (
                ("copy", "true_cue"),
                ("recompute", "foil_cue"),
            )
            selected = (slot["candidate"] == "arithmetic_truth") == positive
            values[index]["score"] = -1.0 if selected else -1.7e308
        with self.assertRaises(AssertionError):
            self.run_values(values)

    def test_aggregate_wrapper_exact_identity_status_and_privacy(self):
        packet = a.audit_outcomes(plan(), records(measurements()))
        self.assertEqual(set(packet), {"record_slots", "context_slots", "analysis"})
        self.assertEqual((packet["record_slots"], packet["context_slots"]), (128, 32))
        for key in (
            "messages",
            "observation_id",
            "evaluation_id",
            "logits_sha256",
            "expected_sums",
            "supplied_answers",
            "paired_switch",
            "supplied_qualification",
        ):
            self.assertNotIn(key, a.canonical(packet).decode())
        for key, value in (
            ("evaluation_index", False),
            ("evaluation_id", "other"),
            ("status", "failed"),
            ("extra", 0),
        ):
            rows = records(measurements())
            rows[0][key] = value
            with self.assertRaises(AssertionError):
                a.audit_outcomes(plan(), rows)
        rows = records([None] * 128)
        rows[0]["status"] = "failed"
        self.assertEqual(a.audit_outcomes(plan(), rows)["analysis"]["resolved_measurements"], 0)
        rows[0]["status"] = "completed"
        with self.assertRaises(AssertionError):
            a.audit_outcomes(plan(), rows)


if __name__ == "__main__":
    unittest.main()
