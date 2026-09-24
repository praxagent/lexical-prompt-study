"""Invented A195 native-text, arithmetic and durable-receipt controls; no real model."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace
import struct
import tempfile
import unittest
from unittest.mock import patch

from lexical_prompt_study import canonical_answer_ranking_acquisition as core


class Tokenizer:
    chat_template = "invented-template"
    all_special_ids = [14, 15]

    def __init__(self):
        self.added_tokens_decoder = {i: SimpleNamespace(special=True) for i in (13, 14, 15, 16, 17)}
        self.added_tokens_decoder[22] = SimpleNamespace(special=False)
        self.direct_calls = []
        self.prompts = {}
        self.decoded = {}

    def convert_tokens_to_ids(self, token):
        assert token == core.EOT
        return 13

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, date_string, return_dict=None
    ):
        if tokenize:
            assert return_dict is False
            self.direct_calls.append(add_generation_prompt)
        else:
            assert return_dict is None
        assert date_string == "24 Sep 2026"
        prompt = "HEADER" + repr(messages[:2]) + "ASSISTANT"
        if prompt not in self.prompts:
            self.prompts[prompt] = [14, 32 + len(self.prompts)]
        text = prompt if add_generation_prompt else prompt + messages[-1]["content"] + core.EOT
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

    def encode(self, text, *, add_special_tokens):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        alphabet = "-,0123456789"
        if text == core.EOT:
            ids = [13]
        elif text in self.prompts:
            ids = self.prompts[text]
        elif text.endswith(core.EOT):
            prompt = next(p for p in self.prompts if text.startswith(p))
            tail = text[len(prompt) : -len(core.EOT)]
            target = [alphabet.index(x) + 1 for x in tail] + [13]
            self.decoded[tuple(target)] = tail + core.EOT
            ids = self.prompts[prompt] + target
        else:
            ids = (
                [alphabet.index(x) + 1 for x in text]
                if all(x in alphabet for x in text)
                else [20, 21]
            )
        self.decoded[tuple(ids)] = text
        return {"input_ids": list(ids)}

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return self.decoded[tuple(ids)]


def plan_and_prepared():
    tokenizer = Tokenizer()
    binding = {"schema": "invented"}
    plan = core.compile_plan(
        provenance={
            "model_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "chat_template_sha256": core.e.sha256(tokenizer.chat_template.encode()),
        },
        geometry={"vocab_size": 64, "hidden_width": 8, "model_layers": 1, "context_limit": 512},
        eos_token_ids=[13, 15, 16],
        runtime_binding_sha256=core.e.object_hash(binding),
        protocol_sha256="c" * 64,
        tests_sha256="d" * 64,
    )
    return plan, core.prepare_inputs(plan, tokenizer), binding


def receipt_forward(model, plan, directory, attempt, counts):
    directory.mkdir()
    core._write(directory / "attempt.json", attempt)
    counts["scoring"] += 1
    entry = {
        "schema_version": core.SCHEMA,
        "event": "model_forward_pre_hook_entry",
        "attempt_sha256": core.e.object_hash(attempt),
        "entry_index": 0,
    }
    core._write(directory / "entry.json", entry)
    # Independent uniform-row likelihood: score = -target length * log(vocab).
    raw = bytes(4 * plan["geometry"]["vocab_size"] * len(attempt["target_token_ids"]))
    core.e._publish(directory / "logits.fp32", raw)
    result = {
        "schema_version": core.SCHEMA,
        "status": "completed",
        "attempt_sha256": core.e.object_hash(attempt),
        "entry_sha256": core.e.object_hash(entry),
        "entries": 1,
        "model_call_returned": True,
        "logits_sha256": core.e.sha256(raw),
        **core.score_readout(raw, attempt["target_token_ids"], plan["geometry"]["vocab_size"]),
    }
    core._write(directory / "result.json", result)
    return result


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.plan, self.prepared, self.binding = plan_and_prepared()

    def run_fixture(self, forward=receipt_forward):
        with patch.object(core, "_forward", forward):
            return core.run_acquisition(
                None,
                None,
                plan=self.plan,
                prepared=self.prepared,
                runtime_binding=self.binding,
                directory=self.root,
            )

    def test_fixed_native_geometry_and_shift(self):
        self.assertEqual(len(self.plan["schedule"]), 64)
        for slot in self.plan["schedule"]:
            a = core._attempt(self.plan, self.prepared, "f" * 64, slot)
            obs = self.prepared["observations"][slot["sequence_index"]]
            target = obs["candidates"][slot["candidate"]]["target_token_ids"]
            self.assertEqual(a["input_token_ids"], obs["prompt_token_ids"] + target[:-1])
            self.assertEqual(a["logits_to_keep"], len(target))
            self.assertEqual(a["first_scored_absolute_position"], len(obs["prompt_token_ids"]) - 1)
            self.assertEqual(target[-1], 13)

    def test_native_snapshots(self):
        original = core.e.object_hash(self.plan)
        clone = core.validate_plan(self.plan)
        clone["rows"][0]["messages"][0]["content"] = "changed"
        self.assertEqual(core.e.object_hash(self.plan), original)

    def test_logsumexp_uniform_no_length_normalization(self):
        result = core.score_readout(bytes(4 * 5 * 3), [0, 2, 4], 5)
        self.assertEqual(result["scored_tokens_including_eot"], 3)
        self.assertAlmostEqual(result["score"], -3 * math.log(5), places=14)

    def test_large_equal_logits_avoid_normalizer_cancellation(self):
        result = core.score_readout(struct.pack("<ff", 3e38, 3e38), [1], 2)
        self.assertEqual(result["score"], -math.log(2))
        self.assertNotEqual(result["score"], 0)

    def test_analytic_nonuniform_rows_and_eot(self):
        raw = struct.pack("<ffffff", 0, 0, 0, 0, math.log(2), 0)
        result = core.score_readout(raw, [0, 1], 3)
        widened = struct.unpack("<f", struct.pack("<f", math.log(2)))[0]
        expected = -math.log(3) + widened - math.log(2 + math.exp(widened))
        self.assertAlmostEqual(result["score"], expected, places=14)
        self.assertEqual(len(result["token_logprobs"]), 2)

    def test_extreme_opposite_finite_values(self):
        result = core.score_readout(struct.pack("<ff", -3e38, 3e38), [0], 2)
        self.assertTrue(math.isfinite(result["score"]))
        self.assertLess(result["score"], -5e38)

    def test_complete_all64_and_model_free_replay(self):
        result = self.run_fixture()
        self.assertEqual(
            result["summary"],
            {
                "planned_measurements": 64,
                "completed_measurements": 64,
                "failed_measurements": 0,
                "unattempted_measurements": 0,
                "scoring_entries": 64,
                "total_entries": 64,
                "entries_complete": True,
            },
        )
        self.assertEqual(core.load_acquisition(self.root, self.plan), result)
        self.assertEqual(len(result["records"]), 64)
        self.assertEqual(len(result["context_records"]), 16)
        with self.assertRaises((FileExistsError, ValueError)):
            self.run_fixture()

    def test_event_science_once(self):
        events = []
        with patch.object(core, "_forward", receipt_forward):
            core.run_acquisition(
                None,
                None,
                plan=self.plan,
                prepared=self.prepared,
                runtime_binding=self.binding,
                directory=self.root,
                event=events.append,
            )
        self.assertEqual(events, ["science"])

    def test_repeat_mismatch_retains_all64(self):
        def forward(model, plan, directory, attempt, counts):
            result = receipt_forward(model, plan, directory, attempt, counts)
            if attempt["evaluation_index"] == 3:
                raw = struct.pack("<f", 1.0) * (len(attempt["target_token_ids"]) * 64)
                (directory / "logits.fp32").write_bytes(raw)
                result.update(
                    logits_sha256=core.e.sha256(raw),
                    **core.score_readout(raw, attempt["target_token_ids"], 64),
                )
                (directory / "result.json").write_bytes(core.e.canonical(result))
            return result

        result = self.run_fixture(forward)
        self.assertEqual(result["summary"]["completed_measurements"], 64)
        self.assertFalse(result["context_records"][0]["numerical_valid"])
        self.assertIsNone(
            result["analysis"]["contrasts"]["margin_nats"]["supplied_minus_computed"]["point"]
        )

    def test_fail_after_three_no_replacement(self):
        marker = KeyboardInterrupt("invented")

        def forward(*args):
            if args[3]["evaluation_index"] == 3:
                raise marker
            return receipt_forward(*args)

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_fixture(forward)
        self.assertIs(caught.exception, marker)
        value = core.load_acquisition(self.root, self.plan)
        self.assertEqual(value["summary"]["completed_measurements"], 3)
        self.assertEqual(value["summary"]["failed_measurements"], 1)
        self.assertEqual(value["summary"]["unattempted_measurements"], 60)
        self.assertEqual(value["summary"]["total_entries"], 3)

    def test_repeat_uses_bytes_even_under_invented_hash_collision(self):
        real_hash = core.e.sha256

        def collision(raw):
            if (
                raw
                and len(raw) % 4 == 0
                and (raw == bytes(len(raw)) or raw == struct.pack("<f", 1.0) * (len(raw) // 4))
            ):
                return "f" * 64
            return real_hash(raw)

        def forward(model, plan, directory, attempt, counts):
            result = receipt_forward(model, plan, directory, attempt, counts)
            if attempt["evaluation_index"] == 3:
                raw = struct.pack("<f", 1.0) * (len(attempt["target_token_ids"]) * 64)
                (directory / "logits.fp32").write_bytes(raw)
                result.update(
                    logits_sha256=core.e.sha256(raw),
                    **core.score_readout(raw, attempt["target_token_ids"], 64),
                )
                (directory / "result.json").write_bytes(core.e.canonical(result))
            return result

        with (
            patch.object(core.e, "sha256", collision),
            self.assertRaisesRegex(ValueError, "exact_repeat_bytes"),
        ):
            self.run_fixture(forward)

    def test_orphan_completed_forward_never_promoted(self):
        original = core._write

        def write(path, value):
            if path.name == "return.json":
                raise SystemExit("invented")
            return original(path, value)

        with patch.object(core, "_write", write), self.assertRaises(SystemExit):
            self.run_fixture()
        value = core.load_acquisition(self.root, self.plan)
        self.assertEqual(value["summary"]["completed_measurements"], 0)
        self.assertIsNone(value["summary"]["total_entries"])
        self.assertEqual(value["summary"]["total_entries_lower_bound"], 1)
        self.assertIsNone(value["records"][0]["measurement"])

    def test_durable_normal_return_survives_later_interrupt(self):
        original = core._write

        def write(path, value):
            original(path, value)
            if path.name == "return.json":
                raise KeyboardInterrupt("after publication")

        with patch.object(core, "_write", write), self.assertRaises(KeyboardInterrupt):
            self.run_fixture()
        value = core.load_acquisition(self.root, self.plan)
        self.assertEqual(value["summary"]["completed_measurements"], 1)
        self.assertEqual(value["summary"]["total_entries"], 1)
        self.assertIsNotNone(value["records"][0]["measurement"])

    def test_repeated_callback_lost_failure_is_unknown(self):
        def forward(model, plan, directory, attempt, counts):
            directory.mkdir()
            core._write(directory / "attempt.json", attempt)
            core._write(
                directory / "entry.json",
                {
                    "schema_version": core.SCHEMA,
                    "event": "model_forward_pre_hook_entry",
                    "attempt_sha256": core.e.object_hash(attempt),
                    "entry_index": 0,
                },
            )
            counts["scoring"] += 2
            raise RuntimeError("second callback before body")

        with self.assertRaises(RuntimeError):
            self.run_fixture(forward)
        value = core.load_acquisition(self.root, self.plan)
        self.assertIsNone(value["summary"]["total_entries"])
        self.assertEqual(value["summary"]["total_entries_lower_bound"], 1)
        self.assertEqual(value["reported_summary"]["total_entries"], 2)

    def test_missing_attempt_with_orphan_rejected(self):
        slot = self.plan["schedule"][0]
        attempt = core._attempt(self.plan, self.prepared, "a" * 64, slot)
        self.root.mkdir()
        (self.root / "logits.fp32").write_bytes(b"invented")
        with self.assertRaises(ValueError):
            core._partial_count(self.root, attempt)

    def test_partial_cannot_prove_return_or_second_entry_without_first_receipt(self):
        self.root.mkdir()
        attempt = core._attempt(self.plan, self.prepared, "a" * 64, self.plan["schedule"][0])
        core._write(self.root / "attempt.json", attempt)
        failure = {
            "schema_version": core.SCHEMA,
            "attempt_sha256": core.e.object_hash(attempt),
            "entries": 1,
            "model_call_returned": True,
            "cleanup_failure_type": None,
            "failure_type": "Interrupted",
            "usable": False,
        }
        path = self.root / "failure.json"
        path.write_bytes(core.e.canonical(failure))
        with self.assertRaises(ValueError):
            core._partial_count(self.root, attempt)
        failure.update(entries=2, model_call_returned=False)
        path.write_bytes(core.e.canonical(failure))
        with self.assertRaises(ValueError):
            core._partial_count(self.root, attempt)

    def test_unpublished_attempt_only_zero(self):
        self.root.mkdir()
        (self.root / ".attempt.json.tmp").write_bytes(b"invented")
        self.assertEqual(core._partial_count(self.root, {}), (0, True))

    def test_extra_root_file_rejected(self):
        self.run_fixture()
        (self.root / "extra").write_bytes(b"invented")
        with self.assertRaises(ValueError):
            core.load_acquisition(self.root, self.plan)

    def test_readout_tamper_rejected(self):
        self.run_fixture()
        path = self.root / "calls/000/logits.fp32"
        raw = bytearray(path.read_bytes())
        raw[0] = 1
        path.write_bytes(raw)
        with self.assertRaises(ValueError):
            core.load_acquisition(self.root, self.plan)

    def test_bool_declared_failure_count_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_fixture(lambda *args: (_ for _ in ()).throw(SystemExit("invented")))
        value = core._read(self.root / "failure.json")
        value["summary"]["scoring_entries"] = False
        value["summary"]["total_entries"] = False
        (self.root / "failure.json").write_bytes(core.e.canonical(value))
        with self.assertRaises(ValueError):
            core.load_acquisition(self.root, self.plan)

    def test_symlink_inventory_rejected(self):
        self.root.mkdir()
        (self.root / "link").symlink_to(Path(self.temp.name) / "missing")
        with self.assertRaises(ValueError):
            core._inventory(self.root)


def bad_score(raw, ids, vocab):
    def test(self):
        with self.assertRaises(ValueError):
            core.score_readout(raw, ids, vocab)

    return test


for name, raw, ids, vocab in (
    ("nan", struct.pack("<I", 0x7FC00000), [0], 1),
    ("inf", struct.pack("<I", 0x7F800000), [0], 1),
    ("neginf", struct.pack("<I", 0xFF800000), [0], 1),
    ("size", bytes(7), [0], 2),
    ("mutable", bytearray(8), [0], 2),
    ("bool_id", bytes(8), [True], 2),
    ("out_of_range", bytes(8), [2], 2),
    ("bool_vocab", bytes(4), [0], True),
):
    setattr(CoreTests, "test_bad_score_" + name, bad_score(raw, ids, vocab))


def bad_prepared(change):
    def test(self):
        value = copy.deepcopy(self.prepared)
        obs = value["observations"][0]
        if change == "eot_bool":
            value["eot_token_id"] = True
        elif change == "special_bool":
            value["special_token_ids"][0] = True
        elif change == "payload_special":
            obs["payload_token_ids"]["user"] = [13]
            obs["payload_ids_sha256"] = core.e.object_hash(obs["payload_token_ids"])
        elif change == "paired_target":
            obs["candidates"]["correct"]["target_token_ids"][0] = 20
            obs["candidates"]["correct"]["target_ids_sha256"] = core.e.object_hash(
                obs["candidates"]["correct"]["target_token_ids"]
            )
        elif change == "early_special":
            obs["candidates"]["correct"]["target_token_ids"][0] = 14
            obs["candidates"]["correct"]["target_ids_sha256"] = core.e.object_hash(
                obs["candidates"]["correct"]["target_token_ids"]
            )
        elif change == "last_not_eot":
            obs["candidates"]["correct"]["target_token_ids"][-1] = 12
        elif change == "prompt_limit":
            obs["prompt_token_ids"] = [20] * 257
            obs["prompt_ids_sha256"] = core.e.object_hash(obs["prompt_token_ids"])
        elif change == "target_limit":
            obs["candidates"]["correct"]["target_token_ids"] = [1] * 16 + [13]
            obs["candidates"]["correct"]["target_ids_sha256"] = core.e.object_hash(
                obs["candidates"]["correct"]["target_token_ids"]
            )
        with self.assertRaises(ValueError):
            core.validate_prepared(self.plan, value)

    return test


for change in (
    "eot_bool",
    "special_bool",
    "payload_special",
    "paired_target",
    "early_special",
    "last_not_eot",
    "prompt_limit",
    "target_limit",
):
    setattr(CoreTests, "test_prepared_reject_" + change, bad_prepared(change))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.prepared, _ = plan_and_prepared()

    def failed(self, tokenizer, stage, code, original="ValueError"):
        with self.assertRaises(core.PreparationError) as caught:
            core.prepare_inputs(self.plan, tokenizer)
        self.assertEqual(
            core.preparation_failure(caught.exception),
            {
                "stage": stage,
                "code": code,
                "original_error_type": original,
            },
        )
        return caught.exception

    def test_named_and_backend_union_handles_three_stops(self):
        snapshot = self.prepared["special_inventory"]
        self.assertEqual(len(snapshot["declared_special_ids"]), 2)
        self.assertEqual(len(self.plan["eos_token_ids"]), 3)
        self.assertEqual(
            len(set(snapshot["declared_special_ids"]) & set(self.plan["eos_token_ids"])), 1
        )
        self.assertEqual(snapshot["special_ids"], [13, 14, 15, 16, 17])
        self.assertNotIn(22, snapshot["special_ids"])
        self.assertEqual(self.prepared["terminal_metadata"]["terminal_token_id"], 13)

    def test_direct_templates_explicit_false_and_singleton_terminal(self):
        tokenizer = Tokenizer()
        core.prepare_inputs(self.plan, tokenizer)
        self.assertEqual(tokenizer.direct_calls.count(True), 16)
        self.assertEqual(tokenizer.direct_calls.count(False), 32)
        self.assertEqual(tokenizer.decoded[(13,)], core.EOT)

    def test_defensive_inventory_snapshot(self):
        tokenizer = Tokenizer()
        prepared = core.prepare_inputs(self.plan, tokenizer)
        tokenizer.added_tokens_decoder[13].special = False
        self.assertEqual(core.validate_prepared(self.plan, prepared), prepared)
        self.assertIn(13, prepared["special_inventory"]["added_special_ids"])

    def test_backend_only_payload_special_rejected(self):
        tokenizer = Tokenizer()
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kw: [17] if text != core.EOT else original(text, **kw)
        self.failed(tokenizer, "payload_encoding", "payload_special_token")

    def test_terminal_missing_from_union_rejected(self):
        tokenizer = Tokenizer()
        tokenizer.added_tokens_decoder[13].special = False
        self.failed(tokenizer, "terminal_identity", "operation_exception")

    def test_nonboolean_backend_flag_rejected(self):
        tokenizer = Tokenizer()
        tokenizer.added_tokens_decoder[13].special = 1
        self.failed(tokenizer, "special_inventory_before", "operation_exception")

    def test_bool_terminal_identity_rejected(self):
        tokenizer = Tokenizer()
        tokenizer.convert_tokens_to_ids = lambda _: True
        self.failed(tokenizer, "terminal_identity", "native_eot")

    def test_literal_terminal_encoding_mismatch(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *args, **kw: [15]
        self.failed(tokenizer, "terminal_identity", "literal_eot_encoding")

    def test_literal_terminal_decode_mismatch(self):
        tokenizer = Tokenizer()
        tokenizer.decode = lambda *args, **kw: "invented mismatch"
        self.failed(tokenizer, "terminal_identity", "literal_eot_roundtrip")

    def test_encoding_must_return_list(self):
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *args, **kw: (13,)
        self.failed(tokenizer, "terminal_identity", "encoded_ids")

    def test_native_direct_batch_dict_rejected(self):
        tokenizer = Tokenizer()
        original = tokenizer.apply_chat_template

        def apply(*args, **kw):
            value = original(*args, **kw)
            return {"input_ids": value} if kw["tokenize"] else value

        tokenizer.apply_chat_template = apply
        self.failed(tokenizer, "prompt_construction", "native_prompt_ids")

    def test_native_direct_boolean_id_rejected(self):
        tokenizer = Tokenizer()
        original = tokenizer.apply_chat_template

        def apply(*args, **kw):
            value = original(*args, **kw)
            return [True, *value[1:]] if kw["tokenize"] else value

        tokenizer.apply_chat_template = apply
        self.failed(tokenizer, "prompt_construction", "native_prompt_ids")

    def test_closed_template_suffix_mismatch(self):
        tokenizer = Tokenizer()
        original = tokenizer.apply_chat_template

        def apply(*args, **kw):
            value = original(*args, **kw)
            return (
                value + "extra" if not kw["add_generation_prompt"] and not kw["tokenize"] else value
            )

        tokenizer.apply_chat_template = apply
        self.failed(tokenizer, "candidate_construction", "native_assistant_completion")

    def test_inventory_mutation_detected_after_all_preparation(self):
        tokenizer = Tokenizer()
        original = tokenizer.decode

        def decode(ids, **kw):
            value = original(ids, **kw)
            if len(ids) > 1:
                tokenizer.added_tokens_decoder[18] = SimpleNamespace(special=True)
            return value

        tokenizer.decode = decode
        self.failed(tokenizer, "special_inventory_after", "special_inventory_changed")
        self.assertEqual(len(tokenizer.direct_calls), 48)

    def test_template_mutation_detected_after_preparation(self):
        tokenizer = Tokenizer()
        original = tokenizer.decode

        def decode(ids, **kw):
            value = original(ids, **kw)
            if len(ids) > 1:
                tokenizer.chat_template = "changed"
            return value

        tokenizer.decode = decode
        self.failed(tokenizer, "template_after", "chat_template")

    def test_internal_prepared_cap_has_finite_contract_code(self):
        tokenizer = Tokenizer()
        original = tokenizer.apply_chat_template

        def apply(*args, **kw):
            value = original(*args, **kw)
            if not kw["tokenize"] and kw["add_generation_prompt"]:
                tokenizer.prompts[value] = [20] * 257
            return value

        tokenizer.apply_chat_template = apply
        self.failed(tokenizer, "prepared_validation", "prepared_contract")

    def test_external_error_type_and_cause_without_message(self):
        tokenizer = Tokenizer()
        marker = LookupError("invented private payload")

        def encode(*args, **kw):
            raise marker

        tokenizer.encode = encode
        error = self.failed(tokenizer, "terminal_identity", "operation_exception", "LookupError")
        self.assertIs(error.__cause__, marker)
        self.assertNotIn("private", str(error))
        self.assertNotIn("private", repr(core.preparation_failure(error)))

    def test_unsafe_exception_type_sanitized(self):
        tokenizer = Tokenizer()
        marker = type("Private\npayload", (ValueError,), {})("hidden")

        def encode(*args, **kw):
            raise marker

        tokenizer.encode = encode
        self.failed(tokenizer, "terminal_identity", "operation_exception", "UnclassifiedException")

    def test_interruptions_preserve_original_identity(self):
        for cls in (KeyboardInterrupt, SystemExit):
            with self.subTest(cls=cls):
                tokenizer = Tokenizer()
                marker = cls("private")

                def encode(*args, **kw):
                    raise marker

                tokenizer.encode = encode
                with self.assertRaises(cls) as caught:
                    core.prepare_inputs(self.plan, tokenizer)
                self.assertIs(caught.exception, marker)
                self.assertEqual(
                    core.preparation_failure(marker),
                    {
                        "stage": "unclassified",
                        "code": "operation_exception",
                        "original_error_type": cls.__name__,
                    },
                )

    def test_preparation_metadata_finite_and_typed(self):
        for args in (
            ("other", "operation_exception", "ValueError"),
            ("unclassified", "private", "ValueError"),
            ("unclassified", "operation_exception", "Bad\nType"),
            (True, "operation_exception", "ValueError"),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                core.PreparationError(*args)
        error = core.PreparationError("terminal_identity", "native_eot", "ValueError")
        error.stage = "private"
        with self.assertRaises(ValueError):
            core.preparation_failure(error)

    def test_prepared_inventory_and_terminal_replay_tampering(self):
        for change in ("flag", "terminal", "union", "width"):
            with self.subTest(change=change):
                prepared = copy.deepcopy(self.prepared)
                if change == "flag":
                    prepared["special_inventory"]["added_tokens"][0]["special"] = 1
                elif change == "terminal":
                    prepared["terminal_metadata"]["stop_token_ids"] = [13]
                elif change == "union":
                    prepared["special_token_ids"] = [13, 14, 15, 16]
                else:
                    prepared["special_inventory"]["vocab_size"] = 65
                with self.assertRaises(ValueError):
                    core.validate_prepared(self.plan, prepared)


if __name__ == "__main__":
    unittest.main()
