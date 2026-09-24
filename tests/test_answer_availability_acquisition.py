"""Invented A191 coordinator controls, with a separately selected native class.

Run PureAcquisitionTests without Torch. NativeTinyFP32Tests imports Torch only
in setUpClass and is reserved for root's dedicated untrained-model qualification.
No pretrained model, native tokenizer, or historical evidence is used.
"""

from contextlib import contextmanager
import copy
import inspect
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

from lexical_prompt_study import answer_availability_acquisition as a


RUNTIME = {"regime": "invented_cpu_fp32_eight_threads"}


def make_plan(eos=63):
    return a.compile_plan(
        provenance={
            "model_sha256": "1" * 64,
            "tokenizer_sha256": "2" * 64,
            "chat_template_sha256": "3" * 64,
        },
        geometry={"vocab_size": 64, "hidden_width": 16, "model_layers": 1, "context_limit": 128},
        eos_token_ids=[eos],
        runtime_binding_sha256=a.e.object_hash(RUNTIME),
        protocol_sha256="4" * 64,
        tests_sha256="5" * 64,
    )


class InventedTokenizer:
    """A table-valued test double; its invented token is an entire answer string."""

    def __init__(self, plan):
        self.rows = plan["rows"]
        self.texts = {}
        self.decode_calls = []
        self.template_calls = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, date_string, **kwargs
    ):
        if tokenize is not False or add_generation_prompt is not True or date_string != a.DATE:
            raise AssertionError("unexpected_invented_template_policy")
        text = "<invented>" + messages[0]["content"] + "\n" + messages[1]["content"] + "<assistant>"
        index = next(i for i, row in enumerate(self.rows) if row["messages"] == messages)
        self.texts[text] = [1, 2, index + 3]
        self.template_calls.append((tokenize, add_generation_prompt, date_string))
        return text

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens is not False:
            raise AssertionError("unexpected_special_token_insertion")
        return {"input_ids": list(self.texts[text])}

    def decode(self, ids, **kwargs):
        if kwargs != a.e.DECODER:
            raise AssertionError("unexpected_decode_policy")
        self.decode_calls.append(list(ids))
        if not ids:
            return ""
        if len(ids) == 1 and 16 <= ids[0] <= 31:
            row = self.rows[ids[0] - 16]
            return ",".join(map(str, row["expected_sums"]))
        return "invented nonanswer"


class InventedDispatcher:
    """Write actual primitive-shaped FP32 artifacts without importing Torch."""

    def __init__(
        self,
        *,
        cap=False,
        mismatch=False,
        fail_at=None,
        failure=KeyboardInterrupt,
        publish_entry=True,
        count_entry=True,
        publish_failure=True,
    ):
        self.cap, self.mismatch, self.fail_at = cap, mismatch, fail_at
        self.failure, self.publish_entry, self.count_entry = failure, publish_entry, count_entry
        self.publish_failure = publish_failure
        self.calls = []

    def __call__(self, model, plan, directory, attempt, counts):
        stage = attempt["stage"]
        index, step = attempt["sequence_index"], attempt["step_index"]
        self.calls.append((stage, index, step, list(attempt["input_token_ids"])))
        root = a.e._directory(directory, create=True)
        a._write(root / "attempt.json", attempt)
        failed = self.fail_at == (stage, index, step)
        if not failed or self.count_entry:
            counts[stage]["generation"] += 1
        entry = {
            "schema_version": a.primitive.SCHEMA,
            "event": "model_forward_pre_hook_entry",
            "attempt_sha256": a.e.object_hash(attempt),
            "entry_index": 0,
        }
        if not failed or self.publish_entry:
            a._write(root / "entry.json", entry)
        if failed:
            if self.publish_failure:
                a._write(
                    root / "failure.json",
                    {
                        "schema_version": a.primitive.SCHEMA,
                        "attempt_sha256": a.e.object_hash(attempt),
                        "entries": int(self.count_entry),
                        "model_call_returned": False,
                        "cleanup_failure_type": None,
                        "failure_type": self.failure.__name__,
                        "usable": False,
                    },
                )
            raise self.failure("invented_dispatch_failure")
        chosen = (
            (2 if self.mismatch and step == 1 else 1)
            if stage == "technical"
            else (7 if self.cap else 16 + index if step == 0 else plan["eos_token_ids"][0])
        )
        values = [-1.0] * plan["geometry"]["vocab_size"]
        values[chosen] = 1.0
        raw = struct.pack("<" + "f" * len(values), *values)
        a.e._publish(root / "logits.fp32", raw)
        result = {
            "schema_version": a.primitive.SCHEMA,
            "status": "completed",
            "attempt_sha256": a.e.object_hash(attempt),
            "entry_sha256": a.e.object_hash(entry),
            "entries": 1,
            "model_call_returned": True,
            "logits_sha256": a.e.sha256(raw),
            "chosen_token_id": chosen,
        }
        a._write(root / "result.json", result)
        return result


class PureAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a191-invented-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.plan = make_plan()
        self.tokenizer = InventedTokenizer(self.plan)
        self.prepared = a.prepare_inputs(self.plan, self.tokenizer)

    def run_fake(self, dispatcher=None, **kwargs):
        dispatcher = dispatcher or InventedDispatcher()
        with mock.patch.object(a, "_dispatch", dispatcher):
            return a.run_acquisition(
                object(),
                self.tokenizer,
                plan=self.plan,
                prepared=self.prepared,
                runtime_binding=RUNTIME,
                directory=self.root,
                **kwargs,
            )

    def rewrite(self, path, value):
        path.write_bytes(a.e.canonical(value))

    def rebind_terminal(self, name="terminal.json", changes=None):
        path = self.root / name
        terminal = a._read(path)
        if changes is not None:
            changes(terminal)
        inventory = a.primitive._inventory(self.root)
        inventory.pop(name)
        terminal["artifact_sha256"] = inventory
        self.rewrite(path, terminal)

    def test_plan_fixed_policy_and_deep_bindings(self):
        self.assertEqual(self.plan["threads"], 8)
        self.assertEqual(
            (self.plan["max_entries"], self.plan["max_scientific_entries"]), (1026, 1024)
        )
        self.assertEqual(len(self.plan["rows"]), 16)
        self.assertFalse(self.plan["generation_policy"]["cache"])
        self.assertEqual(a.validate_plan(self.plan), self.plan)
        changed = copy.deepcopy(self.plan)
        changed["threads"] = 4
        with self.assertRaises(ValueError):
            a.validate_plan(changed)

    def test_native_preparation_has_exact_row_and_prompt_bindings(self):
        self.assertEqual(len(self.prepared["observations"]), 16)
        self.assertEqual(len(self.tokenizer.template_calls), 16)
        for row, obs in zip(self.plan["rows"], self.prepared["observations"], strict=True):
            self.assertEqual(obs["row_sha256"], a.e.object_hash(row))
            self.assertEqual(obs["prompt_ids_sha256"], a.e.object_hash(obs["prompt_token_ids"]))
            self.assertIn(row["messages"][1]["content"], obs["prompt_utf8"])

    def test_prepared_tamper_and_oversize_rejected(self):
        altered = copy.deepcopy(self.prepared)
        altered["observations"][0]["prompt_token_ids"] = [True]
        with self.assertRaises(ValueError):
            a.validate_prepared(self.plan, altered)
        altered = copy.deepcopy(self.prepared)
        altered["observations"][0]["prompt_token_ids"] = [1] * 65
        altered["observations"][0]["prompt_ids_sha256"] = a.e.object_hash([1] * 65)
        with self.assertRaises(ValueError):
            a.validate_prepared(self.plan, altered)

    def test_all16_complete_eos_and_normal_return_replay(self):
        stages = []
        dispatcher = InventedDispatcher()
        result = self.run_fake(dispatcher, event=stages.append)
        self.assertEqual(stages, ["technical", "science"])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(
            result["summary"],
            {
                "technical_entries": 2,
                "scientific_entries": 32,
                "total_entries": 34,
                "entries_complete": True,
                "planned_rows": 16,
                "completed_rows": 16,
                "failed_rows": 0,
                "unattempted_rows": 0,
            },
        )
        self.assertEqual(
            result["analysis"]["overall"]["labels"], {"success": 16, "failure": 0, "unknown": 0}
        )
        self.assertTrue(result["analysis"]["supplied_qualification"])
        self.assertEqual(result, a.load_acquisition(self.root, self.plan))
        self.assertEqual(dispatcher.calls[0][3], dispatcher.calls[1][3])
        for index in range(16):
            pair = [call for call in dispatcher.calls if call[:2] == ("science", index)]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[1][3], pair[0][3] + [16 + index])
            self.assertTrue((self.root / "rows" / f"{index:03d}" / "return.json").is_file())
        self.assertEqual(len(self.tokenizer.decode_calls), 16)

    def test_cap_ceiling_all16_no_reference_or_capture_calls(self):
        dispatcher = InventedDispatcher(cap=True)
        result = self.run_fake(dispatcher)
        self.assertEqual(len(dispatcher.calls), 1026)
        self.assertEqual(result["summary"]["scientific_entries"], 1024)
        self.assertEqual(result["analysis"]["overall"]["capped"], 16)
        self.assertEqual(
            result["analysis"]["overall"]["labels"], {"success": 0, "failure": 16, "unknown": 0}
        )
        self.assertFalse(result["analysis"]["supplied_qualification"])
        self.assertFalse(
            any("prefix" in str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        )

    def test_technical_repeatability_failure_never_starts_science(self):
        dispatcher = InventedDispatcher(mismatch=True)
        with self.assertRaisesRegex(ValueError, "technical_repeatability"):
            self.run_fake(dispatcher)
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(len(dispatcher.calls), 2)
        self.assertEqual(result["summary"]["unattempted_rows"], 16)
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)
        self.assertIsNone(result["analysis"]["supplied_qualification"])

    def test_technical_body_failure_has_no_completed_endpoint(self):
        dispatcher = InventedDispatcher(fail_at=("technical", 0, 1))
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(dispatcher)
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(result["summary"]["technical_entries"], 2)
        self.assertEqual(result["summary"]["scientific_entries"], 0)
        self.assertEqual(result["summary"]["unattempted_rows"], 16)

    def test_science_interruption_preserves16_slots_and_known_completed_row(self):
        dispatcher = InventedDispatcher(fail_at=("science", 1, 1))
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(dispatcher)
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(result["summary"]["completed_rows"], 1)
        self.assertEqual(result["summary"]["failed_rows"], 1)
        self.assertEqual(result["summary"]["unattempted_rows"], 14)
        self.assertEqual(
            result["analysis"]["overall"]["labels"], {"success": 1, "failure": 0, "unknown": 15}
        )
        self.assertIsNone(
            result["analysis"]["contrasts"]["success_rate"]["supplied_minus_computed"]["point"]
        )
        self.assertTrue(result["summary"]["entries_complete"])
        self.assertEqual(result["summary"]["total_entries"], 6)

    def test_bound_failure_receipt_authenticates_entry_without_entry_file(self):
        dispatcher = InventedDispatcher(fail_at=("science", 0, 0), publish_entry=False)
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(dispatcher)
        result = a.load_acquisition(self.root, self.plan)
        self.assertTrue(result["summary"]["entries_complete"])
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)
        self.assertEqual(result["summary"]["total_entries"], 3)

    def test_unpublished_entry_and_failure_keep_exact_count_unknown(self):
        dispatcher = InventedDispatcher(
            fail_at=("science", 0, 0), publish_entry=False, publish_failure=False
        )
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(dispatcher)
        result = a.load_acquisition(self.root, self.plan)
        self.assertFalse(result["summary"]["entries_complete"])
        self.assertIsNone(result["summary"]["scientific_entries"])
        self.assertIsNone(result["summary"]["total_entries"])
        self.assertEqual(result["summary"]["scientific_entries_lower_bound"], 0)
        self.assertEqual(result["summary"]["total_entries_lower_bound"], 2)
        self.assertEqual(result["reported_summary"]["total_entries"], 3)
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)

    def test_failed_after_result_publication_keeps_final_saved_eos_unusable(self):
        normal = InventedDispatcher()

        def dispatch(model, plan, directory, attempt, counts):
            result = normal(model, plan, directory, attempt, counts)
            if (attempt["stage"], attempt["sequence_index"], attempt["step_index"]) == (
                "science",
                0,
                1,
            ):
                a._write(
                    Path(directory) / "failure.json",
                    {
                        "schema_version": a.primitive.SCHEMA,
                        "attempt_sha256": a.e.object_hash(attempt),
                        "entries": 1,
                        "model_call_returned": True,
                        "cleanup_failure_type": None,
                        "failure_type": "KeyboardInterrupt",
                        "usable": False,
                    },
                )
                raise KeyboardInterrupt("invented_after_result")
            return result

        with self.assertRaisesRegex(KeyboardInterrupt, "invented_after_result"):
            self.run_fake(dispatch)
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(result["summary"]["total_entries"], 4)
        self.assertIsNone(result["records"][0]["label"])
        self.assertFalse(result["records"][0]["terminal_eos"])
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)

    def test_decode_failure_does_not_promote_saved_eos_to_success(self):
        self.tokenizer.decode = mock.Mock(side_effect=RuntimeError("invented_decode_failure"))
        with self.assertRaisesRegex(RuntimeError, "invented_decode_failure"):
            self.run_fake()
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(result["summary"]["completed_rows"], 0)
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)
        self.assertTrue(result["records"][0]["terminal_eos"])

    def test_row_return_publication_failure_does_not_promote_orphan_result(self):
        original = a._write

        def write(path, value):
            if path.name == "return.json":
                raise OSError("invented_return_publication")
            return original(path, value)

        with mock.patch.object(a, "_write", write):
            with self.assertRaisesRegex(OSError, "invented_return_publication"):
                self.run_fake()
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(result["analysis"]["overall"]["labels"]["unknown"], 16)
        self.assertTrue(result["records"][0]["terminal_eos"])

    def test_published_normal_return_preserves_completed_endpoint_on_signal(self):
        original = a._write

        def write(path, value):
            original(path, value)
            if path.name == "return.json":
                raise KeyboardInterrupt("invented_after_normal_return")

        with mock.patch.object(a, "_write", write):
            with self.assertRaisesRegex(KeyboardInterrupt, "invented_after_normal_return"):
                self.run_fake()
        result = a.load_acquisition(self.root, self.plan)
        self.assertEqual(result["summary"]["completed_rows"], 1)
        self.assertEqual(result["summary"]["failed_rows"], 0)
        self.assertEqual(
            result["analysis"]["overall"]["labels"], {"success": 1, "failure": 0, "unknown": 15}
        )

    def test_failure_return_implication_and_typed_entry_tamper_rejected(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(
                InventedDispatcher(
                    fail_at=("science", 0, 0), publish_entry=False, count_entry=False
                )
            )
        path = self.root / "rows/000/step_00/failure.json"
        original = a._read(path)
        for changes in ({"model_call_returned": True}, {"entries": False}):
            with self.subTest(changes=changes):
                value = {**original, **changes}
                self.rewrite(path, value)
                self.rebind_terminal(name="failure.json")
                with self.assertRaises(ValueError):
                    a.load_acquisition(self.root, self.plan)

    def test_existing_output_is_consumed(self):
        self.run_fake()
        dispatcher = InventedDispatcher()
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_fake(dispatcher)
        self.assertEqual(dispatcher.calls, [])

    def test_runtime_binding_mismatch_before_directory_claim(self):
        with self.assertRaisesRegex(ValueError, "runtime_binding"):
            a.run_acquisition(
                object(),
                self.tokenizer,
                plan=self.plan,
                prepared=self.prepared,
                runtime_binding={"wrong": True},
                directory=self.root,
            )
        self.assertFalse(self.root.exists())

    def test_raw_logit_tamper_fails_even_after_inventory_rebinding(self):
        self.run_fake()
        path = self.root / "rows/000/step_00/logits.fp32"
        path.write_bytes(b"\x00" * 256)
        self.rebind_terminal()
        with self.assertRaises(ValueError):
            a.load_acquisition(self.root, self.plan)

    def test_response_tamper_fails_even_after_inventory_rebinding(self):
        self.run_fake()
        (self.root / "rows/000/response.utf8").write_text("unrelated invented text")
        self.rebind_terminal()
        with self.assertRaises(ValueError):
            a.load_acquisition(self.root, self.plan)

    def test_typed_row_return_tamper_fails(self):
        self.run_fake()
        path = self.root / "rows/000/return.json"
        value = a._read(path)
        value["event"] = True
        self.rewrite(path, value)
        self.rebind_terminal()
        with self.assertRaises(ValueError):
            a.load_acquisition(self.root, self.plan)

    def test_finished_analysis_tamper_fails(self):
        self.run_fake()
        self.rebind_terminal(changes=lambda x: x["analysis"]["overall"]["labels"].update(success=0))
        with self.assertRaises(ValueError):
            a.load_acquisition(self.root, self.plan)

    def test_dispatch_budget_checked_before_model_or_after_extra_entry(self):
        attempt = a._attempt(self.plan, "6" * 64, "science", 0, 0, [1])
        for stage, counts in (
            ("science", {"technical": {"generation": 2}, "science": {"generation": 1024}}),
            ("technical", {"technical": {"generation": 2}, "science": {"generation": 0}}),
        ):
            attempt["stage"] = stage
            with (
                mock.patch.object(a, "_preflight") as preflight,
                mock.patch.object(a.primitive, "_forward") as forward,
            ):
                with self.assertRaisesRegex(ValueError, "entry_ceiling_before"):
                    a._dispatch(object(), self.plan, self.root, attempt, counts)
                preflight.assert_not_called()
                forward.assert_not_called()
        attempt["stage"] = "science"
        counts = {"technical": {"generation": 2}, "science": {"generation": 1023}}

        def too_many(_model, _plan, _directory, _attempt, count):
            count["generation"] += 2
            return {}

        with (
            mock.patch.object(a, "_preflight"),
            mock.patch.object(a.primitive, "_forward", too_many),
        ):
            with self.assertRaisesRegex(ValueError, "entry_ceiling_after"):
                a._dispatch(object(), self.plan, self.root, attempt, counts)

    def test_failed_summary_rejects_bad_types_and_above_budget_counts(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_fake(InventedDispatcher(fail_at=("science", 0, 0)))
        original = a._read(self.root / "failure.json")
        changes = (
            {"total_entries": True},
            {"total_entries": 1027},
            {"scientific_entries": -1},
            {"technical_entries": 3},
            {"total_entries": 3.0},
            {"entries_complete": True},
        )
        for mutation in changes:
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(original)
                value["summary"].update(mutation)
                self.rewrite(self.root / "failure.json", value)
                with self.assertRaises(ValueError):
                    a.load_acquisition(self.root, self.plan)
        self.rewrite(self.root / "failure.json", original)

    def test_original_exception_survives_damaged_return_bookkeeping(self):
        original = a._write

        def write(path, value):
            if path.name == "return.json":
                original(path, {"event": "damaged", "result_sha256": "0" * 64})
                raise KeyboardInterrupt("original_stop")
            return original(path, value)

        with mock.patch.object(a, "_write", write):
            with self.assertRaisesRegex(KeyboardInterrupt, "original_stop"):
                self.run_fake()


@contextmanager
def tiny_body_profile(model):
    codes = {
        inspect.unwrap(type(model).forward).__code__: "lm",
        inspect.unwrap(type(model.model).forward).__code__: "decoder",
    }
    counts = {"lm": 0, "decoder": 0}
    prior = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1
        if prior is not None:
            prior(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield counts
    finally:
        sys.setprofile(prior)


class NativeTinyFP32Tests(unittest.TestCase):
    """Root-only: actual untrained CPU FP32 eight-thread dispatch, no tokenizer."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest("dedicated Torch environment required") from exc
        cls.torch, cls.Config, cls.LM = torch, LlamaConfig, LlamaForCausalLM
        cls.old_threads = torch.get_num_threads()
        cls.old_deterministic = torch.are_deterministic_algorithms_enabled()
        cls.old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        cls.old_sdp = {
            name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("math", "flash", "mem_efficient", "cudnn")
        }
        torch.set_num_threads(8)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        for name in cls.old_sdp:
            getattr(torch.backends.cuda, "enable_" + name + "_sdp")(name == "math")

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)
        cls.torch.use_deterministic_algorithms(cls.old_deterministic, warn_only=cls.old_warn)
        for name, value in cls.old_sdp.items():
            getattr(cls.torch.backends.cuda, "enable_" + name + "_sdp")(value)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a191-tiny-untrained-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.plan = make_plan(eos=0)
        cfg = self.Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=0,
            bos_token_id=1,
            tie_word_embeddings=True,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = "sdpa"
        self.model = self.LM(cfg).to(device="cpu", dtype=self.torch.float32).eval()
        with self.torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        self.tokenizer = InventedTokenizer(self.plan)
        self.prepared = a.prepare_inputs(self.plan, self.tokenizer)

    def assert_no_hooks(self):
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_actual_pinned_primitive_eight_threads_fp32_first_argmax(self):
        counts = {"technical": {"generation": 0}, "science": {"generation": 0}}
        attempt = a._attempt(self.plan, "7" * 64, "technical", 0, 0, [1, 2, 3])
        tensors = []
        original = a.primitive._raw

        def raw(torch, tensor, shape):
            tensors.append((tensor.dtype, tensor.device.type, tuple(tensor.shape)))
            return original(torch, tensor, shape)

        with mock.patch.object(a.primitive, "_raw", raw), tiny_body_profile(self.model) as bodies:
            result = a._dispatch(self.model, self.plan, self.root, attempt, counts)
        self.assertEqual(bodies, {"lm": 1, "decoder": 1})
        self.assertEqual(self.torch.get_num_threads(), 8)
        self.assertEqual(tensors, [(self.torch.float32, "cpu", (1, 1, 64))])
        self.assertEqual(result["chosen_token_id"], 0)
        self.assertEqual(
            struct.unpack("<64f", (self.root / "logits.fp32").read_bytes()), (0.0,) * 64
        )
        self.assertEqual(
            a.primitive._replay_call(self.root, a._primitive_plan(self.plan), attempt), result
        )
        self.assert_no_hooks()

    def test_full_native_eos_run_exact18_nonadditive_entries(self):
        with tiny_body_profile(self.model) as bodies:
            result = a.run_acquisition(
                self.model,
                self.tokenizer,
                plan=self.plan,
                prepared=self.prepared,
                runtime_binding=RUNTIME,
                directory=self.root,
            )
        self.assertEqual(bodies, {"lm": 18, "decoder": 18})
        self.assertEqual(result["summary"]["total_entries"], 18)
        self.assertEqual(result["summary"]["completed_rows"], 16)
        self.assertEqual(result["analysis"]["overall"]["terminal_eos"], 16)
        self.assertEqual(result["analysis"]["overall"]["labels"]["failure"], 16)
        self.assertFalse(result["analysis"]["supplied_qualification"])
        self.assertEqual(result, a.load_acquisition(self.root, self.plan))
        self.assert_no_hooks()

    def test_non_fp32_or_training_preflight_rejected_without_forward(self):
        with tiny_body_profile(self.model) as bodies:
            self.model.train()
            with self.assertRaises(ValueError):
                a._preflight(self.model, self.plan)
            self.model.eval().to(dtype=self.torch.bfloat16)
            with self.assertRaises(ValueError):
                a._preflight(self.model, self.plan)
        self.assertEqual(bodies, {"lm": 0, "decoder": 0})


if __name__ == "__main__":
    unittest.main()
