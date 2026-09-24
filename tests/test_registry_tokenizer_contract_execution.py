"""Invented stdlib apparatus fixtures only: no native tokenizer or model imports."""

import hashlib
import copy
import math
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


r = load(ROOT / "scripts/qualify_registry_tokenizer_contract.py", "a194_fixture_runner")
h = load(ROOT / "src/lexical_prompt_study/special_token_inventory.py", "a194_fixture_inventory")

g = load(ROOT / "src/lexical_prompt_study/tokenizer_model_guard.py", "a194_fixture_guard")
REGISTRY = {"relative_path": g.REGISTRY_RELATIVE_PATH, "sha256": g.REGISTRY_SOURCE_SHA256}


def fake_auto_tokenizer():
    return type("AutoTokenizer", (), {"__module__": "transformers.models.auto.tokenization_auto"})


def fake_guard(mode="selected"):
    return g.NoModelGuard(transformers_root=ROOT / "invented-transformers", mode=mode)


class FakeTokenizer:
    """A deterministic character codec with invented specials, not a native tokenizer."""

    chat_template = "invented template"
    all_special_ids = [500, 501]
    added_tokens_decoder = {i: SimpleNamespace(special=True) for i in (500, 501, 502, 503)}

    def __init__(self):
        self.native_calls = []
        self.inventory_reads = 0

    def convert_tokens_to_ids(self, text):
        self.native_calls.append("convert")
        assert text == r.EOT
        return 502

    @staticmethod
    def raw_encode(text):
        values = []
        while text:
            if text.startswith(r.EOT):
                values.append(502)
                text = text[len(r.EOT) :]
            else:
                values.append(ord(text[0]))
                text = text[1:]
        return values

    def encode(self, text, *, add_special_tokens):
        self.native_calls.append("encode")
        assert add_special_tokens is False
        return self.raw_encode(text)

    def decode(self, values, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.native_calls.append("decode")
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return "".join(r.EOT if i == 502 else chr(i) for i in values)

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, date_string, return_dict=None
    ):
        self.native_calls.append("direct" if tokenize else "render")
        assert date_string == r.DATE
        assert return_dict is False if tokenize else return_dict is None
        prompt = messages[0]["content"] + "|" + messages[1]["content"] + "|assistant:"
        if add_generation_prompt:
            assert len(messages) == 2
            text = prompt
        else:
            assert len(messages) == 3
            text = prompt + messages[-1]["content"] + r.EOT
        # An actual template backend may tokenize internally; caller counters exclude that.
        return self.raw_encode(text) if tokenize else text


def completed(fake=None, directory=None):
    ledger = r.Ledger(directory)
    ledger.stage = "tokenizer_load"
    tokenizer = ledger.call("tokenizer_load", lambda: fake or FakeTokenizer())
    evidence = r.run_contract(tokenizer, h, ledger, vocab_size=600, eos_token_ids=[501, 502, 503])
    return tokenizer, ledger, evidence


class ContractTests(unittest.TestCase):
    def test_exact_success_schedule_with_extra_backend_eos(self):
        tokenizer, ledger, evidence = completed()
        self.assertEqual(ledger.entries, r.EXPECTED_OPERATIONS)
        self.assertEqual(ledger.returns, r.EXPECTED_OPERATIONS)
        self.assertEqual(evidence["inventory_before"]["declared_special_ids"], [500, 501])
        self.assertEqual(evidence["terminal"]["stop_token_ids"], [501, 502, 503])
        self.assertEqual(
            r.completed_checks(evidence),
            {
                "planned_contexts": 2,
                "completed_contexts": 2,
                "planned_paths": 4,
                "completed_paths": 4,
            },
        )
        self.assertEqual(tokenizer.native_calls.count("encode"), 15)
        self.assertEqual(tokenizer.native_calls.count("decode"), 9)
        self.assertEqual(tokenizer.native_calls.count("direct"), 6)
        for candidate in range(2):
            self.assertEqual(
                evidence["contexts"][0]["paths"][candidate]["target_token_ids"],
                evidence["contexts"][1]["paths"][candidate]["target_token_ids"],
            )

    def test_exception_keeps_entry_not_return(self):
        ledger = r.Ledger()

        def fail():
            raise KeyboardInterrupt("PRIVATE_SENTINEL")

        with self.assertRaises(KeyboardInterrupt):
            ledger.call("tokenizer_load", fail)
        self.assertEqual(ledger.entries["tokenizer_load"], 1)
        self.assertEqual(ledger.returns["tokenizer_load"], 0)
        with self.assertRaisesRegex(r.ContractError, "operation_budget"):
            ledger.call("tokenizer_load", lambda: None)

    def test_receipt_publication_failure_prevents_call(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = r.Ledger(Path(temp) / "operations")
            invoked = []
            with patch.object(r, "publish", side_effect=OSError("private")):
                with self.assertRaises(OSError):
                    ledger.call("tokenizer_load", lambda: invoked.append(True))
            self.assertEqual(invoked, [])
            self.assertEqual(ledger.returns["tokenizer_load"], 0)

    def test_return_publication_failure_keeps_actual_return(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = r.Ledger(Path(temp) / "operations")
            publish = r.publish

            def fail_return(path, value):
                if path.name.endswith("return.json"):
                    raise OSError("private")
                return publish(path, value)

            with patch.object(r, "publish", side_effect=fail_return):
                with self.assertRaises(OSError):
                    ledger.call("tokenizer_load", lambda: "returned")
            self.assertEqual(ledger.returns["tokenizer_load"], 1)
            self.assertEqual(len(list(ledger.directory.iterdir())), 1)

    def test_partial_contexts_survive_failure(self):
        fake = FakeTokenizer()
        original = fake.apply_chat_template

        def fail_later(messages, **kwargs):
            if messages[1]["content"] == r.USERS[1]:
                raise RuntimeError("PRIVATE_CONTENT")
            return original(messages, **kwargs)

        fake.apply_chat_template = fail_later
        ledger, evidence = r.Ledger(), {}
        with self.assertRaises(RuntimeError):
            r.run_contract(
                fake, h, ledger, vocab_size=600, eos_token_ids=[501, 502, 503], evidence=evidence
            )
        self.assertEqual(r.completed_checks(evidence)["completed_contexts"], 1)
        self.assertEqual(r.completed_checks(evidence)["completed_paths"], 2)
        self.assertEqual(ledger.entries["render_template"], 4)
        self.assertEqual(ledger.returns["render_template"], 3)
        self.assertEqual(ledger.entries["inventory_collection"], 1)

    def test_inventory_changes_reject_after_all_fixtures(self):
        fake = FakeTokenizer()
        collect = h.collect_special_token_inventory
        n = 0

        def changing(tokenizer, **kwargs):
            nonlocal n
            n += 1
            if n == 2:
                tokenizer.all_special_ids = [500]
            return collect(tokenizer, **kwargs)

        instrument = SimpleNamespace(
            collect_special_token_inventory=changing,
            validate_terminal_token=h.validate_terminal_token,
        )
        ledger = r.Ledger()
        with self.assertRaisesRegex(r.ContractError, "inventory_changed"):
            r.run_contract(fake, instrument, ledger, vocab_size=600, eos_token_ids=[501, 502, 503])
        self.assertEqual(ledger.returns["inventory_collection"], 2)

    def test_undeclared_guard_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "unregistered_guard"):
            r.ContractError("PRIVATE_PAYLOAD")

    def test_library_error_has_safe_generic_code(self):
        self.assertEqual(r.failure_code(ValueError("PRIVATE_PAYLOAD")), "operation_exception")

    def test_typed_ids_reject_bool_and_out_of_range(self):
        for values in ([True], [False], [-1], [600], [1.0], [], (1,)):
            with self.subTest(values=values), self.assertRaises(r.ContractError):
                r.ids(values, 600)

    def test_explicit_wrong_eot_identity(self):
        fake = FakeTokenizer()
        fake.convert_tokens_to_ids = lambda text: 501
        with self.assertRaisesRegex(r.ContractError, "eot_identity"):
            completed(fake)

    def test_payload_special_rejected(self):
        fake = FakeTokenizer()
        original = fake.encode
        fake.encode = lambda text, **kw: [502] if text == r.SYSTEM else original(text, **kw)
        with self.assertRaisesRegex(r.ContractError, "payload_special"):
            completed(fake)

    def test_template_direct_bool_id_rejected(self):
        fake = FakeTokenizer()
        original = fake.apply_chat_template
        fake.apply_chat_template = lambda messages, **kw: (
            [True] if kw["tokenize"] else original(messages, **kw)
        )
        with self.assertRaisesRegex(r.ContractError, "invalid_ids"):
            completed(fake)

    def test_open_token_cap(self):
        fake = FakeTokenizer()
        original = fake.apply_chat_template

        def long(messages, **kw):
            original(messages, **kw)
            return [65] * 257 if kw["tokenize"] else "A" * 257

        fake.apply_chat_template = long
        with self.assertRaisesRegex(r.ContractError, "prompt_contract"):
            completed(fake)

    def test_closed_template_suffix_rejected(self):
        fake = FakeTokenizer()
        original = fake.apply_chat_template

        def suffix(messages, **kw):
            value = original(messages, **kw)
            return value + " " if not kw["tokenize"] and not kw["add_generation_prompt"] else value

        fake.apply_chat_template = suffix
        with self.assertRaisesRegex(r.ContractError, "closed_contract"):
            completed(fake)

    def test_joint_prefix_retokenization_rejected(self):
        fake = FakeTokenizer()
        original_encode = fake.encode
        original_template = fake.apply_chat_template
        fake.encode = lambda text, **kw: (
            [100] + original_encode(text, **kw)
            if text.endswith(r.EOT) and text != r.EOT
            else original_encode(text, **kw)
        )
        fake.apply_chat_template = lambda messages, **kw: (
            [100] + original_template(messages, **kw)
            if kw["tokenize"] and not kw["add_generation_prompt"]
            else original_template(messages, **kw)
        )
        with self.assertRaisesRegex(r.ContractError, "closed_contract"):
            completed(fake)

    def test_wrong_decode_rejected(self):
        fake = FakeTokenizer()
        original = fake.decode
        fake.decode = lambda values, **kw: "wrong" if len(values) > 1 else original(values, **kw)
        with self.assertRaisesRegex(r.ContractError, "path_roundtrip"):
            completed(fake)


class GuardAndIOTests(unittest.TestCase):
    def test_publish_duplicate_does_not_replace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            r.publish(path, {"x": 1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                r.publish(path, {"x": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue((path.parent / ".result.json.tmp").exists())

    def test_publish_link_failure_leaves_orphan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            with patch.object(r.os, "link", side_effect=OSError("private")):
                with self.assertRaises(OSError):
                    r.publish(path, {"x": 1})
            self.assertFalse(path.exists())
            self.assertTrue((path.parent / ".result.json.tmp").exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.json"
            r.publish(path, {})
            link = Path(temp) / "link.json"
            link.symlink_to(path)
            with self.assertRaises(r.ContractError):
                r.read_json(link)
            with self.assertRaises(r.ContractError):
                r.inventory(temp)

    def test_json_noncanonical_duplicate_nonfinite_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{ "x": 1 }'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "x.json"
                path.write_bytes(raw)
                with self.assertRaises((r.ContractError, ValueError)):
                    r.read_json(path)


def asset_fixture(root):
    template = "invented template"
    contents = {
        "config.json": {"vocab_size": 128256},
        "generation_config.json": {"eos_token_id": [501, 502, 503]},
        "tokenizer_config.json": {"chat_template": template},
        "tokenizer.json": {"invented": True},
        "special_tokens_map.json": {},
    }
    root.mkdir()
    hashes = {}
    for name in r.SNAPSHOT_NAMES:
        raw = r.canonical(contents[name]) if name in contents else b"not a weight"
        (root / name).write_bytes(raw)
        if name in contents:
            hashes[name] = hashlib.sha256(raw).hexdigest()
    return {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "model_revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "model_path": str(root),
        "vocab_size": 128256,
        "eos_token_ids": [501, 502, 503],
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "runtime_versions": {key: "invented" for key in r.PACKAGES + ("python",)},
        "assets_sha256": hashes,
        "snapshot_names": sorted(r.SNAPSHOT_NAMES),
    }


class AssetTests(unittest.TestCase):
    def test_only_relevant_assets_read(self):
        with tempfile.TemporaryDirectory() as temp:
            target = asset_fixture(Path(temp) / "snapshot")
            original = Path.read_bytes
            reads = []

            def recorded(path):
                reads.append(path.name)
                return original(path)

            with patch.object(Path, "read_bytes", recorded):
                receipt = r.asset_check(target)
            self.assertEqual(set(reads), r.RELEVANT_ASSETS)
            self.assertEqual(receipt["weight_bytes_read"], 0)
            self.assertFalse(receipt["weight_authentication"])

    def test_expected_snapshot_file_symlink_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "snapshot"
            target = asset_fixture(root)
            file = root / "tokenizer.json"
            blob = Path(temp) / "blob"
            file.rename(blob)
            file.symlink_to(blob)
            self.assertEqual(
                r.asset_check(target)["relevant_assets_sha256"], target["assets_sha256"]
            )

    def test_unlisted_auxiliary_tokenizer_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "snapshot"
            target = asset_fixture(root)
            (root / "added_tokens.json").write_text("{}")
            with self.assertRaisesRegex(r.ContractError, "asset_inventory"):
                r.asset_check(target)

    def test_asset_content_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "snapshot"
            target = asset_fixture(root)
            (root / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(r.ContractError, "asset_hash"):
                r.asset_check(target)

    def test_full_vocab_bound_rejects_boolean(self):
        with tempfile.TemporaryDirectory() as temp:
            target = asset_fixture(Path(temp) / "snapshot")
            target["vocab_size"] = True
            with self.assertRaisesRegex(r.ContractError, "vocabulary"):
                r.validate_target(target)


def worker_fixture(root, mode="selected"):
    output = root / ("run-001" if mode == "selected" else "import-001")
    output.mkdir()
    supervisor = r.identity(os.getpid())
    child = r.identity(os.getpid())
    child["parent_pid"] = supervisor["pid"]
    child["session"] = child["pid"]
    r.publish(
        output / "claim.json",
        {
            "schema_version": r.SCHEMA,
            "freeze_sha256": "a" * 64,
            "supervisor": supervisor,
            "one_attempt": True,
        },
    )
    r.publish(
        output / "launch.json",
        {"freeze_sha256": "a" * 64, "supervisor": supervisor, "limits": r.LIMITS},
    )
    r.publish(output / "worker-launch.json", {"freeze_sha256": "a" * 64, "worker": child})
    return {
        "mode": mode,
        "runtime_versions": {"fixture": "version"},
        "transformers_root": str(ROOT / "invented-transformers"),
        "registry_source": REGISTRY,
        "output_root": str(output),
        "source_root": str(ROOT),
        "target": {
            "runtime_versions": {"fixture": "version"},
            "chat_template_sha256": hashlib.sha256(
                FakeTokenizer.chat_template.encode()
            ).hexdigest(),
            "vocab_size": 600,
            "eos_token_ids": [501, 502, 503],
        },
        "sources": r.PINS.copy(),
    }, child


class WorkerTests(unittest.TestCase):
    def run_fake(self, root, factory=None, extra_patch=None, mode="selected"):
        frozen, child = worker_fixture(root, mode=mode)
        with (
            patch.object(r, "verify_freeze", return_value=frozen),
            patch.object(r, "runtime_versions", return_value={"fixture": "version"}),
            patch.object(r, "asset_check", return_value={"invented": True}),
            patch.object(r, "identity", return_value=child),
            patch.object(r, "guard_module", return_value=g),
            patch.object(g, "verify_registry_source", return_value=REGISTRY),
            patch.dict(os.environ, r.ENV),
            extra_patch or patch.object(r, "default_factory", return_value=FakeTokenizer()),
        ):
            value = r.run_worker(root / "freeze.json", "a" * 64, factory=factory)
        return frozen, value

    def test_whole_worker_success_and_terminal_replay_without_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen, result = self.run_fake(Path(temp))
            self.assertEqual(result["status"], "qualified")
            self.assertEqual(result["operations"]["entries"], r.EXPECTED_OPERATIONS)
            self.assertEqual(
                r.validate_worker_terminal(Path(frozen["output_root"]) / "worker", "a" * 64),
                r.digest(Path(frozen["output_root"]) / "worker/worker-terminal.json"),
            )

    def test_worker_failure_preserves_original_and_partial_counts(self):
        with tempfile.TemporaryDirectory() as temp:

            def failed_factory(target):
                raise KeyboardInterrupt("PRIVATE_TOKENIZER_TEXT")

            with self.assertRaises(KeyboardInterrupt):
                self.run_fake(Path(temp), factory=failed_factory)
            root = Path(temp) / "run-001/worker"
            failure = r.read_json(root / "worker-failure.json")
            self.assertEqual(failure["stage"], "tokenizer_load")
            self.assertEqual(failure["guard_code"], "operation_exception")
            self.assertEqual(failure["error_type"], "KeyboardInterrupt")
            self.assertEqual(failure["operations"]["entries"]["tokenizer_load"], 1)
            self.assertEqual(failure["operations"]["returns"]["tokenizer_load"], 0)
            self.assertNotIn("PRIVATE_TOKENIZER_TEXT", (root / "worker-failure.json").read_text())
            self.assertFalse((root / "worker-terminal.json").exists())

    def test_exception_after_terminal_publication_retains_failure_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            publish = r.publish

            def after(path, value):
                result = publish(path, value)
                if Path(path).name == "worker-terminal.json":
                    raise OSError("private")
                return result

            with self.assertRaises(OSError):
                self.run_fake(
                    Path(temp),
                    extra_patch=patch.object(r, "publish", side_effect=after),
                    factory=lambda _: FakeTokenizer(),
                )
            with self.assertRaises(r.ContractError):
                r.validate_worker_terminal(Path(temp) / "run-001/worker", "a" * 64)

    def test_duplicate_worker_root_never_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen, _ = self.run_fake(Path(temp))
            with (
                patch.object(r, "verify_freeze", return_value=frozen),
                self.assertRaises(FileExistsError),
            ):
                r.run_worker(
                    Path(temp) / "freeze.json", "a" * 64, factory=lambda _: self.fail("retry")
                )

    def test_terminal_bool_counter_and_lineage_tamper_rejected(self):
        for kind in ("counter", "birth", "parent", "orphan"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                frozen, _ = self.run_fake(Path(temp))
                root = Path(frozen["output_root"]) / "worker"
                if kind in ("birth", "parent"):
                    path = root.parent / "worker-launch.json"
                    value = r.read_json(path)
                    value["worker"]["start_ticks" if kind == "birth" else "parent_pid"] = (
                        True if kind == "birth" else 999999
                    )
                    path.write_bytes(r.canonical(value))
                elif kind == "orphan":
                    (root / ".unpublished.tmp").write_bytes(b"{}")
                else:
                    path = root / "summary.json"
                    value = r.read_json(path)
                    value["operations"]["entries"]["tokenizer_load"] = True
                    path.write_bytes(r.canonical(value))
                    terminal_path = root / "worker-terminal.json"
                    terminal = r.read_json(terminal_path)
                    terminal["summary_sha256"] = r.digest(path)
                    terminal["artifact_sha256"]["summary.json"] = r.digest(path)
                    terminal_path.write_bytes(r.canonical(terminal))
                with self.assertRaises(r.ContractError):
                    r.validate_worker_terminal(root, "a" * 64)


def import_qualification(frozen):
    return {
        "status": "qualified",
        "exit_code": 0,
        "stop_reason": None,
        "owned_live_processes_remaining": 0,
        "freeze_sha256": "d" * 64,
        "worker_terminal_sha256": "e" * 64,
        "peak_sampled_owned_rss_bytes": 123,
        "peak_sampled_raw_run_bytes": 456,
        "elapsed_seconds": 1.25,
        "import_qualification": {
            "schema_version": "a194-cold-import-qualification-v1",
            **{key: frozen[key] for key in ("sources", "runtime_versions", "registry_source")},
            "operations": {
                "entries": r.IMPORT_OPERATIONS,
                "returns": r.IMPORT_OPERATIONS,
                "scope": "explicit_caller_operations_not_all_backend_activity",
            },
            "prohibited_entries": dict.fromkeys(r.PROHIBITED_KEYS, 0),
            "guard": {
                "schema_version": g.SCHEMA,
                "mode": "import_only",
                "prohibited_entries": dict.fromkeys(r.PROHIBITED_KEYS, 0),
                "allowed_registry_requests": 1,
                "failure_classification": None,
                "profile_restored": True,
                "cleanup_error_type": None,
            },
            "imports": {
                "torch_imported": True,
                "architecture_implementation_imported": False,
                "generic_registry_imported": True,
                "cuda_initialized": False,
            },
        },
    }


class FreezeTests(unittest.TestCase):
    def frozen(self, base):
        prepared = base / "prepared"
        source = prepared / "source"
        sources = {}
        for name in r.SOURCE_PATHS:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
            sources[name] = r.digest(path)
        target = asset_fixture(base / "snapshot")
        artifacts = {}
        for name in ("QUESTION.json", "QUALIFICATION.json", "PROTOCOL.md", "PRIOR-CLOSED.json"):
            path = prepared / name
            path.write_bytes(r.canonical({"invented": name}))
            artifacts[name] = r.digest(path)
        value = {
            "schema_version": "a194-tokenizer-freeze-v1",
            "mode": "selected",
            "runtime_versions": target["runtime_versions"],
            "transformers_root": str(base / "invented-transformers"),
            "registry_source": REGISTRY,
            "prepared_root": str(prepared),
            "source_root": str(source),
            "sources": sources,
            "files_sha256": r.inventory(prepared),
            "output_root": str(base / "run-001"),
            "native_python": r.NATIVE_PYTHON,
            "public_commit": "c" * 40,
            "selection_sha256": artifacts["QUESTION.json"],
            "qualification_sha256": artifacts["QUALIFICATION.json"],
            "protocol_sha256": artifacts["PROTOCOL.md"],
            "prior_study_closed": True,
            "prior_closure_sha256": artifacts["PRIOR-CLOSED.json"],
            "target": target,
            "limits": r.LIMITS.copy(),
            "expected_operations": r.EXPECTED_OPERATIONS.copy(),
        }
        qualification = import_qualification(value)
        qualification_path = prepared / "IMPORT-QUALIFICATION.json"
        r.publish(qualification_path, qualification)
        value["import_qualification_sha256"] = r.digest(qualification_path)
        value["files_sha256"] = r.inventory(prepared)
        path = prepared / "freeze.json"
        r.publish(path, value)
        return path, value

    def check(self, path):
        with patch.object(
            r,
            "__file__",
            str(path.parent / "source/scripts/qualify_registry_tokenizer_contract.py"),
        ):
            return r.verify_freeze(path, r.digest(path), enforce_executable=False)

    def test_whole_frozen_source_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.frozen(Path(temp))
            self.assertEqual(self.check(path), value)

    def test_unclosed_prior_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.frozen(Path(temp))
            value["prior_study_closed"] = False
            path.write_bytes(r.canonical(value))
            with self.assertRaisesRegex(r.ContractError, "artifact_binding"):
                self.check(path)

    def test_missing_transitive_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.frozen(Path(temp))
            value["sources"].pop("src/lexical_prompt_study/__init__.py")
            path.write_bytes(r.canonical(value))
            with self.assertRaisesRegex(r.ContractError, "source_closure"):
                self.check(path)

    def test_copied_source_modified_after_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.frozen(Path(temp))
            (path.parent / "source/src/lexical_prompt_study/__init__.py").write_bytes(b"pass\n")
            with self.assertRaisesRegex(r.ContractError, "freeze_inventory"):
                self.check(path)

    def test_changed_fixed_operation_budget_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.frozen(Path(temp))
            value["expected_operations"]["tokenizer_load"] = 2
            path.write_bytes(r.canonical(value))
            with self.assertRaisesRegex(r.ContractError, "limits"):
                self.check(path)

    def test_boolean_resource_bound_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.frozen(Path(temp))
            value["limits"]["wall_seconds"] = True
            path.write_bytes(r.canonical(value))
            with self.assertRaisesRegex(r.ContractError, "limits"):
                self.check(path)

    def test_unbound_qualification_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, value = self.frozen(Path(temp))
            value["qualification_sha256"] = "f" * 64
            path.write_bytes(r.canonical(value))
            with self.assertRaisesRegex(r.ContractError, "artifact_binding"):
                self.check(path)


class SupervisorTests(unittest.TestCase):
    def setup_fake(self, temp, memory=None, mode="selected"):
        from contextlib import nullcontext

        process = SimpleNamespace(pid=321, poll=lambda: 0, wait=lambda: 0)
        terminated = []
        lifecycle = SimpleNamespace(
            owned_interruptions=nullcontext,
            memory_available=lambda: memory if memory is not None else 8 * 1024**3,
            open_pidfd=lambda pid: os.open(os.devnull, os.O_RDONLY),
            signal_pidfd=lambda fd, number: None,
            process_session_snapshot=lambda pid: [],
            terminate_owned_session=lambda proc, grace: terminated.append(proc.pid),
        )
        frozen = {
            "mode": mode,
            "output_root": str(Path(temp) / ("run-001" if mode == "selected" else "import-001")),
            "source_root": str(ROOT),
            "sources": r.PINS.copy(),
            "runtime_versions": {"fixture": "version"},
            "registry_source": REGISTRY,
        }
        result = {
            "exit_code": 0,
            "stop_reason": None,
            "peak_sampled_owned_rss_bytes": 1,
            "peak_sampled_raw_run_bytes": 2,
            "elapsed_seconds": 0.3,
            "owned_live_processes_remaining": 0,
        }
        return frozen, lifecycle, process, result, terminated

    def invoke(self, temp, *, memory=None, monitor_error=None, stop_reason=None, mode="selected"):
        from contextlib import ExitStack

        frozen, lifecycle, process, result, terminated = self.setup_fake(temp, memory, mode=mode)
        result["stop_reason"] = stop_reason
        if stop_reason:
            result["exit_code"] = 1
        parent = {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "session": os.getsid(0),
            "start_ticks": 123,
        }
        child = {
            "pid": process.pid,
            "parent_pid": os.getpid(),
            "session": process.pid,
            "start_ticks": 456,
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(r, "verify_freeze", return_value=frozen))
            stack.enter_context(patch.object(r, "helper", return_value=lifecycle))
            stack.enter_context(
                patch.object(
                    r, "identity", side_effect=lambda pid: child if pid == process.pid else parent
                )
            )
            stack.enter_context(
                patch.object(r.shutil, "disk_usage", return_value=SimpleNamespace(free=1024**3))
            )

            def monitor_result(*args, **kwargs):
                if mode == "import_only":
                    worker = Path(frozen["output_root"]) / "worker"
                    worker.mkdir()
                    payload = import_qualification(frozen)["import_qualification"]
                    r.publish(
                        worker / "summary.json",
                        {
                            key: payload[key]
                            for key in ("operations", "prohibited_entries", "guard", "imports")
                        },
                    )
                return result

            monitor = SimpleNamespace(monitor=monitor_result)
            if monitor_error:

                def fail(*args, **kwargs):
                    raise monitor_error

                monitor.monitor = fail
                process.poll = lambda: None
            stack.enter_context(patch.object(r, "monitor_module", return_value=monitor))
            launch = stack.enter_context(patch.object(r.subprocess, "Popen", return_value=process))
            terminal = stack.enter_context(
                patch.object(r, "validate_worker_terminal", return_value="a" * 64)
            )
            actual = r.supervise(Path(temp) / "freeze.json", "b" * 64, mode=mode)
            return actual, launch, terminal, terminated

    def test_normal_parent_path_once_with_owned_child(self):
        with tempfile.TemporaryDirectory() as temp:
            actual, launch, terminal, _ = self.invoke(temp)
            self.assertEqual(actual["status"], "qualified")
            launch.assert_called_once()
            args, kwargs = launch.call_args
            self.assertEqual(args[0][0:2], [r.NATIVE_PYTHON, "-B"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            terminal.assert_called_once()
            with self.assertRaises(FileExistsError):
                self.invoke(temp)

    def test_failed_readiness_claim_consumed_no_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(r.ContractError, "insufficient_resources"):
                self.invoke(temp, memory=4 * 1024**3 - 1)
            root = Path(temp) / "run-001"
            self.assertTrue((root / "claim.json").exists())
            self.assertFalse((root / "launch.json").exists())
            self.assertFalse(r.read_json(root / "supervisor-failure.json")["worker_started"])
            with self.assertRaises(FileExistsError):
                self.invoke(temp)

    def test_resource_stop_never_validates_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            actual, launch, terminal, _ = self.invoke(temp, stop_reason="host_memory_pressure")
            self.assertEqual(actual["status"], "failed")
            terminal.assert_not_called()
            launch.assert_called_once()

    def test_monitor_interruption_preserves_original_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            error = KeyboardInterrupt("PRIVATE")
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.invoke(temp, monitor_error=error)
            self.assertIs(caught.exception, error)
            failure = r.read_json(Path(temp) / "run-001/supervisor-failure.json")
            self.assertEqual(failure["error_type"], "KeyboardInterrupt")
            self.assertNotIn("PRIVATE", r.canonical(failure).decode())


class TerminalSemanticTests(WorkerTests):
    # Avoid inheriting the worker suite as duplicate tests; only reuse fixture method.
    test_whole_worker_success_and_terminal_replay_without_tokenizer = None
    test_worker_failure_preserves_original_and_partial_counts = None
    test_exception_after_terminal_publication_retains_failure_marker = None
    test_duplicate_worker_root_never_retried = None
    test_terminal_bool_counter_and_lineage_tamper_rejected = None

    def rebind(self, root):
        summary_sha = r.digest(root / "summary.json")
        returned = {"event": "tokenizer_contract_returned_normally", "summary_sha256": summary_sha}
        (root / "normal-return.json").write_bytes(r.canonical(returned))
        terminal = r.read_json(root / "worker-terminal.json")
        terminal.update(
            summary_sha256=summary_sha, normal_return_sha256=r.digest(root / "normal-return.json")
        )
        artifacts = r.inventory(root)
        artifacts.pop("worker-terminal.json")
        terminal["artifact_sha256"] = artifacts
        (root / "worker-terminal.json").write_bytes(r.canonical(terminal))

    def test_summary_privacy_and_import_semantic_tampering(self):
        changes = (
            lambda s: s.update(hidden_payload="PRIVATE"),
            lambda s: s["imports"].update(cuda_initialized=True),
            lambda s: s["imports"].update(architecture_implementation_imported=True),
            lambda s: s["imports"].update(torch_imported=1),
            lambda s: s.update(elapsed_seconds=True),
            lambda s: s.update(elapsed_seconds=-1.0),
            lambda s: s.update(schema_version="other"),
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                frozen, _ = self.run_fake(Path(temp))
                root = Path(frozen["output_root"]) / "worker"
                path = root / "summary.json"
                value = r.read_json(path)
                change(value)
                path.write_bytes(r.canonical(value))
                self.rebind(root)
                with self.assertRaises(r.ContractError):
                    r.validate_worker_terminal(root, "a" * 64)

    def test_same_count_stage_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen, _ = self.run_fake(Path(temp))
            root = Path(frozen["output_root"]) / "worker"
            for kind in ("entry", "return"):
                path = root / "operations" / f"002-{kind}.json"
                entry = r.read_json(path)
                entry["stage"] = "inventory_after"
                path.write_bytes(r.canonical(entry))
            self.rebind(root)
            with self.assertRaisesRegex(r.ContractError, "operation_counts"):
                r.validate_worker_terminal(root, "a" * 64)

    def test_rehashed_unknown_worker_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            frozen, _ = self.run_fake(Path(temp))
            root = Path(frozen["output_root"]) / "worker"
            (root / "extra.json").write_bytes(b"{}")
            self.rebind(root)
            with self.assertRaisesRegex(r.ContractError, "terminal_inventory"):
                r.validate_worker_terminal(root, "a" * 64)

    def test_live_template_mismatch_fails_before_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeTokenizer()
            fake.chat_template = "unexpected actual template"
            with self.assertRaisesRegex(r.ContractError, "template"):
                self.run_fake(Path(temp), factory=lambda _: fake)
            failure = r.read_json(Path(temp) / "run-001/worker/worker-failure.json")
            self.assertEqual(failure["operations"]["returns"]["tokenizer_load"], 1)
            self.assertEqual(failure["operations"]["entries"]["inventory_collection"], 0)
            self.assertEqual(failure["checks"]["completed_paths"], 0)
            partial = r.read_json(Path(temp) / "run-001/worker/partial-evidence.json")
            self.assertEqual(len(partial["contexts"]), 2)
            self.assertEqual(sum(len(x["paths"]) for x in partial["contexts"]), 4)

    def test_guard_failure_exposes_original_helper_exception_type(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = FakeTokenizer()
            fake.all_special_ids = [True]
            with self.assertRaisesRegex(r.ContractError, "inventory_contract"):
                self.run_fake(Path(temp), factory=lambda _: fake)
            failure = r.read_json(Path(temp) / "run-001/worker/worker-failure.json")
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertEqual(failure["guard_code"], "inventory_contract")
            self.assertEqual(failure["stage"], "inventory_before")


class ColdImportTests(unittest.TestCase):
    run_fake = WorkerTests.run_fake

    def test_cold_import_success_without_any_asset_or_tokenizer_call(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(r, "run_contract", side_effect=AssertionError("tokenizer work")):
                frozen, result = self.run_fake(
                    Path(temp),
                    factory=fake_auto_tokenizer,
                    mode="import_only",
                    extra_patch=patch.object(
                        r, "asset_check", side_effect=AssertionError("selected asset read")
                    ),
                )
            self.assertEqual(result["operations"]["entries"], {"dependency_import": 1})
            self.assertEqual(result["operations"]["returns"], {"dependency_import": 1})
            self.assertEqual(result["prohibited_entries"], dict.fromkeys(r.PROHIBITED_KEYS, 0))
            root = Path(frozen["output_root"]) / "worker"
            self.assertFalse((root / "assets-before.json").exists())
            self.assertFalse((root / "assets-after.json").exists())
            self.assertEqual(
                r.validate_worker_terminal(root, "a" * 64, mode="import_only"),
                r.digest(root / "worker-terminal.json"),
            )
            with self.assertRaises(r.ContractError):
                r.validate_worker_terminal(root, "a" * 64, mode="selected")

    def test_cold_wrong_return_object_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(r.ContractError, "binding"):
                self.run_fake(Path(temp), factory=lambda: object(), mode="import_only")
            failure = r.read_json(Path(temp) / "import-001/worker/worker-failure.json")
            self.assertEqual(failure["operations"]["returns"], {"dependency_import": 1})
            self.assertEqual(failure["checks"]["completed_imports"], 1)
            self.assertFalse((Path(temp) / "import-001/worker/worker-terminal.json").exists())

    def test_private_blocked_fullname_and_safe_classification(self):
        def attempted_import():
            sys.meta_path[0].find_spec("transformers.models.llama.modeling_llama")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(g.GuardError, "architecture_import"):
                self.run_fake(Path(temp), factory=attempted_import, mode="import_only")
            failure = r.read_json(Path(temp) / "import-001/worker/worker-failure.json")
            self.assertEqual(
                failure["guard"]["blocked_module"], "transformers.models.llama.modeling_llama"
            )
            self.assertEqual(failure["failure_classification"], "concrete_architecture_blocked")
            self.assertEqual(failure["operations"]["returns"], {"dependency_import": 0})
            self.assertTrue(failure["profile_restored"])
            postflight = r.read_json(Path(temp) / "import-001/worker/failure-postflight.json")
            self.assertEqual(set(postflight), {"sources", "runtime", "registry"})
            self.assertTrue(postflight["registry"]["verified"])

    def test_cold_tokenizer_constructor_entry_rejected_before_body(self):
        namespace = {"__name__": "transformers.tokenization_utils_base", "entered": []}
        exec(
            "class PreTrainedTokenizerBase:\n def __init__(self):\n  entered.append(True)\n",
            namespace,
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(g.GuardError, "tokenizer_construction"):
                self.run_fake(
                    Path(temp), factory=namespace["PreTrainedTokenizerBase"], mode="import_only"
                )
            self.assertEqual(namespace["entered"], [])
            failure = r.read_json(Path(temp) / "import-001/worker/worker-failure.json")
            self.assertEqual(failure["prohibited_entries"]["tokenizer_construction"], 1)
            self.assertEqual(failure["failure_classification"], "tokenizer_execution_blocked")

    def test_cold_auto_model_factory_rejected_before_body(self):
        namespace = {"__name__": "transformers.models.auto.auto_factory", "entered": []}
        exec(
            "class _BaseAutoModelClass:\n @classmethod\n def from_config(cls):\n  entered.append(True)\n",
            namespace,
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(g.GuardError, "auto_factory_from_config"):
                self.run_fake(
                    Path(temp),
                    factory=namespace["_BaseAutoModelClass"].from_config,
                    mode="import_only",
                )
            self.assertEqual(namespace["entered"], [])
            failure = r.read_json(Path(temp) / "import-001/worker/worker-failure.json")
            self.assertEqual(failure["prohibited_entries"]["auto_factory_from_config"], 1)
            self.assertEqual(failure["failure_classification"], "model_execution_blocked")

    def test_import_report_ignores_unrelated_hyphenated_module_names(self):
        with patch.dict(sys.modules, {"_sysconfigdata_x86_64-linux-gnu": SimpleNamespace()}):
            report = r.import_report(g)
        self.assertFalse(report["architecture_implementation_imported"])

    def test_failure_classification_cannot_emit_arbitrary_code(self):
        error = RuntimeError("PRIVATE")
        error.code = "PRIVATE_UNTRUSTED_TEXT"
        self.assertEqual(r.public_failure_classification(error), "operation_failure")


class ImportQualificationTests(unittest.TestCase):
    def context(self):
        return {
            "sources": {"invented": "a" * 64},
            "runtime_versions": {"python": "invented"},
            "registry_source": REGISTRY,
        }

    def test_complete_qualified_import_binding(self):
        frozen = self.context()
        r.validate_import_qualification(import_qualification(frozen), frozen)

    def test_missing_terminal_provenance_rejected(self):
        for key in (
            "freeze_sha256",
            "worker_terminal_sha256",
            "elapsed_seconds",
            "peak_sampled_owned_rss_bytes",
        ):
            with self.subTest(key=key):
                frozen = self.context()
                receipt = import_qualification(frozen)
                receipt.pop(key)
                with self.assertRaises(r.ContractError):
                    r.validate_import_qualification(receipt, frozen)

    def test_import_cannot_qualify_changed_selected_sources_or_runtime(self):
        for key in ("sources", "runtime_versions", "registry_source"):
            with self.subTest(key=key):
                frozen = self.context()
                receipt = import_qualification(frozen)
                receipt["import_qualification"][key] = {"different": "b" * 64}
                with self.assertRaises(r.ContractError):
                    r.validate_import_qualification(receipt, frozen)

    def test_partial_or_prohibited_import_rejected(self):
        changes = (
            lambda v: v.update(status="failed"),
            lambda v: v.update(exit_code=False),
            lambda v: v.update(owned_live_processes_remaining=False),
            lambda v: v.update(peak_sampled_owned_rss_bytes=True),
            lambda v: v.update(elapsed_seconds=math.inf),
            lambda v: v["import_qualification"]["operations"]["returns"].update(
                dependency_import=0
            ),
            lambda v: v["import_qualification"]["prohibited_entries"].update(model_load=1),
            lambda v: v["import_qualification"]["guard"].update(profile_restored=False),
            lambda v: v["import_qualification"]["imports"].update(cuda_initialized=True),
        )
        for change in changes:
            with self.subTest(change=change):
                frozen = self.context()
                receipt = copy.deepcopy(import_qualification(frozen))
                change(receipt)
                with self.assertRaises(r.ContractError):
                    r.validate_import_qualification(receipt, frozen)


class ColdPackageTests(unittest.TestCase):
    def cold(self, base):
        builder = FreezeTests()
        path, value = builder.frozen(base)
        value["mode"] = "import_only"
        value["schema_version"] = "a194-import-freeze-v1"
        value["output_root"] = str(base / "import-001")
        value["expected_operations"] = r.IMPORT_OPERATIONS.copy()
        value.pop("target")
        value.pop("import_qualification_sha256")
        (path.parent / "IMPORT-QUALIFICATION.json").unlink()
        value["files_sha256"].pop("IMPORT-QUALIFICATION.json")
        path.write_bytes(r.canonical(value))
        return builder, path, value

    def test_cold_freeze_has_no_target_or_prior_import_qualification(self):
        with tempfile.TemporaryDirectory() as temp:
            builder, path, value = self.cold(Path(temp))
            self.assertEqual(builder.check(path), value)
            self.assertNotIn("target", value)
            self.assertNotIn("model_path", r.canonical(value).decode())
            self.assertEqual(value["expected_operations"], {"dependency_import": 1})

    def test_cold_package_rejects_added_selected_asset_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            builder, path, value = self.cold(Path(temp))
            value["target"] = {"model_path": "/unexpected"}
            path.write_bytes(r.canonical(value))
            with self.assertRaisesRegex(r.ContractError, "freeze_schema"):
                builder.check(path)

    def test_cli_mode_cannot_launch_different_frozen_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            _, path, value = self.cold(Path(temp))
            with (
                patch.object(r, "verify_freeze", return_value=value),
                self.assertRaisesRegex(r.ContractError, "freeze_schema"),
            ):
                r.supervise(path, "a" * 64, mode="selected")
            self.assertFalse((Path(temp) / "import-001").exists())
            self.assertFalse((Path(temp) / "run-001").exists())

    def test_cold_supervisor_produces_usable_import_qualification(self):
        with tempfile.TemporaryDirectory() as temp:
            tester = SupervisorTests()
            result, launch, terminal, _ = tester.invoke(temp, mode="import_only")
            self.assertEqual(result["status"], "qualified")
            self.assertEqual(launch.call_args.args[0][3], "import-worker")
            self.assertEqual(terminal.call_args.kwargs, {"mode": "import_only"})
            frozen, *_ = tester.setup_fake(temp, mode="import_only")
            r.validate_import_qualification(result, frozen)
            self.assertEqual(r.read_json(Path(temp) / "import-001/supervisor-result.json"), result)

    def test_inherited_immutable_dependency_pins_unchanged(self):
        for path, sha in r.PINS.items():
            with self.subTest(path=path):
                self.assertEqual(r.digest(ROOT / path), sha)


if __name__ == "__main__":
    unittest.main()
