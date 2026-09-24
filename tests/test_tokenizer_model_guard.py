"""Invented frames and registry bytes only; no Transformers/tokenizer imports."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lexical_prompt_study import tokenizer_model_guard as g


@contextmanager
def registry():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "transformers"
        path = root / g.REGISTRY_RELATIVE_PATH
        path.parent.mkdir(parents=True)
        raw = b"# invented registry, never executed\n"
        path.write_bytes(raw)
        with patch.object(g, "REGISTRY_SOURCE_SHA256", hashlib.sha256(raw).hexdigest()):
            yield root, path


def frame(module, name, cls=None, *, instance=False):
    return SimpleNamespace(
        f_globals={"__name__": module},
        f_code=SimpleNamespace(co_name=name),
        f_locals={"self" if instance else "cls": object.__new__(cls) if instance else cls},
    )


def family(base_name, child_name="Derived"):
    base = type(base_name, (), {})
    return type(child_name, (base,), {})


class ClassificationTests(unittest.TestCase):
    def test_exact_pinned_registry_constant(self):
        self.assertEqual(
            g.REGISTRY_SOURCE_SHA256,
            "b439f15d7b0f1ed3e1339c0e71cea0ebc1bac4a87ff89596d39d78e4cd1b0fd4",
        )
        self.assertEqual(g.classify_import(g.REGISTRY_MODULE), "generic_registry")
        self.assertEqual(g.REGISTRY_RELATIVE_PATH, "models/auto/modeling_auto.py")

    def test_configuration_and_tokenizer_metadata_allowed(self):
        for name in (
            "transformers.models.llama.configuration_llama",
            "transformers.models.auto.tokenization_auto",
            "transformers.modeling_utils",
            "torch",
        ):
            with self.subTest(name=name):
                self.assertEqual(g.classify_import(name), "other")

    def test_safe_error_code_only(self):
        with self.assertRaisesRegex(ValueError, "unregistered_guard_code"):
            g.GuardError("arbitrary private error text")
        self.assertEqual(str(g.GuardError("model_load")), "model_load")


def rejected_name(name, expected):
    def test(self):
        if expected is None:
            with self.assertRaisesRegex(g.GuardError, "invalid_module_name"):
                g.classify_import(name)
        else:
            self.assertEqual(g.classify_import(name), expected)

    return test


for index, name in enumerate(
    (
        "transformers.models.llama.modeling_llama",
        "transformers.models.auto.modeling_other",
        "transformers.models.auto.modeling_auto.child",
        "transformers.models.auto.modeling_auto_extra",
        "transformers.models.fake.modeling_auto",
    )
):
    setattr(
        ClassificationTests, f"test_concrete_{index}", rejected_name(name, "concrete_architecture")
    )
for index, name in enumerate(
    (None, True, "", "x." * 200, "foo\nprivate", "foo/token", "a..b", "é.foo")
):
    setattr(ClassificationTests, f"test_invalid_name_{index}", rejected_name(name, None))


class SourceTests(unittest.TestCase):
    def test_exact_bytes_and_search_path(self):
        with registry() as (root, path):
            receipt = g.verify_registry_source(root)
            self.assertEqual(receipt["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            guard = g.NoModelGuard(transformers_root=root)
            self.assertIsNone(guard.find_spec(g.REGISTRY_MODULE, [str(path.parent)]))
            self.assertEqual(guard.allowed_registry_requests, 1)
            self.assertFalse(any(guard.counts.values()))

    def test_modified_registry_rejected(self):
        with registry() as (root, path):
            path.write_bytes(b"modified")
            guard = g.NoModelGuard(transformers_root=root)
            with self.assertRaisesRegex(g.GuardError, "registry_source"):
                guard.find_spec(g.REGISTRY_MODULE, [str(path.parent)])
            self.assertEqual(guard.allowed_registry_requests, 0)
            self.assertEqual(guard.blocked_module, g.REGISTRY_MODULE)

    def test_registry_symlink_rejected(self):
        with registry() as (root, path):
            target = root / "invented.py"
            path.rename(target)
            path.symlink_to(target)
            with self.assertRaisesRegex(g.GuardError, "registry_source"):
                g.verify_registry_source(root)

    def test_registry_directory_rejected(self):
        with registry() as (root, path):
            path.unlink()
            path.mkdir()
            with self.assertRaisesRegex(g.GuardError, "registry_source"):
                g.verify_registry_source(root)

    def test_other_search_path_cannot_use_pinned_allowance(self):
        with registry() as (root, path):
            for paths in (None, [], [str(root)], [str(path.parent), "/other"], str(path.parent)):
                with self.subTest(paths=paths):
                    guard = g.NoModelGuard(transformers_root=root)
                    with self.assertRaisesRegex(g.GuardError, "registry_search_path"):
                        guard.find_spec(g.REGISTRY_MODULE, paths)
                    self.assertEqual(guard.allowed_registry_requests, 0)

    def test_concrete_import_records_bounded_private_name(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        name = "transformers.models.llama.modeling_llama"
        with self.assertRaisesRegex(g.GuardError, "architecture_import"):
            guard.find_spec(name)
        self.assertEqual(guard.snapshot()["blocked_module"], name)
        self.assertNotIn("blocked_module", guard.summary())
        self.assertNotIn(name, str(guard.summary()))
        self.assertEqual(guard.failure_classification, "architecture_import")

    def test_invalid_private_name_not_retained(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        with self.assertRaises(g.GuardError):
            guard.find_spec("transformers.models.private\ncontent")
        self.assertIsNone(guard.blocked_module)
        self.assertEqual(guard.failure_classification, "invalid_module_name")

    def test_unrelated_nonidentifier_import_name_passes(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        self.assertIsNone(guard.find_spec("_sysconfigdata__linux_x86_64-linux-gnu"))
        self.assertEqual(guard.allowed_registry_requests, 0)
        self.assertFalse(any(guard.counts.values()))
        self.assertIsNone(guard.failure_classification)


class ProfileTests(unittest.TestCase):
    def test_selected_tokenizer_execution_not_counted_as_prohibited(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        for name in ("__init__", "from_pretrained"):
            guard.profile(
                frame(
                    "transformers.tokenization_utils_base", name, family("PreTrainedTokenizerBase")
                ),
                "call",
                None,
            )
        self.assertFalse(any(guard.counts.values()))

    def test_registry_metadata_class_definition_allowed(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        guard.profile(
            frame(g.REGISTRY_MODULE, "AutoModel", family("_BaseAutoModelClass")), "call", None
        )
        guard.profile(
            frame("transformers.models.auto.auto_factory", "auto_class_update"), "call", None
        )
        self.assertFalse(any(guard.counts.values()))

    def test_unrelated_method_named_forward_allowed(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        guard.profile(frame("transformers.generic", "forward", family("Ordinary")), "call", None)
        self.assertFalse(any(guard.counts.values()))

    def test_noncall_event_does_not_double_count(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        guard.profile(frame("torch.cuda", "_lazy_init"), "return", None)
        self.assertEqual(guard.counts["gpu_initialization"], 0)

    def test_snapshots_do_not_alias_counters(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        guard.counts["model_load"] = 9
        guard.snapshot()["prohibited_entries"]["model_load"] = 9
        self.assertEqual(guard.counts["model_load"], 0)

    def test_first_violation_retained_if_later_cleanup_fails(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        with self.assertRaises(g.GuardError):
            guard.profile(frame("torch.cuda", "_lazy_init"), "call", None)
        with self.assertRaises(g.GuardError):
            guard.find_spec("transformers.models.llama.modeling_llama")
        self.assertEqual(guard.failure_classification, "gpu_initialization")
        self.assertEqual(guard.blocked_module, "torch.cuda")


def blocked_frame(mode, module, name, base, counter, class_name="Derived"):
    def test(self):
        guard = g.NoModelGuard(transformers_root="/unused", mode=mode)
        cls = family(base, class_name) if base else None
        f = frame(module, name, cls, instance=name in {"__init__", "forward"} and cls is not None)
        with self.assertRaisesRegex(g.GuardError, counter):
            guard.profile(f, "call", None)
        self.assertEqual(guard.counts[counter], 1)
        self.assertEqual(sum(guard.counts.values()), 1)
        self.assertEqual(guard.blocked_module, module)

    return test


for index, case in enumerate(
    (
        (
            "selected",
            "transformers.modeling_utils",
            "__init__",
            "PreTrainedModel",
            "model_construction",
        ),
        (
            "selected",
            "transformers.modeling_utils",
            "from_pretrained",
            "PreTrainedModel",
            "model_load",
        ),
        (
            "selected",
            "transformers.models.llama.modeling_llama",
            "forward",
            "PreTrainedModel",
            "model_forward",
        ),
        (
            "selected",
            "transformers.models.llama.modeling_llama",
            "__init__",
            "Ordinary",
            "model_construction",
        ),
        (
            "selected",
            "transformers.models.auto.auto_factory",
            "from_config",
            "_BaseAutoModelClass",
            "auto_factory_from_config",
        ),
        (
            "selected",
            "transformers.models.auto.auto_factory",
            "from_pretrained",
            "_BaseAutoModelClass",
            "auto_factory_from_pretrained",
        ),
        (
            "import_only",
            "transformers.models.auto.auto_factory",
            "from_config",
            "_BaseAutoModelClass",
            "auto_factory_from_config",
        ),
        ("selected", "torch.cuda", "_lazy_init", None, "gpu_initialization"),
        (
            "import_only",
            "transformers.tokenization_utils_tokenizers",
            "__init__",
            "PreTrainedTokenizerBase",
            "tokenizer_construction",
        ),
        (
            "import_only",
            "transformers.tokenization_utils_base",
            "from_pretrained",
            "PreTrainedTokenizerBase",
            "tokenizer_load",
        ),
        (
            "import_only",
            "transformers.models.auto.tokenization_auto",
            "from_pretrained",
            "Ordinary",
            "tokenizer_load",
            "AutoTokenizer",
        ),
    )
):
    setattr(ProfileTests, f"test_prohibited_entry_{index}", blocked_frame(*case))


class LifecycleTests(unittest.TestCase):
    def test_context_restores_and_cannot_be_reused(self):
        with registry() as (root, _):
            before = list(sys.meta_path)
            guard = g.NoModelGuard(transformers_root=root)
            with guard:
                self.assertIn(guard, sys.meta_path)
                self.assertIsNotNone(sys.getprofile())
            self.assertEqual(sys.meta_path, before)
            self.assertIsNone(sys.getprofile())
            self.assertTrue(guard.restored)
            with self.assertRaisesRegex(g.GuardError, "guard_reuse"):
                with guard:
                    self.fail("reused")

    def test_original_exception_identity_survives_cleanup_error(self):
        with registry() as (root, _):
            guard = g.NoModelGuard(transformers_root=root)
            guard.__enter__()
            original = KeyboardInterrupt("private")
            try:
                with patch.object(g.sys, "setprofile", side_effect=OSError("private")):
                    self.assertFalse(guard.__exit__(type(original), original, None))
                self.assertNotIn(guard, sys.meta_path)
                self.assertEqual(guard.cleanup_error_type, "OSError")
                self.assertFalse(guard.restored)
            finally:
                sys.setprofile(None)

    def test_original_raised_object_preserved(self):
        with registry() as (root, _):
            original = SystemExit(3)
            guard = g.NoModelGuard(transformers_root=root)
            with self.assertRaises(SystemExit) as caught:
                with guard:
                    raise original
            self.assertIs(caught.exception, original)
            self.assertTrue(guard.restored)

    def test_post_context_source_tamper_rejects(self):
        with registry() as (root, path):
            guard = g.NoModelGuard(transformers_root=root)
            with self.assertRaisesRegex(g.GuardError, "registry_source"):
                with guard:
                    path.write_bytes(b"changed")
            self.assertTrue(guard.restored)
            self.assertEqual(guard.cleanup_error_type, "GuardError")
            self.assertIsNone(sys.getprofile())

    def test_existing_profile_not_overwritten(self):
        guard = g.NoModelGuard(transformers_root="/unused")
        marker = object()
        with patch.object(g.sys, "getprofile", return_value=marker):
            with self.assertRaisesRegex(g.GuardError, "profile_existing"):
                guard.__enter__()
        self.assertNotIn(guard, sys.meta_path)

    def test_failed_install_removes_finder(self):
        with registry() as (root, _):
            guard = g.NoModelGuard(transformers_root=root)
            with patch.object(g.sys, "setprofile", side_effect=OSError("private")):
                with self.assertRaises(OSError):
                    guard.__enter__()
            self.assertNotIn(guard, sys.meta_path)

    def test_preloaded_registry_rejected(self):
        with registry() as (root, _):
            guard = g.NoModelGuard(transformers_root=root)
            with patch.dict(sys.modules, {g.REGISTRY_MODULE: SimpleNamespace()}):
                with self.assertRaisesRegex(g.GuardError, "registry_preloaded"):
                    guard.__enter__()
            self.assertNotIn(guard, sys.meta_path)

    def test_preloaded_architecture_rejected(self):
        with registry() as (root, _):
            guard = g.NoModelGuard(transformers_root=root)
            with patch.dict(
                sys.modules, {"transformers.models.llama.modeling_llama": SimpleNamespace()}
            ):
                with self.assertRaisesRegex(g.GuardError, "architecture_import"):
                    guard.__enter__()
            self.assertNotIn(guard, sys.meta_path)


if __name__ == "__main__":
    unittest.main()
