"""Source-bound tokenizer import guard; no tokenizer/model imports at module load.

The single generic registry allowance is not permission to execute its factories.
Counts refer to prohibited entry attempts, not completed calls or object instances.
In selected mode tokenizer execution is permitted and is not counted here. Use a
separate caller ledger for the selected tokenizer operations. This is a guard for
a trusted, frozen runner, not a sandbox for arbitrary code or prior execution.
"""

from __future__ import annotations

import hashlib
import importlib.abc
from pathlib import Path
import re
import stat
import sys

SCHEMA = "tokenizer-model-guard-v1"
REGISTRY_MODULE = "transformers.models.auto.modeling_auto"
REGISTRY_RELATIVE_PATH = "models/auto/modeling_auto.py"
REGISTRY_SOURCE_SHA256 = "b439f15d7b0f1ed3e1339c0e71cea0ebc1bac4a87ff89596d39d78e4cd1b0fd4"
CODES = frozenset(
    {
        "invalid_mode",
        "invalid_module_name",
        "registry_source",
        "registry_search_path",
        "registry_preloaded",
        "architecture_import",
        "model_construction",
        "model_load",
        "model_forward",
        "auto_factory_from_config",
        "auto_factory_from_pretrained",
        "tokenizer_construction",
        "tokenizer_load",
        "gpu_initialization",
        "profile_existing",
        "guard_reuse",
        "guard_cleanup",
    }
)
COUNTERS = (
    "model_construction",
    "model_load",
    "model_forward",
    "auto_factory_from_config",
    "auto_factory_from_pretrained",
    "tokenizer_construction",
    "tokenizer_load",
    "gpu_initialization",
)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", re.ASCII)


class GuardError(ValueError):
    def __init__(self, code):
        if type(code) is not str or code not in CODES:
            raise ValueError("unregistered_guard_code")
        self.code = code
        super().__init__(code)


def _require(ok, code):
    if not ok:
        raise GuardError(code)


def _valid_name(value):
    return type(value) is str and 0 < len(value) <= 256 and _NAME.fullmatch(value) is not None


def classify_import(fullname):
    """Classify a bounded module name; only the exact registry receives an exception."""
    _require(_valid_name(fullname), "invalid_module_name")
    if fullname == REGISTRY_MODULE:
        return "generic_registry"
    if fullname.startswith("transformers.models.") and any(
        part.startswith("modeling_") for part in fullname.split(".")
    ):
        return "concrete_architecture"
    return "other"


def _stat(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def verify_registry_source(transformers_root):
    """Read the exact registry source bytes only; never import or execute them."""
    root = Path(transformers_root)
    path = root / REGISTRY_RELATIVE_PATH
    try:
        _require(root.is_absolute() and root.is_dir(), "registry_source")
        _require(all(not p.is_symlink() for p in (path, *path.parents)), "registry_source")
        before = path.stat()
        _require(stat.S_ISREG(before.st_mode), "registry_source")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(_stat(before) == _stat(path.stat()), "registry_source")
        _require(actual == REGISTRY_SOURCE_SHA256, "registry_source")
    except (OSError, TypeError) as error:
        raise GuardError("registry_source") from error
    return {"relative_path": REGISTRY_RELATIVE_PATH, "sha256": actual}


class NoModelGuard(importlib.abc.MetaPathFinder):
    """Reject concrete model imports/execution, allowing one pinned generic registry.

    Install in the cold worker before importing Transformers. ``import_only``
    also blocks tokenizer construction and loading; ``selected`` permits these.
    ``snapshot`` includes a private bounded module fullname. ``summary`` omits it.
    A registry request count is not proof of a completed registry import.
    """

    def __init__(self, *, transformers_root, mode="selected"):
        _require(type(mode) is str and mode in {"selected", "import_only"}, "invalid_mode")
        self.transformers_root = Path(transformers_root)
        self.mode = mode
        self._counts = dict.fromkeys(COUNTERS, 0)
        self.allowed_registry_requests = 0
        self.restored = False
        self.cleanup_error_type = None
        self.blocked_module = None
        self.failure_classification = None
        self._used = False
        self._callback = self.profile

    @property
    def counts(self):
        return dict(self._counts)

    def summary(self):
        return {
            "schema_version": SCHEMA,
            "mode": self.mode,
            "prohibited_entries": self.counts,
            "allowed_registry_requests": self.allowed_registry_requests,
            "failure_classification": self.failure_classification,
            "profile_restored": self.restored,
            "cleanup_error_type": self.cleanup_error_type,
        }

    def snapshot(self):
        return {**self.summary(), "blocked_module": self.blocked_module}

    def _reject(self, code, module=None, *, counter=None):
        if counter is not None:
            self._counts[counter] += 1
        if self.failure_classification is None:
            self.failure_classification = code
            self.blocked_module = module if _valid_name(module) else None
        raise GuardError(code)

    def _source(self):
        try:
            return verify_registry_source(self.transformers_root)
        except GuardError:
            self._reject("registry_source", REGISTRY_MODULE)

    def find_spec(self, fullname, path=None, target=None):
        if type(fullname) is str and not fullname.startswith("transformers.models."):
            return None
        try:
            kind = classify_import(fullname)
        except GuardError:
            self._reject("invalid_module_name")
        if kind == "concrete_architecture":
            self._reject("architecture_import", fullname)
        if kind == "generic_registry":
            expected = str(self.transformers_root / "models/auto")
            if type(path) not in (list, tuple) or len(path) != 1 or path[0] != expected:
                self._reject("registry_search_path", fullname)
            self._source()
            self.allowed_registry_requests += 1
        return None

    def profile(self, frame, event, arg):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        if module == "torch.cuda" and name == "_lazy_init":
            self._reject("gpu_initialization", module, counter="gpu_initialization")
        if type(module) is not str or not module.startswith("transformers."):
            return
        value = frame.f_locals.get("self", frame.f_locals.get("cls"))
        cls = value if isinstance(value, type) else type(value)
        bases = {base.__name__ for base in cls.__mro__}
        if "_BaseAutoModelClass" in bases and name in {"from_config", "from_pretrained"}:
            key = "auto_factory_" + name
            self._reject(key, module, counter=key)
        if self.mode == "import_only" and (
            "PreTrainedTokenizerBase" in bases
            or (
                module == "transformers.models.auto.tokenization_auto"
                and cls.__name__ == "AutoTokenizer"
            )
        ):
            key = {"__init__": "tokenizer_construction", "from_pretrained": "tokenizer_load"}.get(
                name
            )
            if key:
                self._reject(key, module, counter=key)
        key = {
            "__init__": "model_construction",
            "from_pretrained": "model_load",
            "forward": "model_forward",
        }.get(name)
        if key and (
            "PreTrainedModel" in bases
            or (_valid_name(module) and classify_import(module) == "concrete_architecture")
        ):
            self._reject(key, module, counter=key)

    def __enter__(self):
        if self._used:
            self._reject("guard_reuse")
        self._used = True
        if sys.getprofile() is not None:
            self._reject("profile_existing")
        self._source()
        if REGISTRY_MODULE in sys.modules:
            self._reject("registry_preloaded", REGISTRY_MODULE)
        for name in tuple(sys.modules):
            if _valid_name(name) and classify_import(name) == "concrete_architecture":
                self._reject("architecture_import", name)
        sys.meta_path.insert(0, self)
        try:
            sys.setprofile(self._callback)
        except BaseException:
            sys.meta_path.remove(self)
            raise
        return self

    def __exit__(self, kind, error, traceback):
        cleanup_error = None
        try:
            sys.setprofile(None)
        except BaseException as cleanup:
            cleanup_error = cleanup
        try:
            sys.meta_path.remove(self)
        except BaseException as cleanup:
            cleanup_error = cleanup_error or cleanup
        self.restored = cleanup_error is None
        if cleanup_error is None:
            try:
                self._source()
            except BaseException as cleanup:
                cleanup_error = cleanup
        self.cleanup_error_type = type(cleanup_error).__name__ if cleanup_error else None
        if cleanup_error is not None and error is None:
            raise cleanup_error
        return False
