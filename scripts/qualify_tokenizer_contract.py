"""A193 one-shot tokenizer instrument; import is inert and stdlib-only.

Caller-level operation entries/returns are distinct from nested backend activity.
No model architecture is imported; sampled resource limits are not reservations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

if not __debug__:
    raise RuntimeError("a193_optimization_forbidden")

SCHEMA = "a193-tokenizer-contract-v1"
NATIVE_PYTHON = "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python"
PINS = {
    "scripts/run_native_prefix_qualification.py": "af4038d8422a33e67e507a230e16a89c5ce48002d9b6aa81888cfc838d4e94d0",
    "scripts/run_matched_prefix_prediction.py": "9ee523132bf54a84131052f50c35f1cec3a5cb1ec816302512d28b817df276cd",
    "src/lexical_prompt_study/__init__.py": "38df51bcc0f0179554da19316a656daaa4ba07dd998a704399b29e3085bce948",
    "src/lexical_prompt_study/special_token_inventory.py": "75e9425bbde7fefa56798613c6ad17a643f077463af1fce8b4c2c40e716391f0",
}
SOURCE_PATHS = tuple(PINS) + ("scripts/qualify_tokenizer_contract.py",)
EXPECTED_OPERATIONS = {
    "tokenizer_load": 1,
    "inventory_collection": 2,
    "token_to_id": 1,
    "encode": 15,
    "render_template": 6,
    "tokenize_template": 6,
    "decode": 9,
}
LIMITS = {
    "minimum_available_bytes": 4 * 1024**3,
    "maximum_owned_rss_bytes": 4 * 1024**3,
    "host_pressure_floor_bytes": 2 * 1024**3,
    "minimum_free_scratch_bytes": 256 * 1024**2,
    "minimum_remaining_scratch_bytes": 128 * 1024**2,
    "maximum_raw_run_bytes": 64 * 1024**2,
    "wall_seconds": 600,
    "termination_grace_seconds": 60,
    "poll_seconds": 0.2,
    "storage_poll_seconds": 5.0,
}
ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONOPTIMIZE": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}
RELEVANT_ASSETS = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
SNAPSHOT_NAMES = RELEVANT_ASSETS | {
    "LICENSE.txt",
    "USE_POLICY.md",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
}
PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "jinja2",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)
FREEZE_KEYS = set(
    "schema_version prepared_root source_root sources files_sha256 output_root native_python public_commit selection_sha256 qualification_sha256 protocol_sha256 prior_study_closed prior_closure_sha256 target limits expected_operations".split()
)
TARGET_KEYS = set(
    "model_id model_revision model_path vocab_size eos_token_ids chat_template_sha256 runtime_versions assets_sha256 snapshot_names".split()
)
STAGES = {
    "startup",
    "binding",
    "assets_before",
    "tokenizer_import",
    "tokenizer_load",
    "inventory_before",
    "terminal_identity",
    "fixtures",
    "inventory_after",
    "release",
    "postflight",
    "finished",
}
GUARDS = set(
    "binding canonical_json regular_file symlink_parent hash_changed publish_parent helper_hash source_changed freeze_path freeze_schema native_python actual_python limits source_path output_path inventory_type inventory_entry freeze_inventory source_closure source_binding dependency_pins executing_source commit artifact_binding target_schema target_identity vocabulary eos template runtime asset_inventory asset_regular asset_changed asset_hash asset_metadata environment resource_sample insufficient_resources lineage terminal_schema terminal_inventory terminal_lineage operation_name operation_budget operation_counts invalid_ids inventory_contract terminal_contract eot_identity payload_special prompt_contract closed_contract target_contract path_roundtrip candidate_consistency inventory_changed architecture_import model_execution gpu_initialization profile_existing profile_restore guard_cleanup tokenizer_release failure_artifact".split()
)
SYSTEM = "Return exactly two comma-separated integers and no other text."
USERS = ("Copy this ordered pair: 17,-6.", "Preserve the order of this pair: 17,-6.")
CANDIDATES = ("17,-6", "18,-6")
DATE = "24 Sep 2026"
EOT = "<|eot_id|>"


class ContractError(ValueError):
    def __init__(self, code):
        if code not in GUARDS:
            raise RuntimeError("unregistered_guard")
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise ContractError(code)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def regular(path):
    path = Path(path)
    require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "regular_file")
    require(all(not p.is_symlink() for p in path.parents), "symlink_parent")
    return path


def stable_stat(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def digest(path):
    path = regular(path)
    before = path.stat()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    require(stable_stat(before) == stable_stat(path.stat()), "hash_changed")
    return value


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, value):
    path = Path(path)
    require(all(not p.is_symlink() for p in path.parents), "publish_parent")
    raw = canonical(value)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    sync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "canonical_json")
            result[key] = value
        return result

    def finite(value):
        result = float(value)
        require(math.isfinite(result), "canonical_json")
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite, parse_constant=finite)


def read_json(path, expected=None):
    raw = regular(path).read_bytes()
    require(expected is None or hashlib.sha256(raw).hexdigest() == expected, "binding")
    result = parse_json(raw)
    require(canonical(result) == raw, "canonical_json")
    return result


def inventory(root):
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), "inventory_type")
    result = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "inventory_type")
        require(path.is_dir() or stat.S_ISREG(path.stat().st_mode), "inventory_type")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def module(path, name, wanted):
    require(digest(path) == wanted, "helper_hash")
    raw = regular(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == wanted, "helper_hash")
    result = SimpleNamespace(__file__=str(path), __name__=name)
    exec(compile(raw, str(path), "exec"), result.__dict__)
    require(digest(path) == wanted, "source_changed")
    return result


def helper():
    name = "scripts/run_native_prefix_qualification.py"
    return module(Path(__file__).parent / Path(name).name, "a193_lifecycle", PINS[name])


def monitor_module():
    name = "scripts/run_matched_prefix_prediction.py"
    return module(Path(__file__).parent / Path(name).name, "a193_monitor", PINS[name])


def runtime_versions():
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in PACKAGES},
    }


def validate_target(target):
    require(type(target) is dict and set(target) == TARGET_KEYS, "target_schema")
    require(
        target["model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
        and target["model_revision"] == "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "target_identity",
    )
    require(type(target["vocab_size"]) is int and target["vocab_size"] == 128256, "vocabulary")
    eos = target["eos_token_ids"]
    require(
        type(eos) is list
        and eos
        and all(type(x) is int and 0 <= x < target["vocab_size"] for x in eos)
        and eos == sorted(set(eos)),
        "eos",
    )
    require(is_hash(target["chat_template_sha256"]), "template")
    require(
        type(target["runtime_versions"]) is dict
        and set(target["runtime_versions"]) == set(PACKAGES) | {"python"}
        and all(type(x) is str and x for x in target["runtime_versions"].values()),
        "runtime",
    )
    require(
        type(target["assets_sha256"]) is dict
        and set(target["assets_sha256"]) == RELEVANT_ASSETS
        and all(is_hash(x) for x in target["assets_sha256"].values()),
        "asset_inventory",
    )
    require(
        target["snapshot_names"] == sorted(SNAPSHOT_NAMES)
        and Path(target["model_path"]).is_absolute(),
        "asset_inventory",
    )


def asset_check(target):
    """Hash only five tokenizer/config files; weights are neither read nor authenticated."""
    validate_target(target)
    root = Path(target["model_path"])
    require(
        root.is_dir() and not root.is_symlink() and all(not p.is_symlink() for p in root.parents),
        "asset_inventory",
    )
    require({p.name for p in root.iterdir()} == SNAPSHOT_NAMES, "asset_inventory")
    actual, metadata = {}, {}
    for name in sorted(SNAPSHOT_NAMES):
        path = root / name
        require(stat.S_ISREG(path.stat().st_mode), "asset_regular")
        if name not in RELEVANT_ASSETS:
            continue
        link_before, before = path.lstat(), path.stat()
        raw = path.read_bytes()
        require(
            stable_stat(before) == stable_stat(path.stat())
            and stable_stat(link_before) == stable_stat(path.lstat()),
            "asset_changed",
        )
        actual[name] = hashlib.sha256(raw).hexdigest()
        require(actual[name] == target["assets_sha256"][name], "asset_hash")
        metadata[name] = parse_json(raw)
    require(
        type(metadata["config.json"].get("vocab_size")) is int
        and metadata["config.json"]["vocab_size"] == target["vocab_size"],
        "asset_metadata",
    )
    eos = metadata["generation_config.json"].get("eos_token_id")
    require(
        type(eos) is list
        and all(type(x) is int for x in eos)
        and sorted(eos) == target["eos_token_ids"],
        "asset_metadata",
    )
    template = metadata["tokenizer_config.json"].get("chat_template")
    require(
        type(template) is str
        and hashlib.sha256(template.encode()).hexdigest() == target["chat_template_sha256"],
        "template",
    )
    return {
        "relevant_assets_sha256": actual,
        "snapshot_names": sorted(SNAPSHOT_NAMES),
        "weight_bytes_read": 0,
        "weight_authentication": False,
    }


def verify_freeze(path, expected_sha, *, enforce_executable=True):
    path = Path(path)
    require(
        path.is_absolute() and path.name == "freeze.json" and is_hash(expected_sha), "freeze_path"
    )
    frozen = read_json(path, expected_sha)
    require(
        type(frozen) is dict
        and set(frozen) == FREEZE_KEYS
        and frozen["schema_version"] == "a193-tokenizer-freeze-v1",
        "freeze_schema",
    )
    require(frozen["native_python"] == NATIVE_PYTHON, "native_python")
    require(not enforce_executable or sys.executable == NATIVE_PYTHON, "actual_python")
    require(
        canonical(frozen["limits"]) == canonical(LIMITS)
        and canonical(frozen["expected_operations"]) == canonical(EXPECTED_OPERATIONS),
        "limits",
    )
    prepared, source = Path(frozen["prepared_root"]), Path(frozen["source_root"])
    require(prepared == path.parent and source == prepared / "source", "source_path")
    require(Path(frozen["output_root"]) == prepared.parent / "run-001", "output_path")
    expected = frozen["files_sha256"]
    require(type(expected) is dict and bool(expected), "inventory_type")
    for name, sha in expected.items():
        require(
            type(name) is str
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and str(Path(name)) == name
            and is_hash(sha),
            "inventory_entry",
        )
    actual = inventory(prepared)
    require(
        actual.pop("freeze.json", None) == expected_sha and actual == expected, "freeze_inventory"
    )
    require(
        type(frozen["sources"]) is dict and set(frozen["sources"]) == set(SOURCE_PATHS),
        "source_closure",
    )
    require(
        all(expected.get("source/" + name) == sha for name, sha in frozen["sources"].items()),
        "source_binding",
    )
    require(all(frozen["sources"][name] == sha for name, sha in PINS.items()), "dependency_pins")
    require(
        Path(__file__).absolute() == source / "scripts/qualify_tokenizer_contract.py",
        "executing_source",
    )
    commit = frozen["public_commit"]
    require(
        type(commit) is str and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
        "commit",
    )
    for key in (
        "selection_sha256",
        "qualification_sha256",
        "protocol_sha256",
        "prior_closure_sha256",
    ):
        require(is_hash(frozen[key]) and frozen[key] in expected.values(), "artifact_binding")
    require(frozen["prior_study_closed"] is True, "artifact_binding")
    validate_target(frozen["target"])
    return frozen


class Ledger:
    """Entry publication precedes call; a return receipt never means all checks passed."""

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(mode=0o700)
        self.entries = dict.fromkeys(EXPECTED_OPERATIONS, 0)
        self.returns = dict.fromkeys(EXPECTED_OPERATIONS, 0)
        self.sequence = 0
        self.stage = "startup"
        self.last_operation = None

    def call(self, operation, function, *args, **kwargs):
        require(operation in EXPECTED_OPERATIONS, "operation_name")
        require(self.entries[operation] < EXPECTED_OPERATIONS[operation], "operation_budget")
        self.sequence += 1
        self.entries[operation] += 1
        self.last_operation = operation
        receipt = {
            "sequence": self.sequence,
            "operation": operation,
            "entry_index": self.entries[operation],
            "stage": self.stage,
            "pid": os.getpid(),
        }
        if self.directory is not None:
            publish(self.directory / f"{self.sequence:03d}-entry.json", receipt)
        result = function(*args, **kwargs)
        self.returns[operation] += 1
        if self.directory is not None:
            publish(self.directory / f"{self.sequence:03d}-return.json", receipt)
        return result

    def snapshot(self):
        return {
            "entries": dict(self.entries),
            "returns": dict(self.returns),
            "scope": "explicit_caller_operations_not_all_backend_activity",
        }


def ids(value, vocab_size):
    require(
        type(value) is list
        and bool(value)
        and all(type(x) is int and 0 <= x < vocab_size for x in value),
        "invalid_ids",
    )
    return list(value)


def helper_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ValueError as error:
        raise ContractError("inventory_contract") from error


def operation_schedule():
    result = [
        ("tokenizer_load", "tokenizer_load"),
        ("inventory_collection", "inventory_before"),
        ("token_to_id", "terminal_identity"),
        ("encode", "terminal_identity"),
        ("decode", "terminal_identity"),
    ]
    for _ in USERS:
        result.extend([("encode", "fixtures")] * 4)
        result.extend((op, "fixtures") for op in ("render_template", "encode", "tokenize_template"))
        for _ in CANDIDATES:
            result.extend(
                (op, "fixtures")
                for op in ("render_template", "encode", "tokenize_template", "decode", "decode")
            )
    result.append(("inventory_collection", "inventory_after"))
    return result


def fresh_evidence():
    return {
        "schema_version": SCHEMA,
        "contexts": [
            {"status": "unattempted", "paths": [{"status": "unattempted"} for _ in CANDIDATES]}
            for _ in USERS
        ],
    }


def run_contract(tokenizer, inventory_helper, ledger, *, vocab_size, eos_token_ids, evidence=None):
    """Fixed fresh fixtures only; supplied evidence dict retains partial private state."""
    evidence = evidence if evidence is not None else {}
    evidence.update(fresh_evidence())
    ledger.stage = "inventory_before"
    before = ledger.call(
        "inventory_collection",
        helper_call,
        inventory_helper.collect_special_token_inventory,
        tokenizer,
        vocab_size=vocab_size,
    )
    evidence["inventory_before"] = before
    special = set(before["special_ids"])
    ledger.stage = "terminal_identity"
    eot = ledger.call("token_to_id", tokenizer.convert_tokens_to_ids, EOT)
    evidence["terminal"] = helper_call(
        inventory_helper.validate_terminal_token,
        before,
        terminal_token_id=eot,
        stop_token_ids=eos_token_ids,
    )
    eot_ids = ids(
        ledger.call("encode", tokenizer.encode, EOT, add_special_tokens=False), vocab_size
    )
    decoded = ledger.call(
        "decode",
        tokenizer.decode,
        [eot],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    require(eot_ids == [eot] and type(decoded) is str and decoded == EOT, "eot_identity")
    prior_targets = None
    ledger.stage = "fixtures"
    for index, user in enumerate(USERS):
        context = evidence["contexts"][index]
        context["status"] = "started"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        context["messages"] = messages
        context["payload_token_ids"] = []
        for payload in (SYSTEM, user, *CANDIDATES):
            payload_ids = ids(
                ledger.call("encode", tokenizer.encode, payload, add_special_tokens=False),
                vocab_size,
            )
            context["payload_token_ids"].append(payload_ids)
            require(not special.intersection(payload_ids), "payload_special")
        prompt = ledger.call(
            "render_template",
            tokenizer.apply_chat_template,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            date_string=DATE,
        )
        require(type(prompt) is str and bool(prompt), "prompt_contract")
        context["rendered_prompt"] = prompt
        prompt_ids = ids(
            ledger.call("encode", tokenizer.encode, prompt, add_special_tokens=False), vocab_size
        )
        context["prompt_token_ids"] = prompt_ids
        direct = ids(
            ledger.call(
                "tokenize_template",
                tokenizer.apply_chat_template,
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                date_string=DATE,
            ),
            vocab_size,
        )
        require(prompt_ids == direct and len(prompt_ids) <= 256, "prompt_contract")
        targets = []
        for candidate_index, candidate in enumerate(CANDIDATES):
            record = context["paths"][candidate_index]
            record.update(status="started", candidate=candidate)
            closed_messages = messages + [{"role": "assistant", "content": candidate}]
            closed = ledger.call(
                "render_template",
                tokenizer.apply_chat_template,
                closed_messages,
                tokenize=False,
                add_generation_prompt=False,
                date_string=DATE,
            )
            require(type(closed) is str and closed == prompt + candidate + EOT, "closed_contract")
            record["rendered_closed"] = closed
            joint = ids(
                ledger.call("encode", tokenizer.encode, closed, add_special_tokens=False),
                vocab_size,
            )
            record["closed_token_ids"] = joint
            direct = ids(
                ledger.call(
                    "tokenize_template",
                    tokenizer.apply_chat_template,
                    closed_messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=False,
                    date_string=DATE,
                ),
                vocab_size,
            )
            require(joint == direct and joint[: len(prompt_ids)] == prompt_ids, "closed_contract")
            target = joint[len(prompt_ids) :]
            record["target_token_ids"] = target
            require(
                bool(target)
                and len(target) <= 16
                and target[-1] == eot
                and not special.intersection(target[:-1]),
                "target_contract",
            )
            target_text = ledger.call(
                "decode",
                tokenizer.decode,
                target,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            full_text = ledger.call(
                "decode",
                tokenizer.decode,
                joint,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            require(
                type(target_text) is str
                and type(full_text) is str
                and target_text == candidate + EOT
                and full_text == closed,
                "path_roundtrip",
            )
            record.update(status="completed", decoded_target=target_text, decoded_full=full_text)
            targets.append(target)
        require(prior_targets is None or prior_targets == targets, "candidate_consistency")
        prior_targets = targets
        context["status"] = "completed"
    ledger.stage = "inventory_after"
    after = ledger.call(
        "inventory_collection",
        helper_call,
        inventory_helper.collect_special_token_inventory,
        tokenizer,
        vocab_size=vocab_size,
    )
    evidence["inventory_after"] = after
    require(canonical(before) == canonical(after), "inventory_changed")
    return evidence


class NoModelGuard(importlib.abc.MetaPathFinder):
    """Architecture imports and actual model/GPU entry frames are refused, not inferred."""

    def __init__(self):
        self.counts = {
            "model_construction": 0,
            "model_load": 0,
            "model_forward": 0,
            "gpu_initialization": 0,
        }
        self.restored = False
        self.cleanup_error_type = None

    @staticmethod
    def architecture(name):
        return name.startswith("transformers.models.") and any(
            part.startswith("modeling_") for part in name.split(".")
        )

    def find_spec(self, fullname, path=None, target=None):
        require(not self.architecture(fullname), "architecture_import")
        return None

    def profile(self, frame, event, arg):
        if event != "call":
            return
        module_name = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        if module_name == "torch.cuda" and name == "_lazy_init":
            self.counts["gpu_initialization"] += 1
            raise ContractError("gpu_initialization")
        if not module_name.startswith("transformers"):
            return
        value = frame.f_locals.get("self", frame.f_locals.get("cls"))
        cls = value if isinstance(value, type) else type(value)
        is_model = any(base.__name__ == "PreTrainedModel" for base in cls.__mro__)
        key = {
            "__init__": "model_construction",
            "from_pretrained": "model_load",
            "forward": "model_forward",
        }.get(name)
        if key and (is_model or self.architecture(module_name)):
            self.counts[key] += 1
            raise ContractError("model_execution")

    def __enter__(self):
        require(sys.getprofile() is None, "profile_existing")
        require(not any(self.architecture(name) for name in sys.modules), "architecture_import")
        sys.meta_path.insert(0, self)
        try:
            sys.setprofile(self.profile)
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
        self.cleanup_error_type = type(cleanup_error).__name__ if cleanup_error else None
        if cleanup_error is not None and error is None:
            raise cleanup_error
        return False


def import_report():
    torch = sys.modules.get("torch")
    cuda_initialized = bool(
        torch is not None and hasattr(torch, "cuda") and torch.cuda.is_initialized()
    )
    return {
        "torch_imported": torch is not None,
        "architecture_implementation_imported": any(
            NoModelGuard.architecture(n) for n in sys.modules
        ),
        "cuda_initialized": cuda_initialized,
    }


def default_factory(target):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        target["model_path"], local_files_only=True, trust_remote_code=False, use_fast=True
    )


def failure_code(error):
    return error.code if type(error) is ContractError else "operation_exception"


def original_error_type(error):
    return (
        type(error.__cause__).__name__
        if type(error) is ContractError and error.__cause__ is not None
        else type(error).__name__
    )


def live_template_check(tokenizer, target):
    template = tokenizer.chat_template
    require(
        type(template) is str
        and hashlib.sha256(template.encode()).hexdigest() == target["chat_template_sha256"],
        "template",
    )


def completed_checks(evidence):
    contexts = evidence.get("contexts", [])
    return {
        "planned_contexts": 2,
        "completed_contexts": sum(x.get("status") == "completed" for x in contexts),
        "planned_paths": 4,
        "completed_paths": sum(
            p.get("status") == "completed" for x in contexts for p in x["paths"]
        ),
    }


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def run_worker(path, expected_sha, *, factory=None):
    frozen = verify_freeze(path, expected_sha)
    root = Path(frozen["output_root"]) / "worker"
    root.mkdir(mode=0o700)
    sync_directory(root.parent)
    ledger, guard, evidence, tokenizer = (
        Ledger(root / "operations"),
        NoModelGuard(),
        fresh_evidence(),
        None,
    )
    started = time.monotonic()
    publish(
        root / "run.json",
        {
            "schema_version": SCHEMA,
            "freeze_sha256": expected_sha,
            "identity": identity(os.getpid()),
        },
    )
    try:
        ledger.stage = "binding"
        require(all(os.environ.get(k) == v for k, v in ENV.items()), "environment")
        require(
            canonical(runtime_versions()) == canonical(frozen["target"]["runtime_versions"]),
            "runtime",
        )
        source = Path(frozen["source_root"])
        name = "src/lexical_prompt_study/special_token_inventory.py"
        instrument = module(source / name, "a193_special_token_inventory", frozen["sources"][name])
        ledger.stage = "assets_before"
        before = asset_check(frozen["target"])
        publish(root / "assets-before.json", before)
        with guard:
            require(not import_report()["cuda_initialized"], "gpu_initialization")
            ledger.stage = "tokenizer_load"
            tokenizer = ledger.call("tokenizer_load", factory or default_factory, frozen["target"])
            live_template_check(tokenizer, frozen["target"])
            run_contract(
                tokenizer,
                instrument,
                ledger,
                vocab_size=frozen["target"]["vocab_size"],
                eos_token_ids=frozen["target"]["eos_token_ids"],
                evidence=evidence,
            )
            live_template_check(tokenizer, frozen["target"])
            require(
                ledger.entries == EXPECTED_OPERATIONS and ledger.returns == EXPECTED_OPERATIONS,
                "operation_counts",
            )
            require(not any(guard.counts.values()), "model_execution")
            imports = import_report()
            require(
                not imports["cuda_initialized"]
                and not imports["architecture_implementation_imported"],
                "gpu_initialization",
            )
            ledger.stage = "release"
            tokenizer = None
            gc.collect()
        require(guard.restored, "profile_restore")
        ledger.stage = "postflight"
        after = asset_check(frozen["target"])
        require(canonical(before) == canonical(after), "asset_changed")
        publish(root / "assets-after.json", after)
        require(
            canonical(runtime_versions()) == canonical(frozen["target"]["runtime_versions"]),
            "runtime",
        )
        verify_freeze(path, expected_sha)
        evidence_sha = publish(root / "evidence.json", evidence)
        summary = {
            "schema_version": SCHEMA,
            "status": "qualified",
            "freeze_sha256": expected_sha,
            "operations": ledger.snapshot(),
            "prohibited_entries": dict(guard.counts),
            "checks": completed_checks(evidence),
            "imports": imports,
            "profile_restored": True,
            "tokenizer_reference_released": True,
            "evidence_sha256": evidence_sha,
            "elapsed_seconds": time.monotonic() - started,
        }
        summary_sha = publish(root / "summary.json", summary)
        returned_sha = publish(
            root / "normal-return.json",
            {"event": "tokenizer_contract_returned_normally", "summary_sha256": summary_sha},
        )
        terminal = {
            "schema_version": SCHEMA,
            "status": "qualified",
            "freeze_sha256": expected_sha,
            "summary_sha256": summary_sha,
            "normal_return_sha256": returned_sha,
            "artifact_sha256": inventory(root),
        }
        publish(root / "worker-terminal.json", terminal)
        ledger.stage = "finished"
        return summary
    except BaseException as error:
        tokenizer = None
        try:
            publish(
                root / "worker-failure.json",
                {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "freeze_sha256": expected_sha,
                    "stage": ledger.stage,
                    "guard_code": failure_code(error),
                    "error_type": original_error_type(error),
                    "operations": ledger.snapshot(),
                    "prohibited_entries": dict(guard.counts),
                    "checks": completed_checks(evidence),
                    "profile_restored": guard.restored,
                    "cleanup_error_type": guard.cleanup_error_type,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            publish(root / "partial-evidence.json", evidence)
        except BaseException:
            pass
        postflight = {}
        for check, function in (
            ("sources", lambda: verify_freeze(path, expected_sha)),
            (
                "runtime",
                lambda: require(
                    canonical(runtime_versions())
                    == canonical(frozen["target"]["runtime_versions"]),
                    "runtime",
                ),
            ),
            ("assets", lambda: asset_check(frozen["target"])),
        ):
            try:
                function()
                postflight[check] = {"verified": True, "error_type": None}
            except BaseException as cleanup:
                postflight[check] = {"verified": False, "error_type": type(cleanup).__name__}
        try:
            publish(root / "failure-postflight.json", postflight)
        except BaseException:
            pass
        raise


def validate_worker_terminal(root, freeze_sha):
    root = Path(root)
    terminal = read_json(root / "worker-terminal.json")
    require(
        set(terminal)
        == {
            "schema_version",
            "status",
            "freeze_sha256",
            "summary_sha256",
            "normal_return_sha256",
            "artifact_sha256",
        }
        and terminal["schema_version"] == SCHEMA
        and terminal["status"] == "qualified"
        and terminal["freeze_sha256"] == freeze_sha,
        "terminal_schema",
    )
    actual = inventory(root)
    actual.pop("worker-terminal.json")
    require(
        actual == terminal["artifact_sha256"]
        and not any("failure" in Path(n).name or Path(n).name.startswith(".") for n in actual),
        "terminal_inventory",
    )
    summary = read_json(root / "summary.json", terminal["summary_sha256"])
    require(
        type(summary) is dict
        and set(summary)
        == set(
            "schema_version status freeze_sha256 operations prohibited_entries checks imports profile_restored tokenizer_reference_released evidence_sha256 elapsed_seconds".split()
        )
        and summary["schema_version"] == SCHEMA
        and type(summary["elapsed_seconds"]) in (int, float)
        and math.isfinite(summary["elapsed_seconds"])
        and summary["elapsed_seconds"] > 0
        and summary["status"] == "qualified"
        and summary["freeze_sha256"] == freeze_sha
        and summary["profile_restored"] is True
        and summary["tokenizer_reference_released"] is True,
        "terminal_schema",
    )
    require(
        type(summary["imports"]) is dict
        and set(summary["imports"])
        == {"torch_imported", "architecture_implementation_imported", "cuda_initialized"}
        and type(summary["imports"]["torch_imported"]) is bool
        and summary["imports"]["architecture_implementation_imported"] is False
        and summary["imports"]["cuda_initialized"] is False,
        "model_execution",
    )
    require(
        canonical(summary["operations"])
        == canonical(
            {
                "entries": EXPECTED_OPERATIONS,
                "returns": EXPECTED_OPERATIONS,
                "scope": "explicit_caller_operations_not_all_backend_activity",
            }
        ),
        "operation_counts",
    )
    require(
        canonical(summary["prohibited_entries"])
        == canonical(
            {"model_construction": 0, "model_load": 0, "model_forward": 0, "gpu_initialization": 0}
        ),
        "model_execution",
    )
    require(
        canonical(summary["checks"])
        == canonical(
            {
                "planned_contexts": 2,
                "completed_contexts": 2,
                "planned_paths": 4,
                "completed_paths": 4,
            }
        ),
        "terminal_schema",
    )
    require(
        read_json(root / "normal-return.json", terminal["normal_return_sha256"])
        == {
            "event": "tokenizer_contract_returned_normally",
            "summary_sha256": terminal["summary_sha256"],
        },
        "terminal_schema",
    )
    header = read_json(root / "run.json")
    launch = read_json(root.parent / "launch.json")
    worker_launch = read_json(root.parent / "worker-launch.json")
    claim = read_json(root.parent / "claim.json")
    require(
        set(header) == {"schema_version", "freeze_sha256", "identity"}
        and header["schema_version"] == SCHEMA
        and set(launch) == {"freeze_sha256", "supervisor", "limits"}
        and canonical(launch["limits"]) == canonical(LIMITS)
        and set(worker_launch) == {"freeze_sha256", "worker"}
        and set(claim) == {"schema_version", "freeze_sha256", "supervisor", "one_attempt"}
        and claim["schema_version"] == SCHEMA
        and claim["one_attempt"] is True
        and claim["freeze_sha256"] == freeze_sha
        and canonical(claim["supervisor"]) == canonical(launch["supervisor"]),
        "terminal_lineage",
    )
    require(
        header["freeze_sha256"]
        == freeze_sha
        == launch["freeze_sha256"]
        == worker_launch["freeze_sha256"],
        "terminal_lineage",
    )
    child, parent = worker_launch["worker"], launch["supervisor"]
    for entry in (child, parent):
        require(
            type(entry) is dict
            and set(entry) == {"pid", "parent_pid", "session", "start_ticks"}
            and all(type(x) is int and x > 0 for x in entry.values()),
            "terminal_lineage",
        )
    require(
        canonical(header["identity"]) == canonical(child)
        and child["parent_pid"] == parent["pid"]
        and child["session"] == child["pid"],
        "terminal_lineage",
    )
    require(
        {p.name for p in root.iterdir()}
        == {
            "run.json",
            "operations",
            "assets-before.json",
            "assets-after.json",
            "evidence.json",
            "summary.json",
            "normal-return.json",
            "worker-terminal.json",
        },
        "terminal_inventory",
    )
    require(
        canonical(read_json(root / "assets-before.json"))
        == canonical(read_json(root / "assets-after.json")),
        "asset_changed",
    )
    counts = dict.fromkeys(EXPECTED_OPERATIONS, 0)
    n = sum(EXPECTED_OPERATIONS.values())
    require(
        {p.name for p in (root / "operations").iterdir()}
        == {f"{i:03d}-{kind}.json" for i in range(1, n + 1) for kind in ("entry", "return")},
        "operation_counts",
    )
    for i in range(1, n + 1):
        entry = read_json(root / "operations" / f"{i:03d}-entry.json")
        returned = read_json(root / "operations" / f"{i:03d}-return.json")
        require(
            type(entry) is dict
            and set(entry) == {"sequence", "operation", "entry_index", "stage", "pid"}
            and entry["operation"] in counts,
            "operation_counts",
        )
        counts[entry["operation"]] += 1
        require(
            canonical(entry) == canonical(returned)
            and type(entry["sequence"]) is int
            and entry["sequence"] == i
            and type(entry["entry_index"]) is int
            and entry["entry_index"] == counts[entry["operation"]]
            and type(entry["pid"]) is int
            and entry["pid"] == child["pid"]
            and (entry["operation"], entry["stage"]) == operation_schedule()[i - 1],
            "operation_counts",
        )
    require(counts == EXPECTED_OPERATIONS, "operation_counts")
    require(digest(root / "evidence.json") == summary["evidence_sha256"], "binding")
    return digest(root / "worker-terminal.json")


def environment(frozen):
    result = os.environ.copy()
    result.update(ENV)
    result["PYTHONPATH"] = str(Path(frozen["source_root"]) / "src")
    return result


def supervise(path, expected_sha):
    frozen = verify_freeze(path, expected_sha)
    lifecycle = helper()
    root = Path(frozen["output_root"])
    root.mkdir(mode=0o700)
    sync_directory(root.parent)
    process = None
    started = time.monotonic()
    try:
        publish(
            root / "claim.json",
            {
                "schema_version": SCHEMA,
                "freeze_sha256": expected_sha,
                "supervisor": identity(os.getpid()),
                "one_attempt": True,
            },
        )
        with lifecycle.owned_interruptions():
            memory, disk = lifecycle.memory_available(), shutil.disk_usage(root).free
            require(
                type(memory) is int and memory >= 0 and type(disk) is int and disk >= 0,
                "resource_sample",
            )
            publish(
                root / "readiness.json",
                {"available_memory_bytes": memory, "free_scratch_bytes": disk, "limits": LIMITS},
            )
            require(
                memory >= LIMITS["minimum_available_bytes"]
                and disk >= LIMITS["minimum_free_scratch_bytes"],
                "insufficient_resources",
            )
            probe = lifecycle.open_pidfd(os.getpid())
            try:
                lifecycle.signal_pidfd(probe, 0)
            finally:
                os.close(probe)
            publish(
                root / "launch.json",
                {
                    "freeze_sha256": expected_sha,
                    "supervisor": identity(os.getpid()),
                    "limits": LIMITS,
                },
            )
            with (root / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        NATIVE_PYTHON,
                        "-B",
                        str(Path(frozen["source_root"]) / "scripts/qualify_tokenizer_contract.py"),
                        "worker",
                        "--freeze",
                        str(path),
                        "--freeze-sha256",
                        expected_sha,
                        "--parent",
                        str(os.getpid()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment(frozen),
                    start_new_session=True,
                )
                child = identity(process.pid)
                require(
                    child["parent_pid"] == os.getpid() and child["session"] == process.pid,
                    "lineage",
                )
                publish(
                    root / "worker-launch.json", {"freeze_sha256": expected_sha, "worker": child}
                )
                result = monitor_module().monitor(
                    process, LIMITS, started, root, lifecycle=lifecycle
                )
            result.update(status="failed", freeze_sha256=expected_sha, worker_terminal_sha256=None)
            if result["exit_code"] == 0 and result["stop_reason"] is None:
                verify_freeze(path, expected_sha)
                result["worker_terminal_sha256"] = validate_worker_terminal(
                    root / "worker", expected_sha
                )
                result["status"] = "qualified"
            publish(root / "supervisor-result.json", result)
            return result
    except BaseException as error:
        try:
            publish(
                root / "supervisor-failure.json",
                {
                    "schema_version": SCHEMA,
                    "freeze_sha256": expected_sha,
                    "guard_code": failure_code(error),
                    "error_type": original_error_type(error),
                    "worker_started": process is not None,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        except BaseException:
            pass
        raise
    finally:
        original = sys.exception()
        if process is not None:
            try:
                if process.poll() is None or lifecycle.process_session_snapshot(process.pid):
                    lifecycle.terminate_owned_session(process, LIMITS["termination_grace_seconds"])
                require(not lifecycle.process_session_snapshot(process.pid), "lineage")
            except BaseException:
                if original is None:
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("supervise", "worker"))
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    sys.dont_write_bytecode = True
    try:
        if args.mode == "worker":
            require(type(args.parent) is int and args.parent > 0, "lineage")
            lifecycle = helper()
            lifecycle.bind_parent_lifetime(args.parent)
            with lifecycle.owned_interruptions():
                result = run_worker(args.freeze, args.freeze_sha256)
            output = {
                "status": result["status"],
                "operations": result["operations"],
                "checks": result["checks"],
            }
        else:
            output = supervise(args.freeze, args.freeze_sha256)
        print(canonical(output).decode())
        return 0 if output["status"] == "qualified" else 1
    except BaseException as error:
        print(
            canonical(
                {
                    "status": "failed",
                    "guard_code": failure_code(error),
                    "error_type": original_error_type(error),
                }
            ).decode()
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
