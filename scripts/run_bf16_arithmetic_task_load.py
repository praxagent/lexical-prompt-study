#!/usr/bin/env python3
"""One-shot A188 queue, CPU worker and resource supervisor.

Import is stdlib-only and inert. Only the new worker can load the selected model;
no old study runner is dispatched. Resource checks are sampled thresholds, not
reservations. Private receipts retain attempted and completed body calls
separately; nested decoder calls must never be added to causal-LM calls.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import weakref

if not __debug__:
    raise RuntimeError("a188_optimization_forbidden")

NATIVE_PYTHON = "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python"
HELPER_SHA = "af4038d8422a33e67e507a230e16a89c5ce48002d9b6aa81888cfc838d4e94d0"
MONITOR_SHA = "9ee523132bf54a84131052f50c35f1cec3a5cb1ec816302512d28b817df276cd"
SCHEMA = "a188-native-worker-v1"
MAX_ENTRIES = 2050
MAX_SCIENCE_ENTRIES = 2048
LIMITS = {
    "minimum_available_bytes": 48 * 1024**3,
    "maximum_owned_rss_bytes": 32 * 1024**3,
    "host_pressure_floor_bytes": 8 * 1024**3,
    "minimum_free_scratch_bytes": 64 * 1024**3,
    "minimum_remaining_scratch_bytes": 16 * 1024**3,
    "maximum_raw_run_bytes": 3 * 1024**3,
    "wall_seconds": 21600,
    "termination_grace_seconds": 60,
    "poll_seconds": 0.2,
    "storage_poll_seconds": 5.0,
    "queue_wall_seconds": 43200,
    "queue_poll_seconds": 60,
}
SOURCE_PATHS = (
    "scripts/run_bf16_arithmetic_task_load.py",
    "scripts/run_matched_prefix_prediction.py",
    "scripts/run_native_prefix_qualification.py",
    "scripts/load_bf16_reference.py",
    "src/lexical_prompt_study/__init__.py",
    "src/lexical_prompt_study/landmark_evidence.py",
    "src/lexical_prompt_study/prefix_only_capture.py",
    "src/lexical_prompt_study/native_prefix_qualification.py",
    "src/lexical_prompt_study/instruction_binding_runtime.py",
    "src/lexical_prompt_study/instruction_binding_tasks.py",
    "src/lexical_prompt_study/bf16_arithmetic_task_load.py",
    "src/lexical_prompt_study/bf16_arithmetic_task_load_acquisition.py",
    "src/lexical_prompt_study/bf16_forward.py",
)
PINS = {
    "scripts/run_native_prefix_qualification.py": HELPER_SHA,
    "scripts/run_matched_prefix_prediction.py": MONITOR_SHA,
    "src/lexical_prompt_study/landmark_evidence.py": "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14",
    "src/lexical_prompt_study/prefix_only_capture.py": "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69",
    "src/lexical_prompt_study/native_prefix_qualification.py": "2542d54eaf839a253ae0b11c028a036ca0d6f697935dd456d0f4e3fc4d929526",
}
FREEZE_KEYS = set(
    "schema_version source_root sources target_config_path target_config_sha256 "
    "runtime_binding plan native_python limits output_root prepared_root "
    "files_sha256 public_commit selection_sha256 qualification_sha256".split()
)
ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONOPTIMIZE": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}


def require(ok, reason):
    if not ok:
        raise ValueError("a188_outer_" + reason)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def regular(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), "regular_file")
    require(all(not p.is_symlink() for p in path.parents), "parent_symlink")
    return path


def digest(path):
    path = regular(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "file_changed",
    )
    return h.hexdigest()


def publish(path, value):
    """Receipt-last exclusive publication; failed temporaries remain unusable."""
    path = Path(path)
    temporary = path.with_name(".pending-" + path.name)
    with temporary.open("xb") as stream:
        os.chmod(temporary, 0o600)
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(path)


def _module(path, name, expected):
    require(digest(path) == expected, "helper_source")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(digest(path) == expected, "helper_source_after_import")
    return module


def helper():
    return _module(
        Path(__file__).with_name("run_native_prefix_qualification.py"), "a188_lifecycle", HELPER_SHA
    )


def monitor_module():
    return _module(
        Path(__file__).with_name("run_matched_prefix_prediction.py"),
        "a188_monitor_only",
        MONITOR_SHA,
    )


def read_json(path, expected=None):
    sha = digest(path)
    require(expected is None or sha == expected, "artifact_hash")
    value = helper().hash_bound_json(path, sha)
    require(canonical(value) == Path(path).read_bytes(), "canonical_json")
    return value


def inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "inventory_root")
    result = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "inventory_special_file")
        if stat.S_ISREG(mode):
            result[str(path.relative_to(root))] = digest(path)
    return result


def raw_bytes(root):
    total = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            try:
                item = (Path(parent) / name).lstat()
            except FileNotFoundError:
                continue
            require(stat.S_ISREG(item.st_mode) or stat.S_ISDIR(item.st_mode), "output_special_file")
            if stat.S_ISREG(item.st_mode):
                total += item.st_size
    return total


def verify_freeze(path, expected_sha, *, enforce_executable=True):
    path = Path(path)
    require(path.is_absolute() and is_hash(expected_sha), "freeze_path_hash")
    frozen = read_json(path, expected_sha)
    require(type(frozen) is dict and set(frozen) == FREEZE_KEYS, "freeze_schema")
    require(frozen["schema_version"] == "a188-native-freeze-v1", "freeze_version")
    require(frozen["native_python"] == NATIVE_PYTHON, "native_python")
    if enforce_executable:
        require(sys.executable == NATIVE_PYTHON, "actual_executable")
    require(canonical(frozen["limits"]) == canonical(LIMITS), "selected_limits")
    prepared, source = Path(frozen["prepared_root"]), Path(frozen["source_root"])
    require(prepared == path.parent and source == prepared / "source", "prepared_source_path")
    require(Path(frozen["output_root"]) == prepared.parent / "run-001", "one_shot_output")
    expected = frozen["files_sha256"]
    require(type(expected) is dict and bool(expected), "inventory_type")
    for relative, sha in expected.items():
        require(
            type(relative) is str
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and str(Path(relative)) == relative
            and is_hash(sha),
            "inventory_entry",
        )
    actual = inventory(prepared)
    require(actual.pop(path.name, None) == expected_sha and actual == expected, "frozen_inventory")
    require(
        type(frozen["sources"]) is dict and set(frozen["sources"]) == set(SOURCE_PATHS),
        "source_closure",
    )
    require(
        all(expected.get("source/" + name) == sha for name, sha in frozen["sources"].items()),
        "source_hashes",
    )
    require(all(frozen["sources"][name] == sha for name, sha in PINS.items()), "inherited_pins")
    require(
        Path(__file__).absolute() == source / "scripts/run_bf16_arithmetic_task_load.py",
        "actual_frozen_runner",
    )
    config = Path(frozen["target_config_path"])
    require(
        config.parent == prepared and expected.get(config.name) == frozen["target_config_sha256"],
        "target_config_binding",
    )
    require(
        type(frozen["public_commit"]) is str
        and len(frozen["public_commit"]) == 40
        and all(c in "0123456789abcdef" for c in frozen["public_commit"]),
        "public_commit",
    )
    for key in ("selection_sha256", "qualification_sha256"):
        require(is_hash(frozen[key]) and frozen[key] in expected.values(), key)
    return frozen


def expected_runtime_binding(config, config_sha):
    result = helper().expected_runtime_binding(config, config_sha)
    result["schema_version"] = "a188-runtime-binding-v1"
    result["cpu_loader_sha256"] = digest(Path(__file__).with_name("load_bf16_reference.py"))
    result["actual_requested_execution"]["sampled_token_cap"] = 64
    result["actual_requested_execution"]["dtype"] = "bfloat16"
    result["actual_requested_execution"]["autocast"] = False
    result["actual_requested_execution"]["technical_forwards"] = 2
    result["note"] = (
        "Legacy NF4/cap64 descriptors remain unchanged as loader input bindings; "
        "A188 directly loads original CPU BF16 parameters with SDPA; buffer/native logit dtypes are recorded. "
        "Two fixed technical forwards precede cache-free greedy generation capped at64."
    )
    return result


def environment(frozen):
    env = os.environ.copy()
    env.update(ENV)
    env["PYTHONPATH"] = str(Path(frozen["source_root"]) / "src")
    # Never inherit Python startup injection into the selected interpreter.
    for name in ("PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONHOME"):
        env.pop(name, None)
    return env


def imported_sources(frozen):
    source = Path(frozen["source_root"])
    for relative in SOURCE_PATHS:
        if not relative.startswith("src/"):
            continue
        name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        if name.endswith(".__init__"):
            name = name.removesuffix(".__init__")
        module = sys.modules.get(name)
        require(
            module is not None and Path(module.__file__).absolute() == source / relative,
            "actual_import_path",
        )
        require(digest(module.__file__) == frozen["sources"][relative], "actual_import_hash")


def native_dependencies(frozen):
    source = Path(frozen["source_root"])
    sys.path.insert(0, str(source / "src"))
    from lexical_prompt_study import bf16_arithmetic_task_load_acquisition as core
    from lexical_prompt_study import instruction_binding_runtime as engine
    from lexical_prompt_study import bf16_forward as primitive

    imported_sources(frozen)
    loader = _module(
        source / "scripts/load_bf16_reference.py",
        "a188_cpu_loader",
        frozen["sources"]["scripts/load_bf16_reference.py"],
    )
    import torch
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

    return core, primitive, engine, loader, torch, LlamaForCausalLM, LlamaModel


def validate_target(config, plan, binding, config_sha, lifecycle):
    require(
        canonical(binding) == canonical(expected_runtime_binding(config, config_sha)),
        "selected_runtime",
    )
    require(
        canonical(plan["geometry"])
        == canonical(
            {
                "vocab_size": 128256,
                "hidden_width": 4096,
                "model_layers": 32,
                "context_limit": 131072,
            }
        ),
        "native_geometry",
    )
    require(object_hash(binding) == plan["bindings"]["runtime_binding_sha256"], "plan_runtime")
    provenance = plan["provenance"]
    require(
        object_hash(config["model_files_sha256"]) == provenance["model_sha256"], "model_binding"
    )
    subset = {
        k: v
        for k, v in config["model_files_sha256"].items()
        if k in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    }
    require(object_hash(subset) == provenance["tokenizer_sha256"], "tokenizer_binding")
    require(
        config["chat_template_sha256"] == provenance["chat_template_sha256"], "template_binding"
    )
    model_config = lifecycle.hash_bound_json(
        Path(config["model_path"]) / "config.json",
        config["model_files_sha256"]["config.json"],
        allow_snapshot_link=True,
    )
    for field, wanted in (
        ("vocab_size", 128256),
        ("hidden_size", 4096),
        ("num_hidden_layers", 32),
        ("max_position_embeddings", 131072),
    ):
        require(
            type(model_config[field]) is int and model_config[field] == wanted, "model_geometry"
        )
    generation = lifecycle.hash_bound_json(
        Path(config["model_path"]) / "generation_config.json",
        config["model_files_sha256"]["generation_config.json"],
        allow_snapshot_link=True,
    )
    eos = generation["eos_token_id"]
    eos = eos if type(eos) is list else [eos]
    require(bool(eos) and all(type(t) is int and 0 <= t < 128256 for t in eos), "eos_type")
    require(canonical(sorted(set(eos))) == canonical(plan["eos_token_ids"]), "native_eos")


class BodyProfile:
    """Count body entry before execution; failed entries stay counted privately."""

    def __init__(self, directory, classes):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        self.codes = {inspect.unwrap(cls.forward).__code__: name for cls, name in classes}
        require(
            len(self.codes) == 2
            and set(self.codes.values()) == {"causal_lm_entries", "nested_decoder_entries"},
            "profile_code_identity",
        )
        self.counts = dict.fromkeys(self.codes.values(), 0)
        self.returns = dict.fromkeys(self.codes.values(), 0)
        self.stage = "nonstudy"
        self.by_stage = {
            stage: dict.fromkeys(self.codes.values(), 0)
            for stage in ("nonstudy", "technical", "science")
        }
        self.previous = sys.getprofile()
        self.cleanup_errors = []
        require(self.previous is None, "owned_profiler")

    def __call__(self, frame, event, _arg):
        name = self.codes.get(frame.f_code)
        if name is None:
            return
        if event == "call":
            require(self.stage in self.by_stage, "profile_stage")
            self.counts[name] += 1
            self.by_stage[self.stage][name] += 1
            publish(
                self.directory / f"{name}_{self.counts[name]:04d}.json",
                {
                    "schema_version": "a188-body-entry-v1",
                    "kind": name,
                    "stage": self.stage,
                    "entry_index": self.counts[name],
                    "pid": os.getpid(),
                },
            )
            require(self.counts[name] <= MAX_ENTRIES, "body_ceiling")
        elif event == "return":
            # CPython emits return events on exceptional unwinding too. These are
            # exit events, not proof that a model call returned successfully.
            self.returns[name] += 1

    def __enter__(self):
        sys.setprofile(self)
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            sys.setprofile(self.previous)
        except BaseException as cleanup:
            self.cleanup_errors.append(type(cleanup).__name__)
            if error is None:
                raise
        return False


def acquisition_complete(acquired, counts):
    summary = acquired["summary"]
    require(
        acquired["status"] == "finished"
        and summary["schedule_finished"] is True
        and summary["entries_complete"] is True,
        "acquisition_complete",
    )
    for key, wanted in (
        ("planned_generations", 32),
        ("completed_rows", 32),
        ("failed_rows", 0),
        ("unattempted_rows", 0),
        ("completed_generations", 32),
    ):
        require(type(summary[key]) is int and summary[key] == wanted, "acquisition_rows")
    entries = summary["total_entries"]
    require(
        type(entries) is int
        and 32 <= entries <= MAX_SCIENCE_ENTRIES
        and all(type(v) is int and v == entries for v in counts.values())
        and set(counts) == {"causal_lm_entries", "nested_decoder_entries"},
        "body_count_match",
    )
    require(
        type(summary["generation_entries"]) is int and summary["generation_entries"] == entries,
        "generation_count_match",
    )


def technical_complete(result):
    require(
        result["status"] == "completed"
        and type(result["entries"]) is int
        and result["entries"] == 2,
        "technical_complete",
    )
    require(
        all(
            result[key] is True for key in ("qualified", "native_repeatable", "widened_repeatable")
        ),
        "technical_qualified",
    )


def stage_counts_complete(counts, stages):
    require(
        type(stages) is dict and set(stages) == {"nonstudy", "technical", "science"},
        "stage_count_keys",
    )
    for stage, values in stages.items():
        require(
            type(values) is dict
            and set(values) == set(counts)
            and all(type(v) is int and v >= 0 for v in values.values()),
            "stage_counts",
        )
        require(
            values["causal_lm_entries"] == values["nested_decoder_entries"], "nested_stage_counts"
        )
    for name, total in counts.items():
        require(
            type(total) is int
            and 34 <= total <= MAX_ENTRIES
            and total == sum(values[name] for values in stages.values()),
            "stage_count_sum",
        )
        require(
            stages["nonstudy"][name] == 0
            and stages["technical"][name] == 2
            and 32 <= stages["science"][name] <= MAX_SCIENCE_ENTRIES,
            "selected_stage_counts",
        )


def run_worker(frozen):
    root = Path(frozen["output_root"]) / "worker"
    root.mkdir(mode=0o700)
    lifecycle = helper()
    phase, profiler, runtime, original = "preflight", None, None, None
    counts = {"causal_lm_entries": 0, "nested_decoder_entries": 0}
    cleanup_errors = []
    try:
        publish(
            root / "run.json",
            {
                "schema_version": SCHEMA,
                "freeze_sha256": object_hash(frozen),
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
            },
        )
        require(all(os.environ.get(k) == v for k, v in ENV.items()), "actual_environment")
        core, primitive, engine, loader, torch, lm, decoder = native_dependencies(frozen)
        config = lifecycle.hash_bound_json(
            frozen["target_config_path"], frozen["target_config_sha256"]
        )
        engine.validate_config(config)
        plan = core.validate_plan(frozen["plan"])
        validate_target(
            config, plan, frozen["runtime_binding"], frozen["target_config_sha256"], lifecycle
        )
        require(
            canonical(engine.runtime_versions()) == canonical(config["runtime_versions"]),
            "runtime_versions",
        )
        assets = lifecycle.asset_check(config)
        publish(root / "assets-before.json", assets)
        require(lifecycle.memory_available() >= LIMITS["minimum_available_bytes"], "preload_memory")
        require(
            shutil.disk_usage(root).free >= LIMITS["minimum_free_scratch_bytes"], "preload_scratch"
        )
        profiler = BodyProfile(
            root / "entries", ((lm, "causal_lm_entries"), (decoder, "nested_decoder_entries"))
        )
        counts = profiler.counts
        with profiler:
            phase = "target_load"
            runtime = loader.load_cpu_bf16(config, engine)
            require(not any(counts.values()), "forward_during_load")
            require(
                type(runtime.model) is lm and type(runtime.model.model) is decoder, "target_class"
            )
            require(canonical(runtime.eos_ids) == canonical(plan["eos_token_ids"]), "target_eos")
            phase = "native_preparation"
            technical_plan = core._primitive_plan(plan)
            model_inventory = primitive.clean_model(runtime.model, technical_plan)
            publish(root / "model-inventory-before.json", model_inventory)
            prepared = core.prepare_inputs(plan, runtime.tokenizer)
            technical_prepared = primitive.prepare_technical(technical_plan, runtime.tokenizer)
            require(
                technical_prepared["chat_template_sha256"]
                == plan["provenance"]["chat_template_sha256"],
                "technical_template",
            )
            require(not any(counts.values()), "forward_during_preparation")
            publish(root / "native-prepared.json", prepared)
            publish(root / "technical-prepared.json", technical_prepared)
            phase = "technical_qualification"
            profiler.stage = "technical"
            checked = primitive.run_technical(
                runtime.model,
                plan=technical_plan,
                prepared=technical_prepared,
                directory=root / "technical",
            )
            profiler.stage = "nonstudy"
            technical_normal_sha = publish(
                root / "technical-normal-return.json",
                {
                    "event": "run_technical_returned_normally",
                    "terminal_sha256": checked["terminal_sha256"],
                },
            )
            technical_counts = dict(counts)
            require(
                canonical(checked)
                == canonical(
                    primitive.load_technical(root / "technical", technical_plan, technical_prepared)
                ),
                "technical_replay",
            )
            require(
                counts == technical_counts
                and all(v == 2 for v in counts.values())
                and not any(profiler.by_stage["nonstudy"].values()),
                "technical_body_counts",
            )
            technical_complete(checked)
            phase = "acquisition"
            profiler.stage = "science"
            acquired = core.run_acquisition(
                runtime.model,
                runtime.tokenizer,
                plan=plan,
                prepared=prepared,
                runtime_binding=frozen["runtime_binding"],
                directory=root / "acquisition",
            )
            profiler.stage = "nonstudy"
            normal_sha = publish(
                root / "acquisition-normal-return.json",
                {
                    "event": "run_acquisition_returned_normally",
                    "terminal_sha256": acquired["terminal_sha256"],
                },
            )
            phase = "replay"
            before_replay = dict(counts)
            require(
                canonical(acquired) == canonical(core.load_acquisition(root / "acquisition", plan)),
                "acquisition_replay",
            )
            require(counts == before_replay, "forward_during_replay")
            acquisition_complete(acquired, profiler.by_stage["science"])
            stage_counts_complete(counts, profiler.by_stage)
            phase = "postflight"
            require(not torch.cuda.is_initialized(), "unexpected_cuda")
            require(
                canonical(engine.runtime_versions()) == canonical(config["runtime_versions"]),
                "runtime_after",
            )
            model_after = primitive.clean_model(runtime.model, technical_plan)
            publish(root / "model-inventory-after.json", model_after)
            after = lifecycle.asset_check(config)
            require(canonical(after) == canonical(assets), "assets_changed")
            publish(root / "assets-after.json", after)
            imported_sources(frozen)
            verify_freeze(Path(frozen["prepared_root"]) / "freeze.json", object_hash(frozen))
            reference = weakref.ref(runtime.model)
            runtime = None
            gc.collect()
            require(reference() is None, "target_reference_retained")
            require(counts == before_replay, "forward_after_acquisition")
        require(sys.getprofile() is profiler.previous, "profile_restored")
        phase = "terminal"
        terminal = {
            "schema_version": SCHEMA,
            "status": "completed",
            "freeze_sha256": object_hash(frozen),
            "acquisition_terminal_sha256": acquired["terminal_sha256"],
            "normal_return_sha256": normal_sha,
            "technical_terminal_sha256": checked["terminal_sha256"],
            "technical_normal_return_sha256": technical_normal_sha,
            "technical_prepared_sha256": digest(root / "technical-prepared.json"),
            "model_inventory_before_sha256": digest(root / "model-inventory-before.json"),
            "model_inventory_after_sha256": digest(root / "model-inventory-after.json"),
            "body_entry_counts_by_stage": profiler.by_stage,
            "body_entry_counts": dict(counts),
            "body_exit_events": dict(profiler.returns),
            "nested_counts_nonadditive": True,
            "profile_restored": True,
            "assets_before_sha256": digest(root / "assets-before.json"),
            "assets_after_sha256": digest(root / "assets-after.json"),
            "native_prepared_sha256": digest(root / "native-prepared.json"),
            "model_free_replay": True,
            "target_released": True,
        }
        publish(root / "worker-terminal.json", terminal)
        return terminal
    except BaseException as error:
        original = error
        try:
            publish(
                root / "worker-failure.json",
                {
                    "schema_version": SCHEMA,
                    "phase": phase,
                    "freeze_sha256": object_hash(frozen),
                    "error_type": type(error).__name__,
                    "body_entry_counts": dict(counts),
                    "body_entry_counts_by_stage": profiler.by_stage
                    if profiler is not None
                    else None,
                    "nested_counts_nonadditive": True,
                },
            )
        except BaseException:
            pass
        raise
    finally:
        if profiler is not None:
            cleanup_errors.extend(profiler.cleanup_errors)
        try:
            if profiler is not None:
                sys.setprofile(profiler.previous)
        except BaseException as error:
            cleanup_errors.append(type(error).__name__)
        try:
            runtime = None
            gc.collect()
        except BaseException as error:
            cleanup_errors.append(type(error).__name__)
        if cleanup_errors:
            try:
                publish(
                    root / "worker-cleanup-failure.json",
                    {
                        "error_types": cleanup_errors,
                        "original_error_type": type(original).__name__
                        if original is not None
                        else None,
                    },
                )
            except BaseException:
                pass
            if original is None:
                raise RuntimeError("a188_worker_cleanup_failed")


def validate_worker_terminal(root, expected_freeze_sha):
    root = Path(root)
    for path in root.rglob("*"):
        require(not path.is_symlink(), "terminal_symlink")
        require(
            path.name not in ("failure.json", "worker-failure.json", "worker-cleanup-failure.json")
            and not path.name.startswith(".pending-")
            and not path.name.endswith(".tmp"),
            "failure_accompanies_terminal",
        )
    inventory(root)  # Reject special files before replay.
    terminal = read_json(root / "worker-terminal.json")
    require(
        set(terminal)
        == set(
            "schema_version status freeze_sha256 acquisition_terminal_sha256 "
            "normal_return_sha256 body_entry_counts body_exit_events nested_counts_nonadditive "
            "profile_restored assets_before_sha256 assets_after_sha256 native_prepared_sha256 "
            "model_free_replay target_released technical_terminal_sha256 technical_normal_return_sha256 "
            "technical_prepared_sha256 model_inventory_before_sha256 model_inventory_after_sha256 "
            "body_entry_counts_by_stage".split()
        ),
        "worker_terminal_schema",
    )
    require(
        terminal["schema_version"] == SCHEMA
        and terminal["status"] == "completed"
        and terminal["freeze_sha256"] == expected_freeze_sha,
        "worker_terminal_binding",
    )
    for key in (
        "nested_counts_nonadditive",
        "profile_restored",
        "model_free_replay",
        "target_released",
    ):
        require(terminal[key] is True, "worker_terminal_flag")
    counts = terminal["body_entry_counts"]
    exits = terminal["body_exit_events"]
    require(
        type(counts) is dict and set(counts) == {"causal_lm_entries", "nested_decoder_entries"},
        "count_schema",
    )
    require(canonical(counts) == canonical(exits), "exit_event_counts")
    acquired = read_json(
        root / "acquisition/terminal.json", terminal["acquisition_terminal_sha256"]
    )
    stages = terminal["body_entry_counts_by_stage"]
    stage_counts_complete(counts, stages)
    acquisition_complete(acquired, stages["science"])
    technical = read_json(root / "technical/terminal.json", terminal["technical_terminal_sha256"])
    require(
        technical.get("schema_version") == "a188-bf16-forward-v1"
        and technical.get("status") == "completed",
        "technical_terminal",
    )
    technical_result = read_json(root / "technical/result.json", technical["result_sha256"])
    technical_complete(technical_result)
    technical_inventory = inventory(root / "technical")
    technical_inventory.pop("terminal.json")
    require(
        canonical(technical_inventory) == canonical(technical["artifact_sha256"]),
        "technical_inventory",
    )
    technical_return = read_json(
        root / "technical-normal-return.json", terminal["technical_normal_return_sha256"]
    )
    require(
        canonical(technical_return)
        == canonical(
            {
                "event": "run_technical_returned_normally",
                "terminal_sha256": terminal["technical_terminal_sha256"],
            }
        ),
        "technical_observed_return",
    )
    read_json(root / "technical-prepared.json", terminal["technical_prepared_sha256"])
    read_json(root / "model-inventory-before.json", terminal["model_inventory_before_sha256"])
    read_json(root / "model-inventory-after.json", terminal["model_inventory_after_sha256"])
    normal = read_json(root / "acquisition-normal-return.json", terminal["normal_return_sha256"])
    require(
        canonical(normal)
        == canonical(
            {
                "event": "run_acquisition_returned_normally",
                "terminal_sha256": terminal["acquisition_terminal_sha256"],
            }
        ),
        "observed_normal_return",
    )
    before = read_json(root / "assets-before.json", terminal["assets_before_sha256"])
    after = read_json(root / "assets-after.json", terminal["assets_after_sha256"])
    require(canonical(before) == canonical(after), "terminal_assets")
    read_json(root / "native-prepared.json", terminal["native_prepared_sha256"])
    header = read_json(root / "run.json")
    require(
        header.get("freeze_sha256") == expected_freeze_sha
        and type(header.get("pid")) is int
        and header["pid"] > 0,
        "worker_header",
    )
    expected_names = set()
    for kind, n in counts.items():
        for i in range(1, n + 1):
            name = f"{kind}_{i:04d}.json"
            expected_names.add(name)
            entry = read_json(root / "entries" / name)
            require(
                canonical(entry)
                == canonical(
                    {
                        "schema_version": "a188-body-entry-v1",
                        "kind": kind,
                        "stage": "technical" if i <= 2 else "science",
                        "entry_index": i,
                        "pid": header["pid"],
                    }
                ),
                "body_entry_receipt",
            )
    require(
        {p.name for p in (root / "entries").iterdir()} == expected_names, "body_entry_inventory"
    )
    return digest(root / "worker-terminal.json")


@contextmanager
def cooperative_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "lock_regular")
        os.fchmod(fd, 0o600)
        yield fd
    finally:
        os.close(fd)


def wait_for_resources(
    root,
    lock_fd,
    lifecycle,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    free_bytes=None,
    limits=LIMITS,
):
    """One bounded queue; True returns with lock held, False means expired."""
    root = Path(root)
    (root / "queue").mkdir(mode=0o700)
    free_bytes = free_bytes or (lambda: shutil.disk_usage(root).free)
    started, index = clock(), 0
    while True:
        elapsed = clock() - started
        if elapsed >= limits["queue_wall_seconds"]:
            return False
        locked = False
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            pass
        ready = False
        try:
            memory, disk = lifecycle.memory_available(), free_bytes()
            require(
                type(memory) is int and memory >= 0 and type(disk) is int and disk >= 0,
                "resource_sample",
            )
            ready = (
                locked
                and memory >= limits["minimum_available_bytes"]
                and disk >= limits["minimum_free_scratch_bytes"]
                and clock() - started < limits["queue_wall_seconds"]
            )
            publish(
                root / "queue" / f"check_{index:04d}.json",
                {
                    "elapsed_seconds": elapsed,
                    "cooperative_lock_acquired": locked,
                    "available_memory_bytes": memory,
                    "free_scratch_bytes": disk,
                    "ready": ready,
                },
            )
            if ready:
                return True
        finally:
            if locked and not ready:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        index += 1
        sleep(
            min(
                limits["queue_poll_seconds"],
                max(0, limits["queue_wall_seconds"] - (clock() - started)),
            )
        )


def supervise(path, expected_sha):
    frozen = verify_freeze(path, expected_sha)
    lifecycle, monitor = helper(), monitor_module().monitor
    root = Path(frozen["output_root"])
    root.mkdir(mode=0o700)  # Claim precedes queue, so expiry and failure cannot retry.
    process, started = None, time.monotonic()
    try:
        publish(
            root / "claim.json",
            {
                "schema_version": "a188-one-shot-claim-v1",
                "freeze_sha256": expected_sha,
                "supervisor_pid": os.getpid(),
                "queue_limit_seconds": LIMITS["queue_wall_seconds"],
            },
        )
        with (
            lifecycle.owned_interruptions(),
            cooperative_lock(root.parent / ".cpu-study.lock") as lock_fd,
        ):
            if not wait_for_resources(root, lock_fd, lifecycle):
                result = {
                    "status": "deferred",
                    "reason": "queue_expired",
                    "freeze_sha256": expected_sha,
                    "worker_started": False,
                    "target_body_entries": 0,
                }
                publish(root / "supervisor-result.json", result)
                return result
            verify_freeze(path, expected_sha)
            probe = lifecycle.open_pidfd(os.getpid())
            try:
                lifecycle.signal_pidfd(probe, 0)
            finally:
                os.close(probe)
            publish(
                root / "launch.json",
                {"freeze_sha256": expected_sha, "supervisor_pid": os.getpid(), "limits": LIMITS},
            )
            started = time.monotonic()
            with (root / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        NATIVE_PYTHON,
                        "-B",
                        str(
                            Path(frozen["source_root"]) / "scripts/run_bf16_arithmetic_task_load.py"
                        ),
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
                outcome = monitor(
                    process,
                    LIMITS,
                    started,
                    root,
                    lifecycle=lifecycle,
                    usage=lambda: raw_bytes(root),
                )
            outcome.update(freeze_sha256=expected_sha, status="failed", worker_terminal_sha256=None)
            if outcome["exit_code"] == 0 and outcome["stop_reason"] is None:
                verify_freeze(path, expected_sha)
                outcome["worker_terminal_sha256"] = validate_worker_terminal(
                    root / "worker", expected_sha
                )
                outcome["status"] = "completed"
            publish(root / "supervisor-result.json", outcome)
            return outcome
    except BaseException as error:
        try:
            publish(
                root / "supervisor-failure.json",
                {
                    "freeze_sha256": expected_sha,
                    "error_type": type(error).__name__,
                    "elapsed_seconds": time.monotonic() - started,
                    "worker_started": process is not None,
                },
            )
        except BaseException:
            pass
        raise
    finally:
        original = sys.exception()
        try:
            if process is not None and (
                process.poll() is None or lifecycle.process_session_snapshot(process.pid)
            ):
                lifecycle.terminate_owned_session(process, LIMITS["termination_grace_seconds"])
        except BaseException as error:
            try:
                publish(
                    root / "supervisor-cleanup-failure.json",
                    {
                        "error_type": type(error).__name__,
                        "original_error_type": type(original).__name__
                        if original is not None
                        else None,
                    },
                )
            except BaseException:
                pass
            if original is None:
                raise


def worker(path, expected_sha, parent):
    require(type(parent) is int and parent > 0, "parent_pid")
    lifecycle = helper()
    lifecycle.bind_parent_lifetime(parent)
    with lifecycle.owned_interruptions():
        frozen = verify_freeze(path, expected_sha)
        launch = read_json(Path(frozen["output_root"]) / "launch.json")
        require(
            launch.get("freeze_sha256") == expected_sha
            and type(launch.get("supervisor_pid")) is int
            and launch["supervisor_pid"] == parent == os.getppid(),
            "launch_parent",
        )
        return run_worker(frozen)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "worker"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    try:
        if args.mode == "worker":
            value = worker(args.freeze, args.freeze_sha256, args.parent)
            summary = {
                "status": value["status"],
                "body_entry_counts": value["body_entry_counts"],
                "nested_counts_nonadditive": True,
                "freeze_sha256": args.freeze_sha256,
            }
        else:
            require(args.parent is None, "supervisor_parent_argument")
            value = supervise(args.freeze, args.freeze_sha256)
            summary = {k: value[k] for k in ("status", "freeze_sha256")}
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if value["status"] == "completed" else 1
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
