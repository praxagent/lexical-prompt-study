"""A197 one-shot CPU full-candidate conditional scoring. Import is stdlib-only and inert.

Only pinned lifecycle/resource helpers are called; old study entrypoints are not.
Resource thresholds are sampled, not reservations. Nested decoder counts are
separate from (and never added to) causal-LM body entries.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import weakref

if not __debug__:
    raise RuntimeError("a197_optimization_forbidden")

NATIVE_PYTHON = "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python"
SCHEMA = "a197-native-worker-v1"
INSTRUMENT_QUALIFICATION_SHA256 = "228f5fe758022379ac189663153c3063dd5b3238e034b5fd31c8928c20651f47"
PREDECESSOR_VERIFICATION_SHA256 = "641c3c723d45ee43427983ca56a5238827fd4595558abfd32c4e00fd549a8b76"
PREDECESSOR_CLOSURE_SHA256 = "5d6b90faaee47b099fbd1d36f3471c36cd51032b2d3160d29bf47983d63598b2"
EXECUTION = {
    "device": "cpu",
    "dtype": "float32",
    "quantization": None,
    "attention": "sdpa_math",
    "threads": 8,
    "interop_threads": 1,
    "deterministic_algorithms": True,
    "autocast": False,
    "cache": False,
    "candidate_token_cap": 16,
    "planned_forwards": 128,
}
GEOMETRY = {"vocab_size": 128256, "hidden_width": 3072, "model_layers": 28, "context_limit": 131072}
LIMITS = {
    "minimum_available_bytes": 36 * 1024**3,
    "maximum_owned_rss_bytes": 24 * 1024**3,
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
ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONOPTIMIZE": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
}
PINS = {
    "scripts/run_native_prefix_qualification.py": "af4038d8422a33e67e507a230e16a89c5ce48002d9b6aa81888cfc838d4e94d0",
    "scripts/run_matched_prefix_prediction.py": "9ee523132bf54a84131052f50c35f1cec3a5cb1ec816302512d28b817df276cd",
    "src/lexical_prompt_study/__init__.py": "38df51bcc0f0179554da19316a656daaa4ba07dd998a704399b29e3085bce948",
    "src/lexical_prompt_study/landmark_evidence.py": "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14",
    "src/lexical_prompt_study/instruction_binding_runtime.py": "007d7dc3aea547e42564fd35295149cdaef7ea360dd43e573904f20c94bdebf2",
    "src/lexical_prompt_study/instruction_binding_tasks.py": "6dd403bb5e813d015c8bbe878cac59b908beff330a105855d444362c5bde855a",
    "src/lexical_prompt_study/special_token_inventory.py": "75e9425bbde7fefa56798613c6ad17a643f077463af1fce8b4c2c40e716391f0",
}
SOURCE_PATHS = tuple(PINS) + (
    "scripts/run_rule_cue_interaction.py",
    "src/lexical_prompt_study/rule_cue_interaction_tasks.py",
    "src/lexical_prompt_study/rule_cue_interaction_acquisition.py",
)
FREEZE_KEYS = set(
    "schema_version source_root sources target_config_path target_config_sha256 runtime_binding plan native_python limits output_root prepared_root files_sha256 public_commit selection_sha256 qualification_sha256 instrument_qualification_sha256 predecessor_verification_sha256 predecessor_closure_sha256".split()
)


def require(ok, code):
    if not ok:
        raise ValueError("a197_outer_" + code)


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


def digest(path):
    path = regular(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "hash_changed",
    )
    return h.hexdigest()


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


def module(path, name, wanted):
    require(digest(path) == wanted, "helper_hash")
    value = SimpleNamespace(__file__=str(path), __name__=name)
    # Execute authenticated source bytes without creating a bytecode cache.
    raw = regular(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == wanted, "executed_helper_bytes")
    exec(compile(raw, str(path), "exec"), value.__dict__)
    require(digest(path) == wanted, "helper_changed")
    return value


def helper():
    relative = "scripts/run_native_prefix_qualification.py"
    return module(Path(__file__).parent / Path(relative).name, "a197_lifecycle", PINS[relative])


def monitor_module():
    relative = "scripts/run_matched_prefix_prediction.py"
    return module(Path(__file__).parent / Path(relative).name, "a197_monitor", PINS[relative])


def read_json(path, expected=None):
    regular(path)
    value = helper().hash_bound_json(path, expected or digest(path))
    require(canonical(value) == Path(path).read_bytes(), "canonical_json")
    return value


def inventory(root):
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), "inventory_root")
    found = {}
    for path in root.rglob("*"):
        require(not path.is_symlink(), "inventory_link")
        mode = path.stat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "inventory_special")
        if stat.S_ISREG(mode):
            found[str(path.relative_to(root))] = digest(path)
    return found


def raw_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        require(not path.is_symlink(), "raw_link")
        mode = path.stat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "raw_special")
        if stat.S_ISREG(mode):
            total += path.stat().st_size
    return total


def verify_freeze(path, expected_sha, *, enforce_executable=True):
    path = Path(path)
    require(
        path.is_absolute() and path.name == "freeze.json" and is_hash(expected_sha), "freeze_path"
    )
    frozen = read_json(path, expected_sha)
    require(
        set(frozen) == FREEZE_KEYS and frozen["schema_version"] == "a197-native-freeze-v1",
        "freeze_schema",
    )
    require(frozen["native_python"] == NATIVE_PYTHON, "native_python")
    require(not enforce_executable or sys.executable == NATIVE_PYTHON, "actual_python")
    require(canonical(frozen["limits"]) == canonical(LIMITS), "limits")
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
    require(actual.pop(path.name, None) == expected_sha and actual == expected, "freeze_inventory")
    require(set(frozen["sources"]) == set(SOURCE_PATHS), "source_closure")
    require(
        all(expected.get("source/" + name) == sha for name, sha in frozen["sources"].items()),
        "source_binding",
    )
    require(all(frozen["sources"][name] == sha for name, sha in PINS.items()), "dependency_pins")
    require(
        Path(__file__).absolute() == source / "scripts/run_rule_cue_interaction.py",
        "executing_source",
    )
    config = Path(frozen["target_config_path"])
    require(
        config.parent == prepared and expected.get(config.name) == frozen["target_config_sha256"],
        "config_binding",
    )
    require(
        type(frozen["public_commit"]) is str
        and len(frozen["public_commit"]) == 40
        and all(c in "0123456789abcdef" for c in frozen["public_commit"]),
        "commit",
    )
    for key in ("selection_sha256", "qualification_sha256", "instrument_qualification_sha256"):
        require(is_hash(frozen[key]) and frozen[key] in expected.values(), key)
    require(
        frozen["instrument_qualification_sha256"] == INSTRUMENT_QUALIFICATION_SHA256,
        "instrument_qualification",
    )
    for key, name, wanted in (
        (
            "predecessor_verification_sha256",
            "A196-VERIFICATION.json",
            PREDECESSOR_VERIFICATION_SHA256,
        ),
        ("predecessor_closure_sha256", "A196-CLOSED.json", PREDECESSOR_CLOSURE_SHA256),
    ):
        require(is_hash(wanted) and frozen[key] == wanted and expected.get(name) == wanted, key)
    return frozen


def expected_runtime_binding(config, config_sha):
    return {
        "schema_version": "a197-runtime-binding-v1",
        "target_config_sha256": config_sha,
        **{
            k: config[k]
            for k in (
                "model_revision",
                "model_files_sha256",
                "chat_template_sha256",
                "runtime_versions",
                "seed",
            )
        },
        "actual_requested_execution": EXECUTION.copy(),
    }


def validate_config(config):
    require(
        set(config)
        == set(
            "schema_version model_id model_path model_revision model_files_sha256 chat_template_sha256 runtime_versions seed execution".split()
        ),
        "config_keys",
    )
    require(
        config["schema_version"] == "a197-cpu-fp32-config-v1"
        and config["model_id"] == "meta-llama/Llama-3.2-3B-Instruct"
        and config["model_revision"] == "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "target",
    )
    require(canonical(config["execution"]) == canonical(EXECUTION), "execution_policy")
    require(type(config["seed"]) is int and config["seed"] == 1729, "seed")
    require(
        Path(config["model_path"]).is_absolute() and is_hash(config["chat_template_sha256"]),
        "model_path",
    )
    require(
        type(config["model_files_sha256"]) is dict
        and len(config["model_files_sha256"]) == 10
        and all(Path(k).name == k and is_hash(v) for k, v in config["model_files_sha256"].items()),
        "assets",
    )


def validate_target(config, plan, binding, config_sha, lifecycle):
    validate_config(config)
    require(
        canonical(binding) == canonical(expected_runtime_binding(config, config_sha)),
        "runtime_binding",
    )
    require(canonical(plan["geometry"]) == canonical(GEOMETRY), "geometry")
    require(plan["runtime_binding_sha256"] == object_hash(binding), "plan_runtime")
    manifest = config["model_files_sha256"]
    subset = {
        k: v
        for k, v in manifest.items()
        if k in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    }
    require(
        canonical(plan["provenance"])
        == canonical(
            {
                "model_sha256": object_hash(manifest),
                "tokenizer_sha256": object_hash(subset),
                "chat_template_sha256": config["chat_template_sha256"],
            }
        ),
        "provenance",
    )
    meta = lifecycle.hash_bound_json(
        Path(config["model_path"]) / "config.json",
        manifest["config.json"],
        allow_snapshot_link=True,
    )
    for key, value in zip(
        ("vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings"),
        GEOMETRY.values(),
        strict=True,
    ):
        require(type(meta[key]) is int and meta[key] == value, "asset_geometry")
    generation = lifecycle.hash_bound_json(
        Path(config["model_path"]) / "generation_config.json",
        manifest["generation_config.json"],
        allow_snapshot_link=True,
    )
    eos = generation["eos_token_id"]
    eos = eos if type(eos) is list else [eos]
    require(
        bool(eos) and all(type(x) is int and 0 <= x < GEOMETRY["vocab_size"] for x in eos),
        "asset_eos",
    )
    require(canonical(plan["eos_token_ids"]) == canonical(sorted(set(eos))), "eos_binding")


def environment(frozen):
    env = os.environ.copy()
    env.update(ENV)
    env["PYTHONPATH"] = str(Path(frozen["source_root"]) / "src")
    for key in ("PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONHOME"):
        env.pop(key, None)
    return env


def imported_sources(frozen):
    source = Path(frozen["source_root"])
    for name, value in list(sys.modules.items()):
        if name == "lexical_prompt_study" or name.startswith("lexical_prompt_study."):
            relative = (
                "src/"
                + name.replace(".", "/")
                + ("/__init__.py" if name == "lexical_prompt_study" else ".py")
            )
            require(
                relative in frozen["sources"]
                and Path(value.__file__).absolute() == source / relative,
                "actual_import_path",
            )
            require(digest(value.__file__) == frozen["sources"][relative], "actual_import_hash")


def native_dependencies(frozen):
    sys.path.insert(0, str(Path(frozen["source_root"]) / "src"))
    from lexical_prompt_study import rule_cue_interaction_acquisition as core
    from lexical_prompt_study import instruction_binding_runtime as engine
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

    imported_sources(frozen)
    return (
        core,
        engine,
        torch,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaForCausalLM,
        LlamaModel,
    )


def model_metadata(model, torch):
    require(
        not torch.cuda.is_initialized()
        and not torch.is_autocast_enabled("cpu")
        and not torch.is_autocast_enabled("cuda"),
        "cpu_only",
    )
    require(torch.get_float32_matmul_precision() == "highest", "float32_precision")
    require(torch.get_num_threads() == 8 and torch.get_num_interop_threads() == 1, "threads")
    require(
        torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "determinism",
    )
    require(
        torch.backends.cuda.math_sdp_enabled()
        and not any(
            (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.cudnn_sdp_enabled(),
            )
        ),
        "math_attention",
    )
    require(
        not getattr(model, "is_loaded_in_4bit", False)
        and not getattr(model, "is_loaded_in_8bit", False)
        and not hasattr(model, "_orig_mod"),
        "unquantized_uncompiled",
    )
    require(
        all(
            not m.training and not type(m).__module__.startswith("bitsandbytes")
            for m in model.modules()
        ),
        "eval_modules",
    )
    require(
        model.config._attn_implementation == "sdpa"
        and model.config.output_hidden_states is False
        and model.config.output_attentions is False,
        "collectors_attention",
    )
    require(
        {
            k: getattr(model.config, attr)
            for k, attr in zip(
                GEOMETRY,
                ("vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings"),
                strict=True,
            )
        }
        == GEOMETRY,
        "actual_geometry",
    )
    tensors = []
    for kind, items in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in items:
            require(
                value.device.type == "cpu"
                and value.layout == torch.strided
                and not value.is_complex()
                and (not value.is_floating_point() or value.dtype == torch.float32),
                "tensor_dtype_device",
            )
            tensors.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            )
    require(any(x["kind"] == "parameter" for x in tensors), "empty_model")
    return {
        "geometry": GEOMETRY.copy(),
        "tensors": tensors,
        "tensor_metadata_sha256": object_hash(tensors),
        "weight_content_authenticated_by_this_table": False,
        "execution": EXECUTION.copy(),
    }


def load_cpu_fp32(config, torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer):
    require(all(os.environ.get(k) == v for k, v in ENV.items()), "environment")
    require(not torch.cuda.is_initialized(), "prior_cuda")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    path = Path(config["model_path"])
    meta = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    require(getattr(meta, "quantization_config", None) is None, "quantized_checkpoint")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        device_map={"": "cpu"},
        attn_implementation="sdpa",
    ).eval()
    model_metadata(model, torch)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    require(
        hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
        == config["chat_template_sha256"],
        "template",
    )
    eos = model.generation_config.eos_token_id
    eos = eos if type(eos) is list else [eos]
    require(
        bool(eos) and all(type(x) is int and 0 <= x < GEOMETRY["vocab_size"] for x in eos),
        "loaded_eos",
    )
    return SimpleNamespace(model=model, tokenizer=tokenizer, eos_ids=sorted(set(eos)))


class BodyProfile:
    def __init__(self, directory, classes):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        self.codes = {inspect.unwrap(cls.forward).__code__: name for cls, name in classes}
        self.counts = {name: 0 for name in self.codes.values()}
        self.stages = {stage: dict(self.counts) for stage in ("nonstudy", "science")}
        self.stage = "nonstudy"
        self.previous = sys.getprofile()
        self.cleanup_errors = []

    def set_stage(self, stage):
        require((self.stage, stage) == ("nonstudy", "science"), "stage_order")
        require(not any(self.counts.values()), "pre_scoring_body_entry")
        self.stage = stage

    def __call__(self, frame, event, arg):
        name = self.codes.get(frame.f_code)
        if name is None or event != "call":
            return
        self.counts[name] += 1
        self.stages[self.stage][name] += 1
        publish(
            self.directory / f"{name}_{self.counts[name]:04d}.json",
            {
                "schema_version": "a197-body-entry-v1",
                "kind": name,
                "entry_index": self.counts[name],
                "stage": self.stage,
                "pid": os.getpid(),
            },
        )
        require(
            self.stage != "nonstudy"
            and self.counts[name] <= 128
            and self.stages[self.stage][name] <= 128,
            "body_ceiling",
        )

    def __enter__(self):
        sys.setprofile(self)
        return self

    def __exit__(self, kind, original, tb):
        try:
            sys.setprofile(self.previous)
        except BaseException as error:
            self.cleanup_errors.append(type(error).__name__)
            if original is None:
                raise
        return False


def complete_counts(acquired, profiler):
    summary = acquired["summary"]
    require(
        acquired["status"] == "finished" and summary["entries_complete"] is True, "core_finished"
    )
    for key, value in (
        ("planned_measurements", 128),
        ("completed_measurements", 128),
        ("failed_measurements", 0),
        ("unattempted_measurements", 0),
        ("scoring_entries", 128),
        ("total_entries", 128),
    ):
        require(type(summary[key]) is int and summary[key] == value, "core_counts")
    for kind in ("causal_lm_entries", "nested_decoder_entries"):
        require(
            profiler.counts[kind] == 128
            and profiler.stages["nonstudy"][kind] == 0
            and profiler.stages["science"][kind] == 128,
            "body_dispatch_match",
        )


def safe_exception_type(error):
    name = type(error).__name__
    return (
        name
        if type(name) is str and name.isascii() and name.isidentifier() and len(name) <= 128
        else "unclassified_exception"
    )


def retained_preparation_failure(error, core):
    """Keep bounded source-enumerated metadata without replacing the primary error."""
    fallback = {
        "stage": "unclassified",
        "code": "preparation_metadata_invalid",
        "original_error_type": safe_exception_type(error),
    }
    try:
        value = core.preparation_failure(error)
        require(
            type(value) is dict and set(value) == {"stage", "code", "original_error_type"},
            "preparation_failure_schema",
        )
        require(
            type(value["stage"]) is str and value["stage"] in core.PREPARATION_STAGES,
            "preparation_failure_stage",
        )
        require(
            type(value["code"]) is str and value["code"] in core.PREPARATION_GUARD_CODES,
            "preparation_failure_code",
        )
        name = value["original_error_type"]
        require(
            type(name) is str and name.isascii() and name.isidentifier() and len(name) <= 128,
            "preparation_failure_type",
        )
        return dict(value)
    except BaseException:
        return fallback


def run_worker(frozen):
    root = Path(frozen["output_root"]) / "worker"
    root.mkdir(mode=0o700)
    lifecycle = helper()
    phase = "preflight"
    profiler = runtime = original = None
    try:
        publish(
            root / "run.json",
            {
                "schema_version": SCHEMA,
                "freeze_sha256": object_hash(frozen),
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "session": os.getsid(0),
            },
        )
        require(all(os.environ.get(k) == v for k, v in ENV.items()), "actual_environment")
        core, engine, torch, ac, am, at, lm, decoder = native_dependencies(frozen)
        config = lifecycle.hash_bound_json(
            frozen["target_config_path"], frozen["target_config_sha256"]
        )
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
        require(
            lifecycle.memory_available() >= LIMITS["minimum_available_bytes"]
            and shutil.disk_usage(root).free >= LIMITS["minimum_free_scratch_bytes"],
            "preload_resources",
        )
        profiler = BodyProfile(
            root / "entries", ((lm, "causal_lm_entries"), (decoder, "nested_decoder_entries"))
        )
        with profiler:
            phase = "target_load"
            runtime = load_cpu_fp32(config, torch, ac, am, at)
            require(
                type(runtime.model) is lm and type(runtime.model.model) is decoder, "model_classes"
            )
            require(canonical(runtime.eos_ids) == canonical(plan["eos_token_ids"]), "model_eos")
            publish(root / "model-before.json", model_metadata(runtime.model, torch))
            phase = "native_preparation"
            prepared = core.prepare_inputs(plan, runtime.tokenizer)
            publish(root / "native-prepared.json", prepared)
            require(not any(profiler.counts.values()), "preparation_forward")
            phase = "acquisition"
            acquired = core.run_acquisition(
                runtime.model,
                runtime.tokenizer,
                plan=plan,
                prepared=prepared,
                runtime_binding=frozen["runtime_binding"],
                directory=root / "acquisition",
                event=profiler.set_stage,
            )
            normal = publish(
                root / "acquisition-normal-return.json",
                {
                    "event": "run_acquisition_returned_normally",
                    "terminal_sha256": acquired["terminal_sha256"],
                },
            )
            phase = "replay"
            counts_before = canonical(profiler.counts)
            require(
                canonical(core.load_acquisition(root / "acquisition", plan)) == canonical(acquired),
                "core_replay",
            )
            complete_counts(acquired, profiler)
            phase = "postflight"
            publish(root / "model-after.json", model_metadata(runtime.model, torch))
            require(
                read_json(root / "model-before.json") == read_json(root / "model-after.json"),
                "model_changed",
            )
            require(
                canonical(engine.runtime_versions()) == canonical(config["runtime_versions"]),
                "runtime_after",
            )
            after = lifecycle.asset_check(config)
            require(canonical(after) == canonical(assets), "assets_changed")
            publish(root / "assets-after.json", after)
            imported_sources(frozen)
            verify_freeze(Path(frozen["prepared_root"]) / "freeze.json", object_hash(frozen))
            ref = weakref.ref(runtime.model)
            runtime = None
            gc.collect()
            require(ref() is None, "model_not_released")
            require(canonical(profiler.counts) == counts_before, "extra_forward")
        require(
            sys.getprofile() is profiler.previous and not profiler.cleanup_errors,
            "profile_restored",
        )
        terminal = {
            "schema_version": SCHEMA,
            "status": "completed",
            "freeze_sha256": object_hash(frozen),
            "acquisition_terminal_sha256": acquired["terminal_sha256"],
            "normal_return_sha256": normal,
            "body_entry_counts": profiler.counts,
            "body_stage_counts": profiler.stages,
            "nested_counts_nonadditive": True,
            "profile_restored": True,
            "target_released": True,
            "model_free_replay": True,
            "artifact_sha256": inventory(root),
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
                    "error_type": safe_exception_type(error),
                    "preparation_failure": retained_preparation_failure(error, core)
                    if phase == "native_preparation"
                    else None,
                    "body_entry_counts": None if profiler is None else profiler.counts,
                    "body_stage_counts": None if profiler is None else profiler.stages,
                    "nested_counts_nonadditive": True,
                },
            )
        except BaseException:
            pass
        raise
    finally:
        runtime = None
        errors = [] if profiler is None else list(profiler.cleanup_errors)
        try:
            if profiler is not None:
                sys.setprofile(profiler.previous)
            gc.collect()
        except BaseException as error:
            errors.append(type(error).__name__)
        if errors:
            try:
                publish(
                    root / "worker-cleanup-failure.json",
                    {
                        "error_types": errors,
                        "original_error_type": None
                        if original is None
                        else type(original).__name__,
                    },
                )
            except BaseException:
                pass
            if original is None:
                raise RuntimeError("a197_cleanup_failure")


def validate_worker_terminal(root, freeze_sha):
    root = Path(root)
    terminal = read_json(root / "worker-terminal.json")
    require(
        set(terminal)
        == set(
            "schema_version status freeze_sha256 acquisition_terminal_sha256 normal_return_sha256 body_entry_counts body_stage_counts nested_counts_nonadditive profile_restored target_released model_free_replay artifact_sha256".split()
        ),
        "terminal_keys",
    )
    require(
        terminal["schema_version"] == SCHEMA
        and terminal["status"] == "completed"
        and terminal["freeze_sha256"] == freeze_sha,
        "terminal_binding",
    )
    for key in (
        "nested_counts_nonadditive",
        "profile_restored",
        "target_released",
        "model_free_replay",
    ):
        require(terminal[key] is True, key)
    actual = inventory(root)
    actual.pop("worker-terminal.json")
    require(actual == terminal["artifact_sha256"], "terminal_inventory")
    require(
        not any("failure" in Path(p).name or Path(p).name.startswith(".") for p in actual),
        "failure_with_terminal",
    )
    require(
        read_json(root / "acquisition-normal-return.json", terminal["normal_return_sha256"])
        == {
            "event": "run_acquisition_returned_normally",
            "terminal_sha256": terminal["acquisition_terminal_sha256"],
        },
        "normal_return",
    )
    counts, stages = terminal["body_entry_counts"], terminal["body_stage_counts"]
    require(
        set(counts) == {"causal_lm_entries", "nested_decoder_entries"}
        and set(stages) == {"nonstudy", "science"},
        "body_keys",
    )
    for stage in stages:
        require(set(stages[stage]) == set(counts), "stage_keys")
    header = read_json(root / "run.json")
    require(
        set(header) == {"schema_version", "freeze_sha256", "pid", "parent_pid", "session"}
        and header["schema_version"] == SCHEMA
        and header["freeze_sha256"] == freeze_sha
        and all(type(header[k]) is int and header[k] > 0 for k in ("pid", "parent_pid", "session")),
        "worker_header",
    )
    launch = read_json(root.parent / "launch.json")
    child_launch = read_json(root.parent / "worker-launch.json")
    require(
        set(launch) == {"freeze_sha256", "supervisor", "limits"}
        and launch["freeze_sha256"] == freeze_sha
        and canonical(launch["limits"]) == canonical(LIMITS)
        and set(child_launch) == {"freeze_sha256", "worker"}
        and child_launch["freeze_sha256"] == freeze_sha,
        "terminal_launch_binding",
    )
    supervisor, child = launch["supervisor"], child_launch["worker"]
    for owned in (supervisor, child):
        require(
            type(owned) is dict
            and set(owned) == {"pid", "parent_pid", "session", "start_ticks"}
            and all(type(value) is int and value > 0 for value in owned.values()),
            "terminal_process_identity",
        )
    require(
        child["parent_pid"] == supervisor["pid"]
        and child["session"] == child["pid"]
        and canonical({key: header[key] for key in ("pid", "parent_pid", "session")})
        == canonical({key: child[key] for key in ("pid", "parent_pid", "session")}),
        "terminal_worker_lineage",
    )
    require(
        digest(root / "acquisition/terminal.json") == terminal["acquisition_terminal_sha256"],
        "core_terminal_binding",
    )
    names = set()
    for kind, n in counts.items():
        require(type(n) is int and n == 128, "body_count")
        require(
            all(type(stages[s][kind]) is int for s in stages)
            and stages["nonstudy"][kind] == 0
            and stages["science"][kind] == n,
            "stage_count",
        )
        for i in range(1, n + 1):
            name = f"{kind}_{i:04d}.json"
            names.add(name)
            require(
                canonical(read_json(root / "entries" / name))
                == canonical(
                    {
                        "schema_version": "a197-body-entry-v1",
                        "kind": kind,
                        "entry_index": i,
                        "stage": "science",
                        "pid": header["pid"],
                    }
                ),
                "entry_receipt",
            )
    require(
        counts["causal_lm_entries"] == counts["nested_decoder_entries"]
        and {p.name for p in (root / "entries").iterdir()} == names,
        "entry_inventory",
    )
    return digest(root / "worker-terminal.json")


@contextmanager
def cooperative_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "lock_regular")
        yield fd
    finally:
        os.close(fd)


def wait_for_resources(
    root,
    lock_fd,
    lifecycle,
    *,
    limits=LIMITS,
    clock=time.monotonic,
    sleep=time.sleep,
    free_bytes=None,
):
    root = Path(root)
    (root / "queue").mkdir(mode=0o700)
    free_bytes = free_bytes or (lambda: shutil.disk_usage(root).free)
    started, index = clock(), 0
    while clock() - started < limits["queue_wall_seconds"]:
        locked = False
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            pass
        ready = False
        try:
            memory, disk = lifecycle.memory_available(), free_bytes()
            elapsed = clock() - started
            require(
                type(memory) is int and memory >= 0 and type(disk) is int and disk >= 0,
                "resource_sample",
            )
            ready = (
                locked
                and memory >= limits["minimum_available_bytes"]
                and disk >= limits["minimum_free_scratch_bytes"]
                and elapsed < limits["queue_wall_seconds"]
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
    return False


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


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
                "schema_version": "a197-one-shot-claim-v1",
                "freeze_sha256": expected_sha,
                "supervisor": identity(os.getpid()),
                "queue_limit_seconds": LIMITS["queue_wall_seconds"],
            },
        )
        with (
            lifecycle.owned_interruptions(),
            cooperative_lock(root.parent / ".cpu-study.lock") as fd,
        ):
            if not wait_for_resources(root, fd, lifecycle):
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
            launch = {
                "freeze_sha256": expected_sha,
                "supervisor": identity(os.getpid()),
                "limits": LIMITS,
            }
            publish(root / "launch.json", launch)
            started = time.monotonic()
            with (root / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        NATIVE_PYTHON,
                        "-B",
                        str(Path(frozen["source_root"]) / "scripts/run_rule_cue_interaction.py"),
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
                    "worker_lineage",
                )
                publish(
                    root / "worker-launch.json", {"freeze_sha256": expected_sha, "worker": child}
                )
                result = monitor_module().monitor(
                    process,
                    LIMITS,
                    started,
                    root,
                    lifecycle=lifecycle,
                    usage=lambda: raw_bytes(root),
                )
            result.update(status="failed", freeze_sha256=expected_sha, worker_terminal_sha256=None)
            if result["exit_code"] == 0 and result["stop_reason"] is None:
                verify_freeze(path, expected_sha)
                result["worker_terminal_sha256"] = validate_worker_terminal(
                    root / "worker", expected_sha
                )
                result["status"] = "completed"
            publish(root / "supervisor-result.json", result)
            return result
    except BaseException as error:
        try:
            publish(
                root / "supervisor-failure.json",
                {
                    "freeze_sha256": expected_sha,
                    "error_type": type(error).__name__,
                    "worker_started": process is not None,
                    "elapsed_seconds": time.monotonic() - started,
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
                        "original_error_type": None
                        if original is None
                        else type(original).__name__,
                    },
                )
            except BaseException:
                pass
            if original is None:
                raise


def worker(path, expected_sha, parent):
    require(type(parent) is int and parent > 0, "parent")
    lifecycle = helper()
    lifecycle.bind_parent_lifetime(parent)
    with lifecycle.owned_interruptions():
        frozen = verify_freeze(path, expected_sha)
        launch = read_json(Path(frozen["output_root"]) / "launch.json")
        require(
            launch["freeze_sha256"] == expected_sha
            and canonical(launch["supervisor"]) == canonical(identity(parent))
            and parent == os.getppid()
            and os.getsid(0) == os.getpid(),
            "parent_lineage",
        )
        return run_worker(frozen)


def main():
    os.umask(0o077)
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("supervise", "worker"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    try:
        result = (
            supervise(args.freeze, args.freeze_sha256)
            if args.mode == "supervise"
            else worker(args.freeze, args.freeze_sha256, args.parent)
        )
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1
    print(json.dumps({"status": result["status"], "model_payloads_printed": False}, sort_keys=True))
    return 0 if result["status"] in ("completed", "deferred") else 1


if __name__ == "__main__":
    raise SystemExit(main())
