#!/usr/bin/env python3
"""A186 frozen sequential native worker; no work occurs on import.

The outer supervisor owns resource enforcement and execution selection. This
worker owns one target load/acquisition, one encoder load, and one fit handoff.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import subprocess
import sys
import weakref

SCHEMA = "a186-native-worker-v1"
PYTHON = "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python"
HELPER_SHA = "af4038d8422a33e67e507a230e16a89c5ce48002d9b6aa81888cfc838d4e94d0"
LOADER_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
SEMANTIC_FILES = {
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
}
REQUIRED_FREEZE = {
    "source_root",
    "sources",
    "target_config_path",
    "target_config_sha256",
    "runtime_binding",
    "plan",
    "semantic_assets",
    "encoder_binding_sha256",
    "encoder_binding",
    "encoder_runtime",
    "fit_runtime",
    "fit_python",
    "fit_worker",
    "native_python",
    "output_root",
}


def require(ok, reason):
    if not ok:
        raise ValueError("a186_worker_" + reason)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def tree_hash(value):
    """Type-preserving comparison of JSON metadata and immutable byte payloads."""

    def bind(item):
        if type(item) is bytes:
            return {"bytes_sha256": hashlib.sha256(item).hexdigest(), "byte_length": len(item)}
        if type(item) is dict:
            return {key: bind(child) for key, child in item.items()}
        if type(item) is list:
            return [bind(child) for child in item]
        return item

    return object_hash(bind(value))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "module_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_helper():
    path = Path(__file__).absolute().with_name("run_native_prefix_qualification.py")
    require(not path.is_symlink() and digest(path) == HELPER_SHA, "helper_source")
    return _module(path, "a186_pinned_supervisor_helpers")


def expected_runtime_binding(config, config_sha):
    binding = pinned_helper().expected_runtime_binding(config, config_sha)
    binding["schema_version"] = "a186-runtime-binding-v1"
    binding["actual_requested_execution"]["sampled_token_cap"] = 64
    binding["note"] = (
        "Legacy NF4/cap64 descriptors remain unchanged as loader input bindings; "
        "A186 uses original CPU FP32 SDPA cache-free greedy generation capped at64."
    )
    return binding


def verify_sources(freeze):
    require(type(freeze) is dict and REQUIRED_FREEZE <= set(freeze), "freeze_fields")
    root = Path(freeze["source_root"])
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "source_root")
    sources = freeze["sources"]
    require(type(sources) is dict and bool(sources), "sources")
    required = {
        "scripts/matched_prefix_native_worker.py",
        "scripts/run_native_prefix_qualification.py",
        "scripts/run_cpu_reference_audit.py",
        "scripts/matched_prefix_fit_worker.py",
        *{
            "src/lexical_prompt_study/" + name + ".py"
            for name in (
                "landmark_evidence",
                "prefix_only_capture",
                "native_prefix_qualification",
                "instruction_binding_runtime",
                "matched_prefix_tasks",
                "matched_prefix_acquisition",
                "matched_prefix_encoder",
                "matched_prefix_prediction",
            )
        },
    }
    require(required <= set(sources), "source_closure")
    for relative, wanted in sources.items():
        pure = PurePosixPath(relative)
        require(
            type(relative) is str
            and not pure.is_absolute()
            and ".." not in pure.parts
            and str(pure) == relative,
            "source_relative_path",
        )
        path = root / relative
        require(
            all(not p.is_symlink() for p in [path, *path.parents] if p != root.parent),
            "source_link",
        )
        require(stat.S_ISREG(path.lstat().st_mode) and digest(path) == wanted, "source_hash")
    require(
        Path(__file__).absolute() == root / "scripts/matched_prefix_native_worker.py",
        "actual_worker_source",
    )
    require(sources["scripts/run_native_prefix_qualification.py"] == HELPER_SHA, "helper_pin")
    require(sources["scripts/run_cpu_reference_audit.py"] == LOADER_SHA, "loader_pin")
    require(
        freeze["native_python"] == PYTHON and str(Path(sys.executable).absolute()) == PYTHON,
        "native_python",
    )
    require(freeze["fit_worker"] == "scripts/matched_prefix_fit_worker.py", "fit_worker_path")
    require(Path(freeze["fit_python"]).is_absolute(), "fit_python")
    require(
        object_hash(freeze["encoder_binding"]) == freeze["encoder_binding_sha256"],
        "encoder_binding",
    )
    return root


def frozen_imports(freeze, source):
    sys.path.insert(0, str(source / "src"))
    from lexical_prompt_study import instruction_binding_runtime as engine
    from lexical_prompt_study import landmark_evidence as evidence
    from lexical_prompt_study import matched_prefix_acquisition as acquisition
    from lexical_prompt_study import matched_prefix_encoder as encoder
    from lexical_prompt_study import matched_prefix_tasks as tasks

    for module in (engine, evidence, acquisition, encoder, tasks):
        relative = "src/lexical_prompt_study/" + module.__name__.rsplit(".", 1)[1] + ".py"
        require(Path(module.__file__).absolute() == source / relative, "actual_import_path")
        require(digest(module.__file__) == freeze["sources"][relative], "actual_import_hash")
    return evidence, engine, acquisition, encoder, tasks


def semantic_asset_check(spec):
    require(type(spec) is dict and set(spec) == {"path", "files"}, "semantic_spec")
    root = Path(spec["path"])
    require(root.is_absolute() and not root.is_symlink() and root.is_dir(), "semantic_root")
    require(
        set(spec["files"]) == SEMANTIC_FILES == {p.name for p in root.iterdir()},
        "semantic_inventory",
    )
    observed = {}
    for name, expected in spec["files"].items():
        require(type(expected) is dict and set(expected) == {"bytes", "sha256"}, "semantic_receipt")
        path = root / name
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode), "semantic_regular")
        value = digest(path)
        after = path.lstat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "semantic_changed",
        )
        require(
            type(expected["bytes"]) is int
            and after.st_size == expected["bytes"]
            and value == expected["sha256"],
            "semantic_hash",
        )
        observed[name] = {"bytes": after.st_size, "sha256": value}
    return observed


def validate_target(config, plan, binding, config_sha, helper):
    require(
        canonical(binding) == canonical(expected_runtime_binding(config, config_sha)),
        "selected_runtime",
    )
    manifest = plan["manifest"]
    require(
        manifest["capture"]["model_layers"] == 32
        and manifest["capture"]["layer_index"] == 31
        and manifest["capture"]["hidden_width"] == 4096
        and manifest["vocab_size"] == 128256,
        "native_geometry",
    )
    require(object_hash(binding) == plan["bindings"]["runtime_binding_sha256"], "plan_runtime")
    require(
        object_hash(config["model_files_sha256"]) == manifest["provenance"]["model_sha256"],
        "target_assets",
    )
    subset = {
        k: v
        for k, v in config["model_files_sha256"].items()
        if k in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    }
    require(object_hash(subset) == manifest["provenance"]["tokenizer_sha256"], "target_tokenizer")
    require(
        config["chat_template_sha256"] == manifest["provenance"]["chat_template_sha256"],
        "target_template",
    )
    generation = helper.hash_bound_json(
        Path(config["model_path"]) / "generation_config.json",
        config["model_files_sha256"]["generation_config.json"],
        allow_snapshot_link=True,
    )
    eos = generation["eos_token_id"]
    eos = eos if type(eos) is list else [eos]
    require(bool(eos) and all(type(t) is int and 0 <= t < 128256 for t in eos), "eos_types")
    require(sorted(set(eos)) == manifest["eos_token_ids"], "native_eos")


def _float32(raw, width):
    require(type(raw) is bytes and len(raw) == width * 4, "vector_bytes")
    values = list(struct.unpack("<" + "f" * width, raw))
    require(all(math.isfinite(v) for v in values), "finite_vector")
    return values


def assemble_fit_request(freeze, views, acquisition, encoding, tasks):
    """Join strict common-time bytes without exposing labels to an encoder."""
    roster = tasks.prediction_roster(tasks.build_roster())
    ids = [row["observation_id"] for row in roster]
    require(set(views) == {"semantic_packets", "internal_packets", "labels"}, "view_keys")
    semantic, internal, records = (
        views["semantic_packets"],
        views["internal_packets"],
        acquisition["records"],
    )
    require(
        all(type(x) is list and len(x) == 112 for x in (semantic, internal, records)),
        "view_coverage",
    )
    require(
        set(views["labels"]) == set(ids) == set(encoding["embeddings"]), "label_embedding_coverage"
    )
    packets, reached = [], {}
    for row, text, vector, record in zip(roster, semantic, internal, records, strict=True):
        key = row["observation_id"]
        require(
            set(text) == {"schema_version", "observation_id", "common", "features_sha256"}
            and text["schema_version"] == "a186-common-prefix-v1"
            and set(vector)
            == {
                "schema_version",
                "observation_id",
                "residual_fp32le",
                "residual_shape",
                "features_sha256",
            }
            and vector["schema_version"] == "a186-internal-prefix-v1",
            "handoff_packet_keys",
        )
        require(
            text["observation_id"] == vector["observation_id"] == record["observation_id"] == key,
            "handoff_order",
        )
        require(
            record["status"] == "completed" and type(record["prefix_available"]) is bool,
            "record_complete",
        )
        reached[key] = record["prefix_available"]
        label = views["labels"][key]
        require(
            type(label) is int and label in (0, 1) and label == record["label"], "complete_label"
        )
        common = text["common"]
        require((common is not None) == reached[key], "common_reached")
        if common is not None:
            require(
                set(common)
                == {
                    "rendered_prompt_utf8",
                    "prompt_token_ids",
                    "prefix_utf8",
                    "prefix_token_ids",
                    "completed_fields",
                    "prefix_known_error",
                },
                "common_keys",
            )
            common = dict(common)
            for field in ("rendered_prompt_utf8", "prefix_utf8"):
                require(type(common[field]) is bytes, "common_bytes")
                common[field] = common[field].decode("utf-8", errors="strict")
            check = tasks.prefix_features(row["core_index"], common["prefix_utf8"])
            require(
                type(common["completed_fields"]) is int
                and common["completed_fields"] == check["completed_fields"]
                and common["prefix_known_error"] is check["prefix_known_error"],
                "common_check",
            )
        embeddings = encoding["embeddings"][key]
        require(set(embeddings) == {"prompt", "prefix"}, "embedding_views")
        if common is None:
            require(
                all(raw is None for raw in embeddings.values())
                and vector["residual_fp32le"] is None,
                "unreached_features",
            )
            semantic_values = internal_values = None
        else:
            semantic_values = _float32(embeddings["prompt"], 384) + _float32(
                embeddings["prefix"], 384
            )
            require(vector["residual_shape"] == [1, 4096], "internal_shape")
            internal_values = _float32(vector["residual_fp32le"], 4096)
        packets.append(
            {
                "schema_version": "a186-prediction-features-v1",
                "observation_id": key,
                "common": common,
                "text_semantic": semantic_values,
                "internal": internal_values,
            }
        )
    bindings = {
        "plan_sha256": object_hash(freeze["plan"]),
        "acquisition_terminal_sha256": acquisition["terminal_sha256"],
        "encoder_terminal_sha256": encoding["terminal_sha256"],
        "feature_packets_sha256": object_hash(packets),
        "labels_sha256": object_hash(views["labels"]),
        "landmark_reached_sha256": object_hash(reached),
        "fit_runtime_sha256": object_hash(freeze["fit_runtime"]),
        "prediction_source_sha256": freeze["sources"][
            "src/lexical_prompt_study/matched_prefix_prediction.py"
        ],
    }
    return {
        "roster": roster,
        "feature_packets": packets,
        "labels": views["labels"],
        "bindings": bindings,
        "landmark_reached": reached,
    }


def _load_encoder(spec, torch):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec["path"], local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        spec["path"],
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        device_map={"": "cpu"},
        attn_implementation="eager",
    ).eval()
    require(
        type(model).__module__ == "transformers.models.bert.modeling_bert"
        and type(model).__name__ == "BertModel",
        "encoder_class",
    )
    cfg = model.config
    require(
        cfg.hidden_size == 384
        and cfg.num_hidden_layers == 12
        and cfg.num_attention_heads == 12
        and cfg.max_position_embeddings == 512
        and cfg.vocab_size == 30522,
        "encoder_geometry",
    )
    return model, tokenizer


def validate_fit_return(root, request_sha, freeze, expected_pid, e):
    """Check the separate worker's completed writer envelope without a refit."""
    require(not root.is_symlink() and root.is_dir(), "fit_directory")
    require(
        {p.name for p in root.iterdir()}
        == {
            "run.json",
            "request.json",
            "events",
            "entries",
            "result.json",
            "normal-return.json",
            "terminal.json",
        },
        "fit_inventory",
    )
    for path in root.rglob("*"):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "fit_artifact_type")

    def read(name):
        return e._read_json(e._read_file(root / name))

    terminal, normal, header = read("terminal.json"), read("normal-return.json"), read("run.json")
    require(
        header["schema_version"] == "a186-fit-worker-v1"
        and header["request_sha256"] == request_sha
        and header["pid"] == expected_pid
        and header["parent_pid"] == os.getpid()
        and header["worker_source_sha256"] == freeze["sources"][freeze["fit_worker"]]
        and header["prediction_source_sha256"]
        == freeze["sources"]["src/lexical_prompt_study/matched_prefix_prediction.py"]
        and canonical(header["fit_runtime"]) == canonical(freeze["fit_runtime"]),
        "fit_header",
    )
    require(
        all(type(header[key]) is int and header[key] > 0 for key in ("pid", "parent_pid")),
        "fit_header_pid_types",
    )
    for name in ("actual_fit_entries", "fit_claims", "fit_body_returns", "event_count"):
        require(type(terminal[name]) is int and terminal[name] >= 0, "fit_counters")
    require(
        terminal["schema_version"] == "a186-fit-worker-v1"
        and terminal["status"] == "completed"
        and terminal["request_sha256"] == request_sha
        and terminal["run_sha256"] == object_hash(header)
        and terminal["model_free_replay"] is True
        and terminal["profile_restored"] is True
        and terminal["actual_fit_entries"]
        == terminal["fit_claims"]
        == terminal["fit_body_returns"]
        <= 2,
        "fit_terminal",
    )
    require(
        digest(root / "request.json") == request_sha
        and digest(root / "result.json") == terminal["result_sha256"],
        "fit_files",
    )
    require(
        object_hash(normal) == terminal["normal_return_sha256"]
        and normal["fit_predict_returned"] is True
        and normal["request_sha256"] == request_sha
        and normal["run_sha256"] == object_hash(header)
        and type(normal["actual_fit_entries"]) is int
        and normal["actual_fit_entries"] == terminal["actual_fit_entries"]
        and normal["returned_result_sha256"] == terminal["result_sha256"]
        and normal["terminal_error"] is None,
        "fit_normal_return",
    )
    for folder, key, names in (
        ("events", "event_sha256", [f"event_{i:03d}.json" for i in range(terminal["event_count"])]),
        (
            "entries",
            "entry_sha256",
            [f"fit_{i:02d}.json" for i in range(1, terminal["actual_fit_entries"] + 1)],
        ),
    ):
        hashes = terminal[key]
        require(
            type(hashes) is list
            and len(hashes) == len(names)
            and {p.name for p in (root / folder).iterdir()} == set(names),
            "fit_child_inventory",
        )
        require(
            all(
                digest(root / folder / name) == wanted
                for name, wanted in zip(names, hashes, strict=True)
            ),
            "fit_child_hash",
        )
    return terminal


def run(freeze, directory):
    source = verify_sources(freeze)
    helper = pinned_helper()
    e, engine, core, encoder, tasks = frozen_imports(freeze, source)
    root = Path(directory).absolute()
    require(root == Path(freeze["output_root"]) / "worker", "worker_output")
    root = e._directory(root, create=True)
    counts = {
        "causal_lm_entries": 0,
        "nested_decoder_entries": 0,
        "bert_model_entries": 0,
        "nested_bert_encoder_entries": 0,
    }
    phase, previous_profile = "preflight", sys.getprofile()
    fit_started, fit_entries, original_error = False, 0, None

    def write(name, value):
        e._publish(root / name, e.canonical(value))

    try:
        require(previous_profile is None, "owned_profiler")
        write(
            "run.json",
            {"schema_version": SCHEMA, "freeze_sha256": object_hash(freeze), "pid": os.getpid()},
        )
        for key, value in {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }.items():
            os.environ[key] = value
        config = helper.hash_bound_json(
            freeze["target_config_path"], freeze["target_config_sha256"]
        )
        plan = core.validate_plan(freeze["plan"])
        validate_target(
            config, plan, freeze["runtime_binding"], freeze["target_config_sha256"], helper
        )
        require(engine.runtime_versions() == config["runtime_versions"], "target_runtime")
        require(encoder.runtime_descriptor() == freeze["encoder_runtime"], "encoder_runtime")
        assets = helper.asset_check(config)
        semantic_assets = semantic_asset_check(freeze["semantic_assets"])
        write("assets-before.json", {"target": assets, "semantic": semantic_assets})
        require(helper.memory_available() >= 64 * 1024**3, "target_preload_memory")
        loader = _module(source / "scripts/run_cpu_reference_audit.py", "a186_cpu_reference_loader")
        import torch
        from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
        from transformers.models.bert.modeling_bert import BertModel, BertEncoder

        codes = {
            inspect.unwrap(cls.forward).__code__: name
            for cls, name in (
                (LlamaForCausalLM, "causal_lm_entries"),
                (LlamaModel, "nested_decoder_entries"),
                (BertModel, "bert_model_entries"),
                (BertEncoder, "nested_bert_encoder_entries"),
            )
        }
        require(len(codes) == 4, "profile_code_identity")

        def profile(frame, event, _arg):
            if event == "call" and frame.f_code in codes:
                name = codes[frame.f_code]
                counts[name] += 1
                require(counts[name] <= (7280 if "bert" not in name else 1792), "body_ceiling")

        sys.setprofile(profile)
        phase = "target_load"
        runtime = loader.load_cpu_reference(config, engine)
        require(not any(counts.values()), "forward_during_target_load")
        require(
            type(runtime.model) is LlamaForCausalLM and type(runtime.model.model) is LlamaModel,
            "target_class",
        )
        require(runtime.eos_ids == plan["manifest"]["eos_token_ids"], "target_eos")
        phase = "native_preparation"
        prepared = core.prepare_inputs(plan, runtime.tokenizer)
        require(not any(counts.values()), "forward_during_preparation")
        write("native-prepared.json", prepared)
        phase = "acquisition"
        acquired = core.run_acquisition(
            runtime.model,
            runtime.tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=freeze["runtime_binding"],
            directory=root / "acquisition",
        )
        write(
            "acquisition-normal-return.json",
            {
                "event": "run_acquisition_returned_normally",
                "terminal_sha256": acquired["terminal_sha256"],
            },
        )
        require(
            canonical(acquired) == canonical(core.load_acquisition(root / "acquisition", plan)),
            "acquisition_replay",
        )
        require(
            acquired["status"] == "finished" and acquired["summary"]["completed_rows"] == 112,
            "acquisition_complete",
        )
        require(
            counts["causal_lm_entries"]
            == counts["nested_decoder_entries"]
            == acquired["summary"]["total_entries"]
            <= 7280,
            "target_counts",
        )
        require(
            not counts["bert_model_entries"] and not counts["nested_bert_encoder_entries"],
            "premature_encoder",
        )
        views = core.feature_views(root / "acquisition", plan)
        after = helper.asset_check(config)
        require(after == assets, "target_assets_changed")
        write("target-assets-after.json", after)
        target_reference = weakref.ref(runtime.model)
        del runtime
        gc.collect()
        require(target_reference() is None, "target_reference_retained")
        write(
            "target-released.json",
            {
                "event": "target_weakref_dead_after_owned_reference_release_gc",
                "body_counts": counts,
            },
        )
        phase = "encoder_load"
        require(helper.memory_available() >= 64 * 1024**3, "encoder_preload_memory")
        require(
            semantic_asset_check(freeze["semantic_assets"]) == semantic_assets,
            "encoder_assets_before_load",
        )
        target_counts = dict(counts)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.random.default_generator.manual_seed(20260915)
        model, tokenizer = _load_encoder(freeze["semantic_assets"], torch)
        require(counts == target_counts, "forward_during_encoder_load")
        phase = "encoding"
        encoded = encoder.encode_packets(
            model,
            tokenizer,
            views["semantic_packets"],
            binding=freeze["encoder_binding_sha256"],
            directory=root / "encoding",
        )
        write(
            "encoder-normal-return.json",
            {
                "event": "encode_packets_returned_normally",
                "terminal_sha256": encoded["terminal_sha256"],
            },
        )
        replay = encoder.load_encoding(
            root / "encoding", views["semantic_packets"], binding=freeze["encoder_binding_sha256"]
        )
        require(tree_hash(encoded) == tree_hash(replay), "encoder_replay")
        require(
            counts["bert_model_entries"]
            == counts["nested_bert_encoder_entries"]
            == encoded["entries"]
            <= 1792,
            "encoder_counts",
        )
        require(
            all(
                counts[k] == target_counts[k]
                for k in ("causal_lm_entries", "nested_decoder_entries")
            ),
            "target_after_release",
        )
        require(not torch.cuda.is_initialized(), "unexpected_cuda")
        require(
            semantic_asset_check(freeze["semantic_assets"]) == semantic_assets,
            "encoder_assets_changed",
        )
        write("semantic-assets-after.json", semantic_assets)
        encoder_reference = weakref.ref(model)
        del model, tokenizer
        gc.collect()
        require(encoder_reference() is None, "encoder_reference_retained")
        phase = "fit_request"
        request = assemble_fit_request(freeze, views, acquired, encoded, tasks)
        write("fit-request.json", request)
        phase = "fit_subprocess"
        command = [
            freeze["fit_python"],
            str(source / freeze["fit_worker"]),
            "--request",
            str(root / "fit-request.json"),
            "--request-sha256",
            object_hash(request),
            "--output",
            str(root / "fit"),
            "--parent",
            str(os.getpid()),
        ]
        with (root / "fit-process.log").open("xb") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=False
            )
            fit_started, fit_entries = True, None
            write(
                "fit-process.json",
                {
                    "pid": process.pid,
                    "parent": os.getpid(),
                    "command": command,
                    "sid": os.getsid(process.pid),
                },
            )
            require(os.getsid(process.pid) == os.getsid(0), "fit_owned_session")
            code = process.wait()
        write("fit-process-exit.json", {"exit_code": code})
        require(code == 0, "fit_exit")
        terminal = validate_fit_return(root / "fit", object_hash(request), freeze, process.pid, e)
        fit_entries = terminal["actual_fit_entries"]
        verify_sources(freeze)
        write(
            "worker-terminal.json",
            {
                "schema_version": SCHEMA,
                "status": "completed",
                "freeze_sha256": object_hash(freeze),
                "acquisition_terminal_sha256": acquired["terminal_sha256"],
                "encoder_terminal_sha256": encoded["terminal_sha256"],
                "fit_terminal_sha256": object_hash(terminal),
                "body_entry_counts": counts,
                "actual_fit_entries": terminal["actual_fit_entries"],
                "nested_counts_nonadditive": True,
            },
        )
        return {
            "status": "completed",
            "body_entry_counts": counts,
            "actual_fit_entries": terminal["actual_fit_entries"],
        }
    except BaseException as error:
        original_error = error
        try:
            write(
                "worker-failure.json",
                {
                    "schema_version": SCHEMA,
                    "status": "failed",
                    "freeze_sha256": object_hash(freeze),
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "body_entry_counts": counts,
                    "fit_subprocess_started": fit_started,
                    "actual_fit_entries": fit_entries,
                    "complete": False,
                },
            )
        except BaseException:
            pass
        raise
    finally:
        try:
            sys.setprofile(previous_profile)
        except BaseException as cleanup_error:
            try:
                write(
                    "worker-cleanup-failure.json",
                    {
                        "schema_version": SCHEMA,
                        "status": "failed",
                        "cleanup_error_type": type(cleanup_error).__name__,
                        "original_error_type": None
                        if original_error is None
                        else type(original_error).__name__,
                        "phase": phase,
                        "body_entry_counts": counts,
                    },
                )
            except BaseException:
                pass
            if original_error is None:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", required=True, type=int)
    args = parser.parse_args()
    helper = pinned_helper()
    with helper.owned_interruptions():
        helper.bind_parent_lifetime(args.parent)
        freeze = helper.hash_bound_json(args.freeze, args.freeze_sha256)
        result = run(freeze, args.output)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
