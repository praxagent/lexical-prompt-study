#!/usr/bin/env python3
"""One-shot CPU FP32 reference for the already-observed A164 development set.

No held-out execution, downloads, GPU use, quantization, retries or prompt edits.
Launch under an external 7,200-second timeout. All responses remain private.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

MIN_AVAILABLE_BYTES = 48 * 1024**3
CELL_SCHEMA = "cpu-reference-a164-cell-v1"


class ReferenceInterrupted(BaseException):
    """Bypass the frozen evaluator's per-trial Exception recovery."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_json(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("input_hash_drift")
    return json.loads(raw)


def memory_available(path=Path("/proc/meminfo")):
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("memory_availability_unavailable")


def require_memory():
    available = memory_available()
    if available < MIN_AVAILABLE_BYTES:
        raise RuntimeError("insufficient_cpu_reference_memory")
    return available


def load_frozen(source):
    sys.path.insert(0, str(source))
    from lexical_prompt_study import instruction_selection_runtime as runtime
    from lexical_prompt_study import instruction_selection_tasks as tasks
    for module in (runtime, tasks, runtime.engine):
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError("frozen_source_import_mismatch")
    return runtime, tasks


def prepare_inputs(args, runtime, tasks):
    plan = bound_json(args.plan, args.plan_sha256)
    config = bound_json(args.config, args.config_sha256)
    protocol = args.protocol.read_bytes()
    if sha(protocol) != args.protocol_sha256:
        raise ValueError("reference_protocol_hash_drift")
    tasks.validate_plan(plan)
    runtime.engine.validate_config(config)
    if plan["partition"] != "development" or plan["stage"] != "development_baseline" or len(plan["trials"]) != 32:
        raise ValueError("observed_development_only")
    if config["cpu_threads"] != 4 or config["attention_implementation"] != "sdpa":
        raise ValueError("reference_requires_shared_sdpa_four_thread_config")
    # Receipt export replays native tokens, EOS and scores, without a model load.
    nf4_records = runtime.export_records(plan_path=args.plan, config_path=args.config,
        output_root=args.nf4_run_root, expected_plan_sha256=args.plan_sha256,
        expected_config_sha256=args.config_sha256)
    if len(nf4_records) != 32 or any(row["generation_status"] != "completed" for row in nf4_records):
        raise ValueError("all_prior_development_cells_must_be_observed")
    tokenizer = runtime.load_tokenizer(config)
    prepared = {trial["trial_id"]: runtime.prepare_generation(tokenizer, trial, config) for trial in plan["trials"]}
    eos_ids = runtime.pinned_eos_ids(config)
    # Hash every original weight before creating the one-shot freeze or loading.
    if runtime.engine.snapshot_manifest(Path(config["model_path"])) != config["model_files_sha256"]:
        raise ValueError("original_weight_manifest_drift")
    available = require_memory()
    prior_header_path = args.nf4_run_root / "run.json"
    prior_header = json.loads(prior_header_path.read_bytes())
    nf4_diagnostics = {}
    numeric_by_id = {row["trial_id"]: row for row in nf4_records}
    with (args.nf4_run_root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        for trial in plan["trials"]:
            history = runtime._history(args.nf4_run_root / "trials" / trial["trial_id"], trial,
                                       prior_header, tokenizer, config, eos_ids)
            result = history[-1][2]
            if result is None or result["record"] != numeric_by_id[trial["trial_id"]]:
                raise ValueError("prior_receipt_changed")
            nf4_diagnostics[trial["trial_id"]] = diagnose(result["private"], tasks)
    header = {"schema_version": "cpu-reference-run-v1", "script_sha256": sha(Path(__file__).read_bytes()),
        "reference_protocol_sha256": args.protocol_sha256,
        "plan_sha256": args.plan_sha256, "shared_config_sha256": args.config_sha256,
        "nf4_run_header_sha256": sha(prior_header_path.read_bytes()),
        "nf4_numeric_records_sha256": sha(canonical(nf4_records)),
        "nf4_diagnostics_sha256": sha(canonical(nf4_diagnostics)),
        "frozen_source_hashes": prior_header["source_hashes"],
        "native_prepared_sha256": sha(canonical(prepared)), "eos_token_ids": eos_ids,
        "reference_settings": {"device": "cpu", "dtype": "float32", "quantization": None,
            "attention_implementation": "sdpa", "do_sample": False, "num_beams": 1,
            "max_new_tokens": 64, "cpu_threads": 4, "seed": config["seed"], "retries": 0},
        "available_memory_bytes_preflight": available, "minimum_available_memory_bytes": MIN_AVAILABLE_BYTES,
        "planned_cells": 32, "heldout_allowed": False, "quantization_causal_claim": False,
        "different_attention_and_arithmetic_backend": True}
    return plan, config, nf4_records, nf4_diagnostics, tokenizer, prepared, header


def load_cpu_reference(config, engine):
    """Original-weight local CPU model; never call the frozen NF4 loader."""
    require_memory()
    if engine.runtime_versions() != config["runtime_versions"]:
        raise ValueError("runtime_version_drift")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    path = Path(config["model_path"])
    original = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if getattr(original, "quantization_config", None) is not None:
        raise ValueError("original_unquantized_weights_required")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, device_map={"": "cpu"}, attn_implementation="sdpa").eval()
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("reference_was_quantized")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("reference_parameter_device_or_dtype")
    if any(type(module).__module__.startswith("bitsandbytes") for module in model.modules()):
        raise ValueError("reference_contains_quantized_module")
    if torch.cuda.is_initialized():
        raise ValueError("unexpected_cuda_initialization")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if engine.digest(tokenizer.get_chat_template().encode()) != config["chat_template_sha256"]:
        raise ValueError("reference_native_template_drift")
    return engine.LocalRuntime(model, tokenizer, torch, config)


def reference_record(trial, runtime):
    record = runtime.numeric_record(trial)
    record["schema_version"] = CELL_SCHEMA
    return record


def validate_reference_result(result, attempt, trial, tokenizer, runtime):
    if result["schema_version"] != "cpu-reference-result-v1":
        raise ValueError("reference_result_schema")
    if result["record"]["schema_version"] != CELL_SCHEMA:
        raise ValueError("reference_record_schema")
    # Reuse the frozen strict token/scorer checks internally, while the durable
    # reference artifact keeps a distinct schema and cannot masquerade as A164.
    wire = {"schema_version": "a164-result-v1", "attempt_sha256": result["attempt_sha256"],
        "record": {**result["record"], "schema_version": "a164-cell-v1"},
        "private": result["private"], "elapsed_seconds": result["elapsed_seconds"]}
    runtime.validate_result(wire, attempt, trial, sha(canonical(attempt)), tokenizer)


def diagnose(private, tasks):
    text = private["response_text"]
    label = False
    if text is not None:
        try:
            value = json.loads(text, object_pairs_hook=tasks._unique_object, parse_constant=tasks._reject_constant)
            label = type(value) is dict and set(value) == {"answer"} and value["answer"] in ("A", "B")
        except (ValueError, RecursionError):
            pass
    tokens = private["generated_token_ids"]
    return {"literal_selector_label": label if text is not None else None,
            "generated_token_count": len(tokens) if tokens is not None else None}


def summarize(plan, nf4_records, reference_records, nf4_diagnostics, reference_diagnostics):
    expected = {trial["trial_id"] for trial in plan["trials"]}
    baseline = {row["trial_id"]: row for row in nf4_records}
    reference = {row["trial_id"]: row for row in reference_records}
    if len(expected) != 32 or set(baseline) != expected or len(baseline) != len(nf4_records):
        raise ValueError("baseline_comparison_identity")
    if len(reference) != len(reference_records) or not set(reference) <= expected:
        raise ValueError("reference_comparison_identity")
    transitions = {old: dict.fromkeys(("exact", "other_selector", "other_answer", "format", "unresolved"), 0)
                   for old in ("exact", "other_selector", "other_answer", "format")}
    completed = improved = worsened = same = strict = cap = 0
    low = high = 0
    for trial in plan["trials"]:
        original = baseline[trial["trial_id"]]
        row = reference.get(trial["trial_id"])
        old = int(original["score"]["strict_correct"])
        resolved = row is not None and row["generation_status"] == "completed"
        category = row["score"]["category"] if resolved else "unresolved"
        transitions[original["score"]["category"]][category] += 1
        low += (int(row["score"]["strict_correct"]) if resolved else 0) - old
        high += (int(row["score"]["strict_correct"]) if resolved else 1) - old
        if resolved:
            correct = int(row["score"]["strict_correct"])
            completed += 1
            strict += correct
            cap += row["finish_reason"] == "length"
            improved += correct > old
            worsened += correct < old
            same += correct == old
    def arm(rows, diagnostics, trials):
        complete = [rows[t["trial_id"]] for t in trials if t["trial_id"] in rows
                    and rows[t["trial_id"]]["generation_status"] == "completed"]
        correct = sum(row["score"]["strict_correct"] for row in complete)
        total = len(trials)
        unknown = total - len(complete)
        lengths = [diagnostics[row["trial_id"]]["generated_token_count"] for row in complete]
        return {"planned": total, "completed": len(complete), "unresolved": unknown,
            "strict_correct": correct, "accuracy_bounds_pp": [100 * correct / total, 100 * (correct + unknown) / total],
            "capped": sum(row["finish_reason"] == "length" for row in complete),
            "generated_tokens": {"sequences": len(lengths), "total": sum(lengths),
                                 "minimum": min(lengths, default=None), "maximum": max(lengths, default=None)},
            "literal_selector_label": sum(diagnostics.get(row["trial_id"], {}).get("literal_selector_label") is True
                                          for row in complete),
            "parser_categories": {category: sum(row["score"]["category"] == category for row in complete)
                                  for category in ("exact", "other_selector", "other_answer", "format")}}
    def pairs(rows):
        groups = {}
        for trial in plan["trials"]:
            row = rows.get(trial["trial_id"])
            groups.setdefault(trial["world_id"], []).append(
                row["score"]["strict_correct"] if row and row["generation_status"] == "completed" else None)
        lower = sum(all(value is True for value in group) for group in groups.values())
        upper = sum(not any(value is False for value in group) for group in groups.values())
        return {"worlds": len(groups), "completed_pairs": sum(all(value is not None for value in group)
                for group in groups.values()), "both_correct": lower,
                "both_correct_accuracy_bounds_pp": [100 * lower / len(groups), 100 * upper / len(groups)]}
    return {"schema_version": "cpu-reference-comparison-v1", "planned": 32, "reference_completed": completed,
        "reference_infrastructure_failed": sum(row["generation_status"] != "completed" for row in reference_records),
        "reference_missing": 32 - len(reference_records), "nf4_strict_correct": sum(
            row["score"]["strict_correct"] for row in nf4_records), "reference_strict_correct": strict,
        "reference_caps": cap, "paired_improved": improved, "paired_worsened": worsened,
        "paired_unchanged": same, "parser_category_transitions": transitions,
        "paired_accuracy_difference_pp": 100 * low / 32 if completed == 32 else None,
        "paired_accuracy_difference_bounds_pp": [100 * low / 32, 100 * high / 32],
        "reference_accuracy_bounds_pp": [100 * strict / 32, 100 * (strict + 32 - completed) / 32],
        "arms": {"nf4": arm(baseline, nf4_diagnostics, plan["trials"]),
                 "cpu_fp32": arm(reference, reference_diagnostics, plan["trials"])},
        "both_selector_pairs": {"nf4": pairs(baseline), "cpu_fp32": pairs(reference)},
        "by_presentation_and_selector": {order + "_" + selector: {
            "nf4": arm(baseline, nf4_diagnostics, [t for t in plan["trials"] if t["query_order"] == order and t["selector"] == selector]),
            "cpu_fp32": arm(reference, reference_diagnostics, [t for t in plan["trials"] if t["query_order"] == order and t["selector"] == selector])}
            for order in ("AB", "BA") for selector in ("A", "B")},
        "quantization_causal_claim": False, "different_precision_device_and_sdpa_backend": True,
        "posthoc_runtime_diagnostic": True, "a164_gate_reopened": False, "heldout_allowed": False}


def interrupt(_signum, _frame):
    raise ReferenceInterrupted()


def run(args, *, source_loader=load_frozen, model_loader=load_cpu_reference):
    runtime, tasks = source_loader(args.frozen_src)
    root = args.output_root.resolve()
    if root.is_relative_to(args.frozen_src.resolve().parent) or root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("private_reference_output_outside_source_required")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    # Claim ownership before expensive validation. Re-running this root is forbidden.
    with (root / "one-shot.json").open("xb") as stream:
        stream.write(canonical({"schema_version": "cpu-reference-claim-v1", "pid": os.getpid(),
                                "script_sha256": sha(Path(__file__).read_bytes())}))
    plan, config, nf4, nf4_diagnostics, tokenizer, prepared, header = prepare_inputs(args, runtime, tasks)
    runtime.engine.write_private(root / "run.json", header)
    runtime.engine.write_private(root / "nf4-records.json", nf4)
    runtime.engine.write_private(root / "nf4-diagnostics.json", nf4_diagnostics)
    records, reference, token_counts, durations = [], None, [], []
    reference_diagnostics = {}
    status = "failed"
    started = time.monotonic()
    try:
        reference = model_loader(config, runtime.engine)
        if reference.eos_ids != header["eos_token_ids"]:
            raise ValueError("reference_eos_drift")
        for trial in plan["trials"]:
            if time.monotonic() - started >= 7200:
                raise TimeoutError("reference_deadline")
            trial_id = trial["trial_id"]
            attempt = {"schema_version": "cpu-reference-attempt-v1", "run_sha256": sha(canonical(header)),
                "trial_sha256": tasks.sha(tasks.canonical(trial)), "prepared": prepared[trial_id],
                "eos_token_ids": reference.eos_ids}
            path = root / "trials" / trial_id
            runtime.engine.write_private(path / "attempt.json", attempt)
            tick = time.monotonic()
            record, private = runtime.evaluate_trial(reference, trial, prepared[trial_id])
            record["schema_version"] = CELL_SCHEMA
            elapsed = time.monotonic() - tick
            result = {"schema_version": "cpu-reference-result-v1", "attempt_sha256": sha(canonical(attempt)),
                      "record": record, "private": private, "elapsed_seconds": elapsed}
            validate_reference_result(result, attempt, trial, tokenizer, runtime)
            runtime.engine.write_private(path / "result.json", result)
            records.append(record)
            reference_diagnostics[trial_id] = diagnose(private, tasks)
            durations.append(elapsed)
            if record["generation_status"] == "completed":
                token_counts.append(len(private["generated_token_ids"]))
        status = "complete" if all(row["generation_status"] == "completed" for row in records) else "incomplete"
    except BaseException as exc:
        runtime.engine.write_private(root / "error.json", runtime.engine.safe_error(exc, "cpu_reference_failed"))
        status = "interrupted" if isinstance(exc, (ReferenceInterrupted, KeyboardInterrupt)) else "failed"
    finally:
        summary = summarize(plan, nf4, records, nf4_diagnostics, reference_diagnostics)
        summary.update(status=status, run_sha256=sha(canonical(header)), numeric_records_sha256=sha(canonical(records)),
            generated_tokens={"completed_sequences": len(token_counts), "total": sum(token_counts),
                              "minimum": min(token_counts, default=None), "maximum": max(token_counts, default=None)},
            generation_seconds=sum(durations), elapsed_seconds=time.monotonic() - started)
        runtime.engine.write_private(root / "records.json", records)
        runtime.engine.write_private(root / "diagnostics.json", reference_diagnostics)
        runtime.engine.write_private(root / "summary.json", summary)
        reference = None
        gc.collect()
    return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen-src", "plan", "config", "nf4-run-root", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "config-sha256", "protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    for key in ("frozen_src", "plan", "config", "nf4_run_root", "protocol", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def main(argv=None):
    os.umask(0o077)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    args = arguments(argv)
    args.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with (args.output_root / "execution.log").open("xb") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = run(args)
        print(json.dumps({key: result[key] for key in ("status", "planned", "reference_completed",
                                                       "reference_strict_correct", "heldout_allowed")}))
        return 0 if result["status"] == "complete" else 1
    except BaseException as exc:
        print(json.dumps({"status": "cpu_reference_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
