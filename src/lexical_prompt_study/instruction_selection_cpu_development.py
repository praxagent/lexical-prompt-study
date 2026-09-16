"""A165 exploratory CPU FP32 scaffold matrix on the 16 observed development worlds.

All 288 cells are fixed before calls. Contemporaneous no-scaffold competence
controls gate the remaining 256 cells. No held-out execution or automatic retries.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

from . import instruction_selection_runtime as native
from . import instruction_selection_tasks as tasks

REFERENCE_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
REFERENCE_PROTOCOL_SHA = "1e2a3074f282acfa4e15331b42b19e6c4ca05e786e8826a93f6dec412aeb8467"
REFERENCE_AMENDMENT_SHA = "503c9382a5f0e2105af9e9afa44b6bc034f71637514d1c83907e69a28f8cd527"
REUSED_SOURCE_HASHES = {
    "instruction_binding_runtime.py": "007d7dc3aea547e42564fd35295149cdaef7ea360dd43e573904f20c94bdebf2",
    "instruction_binding_tasks.py": "6dd403bb5e813d015c8bbe878cac59b908beff330a105855d444362c5bde855a",
    "instruction_selection_tasks.py": "cf36f10a6c1ee98ef103bf5635803fc87d7b401c2faa0a46444bcf3e0e44c85a",
    "instruction_selection_runtime.py": "1a59591fa974f73c71de81119d808218f47c9c9c30de98be1f452ee5e0351d3a",
}


class A165Interrupted(BaseException):
    """Stop outside the frozen evaluator's per-trial Exception recovery."""


def source_bindings(protocol_sha256, base_plan):
    tasks.require(tasks.is_hash(protocol_sha256), "a165_protocol_hash")
    return {"protocol_sha256": protocol_sha256,
        "materials_sha256": base_plan["bindings"]["materials_sha256"],
        "task_source_sha256": base_plan["bindings"]["task_source_sha256"],
        "a165_source_sha256": tasks.sha(Path(__file__).read_bytes()),
        "cpu_reference_script_sha256": REFERENCE_SHA,
        "cpu_reference_protocol_sha256": REFERENCE_PROTOCOL_SHA,
        "cpu_reference_amendment_sha256": REFERENCE_AMENDMENT_SHA}


def _identity(trial, base_sha, bindings):
    return tasks.sha(tasks.canonical({"schema_version": "a165-trial-identity-v1",
        "base_plan_sha256": base_sha, "bindings": bindings,
        "trial": {key: value for key, value in trial.items() if key != "trial_id"}}))[:24]


def compile_plan(base_plan, *, scaffolds, protocol_sha256):
    """Caller supplies already-verified private materials; no outcomes are read."""
    tasks.validate_plan(base_plan)
    tasks.require(base_plan["partition"] == "development" and base_plan["trial_count"] == 32,
                  "a165_requires_original_development_plan")
    tasks.require(set(scaffolds) == set(tasks.SCAFFOLDS), "a165_four_materials_required")
    for kind, value in scaffolds.items():
        receipt = base_plan["material_receipts"][kind]
        tasks.require(type(value) is str and tasks.sha(value.encode()) == receipt["sha256"]
                      and len(value.encode()) == receipt["bytes"], "a165_unchanged_material_required")
    bindings = source_bindings(protocol_sha256, base_plan)
    base_sha = tasks.sha(tasks.canonical(base_plan))
    baselines, scaffold_trials = [], []
    for original in base_plan["trials"]:
        baseline = copy.deepcopy(original)
        baseline["trial_id"] = _identity(baseline, base_sha, bindings)
        baselines.append(baseline)
        user = original["messages"][1]["content"]
        for kind in tasks.SCAFFOLDS:
            for placement in ("before", "after"):
                trial = copy.deepcopy(original)
                trial.update(scaffold_kind=kind, placement=placement)
                trial["messages"][1]["content"] = (scaffolds[kind] + "\n\n" + user if placement == "before"
                                                    else user + "\n\n" + scaffolds[kind])
                trial["trial_id"] = _identity(trial, base_sha, bindings)
                scaffold_trials.append(trial)
    plan = {"schema_version": "a165-plan-v1", "stage": "development_scaffold", "partition": "development",
        "task_family": tasks.FAMILY, "renderer": "json", "world_count": 16, "trial_count": 288,
        "cohort_sha256": base_plan["cohort_sha256"], "base_plan_sha256": base_sha,
        "base_plan": copy.deepcopy(base_plan), "bindings": bindings,
        "material_receipts": copy.deepcopy(base_plan["material_receipts"]),
        "trials": baselines + scaffold_trials, "likelihood_collected": False}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    keys = {"schema_version", "stage", "partition", "task_family", "renderer", "world_count", "trial_count",
            "cohort_sha256", "base_plan_sha256", "base_plan", "bindings", "material_receipts", "trials",
            "likelihood_collected"}
    tasks.require(type(plan) is dict and set(plan) == keys, "a165_plan_schema")
    tasks.require(plan["schema_version"] == "a165-plan-v1" and plan["stage"] == "development_scaffold"
                  and plan["partition"] == "development" and plan["task_family"] == tasks.FAMILY
                  and plan["renderer"] == "json" and plan["world_count"] == 16 and plan["trial_count"] == 288
                  and plan["likelihood_collected"] is False, "a165_plan_identity")
    base = plan["base_plan"]
    tasks.validate_plan(base)
    tasks.require(base["partition"] == "development" and base["trial_count"] == 32
                  and plan["base_plan_sha256"] == tasks.sha(tasks.canonical(base))
                  and plan["cohort_sha256"] == base["cohort_sha256"]
                  and plan["material_receipts"] == base["material_receipts"], "a165_original_development_binding")
    tasks.require(plan["bindings"] == source_bindings(plan["bindings"]["protocol_sha256"], base), "a165_source_binding")
    originals = {(row["world_id"], row["selector"]): row for row in base["trials"]}
    required = {(world, selector, kind, placement) for world, selector in originals
                for kind, placement in [("none", "none"), *((k, p) for k in tasks.SCAFFOLDS for p in ("before", "after"))]}
    actual, identifiers = set(), set()
    tasks.require(type(plan["trials"]) is list and len(plan["trials"]) == 288, "a165_fixed_matrix")
    for index, trial in enumerate(plan["trials"]):
        key = (trial["world_id"], trial["selector"], trial["scaffold_kind"], trial["placement"])
        tasks.require(key in required and key not in actual, "a165_trial_matrix")
        actual.add(key)
        original = originals[key[:2]]
        tasks.require(set(trial) == set(original) and all(trial[k] == v for k, v in original.items()
            if k not in ("trial_id", "scaffold_kind", "placement", "messages")), "a165_original_task_binding")
        tasks.require(trial["messages"][0] == original["messages"][0]
                      and len(trial["messages"]) == 2 and set(trial["messages"][1]) == {"role", "content"}
                      and trial["messages"][1]["role"] == "user", "a165_native_message_roles")
        base_user = original["messages"][1]["content"]
        content = trial["messages"][1]["content"]
        if index < 32:
            tasks.require(trial["scaffold_kind"] == "none" and content == base_user
                          and key[:2] == (base["trials"][index]["world_id"], base["trials"][index]["selector"]),
                          "a165_all_baselines_first")
        else:
            tasks.require(trial["scaffold_kind"] in tasks.SCAFFOLDS, "a165_scaffold_phase")
            offset = index - 32
            ordered_original = base["trials"][offset // 8]
            tasks.require(key == (ordered_original["world_id"], ordered_original["selector"],
                          tasks.SCAFFOLDS[(offset % 8) // 2], ("before", "after")[offset % 2]),
                          "a165_fixed_scaffold_order")
            before = trial["placement"] == "before"
            edge = "\n\n" + base_user if before else base_user + "\n\n"
            tasks.require(content.endswith(edge) if before else content.startswith(edge), "a165_placement")
            material = content[:-len(edge)] if before else content[len(edge):]
            receipt = plan["material_receipts"][trial["scaffold_kind"]]
            tasks.require(tasks.sha(material.encode()) == receipt["sha256"] and len(material.encode()) == receipt["bytes"],
                          "a165_material_binding")
        expected = _identity(trial, plan["base_plan_sha256"], plan["bindings"])
        tasks.require(trial["trial_id"] == expected and expected not in identifiers, "a165_trial_identity")
        identifiers.add(expected)
    tasks.require(actual == required, "a165_complete_planned_matrix")


def numeric_record(trial):
    return {**native.numeric_record(trial), "schema_version": "a165-cell-v1"}


def baseline_gate(plan, records):
    trials = plan["trials"][:32]
    by_id = {row["trial_id"]: row for row in records}
    tasks.require(len(by_id) == len(records), "a165_duplicate_terminal")
    complete, correct, pairs = 0, 0, {}
    for trial in trials:
        row = by_id.get(trial["trial_id"])
        ok = row is not None and row["generation_status"] == "completed"
        complete += ok
        correct += ok and row["score"]["strict_correct"]
        pairs.setdefault(trial["world_id"], []).append(ok and row["score"]["strict_correct"])
    paired = sum(all(group) for group in pairs.values())
    return {"planned": 32, "completed": complete, "strict_correct": correct,
            "worlds": 16, "both_selector_correct": paired,
            "gate_passed": complete == 32 and correct >= 29 and paired >= 15}


def load_reference(path):
    tasks.require(native.engine.file_digest(path) == REFERENCE_SHA, "a165_reference_source_drift")
    spec = importlib.util.spec_from_file_location("a165_frozen_cpu_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sources():
    root = Path(__file__).parent
    for name, expected in REUSED_SOURCE_HASHES.items():
        tasks.require(native.engine.file_digest(root / name) == expected, "a165_reused_source_drift")
    paths = [Path(__file__), root / "instruction_selection_cpu_development_analysis.py"]
    return {**REUSED_SOURCE_HASHES, **{path.name: native.engine.file_digest(path) for path in paths}}


def _interrupt(_signal, _frame):
    raise A165Interrupted()


def export_records(*, plan_path, plan_sha256, config_path, config_sha256, output_root,
                   tokenizer_loader=native.load_tokenizer):
    """Receipt-backed numeric export; native prompt/response/scorer replay, no model."""
    plan = native._read_bound(plan_path, plan_sha256)
    config = native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    sources = validate_sources()
    tokenizer = tokenizer_loader(config)
    eos = native.pinned_eos_ids(config)
    prepared = {row["trial_id"]: native.prepare_generation(tokenizer, row, config) for row in plan["trials"]}
    root = output_root
    with (root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        header = json.loads((root / "run.json").read_bytes())
        tasks.require(header["schema_version"] == "a165-run-v1" and header["plan_sha256"] == plan_sha256
                      and header["config_sha256"] == config_sha256 and header["source_hashes"] == sources
                      and header["cpu_reference_script_sha256"] == REFERENCE_SHA
                      and header["reference_protocol_sha256"] == REFERENCE_PROTOCOL_SHA
                      and header["reference_amendment_sha256"] == REFERENCE_AMENDMENT_SHA
                      and header["native_prepared_sha256"] == tasks.sha(tasks.canonical(prepared))
                      and header["eos_token_ids"] == eos and header["heldout_allowed"] is False
                      and header["exploratory_development_only"] is True and header["retries"] == 0,
                      "a165_run_lineage")
        tasks.require(native.engine.file_digest(Path(header["cpu_reference_script_path"])) == REFERENCE_SHA,
                      "a165_reference_source_drift")
        records = []
        expected = {trial["trial_id"] for trial in plan["trials"]}
        tasks.require(all(path.name in expected for path in (root / "trials").glob("*")), "a165_unknown_trial")
        for trial in plan["trials"]:
            folder = root / "trials" / trial["trial_id"]
            if not (folder / "attempt.json").exists():
                tasks.require(not (folder / "result.json").exists(), "a165_orphan_result")
                continue
            attempt = json.loads((folder / "attempt.json").read_bytes())
            tasks.require(attempt == {"schema_version": "a165-attempt-v1", "run_sha256": tasks.sha(tasks.canonical(header)),
                "trial_sha256": tasks.sha(tasks.canonical(trial)), "prepared": prepared[trial["trial_id"]],
                "eos_token_ids": eos}, "a165_attempt_replay")
            if not (folder / "result.json").exists():
                continue
            result = json.loads((folder / "result.json").read_bytes())
            tasks.require(result["schema_version"] == "a165-result-v1"
                          and result["record"]["schema_version"] == "a165-cell-v1", "a165_result_identity")
            wire = {**result, "schema_version": "a164-result-v1",
                    "record": {**result["record"], "schema_version": "a164-cell-v1"}}
            native.validate_result(wire, attempt, trial, native.engine.file_digest(folder / "attempt.json"), tokenizer)
            records.append(result["record"])
        if any(row["scaffold"] != "none" for row in records):
            gate = baseline_gate(plan, records)
            tasks.require(gate["gate_passed"] and json.loads((root / "baseline-gate.json").read_bytes()) == gate,
                          "a165_concrete_baseline_gate")
        if (root / "records.json").exists():
            tasks.require(json.loads((root / "records.json").read_bytes()) == records, "a165_numeric_export_drift")
    return records


def run_plan(*, plan_path, plan_sha256, config_path, config_sha256, protocol_path, reference_script,
             output_root, reference_loader=load_reference, tokenizer_loader=native.load_tokenizer):
    plan = native._read_bound(plan_path, plan_sha256)
    config = native._read_bound(config_path, config_sha256)
    validate_plan(plan)
    native.engine.validate_config(config)
    tasks.require(native.engine.file_digest(protocol_path) == plan["bindings"]["protocol_sha256"], "a165_protocol_drift")
    tasks.require(config["attention_implementation"] == "sdpa" and config["cpu_threads"] == 4, "a165_shared_settings")
    sources = validate_sources()
    reference = reference_loader(reference_script)
    tokenizer = tokenizer_loader(config)
    prepared = {row["trial_id"]: native.prepare_generation(tokenizer, row, config) for row in plan["trials"]}
    eos_ids = native.pinned_eos_ids(config)
    tasks.require(native.engine.snapshot_manifest(Path(config["model_path"])) == config["model_files_sha256"],
                  "a165_original_model_manifest")
    available = reference.require_memory()
    root = output_root.resolve()
    tasks.require(not root.is_relative_to(Path(__file__).resolve().parents[2]), "a165_private_output_outside_source")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with (root / "one-shot.json").open("xb") as claim:
        claim.write(tasks.canonical({"schema_version": "a165-one-shot-v1", "pid": os.getpid()}))
    header = {"schema_version": "a165-run-v1", "plan_sha256": plan_sha256, "config_sha256": config_sha256,
        "source_hashes": sources, "cpu_reference_script_sha256": REFERENCE_SHA,
        "cpu_reference_script_path": str(reference_script.resolve()),
        "reference_protocol_sha256": REFERENCE_PROTOCOL_SHA, "reference_amendment_sha256": REFERENCE_AMENDMENT_SHA,
        "native_prepared_sha256": tasks.sha(tasks.canonical(prepared)), "eos_token_ids": eos_ids,
        "available_memory_bytes": available, "settings": {"device": "cpu", "dtype": "float32", "quantization": None,
            "attention_implementation": "sdpa", "cpu_threads": 4, "seed": config["seed"], "max_new_tokens": 64},
        "exploratory_development_only": True, "heldout_allowed": False, "retries": 0}
    lock = (root / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    native.engine.write_private(root / "run.json", header)
    native.engine.write_private(root / "native-inputs.json", prepared)
    records, model, status, gate = [], None, "failed", None
    started = time.monotonic()
    try:
        model = reference.load_cpu_reference(config, native.engine)
        tasks.require(model.eos_ids == eos_ids, "a165_reference_eos")
        for index, trial in enumerate(plan["trials"]):
            if index == 32:
                gate = baseline_gate(plan, records)
                native.engine.write_private(root / "baseline-gate.json", gate)
                if not gate["gate_passed"]:
                    status = "stopped_baseline_gate_failed"
                    break
            if time.monotonic() - started >= 18_000:
                raise TimeoutError("a165_deadline")
            attempt = {"schema_version": "a165-attempt-v1", "run_sha256": tasks.sha(tasks.canonical(header)),
                "trial_sha256": tasks.sha(tasks.canonical(trial)), "prepared": prepared[trial["trial_id"]],
                "eos_token_ids": eos_ids}
            folder = root / "trials" / trial["trial_id"]
            native.engine.write_private(folder / "attempt.json", attempt)
            tick = time.monotonic()
            record, private = native.evaluate_trial(model, trial, attempt["prepared"])
            elapsed = time.monotonic() - tick
            wire = {"schema_version": "a164-result-v1", "attempt_sha256": tasks.sha(tasks.canonical(attempt)),
                    "record": record, "private": private, "elapsed_seconds": elapsed}
            native.validate_result(wire, attempt, trial, wire["attempt_sha256"], tokenizer)
            record["schema_version"] = "a165-cell-v1"
            result = {**wire, "schema_version": "a165-result-v1", "record": record}
            native.engine.write_private(folder / "result.json", result)
            records.append(record)
        else:
            status = "complete" if all(row["generation_status"] == "completed" for row in records) else "incomplete"
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (A165Interrupted, KeyboardInterrupt)) else "failed"
        native.engine.write_private(root / "error.json", native.engine.safe_error(exc, "a165_execution_failed"))
    finally:
        gate = gate or baseline_gate(plan, records)
        progress = {"schema_version": "a165-progress-v1", "status": status, "planned": 288,
            "recorded": len(records), "completed": sum(row["generation_status"] == "completed" for row in records),
            "unlaunched": 288 - len(records), "baseline_controls": gate,
            "records_sha256": tasks.sha(tasks.canonical(records)), "run_sha256": tasks.sha(tasks.canonical(header)),
            "elapsed_seconds": time.monotonic() - started, "heldout_allowed": False,
            "exploratory_development_only": True}
        native.engine.write_private(root / "records.json", records)
        native.engine.write_private(root / "progress.json", progress)
        model = None
        gc.collect()
        try:
            from .instruction_selection_cpu_development_analysis import analyze_plan_records
            aggregate = analyze_plan_records(plan, records)
            native.engine.write_private(root / "analysis.json", aggregate)
        except BaseException as exc:
            native.engine.write_private(root / "analysis-error.json", native.engine.safe_error(exc, "a165_analysis_failed"))
        lock.close()
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "config", "protocol", "reference-script", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "config-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _interrupt)
    args.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with (args.output_root / "execution.log").open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = run_plan(plan_path=args.plan, plan_sha256=args.plan_sha256,
                    config_path=args.config, config_sha256=args.config_sha256, protocol_path=args.protocol,
                    reference_script=args.reference_script, output_root=args.output_root)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] in ("complete", "stopped_baseline_gate_failed") else 1
    except BaseException as exc:
        print(json.dumps({"status": "a165_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
