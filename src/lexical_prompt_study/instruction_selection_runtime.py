"""Generation-only A164 execution with private immutable receipts and a dev gate.

The frozen A163 module supplies the local model engine, native tokenization and
private JSON writer. A164 worlds, scores, run identity and records are separate.
No likelihood forward passes are performed or reported.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time

from . import instruction_binding_runtime as engine
from . import instruction_selection_tasks as tasks

META = ("trial_id", "world_id", "partition", "task_family", "renderer", "query_order",
        "scaffold", "placement", "selector")


def numeric_record(trial: dict) -> dict:
    return {"schema_version": "a164-cell-v1", **{
        field: trial["scaffold_kind" if field == "scaffold" else field] for field in META
    }, "generation_status": "infrastructure_failed", "finish_reason": None,
        "score": {"category": None, "strict_correct": None}, "likelihood_collected": False}


def load_tokenizer(config: dict):
    engine.validate_config(config)
    tasks.require(engine.runtime_versions() == config["runtime_versions"], "tokenizer_runtime_version_drift")
    from transformers import AutoTokenizer
    root = Path(config["model_path"])
    for name, expected in config["model_files_sha256"].items():
        if not name.endswith(".safetensors"):
            tasks.require(engine.file_digest(root / name) == expected, "tokenizer_config_file_drift")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tasks.require(engine.digest(tokenizer.get_chat_template().encode()) == config["chat_template_sha256"],
                  "native_template_drift")
    return tokenizer


def prepare_generation(tokenizer, trial: dict, config: dict) -> dict:
    # The shared helper verifies native roles/tokens. Its hypothetical suffix
    # metadata is discarded; no A164 likelihood measurements are collected.
    prepared = engine.prepare_trial(tokenizer, trial, config)
    return {key: prepared[key] for key in ("rendered_text", "prompt_token_ids", "chat_template_sha256")}


def pinned_eos_ids(config: dict) -> list[int]:
    path = Path(config["model_path"]) / "generation_config.json"
    tasks.require(engine.file_digest(path) == config["model_files_sha256"].get(path.name),
                  "generation_config_binding")
    value = json.loads(path.read_bytes())["eos_token_id"]
    values = value if type(value) is list else [value]
    tasks.require(bool(values) and all(type(token) is int and token >= 0 for token in values), "pinned_eos_ids")
    return sorted(set(values))


def _header(plan_sha: str, config_sha: str, development_gate: dict | None) -> dict:
    root = Path(__file__).parent
    dependencies = (Path(__file__), Path(tasks.__file__), Path(engine.__file__),
                    root / "instruction_binding_tasks.py", root / "instruction_selection_analysis.py")
    return {"schema_version": "a164-run-v1", "plan_sha256": plan_sha, "config_sha256": config_sha,
            "source_hashes": {path.name: engine.file_digest(path) for path in dependencies},
            "development_gate": development_gate, "likelihood_collected": False}


def _read_bound(path: Path, expected: str):
    raw = path.read_bytes()
    tasks.require(tasks.is_hash(expected) and engine.digest(raw) == expected, "input_hash_drift")
    return json.loads(raw)


def evaluate_trial(runtime, trial: dict, prepared: dict) -> tuple[dict, dict]:
    record = numeric_record(trial)
    private = {"generated_token_ids": None, "response_text": None, "error": None}
    try:
        tokens, response, finish = runtime.generate(prepared)
        category = tasks.score_response(response, trial["selected_answer"], trial["unselected_answer"])
        record.update(generation_status="completed", finish_reason=finish,
                      score={"category": category, "strict_correct": category == "exact" and finish == "eos"})
        private.update(generated_token_ids=tokens, response_text=response)
    except Exception as exc:
        private["error"] = engine.safe_error(exc, "generation_failed")
    return record, private


def validate_result(result: dict, attempt: dict, trial: dict, attempt_sha: str, tokenizer) -> dict:
    tasks.require(set(result) == {"schema_version", "attempt_sha256", "record", "private", "elapsed_seconds"}
                  and result["schema_version"] == "a164-result-v1"
                  and result["attempt_sha256"] == attempt_sha, "result_binding")
    record, private = result["record"], result["private"]
    expected = numeric_record(trial)
    tasks.require(set(record) == set(expected) and all(record[key] == value for key, value in expected.items()
                  if key not in ("generation_status", "finish_reason", "score")), "numeric_metadata_drift")
    if record["generation_status"] == "completed":
        tokens = engine._ids(private["generated_token_ids"])
        eos_ids = attempt["eos_token_ids"]
        eos = tokens[-1] in eos_ids
        tasks.require(len(tokens) <= 64 and not any(value in eos_ids for value in tokens[:-1])
                      and (eos or len(tokens) == 64), "generation_termination")
        text = tokenizer.decode(tokens[:-1] if eos else tokens, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        tasks.require(text == private["response_text"], "response_token_decode_drift")
        category = tasks.score_response(text, trial["selected_answer"], trial["unselected_answer"])
        finish = "eos" if eos else "length"
        tasks.require(record["finish_reason"] == finish and record["score"] == {
            "category": category, "strict_correct": category == "exact" and finish == "eos"
        } and private["error"] is None, "response_score_drift")
    else:
        tasks.require(record == expected and private["generated_token_ids"] is None
                      and private["response_text"] is None and type(private["error"]) is dict,
                      "infrastructure_failure_record")
    return record


def _history(root: Path, trial: dict, header: dict, tokenizer, config: dict, eos_ids: list[int]) -> list[tuple]:
    result = []
    previous_hash = None
    completed = False
    for index, path in enumerate(sorted(root.glob("attempt-*/attempt.json")), start=1):
        tasks.require(path.parent.name == f"attempt-{index:02d}" and index <= 3 and not completed,
                      "invalid_attempt_sequence")
        attempt = json.loads(path.read_bytes())
        tasks.require(attempt["schema_version"] == "a164-attempt-v1" and attempt["run"] == header
                      and attempt["trial_sha256"] == tasks.sha(tasks.canonical(trial))
                      and attempt["previous_result_sha256"] == previous_hash, "attempt_lineage")
        tasks.require(attempt["prepared"] == prepare_generation(tokenizer, trial, config)
                      and attempt["eos_token_ids"] == eos_ids, "native_prompt_or_eos_replay_drift")
        value = None
        result_path = path.with_name("result.json")
        if result_path.exists():
            value = json.loads(result_path.read_bytes())
            validate_result(value, attempt, trial, engine.file_digest(path), tokenizer)
            completed = value["record"]["generation_status"] == "completed"
            previous_hash = engine.file_digest(result_path)
        result.append((path, attempt, value))
    return result


def export_records(*, plan_path: Path, config_path: Path, output_root: Path,
                   expected_plan_sha256: str, expected_config_sha256: str,
                   tokenizer_loader=load_tokenizer) -> list[dict]:
    plan = _read_bound(plan_path, expected_plan_sha256)
    config = _read_bound(config_path, expected_config_sha256)
    tasks.validate_plan(plan)
    engine.validate_config(config)
    tokenizer = tokenizer_loader(config)
    eos_ids = pinned_eos_ids(config)
    with (output_root / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        header = json.loads((output_root / "run.json").read_bytes())
        tasks.require(header == _header(expected_plan_sha256, expected_config_sha256,
                                        header["development_gate"]), "source_or_input_lineage")
        tasks.require((plan["partition"] == "development") == (header["development_gate"] is None),
                      "development_gate_identity")
        records = []
        for trial in plan["trials"]:
            history = _history(output_root / "trials" / trial["trial_id"], trial, header, tokenizer, config, eos_ids)
            if history and history[-1][2] is not None:
                records.append(history[-1][2]["record"])
    return records


def _development_gate(plan: dict, config_path: Path, config_sha: str,
                      development_plan_path: Path | None, development_plan_sha256: str | None,
                      development_run_root: Path | None, tokenizer_loader) -> dict | None:
    if plan["partition"] == "development":
        tasks.require(all(value is None for value in (
            development_plan_path, development_plan_sha256, development_run_root)), "unexpected_gate_inputs")
        return None
    tasks.require(all(value is not None for value in (
        development_plan_path, development_plan_sha256, development_run_root)), "development_gate_required")
    development = _read_bound(development_plan_path, development_plan_sha256)
    tasks.validate_plan(development)
    tasks.require(development["partition"] == "development" and all(development[key] == plan[key] for key in (
        "cohort_sha256", "bindings", "material_receipts"
    )), "development_instrument_mismatch")
    records = export_records(plan_path=development_plan_path, config_path=config_path,
                             output_root=development_run_root,
                             expected_plan_sha256=development_plan_sha256,
                             expected_config_sha256=config_sha, tokenizer_loader=tokenizer_loader)
    from .instruction_selection_analysis import analyze_plan_records
    summary = analyze_plan_records(development, records)
    tasks.require(summary["baseline_controls"]["gate_passed"], "development_competence_gate_failed")
    return {"development_plan_sha256": development_plan_sha256,
            "development_run_header_sha256": engine.file_digest(development_run_root / "run.json"),
            "records_sha256": tasks.sha(tasks.canonical(records)),
            "summary_sha256": tasks.sha(tasks.canonical(summary)), "gate_passed": True}


def run_plan(*, plan_path: Path, config_path: Path, output_root: Path,
             expected_plan_sha256: str, expected_config_sha256: str,
             development_plan_path: Path | None = None, development_plan_sha256: str | None = None,
             development_run_root: Path | None = None, max_trials: int | None = None,
             retry_failed: bool = False, runtime_loader=engine.load_runtime,
             tokenizer_loader=load_tokenizer) -> dict:
    plan = _read_bound(plan_path, expected_plan_sha256)
    config = _read_bound(config_path, expected_config_sha256)
    tasks.validate_plan(plan)
    engine.validate_config(config)
    tasks.require(max_trials is None or type(max_trials) is int and max_trials > 0, "max_trials")
    # Gate replay precedes GPU/model loading and all held-out execution.
    gate = _development_gate(plan, config_path, expected_config_sha256, development_plan_path,
                             development_plan_sha256, development_run_root, tokenizer_loader)
    header = _header(expected_plan_sha256, expected_config_sha256, gate)
    root = output_root.resolve()
    tasks.require(not root.is_relative_to(Path(__file__).resolve().parents[2]), "private_output_outside_git")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    tokenizer, runtime = tokenizer_loader(config), None
    eos_ids = pinned_eos_ids(config)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine.write_private(root / "run.json", header)
        records, launched = [], 0
        for trial in plan["trials"]:
            trial_root = root / "trials" / trial["trial_id"]
            history = _history(trial_root, trial, header, tokenizer, config, eos_ids)
            last = history[-1][2] if history else None
            if last and (last["record"]["generation_status"] == "completed" or not retry_failed):
                records.append(last["record"])
                continue
            if history and last is None and not retry_failed:
                raise RuntimeError("interrupted_attempt_requires_explicit_retry")
            if max_trials is not None and launched >= max_trials:
                if last:
                    records.append(last["record"])
                continue
            tasks.require(len(history) < 3, "infrastructure_retry_limit")
            prepared = prepare_generation(tokenizer, trial, config)
            if runtime is None:
                runtime = runtime_loader(config)
            # Ensure generation uses the exact tokenizer used by the preflight.
            tasks.require(engine.digest(runtime.tokenizer.get_chat_template().encode())
                          == config["chat_template_sha256"] and runtime.eos_ids == eos_ids,
                          "engine_tokenizer_template_or_eos_drift")
            previous = next((path.with_name("result.json") for path, _, value in reversed(history)
                             if value is not None), None)
            attempt = {"schema_version": "a164-attempt-v1", "run": header,
                       "trial_sha256": tasks.sha(tasks.canonical(trial)), "prepared": prepared,
                       "eos_token_ids": runtime.eos_ids,
                       "previous_result_sha256": engine.file_digest(previous) if previous else None}
            attempt_path = trial_root / f"attempt-{len(history) + 1:02d}" / "attempt.json"
            engine.write_private(attempt_path, attempt)
            started = time.monotonic()
            record, private = evaluate_trial(runtime, trial, prepared)
            result = {"schema_version": "a164-result-v1", "attempt_sha256": engine.file_digest(attempt_path),
                      "record": record, "private": private, "elapsed_seconds": time.monotonic() - started}
            validate_result(result, attempt, trial, engine.file_digest(attempt_path), tokenizer)
            engine.write_private(attempt_path.with_name("result.json"), result)
            records.append(record)
            launched += 1
        engine.write_private(root / f"records-{tasks.sha(tasks.canonical(records))}.json",
                             {"schema_version": "a164-numeric-records-v1", "run": header, "records": records})
        return {"schema_version": "a164-progress-v1", "partition": plan["partition"],
                "planned": len(plan["trials"]), "recorded": len(records), "launched_this_call": launched,
                "generation_completed": sum(row["generation_status"] == "completed" for row in records),
                "strict_correct": sum(row["score"]["strict_correct"] is True for row in records),
                "length_terminated": sum(row["finish_reason"] == "length" for row in records),
                "likelihood_collected": False, "all_cells_recorded": len(records) == len(plan["trials"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "config", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "config-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--development-plan", type=Path)
    parser.add_argument("--development-plan-sha256")
    parser.add_argument("--development-run-root", type=Path)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            result = run_plan(plan_path=args.plan, config_path=args.config, output_root=args.output_root,
                              expected_plan_sha256=args.plan_sha256, expected_config_sha256=args.config_sha256,
                              development_plan_path=args.development_plan,
                              development_plan_sha256=args.development_plan_sha256,
                              development_run_root=args.development_run_root,
                              max_trials=args.max_trials, retry_failed=args.retry_failed)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": engine.safe_error(exc, "a164_execution_failed")}))
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
