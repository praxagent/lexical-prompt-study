"""Supervise the frozen A186 acquisition, encoder and fitter exactly once.

Only the selected study's new process session is owned. Resource limits are
sampled stop thresholds; they are not kernel allocation reservations. All
receipts and logs stay in the caller's private output directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

NATIVE_PYTHON = "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python"
FIT_PYTHON = "/data2/PRAX/lexical-prompt-study/.venv/bin/python"
HELPER_SHA = "af4038d8422a33e67e507a230e16a89c5ce48002d9b6aa81888cfc838d4e94d0"
LIMITS = {
    "minimum_available_bytes": 64 * 1024**3,
    "maximum_owned_rss_bytes": 64 * 1024**3,
    "host_pressure_floor_bytes": 8 * 1024**3,
    "minimum_free_scratch_bytes": 64 * 1024**3,
    "minimum_remaining_scratch_bytes": 32 * 1024**3,
    "maximum_raw_run_bytes": 8 * 1024**3,
    "wall_seconds": 86400,
    "termination_grace_seconds": 60,
    "poll_seconds": 0.2,
    "storage_poll_seconds": 5.0,
}
SOURCE_PATHS = (
    "scripts/run_matched_prefix_prediction.py",
    "scripts/matched_prefix_native_worker.py",
    "scripts/matched_prefix_fit_worker.py",
    "scripts/run_native_prefix_qualification.py",
    "scripts/run_cpu_reference_audit.py",
    "src/lexical_prompt_study/__init__.py",
    "src/lexical_prompt_study/landmark_evidence.py",
    "src/lexical_prompt_study/prefix_only_capture.py",
    "src/lexical_prompt_study/native_prefix_qualification.py",
    "src/lexical_prompt_study/instruction_binding_runtime.py",
    "src/lexical_prompt_study/instruction_binding_tasks.py",
    "src/lexical_prompt_study/matched_prefix_tasks.py",
    "src/lexical_prompt_study/matched_prefix_acquisition.py",
    "src/lexical_prompt_study/matched_prefix_encoder.py",
    "src/lexical_prompt_study/matched_prefix_prediction.py",
)
FREEZE_KEYS = set(
    "schema_version source_root sources target_config_path target_config_sha256 "
    "runtime_binding plan semantic_assets encoder_binding_sha256 fit_runtime "
    "fit_python fit_worker native_python limits output_root prepared_root "
    "files_sha256 public_commit selection_sha256 qualification_sha256 "
    "encoder_runtime encoder_binding".split()
)


def require(ok, reason):
    if not ok:
        raise ValueError("a186_supervisor_" + reason)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def publish(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    return digest(path)


def helper():
    path = Path(__file__).with_name("run_native_prefix_qualification.py")
    require(digest(path) == HELPER_SHA, "lifecycle_helper_source")
    spec = importlib.util.spec_from_file_location("a186_owned_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def verify_freeze(path, expected_sha, *, enforce_executable=True):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), "freeze_path")
    frozen = helper().hash_bound_json(path, expected_sha)
    require(canonical(frozen) == path.read_bytes(), "freeze_canonical")
    require(set(frozen) == FREEZE_KEYS, "freeze_schema")
    require(frozen["schema_version"] == "a186-native-freeze-v1", "freeze_version")
    require(frozen["native_python"] == NATIVE_PYTHON, "native_python")
    require(frozen["fit_python"] == FIT_PYTHON, "fit_python")
    if enforce_executable:
        require(sys.executable == NATIVE_PYTHON, "actual_executable")
    require(canonical(frozen["limits"]) == canonical(LIMITS), "selected_limits")
    prepared, source = Path(frozen["prepared_root"]), Path(frozen["source_root"])
    require(prepared == path.parent and source == prepared / "source", "prepared_source_path")
    require(not prepared.is_symlink(), "prepared_symlink")
    expected = frozen["files_sha256"]
    require(type(expected) is dict and bool(expected), "inventory_type")
    for relative, sha in expected.items():
        require(
            type(relative) is str
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "relative_path",
        )
        require(
            type(sha) is str and len(sha) == 64 and all(x in "0123456789abcdef" for x in sha),
            "file_hash",
        )
    actual = inventory(prepared)
    require(actual.pop(path.name, None) == expected_sha and actual == expected, "frozen_inventory")
    require(set(frozen["sources"]) == set(SOURCE_PATHS), "source_closure")
    require(
        all(expected.get("source/" + name) == sha for name, sha in frozen["sources"].items()),
        "source_hashes",
    )
    require(
        Path(__file__).resolve() == source / "scripts/run_matched_prefix_prediction.py",
        "actual_frozen_supervisor",
    )
    config = Path(frozen["target_config_path"])
    require(
        config.parent == prepared and expected.get(config.name) == frozen["target_config_sha256"],
        "target_config_binding",
    )
    require(frozen["fit_worker"] == "scripts/matched_prefix_fit_worker.py", "fit_worker")
    require(Path(frozen["output_root"]) == prepared.parent / "run-001", "one_shot_output")
    require(
        hashlib.sha256(canonical(frozen["encoder_binding"])).hexdigest()
        == frozen["encoder_binding_sha256"],
        "encoder_binding",
    )
    return frozen


def raw_bytes(root):
    """Account all regular output bytes, rejecting links and special files."""
    result = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            try:
                item = path.lstat()
            except FileNotFoundError:
                continue  # Atomic publication can rename a transient file.
            require(stat.S_ISREG(item.st_mode) or stat.S_ISDIR(item.st_mode), "output_special_file")
            if stat.S_ISREG(item.st_mode):
                result += item.st_size
    return result


def monitor(
    process,
    limits,
    started,
    directory,
    *,
    lifecycle=None,
    available=None,
    free_bytes=None,
    usage=None,
):
    lifecycle = lifecycle or helper()
    available = available or lifecycle.memory_available
    free_bytes = free_bytes or (lambda: shutil.disk_usage(directory).free)
    usage = usage or (lambda: raw_bytes(directory))
    peak_rss, peak_raw, next_storage = 0, 0, 0.0
    stop_reason = None
    try:
        while process.poll() is None:
            now = time.monotonic()
            rss = sum(item["rss"] for item in lifecycle.process_session_snapshot(process.pid))
            peak_rss = max(peak_rss, rss)
            if now - started >= limits["wall_seconds"]:
                stop_reason = "wall_limit"
            elif rss > limits["maximum_owned_rss_bytes"]:
                stop_reason = "owned_rss_limit"
            elif available() < limits["host_pressure_floor_bytes"]:
                stop_reason = "host_memory_pressure"
            if now >= next_storage:
                peak_raw = max(peak_raw, usage())
                if peak_raw > limits["maximum_raw_run_bytes"]:
                    stop_reason = stop_reason or "raw_run_storage_limit"
                elif free_bytes() < limits["minimum_remaining_scratch_bytes"]:
                    stop_reason = stop_reason or "scratch_reserve_limit"
                next_storage = now + limits["storage_poll_seconds"]
            if stop_reason:
                break
            time.sleep(limits["poll_seconds"])
    finally:
        original = sys.exception()
        try:
            if process.poll() is None or lifecycle.process_session_snapshot(process.pid):
                lifecycle.terminate_owned_session(process, limits["termination_grace_seconds"])
        except BaseException:
            if original is None:
                raise
    require(not lifecycle.process_session_snapshot(process.pid), "owned_process_survived")
    peak_raw = max(peak_raw, usage())
    if peak_raw > limits["maximum_raw_run_bytes"]:
        stop_reason = stop_reason or "raw_run_storage_limit"
    return {
        "exit_code": process.wait(),
        "stop_reason": stop_reason,
        "peak_sampled_owned_rss_bytes": peak_rss,
        "peak_sampled_raw_run_bytes": peak_raw,
        "elapsed_seconds": time.monotonic() - started,
        "owned_live_processes_remaining": 0,
    }


def environment(frozen):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(frozen["source_root"]) / "src"),
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
    )
    return env


def validate_worker_terminal(root, expected_freeze_sha):
    """Authenticate the child's final bindings without another inference or fit."""
    root = Path(root)
    lifecycle = helper()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "terminal_symlink")
        require(
            path.name not in ("failure.json", "worker-failure.json", "worker-cleanup-failure.json")
            and not path.name.startswith(".pending-")
            and not (path.name.startswith(".") and path.name.endswith(".tmp")),
            "failure_accompanies_terminal",
        )

    def read(relative, expected=None):
        path = root / relative
        sha = digest(path)
        require(expected is None or sha == expected, "terminal_artifact_hash")
        value = lifecycle.hash_bound_json(path, sha)
        require(canonical(value) == path.read_bytes(), "terminal_artifact_canonical")
        return value, sha

    terminal, terminal_sha = read("worker-terminal.json")
    require(
        set(terminal)
        == set(
            "schema_version status freeze_sha256 acquisition_terminal_sha256 "
            "encoder_terminal_sha256 fit_terminal_sha256 body_entry_counts "
            "actual_fit_entries nested_counts_nonadditive".split()
        ),
        "worker_terminal_schema",
    )
    require(
        terminal["schema_version"] == "a186-native-worker-v1"
        and terminal["status"] == "completed"
        and terminal["freeze_sha256"] == expected_freeze_sha
        and terminal["nested_counts_nonadditive"] is True,
        "worker_terminal_binding",
    )
    counts = terminal["body_entry_counts"]
    require(
        type(counts) is dict
        and set(counts)
        == {
            "causal_lm_entries",
            "nested_decoder_entries",
            "bert_model_entries",
            "nested_bert_encoder_entries",
        },
        "worker_count_keys",
    )
    require(all(type(value) is int and value >= 0 for value in counts.values()), "worker_counts")
    require(
        112 <= counts["causal_lm_entries"] == counts["nested_decoder_entries"] <= 7280
        and counts["bert_model_entries"] == counts["nested_bert_encoder_entries"] <= 1792,
        "worker_entry_ceiling",
    )
    fits = terminal["actual_fit_entries"]
    require(type(fits) is int and 0 <= fits <= 2, "worker_fit_count")
    acquisition, _ = read("acquisition/terminal.json", terminal["acquisition_terminal_sha256"])
    encoding, _ = read("encoding/terminal.json", terminal["encoder_terminal_sha256"])
    fitting, _ = read("fit/terminal.json", terminal["fit_terminal_sha256"])
    require(
        acquisition["status"] == "finished"
        and type(acquisition["summary"]["completed_rows"]) is int
        and acquisition["summary"]["completed_rows"] == 112
        and type(acquisition["summary"]["total_entries"]) is int
        and acquisition["summary"]["total_entries"] == counts["causal_lm_entries"],
        "acquisition_terminal_counts",
    )
    require(
        encoding["status"] == "completed"
        and type(encoding["entries"]) is int
        and encoding["entries"] == counts["bert_model_entries"],
        "encoder_terminal_counts",
    )
    require(
        fitting["status"] == "completed"
        and type(fitting["actual_fit_entries"]) is int
        and fitting["actual_fit_entries"] == fits
        and fitting["model_free_replay"] is True
        and fitting["profile_restored"] is True,
        "fit_terminal_counts",
    )
    for filename, event, sha in (
        (
            "acquisition-normal-return.json",
            "run_acquisition_returned_normally",
            terminal["acquisition_terminal_sha256"],
        ),
        (
            "encoder-normal-return.json",
            "encode_packets_returned_normally",
            terminal["encoder_terminal_sha256"],
        ),
    ):
        normal, _ = read(filename)
        require(
            canonical(normal) == canonical({"event": event, "terminal_sha256": sha}),
            "observed_stage_return",
        )
    normal, _ = read("fit/normal-return.json", fitting["normal_return_sha256"])
    require(
        normal["fit_predict_returned"] is True
        and normal["terminal_error"] is None
        and normal["returned_result_sha256"] == fitting["result_sha256"],
        "observed_fit_return",
    )
    require(digest(root / "fit/result.json") == fitting["result_sha256"], "fitted_result_hash")
    return terminal_sha


def supervise(path, expected_sha):
    frozen = verify_freeze(path, expected_sha)
    lifecycle = helper()
    probe = lifecycle.open_pidfd(os.getpid())
    try:
        lifecycle.signal_pidfd(probe, 0)
    finally:
        os.close(probe)
    root = Path(frozen["output_root"])
    # This cooperative lock coordinates this study, not unrelated server jobs.
    with lifecycle.owned_interruptions(), (root.parent / ".cpu-study.lock").open("a") as lock:
        os.chmod(root.parent / ".cpu-study.lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory, disk = lifecycle.memory_available(), shutil.disk_usage(root.parent).free
        require(memory >= LIMITS["minimum_available_bytes"], "not_ready_memory")
        require(disk >= LIMITS["minimum_free_scratch_bytes"], "not_ready_disk")
        root.mkdir(mode=0o700)  # An existing claim can never be resumed or replaced.
        publish(
            root / "launch.json",
            {
                "freeze_sha256": expected_sha,
                "supervisor_pid": os.getpid(),
                "available_memory_bytes": memory,
                "free_scratch_bytes": disk,
                "limits": LIMITS,
            },
        )
        process, started = None, time.monotonic()
        try:
            with (root / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        NATIVE_PYTHON,
                        "-B",
                        str(
                            Path(frozen["source_root"]) / "scripts/matched_prefix_native_worker.py"
                        ),
                        "--freeze",
                        str(path),
                        "--freeze-sha256",
                        expected_sha,
                        "--output",
                        str(root / "worker"),
                        "--parent",
                        str(os.getpid()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment(frozen),
                    start_new_session=True,
                )
                outcome = monitor(process, LIMITS, started, root, lifecycle=lifecycle)
            outcome["freeze_sha256"] = expected_sha
            outcome["status"] = "failed"
            outcome["worker_terminal_sha256"] = None
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
                    },
                )
            except BaseException:
                pass  # Do not replace the original interruption or disk failure.
            raise
        finally:
            original = sys.exception()
            try:
                if process is not None and (
                    process.poll() is None or lifecycle.process_session_snapshot(process.pid)
                ):
                    lifecycle.terminate_owned_session(process, LIMITS["termination_grace_seconds"])
            except BaseException:
                if original is None:
                    raise


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    try:
        result = supervise(args.freeze, args.freeze_sha256)
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
