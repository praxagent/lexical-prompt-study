"""A186 freeze and new storage supervision; invented fixtures, no models."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_matched_prefix_prediction.py"
SPEC = importlib.util.spec_from_file_location("a186_supervisor_tests", SCRIPT)
execution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(execution)
LIFECYCLE = execution.helper()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    source = prepared / "source"
    source.mkdir(parents=True)
    for name in execution.SOURCE_PATHS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# invented qualification source\n")
    config = prepared / "target-config.json"
    config.write_bytes(b"{}")
    binding = {"invented": True}
    value = {
        "schema_version": "a186-native-freeze-v1",
        "source_root": str(source),
        "sources": {name: execution.digest(source / name) for name in execution.SOURCE_PATHS},
        "target_config_path": str(config),
        "target_config_sha256": execution.digest(config),
        "runtime_binding": {},
        "plan": {},
        "semantic_assets": {},
        "encoder_binding_sha256": hashlib.sha256(execution.canonical(binding)).hexdigest(),
        "fit_runtime": {},
        "fit_python": execution.FIT_PYTHON,
        "fit_worker": "scripts/matched_prefix_fit_worker.py",
        "native_python": execution.NATIVE_PYTHON,
        "limits": dict(execution.LIMITS),
        "output_root": str(tmp_path / "run-001"),
        "prepared_root": str(prepared),
        "files_sha256": execution.inventory(prepared),
        "public_commit": "a" * 40,
        "selection_sha256": "b" * 64,
        "qualification_sha256": "c" * 64,
        "encoder_runtime": {},
        "encoder_binding": binding,
    }
    path = prepared / "freeze.json"
    monkeypatch.setattr(execution, "helper", lambda: LIFECYCLE)
    monkeypatch.setattr(
        execution, "__file__", str(source / "scripts/run_matched_prefix_prediction.py")
    )
    return path, value


def write_freeze(path, value):
    path.write_bytes(execution.canonical(value))
    return execution.digest(path)


def test_complete_exact_freeze_is_accepted(frozen):
    path, value = frozen
    sha = write_freeze(path, value)
    assert execution.verify_freeze(path, sha, enforce_executable=False) == value


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "source", "limits", "symlink", "binding", "escape"]
)
def test_freeze_rejects_unbound_or_reselected_state(frozen, mutation):
    path, value = frozen
    if mutation == "extra":
        (path.parent / "extra.txt").write_text("invented")
    elif mutation == "missing":
        (Path(value["source_root"]) / execution.SOURCE_PATHS[0]).unlink()
    elif mutation == "source":
        (Path(value["source_root"]) / execution.SOURCE_PATHS[0]).write_text("changed")
    elif mutation == "limits":
        value["limits"]["wall_seconds"] += 1
    elif mutation == "symlink":
        (path.parent / "linked").symlink_to(path.parent)
    elif mutation == "binding":
        value["encoder_binding"]["invented"] = False
    else:
        value["files_sha256"]["../outside"] = "f" * 64
    sha = write_freeze(path, value)
    with pytest.raises(ValueError):
        execution.verify_freeze(path, sha, enforce_executable=False)


def test_noncanonical_duplicate_keys_are_rejected(frozen):
    path, value = frozen
    raw = execution.canonical(value)
    path.write_bytes(b'{"schema_version":"invented",' + raw[1:])
    with pytest.raises(ValueError):
        execution.verify_freeze(path, execution.digest(path), enforce_executable=False)


def child(code="import time; time.sleep(10)"):
    return subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def limits(**updates):
    result = dict(execution.LIMITS)
    result.update(
        wall_seconds=2, termination_grace_seconds=0.1, poll_seconds=0.01, storage_poll_seconds=0.01
    )
    result.update(updates)
    return result


@pytest.mark.parametrize("stop", ["raw_run_storage_limit", "scratch_reserve_limit", "wall_limit"])
def test_storage_or_time_stop_leaves_foreign_process_alive(tmp_path, stop):
    owned, foreign = child(), child()
    try:
        result = execution.monitor(
            owned,
            limits(wall_seconds=0 if stop == "wall_limit" else 2),
            time.monotonic(),
            tmp_path,
            lifecycle=LIFECYCLE,
            available=lambda: 100 * 1024**3,
            free_bytes=lambda: 0 if stop == "scratch_reserve_limit" else 100 * 1024**3,
            usage=lambda: 9 * 1024**3 if stop == "raw_run_storage_limit" else 0,
        )
        assert result["stop_reason"] == stop
        assert result["owned_live_processes_remaining"] == 0 and foreign.poll() is None
    finally:
        LIFECYCLE.terminate_owned_session(owned, 0.1)
        LIFECYCLE.terminate_owned_session(foreign, 0.1)


def test_monitor_preserves_sampling_exception_and_cleans_owned_session(tmp_path):
    process = child()
    original = RuntimeError("invented storage failure")

    def broken():
        raise original

    with pytest.raises(RuntimeError) as raised:
        execution.monitor(
            process,
            limits(),
            time.monotonic(),
            tmp_path,
            lifecycle=LIFECYCLE,
            available=lambda: 100 * 1024**3,
            usage=broken,
        )
    assert raised.value is original
    assert not LIFECYCLE.process_session_snapshot(process.pid)


def test_completed_process_still_has_final_storage_check(tmp_path):
    process = child("pass")
    process.wait(timeout=3)
    result = execution.monitor(
        process,
        limits(),
        time.monotonic(),
        tmp_path,
        lifecycle=LIFECYCLE,
        usage=lambda: 9 * 1024**3,
    )
    assert result["exit_code"] == 0 and result["stop_reason"] == "raw_run_storage_limit"


def test_raw_usage_rejects_link_and_counts_nested_bytes(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/fixture").write_bytes(b"12345")
    assert execution.raw_bytes(tmp_path) == 5
    (tmp_path / "link").symlink_to(tmp_path / "nested/fixture")
    with pytest.raises(ValueError):
        execution.raw_bytes(tmp_path)


def test_receipt_claim_is_exclusive(tmp_path):
    path = tmp_path / "receipt.json"
    sha = execution.publish(path, {"invented": 1})
    assert sha == execution.digest(path)
    with pytest.raises(FileExistsError):
        execution.publish(path, {"invented": 2})


def terminal_fixture(root):
    root.mkdir()
    values = {
        "acquisition": {
            "status": "finished",
            "summary": {"completed_rows": 112, "total_entries": 224},
        },
        "encoding": {"status": "completed", "entries": 224},
        "fit": {
            "status": "completed",
            "actual_fit_entries": 2,
            "model_free_replay": True,
            "profile_restored": True,
        },
    }
    hashes = {}
    for name, value in values.items():
        (root / name).mkdir()
        if name == "fit":
            result_sha = execution.publish(root / "fit/result.json", {"invented": True})
            normal_sha = execution.publish(
                root / "fit/normal-return.json",
                {
                    "fit_predict_returned": True,
                    "terminal_error": None,
                    "returned_result_sha256": result_sha,
                },
            )
            value.update(result_sha256=result_sha, normal_return_sha256=normal_sha)
        hashes[name] = execution.publish(root / name / "terminal.json", value)
    for kind, event in (
        ("acquisition", "run_acquisition_returned_normally"),
        ("encoding", "encode_packets_returned_normally"),
    ):
        name = "acquisition" if kind == "acquisition" else "encoder"
        execution.publish(
            root / (name + "-normal-return.json"),
            {
                "event": event,
                "terminal_sha256": hashes[kind],
            },
        )
    terminal = {
        "schema_version": "a186-native-worker-v1",
        "status": "completed",
        "freeze_sha256": "a" * 64,
        "acquisition_terminal_sha256": hashes["acquisition"],
        "encoder_terminal_sha256": hashes["encoding"],
        "fit_terminal_sha256": hashes["fit"],
        "body_entry_counts": {
            "causal_lm_entries": 224,
            "nested_decoder_entries": 224,
            "bert_model_entries": 224,
            "nested_bert_encoder_entries": 224,
        },
        "actual_fit_entries": 2,
        "nested_counts_nonadditive": True,
    }
    execution.publish(root / "worker-terminal.json", terminal)
    return terminal


def test_complete_worker_terminal_is_bound_without_model_or_fit(tmp_path):
    root = tmp_path / "worker"
    terminal_fixture(root)
    assert execution.validate_worker_terminal(root, "a" * 64) == execution.digest(
        root / "worker-terminal.json"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "absent",
        "malformed",
        "wrong_freeze",
        "failure",
        "nested_failure",
        "dangling_failure",
        "bool_count",
        "negative",
        "over_budget",
        "unpaired",
        "fit_bool",
        "nested_flag",
        "unbound_nested",
        "cleanup_failure",
        "orphan_temporary",
        "missing_normal_return",
        "empty_target_count",
    ],
)
def test_false_worker_success_is_rejected(tmp_path, mutation):
    root = tmp_path / "worker"
    terminal = terminal_fixture(root)
    path = root / "worker-terminal.json"
    if mutation == "absent":
        path.unlink()
    elif mutation == "malformed":
        path.write_bytes(b"{")
    elif mutation == "failure":
        (root / "worker-failure.json").write_bytes(b"{}")
    elif mutation == "nested_failure":
        (root / "fit/failure.json").write_bytes(b"{}")
    elif mutation == "dangling_failure":
        (root / "fit/failure.json").symlink_to(root / "missing")
    elif mutation == "unbound_nested":
        (root / "fit/terminal.json").write_bytes(b"{}")
    elif mutation == "cleanup_failure":
        (root / "worker-cleanup-failure.json").write_bytes(b"{}")
    elif mutation == "orphan_temporary":
        (root / "acquisition/.result.json.tmp").write_bytes(b"{}")
    elif mutation == "missing_normal_return":
        (root / "encoder-normal-return.json").unlink()
    else:
        if mutation == "wrong_freeze":
            terminal["freeze_sha256"] = "b" * 64
        elif mutation == "bool_count":
            terminal["body_entry_counts"]["causal_lm_entries"] = True
        elif mutation == "negative":
            terminal["body_entry_counts"]["causal_lm_entries"] = -1
        elif mutation == "over_budget":
            terminal["body_entry_counts"]["causal_lm_entries"] = 7281
        elif mutation == "unpaired":
            terminal["body_entry_counts"]["nested_decoder_entries"] -= 1
        elif mutation == "fit_bool":
            terminal["actual_fit_entries"] = True
        elif mutation == "empty_target_count":
            terminal["body_entry_counts"]["causal_lm_entries"] = 0
            terminal["body_entry_counts"]["nested_decoder_entries"] = 0
        else:
            terminal["nested_counts_nonadditive"] = 1
        path.write_bytes(execution.canonical(terminal))
    with pytest.raises((ValueError, FileNotFoundError)):
        execution.validate_worker_terminal(root, "a" * 64)


def test_existing_run_is_never_adopted_or_launched(tmp_path, monkeypatch):
    output = tmp_path / "run-001"
    output.mkdir()
    existing = output / "existing.json"
    existing.write_bytes(b'{"invented":"preserved"}')
    frozen = {"output_root": str(output)}
    monkeypatch.setattr(execution, "verify_freeze", lambda *_: frozen)
    monkeypatch.setattr(LIFECYCLE, "memory_available", lambda: 100 * 1024**3)
    monkeypatch.setattr(
        execution.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    monkeypatch.setattr(execution, "helper", lambda: LIFECYCLE)

    def forbidden(*args, **kwargs):
        pytest.fail("A previously claimed run may not launch another worker")

    monkeypatch.setattr(execution.subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError):
        execution.supervise(tmp_path / "invented-freeze.json", "a" * 64)
    assert existing.read_bytes() == b'{"invented":"preserved"}'
    assert set(p.name for p in output.iterdir()) == {"existing.json"}
