"""Real subprocess lifecycle tests; no model, native tokenizer or inference."""

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import shutil
import sys
import time

import pytest

_path = Path(__file__).parents[1] / "scripts/run_native_prefix_qualification.py"
_spec = importlib.util.spec_from_file_location("native_prefix_execution_test_module", _path)
execution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(execution)


def limits(**updates):
    result = dict(execution.LIMITS)
    result.update(wall_seconds=2, termination_grace_seconds=0.1, poll_seconds=0.01)
    result.update(updates)
    return result


def child(code="import time; time.sleep(10)"):
    return subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_normal_child_completion_is_recorded_without_retry():
    process = child("pass")
    result = execution.monitor_process(
        process, limits(), time.monotonic(), available=lambda: 100 * 1024**3
    )
    assert result["exit_code"] == 0 and result["stop_reason"] is None
    assert result["owned_live_processes_remaining"] == 0
    assert execution.process_session_snapshot(process.pid) == []


def test_timeout_kills_only_owned_session_and_escalates_if_needed():
    unrelated = child()
    owned = child(
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)"
    )
    try:
        result = execution.monitor_process(
            owned,
            limits(wall_seconds=0.15),
            time.monotonic(),
            available=lambda: 100 * 1024**3,
        )
        assert result["stop_reason"] == "wall_limit"
        assert result["exit_code"] in (-signal.SIGTERM, -signal.SIGKILL)
        assert unrelated.poll() is None
        assert execution.process_session_snapshot(owned.pid) == []
    finally:
        execution.terminate_owned_session(unrelated, 0.1)
        execution.terminate_owned_session(owned, 0.1)


@pytest.mark.parametrize("reason", ["host_memory_pressure", "owned_rss_limit"])
def test_resource_threshold_stops_owned_work(reason):
    process = child()
    selected = limits(maximum_owned_rss_bytes=0) if reason == "owned_rss_limit" else limits()
    result = execution.monitor_process(
        process,
        selected,
        time.monotonic(),
        available=lambda: 0 if reason == "host_memory_pressure" else 100 * 1024**3,
    )
    assert result["stop_reason"] == reason
    assert result["exit_code"] < 0
    assert result["owned_live_processes_remaining"] == 0


def test_monitor_exception_preserves_failure_and_removes_owned_process():
    process = child()
    original = RuntimeError("invented observer failure")

    def fail():
        raise original

    with pytest.raises(RuntimeError) as raised:
        execution.monitor_process(process, limits(), time.monotonic(), available=fail)
    assert raised.value is original
    assert process.poll() is not None
    assert execution.process_session_snapshot(process.pid) == []


@pytest.mark.parametrize("subgroup", [False, True])
def test_owned_descendant_is_included_and_stopped(subgroup):
    process = child(
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']"
        + (",process_group=0" if subgroup else "")
        + "); "
        "time.sleep(10)"
    )
    # Give the owned parent time to create its child before applying the cap.
    deadline = time.monotonic() + 2
    try:
        while len(execution.process_session_snapshot(process.pid)) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        result = execution.monitor_process(
            process,
            limits(wall_seconds=0),
            time.monotonic(),
            available=lambda: 100 * 1024**3,
        )
        assert result["stop_reason"] == "wall_limit"
        assert result["owned_live_processes_remaining"] == 0
    finally:
        execution.terminate_owned_session(process, 0.1)


def wait_ready(path, process):
    until = time.monotonic() + 3
    while not path.exists():
        assert process.poll() is None
        assert time.monotonic() < until
        time.sleep(0.01)
    return int(path.read_text())


def test_supervisor_term_runs_cleanup_for_owned_worker(tmp_path):
    ready = tmp_path / "worker-pid"
    code = (
        "import runpy,subprocess,sys,time; from pathlib import Path; "
        f"api=runpy.run_path({str(_path)!r}); "
        "scope=api['owned_interruptions'](); scope.__enter__(); "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True); "
        f"Path({str(ready)!r}).write_text(str(p.pid)); "
        "limits=dict(api['LIMITS']); limits.update(termination_grace_seconds=0.1,poll_seconds=0.01); "
        "api['monitor_process'](p,limits,time.monotonic(),available=lambda:100*1024**3)"
    )
    parent = child(code)
    worker_pid = None
    try:
        worker_pid = wait_ready(ready, parent)
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=3)
        assert execution.process_session_snapshot(worker_pid) == []
    finally:
        execution.terminate_owned_session(parent, 0.1)
        if worker_pid is not None:
            execution._signal_session(worker_pid, signal.SIGKILL)


def test_worker_parent_death_binding_handles_supervisor_sigkill(tmp_path):
    ready = tmp_path / "worker-pid"
    worker_code = (
        "import runpy,sys,os,time; from pathlib import Path; "
        "api=runpy.run_path(sys.argv[1]); api['bind_parent_lifetime'](int(sys.argv[2])); "
        "Path(sys.argv[3]).write_text(str(os.getpid())); time.sleep(10)"
    )
    parent = child(
        "import subprocess,sys,os; "
        f"p=subprocess.Popen([sys.executable,'-c',{worker_code!r},{str(_path)!r},str(os.getpid()),{str(ready)!r}],start_new_session=True); "
        "p.wait()"
    )
    worker_pid = None
    try:
        worker_pid = wait_ready(ready, parent)
        parent.kill()
        parent.wait(timeout=3)
        until = time.monotonic() + 3
        while execution.process_session_snapshot(worker_pid):
            assert time.monotonic() < until
            time.sleep(0.01)
    finally:
        execution.terminate_owned_session(parent, 0.1)
        if worker_pid is not None:
            execution._signal_session(worker_pid, signal.SIGKILL)


def freeze_fixture(tmp_path, monkeypatch):
    repository = Path(__file__).parents[1]
    prepared = tmp_path / "prepared-001"
    prepared.mkdir()
    for name in execution.SOURCE_PATHS:
        target = prepared / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    for name in ("plan.json", "legacy-runtime.json", "runtime-binding.json"):
        (prepared / name).write_text("{}\n")
    files = {
        str(p.relative_to(prepared)): execution.digest(p)
        for p in prepared.rglob("*")
        if p.is_file()
    }
    freeze = {
        "schema_version": "a185-native-freeze-v1",
        "prepared_root": str(prepared),
        "output_root": str(tmp_path / "run-001"),
        "runtime_executable": sys.executable,
        "limits": dict(execution.LIMITS),
        "files_sha256": files,
        "plan_path": "plan.json",
        "legacy_config_path": "legacy-runtime.json",
        "runtime_binding_path": "runtime-binding.json",
        "loader_path": "source/scripts/run_cpu_reference_audit.py",
        "public_commit": "a" * 40,
    }
    path = prepared / "FREEZE.json"
    path.write_text(json.dumps(freeze, sort_keys=True))
    monkeypatch.setattr(
        execution, "__file__", str(prepared / "source/scripts/run_native_prefix_qualification.py")
    )
    monkeypatch.setattr(execution, "PYTHON", sys.executable)
    return path, freeze


def test_exact_frozen_source_closure_is_accepted_without_execution(tmp_path, monkeypatch):
    path, freeze = freeze_fixture(tmp_path, monkeypatch)
    assert execution.verify_freeze(path, execution.digest(path)) == freeze
    assert not Path(freeze["output_root"]).exists()


@pytest.mark.parametrize(
    "mutation",
    ["source_bytes", "extra_file", "symlink", "missing_source", "limits", "output_escape"],
)
def test_freeze_rejects_unbound_sources_and_policy_changes(tmp_path, monkeypatch, mutation):
    path, freeze = freeze_fixture(tmp_path, monkeypatch)
    prepared = path.parent
    if mutation == "source_bytes":
        (prepared / "plan.json").write_text('{"changed":true}')
    elif mutation == "extra_file":
        (prepared / "source/src/torch.py").write_text('raise RuntimeError("invented shadow")\n')
    elif mutation == "symlink":
        target = prepared / "plan.json"
        target.unlink()
        target.symlink_to(prepared / "legacy-runtime.json")
    elif mutation == "missing_source":
        key = "source/src/lexical_prompt_study/native_prefix_qualification.py"
        (prepared / key).unlink()
        del freeze["files_sha256"][key]
    elif mutation == "limits":
        freeze["limits"]["wall_seconds"] = 1801
    else:
        freeze["output_root"] = str(tmp_path / "different-run")
    path.write_text(json.dumps(freeze, sort_keys=True))
    with pytest.raises(ValueError):
        execution.verify_freeze(path, execution.digest(path))
    assert not (tmp_path / "run-001").exists()


def test_runtime_binding_distinguishes_legacy_metadata_from_selected_policy():
    legacy = {
        "runtime_versions": {"python": "invented"},
        "model_revision": "a" * 40,
        "model_files_sha256": {"invented": "b" * 64},
        "chat_template_sha256": "c" * 64,
        "seed": 20260915,
        "max_new_tokens": 64,
        "quantization": "nf4",
        "device": "cuda:0",
    }
    binding = execution.expected_runtime_binding(legacy, "d" * 64)
    actual = binding["actual_requested_execution"]
    assert actual["sampled_token_cap"] == 16 and actual["cache"] is False
    assert (
        actual["device"] == "cpu"
        and actual["dtype"] == "float32"
        and actual["quantization"] is None
    )
    assert legacy["max_new_tokens"] == 64 and legacy["device"] == "cuda:0"


def test_hash_bound_vendor_json_accepts_original_encoding_and_authenticated_snapshot_link(tmp_path):
    vendor = tmp_path / "vendor.json"
    vendor.write_text('{\n  "eos_token_id": [2, 3],\n  "temperature": 0.6\n}\n')
    snapshot = tmp_path / "snapshot.json"
    snapshot.symlink_to(vendor)
    expected = {"eos_token_id": [2, 3], "temperature": 0.6}
    sha = execution.digest(vendor)
    assert execution.hash_bound_json(vendor, sha) == expected
    assert execution.hash_bound_json(snapshot, sha, allow_snapshot_link=True) == expected
    with pytest.raises(ValueError, match="json_symlink"):
        execution.hash_bound_json(snapshot, sha)
    with pytest.raises(ValueError, match="json_hash"):
        execution.hash_bound_json(vendor, "0" * 64)


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":-Infinity}',
        '{"a":1e999}',
        '{"a":{"b":1,"b":2}}',
        "[]",
        "null",
    ],
)
def test_hash_bound_vendor_json_rejects_ambiguous_or_nonfinite_metadata(tmp_path, payload):
    vendor = tmp_path / "vendor.json"
    vendor.write_text(payload)
    with pytest.raises(ValueError):
        execution.hash_bound_json(vendor, execution.digest(vendor))
