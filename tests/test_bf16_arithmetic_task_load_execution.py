"""Invented stdlib fixtures: no Torch, tokenizer, target, fit or historical data."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/run_bf16_arithmetic_task_load.py"


def module(path=SCRIPT):
    spec = importlib.util.spec_from_file_location("a188_synthetic_outer", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module()


def fresh(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.canonical(value))


@pytest.fixture
def frozen(tmp_path):
    prepared = tmp_path / "prepared"
    source = prepared / "source"
    for relative in m.SOURCE_PATHS:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, target)
    fresh(prepared / "config.json", {"invented_config": True})
    fresh(prepared / "selection.json", {"selected": "invented"})
    fresh(prepared / "qualification.json", {"qualification": "invented"})
    value = {
        "schema_version": "a188-native-freeze-v1",
        "source_root": str(source),
        "sources": {name: m.digest(source / name) for name in m.SOURCE_PATHS},
        "target_config_path": str(prepared / "config.json"),
        "target_config_sha256": m.digest(prepared / "config.json"),
        "runtime_binding": {},
        "plan": {},
        "native_python": m.NATIVE_PYTHON,
        "limits": dict(m.LIMITS),
        "output_root": str(tmp_path / "run-001"),
        "prepared_root": str(prepared),
        "files_sha256": m.inventory(prepared),
        "public_commit": "f" * 40,
        "selection_sha256": m.digest(prepared / "selection.json"),
        "qualification_sha256": m.digest(prepared / "qualification.json"),
    }
    path = prepared / "freeze.json"
    fresh(path, value)
    return module(source / "scripts/run_bf16_arithmetic_task_load.py"), path, value


def test_complete_frozen_package(frozen):
    copied, path, value = frozen
    assert copied.verify_freeze(path, m.digest(path), enforce_executable=False) == value


@pytest.mark.parametrize(
    "change", ["bool_limit", "extra_source", "commit", "selection", "output", "python", "version"]
)
def test_rehashed_freeze_contract_tamper(frozen, change):
    copied, path, value = frozen
    if change == "bool_limit":
        value["limits"]["termination_grace_seconds"] = True
    elif change == "extra_source":
        value["sources"]["scripts/other.py"] = "a" * 64
    elif change == "commit":
        value["public_commit"] = "f" * 39
    elif change == "selection":
        value["selection_sha256"] = "a" * 64
    elif change == "output":
        value["output_root"] += "-retry"
    elif change == "python":
        value["native_python"] = sys.executable
    else:
        value["schema_version"] = "a186-native-freeze-v1"
    fresh(path, value)
    with pytest.raises(ValueError):
        copied.verify_freeze(path, m.digest(path), enforce_executable=False)


@pytest.mark.parametrize("change", ["file", "unlisted", "symlink", "source_pin"])
def test_source_closure_tamper(frozen, change):
    copied, path, value = frozen
    target = Path(value["source_root"]) / "scripts/run_native_prefix_qualification.py"
    if change == "unlisted":
        (path.parent / "extra").write_text("invented")
    elif change == "symlink":
        body = target.read_bytes()
        target.unlink()
        replacement = path.parent.parent / "outside.py"
        replacement.write_bytes(body)
        target.symlink_to(replacement)
    else:
        target.write_bytes(target.read_bytes() + b"\n# invented drift\n")
        if change == "source_pin":
            value["sources"]["scripts/run_native_prefix_qualification.py"] = m.digest(target)
            value["files_sha256"]["source/scripts/run_native_prefix_qualification.py"] = m.digest(
                target
            )
            fresh(path, value)
    with pytest.raises(ValueError):
        copied.verify_freeze(path, m.digest(path), enforce_executable=False)


def test_unfrozen_running_copy_rejected(frozen):
    _, path, _ = frozen
    with pytest.raises(ValueError, match="actual_frozen_runner"):
        m.verify_freeze(path, m.digest(path), enforce_executable=False)


def test_import_is_stdlib_only_and_optimized_run_refuses():
    code = (
        "import importlib.util,sys; s=importlib.util.spec_from_file_location('outer',sys.argv[1]);"
        " m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        " assert not any(x in sys.modules for x in ('torch','transformers','numpy','sklearn'));print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0 and result.stdout == "ok\n"
    result = subprocess.run(
        [sys.executable, "-B", "-O", str(SCRIPT), "--help"], capture_output=True
    )
    assert result.returncode != 0 and b"a188_optimization_forbidden" in result.stderr


def test_environment_excludes_devices_and_startup(monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/invented")
    monkeypatch.setenv("PYTHONSTARTUP", "/invented")
    env = m.environment({"source_root": "/invented/frozen"})
    assert all(env[key] == value for key, value in m.ENV.items())
    assert "PYTHONHOME" not in env and "PYTHONSTARTUP" not in env
    assert env["PYTHONPATH"] == "/invented/frozen/src"


def test_exclusive_publication_and_orphan(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    m.publish(path, {"a": 1})
    with pytest.raises(FileExistsError):
        m.publish(path, {"a": 2})
    assert m.read_json(path) == {"a": 1}
    assert path.with_name(".pending-receipt.json").exists()
    monkeypatch.setattr(m.os, "link", lambda *a, **kw: (_ for _ in ()).throw(OSError("private")))
    with pytest.raises(OSError):
        m.publish(tmp_path / "later.json", {})
    assert not (tmp_path / "later.json").exists()
    assert (tmp_path / ".pending-later.json").exists()


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("failure", ["memory", "scratch", "lock"])
def test_queue_bounded_expiry(tmp_path, failure):
    limits = dict(m.LIMITS, queue_wall_seconds=5, queue_poll_seconds=2)
    clock = Clock()
    life = SimpleNamespace(memory_available=lambda: 0 if failure == "memory" else 64 * 1024**3)
    lock_path = tmp_path / "lock"
    with m.cooperative_lock(lock_path) as holder, m.cooperative_lock(lock_path) as waiter:
        if failure == "lock":
            m.fcntl.flock(holder, m.fcntl.LOCK_EX | m.fcntl.LOCK_NB)
        assert not m.wait_for_resources(
            tmp_path,
            waiter,
            life,
            clock=clock,
            sleep=clock.sleep,
            limits=limits,
            free_bytes=lambda: 0 if failure == "scratch" else 64 * 1024**3,
        )
    assert clock.now == 5
    assert len(list((tmp_path / "queue").iterdir())) == 3
    assert all(m.read_json(p)["ready"] is False for p in (tmp_path / "queue").iterdir())


def test_queue_readiness_holds_lock(tmp_path):
    clock = Clock()
    life = SimpleNamespace(memory_available=lambda: 64 * 1024**3)
    with (
        m.cooperative_lock(tmp_path / "lock") as first,
        m.cooperative_lock(tmp_path / "lock") as second,
    ):
        assert m.wait_for_resources(
            tmp_path, first, life, clock=clock, sleep=clock.sleep, free_bytes=lambda: 64 * 1024**3
        )
        with pytest.raises(BlockingIOError):
            m.fcntl.flock(second, m.fcntl.LOCK_EX | m.fcntl.LOCK_NB)
    assert clock.now == 0


def test_queue_refuses_symlink_lock(tmp_path):
    (tmp_path / "other").write_text("private")
    (tmp_path / "lock").symlink_to(tmp_path / "other")
    with pytest.raises(OSError), m.cooperative_lock(tmp_path / "lock"):
        pass


def test_queue_failure_consumes_one_shot_claim(tmp_path, monkeypatch):
    freeze = {"output_root": str(tmp_path / "run-001")}
    monkeypatch.setattr(m, "verify_freeze", lambda *a, **kw: freeze)
    monkeypatch.setattr(m, "wait_for_resources", lambda *a, **kw: False)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("no launch allowed"))
    result = m.supervise(tmp_path / "freeze", "a" * 64)
    assert result["status"] == "deferred" and result["target_body_entries"] == 0
    with pytest.raises(FileExistsError):
        m.supervise(tmp_path / "freeze", "a" * 64)


def test_queue_interruption_has_no_launch_and_no_retry(tmp_path, monkeypatch):
    freeze = {"output_root": str(tmp_path / "run-001")}
    monkeypatch.setattr(m, "verify_freeze", lambda *a, **kw: freeze)

    def interrupted(*a, **kw):
        raise KeyboardInterrupt("private detail")

    monkeypatch.setattr(m, "wait_for_resources", interrupted)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("no launch allowed"))
    with pytest.raises(KeyboardInterrupt):
        m.supervise(tmp_path / "freeze", "a" * 64)
    raw = (tmp_path / "run-001/supervisor-failure.json").read_bytes()
    assert b"KeyboardInterrupt" in raw and b"private detail" not in raw
    with pytest.raises(FileExistsError):
        m.supervise(tmp_path / "freeze", "a" * 64)


class Decoder:
    def forward(self, fail=False):
        if fail:
            raise RuntimeError("invented private exception")
        return 4


class LM:
    def __init__(self):
        self.model = Decoder()

    def forward(self, fail=False):
        return self.model.forward(fail)


def test_profile_counts_nested_calls_without_adding(tmp_path):
    profile = m.BodyProfile(
        tmp_path / "entries", ((LM, "causal_lm_entries"), (Decoder, "nested_decoder_entries"))
    )
    with profile:
        assert LM().forward() == 4
        with pytest.raises(RuntimeError):
            LM().forward(True)
    assert profile.counts == {"causal_lm_entries": 2, "nested_decoder_entries": 2}
    assert profile.returns == profile.counts  # exits are explicitly not successful-return claims
    assert sys.getprofile() is None
    assert len(list((tmp_path / "entries").iterdir())) == 4


def test_profile_ceiling_counts_attempt_before_body(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MAX_ENTRIES", 1)
    profile = m.BodyProfile(
        tmp_path / "entries", ((LM, "causal_lm_entries"), (Decoder, "nested_decoder_entries"))
    )
    with profile:
        LM().forward()
        with pytest.raises(ValueError, match="body_ceiling"):
            LM().forward()
    assert profile.counts == {"causal_lm_entries": 2, "nested_decoder_entries": 1}
    assert sys.getprofile() is None


def test_profile_existing_owner_refused(tmp_path):
    def prior(*args):
        return None

    sys.setprofile(prior)
    try:
        with pytest.raises(ValueError, match="owned_profiler"):
            m.BodyProfile(
                tmp_path / "entries",
                ((LM, "causal_lm_entries"), (Decoder, "nested_decoder_entries")),
            )
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)


def summary():
    return {
        "planned_generations": 32,
        "completed_rows": 32,
        "failed_rows": 0,
        "unattempted_rows": 0,
        "completed_generations": 32,
        "known_labels": 32,
        "success_labels": 16,
        "generation_entries": 32,
        "entry_lower_bound": 32,
        "entries_complete": True,
        "total_entries": 32,
        "schedule_finished": True,
    }


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    root = tmp_path / "run-001"
    root.mkdir()
    frozen = {
        "output_root": str(root),
        "prepared_root": str(tmp_path / "prepared"),
        "target_config_path": "/invented/config",
        "target_config_sha256": "a" * 64,
        "runtime_binding": {},
        "plan": {"eos_token_ids": [9], "provenance": {"chat_template_sha256": "c" * 64}},
    }
    for k, v in m.ENV.items():
        monkeypatch.setenv(k, v)
    state = {"fail_at": None, "loads": 0, "prepared": 0, "replays": 0, "calls": 0}
    core = SimpleNamespace(
        validate_plan=lambda plan: plan, _primitive_plan=lambda plan: {"invented_geometry": True}
    )

    def prepare(plan, tokenizer):
        state["prepared"] += 1
        return {"invented": "native fixture"}

    core.prepare_inputs = prepare
    primitive = SimpleNamespace(
        clean_model=lambda *a: {
            "tensors": [{"kind": "buffer", "dtype": "torch.float32"}],
            "parameter_dtype": "bfloat16",
        }
    )
    primitive.prepare_technical = lambda *a: {"chat_template_sha256": "c" * 64}

    def technical(model, *, directory, **kwargs):
        directory.mkdir()
        for _ in range(state.get("technical_calls", 2)):
            model.forward(state.get("technical_failure", False))
        terminal = {
            "status": "completed",
            "entries": state.get("technical_calls", 2),
            "qualified": not state.get("mismatch", False),
            "native_repeatable": not state.get("mismatch", False),
            "widened_repeatable": True,
        }
        result_sha = m.publish(directory / "result.json", terminal)
        envelope = {
            "schema_version": "a188-bf16-forward-v1",
            "status": "completed",
            "result_sha256": result_sha,
            "artifact_sha256": m.inventory(directory),
        }
        sha = m.publish(directory / "terminal.json", envelope)
        state["technical_result"] = {**terminal, "terminal_sha256": sha}
        return state["technical_result"]

    primitive.run_technical = technical
    primitive.load_technical = lambda *args: copy.deepcopy(state["technical_result"])
    state["primitive"] = primitive

    def run(model, tokenizer, *, directory, **kwargs):
        directory.mkdir()
        for index in range(32):
            state["calls"] += 1
            model.forward(state["fail_at"] == index)
        terminal = {"status": "finished", "summary": summary()}
        sha = m.publish(directory / "terminal.json", terminal)
        state["result"] = {**terminal, "terminal_sha256": sha}
        return state["result"]

    core.run_acquisition = run

    def replay(*args):
        state["replays"] += 1
        return copy.deepcopy(state["result"])

    core.load_acquisition = replay

    def load(config, engine):
        state["loads"] += 1
        loaded = SimpleNamespace(model=LM(), tokenizer=object(), eos_ids=[9])
        if state.get("forward_in_loader"):
            loaded.model.forward()
        return loaded

    engine = SimpleNamespace(validate_config=lambda config: None, runtime_versions=lambda: {})
    lifecycle = SimpleNamespace(
        hash_bound_json=lambda *a: {"runtime_versions": {}},
        asset_check=lambda config: {"invented": {"sha256": "b" * 64, "bytes": 1}},
        memory_available=lambda: 64 * 1024**3,
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setattr(
        m,
        "native_dependencies",
        lambda freeze: (
            core,
            primitive,
            engine,
            SimpleNamespace(load_cpu_bf16=load),
            torch,
            LM,
            Decoder,
        ),
    )
    original_helper = m.helper
    monkeypatch.setattr(m, "helper", lambda: lifecycle)
    monkeypatch.setattr(m, "validate_target", lambda *a: None)
    monkeypatch.setattr(m, "imported_sources", lambda *a: None)

    def verify_postflight(path, sha):
        assert Path(path) == Path(frozen["prepared_root"]) / "freeze.json"
        assert sha == m.object_hash(frozen)
        state["postflight_checked"] = True
        return frozen

    monkeypatch.setattr(m, "verify_freeze", verify_postflight)
    monkeypatch.setattr(m.shutil, "disk_usage", lambda *a: SimpleNamespace(free=64 * 1024**3))
    return frozen, state, core, original_helper


def test_worker_complete_one_load_one_acquisition_replay(fake_worker, monkeypatch):
    frozen, state, _, real_helper = fake_worker
    terminal = m.run_worker(frozen)
    assert state["loads"] == state["prepared"] == state["replays"] == 1
    assert state["calls"] == 32 and state["postflight_checked"] is True
    assert terminal["body_entry_counts"] == {"causal_lm_entries": 34, "nested_decoder_entries": 34}
    assert terminal["model_free_replay"] is True and sys.getprofile() is None
    monkeypatch.setattr(m, "helper", real_helper)
    assert m.validate_worker_terminal(Path(frozen["output_root"]) / "worker", m.object_hash(frozen))


def test_active_failed_body_is_retained_and_exception_private(fake_worker):
    frozen, state, _, _ = fake_worker
    state["fail_at"] = 2
    with pytest.raises(RuntimeError, match="invented private"):
        m.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    failure = json.loads((root / "worker-failure.json").read_bytes())
    assert failure["body_entry_counts"] == {"causal_lm_entries": 5, "nested_decoder_entries": 5}
    assert failure["phase"] == "acquisition" and failure["error_type"] == "RuntimeError"
    assert b"invented private" not in (root / "worker-failure.json").read_bytes()
    assert state["loads"] == 1 and state["replays"] == 0 and sys.getprofile() is None
    assert not (root / "worker-terminal.json").exists()


def test_native_preparation_failure_no_body_retry(fake_worker):
    frozen, state, core, _ = fake_worker

    def broken(*a):
        raise KeyboardInterrupt()

    core.prepare_inputs = broken
    with pytest.raises(KeyboardInterrupt):
        m.run_worker(frozen)
    failure = json.loads((Path(frozen["output_root"]) / "worker/worker-failure.json").read_bytes())
    assert failure["body_entry_counts"] == {"causal_lm_entries": 0, "nested_decoder_entries": 0}
    assert state["loads"] == 1 and state["calls"] == 0 and sys.getprofile() is None


def test_entry_receipt_failure_retains_declared_active_count(fake_worker, monkeypatch):
    frozen, _, _, _ = fake_worker
    original = m.publish

    def broken(path, value):
        if Path(path).name == "causal_lm_entries_0001.json":
            raise OSError("private disk failure")
        return original(path, value)

    monkeypatch.setattr(m, "publish", broken)
    with pytest.raises(OSError):
        m.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    failure = json.loads((root / "worker-failure.json").read_bytes())
    assert failure["body_entry_counts"] == {"causal_lm_entries": 1, "nested_decoder_entries": 0}
    assert sys.getprofile() is None


def test_after_terminal_publication_failure_is_not_success(fake_worker, monkeypatch):
    frozen, _, _, real_helper = fake_worker
    original = m.publish

    def broken(path, value):
        result = original(path, value)
        if Path(path).name == "worker-terminal.json":
            raise OSError("after publication")
        return result

    monkeypatch.setattr(m, "publish", broken)
    with pytest.raises(OSError):
        m.run_worker(frozen)
    monkeypatch.setattr(m, "helper", real_helper)
    with pytest.raises(ValueError, match="failure_accompanies_terminal"):
        m.validate_worker_terminal(Path(frozen["output_root"]) / "worker", m.object_hash(frozen))


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_count",
        "wrong_rows",
        "incomplete",
        "extra_entry",
        "orphan",
        "normal",
        "assets",
        "freeze",
    ],
)
def test_terminal_rejects_semantic_tamper(fake_worker, monkeypatch, mutation):
    frozen, _, _, real_helper = fake_worker
    m.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    monkeypatch.setattr(m, "helper", real_helper)
    terminal = m.read_json(root / "worker-terminal.json")
    if mutation == "bool_count":
        terminal["body_entry_counts"]["causal_lm_entries"] = True
        terminal["body_exit_events"]["causal_lm_entries"] = True
    elif mutation in ("wrong_rows", "incomplete"):
        path = root / "acquisition/terminal.json"
        value = m.read_json(path)
        value["summary"]["completed_rows" if mutation == "wrong_rows" else "entries_complete"] = (
            31 if mutation == "wrong_rows" else False
        )
        fresh(path, value)
        terminal["acquisition_terminal_sha256"] = m.digest(path)
    elif mutation == "extra_entry":
        fresh(root / "entries/extra.json", {})
    elif mutation == "orphan":
        fresh(root / ".pending-result.json", {})
    elif mutation == "normal":
        path = root / "acquisition-normal-return.json"
        fresh(
            path,
            {"event": "not_returned", "terminal_sha256": terminal["acquisition_terminal_sha256"]},
        )
        terminal["normal_return_sha256"] = m.digest(path)
    elif mutation == "assets":
        path = root / "assets-after.json"
        fresh(path, {"changed": True})
        terminal["assets_after_sha256"] = m.digest(path)
    else:
        terminal["freeze_sha256"] = "c" * 64
    fresh(root / "worker-terminal.json", terminal)
    with pytest.raises(ValueError):
        m.validate_worker_terminal(root, m.object_hash(frozen))


def test_pinned_monitor_stops_owned_only(tmp_path):
    lifecycle = m.helper()
    monitor = m.monitor_module().monitor
    owned = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
    )
    try:
        limits = dict(
            m.LIMITS, wall_seconds=0.05, poll_seconds=0.01, termination_grace_seconds=0.05
        )
        result = monitor(
            owned,
            limits,
            time.monotonic(),
            tmp_path,
            lifecycle=lifecycle,
            available=lambda: 64 * 1024**3,
            free_bytes=lambda: 64 * 1024**3,
            usage=lambda: 0,
        )
        assert (
            result["stop_reason"] == "wall_limit" and result["owned_live_processes_remaining"] == 0
        )
        assert owned.poll() is not None and unrelated.poll() is None
    finally:
        for process in (owned, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait()


@pytest.mark.parametrize("guard", ["owned_rss", "host_memory", "raw", "scratch"])
def test_monitor_resource_guards_without_models(tmp_path, guard):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
    )
    lifecycle = m.helper()
    limits = dict(m.LIMITS, termination_grace_seconds=0.05)
    if guard == "owned_rss":
        limits["maximum_owned_rss_bytes"] = -1
    try:
        result = m.monitor_module().monitor(
            process,
            limits,
            time.monotonic(),
            tmp_path,
            lifecycle=lifecycle,
            available=lambda: 0 if guard == "host_memory" else 64 * 1024**3,
            free_bytes=lambda: 0 if guard == "scratch" else 64 * 1024**3,
            usage=lambda: 4 * 1024**3 if guard == "raw" else 0,
        )
        assert (
            result["stop_reason"]
            == {
                "owned_rss": "owned_rss_limit",
                "host_memory": "host_memory_pressure",
                "raw": "raw_run_storage_limit",
                "scratch": "scratch_reserve_limit",
            }[guard]
        )
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_original_exception_survives_profiler_restoration_failure(fake_worker, monkeypatch):
    frozen, state, _, _ = fake_worker
    state["fail_at"] = 1
    real_setprofile = sys.setprofile
    restored = 0

    def once_failing_restore(value):
        nonlocal restored
        if value is None:
            restored += 1
            if restored == 1:
                raise OSError("private cleanup detail")
        return real_setprofile(value)

    monkeypatch.setattr(m.sys, "setprofile", once_failing_restore)
    try:
        with pytest.raises(RuntimeError, match="invented private exception"):
            m.run_worker(frozen)
    finally:
        real_setprofile(None)
    root = Path(frozen["output_root"]) / "worker"
    failure = json.loads((root / "worker-failure.json").read_bytes())
    cleanup = json.loads((root / "worker-cleanup-failure.json").read_bytes())
    assert failure["error_type"] == "RuntimeError"
    assert cleanup == {"error_types": ["OSError"], "original_error_type": "RuntimeError"}
    assert restored >= 2 and not (root / "worker-terminal.json").exists()


def test_successful_body_with_failed_profile_restore_is_failed(fake_worker, monkeypatch):
    frozen, _, _, _ = fake_worker
    real_setprofile = sys.setprofile
    restored = 0

    def once_failing_restore(value):
        nonlocal restored
        if value is None:
            restored += 1
            if restored == 1:
                raise OSError("private cleanup detail")
        return real_setprofile(value)

    monkeypatch.setattr(m.sys, "setprofile", once_failing_restore)
    try:
        with pytest.raises(OSError):
            m.run_worker(frozen)
    finally:
        real_setprofile(None)
    root = Path(frozen["output_root"]) / "worker"
    assert not (root / "worker-terminal.json").exists()
    assert (root / "worker-cleanup-failure.json").is_file()


def test_raw_storage_rejects_symlinks_and_counts_private_files(tmp_path):
    (tmp_path / "a").write_bytes(b"1234")
    assert m.raw_bytes(tmp_path) == 4
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="output_special_file"):
        m.raw_bytes(tmp_path)


def test_cli_errors_are_class_only(tmp_path):
    missing = tmp_path / "private-sensitive-name"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "supervise",
            "--freeze",
            str(missing),
            "--freeze-sha256",
            "a" * 64,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and result.stderr == ""
    assert json.loads(result.stdout) == {"status": "failed", "error_type": "FileNotFoundError"}
    assert "private-sensitive" not in result.stdout


@pytest.mark.parametrize("technical_calls", [0, 1, 3])
def test_wrong_technical_count_stops_science(fake_worker, technical_calls):
    frozen, state, _, _ = fake_worker
    state["technical_calls"] = technical_calls
    with pytest.raises(ValueError):
        m.run_worker(frozen)
    assert state["calls"] == 0 and state["replays"] == 0
    failure = json.loads((Path(frozen["output_root"]) / "worker/worker-failure.json").read_bytes())
    assert failure["phase"] == "technical_qualification"
    assert (
        failure["body_entry_counts_by_stage"]["technical"]["causal_lm_entries"] == technical_calls
    )
    assert failure["body_entry_counts_by_stage"]["science"]["causal_lm_entries"] == 0


def test_completed_repeatability_mismatch_stops_without_retry(fake_worker):
    frozen, state, _, _ = fake_worker
    state["mismatch"] = True
    with pytest.raises(ValueError, match="technical_qualified"):
        m.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    assert state["loads"] == 1 and state["calls"] == 0
    assert (root / "technical/terminal.json").is_file()
    assert (root / "technical-normal-return.json").is_file()
    assert not (root / "acquisition").exists()


def test_hard_technical_failure_counts_only_technical(fake_worker):
    frozen, state, _, _ = fake_worker
    state["technical_failure"] = True
    with pytest.raises(RuntimeError):
        m.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    failure = json.loads((root / "worker-failure.json").read_bytes())
    assert state["calls"] == 0 and state["loads"] == 1
    assert failure["body_entry_counts_by_stage"] == {
        "nonstudy": {"causal_lm_entries": 0, "nested_decoder_entries": 0},
        "technical": {"causal_lm_entries": 1, "nested_decoder_entries": 1},
        "science": {"causal_lm_entries": 0, "nested_decoder_entries": 0},
    }


def test_unexpected_loader_forward_is_not_technical(fake_worker):
    frozen, state, _, _ = fake_worker
    state["forward_in_loader"] = True
    with pytest.raises(ValueError, match="forward_during_load"):
        m.run_worker(frozen)
    failure = json.loads((Path(frozen["output_root"]) / "worker/worker-failure.json").read_bytes())
    assert failure["body_entry_counts_by_stage"]["nonstudy"]["causal_lm_entries"] == 1
    assert failure["body_entry_counts_by_stage"]["technical"]["causal_lm_entries"] == 0
    assert state["prepared"] == 0 and state["calls"] == 0


def test_technical_template_mismatch_has_zero_body_entries(fake_worker):
    frozen, state, _, _ = fake_worker
    state["primitive"].prepare_technical = lambda *a: {"chat_template_sha256": "d" * 64}
    with pytest.raises(ValueError, match="technical_template"):
        m.run_worker(frozen)
    failure = json.loads((Path(frozen["output_root"]) / "worker/worker-failure.json").read_bytes())
    assert not any(failure["body_entry_counts"].values())
    assert state["calls"] == 0


def test_science_cannot_include_technical_dispatches(fake_worker):
    frozen, _, core, _ = fake_worker
    original = core.run_acquisition

    def mixed(*a, **kw):
        value = original(*a, **kw)
        value["summary"]["total_entries"] += 2
        return value

    core.run_acquisition = mixed
    with pytest.raises(ValueError, match="body_count_match"):
        m.run_worker(frozen)


@pytest.mark.parametrize("tamper", ["stage", "technical_flag", "technical_bytes"])
def test_terminal_separates_technical_provenance(fake_worker, monkeypatch, tamper):
    frozen, _, _, real_helper = fake_worker
    m.run_worker(frozen)
    monkeypatch.setattr(m, "helper", real_helper)
    root = Path(frozen["output_root"]) / "worker"
    if tamper == "stage":
        path = root / "entries/causal_lm_entries_0001.json"
        value = m.read_json(path)
        value["stage"] = "science"
        fresh(path, value)
    elif tamper == "technical_bytes":
        (root / "technical/extra").write_bytes(b"invented unbound")
    else:
        terminal = m.read_json(root / "worker-terminal.json")
        technical = m.read_json(root / "technical/terminal.json")
        result = m.read_json(root / "technical/result.json")
        result["qualified"] = 1  # bool/int equality must not qualify a technical check
        fresh(root / "technical/result.json", result)
        technical["result_sha256"] = m.digest(root / "technical/result.json")
        technical["artifact_sha256"]["result.json"] = technical["result_sha256"]
        fresh(root / "technical/terminal.json", technical)
        terminal["technical_terminal_sha256"] = m.digest(root / "technical/terminal.json")
        fresh(root / "worker-terminal.json", terminal)
    with pytest.raises(ValueError):
        m.validate_worker_terminal(root, m.object_hash(frozen))


@pytest.fixture
def fake_loader(monkeypatch):
    import hashlib

    loader = module(PROJECT / "scripts/load_bf16_reference.py")
    calls = {
        "config": [],
        "model": [],
        "tokenizer": [],
        "threads": [],
        "seed": [],
        "deterministic": [],
    }
    bf16, fp32 = object(), object()
    tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype=bf16)
    buffer = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype=fp32)

    class Model:
        def __init__(self):
            self.training = True
            self.config = SimpleNamespace(_attn_implementation="sdpa", vocab_size=20)
            self.generation_config = SimpleNamespace(eos_token_id=[9])

        def eval(self):
            self.training = False
            return self

        def parameters(self):
            return iter([tensor])

        def buffers(self):
            return iter([buffer])

        def modules(self):
            return iter([self])

        def to(self, *a, **kw):
            pytest.fail("loaded model must not be cast")

        def float(self):
            pytest.fail("FP32 conversion forbidden")

    model = Model()
    original = SimpleNamespace(quantization_config=None)
    tokenizer = SimpleNamespace(get_chat_template=lambda: "invented template")

    def config_load(*a, **kw):
        calls["config"].append((a, kw))
        return original

    def model_load(*a, **kw):
        calls["model"].append((a, kw))
        return model

    def tokenizer_load(*a, **kw):
        calls["tokenizer"].append((a, kw))
        return tokenizer

    fake_torch = SimpleNamespace(
        bfloat16=bf16,
        float32=fp32,
        cuda=SimpleNamespace(is_initialized=lambda: False),
        is_autocast_enabled=lambda *a: False,
        set_num_threads=lambda n: calls["threads"].append(n),
        random=SimpleNamespace(
            default_generator=SimpleNamespace(manual_seed=lambda n: calls["seed"].append(n))
        ),
        use_deterministic_algorithms=lambda yes: calls["deterministic"].append(yes),
    )
    fake_transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=config_load),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_load),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "" if key == "CUDA_VISIBLE_DEVICES" else "1")
    config = {
        "runtime_versions": {"invented": "1"},
        "seed": 7,
        "model_path": "/invented/checkpoint",
        "chat_template_sha256": hashlib.sha256(b"invented template").hexdigest(),
    }
    engine = SimpleNamespace(runtime_versions=lambda: {"invented": "1"})
    return loader, config, engine, calls, model, tensor, buffer, original, fake_torch


def test_loader_direct_bf16_single_load_no_cast_and_fp32_buffer(fake_loader):
    loader, config, engine, calls, model, _, buffer, _, torch = fake_loader
    loaded = loader.load_cpu_bf16(config, engine)
    assert loaded.model is model and loaded.eos_ids == [9]
    assert len(calls["model"]) == len(calls["tokenizer"]) == len(calls["config"]) == 1
    assert calls["model"][0][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "dtype": torch.bfloat16,
        "device_map": {"": "cpu"},
        "attn_implementation": "sdpa",
    }
    assert buffer.dtype is torch.float32  # accurate buffer retention, not a BF16 blanket claim
    assert calls["threads"] == [4] and calls["deterministic"] == [True] and calls["seed"] == [7]


@pytest.mark.parametrize(
    "defect",
    [
        "runtime",
        "autocast",
        "quantized_config",
        "fp32_parameter",
        "gpu_buffer",
        "bool_eos",
        "template",
        "quantized_model",
    ],
)
def test_loader_rejects_unsupported_without_fallback(fake_loader, defect):
    loader, config, engine, calls, model, tensor, buffer, original, torch = fake_loader
    if defect == "runtime":
        engine.runtime_versions = lambda: {"invented": "2"}
    elif defect == "autocast":
        torch.is_autocast_enabled = lambda *a: True
    elif defect == "quantized_config":
        original.quantization_config = {}
    elif defect == "fp32_parameter":
        tensor.dtype = torch.float32
    elif defect == "gpu_buffer":
        buffer.device.type = "cuda"
    elif defect == "bool_eos":
        model.generation_config.eos_token_id = True
    elif defect == "template":
        config["chat_template_sha256"] = "a" * 64
    else:
        model.is_loaded_in_4bit = True
    with pytest.raises(ValueError):
        loader.load_cpu_bf16(config, engine)
    assert len(calls["model"]) <= 1 and len(calls["tokenizer"]) <= 1
