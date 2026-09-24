"""A195 outer qualification: stdlib/fake workers only, no native imports."""

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "a195_outer_test", ROOT / "scripts/run_canonical_answer_ranking.py"
)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def config():
    return {
        "schema_version": "a195-cpu-fp32-config-v1",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "model_path": "/invented/assets",
        "model_revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "model_files_sha256": {
            name: "a" * 64
            for name in [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "weights1",
                "weights2",
                "index",
                "license",
                "policy",
            ]
        },
        "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
        "runtime_versions": {"invented": "1"},
        "seed": 1729,
        "execution": copy.deepcopy(r.EXECUTION),
    }


class Decoder:
    def forward(self):
        return None


class LM:
    def __init__(self):
        self.model = Decoder()

    def forward(self):
        return self.model.forward()


def classes():
    return ((LM, "causal_lm_entries"), (Decoder, "nested_decoder_entries"))


def raw_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.canonical(value))
    return r.digest(path)


class Lifecycle:
    def __init__(self):
        self.killed = []
        self.alive = []

    def hash_bound_json(self, path, sha, **kw):
        assert r.digest(path) == sha
        return json.loads(Path(path).read_bytes())

    def memory_available(self):
        return 100 * 1024**3

    def asset_check(self, config):
        return {"invented_asset": {"sha256": "a" * 64, "bytes": 10}}

    def process_session_snapshot(self, pid):
        return self.alive

    def terminate_owned_session(self, process, grace):
        self.killed.append(process.pid)
        self.alive = []
        process.code = 1

    def owned_interruptions(self):
        return contextlib.nullcontext()

    def open_pidfd(self, pid):
        return os.open(os.devnull, os.O_RDONLY)

    def signal_pidfd(self, descriptor, signum):
        assert signum == 0


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    frozen = {
        "output_root": str(tmp_path / "run-001"),
        "prepared_root": str(tmp_path / "prepared"),
        "source_root": str(tmp_path / "source"),
        "target_config_path": str(tmp_path / "config.json"),
        "plan": {"eos_token_ids": [7]},
        "runtime_binding": {"invented": True},
    }
    Path(frozen["output_root"]).mkdir()
    cfg = config()
    frozen["target_config_sha256"] = raw_json(Path(frozen["target_config_path"]), cfg)
    # Simulate the already-created worker session, without any OS process launch.
    monkeypatch.setattr(r.os, "getpid", lambda: 1001)
    monkeypatch.setattr(r.os, "getppid", lambda: 1000)
    monkeypatch.setattr(r.os, "getsid", lambda pid: 1001)
    raw_json(
        Path(frozen["output_root"]) / "launch.json",
        {
            "freeze_sha256": r.object_hash(frozen),
            "limits": r.LIMITS,
            "supervisor": {"pid": 1000, "parent_pid": 999, "session": 1000, "start_ticks": 2000},
        },
    )
    raw_json(
        Path(frozen["output_root"]) / "worker-launch.json",
        {
            "freeze_sha256": r.object_hash(frozen),
            "worker": {"pid": 1001, "parent_pid": 1000, "session": 1001, "start_ticks": 2010},
        },
    )
    lifecycle = Lifecycle()
    seen = {"model_refs": [], "stages": [], "freeze_paths": [], "replays": 0}
    acquired = {
        "status": "finished",
        "summary": {
            "scoring_entries": 64,
            "total_entries": 64,
            "entries_complete": True,
            "planned_measurements": 64,
            "completed_measurements": 64,
            "failed_measurements": 0,
            "unattempted_measurements": 0,
        },
        "analysis": {"invented": True},
        "terminal_sha256": r.object_hash({"invented": True}),
    }

    def acquire(model, tokenizer, *, plan, prepared, runtime_binding, directory, event):
        Path(directory).mkdir()
        r.publish(Path(directory) / "terminal.json", {"invented": True})
        event("science")
        seen["stages"].append("science")
        for _ in range(64):
            model.forward()
        return copy.deepcopy(acquired)

    def replay(directory, plan):
        seen["replays"] += 1
        return copy.deepcopy(acquired)

    core = NS(
        validate_plan=lambda p: p,
        prepare_inputs=lambda p, t: {"invented": True},
        run_acquisition=acquire,
        load_acquisition=replay,
    )
    engine = NS(runtime_versions=lambda: cfg["runtime_versions"])

    def load(*args):
        model = LM()
        seen["model_refs"].append(weakref.ref(model))
        return NS(model=model, tokenizer=object(), eos_ids=[7])

    monkeypatch.setattr(r, "helper", lambda: lifecycle)
    monkeypatch.setattr(
        r, "native_dependencies", lambda f: (core, engine, object(), None, None, None, LM, Decoder)
    )
    monkeypatch.setattr(r, "load_cpu_fp32", load)
    monkeypatch.setattr(r, "model_metadata", lambda m, t: {"invented": "same"})
    monkeypatch.setattr(r, "validate_target", lambda *args: None)
    monkeypatch.setattr(r, "imported_sources", lambda f: None)
    monkeypatch.setattr(r, "verify_freeze", lambda path, sha: seen["freeze_paths"].append(path))
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: NS(free=100 * 1024**3))
    for k, v in r.ENV.items():
        monkeypatch.setenv(k, v)
    return frozen, core, lifecycle, seen, acquired


def test_complete_fake_worker_and_terminal_replay(worker_case):
    frozen, core, lifecycle, seen, _ = worker_case
    prior = sys.getprofile()
    terminal = r.run_worker(frozen)
    assert sys.getprofile() is prior
    assert terminal["body_entry_counts"] == {"causal_lm_entries": 64, "nested_decoder_entries": 64}
    assert terminal["body_stage_counts"]["science"] == {
        "causal_lm_entries": 64,
        "nested_decoder_entries": 64,
    }
    assert seen["stages"] == ["science"] and seen["replays"] == 1
    assert seen["freeze_paths"][0].name == "freeze.json"
    assert seen["model_refs"][0]() is None
    root = Path(frozen["output_root"]) / "worker"
    assert r.validate_worker_terminal(root, r.object_hash(frozen)) == r.digest(
        root / "worker-terminal.json"
    )
    with pytest.raises(FileExistsError):
        r.run_worker(frozen)


def test_failure_retains_actual_partial_body_entries(worker_case):
    frozen, core, _, _, _ = worker_case

    class Interrupted(BaseException):
        pass

    def fail(model, tokenizer, **kw):
        kw["event"]("science")
        model.forward()
        raise Interrupted("private failure text")

    core.run_acquisition = fail
    with pytest.raises(Interrupted):
        r.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    failure = r.read_json(root / "worker-failure.json")
    assert failure["body_entry_counts"] == {"causal_lm_entries": 1, "nested_decoder_entries": 1}
    assert failure["phase"] == "acquisition" and failure["error_type"] == "Interrupted"
    assert "private failure" not in (root / "worker-failure.json").read_text()
    assert not (root / "worker-terminal.json").exists()


def test_forward_during_load_is_recorded_and_rejected(worker_case, monkeypatch):
    frozen, _, _, _, _ = worker_case

    def wrong_load(*args):
        model = LM()
        model.forward()

    monkeypatch.setattr(r, "load_cpu_fp32", wrong_load)
    with pytest.raises(ValueError, match="body_ceiling"):
        r.run_worker(frozen)
    f = r.read_json(Path(frozen["output_root"]) / "worker/worker-failure.json")
    assert f["body_stage_counts"]["nonstudy"]["causal_lm_entries"] == 1
    assert f["body_entry_counts"]["nested_decoder_entries"] == 0


@pytest.mark.parametrize("change", ["replay", "counts", "assets"])
def test_post_acquisition_failure_never_publishes_terminal(worker_case, change):
    frozen, core, lifecycle, _, acquired = worker_case
    if change == "replay":
        core.load_acquisition = lambda *args: {"different": True}
    elif change == "counts":
        acquired["summary"]["scoring_entries"] = True
    else:
        calls = []

        def asset(cfg):
            calls.append(1)
            return {"call": len(calls)}

        lifecycle.asset_check = asset
    with pytest.raises(ValueError):
        r.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    assert (root / "acquisition-normal-return.json").exists()
    assert (root / "worker-failure.json").exists() and not (root / "worker-terminal.json").exists()


@pytest.mark.parametrize(
    "mutation", ["bool_count", "wrong_stage", "missing_entry", "private_extra", "failure_marker"]
)
def test_completed_terminal_tamper_rejected(worker_case, mutation):
    frozen, *_ = worker_case
    r.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    terminal = r.read_json(root / "worker-terminal.json")
    if mutation == "bool_count":
        terminal["body_entry_counts"]["causal_lm_entries"] = True
    elif mutation == "wrong_stage":
        terminal["body_stage_counts"]["science"]["causal_lm_entries"] = 63
    elif mutation == "missing_entry":
        (root / "entries/causal_lm_entries_0001.json").unlink()
    elif mutation == "private_extra":
        terminal["raw_response"] = "invented"
    else:
        raw_json(root / "worker-failure.json", {"failed": True})
    if mutation in ("missing_entry", "failure_marker"):
        actual = r.inventory(root)
        actual.pop("worker-terminal.json")
        terminal["artifact_sha256"] = actual
    (root / "worker-terminal.json").write_bytes(r.canonical(terminal))
    with pytest.raises((ValueError, FileNotFoundError)):
        r.validate_worker_terminal(root, r.object_hash(frozen))


def test_profile_ceiling_stage_order_and_exception_counts(tmp_path):
    profile = r.BodyProfile(tmp_path / "entries", classes())
    model = LM()
    with profile:
        profile.set_stage("science")
        for _ in range(64):
            model.forward()
        with pytest.raises(ValueError, match="body_ceiling"):
            model.forward()
    assert profile.counts["causal_lm_entries"] == 65
    assert profile.counts["nested_decoder_entries"] == 64
    assert (tmp_path / "entries/causal_lm_entries_0065.json").exists()
    with pytest.raises(ValueError, match="stage_order"):
        profile.set_stage("science")


@pytest.mark.parametrize("stage", ["technical", "generation", "nonstudy", None])
def test_no_separate_technical_or_generation_stage(tmp_path, stage):
    profile = r.BodyProfile(tmp_path / "entries", classes())
    with pytest.raises(ValueError, match="stage_order"):
        profile.set_stage(stage)
    assert profile.counts == {"causal_lm_entries": 0, "nested_decoder_entries": 0}


def test_profile_original_exception_survives_restoration_failure(tmp_path, monkeypatch):
    profile = r.BodyProfile(tmp_path / "entries", classes())
    original = KeyboardInterrupt()
    monkeypatch.setattr(
        r.sys, "setprofile", lambda value: (_ for _ in ()).throw(RuntimeError("cleanup"))
    )
    assert profile.__exit__(KeyboardInterrupt, original, None) is False
    assert profile.cleanup_errors == ["RuntimeError"]
    with pytest.raises(RuntimeError):
        profile.__exit__(None, None, None)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, n):
        self.now += n


def test_queue_readiness_and_expiry_are_bounded(tmp_path):
    life = Lifecycle()
    clock = Clock()
    limits = {**r.LIMITS, "queue_wall_seconds": 120, "queue_poll_seconds": 60}
    with r.cooperative_lock(tmp_path / "lock") as fd:
        ready = tmp_path / "ready"
        ready.mkdir()
        assert r.wait_for_resources(
            ready,
            fd,
            life,
            limits=limits,
            clock=clock,
            sleep=clock.sleep,
            free_bytes=lambda: 100 * 1024**3,
        )
        assert r.read_json(ready / "queue/check_0000.json")["ready"] is True
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
        expired = tmp_path / "expired"
        expired.mkdir()
        life.memory_available = lambda: 0
        assert not r.wait_for_resources(
            expired,
            fd,
            life,
            limits=limits,
            clock=clock,
            sleep=clock.sleep,
            free_bytes=lambda: 100 * 1024**3,
        )
        assert clock.now == 120 and len(list((expired / "queue").iterdir())) == 2
        with pytest.raises(FileExistsError):
            r.wait_for_resources(expired, fd, life)


def test_queue_rejects_boolean_resource_sample(tmp_path):
    life = Lifecycle()
    life.memory_available = lambda: True
    with r.cooperative_lock(tmp_path / "lock") as fd:
        with pytest.raises(ValueError, match="resource_sample"):
            r.wait_for_resources(tmp_path, fd, life, free_bytes=lambda: 100 * 1024**3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads", 4),
        ("threads", True),
        ("dtype", "bfloat16"),
        ("cache", True),
        ("attention", "sdpa_flash"),
    ],
)
def test_config_exact_policy(field, value):
    c = config()
    r.validate_config(c)
    c["execution"][field] = value
    with pytest.raises(ValueError):
        r.validate_config(c)


def test_environment_forces_offline_cpu_eight_threads(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONHOME", "/untrusted")
    env = r.environment({"source_root": str(tmp_path)})
    assert all(env[k] == v for k, v in r.ENV.items()) and "PYTHONHOME" not in env
    assert env["PYTHONPATH"] == str(tmp_path / "src")


def test_publish_is_exclusive_and_orphan_nonreusable(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    r.publish(path, {"a": 1})
    with pytest.raises(FileExistsError):
        r.publish(path, {"a": 2})
    assert json.loads(path.read_bytes()) == {"a": 1}
    assert (tmp_path / ".receipt.json.tmp").exists()
    path2 = tmp_path / "broken.json"
    monkeypatch.setattr(r.os, "link", lambda *args: (_ for _ in ()).throw(OSError("link")))
    with pytest.raises(OSError):
        r.publish(path2, {"b": 1})
    assert not path2.exists() and (tmp_path / ".broken.json.tmp").exists()


def test_regular_and_inventory_reject_links_and_specials(tmp_path):
    file = tmp_path / "x"
    file.write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(file)
    with pytest.raises(ValueError):
        r.digest(link)
    with pytest.raises(ValueError):
        r.inventory(tmp_path)
    link.unlink()
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError):
        r.inventory(tmp_path)


def test_strict_json_rejects_boolean_float_confusion_and_duplicates(tmp_path):
    path = tmp_path / "bad.json"
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e9999}'):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            r.read_json(path)
    assert r.canonical({"n": True}) != r.canonical({"n": 1}) != r.canonical({"n": 1.0})


def test_helper_import_is_inert_and_hash_bound(tmp_path):
    p = tmp_path / "module.py"
    p.write_text('raise AssertionError("must not execute")\n')
    with pytest.raises(ValueError, match="helper_hash"):
        r.module(p, "test", "0" * 64)
    p.write_text("value = 17\n")
    assert r.module(p, "test", r.digest(p)).value == 17


@pytest.fixture
def frozen_case(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    source = prepared / "source"
    source.mkdir(parents=True)
    runner = source / "scripts/run_canonical_answer_ranking.py"
    runner.parent.mkdir()
    runner.write_text("# invented\n")
    file = prepared / "runtime.json"
    raw_json(file, config())
    selection = raw_json(prepared / "selection.json", {"selected": True})
    qualification = raw_json(prepared / "qualification.json", {"qualified": True})
    instrument_qualification = raw_json(
        prepared / "instrument.json", {"invented_instrument_qualified": True}
    )
    monkeypatch.setattr(r, "INSTRUMENT_QUALIFICATION_SHA256", instrument_qualification)
    relative = "scripts/run_canonical_answer_ranking.py"
    monkeypatch.setattr(r, "SOURCE_PATHS", (relative,))
    monkeypatch.setattr(r, "PINS", {})
    monkeypatch.setattr(r, "__file__", str(runner))
    # Avoid importing a helper from this wholly invented source closure.
    monkeypatch.setattr(r, "helper", lambda: Lifecycle())
    frozen = {
        "schema_version": "a195-native-freeze-v1",
        "prepared_root": str(prepared),
        "source_root": str(source),
        "output_root": str(tmp_path / "run-001"),
        "sources": {relative: r.digest(runner)},
        "native_python": r.NATIVE_PYTHON,
        "limits": copy.deepcopy(r.LIMITS),
        "target_config_path": str(file),
        "target_config_sha256": r.digest(file),
        "runtime_binding": {},
        "plan": {},
        "public_commit": "a" * 40,
        "selection_sha256": selection,
        "qualification_sha256": qualification,
        "instrument_qualification_sha256": instrument_qualification,
        "files_sha256": r.inventory(prepared),
    }
    path = prepared / "freeze.json"
    sha = raw_json(path, frozen)
    return frozen, path, sha


def test_frozen_closure_and_duplicate_extra_files(frozen_case):
    f, path, sha = frozen_case
    assert r.verify_freeze(path, sha, enforce_executable=False) == f
    (path.parent / "unlisted").write_text("x")
    with pytest.raises(ValueError, match="freeze_inventory"):
        r.verify_freeze(path, sha, enforce_executable=False)


@pytest.mark.parametrize(
    "key,value",
    [
        ("limits", {}),
        ("sources", {}),
        ("public_commit", True),
        ("native_python", "python"),
        ("output_root", "/tmp/alternate"),
    ],
)
def test_freeze_semantic_tamper(frozen_case, key, value):
    f, path, _ = frozen_case
    f[key] = value
    sha = raw_json(path, f)
    with pytest.raises(ValueError):
        r.verify_freeze(path, sha, enforce_executable=False)


def test_supervisor_expiry_claim_blocks_retry(frozen_case, monkeypatch):
    frozen, path, sha = frozen_case
    life = Lifecycle()
    monkeypatch.setattr(r, "verify_freeze", lambda *args: frozen)
    monkeypatch.setattr(r, "helper", lambda: life)
    monkeypatch.setattr(r, "wait_for_resources", lambda *args: False)
    result = r.supervise(path, sha)
    assert result["status"] == "deferred" and result["target_body_entries"] == 0
    with pytest.raises(FileExistsError):
        r.supervise(path, sha)


@pytest.mark.parametrize("monitor_fails", [False, True])
def test_supervisor_owns_cleanup_and_does_not_retry(frozen_case, monkeypatch, monitor_fails):
    frozen, path, sha = frozen_case
    life = Lifecycle()

    class Process:
        pid = 999991
        code = None

        def poll(self):
            return self.code

    process = Process()
    life.alive = [{"pid": process.pid}]
    monkeypatch.setattr(r, "verify_freeze", lambda *args: frozen)
    monkeypatch.setattr(r, "helper", lambda: life)
    monkeypatch.setattr(r, "wait_for_resources", lambda *args: True)
    monkeypatch.setattr(r.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        r,
        "identity",
        lambda pid: {"pid": pid, "parent_pid": os.getpid(), "session": pid, "start_ticks": 123},
    )

    def monitor(*args, **kwargs):
        if monitor_fails:
            raise KeyboardInterrupt("original")
        process.code = 1
        life.alive = []
        return {
            "exit_code": 1,
            "stop_reason": "host_memory_pressure",
            "owned_live_processes_remaining": 0,
        }

    monkeypatch.setattr(r, "monitor_module", lambda: NS(monitor=monitor))
    if monitor_fails:
        with pytest.raises(KeyboardInterrupt):
            r.supervise(path, sha)
        assert life.killed == [process.pid]
    else:
        result = r.supervise(path, sha)
        assert result["status"] == "failed" and result["worker_terminal_sha256"] is None
    assert (Path(frozen["output_root"]) / "worker-launch.json").exists()


def test_loader_direct_cpu_fp32_and_eight_threads(monkeypatch):
    calls = []
    torch = NS(
        cuda=NS(is_initialized=lambda: False),
        float32="float32",
        set_num_threads=lambda n: calls.append(("threads", n)),
        set_num_interop_threads=lambda n: calls.append(("interop", n)),
        random=NS(default_generator=NS(manual_seed=lambda n: calls.append(("seed", n)))),
        use_deterministic_algorithms=lambda *a, **k: calls.append(("deterministic", a, k)),
        set_float32_matmul_precision=lambda n: calls.append(("precision", n)),
        backends=NS(
            cuda=NS(
                **{
                    name: (lambda v, n=name: calls.append((n, v)))
                    for name in (
                        "enable_flash_sdp",
                        "enable_mem_efficient_sdp",
                        "enable_cudnn_sdp",
                        "enable_math_sdp",
                    )
                }
            )
        ),
    )
    model = NS(generation_config=NS(eos_token_id=[3, 2, 3]))
    model.eval = lambda: model

    def load(*args, **kw):
        calls.append(("load", kw))
        return model

    ac = NS(from_pretrained=lambda *a, **k: NS(quantization_config=None))
    am = NS(from_pretrained=load)
    at = NS(from_pretrained=lambda *a, **k: NS(get_chat_template=lambda: "template"))
    monkeypatch.setattr(r, "model_metadata", lambda *a: {})
    for k, v in r.ENV.items():
        monkeypatch.setenv(k, v)
    result = r.load_cpu_fp32(config(), torch, ac, am, at)
    assert result.eos_ids == [2, 3]
    assert ("threads", 8) in calls and ("interop", 1) in calls
    kw = next(x[1] for x in calls if x[0] == "load")
    assert kw["dtype"] == "float32" and kw["device_map"] == {"": "cpu"}
    assert (
        kw["local_files_only"]
        and not kw["trust_remote_code"]
        and kw["attn_implementation"] == "sdpa"
    )
    assert ("enable_math_sdp", True) in calls and ("enable_flash_sdp", False) in calls


def metadata_fixture():
    tensor = NS(
        device=NS(type="cpu"),
        layout="strided",
        dtype="float32",
        shape=(2,),
        is_complex=lambda: False,
        is_floating_point=lambda: True,
    )
    cfg = NS(
        vocab_size=128256,
        hidden_size=3072,
        num_hidden_layers=28,
        max_position_embeddings=131072,
        _attn_implementation="sdpa",
        output_hidden_states=False,
        output_attentions=False,
    )
    model = NS(config=cfg, training=False)
    model.modules = lambda: [model]
    model.named_parameters = lambda: [("weight", tensor)]
    model.named_buffers = lambda: [("buffer", tensor)]
    torch = NS(
        cuda=NS(is_initialized=lambda: False),
        is_autocast_enabled=lambda device: False,
        get_num_threads=lambda: 8,
        get_num_interop_threads=lambda: 1,
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
        strided="strided",
        float32="float32",
        backends=NS(
            cuda=NS(
                math_sdp_enabled=lambda: True,
                flash_sdp_enabled=lambda: False,
                mem_efficient_sdp_enabled=lambda: False,
                cudnn_sdp_enabled=lambda: False,
            )
        ),
    )
    return model, torch, tensor


def test_actual_tensor_metadata_retained_without_weight_authentication_claim():
    model, torch, _ = metadata_fixture()
    meta = r.model_metadata(model, torch)
    assert len(meta["tensors"]) == 2 and meta["execution"]["threads"] == 8
    assert meta["weight_content_authenticated_by_this_table"] is False
    assert meta["tensor_metadata_sha256"] == r.object_hash(meta["tensors"])


@pytest.mark.parametrize(
    "violation",
    [
        "fp64_buffer",
        "cuda_tensor",
        "threads",
        "interop",
        "training",
        "autocast",
        "flash",
        "geometry",
        "precision",
    ],
)
def test_model_preflight_rejects_execution_drift(violation):
    model, torch, tensor = metadata_fixture()
    if violation == "fp64_buffer":
        tensor.dtype = "float64"
    elif violation == "cuda_tensor":
        tensor.device.type = "cuda"
    elif violation == "threads":
        torch.get_num_threads = lambda: 4
    elif violation == "interop":
        torch.get_num_interop_threads = lambda: 2
    elif violation == "training":
        model.training = True
    elif violation == "autocast":
        torch.is_autocast_enabled = lambda device: True
    elif violation == "flash":
        torch.backends.cuda.flash_sdp_enabled = lambda: True
    elif violation == "geometry":
        model.config.hidden_size = 4096
    else:
        torch.get_float32_matmul_precision = lambda: "medium"
    with pytest.raises(ValueError):
        r.model_metadata(model, torch)


def test_asset_geometry_and_eos_are_bound_without_loading():
    cfg = config()
    binding = r.expected_runtime_binding(cfg, "a" * 64)
    plan = {
        "geometry": copy.deepcopy(r.GEOMETRY),
        "runtime_binding_sha256": r.object_hash(binding),
        "eos_token_ids": [2, 3],
        "provenance": {
            "model_sha256": r.object_hash(cfg["model_files_sha256"]),
            "tokenizer_sha256": r.object_hash(
                {
                    k: v
                    for k, v in cfg["model_files_sha256"].items()
                    if k in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
                }
            ),
            "chat_template_sha256": cfg["chat_template_sha256"],
        },
    }

    def meta(path, sha, **kwargs):
        assert kwargs == {"allow_snapshot_link": True}
        return (
            {
                "vocab_size": 128256,
                "hidden_size": 3072,
                "num_hidden_layers": 28,
                "max_position_embeddings": 131072,
            }
            if Path(path).name == "config.json"
            else {"eos_token_id": [3, 2, 3]}
        )

    life = NS(hash_bound_json=meta)
    r.validate_target(cfg, plan, binding, "a" * 64, life)
    plan["geometry"]["hidden_width"] = 4096
    with pytest.raises(ValueError, match="geometry"):
        r.validate_target(cfg, plan, binding, "a" * 64, life)


def test_worker_parent_binding_precedes_work(frozen_case, monkeypatch):
    frozen, path, sha = frozen_case
    root = Path(frozen["output_root"])
    root.mkdir()
    parent = 71
    identity = {"pid": parent, "parent_pid": 1, "session": parent, "start_ticks": 12}
    raw_json(
        root / "launch.json", {"freeze_sha256": sha, "supervisor": identity, "limits": r.LIMITS}
    )
    calls = []
    life = Lifecycle()
    life.bind_parent_lifetime = lambda p: calls.append(("parent", p))
    monkeypatch.setattr(r, "helper", lambda: life)
    monkeypatch.setattr(r, "verify_freeze", lambda *args: frozen)
    monkeypatch.setattr(r, "identity", lambda p: identity)
    monkeypatch.setattr(r.os, "getppid", lambda: parent)
    monkeypatch.setattr(r.os, "getsid", lambda p: 81)
    monkeypatch.setattr(r.os, "getpid", lambda: 81)
    monkeypatch.setattr(r, "run_worker", lambda f: calls.append(("worker", True)))
    r.worker(path, sha, parent)
    assert calls == [("parent", parent), ("worker", True)]
    monkeypatch.setattr(r.os, "getppid", lambda: 999)
    with pytest.raises(ValueError, match="parent_lineage"):
        r.worker(path, sha, parent)
    assert calls.count(("worker", True)) == 1


@pytest.mark.parametrize("probe_failure", ["open", "signal"])
def test_cleanup_capability_failure_prevents_worker_launch(frozen_case, monkeypatch, probe_failure):
    frozen, path, sha = frozen_case
    life = Lifecycle()
    fds = []
    spawns = []
    monkeypatch.setattr(r, "verify_freeze", lambda *args: frozen)
    monkeypatch.setattr(r, "helper", lambda: life)
    monkeypatch.setattr(r, "wait_for_resources", lambda *args: True)
    monkeypatch.setattr(r.subprocess, "Popen", lambda *a, **k: spawns.append(1))

    def opened(pid):
        if probe_failure == "open":
            raise OSError("unavailable")
        fd = os.open(os.devnull, os.O_RDONLY)
        fds.append(fd)
        return fd

    life.open_pidfd = opened
    life.signal_pidfd = lambda *args: (_ for _ in ()).throw(OSError("unavailable"))
    with pytest.raises(OSError):
        r.supervise(path, sha)
    assert spawns == []
    assert (Path(frozen["output_root"]) / "supervisor-failure.json").exists()
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize(
    "tamper",
    [
        "consistent_pid",
        "header_parent",
        "header_session",
        "worker_birth_zero",
        "worker_birth_bool",
        "supervisor_birth_zero",
        "supervisor_pid",
        "worker_freeze",
        "launch_limits",
    ],
)
def test_terminal_crossbinds_retained_launch_identities(worker_case, tamper):
    frozen, *_ = worker_case
    r.run_worker(frozen)
    root = Path(frozen["output_root"]) / "worker"
    terminal = r.read_json(root / "worker-terminal.json")
    header = r.read_json(root / "run.json")
    child = r.read_json(root.parent / "worker-launch.json")
    launch = r.read_json(root.parent / "launch.json")
    if tamper == "consistent_pid":
        header["pid"] = 2001
        header["session"] = 2001
        # A consistent substituted PID in every worker-owned body receipt used
        # to pass; the independently retained launch identity must now reject it.
        for path in (root / "entries").iterdir():
            entry = r.read_json(path)
            entry["pid"] = 2001
            path.write_bytes(r.canonical(entry))
    elif tamper == "header_parent":
        header["parent_pid"] = 9999
    elif tamper == "header_session":
        header["session"] = 9999
    elif tamper == "worker_birth_zero":
        child["worker"]["start_ticks"] = 0
    elif tamper == "worker_birth_bool":
        child["worker"]["start_ticks"] = True
    elif tamper == "supervisor_birth_zero":
        launch["supervisor"]["start_ticks"] = 0
    elif tamper == "supervisor_pid":
        launch["supervisor"]["pid"] = 9999
    elif tamper == "worker_freeze":
        child["freeze_sha256"] = "0" * 64
    else:
        launch["limits"]["wall_seconds"] = 1
    (root / "run.json").write_bytes(r.canonical(header))
    (root.parent / "worker-launch.json").write_bytes(r.canonical(child))
    (root.parent / "launch.json").write_bytes(r.canonical(launch))
    inventory = r.inventory(root)
    inventory.pop("worker-terminal.json")
    terminal["artifact_sha256"] = inventory
    (root / "worker-terminal.json").write_bytes(r.canonical(terminal))
    with pytest.raises(
        ValueError, match="terminal_(worker_lineage|process_identity|launch_binding)"
    ):
        r.validate_worker_terminal(root, r.object_hash(frozen))


def preparation_adapter(
    core, *, stage="payload_encode", code="payload_special", original_type="ValueError"
):
    core.PREPARATION_STAGES = {
        "unclassified",
        "inventory_before",
        "payload_encode",
        "closed_template",
        "inventory_after",
    }
    core.PREPARATION_GUARD_CODES = {
        "operation_exception",
        "payload_special",
        "closed_render",
        "inventory_changed",
    }
    core.preparation_failure = lambda error: {
        "stage": stage,
        "code": code,
        "original_error_type": original_type,
    }


def test_native_preparation_failure_retains_finite_provenance_and_zero_bodies(worker_case):
    frozen, core, _, seen, _ = worker_case
    preparation_adapter(core)

    class PreparationError(ValueError):
        pass

    original = PreparationError("PRIVATE_NATIVE_PAYLOAD")
    core.prepare_inputs = lambda *args: (_ for _ in ()).throw(original)
    with pytest.raises(PreparationError) as caught:
        r.run_worker(frozen)
    assert caught.value is original
    root = Path(frozen["output_root"]) / "worker"
    failure = r.read_json(root / "worker-failure.json")
    assert failure["phase"] == "native_preparation"
    assert failure["error_type"] == "PreparationError"
    assert failure["preparation_failure"] == {
        "stage": "payload_encode",
        "code": "payload_special",
        "original_error_type": "ValueError",
    }
    assert failure["body_entry_counts"] == {"causal_lm_entries": 0, "nested_decoder_entries": 0}
    assert list((root / "entries").iterdir()) == []
    assert not (root / "acquisition").exists()
    assert "PRIVATE_NATIVE_PAYLOAD" not in (root / "worker-failure.json").read_text()
    assert seen["replays"] == 0


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_preparation_interruption_keeps_original_exception(worker_case, exception):
    frozen, core, *_ = worker_case
    preparation_adapter(
        core, stage="unclassified", code="operation_exception", original_type=exception.__name__
    )
    original = exception("PRIVATE_INTERRUPT")
    core.prepare_inputs = lambda *args: (_ for _ in ()).throw(original)
    with pytest.raises(exception) as caught:
        r.run_worker(frozen)
    assert caught.value is original
    failure = r.read_json(Path(frozen["output_root"]) / "worker/worker-failure.json")
    assert failure["preparation_failure"]["original_error_type"] == exception.__name__
    assert failure["preparation_failure"]["code"] == "operation_exception"
    assert failure["body_entry_counts"] == {"causal_lm_entries": 0, "nested_decoder_entries": 0}


@pytest.mark.parametrize("mutation", ["stage", "code", "type", "extra", "nonobject", "raises"])
def test_bad_preparation_metadata_never_leaks_or_replaces_original(worker_case, mutation):
    frozen, core, *_ = worker_case
    preparation_adapter(core)
    packet = core.preparation_failure(None)
    if mutation in ("stage", "code"):
        packet[mutation] = "PRIVATE_UNREGISTERED_MESSAGE"
    elif mutation == "type":
        packet["original_error_type"] = "PRIVATE message with spaces"
    elif mutation == "extra":
        packet["prompt"] = "PRIVATE_NATIVE_PROMPT"
    elif mutation == "nonobject":
        packet = ["PRIVATE_TEXT"]
    core.preparation_failure = lambda error: packet
    if mutation == "raises":
        core.preparation_failure = lambda error: (_ for _ in ()).throw(
            RuntimeError("PRIVATE_METADATA_ERROR")
        )
    original = ValueError("PRIVATE_ORIGINAL")
    core.prepare_inputs = lambda *args: (_ for _ in ()).throw(original)
    with pytest.raises(ValueError) as caught:
        r.run_worker(frozen)
    assert caught.value is original
    path = Path(frozen["output_root"]) / "worker/worker-failure.json"
    value = r.read_json(path)
    assert value["preparation_failure"] == {
        "stage": "unclassified",
        "code": "preparation_metadata_invalid",
        "original_error_type": "ValueError",
    }
    assert "PRIVATE" not in path.read_text()


def test_non_preparation_failure_does_not_invent_preparation_guard(worker_case):
    frozen, core, *_ = worker_case
    core.run_acquisition = lambda *a, **kw: (_ for _ in ()).throw(OSError("PRIVATE"))
    with pytest.raises(OSError):
        r.run_worker(frozen)
    failure = r.read_json(Path(frozen["output_root"]) / "worker/worker-failure.json")
    assert failure["phase"] == "acquisition" and failure["preparation_failure"] is None


def test_unqualified_instrument_artifact_rejected_even_when_copied(frozen_case):
    frozen, path, _ = frozen_case
    replacement = raw_json(path.parent / "other-instrument.json", {"qualified": "different"})
    frozen["files_sha256"]["other-instrument.json"] = replacement
    frozen["instrument_qualification_sha256"] = replacement
    sha = raw_json(path, frozen)
    with pytest.raises(ValueError, match="instrument_qualification"):
        r.verify_freeze(path, sha, enforce_executable=False)


def test_all_inherited_and_special_inventory_pins_match_actual_source():
    for relative, sha in r.PINS.items():
        assert r.digest(ROOT / relative) == sha
    assert (
        r.PINS["src/lexical_prompt_study/special_token_inventory.py"]
        == "75e9425bbde7fefa56798613c6ad17a643f077463af1fce8b4c2c40e716391f0"
    )
    assert all("answer_path_ranking" not in path for path in r.SOURCE_PATHS)


def test_public_runtime_keeps_exact_selected_cpu_policy_and_64_scope():
    value = json.loads(
        (ROOT / "plans/canonical_answer_ranking_a195_runtime.public.json").read_bytes()
    )
    r.validate_config(value)
    assert value["execution"]["planned_forwards"] == 64
    assert value["execution"]["candidate_token_cap"] == 16
    assert value["execution"]["threads"] == 8 and value["execution"]["interop_threads"] == 1
    assert len(value["model_files_sha256"]) == 10


def test_actual_new_core_preparation_metadata_api_without_native_calls(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(ROOT / "src"))
    core = importlib.import_module("lexical_prompt_study.canonical_answer_ranking_acquisition")
    error = core.PreparationError("prompt_construction", "native_prompt_ids", "ValueError")
    assert r.retained_preparation_failure(error, core) == {
        "stage": "prompt_construction",
        "code": "native_prompt_ids",
        "original_error_type": "ValueError",
    }
    interrupt = KeyboardInterrupt()
    assert r.retained_preparation_failure(interrupt, core) == {
        "stage": "unclassified",
        "code": "operation_exception",
        "original_error_type": "KeyboardInterrupt",
    }
