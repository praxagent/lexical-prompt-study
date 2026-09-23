"""One frozen local A185 process; private output only, no retry or downloads."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

PYTHON = "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python"
LIMITS = {
    "minimum_available_bytes": 64 * 1024**3,
    "maximum_owned_rss_bytes": 64 * 1024**3,
    "host_pressure_floor_bytes": 8 * 1024**3,
    "wall_seconds": 1800,
    "termination_grace_seconds": 60,
    "poll_seconds": 0.2,
}
LOADER_SHA = "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e"
ADAPTER_SHA = "5b7041eaa062d4cdd6297fab062af9c59c11159bc6c593c28d8af36cf25b4c69"
EVIDENCE_SHA = "302a2a788a937a633a5bdc32a273e38fa95e0472f76bb5b7ad150ae748b3fc14"
SOURCE_PATHS = (
    "scripts/run_native_prefix_qualification.py",
    "scripts/run_cpu_reference_audit.py",
    "src/lexical_prompt_study/__init__.py",
    "src/lexical_prompt_study/native_prefix_qualification.py",
    "src/lexical_prompt_study/prefix_only_capture.py",
    "src/lexical_prompt_study/landmark_evidence.py",
    "src/lexical_prompt_study/instruction_binding_runtime.py",
    "src/lexical_prompt_study/instruction_binding_tasks.py",
)


def pidfd_api():
    # The pinned conda Python was built without its optional pidfd wrappers.
    # Use the host libc's named APIs, never architecture-specific syscall IDs.
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "pidfd_open") and hasattr(libc, "pidfd_send_signal"), "libc_pidfd_api")
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int
    return libc


def open_pidfd(pid):
    descriptor = pidfd_api().pidfd_open(pid, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open")
    return descriptor


def signal_pidfd(descriptor, signum):
    if pidfd_api().pidfd_send_signal(descriptor, signum, None, 0) < 0:
        raise OSError(ctypes.get_errno(), "pidfd_send_signal")


@contextmanager
def owned_interruptions():
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in signals}
    try:
        for number in signals:
            signal.signal(number, interrupt)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def bind_parent_lifetime(expected_parent):
    """Stop the model worker even if its supervisor is killed without cleanup."""
    require(os.getppid() == expected_parent, "parent_before_death_binding")
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux PR_SET_PDEATHSIG. Use the libc API and an explicit unsigned-long arg.
    if (
        libc.prctl(
            ctypes.c_int(1),
            ctypes.c_ulong(signal.SIGTERM),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        != 0
    ):
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
    require(os.getppid() == expected_parent, "parent_after_death_binding")


def require(condition, reason):
    if not condition:
        raise ValueError("a185_execution_" + reason)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def hash_bound_json(path, expected_sha, *, allow_snapshot_link=False):
    """Parse authenticated vendor JSON without imposing our receipt encoding."""
    path = Path(path)
    require(allow_snapshot_link or not path.is_symlink(), "json_symlink")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), "json_regular_file")
    raw = path.read_bytes()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "json_changed",
    )
    require(hashlib.sha256(raw).hexdigest() == expected_sha, "json_hash")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "json_duplicate_key")
            result[key] = value
        return result

    def finite(number):
        value = float(number)
        require(math.isfinite(value), "json_nonfinite")
        return value

    value = json.loads(raw, object_pairs_hook=pairs, parse_float=finite, parse_constant=finite)
    require(type(value) is dict, "json_object")
    return value


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("memory_snapshot_unavailable")


def process_session_snapshot(group):
    """Sample only this newly created session; no foreign process command text."""
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
            if int(fields[3]) != group:
                continue
            if fields[0] in ("Z", "X"):
                continue  # Dead children await OS reaping; they cannot execute or hold RSS.
            require(path.stat().st_uid == os.getuid(), "foreign_uid_in_owned_session")
            rss = next(
                (
                    int(line.split()[1]) * 1024
                    for line in (path / "status").read_text().splitlines()
                    if line.startswith("VmRSS:")
                ),
                0,
            )
            result.append({"pid": int(path.name), "start_ticks": int(fields[19]), "rss": rss})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def _signal_session(session, signum):
    """PID fds plus birth-time checks prevent signalling a reused foreign PID."""
    for item in sorted(process_session_snapshot(session), key=lambda item: item["pid"] == session):
        descriptor = None
        try:
            descriptor = open_pidfd(item["pid"])
            stat = Path(f"/proc/{item['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
            if int(stat[19]) != item["start_ticks"] or int(stat[3]) != session:
                continue
            signal_pidfd(descriptor, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)


def terminate_owned_session(process, grace_seconds):
    """Signal the owned session, including children creating new process groups."""
    group = process.pid
    if process.poll() is not None and not process_session_snapshot(group):
        return
    _signal_session(group, signal.SIGTERM)
    until = time.monotonic() + grace_seconds
    while time.monotonic() < until:
        process.poll()
        if not process_session_snapshot(group):
            return
        time.sleep(min(0.2, max(0, until - time.monotonic())))
    if process_session_snapshot(group):
        _signal_session(group, signal.SIGKILL)
    process.wait(timeout=10)


def monitor_process(process, limits, started, *, available=memory_available):
    """Monitored stop thresholds, not an instantaneous kernel RSS allocation cap."""
    peak_rss = 0
    stop_reason = None
    try:
        while process.poll() is None:
            owned = process_session_snapshot(process.pid)
            rss = sum(item["rss"] for item in owned)
            peak_rss = max(peak_rss, rss)
            if time.monotonic() - started >= limits["wall_seconds"]:
                stop_reason = "wall_limit"
            elif rss > limits["maximum_owned_rss_bytes"]:
                stop_reason = "owned_rss_limit"
            elif available() < limits["host_pressure_floor_bytes"]:
                stop_reason = "host_memory_pressure"
            if stop_reason:
                break
            time.sleep(limits["poll_seconds"])
    finally:
        # Also runs when monitoring or its caller is interrupted or fails.
        original = sys.exception()
        try:
            if process.poll() is None or process_session_snapshot(process.pid):
                terminate_owned_session(process, limits["termination_grace_seconds"])
        except BaseException:
            if original is None:
                raise
    require(not process_session_snapshot(process.pid), "owned_process_survived")
    return {
        "exit_code": process.wait(),
        "stop_reason": stop_reason,
        "peak_sampled_owned_rss_bytes": peak_rss,
        "elapsed_seconds": time.monotonic() - started,
        "owned_live_processes_remaining": 0,
    }


def verify_freeze(path, expected_sha):
    require(path.is_absolute() and not path.is_symlink(), "freeze_path")
    require(digest(path) == expected_sha, "freeze_sha256")
    freeze = json.loads(path.read_bytes())
    require(
        set(freeze)
        == {
            "schema_version",
            "prepared_root",
            "output_root",
            "runtime_executable",
            "limits",
            "files_sha256",
            "plan_path",
            "legacy_config_path",
            "loader_path",
            "public_commit",
            "runtime_binding_path",
        },
        "freeze_schema",
    )
    require(freeze["schema_version"] == "a185-native-freeze-v1", "freeze_version")
    require(freeze["runtime_executable"] == sys.executable == PYTHON, "runtime_executable")
    require(freeze["limits"] == LIMITS, "limits")
    prepared = Path(freeze["prepared_root"])
    require(
        prepared.is_absolute() and prepared == path.parent and not prepared.is_symlink(),
        "prepared_root",
    )
    expected = freeze["files_sha256"]
    require(type(expected) is dict and bool(expected), "source_manifest")
    require(
        {"source/" + name for name in SOURCE_PATHS}.issubset(expected), "complete_source_closure"
    )
    inventory = set()
    for entry in prepared.rglob("*"):
        require(not entry.is_symlink(), "prepared_symlink")
        if entry.is_file():
            inventory.add(str(entry.relative_to(prepared)))
        else:
            require(entry.is_dir(), "prepared_special_file")
    require(inventory == set(expected) | {path.name}, "prepared_inventory")
    for name, sha in expected.items():
        relative = Path(name)
        require(
            type(name) is str and not relative.is_absolute() and ".." not in relative.parts,
            "relative_source_path",
        )
        target = prepared / relative
        require(target.is_file() and not target.is_symlink(), "source_file")
        require(
            all(not parent.is_symlink() for parent in target.parents if parent != prepared.parent),
            "source_parent_symlink",
        )
        require(digest(target) == sha, "source_hash")
    require(
        expected.get("source/src/lexical_prompt_study/prefix_only_capture.py") == ADAPTER_SHA,
        "adapter_pin",
    )
    require(
        expected.get("source/src/lexical_prompt_study/landmark_evidence.py") == EVIDENCE_SHA,
        "evidence_pin",
    )
    require(expected.get(freeze["loader_path"]) == LOADER_SHA, "loader_pin")
    require("source/scripts/run_native_prefix_qualification.py" in expected, "entrypoint_pin")
    require(
        str(Path(__file__).resolve())
        == str(prepared / "source/scripts/run_native_prefix_qualification.py"),
        "frozen_entrypoint",
    )
    output = Path(freeze["output_root"])
    require(
        output.is_absolute() and output.parent == prepared.parent and output.name == "run-001",
        "output_root",
    )
    for name in ("plan_path", "legacy_config_path", "loader_path", "runtime_binding_path"):
        require(freeze[name] in expected, "bound_input_path")
    return freeze


def frozen_imports(freeze):
    source = Path(freeze["prepared_root"]) / "source/src"
    sys.path.insert(0, str(source))
    from lexical_prompt_study import landmark_evidence as evidence

    require(
        Path(evidence.__file__).resolve() == source / "lexical_prompt_study/landmark_evidence.py",
        "actual_import_path",
    )

    return evidence


def expected_runtime_binding(config, config_sha):
    return {
        "schema_version": "a185-runtime-binding-v1",
        "legacy_config_sha256": config_sha,
        "cpu_loader_sha256": LOADER_SHA,
        "runtime_versions": config["runtime_versions"],
        "model_revision": config["model_revision"],
        "model_files_sha256": config["model_files_sha256"],
        "chat_template_sha256": config["chat_template_sha256"],
        "actual_requested_execution": {
            "device": "cpu",
            "dtype": "float32",
            "quantization": None,
            "attention": "sdpa",
            "threads": 4,
            "deterministic_algorithms": True,
            "seed": config["seed"],
            "sampled_token_cap": 16,
            "cache": False,
        },
        "note": "Legacy NF4/cap64 descriptors are retained unchanged as a loader input binding; they are not the selected execution settings.",
    }


def environment(freeze):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(freeze["prepared_root"]) / "source/src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONOPTIMIZE": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
    )
    return env


def supervise(path, expected_sha):
    freeze = verify_freeze(path, expected_sha)
    probe = open_pidfd(os.getpid())
    try:
        signal_pidfd(probe, 0)
    finally:
        os.close(probe)
    evidence = frozen_imports(freeze)
    parent = Path(freeze["output_root"]).parent
    # Cooperative study reservation; this does not reserve all host RAM.
    with owned_interruptions(), (parent / ".cpu-study.lock").open("a") as lock:
        os.chmod(parent / ".cpu-study.lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available = memory_available()
        require(available >= LIMITS["minimum_available_bytes"], "not_ready_memory")
        root = evidence._directory(Path(freeze["output_root"]), create=True)
        launch = {
            "freeze_sha256": expected_sha,
            "available_memory_bytes": available,
            "supervisor_pid": os.getpid(),
            "limits": LIMITS,
            "reservation": "exclusive_study_lock_not_host_memory_reservation",
        }
        evidence._publish(root / "launch.json", evidence.canonical(launch))
        process = None
        started = time.monotonic()
        try:
            with (root / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    [
                        PYTHON,
                        "-B",
                        str(Path(__file__)),
                        "worker",
                        "--freeze",
                        str(path),
                        "--freeze-sha256",
                        expected_sha,
                    ],
                    env=environment(freeze),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                outcome = monitor_process(process, LIMITS, started)
            outcome["freeze_sha256"] = expected_sha
            evidence._publish(root / "supervisor-result.json", evidence.canonical(outcome))
            return outcome
        finally:
            if process is not None and (
                process.poll() is None or process_session_snapshot(process.pid)
            ):
                terminate_owned_session(process, LIMITS["termination_grace_seconds"])


class NativeInterrupted(BaseException):
    pass


def interrupt(_signum, _frame):
    raise NativeInterrupted()


def asset_check(config):
    """Authenticate actual local assets, including legitimate HF snapshot links."""
    root = Path(config["model_path"])
    expected = config["model_files_sha256"]
    require(
        root.is_absolute() and {p.name for p in root.iterdir()} == set(expected), "asset_inventory"
    )
    observed = {}
    for name, wanted in expected.items():
        require(Path(name).name == name, "asset_name")
        path = root / name
        before = path.stat()
        value = digest(path)
        after = path.stat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "asset_changed",
        )
        require(value == wanted, "asset_hash")
        observed[name] = {"sha256": value, "bytes": after.st_size}
    return observed


def worker(path, expected_sha):
    freeze = verify_freeze(path, expected_sha)
    evidence = frozen_imports(freeze)
    from lexical_prompt_study import instruction_binding_runtime as engine
    from lexical_prompt_study import native_prefix_qualification as core

    prepared_root, root = Path(freeze["prepared_root"]), Path(freeze["output_root"])
    launch = evidence._read_json(evidence._read_file(root / "launch.json"))
    require(
        launch["freeze_sha256"] == expected_sha and launch["supervisor_pid"] == os.getppid(),
        "owned_worker_parent",
    )
    require(os.getsid(0) == os.getpgrp() == os.getpid(), "owned_worker_session")
    evidence._publish(
        root / "worker-claimed.json",
        evidence.canonical(
            {
                "freeze_sha256": expected_sha,
                "pid": os.getpid(),
                "event": "one_shot_worker_claimed",
            }
        ),
    )
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    bind_parent_lifetime(launch["supervisor_pid"])
    body_counts = {"causal_lm_entries": 0, "nested_decoder_entries": 0}
    previous_profile = sys.getprofile()
    try:
        config = hash_bound_json(
            prepared_root / freeze["legacy_config_path"],
            freeze["files_sha256"][freeze["legacy_config_path"]],
        )
        plan = core.validate_plan(
            evidence._read_json(evidence._read_file(prepared_root / freeze["plan_path"]))
        )
        binding = evidence._read_json(
            evidence._read_file(prepared_root / freeze["runtime_binding_path"])
        )
        require(engine.runtime_versions() == config["runtime_versions"], "runtime_versions")
        require(
            evidence.object_hash(binding) == plan["bindings"]["runtime_binding_sha256"],
            "runtime_binding",
        )
        require(
            evidence.canonical(binding)
            == evidence.canonical(
                expected_runtime_binding(
                    config, freeze["files_sha256"][freeze["legacy_config_path"]]
                )
            ),
            "selected_runtime_binding",
        )
        require(
            plan["manifest"]["capture"]["model_layers"] == 32
            and plan["manifest"]["capture"]["hidden_width"] == 4096
            and plan["manifest"]["capture"]["layer_index"] == 31
            and plan["manifest"]["vocab_size"] == 128256,
            "native_geometry",
        )
        assets = asset_check(config)
        require(
            evidence.object_hash(config["model_files_sha256"])
            == plan["manifest"]["provenance"]["model_sha256"],
            "plan_model_assets",
        )
        tokenizer_files = {
            name: sha
            for name, sha in config["model_files_sha256"].items()
            if name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        }
        require(
            evidence.object_hash(tokenizer_files)
            == plan["manifest"]["provenance"]["tokenizer_sha256"],
            "plan_tokenizer_assets",
        )
        require(
            config["chat_template_sha256"]
            == plan["manifest"]["provenance"]["chat_template_sha256"],
            "plan_template",
        )
        evidence._publish(root / "assets-before.json", evidence.canonical(assets))
        require(memory_available() >= LIMITS["minimum_available_bytes"], "preload_memory")
        require(not Path(prepared_root / freeze["loader_path"]).is_symlink(), "loader_path")
        spec = importlib.util.spec_from_file_location(
            "a185_pinned_cpu_loader", prepared_root / freeze["loader_path"]
        )
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

        codes = {
            inspect.unwrap(LlamaForCausalLM.forward).__code__: "causal_lm_entries",
            inspect.unwrap(LlamaModel.forward).__code__: "nested_decoder_entries",
        }
        require(previous_profile is None and len(codes) == 2, "owned_profiler")

        def observe_body(frame, event, _arg):
            if event == "call" and frame.f_code in codes:
                body_counts[codes[frame.f_code]] += 1

        sys.setprofile(observe_body)
        runtime = loader.load_cpu_reference(config, engine)
        require(not any(body_counts.values()), "unexpected_loader_forward")
        model, tokenizer, torch = runtime.model, runtime.tokenizer, runtime.torch
        require(
            type(model).__module__ == "transformers.models.llama.modeling_llama"
            and type(model).__name__ == "LlamaForCausalLM",
            "native_model_class",
        )
        require(
            not torch.cuda.is_initialized()
            and not torch.is_autocast_enabled("cpu")
            and torch.get_num_threads() == 4
            and torch.are_deterministic_algorithms_enabled(),
            "native_precision",
        )
        require(
            model.config.output_hidden_states is False and model.config.output_attentions is False,
            "native_collectors",
        )
        require(model.config._attn_implementation == "sdpa" and not model.training, "native_mode")
        generation_config = hash_bound_json(
            Path(config["model_path"]) / "generation_config.json",
            config["model_files_sha256"]["generation_config.json"],
            allow_snapshot_link=True,
        )
        eos = generation_config["eos_token_id"]
        eos = eos if type(eos) is list else [eos]
        require(
            bool(eos) and all(type(token) is int and 0 <= token < 128256 for token in eos),
            "native_eos_types",
        )
        expected_eos = sorted(set(eos))
        require(runtime.eos_ids == expected_eos == plan["manifest"]["eos_token_ids"], "native_eos")
        require(
            all(
                t.device.type == "cpu"
                and (not t.is_floating_point() or t.dtype == torch.float32)
                and not t.is_complex()
                for t in list(model.parameters()) + list(model.buffers())
            ),
            "native_tensors",
        )
        require(
            not any(type(m).__module__.startswith("bitsandbytes") for m in model.modules()),
            "native_unquantized",
        )
        evidence._publish(
            root / "runtime-loaded.json",
            evidence.canonical(
                {
                    "freeze_sha256": expected_sha,
                    "loader_sha256": LOADER_SHA,
                    "runtime_binding_sha256": evidence.object_hash(binding),
                    "actual_device": "cpu",
                    "actual_dtype": "float32",
                    "quantization": None,
                    "attention": "sdpa",
                    "threads": 4,
                    "deterministic_algorithms": True,
                    "model_class": type(model).__module__ + "." + type(model).__name__,
                    "cuda_initialized": False,
                    "forward_calls_during_load": 0,
                }
            ),
        )
        prepared = core.prepare_inputs(plan, tokenizer)
        evidence._publish(root / "native-prepared.json", evidence.canonical(prepared))
        result = core.run_qualification(
            model,
            tokenizer,
            plan=plan,
            prepared=prepared,
            runtime_binding=binding,
            directory=root / "evidence",
        )
        # This line can only execute after the core has returned normally.
        normal = {
            "event": "run_qualification_returned_normally",
            "freeze_sha256": expected_sha,
            "terminal_sha256": result["terminal_sha256"],
            "summary": result["summary"],
        }
        evidence._publish(root / "runner-normal-return.json", evidence.canonical(normal))
        replay = core.load_qualification(root / "evidence", plan)
        require(evidence.canonical(result) == evidence.canonical(replay), "whole_run_replay")
        require(
            body_counts["causal_lm_entries"]
            == body_counts["nested_decoder_entries"]
            == result["summary"]["total_entries"]
            <= 40,
            "observed_body_count",
        )
        after = asset_check(config)
        require(after == assets, "postrun_asset_stability")
        evidence._publish(root / "assets-after.json", evidence.canonical(after))
        evidence._publish(
            root / "worker-result.json",
            evidence.canonical(
                {
                    "status": "completed",
                    "freeze_sha256": expected_sha,
                    "normal_return_sha256": evidence.object_hash(normal),
                    "terminal_sha256": replay["terminal_sha256"],
                    "summary": replay["summary"],
                    "assets_stable": True,
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "body_entry_counts": body_counts,
                    "count_note": "Decoder entries are nested; do not add them to causal-LM entries.",
                }
            ),
        )
        return result["summary"]
    except BaseException as exc:
        try:
            evidence._publish(
                root / "worker-failure.json",
                evidence.canonical(
                    {
                        "status": "failed",
                        "freeze_sha256": expected_sha,
                        "failure_type": type(exc).__name__,
                        "complete": False,
                        "observed_body_entry_counts": body_counts,
                    }
                ),
            )
        except BaseException:
            pass
        raise
    finally:
        sys.setprofile(previous_profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "worker"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        result = (
            supervise(args.freeze, args.freeze_sha256)
            if args.mode == "supervise"
            else worker(args.freeze, args.freeze_sha256)
        )
    except BaseException as exc:
        print(json.dumps({"status": "failed", "failure_type": type(exc).__name__}))
        return 1
    if args.mode == "supervise":
        print(json.dumps(result, sort_keys=True))
        return int(result["exit_code"] != 0 or result["stop_reason"] is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
