#!/usr/bin/env python3
"""One frozen A164 development/held-out sequence; no retries or instrument choices."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GPU_LOCK = "/data2/PRAX/lexical-prompt-study-data/runs/a163/local-gpu.lock"
TOTAL_SECONDS = 18_000
CURRENT_CHILD = None


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "process_group": os.getpgid(pid), "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "process_group": pid, "start_ticks": None}


def stop_owned(child):
    if child is None or child.poll() is not None:
        return
    # Every child is launched in its own new session; no foreign process group.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def interrupted(signum, _frame):
    raise InterruptedError(f"signal_{signum}")


class Driver:
    def __init__(self, args):
        self.args, self.root = args, args.run_root.resolve()
        self.started = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=str(args.frozen_src.resolve()), PYTHONDONTWRITEBYTECODE="1",
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
        self.status = {"schema_version": "a164-pipeline-status-v1", "state": "starting",
                       "driver": identity(os.getpid()), "child": None}
        # An existing start marker forbids accidental pipeline retries.
        private_new(self.root / "pipeline-start.json", {
            "schema_version": "a164-pipeline-start-v1", "driver_source_sha256": digest(__file__),
            "driver": self.status["driver"], "frozen_src": str(args.frozen_src.resolve()),
            "python": str(args.python.absolute()), "config_sha256": args.config_sha256,
            "development_plan_sha256": args.dev_plan_sha256,
            "heldout_plan_sha256": args.held_plan_sha256,
            "gpu_lock": GPU_LOCK, "runtime_caps_seconds": {"development": 1200, "heldout": 14400},
            "export_cap_seconds": 600, "whole_cap_seconds": TOTAL_SECONDS, "retries": 0,
        })
        self.event("started")

    def event(self, state, **fields):
        self.status.update(state=state, elapsed_seconds=round(time.monotonic() - self.started, 3), **fields)
        with (self.root / "events.jsonl").open("ab") as stream:
            stream.write(encoded(self.status))
            stream.flush()
            os.fsync(stream.fileno())
        temporary = self.root / "status.json.tmp"
        with temporary.open("wb") as stream:
            stream.write(encoded(self.status))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.root / "status.json")

    def child(self, command, phase, cap):
        global CURRENT_CHILD
        remaining = TOTAL_SECONDS - (time.monotonic() - self.started) - 25
        if remaining <= 0:
            raise TimeoutError("whole_pipeline_deadline")
        deadline = min(cap + 25, remaining)
        self.event("launching", phase=phase, child=None)
        log = self.root / "logs" / (phase + ".log")
        log.parent.mkdir(mode=0o700, exist_ok=True)
        with log.open("xb") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=self.env, start_new_session=True)
            CURRENT_CHILD = child
            self.event("running", phase=phase, child=identity(child.pid))
            try:
                code = child.wait(timeout=deadline)
            except BaseException:
                stop_owned(child)
                raise
            finally:
                CURRENT_CHILD = None
        self.event("child_finished", phase=phase, child=None, child_exit_code=code,
                   log_sha256=digest(log))
        if code != 0:
            raise RuntimeError("child_failed")

    def stage(self, name):
        args = self.args
        plan, plan_sha = ((args.dev_plan, args.dev_plan_sha256) if name == "development"
                          else (args.held_plan, args.held_plan_sha256))
        folder = self.root / name
        cap = 1200 if name == "development" else 14400
        command = ["flock", "-n", GPU_LOCK, "timeout", "--foreground", "--signal=TERM", "--kill-after=20s", str(cap),
                   str(args.python), "-m", "lexical_prompt_study.instruction_selection_runtime",
                   "--plan", str(plan), "--plan-sha256", plan_sha,
                   "--config", str(args.config), "--config-sha256", args.config_sha256,
                   "--output-root", str(folder / "run")]
        if name == "heldout":
            command += ["--development-plan", str(args.dev_plan), "--development-plan-sha256",
                        args.dev_plan_sha256, "--development-run-root", str(self.root / "development" / "run")]
        self.child(command, name + "-runtime", cap)
        worker = ["timeout", "--foreground", "--signal=TERM", "--kill-after=20s", "600", str(args.python),
                  str(Path(__file__).resolve()), *self.args.worker_arguments(), "--worker-stage", name]
        self.child(worker, name + "-export-analysis", 600)
        report_path = folder / "completion.json"
        report = json.loads(report_path.read_bytes())
        expected = {"schema_version", "stage", "records_sha256", "analysis_sha256", "gate_passed",
                    "expected", "completed", "infrastructure_failed", "missing"}
        if set(report) != expected or report["schema_version"] != "a164-pipeline-stage-v1" or report["stage"] != name:
            raise ValueError("invalid_stage_completion")
        if digest(folder / "records.json") != report["records_sha256"] or digest(folder / "analysis.json") != report["analysis_sha256"]:
            raise ValueError("stage_artifact_drift")
        self.event("stage_complete", phase=name, child=None, stage_completion=report)
        return report


def worker(args):
    """CPU tokenizer replay only; all model generation occurs in the runtime child."""
    sys.path.insert(0, str(args.frozen_src.resolve()))
    from lexical_prompt_study import instruction_selection_analysis as analysis
    from lexical_prompt_study import instruction_selection_runtime as runtime
    from lexical_prompt_study import instruction_selection_tasks as tasks
    stage = args.worker_stage
    plan_path, plan_sha = ((args.dev_plan, args.dev_plan_sha256) if stage == "development"
                           else (args.held_plan, args.held_plan_sha256))
    folder = args.run_root / stage
    records = runtime.export_records(plan_path=plan_path, config_path=args.config, output_root=folder / "run",
        expected_plan_sha256=plan_sha, expected_config_sha256=args.config_sha256)
    plan = runtime._read_bound(plan_path, plan_sha)
    summary = analysis.analyze_plan_records(plan, records)
    private_new(folder / "records.json", records)
    private_new(folder / "analysis.json", summary)
    coverage = summary["coverage"]
    private_new(folder / "completion.json", {
        "schema_version": "a164-pipeline-stage-v1", "stage": stage,
        "records_sha256": tasks.sha((folder / "records.json").read_bytes()),
        "analysis_sha256": tasks.sha((folder / "analysis.json").read_bytes()),
        "gate_passed": summary["baseline_controls"]["gate_passed"],
        **{key: coverage[key] for key in ("expected", "completed", "infrastructure_failed", "missing")},
    })


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen-src", "python", "config", "dev-plan", "held-plan", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config-sha256", "dev-plan-sha256", "held-plan-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--worker-stage", choices=("development", "heldout"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Resolving a venv's interpreter symlink bypasses its pyvenv.cfg and changes
    # the environment. Keep the lexical interpreter path for every subprocess.
    args.python = args.python.absolute()
    for name in ("frozen_src", "config", "dev_plan", "held_plan", "run_root"):
        setattr(args, name, getattr(args, name).resolve())
    def worker_arguments():
        result = []
        for name in ("frozen_src", "python", "config", "dev_plan", "held_plan", "run_root",
                     "config_sha256", "dev_plan_sha256", "held_plan_sha256"):
            result.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        return result
    args.worker_arguments = worker_arguments
    return args


def main(argv=None):
    os.umask(0o077)
    args = arguments(argv)
    if args.worker_stage:
        worker(args)
        return 0
    driver = None
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        driver = Driver(args)
        dev = driver.stage("development")
        if dev["gate_passed"] is not True:
            driver.event("stopped_development_gate_failed", child=None)
            return 0
        heldout = driver.stage("heldout")
        state = "complete" if heldout["completed"] == heldout["expected"] else "heldout_incomplete"
        driver.event(state, child=None)
        return 0
    except BaseException as exc:
        stop_owned(CURRENT_CHILD)
        if driver is not None:
            driver.event("failed", child=None, error_type=type(exc).__name__)
        # No exception message, request, token, or response is printed.
        print(json.dumps({"status": "a164_pipeline_failed", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
