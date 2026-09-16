import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_a164_pipeline.py"
spec = importlib.util.spec_from_file_location("pipeline", SCRIPT)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def args(tmp_path):
    argv = []
    values = {"frozen-src": tmp_path / "src", "python": sys.executable,
              "config": tmp_path / "config.json", "dev-plan": tmp_path / "dev.json",
              "held-plan": tmp_path / "held.json", "run-root": tmp_path / "outputs",
              "config-sha256": "a" * 64, "dev-plan-sha256": "b" * 64, "held-plan-sha256": "c" * 64}
    for key, value in values.items():
        argv.extend(["--" + key, str(value)])
    return argv


def test_virtualenv_interpreter_symlink_path_is_preserved(tmp_path):
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    argv = args(tmp_path)
    argv[argv.index("--python") + 1] = str(interpreter)
    parsed = p.arguments(argv)
    assert parsed.python == interpreter.absolute()
    assert parsed.python != interpreter.resolve()
    worker_args = parsed.worker_arguments()
    assert worker_args[worker_args.index("--python") + 1] == str(interpreter.absolute())
    driver = p.Driver(parsed)
    metadata = json.loads((driver.root / "pipeline-start.json").read_bytes())
    assert metadata["python"] == str(interpreter.absolute())


@pytest.mark.parametrize("gate,expected", [(False, ["development"]), (True, ["development", "heldout"])])
def test_gate_sequence_no_retries_or_extra_runs(tmp_path, monkeypatch, gate, expected):
    stages = []
    def stage(self, name):
        stages.append(name)
        return {"gate_passed": gate, "completed": 32 if name == "development" else 1152,
                "expected": 32 if name == "development" else 1152}
    monkeypatch.setattr(p.Driver, "stage", stage)
    assert p.main(args(tmp_path)) == 0
    assert stages == expected
    status = json.loads((tmp_path / "outputs/status.json").read_bytes())
    assert status["state"] == ("complete" if gate else "stopped_development_gate_failed")
    assert p.main(args(tmp_path)) == 1  # Existing marker never resumes automatically.


@pytest.mark.parametrize("name,cap", [("development", "1200"), ("heldout", "14400")])
def test_fixed_commands_and_artifact_bindings(tmp_path, monkeypatch, name, cap):
    driver = p.Driver(p.arguments(args(tmp_path)))
    commands = []
    def child(command, phase, seconds):
        commands.append((command, phase, seconds))
        if phase.endswith("export-analysis"):
            folder = driver.root / name
            p.private_new(folder / "records.json", [])
            p.private_new(folder / "analysis.json", {"synthetic": True})
            p.private_new(folder / "completion.json", {
                "schema_version": "a164-pipeline-stage-v1", "stage": name,
                "records_sha256": p.digest(folder / "records.json"),
                "analysis_sha256": p.digest(folder / "analysis.json"),
                "gate_passed": True, "expected": 32, "completed": 32,
                "missing": 0, "infrastructure_failed": 0})
    monkeypatch.setattr(driver, "child", child)
    assert driver.stage(name)["gate_passed"] is True
    cmd = commands[0][0]
    assert cmd[:8] == ["flock", "-n", p.GPU_LOCK, "timeout", "--foreground", "--signal=TERM", "--kill-after=20s", cap]
    assert "--retry-failed" not in cmd and "--max-trials" not in cmd
    assert ("--development-plan" in cmd) == (name == "heldout")
    assert commands[1][0][:5] == ["timeout", "--foreground", "--signal=TERM", "--kill-after=20s", "600"]
    assert commands[1][0][-2:] == ["--worker-stage", name]


def test_subprocess_output_private_and_identity_recorded(tmp_path):
    driver = p.Driver(p.arguments(args(tmp_path)))
    driver.child([sys.executable, "-c", "print('synthetic private output')"], "synthetic", 1)
    assert (driver.root / "logs/synthetic.log").read_text().strip() == "synthetic private output"
    events = [json.loads(line) for line in (driver.root / "events.jsonl").read_text().splitlines()]
    running = next(row for row in events if row["state"] == "running")
    assert running["child"]["pid"] == running["child"]["process_group"]
    assert isinstance(running["child"]["start_ticks"], int)
    assert "synthetic private output" not in (driver.root / "events.jsonl").read_text()
    assert (driver.root / "logs/synthetic.log").stat().st_mode & 0o777 == 0o600


def test_whole_deadline_terminates_own_process_group(tmp_path, monkeypatch):
    driver = p.Driver(p.arguments(args(tmp_path)))
    monkeypatch.setattr(p, "TOTAL_SECONDS", 25.2)
    with pytest.raises(p.subprocess.TimeoutExpired):
        driver.child([sys.executable, "-c", "import time; time.sleep(30)"], "synthetic", 5)
    running = next(json.loads(line) for line in (driver.root / "events.jsonl").read_text().splitlines()
                   if json.loads(line)["state"] == "running")
    with pytest.raises(ProcessLookupError):
        os.kill(running["child"]["pid"], 0)
    assert p.CURRENT_CHILD is None


def test_flock_timeout_keeps_nested_children_in_owned_group(tmp_path, monkeypatch):
    driver = p.Driver(p.arguments(args(tmp_path)))
    monkeypatch.setattr(p, "TOTAL_SECONDS", 25.3)
    snippet = ("import subprocess,sys,time; "
               "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
               "print(c.pid,flush=True); time.sleep(30)")
    command = ["flock", "-n", str(tmp_path / "synthetic.lock"), "timeout", "--foreground",
               "--signal=TERM", "--kill-after=20s", "30", sys.executable, "-c", snippet]
    with pytest.raises(p.subprocess.TimeoutExpired):
        driver.child(command, "nested-synthetic", 5)
    pid = int((driver.root / "logs/nested-synthetic.log").read_text().strip())
    stat = p.Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_worker_uses_numeric_replay_then_analysis_and_no_model(tmp_path, monkeypatch):
    parsed = p.arguments(args(tmp_path) + ["--worker-stage", "development"])
    called = []
    records = [{"synthetic_numeric": 1}]
    summary = {"baseline_controls": {"gate_passed": True},
               "coverage": {"expected": 32, "completed": 32, "infrastructure_failed": 0, "missing": 0}}
    def export_records(**kwargs):
        called.append(("export", kwargs))
        return records
    def analyze(plan, rows):
        called.append(("analyze", plan, rows))
        return summary
    package = types.ModuleType("lexical_prompt_study")
    package.instruction_selection_runtime = types.SimpleNamespace(export_records=export_records,
        _read_bound=lambda path, sha: {"synthetic_plan": True})
    package.instruction_selection_analysis = types.SimpleNamespace(analyze_plan_records=analyze)
    package.instruction_selection_tasks = types.SimpleNamespace(sha=lambda raw: p.hashlib.sha256(raw).hexdigest())
    monkeypatch.setitem(sys.modules, "lexical_prompt_study", package)
    p.worker(parsed)
    assert [row[0] for row in called] == ["export", "analyze"]
    assert called[0][1]["expected_plan_sha256"] == "b" * 64
    assert json.loads((parsed.run_root / "development/records.json").read_bytes()) == records
    assert json.loads((parsed.run_root / "development/completion.json").read_bytes())["gate_passed"]


def test_signal_records_failure_and_does_not_advance(tmp_path, monkeypatch, capsys):
    stages = []
    def stage(self, name):
        stages.append(name)
        raise InterruptedError("never print synthetic private exception details")
    monkeypatch.setattr(p.Driver, "stage", stage)
    assert p.main(args(tmp_path)) == 1
    assert stages == ["development"]
    assert "never print" not in capsys.readouterr().out
    assert json.loads((tmp_path / "outputs/status.json").read_bytes())["error_type"] == "InterruptedError"
