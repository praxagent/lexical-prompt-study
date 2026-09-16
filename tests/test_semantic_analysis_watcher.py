"""Synthetic process identities and mocked analyses only; no study reads."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/watch_semantic_analysis.py"
SPEC = importlib.util.spec_from_file_location("semantic_analysis_watcher", SCRIPT)
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def stat(state, ticks):
    return f"{watcher.PIPELINE_PID} (command with nested (parentheses)) " + " ".join(
        [state] + ["0"] * 18 + [str(ticks)])


def test_linux_stat_fields_handle_parentheses_in_command():
    assert watcher.process_identity(stat("S", watcher.PIPELINE_START_TICKS)) == ("S", watcher.PIPELINE_START_TICKS)


@pytest.mark.parametrize("state,ticks,finished", [
    ("S", watcher.PIPELINE_START_TICKS, False),
    ("R", watcher.PIPELINE_START_TICKS, False),
    ("Z", watcher.PIPELINE_START_TICKS, True),
    ("X", watcher.PIPELINE_START_TICKS, True),
    ("S", watcher.PIPELINE_START_TICKS + 1, True),
])
def test_exact_identity_zombie_and_pid_reuse(tmp_path, state, ticks, finished):
    process = tmp_path / str(watcher.PIPELINE_PID)
    process.mkdir()
    (process / "stat").write_text(stat(state, ticks))
    assert watcher.original_process_finished(proc_root=tmp_path) is finished


def test_missing_process_is_finished_but_malformed_stat_is_not(tmp_path):
    assert watcher.original_process_finished(proc_root=tmp_path)
    process = tmp_path / str(watcher.PIPELINE_PID)
    process.mkdir()
    (process / "stat").write_text("invalid")
    with pytest.raises((ValueError, IndexError)):
        watcher.original_process_finished(proc_root=tmp_path)


def test_wait_deadline_uses_no_sleep_longer_than_fifteen_seconds():
    clock, sleeps = [0], []
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    assert not watcher.wait_for_pipeline(31, clock=lambda: clock[0], sleep=sleep, check=lambda: False)
    assert sleeps == [15, 15, 1] and clock[0] == 31


def test_wait_returns_when_original_identity_finishes():
    clock, sleeps, checks = [0], [], iter((False, True))
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    assert watcher.wait_for_pipeline(30, clock=lambda: clock[0], sleep=sleep, check=lambda: next(checks))
    assert sleeps == [15]


def test_venv_interpreter_symlink_is_not_resolved(tmp_path, monkeypatch):
    actual = tmp_path / "actual-python"
    actual.write_bytes(b"synthetic executable; never launched")
    link = tmp_path / ".venv/bin/python"
    link.parent.mkdir(parents=True)
    link.symlink_to(actual)
    monkeypatch.setattr(watcher, "PROJECT_PYTHON", link)
    command = watcher.command(tmp_path / "frozen-source")
    assert command[0] == str(link) and command[0] != str(actual)
    assert command[1:4] == ["-B", "-m", "lexical_prompt_study.text_challenger_semantic_analysis"]


def test_frozen_snapshot_rejects_extra_source_and_bytecode(tmp_path, monkeypatch):
    source = tmp_path / "src"
    package = source / "lexical_prompt_study"
    package.mkdir(parents=True)
    path = package / "__init__.py"
    path.write_bytes(b"# harmless synthetic module\n")
    monkeypatch.setattr(watcher, "SOURCE_PINS", {"lexical_prompt_study/__init__.py": watcher.sha_file(path)})
    assert watcher.verify_sources(source) == source
    (package / "unexpected.py").write_bytes(b"# never executed\n")
    with pytest.raises(ValueError):
        watcher.verify_sources(source)
    (package / "unexpected.py").unlink()
    (package / "unexpected.pyc").write_bytes(b"not bytecode")
    with pytest.raises(ValueError):
        watcher.verify_sources(source)


def test_timeout_terminates_only_owned_analysis_group(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(watcher, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(watcher, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(watcher, "verify_sources", lambda value: value)
    def fake_hash(path):
        if Path(path) == watcher.LEXICAL_MANIFEST:
            return watcher.LEXICAL_MANIFEST_SHA
        if Path(path) == watcher.SEMANTIC_LAUNCH:
            return watcher.SEMANTIC_LAUNCH_SHA
        return "0" * 64
    monkeypatch.setattr(watcher, "sha_file", fake_hash)
    process = Mock(pid=4321)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 5),
                               subprocess.TimeoutExpired("synthetic", 5), None]
    popen = Mock(return_value=process)
    killpg = Mock()
    monkeypatch.setattr(watcher.subprocess, "Popen", popen)
    monkeypatch.setattr(watcher.os, "killpg", killpg)
    assert watcher.run_analysis(SimpleNamespace(analysis_seconds=10), source) == 124
    assert [call.args for call in killpg.call_args_list] == [
        (4321, watcher.signal.SIGTERM), (4321, watcher.signal.SIGKILL)]
    assert popen.call_args.kwargs["start_new_session"] is True
    assert popen.call_args.kwargs["env"]["PYTHONPATH"] == str(source)
    assert (tmp_path / "analysis-watch.analysis-attempt.json").is_file()
    assert (tmp_path / "analysis-watch.stdout.log").stat().st_mode & 0o777 == 0o600


def test_consumed_analysis_attempt_cannot_be_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(watcher, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(watcher, "verify_sources", lambda value: value.absolute())
    monkeypatch.setattr(watcher, "PROJECT_PYTHON", SCRIPT)
    (tmp_path / "analysis-watch.analysis-attempt.json").write_text("{}")
    wait = Mock(side_effect=AssertionError("Consumed attempt must not wait or execute"))
    run = Mock(side_effect=AssertionError("Consumed attempt must not execute"))
    monkeypatch.setattr(watcher, "wait_for_pipeline", wait)
    monkeypatch.setattr(watcher, "run_analysis", run)
    assert watcher.main(["--source-root", str(tmp_path)]) == 2
    wait.assert_not_called()
    run.assert_not_called()


def test_check_only_never_waits_or_runs_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "verify_sources", lambda value: value.absolute())
    monkeypatch.setattr(watcher, "PROJECT_PYTHON", SCRIPT)
    wait, run = Mock(), Mock()
    monkeypatch.setattr(watcher, "wait_for_pipeline", wait)
    monkeypatch.setattr(watcher, "run_analysis", run)
    assert watcher.main(["--source-root", str(tmp_path), "--check-only"]) == 0
    wait.assert_not_called()
    run.assert_not_called()
