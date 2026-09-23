"""Invented-data, separate-runtime A186 fit-worker lifecycle qualification."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lexical_prompt_study import matched_prefix_prediction as prediction
from lexical_prompt_study import matched_prefix_tasks as tasks

SCRIPT = Path(__file__).parents[1] / "scripts/matched_prefix_fit_worker.py"
spec = importlib.util.spec_from_file_location("a186_fit_worker_test", SCRIPT)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def request_value():
    roster = tasks.prediction_roster(tasks.build_roster())
    packets, labels, reached = [], {}, {}
    for row in roster:
        semantic = [0.0] * 768
        internal = [0.0] * 4096
        semantic[row["core_index"] % 5] = 1.0
        internal[row["core_index"] % 17] = 1.0
        internal[25] = -0.5 if row["core_index"] % 2 else 0.5
        packets.append(
            {
                "schema_version": "a186-prediction-features-v1",
                "observation_id": row["observation_id"],
                "common": {
                    "rendered_prompt_utf8": "Invented public fixture",
                    "prompt_token_ids": [1, 2],
                    "prefix_utf8": "",
                    "prefix_token_ids": [3, 4, 5, 6, 7, 8, 9, 10],
                    "completed_fields": 0,
                    "prefix_known_error": False,
                },
                "text_semantic": semantic,
                "internal": internal,
            }
        )
        labels[row["observation_id"]] = row["core_index"] % 2
        reached[row["observation_id"]] = True
    bindings = {name: "1" * 64 for name in prediction.BINDING_KEYS}
    bindings.update(
        feature_packets_sha256=worker.object_hash(packets),
        labels_sha256=worker.object_hash(labels),
        landmark_reached_sha256=worker.object_hash(reached),
        fit_runtime_sha256=worker.object_hash(prediction.runtime_descriptor()),
        prediction_source_sha256=prediction.source_sha256(),
    )
    return {
        "roster": roster,
        "feature_packets": packets,
        "labels": labels,
        "bindings": bindings,
        "landmark_reached": reached,
    }


def write_request(tmp_path, value=None):
    value = request_value() if value is None else value
    path = tmp_path / "request.json"
    raw = worker.canonical(value)
    path.write_bytes(raw)
    return path, worker.sha256(raw)


def read(path):
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("fit-worker-completed")
    path, digest = write_request(root)
    output = root / "output"
    summary = worker.run_worker(path, digest, output)
    return path, digest, output, summary


def test_success_two_actual_entries_retained_models_and_no_refit(completed, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    path, digest, output, summary = completed
    assert (
        summary["actual_fit_entries"] == 2 == summary["fit_claims"] == summary["fit_body_returns"]
    )
    assert summary["event_count"] == 5 and summary["model_free_replay"] is True
    assert sys.getprofile() is None
    assert "torch" not in sys.modules
    original = LogisticRegression.fit

    def forbidden(*args, **kwargs):
        pytest.fail("replay must not fit")

    forbidden.__wrapped__ = original  # Keep the authenticated original code identity.
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    replay = worker.load_worker(output, expected_request_sha256=digest)
    assert replay["result"]["actual_fits"] == 2
    assert len(replay["result"]["predictions"]) == 112
    terminal = replay["terminal"]
    assert terminal["request_sha256"] == digest
    assert (
        read(output / "normal-return.json")["returned_result_sha256"] == terminal["result_sha256"]
    )
    assert all((output / "entries" / f"fit_{i:02d}.json").is_file() for i in (1, 2))


def test_zero_fit_support_failure_is_complete_with_null_predictions(tmp_path):
    request = request_value()
    request["labels"] = {key: 1 for key in request["labels"]}
    request["bindings"]["labels_sha256"] = worker.object_hash(request["labels"])
    path, digest = write_request(tmp_path, request)
    summary = worker.run_worker(path, digest, tmp_path / "output")
    assert summary["actual_fit_entries"] == summary["fit_claims"] == summary["event_count"] == 0
    assert read(tmp_path / "output/result.json")["analysis"]["primary_csv"]["point"] is None


@pytest.mark.parametrize(
    "mutation", ["hash", "noncanonical", "extra", "runtime", "source", "reachability"]
)
def test_authentication_rejects_before_output_claim(tmp_path, mutation):
    request = request_value()
    if mutation == "extra":
        request["future_labels"] = {}
    if mutation == "runtime":
        request["bindings"]["fit_runtime_sha256"] = "f" * 64
    if mutation == "source":
        request["bindings"]["prediction_source_sha256"] = "f" * 64
    if mutation == "reachability":
        request["landmark_reached"] = None
    path, digest = write_request(tmp_path, request)
    if mutation == "hash":
        digest = "f" * 64
    if mutation == "noncanonical":
        path.write_bytes(path.read_bytes() + b"\n")
        digest = worker.sha256(path.read_bytes())
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        worker.run_worker(path, digest, output)
    assert not output.exists()
    assert sys.getprofile() is None


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{ "x":1}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        worker._parse(raw)


def test_request_symlink_and_existing_root_not_adopted(tmp_path):
    path, digest = write_request(tmp_path)
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        worker.run_worker(linked, digest, tmp_path / "output")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "preserved"
    sentinel.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        worker.run_worker(path, digest, output)
    assert {item.name for item in output.iterdir()} == {"preserved"}
    assert sentinel.read_bytes() == b"existing"


def test_keyboard_interrupt_preserved_and_profiler_restored(tmp_path, monkeypatch):
    path, digest = write_request(tmp_path)
    original = KeyboardInterrupt("invented private text that must not reach console")

    def stop(*args, **kwargs):
        raise original

    monkeypatch.setattr(prediction, "fit_predict", stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        worker.run_worker(path, digest, tmp_path / "output")
    assert caught.value is original and sys.getprofile() is None
    failure = read(tmp_path / "output/failure.json")
    assert failure["actual_fit_entries"] == 0 and failure["failure_type"] == "KeyboardInterrupt"
    assert failure["profile_restored"] is True
    assert not (tmp_path / "output/terminal.json").exists()


def test_original_error_survives_cleanup_failure(tmp_path, monkeypatch):
    path, digest = write_request(tmp_path)
    original = LookupError("invented original")
    real_setprofile = sys.setprofile

    def setprofile(value):
        real_setprofile(value)
        if value is None:
            raise OSError("invented cleanup")

    def stop(*args, **kwargs):
        raise original

    monkeypatch.setattr(prediction, "fit_predict", stop)
    monkeypatch.setattr(sys, "setprofile", setprofile)
    try:
        with pytest.raises(LookupError) as caught:
            worker.run_worker(path, digest, tmp_path / "output")
        assert caught.value is original
        failure = read(tmp_path / "output/failure.json")
        assert failure["cleanup_failure_type"] == "OSError"
        assert failure["failure_type"] == "LookupError"
    finally:
        real_setprofile(None)


def test_existing_profiler_rejected_and_untouched(tmp_path):
    path, digest = write_request(tmp_path)

    def existing(*args):
        pass

    sys.setprofile(existing)
    try:
        with pytest.raises(ValueError, match="existing_profiler"):
            worker.run_worker(path, digest, tmp_path / "output")
        assert sys.getprofile() is existing
        assert not (tmp_path / "output").exists()
    finally:
        sys.setprofile(None)


def test_one_actual_fit_entry_then_forced_return_journal_failure(tmp_path, monkeypatch):
    path, digest = write_request(tmp_path)
    publish = worker._publish
    sentinel = OSError("invented publication failure")

    def fail_return(path, raw):
        if Path(path).name == "event_002.json":
            raise sentinel
        return publish(path, raw)

    monkeypatch.setattr(worker, "_publish", fail_return)
    with pytest.raises(OSError) as caught:
        worker.run_worker(path, digest, tmp_path / "output")
    assert caught.value is sentinel
    failure = read(tmp_path / "output/failure.json")
    assert (
        failure["actual_fit_entries"] == 1 == failure["fit_body_returns"] == failure["fit_claims"]
    )
    assert len(list((tmp_path / "output/entries").iterdir())) == 1
    assert not (tmp_path / "output/terminal.json").exists()
    assert sys.getprofile() is None


def test_entry_publication_failure_retains_observed_boundary_count(tmp_path, monkeypatch):
    path, digest = write_request(tmp_path)
    publish = worker._publish

    def fail_entry(path, raw):
        if Path(path).name == "fit_01.json":
            raise OSError("invented entry fsync failure")
        return publish(path, raw)

    monkeypatch.setattr(worker, "_publish", fail_entry)
    with pytest.raises(worker.FitPipelineFailed):
        worker.run_worker(path, digest, tmp_path / "output")
    failure = read(tmp_path / "output/failure.json")
    assert failure["actual_fit_entries"] == 1 and failure["fit_body_returns"] == 0
    assert failure["entry_sha256"] == []
    assert read(tmp_path / "output/result.json")["terminal_error"] == "OSError"
    assert sys.getprofile() is None


def test_partial_publication_never_loads_as_complete(completed, tmp_path):
    _, digest, source, _ = completed
    output = tmp_path / "output"
    shutil.copytree(source, output)
    (output / ".pending-orphan").write_bytes(b"invented")
    with pytest.raises(ValueError, match="inventory"):
        worker.load_worker(output, expected_request_sha256=digest)


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_count",
        "missing_entry",
        "extra_event",
        "symlink_result",
        "wrong_normal",
        "changed_request",
    ],
)
def test_completed_tamper_rejected(completed, tmp_path, mutation):
    _, digest, source, _ = completed
    output = tmp_path / "output"
    shutil.copytree(source, output)
    if mutation == "bool_count":
        terminal = read(output / "terminal.json")
        terminal["actual_fit_entries"] = True
        (output / "terminal.json").write_bytes(worker.canonical(terminal))
    if mutation == "missing_entry":
        (output / "entries/fit_02.json").unlink()
    if mutation == "extra_event":
        (output / "events/event_005.json").write_bytes(b"{}")
    if mutation == "symlink_result":
        moved = tmp_path / "result.json"
        (output / "result.json").rename(moved)
        (output / "result.json").symlink_to(moved)
    if mutation == "wrong_normal":
        normal = read(output / "normal-return.json")
        normal["fit_predict_returned"] = False
        (output / "normal-return.json").write_bytes(worker.canonical(normal))
    if mutation == "changed_request":
        (output / "request.json").write_bytes(b"{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.load_worker(output, expected_request_sha256=digest)


def test_import_is_stdlib_only():
    code = "import runpy,sys; runpy.run_path(sys.argv[1],run_name='inert_worker'); assert not any(x in sys.modules for x in ('torch','numpy','sklearn','lexical_prompt_study')); print('inert')"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(SCRIPT)], capture_output=True, text=True, check=True
    )
    assert result.stdout == "inert\n" and result.stderr == ""


def test_cli_failure_sanitizes_private_text_and_checks_parent(tmp_path):
    path, digest = write_request(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--request",
            str(path),
            "--request-sha256",
            digest,
            "--output",
            str(tmp_path / "output"),
            "--parent",
            str(os.getpid() + 999999),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and result.stderr == ""
    summary = json.loads(result.stdout)
    assert summary["status"] == "failed" and summary["failure_type"] == "ValueError"
    assert "Invented public fixture" not in result.stdout
    assert not (tmp_path / "output").exists()


def test_optimized_cli_refuses_before_output(tmp_path):
    path, digest = write_request(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-O",
            str(SCRIPT),
            "--request",
            str(path),
            "--request-sha256",
            digest,
            "--output",
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and json.loads(result.stdout)["failure_type"] == "RuntimeError"
    assert not (tmp_path / "output").exists()


def test_third_fit_claim_blocked_before_estimator_entry(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    path, digest = write_request(tmp_path)
    body_returns = []

    def deliberately_extra(*args, event, **kwargs):
        for index, slot in enumerate(("text", "text_internal", "third"), start=1):
            event(
                worker.canonical(
                    {
                        "schema_version": "a186-fit-event-v1",
                        "kind": "before_fit",
                        "slot": slot,
                        "attempt_index": index,
                        "options": prediction.OPTIONS,
                    }
                )
            )
            model = LogisticRegression(**prediction.OPTIONS)
            model.fit([[0.0], [1.0], [0.5], [-0.5]], [0, 1, 1, 0])
            body_returns.append(index)
            event(
                worker.canonical(
                    {
                        "schema_version": "a186-fit-event-v1",
                        "kind": "fit_return",
                        "slot": slot,
                        "attempt_index": index,
                    }
                )
            )
        pytest.fail("third model fit must not be entered")

    monkeypatch.setattr(prediction, "fit_predict", deliberately_extra)
    with pytest.raises(ValueError, match="claim_count"):
        worker.run_worker(path, digest, tmp_path / "output")
    assert body_returns == [1, 2]
    failure = read(tmp_path / "output/failure.json")
    assert (
        failure["actual_fit_entries"] == failure["fit_claims"] == failure["fit_body_returns"] == 2
    )
    assert len(list((tmp_path / "output/entries").iterdir())) == 2


def test_failure_after_terminal_publication_invalidates_complete_root(tmp_path, monkeypatch):
    request = request_value()
    request["labels"] = {key: 0 for key in request["labels"]}
    request["bindings"]["labels_sha256"] = worker.object_hash(request["labels"])
    path, digest = write_request(tmp_path, request)
    real_load = worker.load_worker
    sentinel = OSError("invented post-terminal caller failure")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(worker, "load_worker", fail)
    output = tmp_path / "output"
    with pytest.raises(OSError) as caught:
        worker.run_worker(path, digest, output)
    assert caught.value is sentinel
    assert (output / "terminal.json").exists() and (output / "failure.json").exists()
    assert read(output / "failure.json")["actual_fit_entries"] == 0
    with pytest.raises(ValueError, match="inventory"):
        real_load(output, expected_request_sha256=digest)


def test_fsynced_orphan_not_reused_and_original_exception_retained(tmp_path, monkeypatch):
    path, digest = write_request(tmp_path)
    original_link = os.link
    sentinel = OSError("invented exclusive publication failure")

    def fail_link(source, target, **kwargs):
        if Path(target).name == "run.json":
            raise sentinel
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", fail_link)
    output = tmp_path / "output"
    with pytest.raises(OSError) as caught:
        worker.run_worker(path, digest, output)
    assert caught.value is sentinel
    assert (output / ".pending-run.json").exists()
    assert read(output / "failure.json")["actual_fit_entries"] == 0
    with pytest.raises(FileExistsError):
        worker.run_worker(path, digest, output)
