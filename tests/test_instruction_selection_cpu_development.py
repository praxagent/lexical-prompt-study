import copy
import importlib.util
import json
from pathlib import Path
import signal
import sys
import types

import pytest

from lexical_prompt_study import instruction_selection_cpu_development as a
from lexical_prompt_study import instruction_selection_tasks as tasks

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a165_synthetic_helpers", ROOT / "tests/test_instruction_selection_runtime.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    data = helper.inputs.__wrapped__(tmp_path)
    protocol = tmp_path / "a165.md"
    protocol.write_text("public synthetic protocol")
    materials = {kind: "Public synthetic " + kind for kind in tasks.SCAFFOLDS}
    plan = a.compile_plan(data["development"], scaffolds=materials,
                          protocol_sha256=a.native.engine.file_digest(protocol))
    path = tmp_path / "a165-plan.json"
    path.write_bytes(tasks.canonical(plan))
    reference = tmp_path / "frozen-reference.py"
    reference.write_text("public synthetic reference source")
    # Synthetic source file is pinned explicitly; production constants stay strict.
    original_digest = a.native.engine.file_digest
    monkeypatch.setattr(a.native.engine, "file_digest", lambda p: a.REFERENCE_SHA if Path(p) == reference else original_digest(p))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: data["config"]["model_files_sha256"])
    args = {"plan_path": path, "plan_sha256": original_digest(path), "config_path": data["config_path"],
            "config_sha256": original_digest(data["config_path"]), "protocol_path": protocol,
            "reference_script": reference, "output_root": tmp_path / "output",
            "tokenizer_loader": lambda config: helper.Tokenizer()}
    return data, plan, args


def run(fixture, mode="correct", interrupt_after=None):
    data, plan, args = fixture
    model = helper.FakeRuntime(plan, mode)
    if interrupt_after is not None:
        original = model.generate
        def generate(prepared):
            if model.calls == interrupt_after:
                a._interrupt(signal.SIGTERM, None)
            return original(prepared)
        model.generate = generate
    reference = types.SimpleNamespace(require_memory=lambda: 48 * 1024**3,
                                      load_cpu_reference=lambda config, engine: model)
    result = a.run_plan(**args, reference_loader=lambda path: reference)
    return result, model


def export(fixture):
    args = fixture[2]
    return a.export_records(**{key: args[key] for key in (
        "plan_path", "plan_sha256", "config_path", "config_sha256", "output_root", "tokenizer_loader")})


def test_exact_development_matrix_unchanged_prompts_materials_and_new_ids(fixture):
    data, plan, _ = fixture
    assert len(plan["trials"]) == len({row["trial_id"] for row in plan["trials"]}) == 288
    assert all(row["scaffold_kind"] == "none" for row in plan["trials"][:32])
    assert not ({row["world_id"] for row in data["heldout"]["trials"]} & {row["world_id"] for row in plan["trials"]})
    for baseline, original in zip(plan["trials"][:32], data["development"]["trials"], strict=True):
        assert baseline["messages"] == original["messages"]
        assert baseline["trial_id"] != original["trial_id"]
    pairs = {}
    for row in plan["trials"]:
        key = row["world_id"], row["scaffold_kind"], row["placement"]
        assert pairs.setdefault(key, row["messages"][1]) == row["messages"][1]
        assert "depth" not in row


@pytest.mark.parametrize("change", ["heldout", "material", "source", "reorder", "missing", "system"])
def test_modified_instrument_or_matrix_rejected(fixture, change):
    data, plan, _ = fixture
    value = copy.deepcopy(plan)
    if change == "heldout":
        with pytest.raises(ValueError):
            a.compile_plan(data["heldout"], scaffolds={}, protocol_sha256="a" * 64)
        return
    if change == "material":
        value["trials"][32]["messages"][1]["content"] += "changed"
    elif change == "source":
        value["bindings"]["a165_source_sha256"] = "0" * 64
    elif change == "reorder":
        value["trials"][32], value["trials"][33] = value["trials"][33], value["trials"][32]
    elif change == "missing":
        value["trials"].pop()
    else:
        value["trials"][0]["messages"][0]["content"] += "changed"
    with pytest.raises(ValueError):
        a.validate_plan(value)


def test_baseline_gate_boundaries_and_missing(fixture):
    plan = fixture[1]
    rows = [a.numeric_record(row) for row in plan["trials"][:32]]
    for row in rows:
        row.update(generation_status="completed", finish_reason="eos", score={"category": "exact", "strict_correct": True})
    assert a.baseline_gate(plan, rows)["gate_passed"]
    rows[0]["score"] = {"category": "other_selector", "strict_correct": False}
    assert a.baseline_gate(plan, rows)["gate_passed"]  # 31 cells,15 pairs.
    rows[2]["score"] = {"category": "other_selector", "strict_correct": False}
    assert not a.baseline_gate(plan, rows)["gate_passed"]  # 30 cells,14 pairs.
    assert not a.baseline_gate(plan, rows[1:])["gate_passed"]


def test_complete_run_export_and_real_analysis_integration(fixture):
    result, model = run(fixture)
    assert result["status"] == "complete" and result["recorded"] == model.calls == 288
    assert result["baseline_controls"]["gate_passed"] and result["heldout_allowed"] is False
    rows = export(fixture)
    assert len(rows) == 288 and all(row["schema_version"] == "a165-cell-v1" for row in rows)
    root = fixture[2]["output_root"]
    header = json.loads((root / "run.json").read_bytes())
    assert a.native.engine.file_digest(root / "native-inputs.json") == header["native_prepared_sha256"]
    assert (root / "analysis.json").is_file() and not (root / "analysis-error.json").exists()
    aggregate = json.loads((root / "analysis.json").read_bytes())
    assert aggregate["baseline_controls"]["gate_passed"]
    with pytest.raises(FileExistsError):
        run(fixture)


@pytest.mark.parametrize("mode", ["wrong", "cap", "infrastructure"])
def test_failed_baselines_stop_before_scaffold_and_preserve_missing(fixture, mode):
    result, model = run(fixture, mode)
    assert result["status"] == "stopped_baseline_gate_failed"
    assert model.calls == result["recorded"] == 32 and result["unlaunched"] == 256
    assert all(row["scaffold"] == "none" for row in export(fixture))
    assert (fixture[2]["output_root"] / "analysis.json").exists()


def test_sigterm_inside_generate_stops_immediately(fixture):
    result, model = run(fixture, interrupt_after=2)
    assert result["status"] == "interrupted" and result["recorded"] == model.calls == 2
    assert len(list(fixture[2]["output_root"].glob("trials/*/attempt.json"))) == 3
    assert len(export(fixture)) == 2
    assert (fixture[2]["output_root"] / "analysis.json").exists()


@pytest.mark.parametrize("tamper", ["prompt", "response", "score", "gate"])
def test_receipt_export_rejects_tampering(fixture, tamper):
    run(fixture)
    root = fixture[2]["output_root"]
    result_path = next(root.glob("trials/*/result.json"))
    attempt_path = result_path.with_name("attempt.json")
    if tamper == "gate":
        path = root / "baseline-gate.json"
        value = json.loads(path.read_bytes())
        value["gate_passed"] = False
    elif tamper == "prompt":
        path = attempt_path
        value = json.loads(path.read_bytes())
        value["prepared"]["prompt_token_ids"][0] += 1
    else:
        path = result_path
        value = json.loads(path.read_bytes())
        if tamper == "response":
            value["private"]["response_text"] = '{"answer":"wrong"}'
        else:
            value["record"]["score"]["strict_correct"] = False
    path.write_bytes(tasks.canonical(value))
    with pytest.raises(ValueError):
        export(fixture)


def test_analysis_failure_does_not_discard_records_progress(fixture, monkeypatch):
    from lexical_prompt_study import instruction_selection_cpu_development_analysis as analysis
    def fail(*args):
        raise ValueError("synthetic private error")
    monkeypatch.setattr(analysis, "analyze_plan_records", fail)
    result, _ = run(fixture, "wrong")
    root = fixture[2]["output_root"]
    assert result["recorded"] == 32 and (root / "records.json").exists() and (root / "progress.json").exists()
    assert (root / "analysis-error.json").exists() and not (root / "analysis.json").exists()
    assert "synthetic private error" not in (root / "analysis-error.json").read_text()


def test_main_text_logging_and_no_private_output(fixture, monkeypatch, capsys):
    args = fixture[2]
    original = a.run_plan
    model = helper.FakeRuntime(fixture[1], "wrong")
    def loader(config, engine):
        print("private synthetic loading")
        sys.stderr.write("private progress \u2588\n")
        return model
    reference = types.SimpleNamespace(require_memory=lambda: 48 * 1024**3, load_cpu_reference=loader)
    monkeypatch.setattr(a, "run_plan", lambda **kwargs: original(**kwargs,
        tokenizer_loader=lambda config: helper.Tokenizer(), reference_loader=lambda path: reference))
    argv = []
    for key in ("plan_path", "plan_sha256", "config_path", "config_sha256", "protocol_path", "reference_script", "output_root"):
        flag = key.removesuffix("_path").replace("_", "-")
        argv += ["--" + flag, str(args[key])]
    assert a.main(argv) == 0
    assert model.calls == 32
    assert "private synthetic loading" in (args["output_root"] / "execution.log").read_text()
    captured = capsys.readouterr()
    assert "private" not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "stopped_baseline_gate_failed"
