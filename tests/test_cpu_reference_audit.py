import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from lexical_prompt_study import instruction_selection_runtime as rt
from lexical_prompt_study import instruction_selection_tasks as tasks

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_reference", ROOT / "scripts/run_cpu_reference_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
HELPER_SPEC = importlib.util.spec_from_file_location("synthetic_runtime_helpers", ROOT / "tests/test_instruction_selection_runtime.py")
helpers = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helpers)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data = helpers.inputs.__wrapped__(tmp_path)
    nf4_args = helpers.args(data)
    rt.run_plan(**nf4_args, runtime_loader=lambda config: helpers.FakeRuntime(data["development"], "wrong"))
    original_export = rt.export_records
    monkeypatch.setattr(rt, "export_records", lambda **kwargs: original_export(
        **kwargs, tokenizer_loader=lambda config: helpers.Tokenizer()))
    monkeypatch.setattr(rt, "load_tokenizer", lambda config: helpers.Tokenizer())
    monkeypatch.setattr(rt.engine, "snapshot_manifest", lambda path: data["config"]["model_files_sha256"])
    monkeypatch.setattr(audit, "require_memory", lambda: audit.MIN_AVAILABLE_BYTES)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("public synthetic protocol")
    args = types.SimpleNamespace(frozen_src=tmp_path / "frozen/src", plan=data["development_path"],
        config=data["config_path"], nf4_run_root=nf4_args["output_root"], protocol=protocol,
        output_root=tmp_path / "reference", plan_sha256=rt.engine.file_digest(data["development_path"]),
        config_sha256=rt.engine.file_digest(data["config_path"]), protocol_sha256=rt.engine.file_digest(protocol))
    return data, args


def test_exact_32_reference_lineage_private_results_and_one_shot(prepared):
    data, args = prepared
    fake = helpers.FakeRuntime(data["development"])
    def loader(config, engine):
        header = json.loads((args.output_root / "run.json").read_bytes())
        assert header["planned_cells"] == 32 and header["reference_settings"]["quantization"] is None
        assert header["nf4_numeric_records_sha256"] == audit.sha((args.output_root / "nf4-records.json").read_bytes())
        assert header["nf4_diagnostics_sha256"] == audit.sha((args.output_root / "nf4-diagnostics.json").read_bytes())
        assert header["frozen_source_hashes"]
        return fake
    summary = audit.run(args, source_loader=lambda source: (rt, tasks), model_loader=loader)
    assert summary["status"] == "complete" and fake.calls == 32
    assert summary["paired_improved"] == summary["reference_strict_correct"] == 32
    assert summary["paired_worsened"] == 0
    assert summary["both_selector_pairs"]["cpu_fp32"]["both_correct"] == 16
    assert all(v["cpu_fp32"]["planned"] == 8 for v in summary["by_presentation_and_selector"].values())
    assert summary["paired_accuracy_difference_bounds_pp"] == [100, 100]
    assert summary["heldout_allowed"] is summary["a164_gate_reopened"] is False
    result = json.loads(next(args.output_root.glob("trials/*/result.json")).read_bytes())
    assert result["schema_version"] == "cpu-reference-result-v1"
    assert result["record"]["schema_version"] == audit.CELL_SCHEMA
    with pytest.raises(FileExistsError):
        audit.run(args, source_loader=lambda source: (rt, tasks), model_loader=loader)
    assert fake.calls == 32


def test_main_captures_text_loader_progress_privately(prepared, monkeypatch, capsys):
    data, args = prepared
    original_run = audit.run
    fake = helpers.FakeRuntime(data["development"])
    def loader(config, engine):
        print("private synthetic loader output")
        sys.stderr.write("\rprivate synthetic loading progress 1/1 \u2588\n")
        sys.stderr.flush()
        return fake
    monkeypatch.setattr(audit, "run", lambda value: original_run(value,
        source_loader=lambda source: (rt, tasks), model_loader=loader))
    argv = []
    for key, value in vars(args).items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert audit.main(argv) == 0
    assert fake.calls == 32
    log = (args.output_root / "execution.log").read_text(encoding="utf-8")
    assert "private synthetic loader output" in log and "loading progress 1/1 \u2588" in log
    captured = capsys.readouterr()
    assert "private synthetic" not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "complete"
    assert (args.output_root / "execution.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("change", ["heldout", "plan_hash", "protocol_hash", "weight_hash", "memory"])
def test_preflight_failures_before_model_load(prepared, monkeypatch, change):
    data, args = prepared
    if change == "heldout":
        args.plan = data["heldout_path"]
        args.plan_sha256 = rt.engine.file_digest(args.plan)
    elif change == "plan_hash":
        args.plan_sha256 = "0" * 64
    elif change == "protocol_hash":
        args.protocol_sha256 = "0" * 64
    elif change == "weight_hash":
        monkeypatch.setattr(rt.engine, "snapshot_manifest", lambda path: {})
    else:
        monkeypatch.setattr(audit, "require_memory", lambda: (_ for _ in ()).throw(RuntimeError("insufficient")))
    with pytest.raises((ValueError, RuntimeError)):
        audit.run(args, source_loader=lambda source: (rt, tasks),
                  model_loader=lambda *args: pytest.fail("model loaded after invalid preflight"))


@pytest.mark.parametrize("mode", ["cap", "infrastructure"])
def test_cap_failure_and_infrastructure_preserved_without_retries(prepared, mode):
    data, args = prepared
    fake = helpers.FakeRuntime(data["development"], mode)
    result = audit.run(args, source_loader=lambda source: (rt, tasks), model_loader=lambda *args: fake)
    assert fake.calls == 32 and result["reference_strict_correct"] == 0
    assert len(list(args.output_root.glob("trials/*/attempt.json"))) == 32
    if mode == "cap":
        assert result["reference_caps"] == 32 and result["reference_completed"] == 32
        assert result["reference_accuracy_bounds_pp"] == [0, 0]
    else:
        assert result["reference_infrastructure_failed"] == 32 and result["reference_completed"] == 0
        assert result["reference_accuracy_bounds_pp"] == [0, 100]
        assert result["paired_accuracy_difference_pp"] is None


@pytest.mark.parametrize("kind", ["keyboard", "sigterm"])
def test_interruption_preserves_prior_results_and_missing_bounds(prepared, kind):
    data, args = prepared
    fake = helpers.FakeRuntime(data["development"])
    original = fake.generate
    def generate(value):
        if fake.calls == 1:
            if kind == "sigterm":
                audit.interrupt(audit.signal.SIGTERM, None)
            raise KeyboardInterrupt()
        return original(value)
    fake.generate = generate
    result = audit.run(args, source_loader=lambda source: (rt, tasks), model_loader=lambda *args: fake)
    assert result["status"] == "interrupted" and result["reference_completed"] == 1
    assert result["reference_missing"] == 31
    assert result["paired_accuracy_difference_bounds_pp"] == [3.125, 100]
    assert len(list(args.output_root.glob("trials/*/attempt.json"))) == 2
    assert len(list(args.output_root.glob("trials/*/result.json"))) == 1


@pytest.mark.parametrize("response,wanted", [('{"answer":"A"}', True), ('{"answer":"B"}', True),
    ('{"answer":"s03"}', False), ('{"answer":"A","extra":1}', False), ('invalid', False),
    ('{"answer":[]}', False), ('{"answer":"A","answer":"B"}', False)])
def test_label_return_diagnostics_are_numeric_only(response, wanted):
    assert audit.diagnose({"response_text": response, "generated_token_ids": [1]}, tasks) == {
        "literal_selector_label": wanted, "generated_token_count": 1}


def test_memory_threshold_reads_memavailable_and_fails_closed(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree: 1 kB\nMemAvailable: 50331648 kB\n")
    assert audit.memory_available(meminfo) == audit.MIN_AVAILABLE_BYTES
    monkeypatch.setattr(audit, "memory_available", lambda: audit.MIN_AVAILABLE_BYTES - 1)
    with pytest.raises(RuntimeError, match="insufficient"):
        audit.require_memory()


@pytest.mark.parametrize("bad", [None, "quantized", "non_fp32", "gpu_parameter", "bnb_module", "cuda_initialized", "original_quantized"])
def test_cpu_loader_explicit_unquantized_original_fp32_cpu(prepared, monkeypatch, bad):
    data, _ = prepared
    calls = []
    parameter = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda" if bad == "gpu_parameter" else "cpu"),
                                      dtype="bf16" if bad == "non_fp32" else "float32")
    class Model:
        is_loaded_in_4bit = bad == "quantized"
        is_loaded_in_8bit = False
        def eval(self):
            return self
        def parameters(self):
            return [parameter]
        def modules(self):
            return [type("Bad", (), {"__module__": "bitsandbytes.nn"})()] if bad == "bnb_module" else [self]
    def model_from_pretrained(path, **kwargs):
        calls.append(kwargs)
        return Model()
    fake_torch = types.SimpleNamespace(float32="float32", set_num_threads=lambda n: calls.append(("threads", n)),
        random=types.SimpleNamespace(default_generator=types.SimpleNamespace(manual_seed=lambda n: calls.append(("seed", n)))),
        use_deterministic_algorithms=lambda x: None,
        cuda=types.SimpleNamespace(is_initialized=lambda: bad == "cuda_initialized"))
    fake_transformers = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(
            quantization_config={} if bad == "original_quantized" else None)),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: helpers.Tokenizer()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    engine = types.SimpleNamespace(runtime_versions=lambda: data["config"]["runtime_versions"],
                                   digest=rt.engine.digest, LocalRuntime=lambda *args: "local-runtime")
    if bad:
        with pytest.raises(ValueError):
            audit.load_cpu_reference(data["config"], engine)
    else:
        assert audit.load_cpu_reference(data["config"], engine) == "local-runtime"
        kwargs = next(value for value in calls if isinstance(value, dict))
        assert kwargs == {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                          "dtype": "float32", "device_map": {"": "cpu"}, "attn_implementation": "sdpa"}
        assert ("threads", 4) in calls and ("seed", data["config"]["seed"]) in calls


def test_comparison_rejects_duplicate_or_unrelated_numeric_rows(prepared):
    data, args = prepared
    values = audit.prepare_inputs(args, rt, tasks)
    nf4, diagnostics = values[2:4]
    rows = copy.deepcopy(nf4)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="reference_comparison_identity"):
        audit.summarize(data["development"], nf4, rows, diagnostics, diagnostics)
