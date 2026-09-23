"""Invented handoff and loader-guard checks; no tokenizer/model assets loaded."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from lexical_prompt_study import landmark_evidence as evidence
from lexical_prompt_study import matched_prefix_tasks as tasks

WORKER = Path(__file__).absolute().parents[1] / "scripts/matched_prefix_native_worker.py"
spec = importlib.util.spec_from_file_location("a186_invented_worker_test", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fixture():
    roster = tasks.prediction_roster(tasks.build_roster())
    semantic, internal, records, labels, embeddings = [], [], [], {}, {}
    for row in roster:
        key = row["observation_id"]
        semantic.append(
            {
                "schema_version": "a186-common-prefix-v1",
                "observation_id": key,
                "common": {
                    "rendered_prompt_utf8": "Invented é prompt".encode(),
                    "prompt_token_ids": [1, 2],
                    "prefix_utf8": "é".encode(),
                    "prefix_token_ids": list(range(8)),
                    "completed_fields": 0,
                    "prefix_known_error": False,
                },
                "features_sha256": "1" * 64,
            }
        )
        internal.append(
            {
                "schema_version": "a186-internal-prefix-v1",
                "observation_id": key,
                "residual_fp32le": struct.pack("<4096f", *([0.5] * 4096)),
                "residual_shape": [1, 4096],
                "features_sha256": "2" * 64,
            }
        )
        records.append({**row, "status": "completed", "prefix_available": True, "label": 0})
        labels[key] = 0
        embeddings[key] = {
            "prompt": struct.pack("<384f", *([0.25] * 384)),
            "prefix": struct.pack("<384f", *([-0.5] * 384)),
        }
    freeze = {
        "plan": {"invented": True},
        "fit_runtime": {"python": "invented"},
        "sources": {"src/lexical_prompt_study/matched_prefix_prediction.py": "3" * 64},
    }
    views = {"semantic_packets": semantic, "internal_packets": internal, "labels": labels}
    acquired = {
        "status": "finished",
        "records": records,
        "terminal_sha256": "4" * 64,
        "summary": {"completed_rows": 112, "total_entries": 112},
    }
    encoded = {"embeddings": embeddings, "terminal_sha256": "5" * 64, "entries": 224}
    return freeze, views, acquired, encoded


def test_exact_handoff_retains112_order_common_time_and_separate_labels():
    freeze, views, acquired, encoded = fixture()
    request = worker.assemble_fit_request(freeze, views, acquired, encoded, tasks)
    assert set(request) == {"roster", "feature_packets", "labels", "bindings", "landmark_reached"}
    assert request["roster"] == tasks.prediction_roster(tasks.build_roster())
    assert len(request["feature_packets"]) == 112
    assert all(request["landmark_reached"].values())
    for packet in request["feature_packets"]:
        assert packet["common"]["prefix_utf8"] == "é"
        assert packet["text_semantic"] == [0.25] * 384 + [-0.5] * 384
        assert packet["internal"] == [0.5] * 4096
        assert set(packet) == {
            "schema_version",
            "observation_id",
            "common",
            "text_semantic",
            "internal",
        }
        assert not {"label", "stop_reason", "future_token_ids"} & set(packet["common"])
    assert request["bindings"]["feature_packets_sha256"] == evidence.object_hash(
        request["feature_packets"]
    )
    assert worker.canonical({"unicode": "é"}) == evidence.canonical({"unicode": "é"})
    assert request["bindings"]["labels_sha256"] == evidence.object_hash(request["labels"])


def test_known_early_eos_keeps_label_but_no_fabricated_features():
    freeze, views, acquired, encoded = fixture()
    key = acquired["records"][0]["observation_id"]
    acquired["records"][0]["prefix_available"] = False
    views["semantic_packets"][0]["common"] = None
    views["internal_packets"][0].update(residual_fp32le=None, residual_shape=None)
    encoded["embeddings"][key] = {"prompt": None, "prefix": None}
    request = worker.assemble_fit_request(freeze, views, acquired, encoded, tasks)
    assert request["landmark_reached"][key] is False and request["labels"][key] == 0
    assert request["feature_packets"][0] == {
        "schema_version": "a186-prediction-features-v1",
        "observation_id": key,
        "common": None,
        "text_semantic": None,
        "internal": None,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "reorder",
        "missing",
        "false_reached",
        "future",
        "utf8",
        "boolean_label",
        "wrong_label",
        "vector_shape",
        "nonfinite",
        "short_embedding",
        "prefix_flag",
        "packet_label",
    ],
)
def test_handoff_rejects_invented_mismatches(mutation):
    freeze, views, acquired, encoded = fixture()
    key = acquired["records"][0]["observation_id"]
    if mutation == "reorder":
        views["internal_packets"].reverse()
    elif mutation == "missing":
        views["semantic_packets"].pop()
    elif mutation == "false_reached":
        acquired["records"][0]["prefix_available"] = False
    elif mutation == "future":
        views["semantic_packets"][0]["common"]["suffix"] = b"later"
    elif mutation == "utf8":
        views["semantic_packets"][0]["common"]["prefix_utf8"] = b"\xff"
    elif mutation == "boolean_label":
        views["labels"][key] = False
    elif mutation == "wrong_label":
        views["labels"][key] = 1
    elif mutation == "vector_shape":
        views["internal_packets"][0]["residual_shape"] = [4096]
    elif mutation == "nonfinite":
        views["internal_packets"][0]["residual_fp32le"] = struct.pack(
            "<4096f", *([float("nan")] * 4096)
        )
    elif mutation == "short_embedding":
        encoded["embeddings"][key]["prompt"] = b"\0" * 4
    elif mutation == "prefix_flag":
        views["semantic_packets"][0]["common"]["prefix_known_error"] = True
    else:
        views["semantic_packets"][0]["label"] = 0
    with pytest.raises((ValueError, UnicodeError)):
        worker.assemble_fit_request(freeze, views, acquired, encoded, tasks)


def test_semantic_inventory_hash_and_no_links(tmp_path):
    root = tmp_path / "semantic"
    root.mkdir()
    files = {}
    for name in worker.SEMANTIC_FILES:
        path = root / name
        path.write_bytes(("invented " + name).encode())
        files[name] = {"bytes": path.stat().st_size, "sha256": worker.digest(path)}
    spec = {"path": str(root), "files": files}
    assert worker.semantic_asset_check(spec) == files
    (root / "config.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="semantic_hash"):
        worker.semantic_asset_check(spec)
    (root / "config.json").unlink()
    (root / "config.json").symlink_to(root / "vocab.txt")
    with pytest.raises(ValueError, match="semantic_regular"):
        worker.semantic_asset_check(spec)


def test_target_binding_guard_precedes_generation_config_read(monkeypatch):
    expected = {"fixed": True}
    monkeypatch.setattr(worker, "expected_runtime_binding", lambda *_: expected)
    helper = SimpleNamespace(
        hash_bound_json=lambda *_a, **_k: pytest.fail("must not read vendor JSON")
    )
    with pytest.raises(ValueError, match="selected_runtime"):
        worker.validate_target({}, {}, {"fixed": False}, "0" * 64, helper)


def fake_dependencies(monkeypatch, tmp_path, *, retain_target=False, fail_load=False):
    freeze, views, acquired, encoded = fixture()
    freeze.update(
        output_root=str(tmp_path),
        runtime_binding={},
        target_config_path="invented-config",
        target_config_sha256="6" * 64,
        semantic_assets={"invented": True},
        encoder_runtime={},
        encoder_binding_sha256="7" * 64,
        fit_python="/invented/python",
        fit_worker="scripts/matched_prefix_fit_worker.py",
    )
    freeze["sources"]["scripts/matched_prefix_fit_worker.py"] = "8" * 64
    held = []

    class LlamaModel:
        def forward(self):
            return None

    class LlamaForCausalLM:
        def __init__(self):
            self.model = LlamaModel()

        def forward(self):
            return self.model.forward()

    class BertEncoder:
        def forward(self):
            return None

    class BertModel:
        def __init__(self):
            self.encoder = BertEncoder()

        def forward(self):
            return self.encoder.forward()

    for path, contents in {
        "transformers.models.llama.modeling_llama": {
            "LlamaModel": LlamaModel,
            "LlamaForCausalLM": LlamaForCausalLM,
        },
        "transformers.models.bert.modeling_bert": {
            "BertModel": BertModel,
            "BertEncoder": BertEncoder,
        },
    }.items():
        module = ModuleType(path)
        module.__dict__.update(contents)
        monkeypatch.setitem(sys.modules, path, module)
    torch = ModuleType("torch")
    torch.set_num_threads = lambda _: None
    torch.use_deterministic_algorithms = lambda _: None
    torch.random = SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda _: None))
    torch.cuda = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def load(*_):
        if fail_load:
            raise OSError("invented load failure")
        model = LlamaForCausalLM()
        if retain_target:
            held.append(model)
        return SimpleNamespace(model=model, tokenizer=object(), eos_ids=[9])

    def acquire(model, _tokenizer, **_kwargs):
        for _ in range(112):
            model.forward()
        return acquired

    def encode(model, _tokenizer, semantic_only, **_kwargs):
        assert semantic_only is views["semantic_packets"]
        assert all("label" not in packet and "internal" not in packet for packet in semantic_only)
        for _ in range(224):
            model.forward()
        return encoded

    core = SimpleNamespace(
        validate_plan=lambda _: {"manifest": {"eos_token_ids": [9]}},
        prepare_inputs=lambda *_: {"invented_native": True},
        run_acquisition=acquire,
        load_acquisition=lambda *_: acquired,
        feature_views=lambda *_: views,
    )
    encoder = SimpleNamespace(
        runtime_descriptor=lambda: {},
        encode_packets=encode,
        load_encoding=lambda *_a, **_k: encoded,
    )
    helper = SimpleNamespace(
        hash_bound_json=lambda *_: {"runtime_versions": {}},
        asset_check=lambda _: {"invented": True},
        memory_available=lambda: 80 * 1024**3,
    )
    monkeypatch.setattr(worker, "verify_sources", lambda _: tmp_path)
    monkeypatch.setattr(worker, "pinned_helper", lambda: helper)
    monkeypatch.setattr(
        worker,
        "frozen_imports",
        lambda *_: (evidence, SimpleNamespace(runtime_versions=lambda: {}), core, encoder, tasks),
    )
    monkeypatch.setattr(worker, "validate_target", lambda *_: None)
    monkeypatch.setattr(worker, "semantic_asset_check", lambda _: {"invented": True})
    monkeypatch.setattr(worker, "_module", lambda *_: SimpleNamespace(load_cpu_reference=load))
    monkeypatch.setattr(worker, "_load_encoder", lambda *_: (BertModel(), object()))

    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is False
            assert command[0] == "/invented/python" and command[-2:] == [
                "--parent",
                str(os.getpid()),
            ]
            self.pid = os.getpid()
            child = Path(command[command.index("--output") + 1])
            child.mkdir()
            request_path = Path(command[command.index("--request") + 1])
            req_sha = command[command.index("--request-sha256") + 1]
            assert worker.digest(request_path) == req_sha
            result = {"invented": True}
            result_sha = evidence.object_hash(result)
            header = {
                "schema_version": "a186-fit-worker-v1",
                "request_sha256": req_sha,
                "pid": self.pid,
                "parent_pid": os.getpid(),
                "fit_runtime": freeze["fit_runtime"],
                "worker_source_sha256": freeze["sources"][freeze["fit_worker"]],
                "prediction_source_sha256": freeze["sources"][
                    "src/lexical_prompt_study/matched_prefix_prediction.py"
                ],
            }
            normal = {
                "fit_predict_returned": True,
                "returned_result_sha256": result_sha,
                "request_sha256": req_sha,
                "terminal_error": None,
                "run_sha256": evidence.object_hash(header),
                "actual_fit_entries": 2,
            }
            terminal = {
                "schema_version": "a186-fit-worker-v1",
                "status": "completed",
                "request_sha256": req_sha,
                "normal_return_sha256": evidence.object_hash(normal),
                "actual_fit_entries": 2,
                "result_sha256": result_sha,
                "model_free_replay": True,
                "profile_restored": True,
                "fit_claims": 2,
                "fit_body_returns": 2,
                "event_count": 0,
                "event_sha256": [],
                "entry_sha256": [],
                "run_sha256": evidence.object_hash(header),
            }
            (child / "events").mkdir()
            (child / "entries").mkdir()
            for i in range(1, 3):
                raw = evidence.canonical({"invented_entry": i})
                entry_path = child / "entries" / f"fit_{i:02d}.json"
                entry_path.write_bytes(raw)
                terminal["entry_sha256"].append(worker.digest(entry_path))
            (child / "request.json").write_bytes(request_path.read_bytes())
            for name, value in (
                ("run.json", header),
                ("result.json", result),
                ("normal-return.json", normal),
                ("terminal.json", terminal),
            ):
                (child / name).write_bytes(evidence.canonical(value))

        def wait(self):
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", Process)
    return freeze, held


def test_invented_whole_pipeline_exact_sequential_counts_and_handoff(monkeypatch, tmp_path):
    freeze, _ = fake_dependencies(monkeypatch, tmp_path)
    result = worker.run(freeze, tmp_path / "worker")
    assert result == {
        "status": "completed",
        "body_entry_counts": {
            "causal_lm_entries": 112,
            "nested_decoder_entries": 112,
            "bert_model_entries": 224,
            "nested_bert_encoder_entries": 224,
        },
        "actual_fit_entries": 2,
    }
    assert sys.getprofile() is None
    root = tmp_path / "worker"
    assert (root / "worker-terminal.json").is_file()
    assert not (root / "worker-failure.json").exists()
    request = json.loads((root / "fit-request.json").read_bytes())
    assert len(request["roster"]) == 112 and len(request["feature_packets"]) == 112
    with pytest.raises((FileExistsError, ValueError)):
        worker.run(freeze, root)


def test_retained_target_blocks_encoder_load(monkeypatch, tmp_path):
    freeze, held = fake_dependencies(monkeypatch, tmp_path, retain_target=True)
    monkeypatch.setattr(worker, "_load_encoder", lambda *_: pytest.fail("target still alive"))
    with pytest.raises(ValueError, match="target_reference_retained"):
        worker.run(freeze, tmp_path / "worker")
    assert held
    failure = json.loads((tmp_path / "worker/worker-failure.json").read_bytes())
    assert failure["body_entry_counts"]["bert_model_entries"] == 0
    assert failure["actual_fit_entries"] == 0 and not failure["fit_subprocess_started"]
    assert sys.getprofile() is None


def test_loader_failure_retains_zero_counts_without_retry(monkeypatch, tmp_path):
    freeze, _ = fake_dependencies(monkeypatch, tmp_path, fail_load=True)
    with pytest.raises(OSError, match="invented load failure"):
        worker.run(freeze, tmp_path / "worker")
    failure = json.loads((tmp_path / "worker/worker-failure.json").read_bytes())
    assert failure["phase"] == "target_load" and not any(failure["body_entry_counts"].values())
    assert not (tmp_path / "worker/native-prepared.json").exists()
    assert sys.getprofile() is None


def test_wrong_output_fails_before_directory_claim(monkeypatch, tmp_path):
    freeze, _ = fake_dependencies(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="worker_output"):
        worker.run(freeze, tmp_path / "unbound")
    assert not (tmp_path / "unbound").exists()


def test_type_preserving_replay_comparison_with_bytes():
    assert worker.tree_hash({"raw": b"\x00", "entries": True}) != worker.tree_hash(
        {"raw": b"\x00", "entries": 1}
    )
    assert worker.tree_hash({"raw": b"\x00"}) != worker.tree_hash({"raw": b"\x01"})


@pytest.mark.parametrize(
    "mutation",
    [
        "failure_marker",
        "entry_tamper",
        "event_extra",
        "result_tamper",
        "normal_bool",
        "header_bool",
        "symlink",
    ],
)
def test_native_parent_rejects_fit_envelope_tampering(monkeypatch, tmp_path, mutation):
    freeze, _ = fake_dependencies(monkeypatch, tmp_path)
    worker.run(freeze, tmp_path / "worker")
    root = tmp_path / "worker/fit"
    request_sha = worker.digest(root / "request.json")
    if mutation == "failure_marker":
        (root / "failure.json").write_bytes(b"{}")
    elif mutation == "entry_tamper":
        (root / "entries/fit_01.json").write_bytes(b"{}")
    elif mutation == "event_extra":
        (root / "events/extra.json").write_bytes(b"{}")
    elif mutation == "result_tamper":
        (root / "result.json").write_bytes(b"{}")
    elif mutation in ("normal_bool", "header_bool"):
        path = root / ("normal-return.json" if mutation == "normal_bool" else "run.json")
        value = json.loads(path.read_bytes())
        value["actual_fit_entries" if mutation == "normal_bool" else "pid"] = True
        path.write_bytes(evidence.canonical(value))
    else:
        path = root / "entries/fit_01.json"
        path.unlink()
        path.symlink_to(root / "result.json")
    with pytest.raises(ValueError):
        worker.validate_fit_return(root, request_sha, freeze, os.getpid(), evidence)


def test_cleanup_failure_preserves_original_loader_exception(monkeypatch, tmp_path):
    freeze, _ = fake_dependencies(monkeypatch, tmp_path, fail_load=True)
    original = sys.setprofile

    def fail_restoration(value):
        original(value)
        if value is None:
            raise RuntimeError("invented cleanup failure")

    monkeypatch.setattr(worker.sys, "setprofile", fail_restoration)
    try:
        with pytest.raises(OSError, match="invented load failure"):
            worker.run(freeze, tmp_path / "worker")
    finally:
        original(None)
    receipt = json.loads((tmp_path / "worker/worker-cleanup-failure.json").read_bytes())
    assert (
        receipt["original_error_type"] == "OSError"
        and receipt["cleanup_error_type"] == "RuntimeError"
    )


def test_source_guard_rejects_absent_closure_without_import(monkeypatch, tmp_path):
    freeze = {key: None for key in worker.REQUIRED_FREEZE}
    freeze.update(
        source_root=str(tmp_path), sources={"scripts/matched_prefix_native_worker.py": "0" * 64}
    )
    monkeypatch.setattr(worker, "_module", lambda *_: pytest.fail("must reject before import"))
    with pytest.raises(ValueError, match="source_closure"):
        worker.verify_sources(freeze)
