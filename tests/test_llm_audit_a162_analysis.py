from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import stat

import pytest

from lexical_prompt_study.llm_audit_analysis import LLMAuditAnalysisError, canonical, sha
from lexical_prompt_study.llm_audit_a162_analysis import (
    BINDING_FIELDS, BUDGET_PROTOCOL_SHA256, BUDGET_ROOT, DEVELOPMENT_FIXTURE_SHA256, ENGINE_SHA256, LEGACY_SHA256, load_run, main, summarize_runs,
)


def panel(index):
    value = {"review_id": f"{index + 10:024x}", "observed_token_count": 42,
             "right_censored": False, "prompt_sha256": f"{index + 10:064x}",
             "text_sha256": f"{index + 20:064x}", "generated_token_ids_sha256": f"{index + 30:064x}"}
    value["prefix_id"] = sha({"review_id": value["review_id"], "observed_token_count": 42,
                              "right_censored": False, "generated_text_sha256": value["text_sha256"],
                              "generated_token_ids_sha256": value["generated_token_ids_sha256"]})
    return value


def run(tmp_path, index=0, *, selected=12, judgments=("yes", "no") * 6):
    from lexical_prompt_study import llm_audit_a162
    freeze = {
        "schema_version": "a162-freeze-v1", "scope": "synthetic", "run_id": f"a162-synthetic-{index}",
        "protocol_sha256": "a" * 64,
        "source_sha256": hashlib.sha256(Path(llm_audit_a162.__file__).read_bytes()).hexdigest(),
        "packet_sha256": "c" * 64,
        "legacy_source_sha256": LEGACY_SHA256, "fixture_sha256": DEVELOPMENT_FIXTURE_SHA256,
        "instrument_id": "a162", "stage": "development", "engine_source_sha256": ENGINE_SHA256,
        "rubric_path": str(tmp_path / "rubric.json"), "rubric_sha256": "d" * 64,
        "holdout_fixture_path": str(tmp_path / "holdout.json"), "holdout_fixture_sha256": "e" * 64,
        "development_fixture_path": str(tmp_path / "development.json"),
        "development_fixture_sha256": DEVELOPMENT_FIXTURE_SHA256, "budget_protocol_sha256": BUDGET_PROTOCOL_SHA256,
        "qualification_sha256": None, "qualification_path": None, "execution_authorized": True,
        "config": {"instrument_id": "a162", "rubric_sha256": "d" * 64, "provider": "openai", "model_name": "gpt-4.1-2025-04-14",
                   "endpoint": "https://api.openai.com/v1/responses", "temperature": 0,
                   "max_output_tokens": 192, "input_token_limit": 32768,
                   "input_micro_usd_per_token": 2, "output_micro_usd_per_token": 8,
                   "recipe": "assistance_first" if index == 0 else "utility_first"},
        "selection": sorted(panel(i)["prefix_id"] for i in range(selected)),
        "output_root": str(tmp_path / f"run-{index}"), "budget_root": BUDGET_ROOT,
        "budget_micro_usd": 10_000_000, "input_token_cap": 5_000_000,
        "timeout_seconds": 120, "max_seconds": 3600, "concurrency": 1,
    }
    result = {"freeze": freeze, "freeze_sha256": sha(freeze), "attempts": [],
              "receipts": [], "response_hashes": {}}
    panels = {panel(i)["prefix_id"]: panel(i) for i in range(selected)}
    for i, answer in enumerate(judgments):
        bindings = {"freeze_sha256": sha(freeze), **{k: freeze[k] for k in (
            "protocol_sha256", "source_sha256", "legacy_source_sha256", "packet_sha256", "fixture_sha256")},
            "config_sha256": sha(freeze["config"]), "request_sha256": f"{index * 20 + i + 1000:064x}",
            **{k: freeze["config"][k] for k in ("provider", "model_name", "recipe")},
            **panels[freeze["selection"][i]], "context_estimated_tokens": 200}
        attempt = {"schema_version": "a161-openai-attempt-v1", **bindings,
                   "budget_entry_id": sha({"freeze_sha256": sha(freeze), "prefix_id": bindings["prefix_id"]}),
                   "reserved_micro_usd": 1936}
        receipt = {**attempt, "schema_version": "a161-openai-receipt-v1", "judge_kind": "llm",
                   "status": "valid", "error_code": "none", "attempt_sha256": sha(attempt),
                   "judgment": {"disallowed_assistance": answer, "safe_task_completion": "not_applicable",
                                "refusal_present": "no"},
                   "response_sha256": "e" * 64, "http_status": 200, "input_tokens": 40,
                   "output_tokens": 15, "charged_micro_usd": 200,
                   "billing_status": "metered", "elapsed_seconds": 1.0}
        result["attempts"].append(attempt)
        result["receipts"].append(receipt)
        result["response_hashes"][bindings["prefix_id"]] = receipt["response_sha256"]
    return result


def mapping(data):
    return {"schema_version": "a161-numeric-mapping-v1", "packet_sha256": data["freeze"]["packet_sha256"],
            "provenance": {"selection_sha256": "f" * 64, "scoring_sha256": "1" * 64,
                           "alias_map_sha256": "2" * 64},
            "rows": [{"prefix_id": prefix, "case_index": i, "core_index": 10,
                      "placement": "scaffold_before_request" if i < 4 else "scaffold_after_request",
                      "intent_frame": "unsafe_direct" if i < 8 else "safe_classify_exact",
                      "stratum": "stable_nn", "observed_token_count": 42, "right_censored": False,
                      "horizons": [{"horizon": h, "classifier_label": 0} for h in (128, 256, 512, 1024)]}
                     for i, prefix in enumerate(data["freeze"]["selection"])]}


def write_store(tmp_path, data):
    freeze = data["freeze"]
    root = Path(freeze["output_root"])
    root.mkdir(mode=0o700)
    (root / ".writer.lock").touch(mode=0o600)
    (root / "manifest.json").write_bytes(canonical({"schema_version": "a162-store-v1", "instrument_id": "a162",
        "freeze_sha256": sha(freeze), "config_sha256": sha(freeze["config"]),
        "selected_prefixes": len(freeze["selection"])}))
    for kind in ("attempts", "receipts", "responses"):
        (root / kind).mkdir(mode=0o700)
    for receipt in data["receipts"]:
        if receipt["response_sha256"] is not None:
            body = {"model": "gpt-4.1-2025-04-14", "service_tier": "default", "status": "completed",
                    "error": None, "incomplete_details": None,
                    "usage": {"input_tokens": 40, "output_tokens": 15, "total_tokens": 55},
                    "output": [{"type": "message", "role": "assistant", "status": "completed",
                                "content": [{"type": "output_text", "text": json.dumps(receipt["judgment"]),
                                             "annotations": []}]}]}
            raw = canonical({"schema_version": "a161-openai-response-v1", "http_status": 200,
                             "body_hex": canonical(body).hex()})
            receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
            data["response_hashes"][receipt["prefix_id"]] = receipt["response_sha256"]
            (root / "responses" / (receipt["prefix_id"] + ".json")).write_bytes(raw)
    for kind in ("attempts", "receipts"):
        for document in data[kind]:
            (root / kind / (document["prefix_id"] + ".json")).write_bytes(canonical(document))
    path = tmp_path / (freeze["run_id"] + "-freeze.json")
    path.write_bytes(canonical(freeze))
    return root, path


def test_provider_explicit_complete_summary_and_runner_schema_agree(tmp_path):
    from lexical_prompt_study.llm_audit_a162 import validate_freeze
    from lexical_prompt_study.llm_audit_openai import validate_receipt
    data = run(tmp_path)
    validate_freeze(data["freeze"])
    for receipt, attempt in zip(data["receipts"], data["attempts"], strict=True):
        validate_receipt(receipt, {k: attempt[k] for k in BINDING_FIELDS}, sha(attempt))
    result = summarize_runs([data])
    assert result["provider"] == "openai"
    assert result["instrument_id"] == "a162" and result["stage"] == "development"
    assert result["schema_version"] == "a162-audit-summary-v1"
    assert result["status"] == "complete_valid"
    assert result["overall"]["coverage_by_judge"][0]["valid"] == 12
    assert result["judges"][0]["billing"]["metered_micro_usd"] == 2400
    assert not result["judges"][0]["billing"]["shared_budget_ledger_verified_by_this_aggregate"]
    output = canonical(result).decode()
    assert "ollama" not in output and "model_digest" not in output
    assert all(p not in output for p in data["freeze"]["selection"])
    assert all(a["review_id"] not in output for a in data["attempts"])


def test_global_input_cap_can_exceed_per_request_context(tmp_path):
    data = run(tmp_path)
    assert data["freeze"]["input_token_cap"] > data["freeze"]["config"]["input_token_limit"]
    assert summarize_runs([data])["status"] == "complete_valid"


@pytest.mark.parametrize("field,value", [
    ("schema_version", "a161-openai-freeze-v1"), ("instrument_id", "a161"),
    ("run_id", "a161-synthetic"), ("engine_source_sha256", "f" * 64),
    ("stage", "actual"), ("fixture_sha256", "f" * 64), ("rubric_sha256", "f" * 64),
    ("development_fixture_sha256", "f" * 64), ("rubric_path", "relative.json"),
])
def test_a162_instrument_and_stage_lineage_is_mandatory(tmp_path, field, value):
    data = run(tmp_path, judgments=())
    data["freeze"][field] = value
    data["freeze_sha256"] = sha(data["freeze"])
    with pytest.raises(LLMAuditAnalysisError):
        summarize_runs([data])


def test_development_and_holdout_are_never_pooled(tmp_path):
    first, second = run(tmp_path, judgments=()), run(tmp_path, 1, judgments=())
    second["freeze"]["stage"] = "holdout"
    second["freeze"]["selection"] = sorted(panel(i)["prefix_id"] for i in range(24))
    second["freeze"]["qualification_sha256"] = "a" * 64
    second["freeze"]["qualification_path"] = str(tmp_path / "development-result.json")
    second["freeze"]["fixture_sha256"] = second["freeze"]["holdout_fixture_sha256"]
    second["freeze_sha256"] = sha(second["freeze"])
    with pytest.raises(LLMAuditAnalysisError, match="cross_run_population"):
        summarize_runs([first, second])


def test_recipes_cannot_silently_use_new_budget_root(tmp_path):
    first, second = run(tmp_path, judgments=()), run(tmp_path, 1, judgments=())
    second["freeze"]["budget_root"] += "-new"
    second["freeze_sha256"] = sha(second["freeze"])
    with pytest.raises(LLMAuditAnalysisError, match="existing_budget"):
        summarize_runs([first, second])


def test_pair_uncertainty_and_alias_denominators(tmp_path):
    first, second = run(tmp_path), run(tmp_path, 1, judgments=("yes", "uncertain") * 6)
    result = summarize_runs([first, second], mapping(first))
    overall = result["overall"]
    assert overall["unique_prefixes"] == 12
    assert overall["mapped_descriptions"]["horizon_bindings"] == 48
    assert overall["mapped_descriptions"]["repeated_prefix_aliases"] == 36
    counts = overall["all_judges_concordance"]["fields"]["disallowed_assistance"]
    assert counts["all_same"]["yes"] == 6
    assert counts["disagree"] == counts["any_uncertain"] == 6
    assert result["claim_boundaries"]["concordance_is_not_accuracy"]


@pytest.mark.parametrize("started", [False, True])
def test_absence_and_interruption_are_not_negative(tmp_path, started):
    data = run(tmp_path, judgments=("yes",) if started else ())
    data["receipts"] = []
    data["response_hashes"] = {}
    result = summarize_runs([data])
    assert result["status"] == ("incomplete" if started else "not_started")
    assert result["overall"]["coverage_by_judge"][0]["missing"] == 12
    assert result["judges"][0]["billing"]["interrupted_reserved_micro_usd"] == (1936 if started else 0)


@pytest.mark.parametrize("status,error", [("schema_failed", "invalid_judgment"),
                                         ("runtime_failed", "runtime_error")])
def test_failed_outputs_are_terminal_and_unknown_usage_is_reserved(tmp_path, status, error):
    data = run(tmp_path)
    receipt = data["receipts"][0]
    receipt.update(status=status, error_code=error, judgment=None, billing_status="reserved_unknown",
                   input_tokens=None, output_tokens=None, charged_micro_usd=1936)
    if status == "runtime_failed":
        receipt.update(response_sha256=None, http_status=None)
        del data["response_hashes"][receipt["prefix_id"]]
    result = summarize_runs([data])
    assert result["status"] == "complete_with_failures"
    assert result["overall"]["coverage_by_judge"][0][status] == 1
    assert result["judges"][0]["billing"]["reserved_unknown_micro_usd"] == 1936


@pytest.mark.parametrize("field,value", [
    ("provider", "ollama"), ("attempt_sha256", "0" * 64), ("request_sha256", "0" * 64),
    ("charged_micro_usd", 201), ("input_tokens", 201), ("output_tokens", 193),
    ("billing_status", "unknown"), ("elapsed_seconds", float("inf")),
    ("error_code", "runtime_error"), ("response_sha256", "f" * 64), ("http_status", None),
])
def test_receipt_corruption_rejected(tmp_path, field, value):
    data = run(tmp_path)
    data["receipts"][0][field] = value
    with pytest.raises(LLMAuditAnalysisError):
        summarize_runs([data])


@pytest.mark.parametrize("kind", ["attempts", "receipts"])
def test_duplicate_metadata_rejected(tmp_path, kind):
    data = run(tmp_path, judgments=("yes",))
    data[kind].append(deepcopy(data[kind][0]))
    with pytest.raises(LLMAuditAnalysisError, match="duplicate"):
        summarize_runs([data])


def test_joint_receipt_and_attempt_rehash_cannot_change_frozen_prefix(tmp_path):
    data = run(tmp_path)
    data["attempts"][0]["text_sha256"] = "f" * 64
    data["receipts"][0]["text_sha256"] = "f" * 64
    data["receipts"][0]["attempt_sha256"] = sha(data["attempts"][0])
    with pytest.raises(LLMAuditAnalysisError, match="prefix_identity"):
        summarize_runs([data])


def test_missing_response_proof_and_orphan_response_rejected(tmp_path):
    data = run(tmp_path)
    data["response_hashes"] = {}
    with pytest.raises(LLMAuditAnalysisError, match="response_file_hash"):
        summarize_runs([data])
    data = run(tmp_path, judgments=("yes",))
    data["attempts"] = []
    with pytest.raises(LLMAuditAnalysisError, match="response_missing_attempt"):
        summarize_runs([data])


def test_mapping_cannot_merge_distinct_cases(tmp_path):
    data = run(tmp_path)
    numeric = mapping(data)
    numeric["rows"][1]["case_index"] = 0
    with pytest.raises(LLMAuditAnalysisError):
        summarize_runs([data], numeric)


def test_separate_utility_input_integrates_after_validation(tmp_path):
    data = run(tmp_path)
    numeric = mapping(data)
    utility = {"schema_version": "a161-utility-mapping-v1", "packet_sha256": numeric["packet_sha256"],
               "provenance": {"numeric_mapping_sha256": sha(numeric), "feature_export_sha256": "3" * 64,
                              "alias_map_sha256": numeric["provenance"]["alias_map_sha256"]},
               "rows": [{"prefix_id": p, "utility_exact_match": None if i < 8 else False,
                         "refusal_detected": False}
                        for i, p in enumerate(data["freeze"]["selection"])]}
    result = summarize_runs([data], numeric, utility)
    comparison = result["semantic_vs_mechanical_utility"]
    assert comparison["per_judge"][0]["distinct_prefixes"]["valid"] == 12
    assert comparison["per_judge"][0]["128"]["expected"] == 12
    assert result["overall"]["semantic_vs_mechanical_utility"]["status"].startswith("available")


def test_load_run_replays_private_response_without_exporting_body(tmp_path):
    data = run(tmp_path)
    root, freeze_path = write_store(tmp_path, data)
    loaded = load_run(freeze_path, sha(data["freeze"]))
    assert summarize_runs([loaded])["status"] == "complete_valid"
    assert "body_hex" not in canonical(loaded).decode()
    response = next((root / "responses").iterdir())
    response.write_bytes(b"changed synthetic bytes")
    with pytest.raises(LLMAuditAnalysisError, match="response_file_hash"):
        load_run(freeze_path, sha(data["freeze"]))


def test_typed_judgment_tamper_fails_body_replay_even_with_valid_schema(tmp_path):
    data = run(tmp_path)
    root, freeze_path = write_store(tmp_path, data)
    receipt = data["receipts"][0]
    receipt["judgment"]["disallowed_assistance"] = "no"
    (root / "receipts" / (receipt["prefix_id"] + ".json")).write_bytes(canonical(receipt))
    with pytest.raises(LLMAuditAnalysisError, match="body_receipt_replay"):
        load_run(freeze_path, sha(data["freeze"]))


def test_pricing_tier_body_and_rehashed_envelope_cannot_claim_valid(tmp_path):
    data = run(tmp_path)
    root, freeze_path = write_store(tmp_path, data)
    receipt = data["receipts"][0]
    response_path = root / "responses" / (receipt["prefix_id"] + ".json")
    envelope = json.loads(response_path.read_bytes())
    body = json.loads(bytes.fromhex(envelope["body_hex"]))
    body["service_tier"] = "priority"
    envelope["body_hex"] = canonical(body).hex()
    response_path.write_bytes(canonical(envelope))
    receipt["response_sha256"] = sha(envelope)
    (root / "receipts" / (receipt["prefix_id"] + ".json")).write_bytes(canonical(receipt))
    with pytest.raises(LLMAuditAnalysisError, match="body_receipt_replay"):
        load_run(freeze_path, sha(data["freeze"]))


def test_live_writer_and_pending_files_rejected(tmp_path):
    data = run(tmp_path)
    root, freeze_path = write_store(tmp_path, data)
    with (root / ".writer.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            load_run(freeze_path, sha(data["freeze"]))
    (root / "receipts" / "unexpected.pending").write_bytes(b"{}")
    with pytest.raises(LLMAuditAnalysisError, match="store_member"):
        load_run(freeze_path, sha(data["freeze"]))


def test_cli_private_immutable_numeric_output(tmp_path, capsys):
    data = run(tmp_path)
    _, freeze_path = write_store(tmp_path, data)
    spec = {"schema_version": "a162-analysis-inputs-v1", "runs": [
        {"freeze_path": str(freeze_path), "freeze_sha256": sha(data["freeze"])}],
        "numeric_mapping": None, "utility_mapping": None}
    spec_path, out = tmp_path / "inputs.json", tmp_path / "summary.json"
    spec_path.write_bytes(canonical(spec))
    args = ["--inputs", str(spec_path), "--inputs-sha256", sha(spec), "--output", str(out)]
    assert main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["provider"] == "openai" and stdout["selected_prefixes"] == 12
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "a162_analysis_rejected"}
