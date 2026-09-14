from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import stat

import pytest

from lexical_prompt_study.llm_audit_analysis import (
    BINDING_KEYS,
    LLMAuditAnalysisError,
    canonical,
    load_run,
    main,
    sha,
    summarize_runs,
    validate_receipt,
)


def panel_metadata(index):
    value = {"review_id": f"{index + 10:024x}", "observed_token_count": 42,
             "right_censored": False, "prompt_sha256": f"{index + 10:064x}",
             "text_sha256": f"{index + 20:064x}", "generated_token_ids_sha256": f"{index + 30:064x}"}
    value["prefix_id"] = sha({"review_id": value["review_id"], "observed_token_count": 42,
                              "right_censored": False, "generated_text_sha256": value["text_sha256"],
                              "generated_token_ids_sha256": value["generated_token_ids_sha256"]})
    return value


def freeze(tmp_path, index=0, *, selected=2):
    return {
        "schema_version": "a161-freeze-v1", "scope": "synthetic", "run_id": f"synthetic-{index}",
        "protocol_sha256": "a" * 64, "source_sha256": "b" * 64, "packet_sha256": "c" * 64,
        "qualification_sha256": None, "execution_authorized": True,
        "config": {"model_name": "synthetic-model:1b", "model_digest": "d" * 64,
                   "ollama_version": "0.18.0", "endpoint": "http://127.0.0.1:11434",
                   "seed": 20260914, "num_ctx": 8192, "num_predict": 96, "num_thread": 2,
                   "num_gpu": 0, "keep_alive": 0, "timeout_seconds": 60, "max_seconds": 600,
                   "recipe": "assistance_first" if index == 0 else "utility_first"},
        "selection": sorted(panel_metadata(i)["prefix_id"] for i in range(selected)),
        "output_root": str(tmp_path / f"run-{index}"),
    }


def run(tmp_path, index=0, *, selected=2, judgments=("yes", "no"), statuses=None):
    f = freeze(tmp_path, index, selected=selected)
    result = {"freeze": f, "freeze_sha256": sha(f), "receipts": [], "attempts": []}
    panels = {panel_metadata(i)["prefix_id"]: panel_metadata(i) for i in range(selected)}
    for i, value in enumerate(judgments):
        bindings = {
            "freeze_sha256": sha(f), "protocol_sha256": f["protocol_sha256"],
            "source_sha256": f["source_sha256"], "packet_sha256": f["packet_sha256"],
            "config_sha256": sha(f["config"]), "request_sha256": f"{1000 + index * 20 + i:064x}",
            "model_name": f["config"]["model_name"], "model_digest": f["config"]["model_digest"],
            "recipe": f["config"]["recipe"], **panels[f["selection"][i]],
            "context_estimated_tokens": 100,
        }
        attempt = {"schema_version": "a161-attempt-v1", **bindings}
        status = statuses[i] if statuses else "valid"
        error = {"valid": "none", "schema_failed": "invalid_judgment",
                 "runtime_failed": "runtime_error", "context_rejected": "context_budget_exceeded"}[status]
        receipt = {
            "schema_version": "a161-receipt-v1", "judge_kind": "llm", **bindings,
            "attempt_sha256": sha(attempt), "status": status, "error_code": error,
            "judgment": {"disallowed_assistance": value, "safe_task_completion": "not_applicable",
                         "refusal_present": "no"} if status == "valid" else None,
            "response_sha256": "e" * 64 if status == "valid" else None,
            "prompt_eval_count": 90 if status == "valid" else None,
            "eval_count": 20 if status == "valid" else None, "elapsed_seconds": 1.0,
        }
        result["attempts"].append(attempt)
        result["receipts"].append(receipt)
    return result


def mapping(f):
    return {
        "schema_version": "a161-numeric-mapping-v1", "packet_sha256": f["packet_sha256"],
        "provenance": {"selection_sha256": "f" * 64, "scoring_sha256": "1" * 64,
                       "alias_map_sha256": "2" * 64},
        "rows": [{"prefix_id": prefix, "case_index": i, "core_index": 10,
                  "placement": "scaffold_before_request", "intent_frame": "unsafe_direct",
                  "stratum": "stable_nn", "observed_token_count": 42, "right_censored": False,
                  "horizons": [{"horizon": h, "classifier_label": 0} for h in (128, 256, 512, 1024)]}
                 for i, prefix in enumerate(f["selection"])],
    }


def write_store(tmp_path, data):
    f = data["freeze"]
    root = tmp_path / f["run_id"]
    root = type(root)(f["output_root"])
    root.mkdir(mode=0o700)
    (root / ".writer.lock").touch(mode=0o600)
    (root / "manifest.json").write_bytes(canonical({
        "schema_version": "a161-store-v1", "freeze_sha256": sha(f),
        "config_sha256": sha(f["config"]), "selected_prefixes": len(f["selection"]),
    }))
    for kind in ("attempts", "receipts"):
        (root / kind).mkdir(mode=0o700)
        for document in data[kind]:
            (root / kind / (document["prefix_id"] + ".json")).write_bytes(canonical(document))
    freeze_path = tmp_path / (f["run_id"] + "-freeze.json")
    freeze_path.write_bytes(canonical(f))
    return root, freeze_path


def test_complete_single_judge_has_no_false_interjudge_agreement(tmp_path):
    data = run(tmp_path)
    result = summarize_runs([data])
    assert result["status"] == "complete_valid"
    assert result["judge_kind"] == "llm"
    assert result["judges"][0]["recipe"] == "assistance_first"
    overall = result["overall"]
    assert overall["coverage_by_judge"][0]["valid"] == 2
    assert overall["paired_concordance"] == []
    assert overall["all_judges_concordance"]["available"] is False
    assert overall["mapped_descriptions"]["status"] == "unavailable_no_numeric_mapping"
    assert "by_placement" not in result
    assert result["claim_boundaries"]["model_judgments_are_not_human_ratings_or_ground_truth"]


def test_missing_and_interrupted_are_not_zero_negative_judgments(tmp_path):
    data = run(tmp_path, selected=3)
    data["receipts"].pop()
    result = summarize_runs([data])
    assert result["status"] == "incomplete"
    assert result["overall"]["coverage_by_judge"][0]["missing"] == 2
    assert result["overall"]["coverage_by_judge"][0]["judgments"]["disallowed_assistance"]["no"] == 0
    assert result["judges"][0]["interrupted_attempts_without_receipt"] == 1
    assert result["judges"][0]["unattempted_prefixes"] == 1


def test_not_started_status_keeps_denominators(tmp_path):
    result = summarize_runs([run(tmp_path, selected=3, judgments=())])
    assert result["status"] == "not_started"
    assert result["overall"]["coverage_by_judge"][0]["expected"] == 3
    assert result["overall"]["coverage_by_judge"][0]["valid"] == 0


def test_attempt_without_any_receipt_is_incomplete_not_not_started(tmp_path):
    data = run(tmp_path)
    data["receipts"] = []
    result = summarize_runs([data])
    assert result["status"] == "incomplete"
    assert result["judges"][0]["interrupted_attempts_without_receipt"] == 2


def test_jointly_changed_attempt_and_receipt_cannot_change_frozen_prefix(tmp_path):
    data = run(tmp_path)
    data["receipts"][0]["text_sha256"] = "f" * 64
    data["attempts"][0]["text_sha256"] = "f" * 64
    data["receipts"][0]["attempt_sha256"] = sha(data["attempts"][0])
    with pytest.raises(LLMAuditAnalysisError, match="prefix_identity"):
        summarize_runs([data])


def test_valid_receipt_cannot_claim_overflowed_context(tmp_path):
    data = run(tmp_path)
    data["receipts"][0]["context_estimated_tokens"] = 8192
    data["attempts"][0]["context_estimated_tokens"] = 8192
    data["receipts"][0]["attempt_sha256"] = sha(data["attempts"][0])
    with pytest.raises(LLMAuditAnalysisError, match="valid_context_bounds"):
        summarize_runs([data])


def test_terminal_failures_separate_from_missing(tmp_path):
    data = run(tmp_path, selected=3, judgments=("no", "no", "no"),
               statuses=("schema_failed", "runtime_failed", "context_rejected"))
    result = summarize_runs([data])
    assert result["status"] == "complete_with_failures"
    counts = result["overall"]["coverage_by_judge"][0]
    assert counts["schema_failed"] == counts["runtime_failed"] == counts["context_rejected"] == 1
    assert counts["missing"] == counts["valid"] == 0
    assert all(v == 0 for v in counts["judgments"]["disallowed_assistance"].values())


def test_pairwise_three_field_uncertainty_and_all_judges_concordance(tmp_path):
    left = run(tmp_path, selected=3, judgments=("yes", "no", "uncertain"))
    right = run(tmp_path, 1, selected=3, judgments=("yes", "yes", "uncertain"))
    right["receipts"][0]["judgment"]["refusal_present"] = "yes"
    result = summarize_runs([left, right])["overall"]
    pair = result["paired_concordance"][0]["counts"]
    assert pair["both_valid"] == 3
    assert pair["fields"]["disallowed_assistance"]["agree"] == 2
    assert pair["fields"]["disallowed_assistance"]["disagree"] == 1
    assert pair["fields"]["disallowed_assistance"]["uncertainty_in_pair"] == 1
    assert pair["fields"]["refusal_present"]["disagree"] == 1
    all_judges = result["all_judges_concordance"]["fields"]["disallowed_assistance"]
    assert all_judges == {"all_same": {"yes": 1, "no": 0, "uncertain": 1},
                          "disagree": 1, "any_uncertain": 1}


def test_failed_member_excludes_pair_and_unanimity(tmp_path):
    left, right = run(tmp_path), run(tmp_path, 1, statuses=("valid", "schema_failed"))
    result = summarize_runs([left, right])["overall"]
    assert result["paired_concordance"][0]["counts"]["excluded_failed_or_missing"] == 1
    assert result["all_judges_concordance"]["excluded_failed_or_missing_prefixes"] == 1


def test_numeric_mapping_keeps_aliases_cases_cores_and_classifier_concordance(tmp_path):
    data = run(tmp_path)
    result = summarize_runs([data], mapping(data["freeze"]))
    descriptions = result["overall"]["mapped_descriptions"]
    assert descriptions["selected_cases"] == 2
    assert descriptions["distinct_request_cores"] == 1
    assert descriptions["horizon_bindings"] == 8
    assert descriptions["repeated_prefix_aliases"] == 6
    concordance = descriptions["classifier_concordance_by_judge_and_horizon"][0]
    assert all(v["binary_comparable"] == 2 and v["agree"] == v["disagree"] == 1
               for v in concordance.values())
    assert len(result["by_sampling_stratum"]) == 20
    serialized = json.dumps(result)
    assert '"prefix_id"' not in serialized and '"review_id"' not in serialized
    assert '"case_index"' not in serialized and '"core_index"' not in serialized


def test_unknown_classifier_and_uncertain_judge_not_concordance_failure(tmp_path):
    data = run(tmp_path, judgments=("uncertain", "no"))
    numeric = mapping(data["freeze"])
    numeric["rows"][0]["horizons"][0]["classifier_label"] = None
    numeric["rows"][0]["stratum"] = "unknown_or_final_capped_negative"
    result = summarize_runs([data], numeric)
    counts = result["overall"]["mapped_descriptions"]["classifier_concordance_by_judge_and_horizon"][0]["128"]
    assert counts["classifier_unknown"] == counts["judge_uncertain"] == 1
    assert counts["binary_comparable"] == counts["agree"] == 1
    assert counts["disagree"] == 0


@pytest.mark.parametrize("mutation", [
    lambda r: r["receipts"][0].update(text="RESTRICTED_SENTINEL"),
    lambda r: r["receipts"][0]["judgment"].update(rationale="RESTRICTED_SENTINEL"),
    lambda r: r["receipts"][0].update(judge_kind="human"),
    lambda r: r["receipts"][0].update(config_sha256="f" * 64),
    lambda r: r["receipts"][0].update(model_digest="f" * 64),
    lambda r: r["receipts"][0].update(recipe="utility_first"),
    lambda r: r["receipts"][0].update(prefix_id="f" * 64),
    lambda r: r["receipts"][0].update(attempt_sha256="f" * 64),
    lambda r: r["receipts"][0].update(error_code="runtime_error"),
    lambda r: r["receipts"][0].update(elapsed_seconds=float("nan")),
    lambda r: r["receipts"][0].update(observed_token_count=True),
    lambda r: r["receipts"][0].update(prompt_eval_count=None),
    lambda r: r["receipts"][0]["judgment"].update(safe_task_completion="complete"),
    lambda r: r["attempts"].pop(0),
    lambda r: r["attempts"].append(deepcopy(r["attempts"][0])),
    lambda r: r["receipts"].append(deepcopy(r["receipts"][0])),
    lambda r: r["freeze"]["config"].update(seed=True),
])
def test_strict_rejection_never_echoes_private_values(tmp_path, mutation):
    data = run(tmp_path)
    mutation(data)
    with pytest.raises(LLMAuditAnalysisError) as error:
        summarize_runs([data])
    assert "RESTRICTED_SENTINEL" not in str(error.value)


def test_cross_run_selection_and_exact_prefix_binding_rejected(tmp_path):
    left, right = run(tmp_path), run(tmp_path, 1, selected=3)
    with pytest.raises(LLMAuditAnalysisError, match="cross_judge_population"):
        summarize_runs([left, right])
    right = run(tmp_path, 1)
    right["receipts"][0]["text_sha256"] = "f" * 64
    right["attempts"][0]["text_sha256"] = "f" * 64
    right["receipts"][0]["attempt_sha256"] = sha(right["attempts"][0])
    with pytest.raises(LLMAuditAnalysisError, match="prefix_identity"):
        summarize_runs([left, right])
    with pytest.raises(LLMAuditAnalysisError, match="duplicate_run"):
        summarize_runs([left, left])


@pytest.mark.parametrize("mutation", [
    lambda m: m["rows"].pop(),
    lambda m: m["rows"].append(deepcopy(m["rows"][0])),
    lambda m: m["rows"][0].update(text="RESTRICTED_SENTINEL"),
    lambda m: m["rows"][0].update(stratum="stable_pp"),
    lambda m: m["rows"][0].update(observed_token_count=43),
    lambda m: m["rows"][0]["horizons"].pop(),
    lambda m: m["rows"][0]["horizons"][0].update(classifier_label=True),
])
def test_bad_numeric_mapping_is_rejected_not_silently_dropped(tmp_path, mutation):
    data = run(tmp_path)
    numeric = mapping(data["freeze"])
    mutation(numeric)
    with pytest.raises(LLMAuditAnalysisError):
        summarize_runs([data], numeric)


def test_native_metadata_loader_verifies_attempt_hash_and_manifest(tmp_path):
    data = run(tmp_path)
    root, freeze_path = write_store(tmp_path, data)
    loaded = load_run(freeze_path, sha(data["freeze"]))
    assert summarize_runs([loaded]) == summarize_runs([data])
    path = root / "attempts" / (data["freeze"]["selection"][0] + ".json")
    changed = json.loads(path.read_bytes())
    changed["request_sha256"] = "f" * 64
    path.write_bytes(canonical(changed))
    with pytest.raises(LLMAuditAnalysisError, match="attempt_hash"):
        load_run(freeze_path, sha(data["freeze"]))


def test_unstarted_store_missing_file_and_pending_are_distinguished(tmp_path):
    data = run(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_bytes(canonical(data["freeze"]))
    assert summarize_runs([load_run(freeze_path, sha(data["freeze"]))])["status"] == "not_started"
    root, freeze_path = write_store(tmp_path, data)
    (root / "receipts" / "unexpected.pending").write_bytes(b"{}")
    with pytest.raises(LLMAuditAnalysisError, match="store_member"):
        load_run(freeze_path, sha(data["freeze"]))


def test_active_writer_snapshot_rejected(tmp_path):
    data = run(tmp_path)
    root, freeze_path = write_store(tmp_path, data)
    with (root / ".writer.lock").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(LLMAuditAnalysisError, match="writer_active"):
            load_run(freeze_path, sha(data["freeze"]))


def test_cli_private_immutable_output_has_only_safe_stdout(tmp_path, capsys):
    data = run(tmp_path)
    _, freeze_path = write_store(tmp_path, data)
    spec = {"schema_version": "a161-analysis-inputs-v1",
            "runs": [{"freeze_path": str(freeze_path), "freeze_sha256": sha(data["freeze"])}],
            "numeric_mapping": None}
    source, output = tmp_path / "inputs.json", tmp_path / "summary.private.json"
    source.write_bytes(canonical(spec))
    args = ["--inputs", str(source), "--inputs-sha256", sha(spec), "--output", str(output)]
    assert main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert set(stdout) == {"status", "judge_configurations", "selected_prefixes", "result_sha256"}
    assert stdout["status"] == "complete_valid"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    before = output.read_bytes()
    assert hashlib.sha256(before).hexdigest() == stdout["result_sha256"]
    assert main(args) == 1
    assert capsys.readouterr().out == '{"status":"a161_analysis_rejected"}\n'
    assert output.read_bytes() == before


def test_cli_raw_input_error_does_not_echo_or_write(tmp_path, capsys):
    source = tmp_path / "RESTRICTED_PATH"
    source.write_bytes(b'{"text":"RESTRICTED_SENTINEL"}\n')
    assert main(["--inputs", str(source), "--inputs-sha256", hashlib.sha256(source.read_bytes()).hexdigest(),
                 "--output", str(tmp_path / "out.json")]) == 1
    assert capsys.readouterr().out == '{"status":"a161_analysis_rejected"}\n'


def test_runner_receipt_validator_accepts_same_synthetic_schema(tmp_path):
    from lexical_prompt_study.llm_audit_runner import validate_receipt as runner_validate
    data = run(tmp_path)
    for receipt in data["receipts"]:
        assert runner_validate(receipt, {k: receipt[k] for k in BINDING_KEYS}, receipt["attempt_sha256"])
        validate_receipt(receipt, data["freeze"], data["freeze_sha256"])
