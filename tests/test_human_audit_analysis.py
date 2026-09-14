from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import stat

import pytest

from lexical_prompt_study.human_audit_analysis import (
    HumanAuditError,
    PROVENANCE_KEYS,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
    canonical_bytes,
    main,
    summarize,
    validate_input,
)


def rating(index=0, harm="no", exposure="none", **changes):
    value = {
        "reviewer_index": index,
        "status": "rated",
        "prior_exposure": exposure,
        "answers": {
            "disallowed_assistance": harm,
            "safe_task_completion": "not_applicable",
            "refusal_present": "no",
            "earliest_assistance": "none_observed" if harm == "no" else "unresolved",
        },
    }
    value.update(changes)
    return value


def case(index=0, *, labels=(0, 0, 0, 0), ratings=None, alias=False, **changes):
    labels = dict(zip((128, 256, 512, 1024), labels, strict=True))
    prefixes = []
    if alias:
        prefixes.append({
            "prefix_index": 0,
            "observed_token_count": 42,
            "right_censored": False,
            "horizons": [{"horizon": h, "classifier_label": label} for h, label in labels.items()],
            "ratings": ratings or [],
        })
    else:
        for i, (horizon, label) in enumerate(labels.items()):
            prefixes.append({
                "prefix_index": i,
                "observed_token_count": horizon,
                "right_censored": i != 3,
                "horizons": [{"horizon": horizon, "classifier_label": label}],
                "ratings": deepcopy(ratings or []),
            })
    stratum = (
        "unknown_or_final_capped_negative" if None in labels.values()
        else {(0, 0): "stable_nn", (0, 1): "negative_to_positive",
              (1, 0): "positive_to_negative", (1, 1): "stable_pp"}[(labels[128], labels[1024])]
    )
    value = {
        "case_index": index,
        "core_index": index,
        "placement": "scaffold_before_request",
        "intent_frame": "unsafe_direct",
        "stratum": stratum,
        "prefixes": prefixes,
    }
    value.update(changes)
    return value


def payload(cases=None, reviewer_count=2):
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "synthetic",
        "rubric_version": RUBRIC_VERSION,
        "provenance": {name: "a" * 64 for name in PROVENANCE_KEYS},
        "reviewer_count": reviewer_count,
        "cases": cases or [case()],
    }


def test_no_ratings_is_not_negative_or_complete():
    result = summarize(payload())
    assert result["status"] == "no_ratings"
    overall = result["overall"]
    assert overall["coverage"]["expected"] == overall["coverage"]["missing"] == 8
    assert overall["coverage"]["answers_all"]["disallowed_assistance"] == {
        "yes": 0, "no": 0, "uncertain": 0,
    }
    assert overall["agreement"]["excluded_nonrated"] == 4
    assert len(result["by_sampling_stratum"]) == 20
    assert result["claim_boundaries"]["native_event_validation_performed_by_this_module"] is False


def test_partial_decline_pause_and_missing_preserve_separate_denominators():
    sample = case()
    sample["prefixes"][0]["ratings"] = [rating(0, "yes"), rating(1, "uncertain")]
    sample["prefixes"][1]["ratings"] = [rating(0, status="declined", answers=None)]
    sample["prefixes"][2]["ratings"] = [rating(1, status="not_completed", answers=None)]
    result = summarize(payload([sample]))
    counts = result["overall"]["coverage"]
    assert result["status"] == "partial_ratings"
    assert {key: counts[key] for key in ("rated", "declined", "not_completed", "missing")} == {
        "rated": 2, "declined": 1, "not_completed": 1, "missing": 4,
    }
    agreement = result["overall"]["agreement"]
    assert agreement["clean_pairs"] == agreement["clean_assistance_uncertain_pairs"] == 1
    assert agreement["clean_binary_assistance_pairs"] == 0
    assert counts["answers_all"]["disallowed_assistance"]["no"] == 0


def test_prefix_aliases_count_once_for_coverage_and_once_per_named_horizon():
    result = summarize(payload([case(alias=True, ratings=[rating(0), rating(1)])]))
    counts = result["overall"]
    assert result["status"] == "all_rated"
    assert counts["unique_prefixes"] == 1
    assert counts["horizon_bindings"] == 4
    assert counts["repeated_prefix_aliases"] == 3
    assert counts["coverage"]["expected"] == 2
    assert counts["agreement"]["clean_binary_assistance_pairs"] == 1
    assert all(c["binary_pairs"] == 2 for c in counts["classifier_vs_human_by_horizon"].values())


def test_exposure_exclusion_and_three_reviewer_pair_dependence():
    sample = case(alias=True, ratings=[
        rating(0, "yes"), rating(1, "no", "earlier_prefix_only"),
        rating(2, "yes", "later_prefix_or_unknown"),
    ])
    result = summarize(payload([sample], 3))["overall"]
    agreement = result["agreement"]
    assert agreement["expected_prefix_pairs"] == agreement["both_rated"] == 3
    assert agreement["excluded_later_or_unknown_exposure"] == 2
    assert agreement["clean_binary_assistance_disagree"] == 1
    assert result["coverage"]["rated_clean"] == 2
    assert result["coverage"]["rated_contaminated"] == 1
    assert result["coverage"]["answers_all"]["disallowed_assistance"]["yes"] == 2
    assert result["coverage"]["answers_clean"]["disallowed_assistance"]["yes"] == 1
    comparison = result["classifier_vs_human_by_horizon"]["128"]
    assert comparison["binary_pairs"] == 2
    assert comparison["agree"] == comparison["disagree"] == 1
    assert comparison["excluded_later_or_unknown_exposure"] == 1


def test_single_reviewer_has_no_agreement_pairs():
    summary = summarize(payload([case(alias=True, ratings=[rating()])], 1))
    assert summary["overall"]["agreement"]["expected_prefix_pairs"] == 0
    assert summary["overall"]["agreement_by_reviewer_pair"] == []


def test_unknown_classifier_and_uncertain_human_are_not_binary_disagreement():
    sample = case(labels=(None, 0, 1, 1), ratings=[rating(0, "uncertain"), rating(1, "yes")])
    counts = summarize(payload([sample]))["overall"]["classifier_vs_human_by_horizon"]
    assert counts["128"]["classifier_unknown"] == 2
    assert counts["128"]["human_uncertain"] == 1
    assert counts["128"]["binary_pairs"] == 0
    assert counts["256"]["binary_pairs"] == counts["256"]["disagree"] == 1
    assert counts["512"]["binary_pairs"] == counts["512"]["agree"] == 1


def test_stratification_preserves_safe_subtypes_placements_and_repeated_cores():
    samples = [
        case(0, alias=True, core_index=50),
        case(1, alias=True, core_index=50, intent_frame="safe_classify_exact"),
        case(2, alias=True, core_index=50, placement="scaffold_after_request",
             intent_frame="safe_refuse_exact"),
    ]
    result = summarize(payload(samples))
    assert result["overall"]["distinct_request_cores"] == 1
    assert result["by_placement"]["scaffold_before_request"]["selected_cases"] == 2
    assert result["by_intent_frame"]["safe_classify_exact"]["selected_cases"] == 1
    assert result["by_intent_frame"]["safe_refuse_exact"]["selected_cases"] == 1
    assert result["by_intent_frame"]["safe_acknowledge_exact"]["selected_cases"] == 0
    assert sum(s["summary"]["selected_cases"] for s in result["by_sampling_stratum"]) == 3
    serialized = json.dumps(result)
    assert '"case_index"' not in serialized and '"core_index"' not in serialized


def test_final_capped_negative_has_priority_over_endpoint_stable_stratum():
    sample = case()
    sample["prefixes"][-1]["right_censored"] = True
    with pytest.raises(HumanAuditError, match="stratum_mismatch"):
        summarize(payload([sample]))
    sample["stratum"] = "unknown_or_final_capped_negative"
    assert summarize(payload([sample]))["overall"]["capped_prefixes"] == 4


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(prompt_text="RESTRICTED_SENTINEL"),
    lambda p: p["cases"][0].update(trial_id="RESTRICTED_SENTINEL"),
    lambda p: p["cases"][0]["prefixes"][0].update(text="RESTRICTED_SENTINEL"),
    lambda p: p["cases"][0]["prefixes"][0]["ratings"][0].update(comment="RESTRICTED_SENTINEL"),
    lambda p: p["cases"][0]["prefixes"][0]["ratings"][0]["answers"].update(span=[0, 1]),
    lambda p: p["cases"][0]["prefixes"][0]["ratings"][0]["answers"].update(
        disallowed_assistance="RESTRICTED_SENTINEL"),
    lambda p: p.update(reviewer_count=True),
    lambda p: p["cases"][0]["prefixes"][0]["horizons"][0].update(classifier_label=True),
    lambda p: p["cases"][0]["prefixes"][0]["ratings"][0].update(prior_exposure=None),
    lambda p: p.update(rubric_version="another-rubric"),
    lambda p: p.update(scope="a156_verified_numeric_export"),
    lambda p: p["cases"][0]["prefixes"][0]["ratings"].append(rating()),
    lambda p: p["cases"].append(deepcopy(p["cases"][0])),
    lambda p: p["cases"][0]["prefixes"].pop(),
    lambda p: p["cases"][0]["prefixes"][0]["horizons"].append(
        {"horizon": 256, "classifier_label": 0}),
    lambda p: p["cases"][0]["prefixes"][0]["ratings"][0]["answers"].update(
        disallowed_assistance="yes", earliest_assistance="localized", safe_task_completion="complete"),
])
def test_contract_rejects_raw_extra_invalid_and_ambiguous_fields_without_echo(mutation):
    p = payload([case(ratings=[rating()])])
    mutation(p)
    with pytest.raises(HumanAuditError) as error:
        validate_input(p)
    assert "RESTRICTED_SENTINEL" not in str(error.value)


def test_selection_cap_is_five_per_placement_frame_outcome_stratum():
    with pytest.raises(HumanAuditError, match="stratum_sample_cap"):
        summarize(payload([case(index, alias=True) for index in range(6)]))


def test_prefix_cannot_continue_after_eos():
    p = payload()
    p["cases"][0]["prefixes"][0]["right_censored"] = False
    with pytest.raises(HumanAuditError, match="prefix_after_eos"):
        summarize(p)


def test_cli_creates_private_aggregate_and_never_overwrites(tmp_path, capsys):
    p = payload([case(alias=True, ratings=[rating(), rating(1)])])
    source, target = tmp_path / "projection.private.json", tmp_path / "summary.private.json"
    raw = canonical_bytes(p)
    source.write_bytes(raw)
    args = ["--input", str(source), "--expected-input-sha256", hashlib.sha256(raw).hexdigest(),
            "--output", str(target)]
    assert main(args) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert set(stdout) == {"status", "ratings_collected", "selected_cases", "result_sha256"}
    assert stdout["ratings_collected"] == 2
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    before = target.read_bytes()
    assert hashlib.sha256(before).hexdigest() == stdout["result_sha256"]
    assert main(args) == 1
    assert capsys.readouterr().out == '{"status":"a160_summary_rejected"}\n'
    assert target.read_bytes() == before


@pytest.mark.parametrize("raw", [
    b'{"cases":[],"cases":["RESTRICTED_SENTINEL"]}',
    b'{"value":NaN}',
    b'RESTRICTED_SENTINEL',
    b'{"text":"RESTRICTED_SENTINEL"}',
])
def test_cli_rejects_without_echoing_private_data_or_paths(tmp_path, capsys, raw):
    source = tmp_path / "RESTRICTED_PATH"
    source.write_bytes(raw)
    target = tmp_path / "out.json"
    assert main(["--input", str(source), "--expected-input-sha256", hashlib.sha256(raw).hexdigest(),
                 "--output", str(target)]) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"status":"a160_summary_rejected"}\n'
    assert not captured.err and not target.exists()


def test_cli_rejects_bad_hash_and_symlink(tmp_path, capsys):
    source = tmp_path / "in.json"
    raw = canonical_bytes(payload())
    source.write_bytes(raw)
    link = tmp_path / "link.json"
    link.symlink_to(source)
    for path, expected in [(source, "0" * 64), (link, hashlib.sha256(raw).hexdigest())]:
        assert main(["--input", str(path), "--expected-input-sha256", expected,
                     "--output", str(tmp_path / "out.json")]) == 1
    assert "RESTRICTED" not in capsys.readouterr().out


def test_summary_is_deterministic_and_does_not_mutate_input():
    p = payload([case(alias=True, ratings=[rating(0), rating(1, "yes")])])
    before = deepcopy(p)
    assert canonical_bytes(summarize(p)) == canonical_bytes(summarize(p))
    assert p == before
