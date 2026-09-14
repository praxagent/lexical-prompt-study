from __future__ import annotations

from copy import deepcopy

import pytest

from lexical_prompt_study.llm_audit_analysis import LLMAuditAnalysisError, sha
from lexical_prompt_study.llm_audit_mapping import build_mapping


def inputs():
    selected, requested, aliases, continued, original = [], [], [], {}, {}
    for index, origin in enumerate(("original_eos", "continued")):
        trial, review = f"{index + 1:024x}", f"{index + 11:024x}"
        core_sha = "9" * 64
        prefix_bindings, scores, request_bindings = [], [], []
        for position, horizon in enumerate((128, 256, 512, 1024)):
            count = 20 if origin == "original_eos" else horizon
            token_sha = f"{count + 1:064x}"
            text_sha = f"{count + 2:064x}"
            generation_sha = f"{count + 3:064x}"
            binding = {"requested_horizon": horizon, "observed_token_count": count,
                       "right_censored": origin == "continued" and position < 3,
                       "generated_token_ids_sha256": token_sha, "generated_text_sha256": text_sha}
            label = origin == "continued" and position > 1
            prefix_bindings.append(binding)
            request_bindings.append({**binding, "generation_receipt_sha256": generation_sha})
            score = {**binding, "binary_prediction": label}
            scores.append(score)
            if origin == "continued":
                continued[trial, horizon] = {**score, "trial_id": trial,
                                             "classifier_behavior_sha256": core_sha,
                                             "generation_receipt_sha256": generation_sha}
            else:
                original[trial] = {"trial_id": trial, "binary_prediction": False,
                                   "classifier_behavior_sha256": core_sha,
                                   "generated_text_sha256": text_sha,
                                   "generation_receipt_sha256": generation_sha}
            if position == 0 or origin == "continued":
                identity = {"review_id": review, **{k: binding[k] for k in (
                    "observed_token_count", "generated_token_ids_sha256", "generated_text_sha256",
                    "right_censored",
                )}}
                aliases.append({**identity, "trial_id": trial, "prefix_id": sha(identity),
                                "requested_horizons": [128, 256, 512, 1024] if origin == "original_eos" else [horizon],
                                "prompt_sha256": "7" * 64, "prompt_token_ids_sha256": "8" * 64,
                                "generation_receipt_sha256": generation_sha})
        numeric_row = {"trial_id": trial, "placement": "scaffold_before_request",
                       "intent_frame": "unsafe_direct", "safe_intent": False, "origin": origin,
                       "request_core_id": "synthetic-core", "request_core_sha256": core_sha,
                       "scores": scores}
        selected.append({"review_id": review, "trial_id": trial, "placement": numeric_row["placement"],
                         "frame_group": "direct", "intent_frame": "unsafe_direct",
                         "outcome_stratum": "stable_nn" if origin == "original_eos" else "negative_to_positive",
                         "origin": origin, "request_core_id": "synthetic-core", "request_core_sha256": core_sha,
                         "selection_rank_sha256": "6" * 64, "source_numeric_row_sha256": sha(numeric_row),
                         "prefix_bindings": prefix_bindings})
        requested.append({"review_id": review, "trial_id": trial, "origin": origin,
                          "original": {"prompt_token_ids_sha256": "8" * 64},
                          "prefix_bindings": request_bindings})
    selection = {"status": "a155_numeric_human_audit_selection_complete", "selected_count": 2,
                 "selected_rows": selected}
    request = {"status": "a156_selected_artifact_request_compiled",
               "provenance": {"selection_sha256": sha(selection)}, "cases": requested}
    alias_map = {"schema_version": "a156-review-alias-map-v1",
                 "provenance": {"request_sha256": sha(request)}, "mappings": aliases}
    completion = {"status": "a156_blinded_display_packet_built", "alias_map_sha256": sha(alias_map),
                  "selected_cases": 2, "unique_prefixes": 5, "horizon_slots": 8,
                  "display_packet_sha256": "5" * 64}
    return {"selection": selection, "request": request, "aliases": alias_map,
            "completion": completion, "continued_scores": continued, "original_scores": original,
            "selection_sha256": sha(selection), "aliases_sha256": sha(alias_map),
            "request_sha256": sha(request), "scoring_sha256": "4" * 64,
            "expected_cases": 2, "expected_prefixes": 5}


def test_numeric_mapping_recovers_both_lineages_and_exact_aliases():
    source = inputs()
    result = build_mapping(**source)
    assert result["schema_version"] == "a161-numeric-mapping-v1"
    assert len(result["rows"]) == 5
    assert sum(len(row["horizons"]) for row in result["rows"]) == 8
    assert len({row["core_index"] for row in result["rows"]}) == 1
    eos = next(row for row in result["rows"] if row["observed_token_count"] == 20)
    assert len(eos["horizons"]) == 4
    assert all(binding["classifier_label"] == 0 for binding in eos["horizons"])
    assert {"trial_id", "review_id", "prompt_sha256", "text", "request_core_id"}.isdisjoint(eos)


def test_changed_classifier_label_rejected_by_original_a155_row_hash():
    source = inputs()
    source["continued_scores"][(f"{2:024x}", 256)]["binary_prediction"] = True
    with pytest.raises(LLMAuditAnalysisError, match="a155_numeric_row_hash"):
        build_mapping(**source)
    source = inputs()
    source["original_scores"][f"{1:024x}"]["binary_prediction"] = True
    with pytest.raises(LLMAuditAnalysisError, match="a155_numeric_row_hash"):
        build_mapping(**source)


@pytest.mark.parametrize("mutation", [
    lambda x: x["continued_scores"].pop((f"{2:024x}", 256)),
    lambda x: x["original_scores"].clear(),
    lambda x: x["continued_scores"][(f"{2:024x}", 256)].update(generated_text_sha256="f" * 64),
    lambda x: x["continued_scores"][(f"{2:024x}", 256)].update(generation_receipt_sha256="f" * 64),
    lambda x: x["original_scores"][f"{1:024x}"].update(classifier_behavior_sha256="f" * 64),
    lambda x: x["completion"].update(alias_map_sha256="f" * 64),
    lambda x: x["selection"].update(selected_count=3),
    lambda x: x["aliases"]["mappings"][0].update(prompt_text="RESTRICTED_SENTINEL"),
    lambda x: x["aliases"]["mappings"][0].update(prefix_id="f" * 64),
])
def test_bad_numeric_source_cannot_silently_change_the_join(mutation):
    source = inputs()
    mutation(source)
    with pytest.raises(LLMAuditAnalysisError) as error:
        build_mapping(**source)
    assert "RESTRICTED_SENTINEL" not in str(error.value)


def test_no_input_mutation_and_output_deterministic():
    source = inputs()
    before = deepcopy(source)
    assert build_mapping(**source) == build_mapping(**source)
    assert source == before


def rebind(source):
    for row in source["selection"]["selected_rows"]:
        scores = []
        for prefix in row["prefix_bindings"]:
            label = (source["original_scores"][row["trial_id"]]["binary_prediction"]
                     if row["origin"] == "original_eos" else
                     source["continued_scores"][row["trial_id"], prefix["requested_horizon"]]["binary_prediction"])
            scores.append({**prefix, "binary_prediction": label})
        row["source_numeric_row_sha256"] = sha({
            "trial_id": row["trial_id"], "placement": row["placement"], "intent_frame": row["intent_frame"],
            "safe_intent": row["frame_group"] == "safe", "origin": row["origin"],
            "request_core_id": row["request_core_id"], "request_core_sha256": row["request_core_sha256"],
            "scores": scores,
        })
    source["selection_sha256"] = sha(source["selection"])
    source["request"]["provenance"]["selection_sha256"] = source["selection_sha256"]
    source["request_sha256"] = sha(source["request"])
    source["aliases"]["provenance"]["request_sha256"] = source["request_sha256"]
    source["aliases_sha256"] = sha(source["aliases"])
    source["completion"]["alias_map_sha256"] = source["aliases_sha256"]


def test_null_classifier_score_remains_null():
    source = inputs()
    source["continued_scores"][(f"{2:024x}", 512)]["binary_prediction"] = None
    source["selection"]["selected_rows"][1]["outcome_stratum"] = "unknown_or_final_capped_negative"
    rebind(source)
    result = build_mapping(**source)
    target = next(row for row in result["rows"] if row["observed_token_count"] == 512)
    assert target["horizons"][0]["classifier_label"] is None


def test_continued_eos_aliases_are_not_separate_prefixes():
    source = inputs()
    trial = f"{2:024x}"
    selected = source["selection"]["selected_rows"][1]
    requested = source["request"]["cases"][1]
    template = {**requested["prefix_bindings"][1], "right_censored": False}
    for index, horizon in enumerate((256, 512, 1024), 1):
        requested["prefix_bindings"][index] = {**template, "requested_horizon": horizon}
        selected["prefix_bindings"][index] = {
            k: v for k, v in requested["prefix_bindings"][index].items() if k != "generation_receipt_sha256"
        }
        source["continued_scores"][trial, horizon] = {
            **requested["prefix_bindings"][index], "trial_id": trial, "binary_prediction": True,
            "classifier_behavior_sha256": "9" * 64,
        }
    kept = [a for a in source["aliases"]["mappings"]
            if a["trial_id"] != trial or a["observed_token_count"] <= 256]
    target = next(a for a in kept if a["trial_id"] == trial and a["observed_token_count"] == 256)
    target.update(right_censored=False, requested_horizons=[256, 512, 1024])
    target["prefix_id"] = sha({k: target[k] for k in (
        "review_id", "observed_token_count", "generated_token_ids_sha256", "generated_text_sha256", "right_censored",
    )})
    source["aliases"]["mappings"] = kept
    source["expected_prefixes"] = source["completion"]["unique_prefixes"] = 3
    rebind(source)
    result = build_mapping(**source)
    assert len(result["rows"]) == 3
    assert sum(len(row["horizons"]) for row in result["rows"]) == 8


def test_rehashed_alias_cannot_crosswire_generation_receipt():
    source = inputs()
    source["aliases"]["mappings"][0]["generation_receipt_sha256"] = "f" * 64
    rebind(source)
    with pytest.raises(LLMAuditAnalysisError, match="alias_generation"):
        build_mapping(**source)
