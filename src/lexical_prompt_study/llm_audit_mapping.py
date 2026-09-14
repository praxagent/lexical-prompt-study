"""A161 numeric alias/classifier join; no prompt, response, or detector inputs.

The CLI reads hash-pinned A155 selection, A156 alias/completion/request metadata,
the original EOS classifier receipts and verified A148 classifier journal only.
It never opens the display packet. Outputs remain private; stdout is counts/hashes.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

from .llm_audit_analysis import (
    HORIZONS,
    _safe_path,
    canonical,
    digest,
    keys,
    read_json,
    require,
    sha,
    validate_mapping,
)

PREFIX_FIELDS = {
    "requested_horizon", "observed_token_count", "right_censored",
    "generated_token_ids_sha256", "generated_text_sha256",
}
SELECTED_FIELDS = {
    "review_id", "trial_id", "placement", "frame_group", "intent_frame", "outcome_stratum",
    "origin", "request_core_id", "request_core_sha256", "selection_rank_sha256",
    "source_numeric_row_sha256", "prefix_bindings",
}
ALIAS_FIELDS = {
    "review_id", "trial_id", "prefix_id", "requested_horizons", "prompt_sha256",
    "prompt_token_ids_sha256", "observed_token_count", "right_censored",
    "generated_token_ids_sha256", "generated_text_sha256", "generation_receipt_sha256",
}


def build_mapping(*, selection: dict, aliases: dict, request: dict, completion: dict,
                  continued_scores: dict[tuple[str, int], dict], original_scores: dict[str, dict],
                  selection_sha256: str, aliases_sha256: str, request_sha256: str,
                  scoring_sha256: str, expected_cases: int = 87,
                  expected_prefixes: int = 287) -> dict:
    """Pure join that reproduces each selected A155 numeric row hash exactly."""
    for value in (selection_sha256, aliases_sha256, request_sha256, scoring_sha256):
        digest(value)
    require(sha(selection) == selection_sha256 and sha(aliases) == aliases_sha256
            and sha(request) == request_sha256, "mapping_source_hashes")
    require(selection.get("status") == "a155_numeric_human_audit_selection_complete",
            "mapping_selection_status")
    require(type(selection.get("selected_rows")) is list
            and len(selection["selected_rows"]) == expected_cases
            and selection.get("selected_count") == expected_cases, "mapping_selected_count")
    require(aliases.get("schema_version") == "a156-review-alias-map-v1"
            and type(aliases.get("mappings")) is list
            and len(aliases["mappings"]) == expected_prefixes, "mapping_alias_schema")
    require(completion.get("status") == "a156_blinded_display_packet_built"
            and completion.get("alias_map_sha256") == aliases_sha256
            and completion.get("selected_cases") == expected_cases
            and completion.get("unique_prefixes") == expected_prefixes
            and completion.get("horizon_slots") == expected_cases * 4, "mapping_completion")
    packet_sha256 = completion["display_packet_sha256"]
    digest(packet_sha256)
    require(type(aliases.get("provenance")) is dict
            and aliases["provenance"].get("request_sha256") == request_sha256,
            "mapping_alias_request")
    require(request.get("status") == "a156_selected_artifact_request_compiled"
            and type(request.get("provenance")) is dict
            and request["provenance"].get("selection_sha256") == selection_sha256,
            "mapping_request_selection")
    selected = {}
    for row in selection["selected_rows"]:
        keys(row, SELECTED_FIELDS, "mapping_selected_fields")
        digest(row["trial_id"], 24)
        digest(row["review_id"], 24)
        digest(row["request_core_sha256"])
        digest(row["source_numeric_row_sha256"])
        require(row["trial_id"] not in selected, "mapping_duplicate_selected")
        selected[row["trial_id"]] = row
    require(len({row["review_id"] for row in selected.values()}) == expected_cases,
            "mapping_duplicate_review")
    request_cases = request.get("cases")
    require(type(request_cases) is list and len(request_cases) == expected_cases,
            "mapping_request_count")
    requested = {row["trial_id"]: row for row in request_cases}
    require(set(requested) == set(selected) and len(requested) == expected_cases,
            "mapping_request_case_set")
    ordered = sorted(selected.values(), key=lambda row: row["review_id"])
    case_indices = {row["trial_id"]: i for i, row in enumerate(ordered)}
    core_indices = {value: i for i, value in enumerate(sorted({
        row["request_core_sha256"] for row in selected.values()
    }))}
    numeric_scores = {}
    for trial, row in selected.items():
        require(row["frame_group"] == ("direct" if row["intent_frame"] == "unsafe_direct" else "safe"),
                "mapping_frame_group")
        require(row["origin"] in ("continued", "original_eos"), "mapping_origin")
        require(type(row["prefix_bindings"]) is list and len(row["prefix_bindings"]) == 4,
                "mapping_selected_prefix_count")
        request_case = requested[trial]
        require(request_case["review_id"] == row["review_id"] and request_case["origin"] == row["origin"],
                "mapping_requested_identity")
        require(type(request_case["prefix_bindings"]) is list
                and len(request_case["prefix_bindings"]) == 4, "mapping_requested_prefix_count")
        scores = []
        for horizon, binding, requested_binding in zip(
            HORIZONS, row["prefix_bindings"], request_case["prefix_bindings"], strict=True,
        ):
            keys(binding, PREFIX_FIELDS, "mapping_selected_prefix_fields")
            require(binding["requested_horizon"] == horizon
                    and all(requested_binding[k] == binding[k] for k in PREFIX_FIELDS),
                    "mapping_selected_requested_prefix")
            if row["origin"] == "continued":
                require((trial, horizon) in continued_scores, "mapping_missing_continued_score")
                score = continued_scores[trial, horizon]
                require(all(score[k] == binding[k] for k in PREFIX_FIELDS)
                        and score["classifier_behavior_sha256"] == row["request_core_sha256"]
                        and score["generation_receipt_sha256"] == requested_binding["generation_receipt_sha256"],
                        "mapping_continued_score_binding")
            else:
                require(trial in original_scores, "mapping_missing_original_score")
                score = original_scores[trial]
                require(score["trial_id"] == trial
                        and score["classifier_behavior_sha256"] == row["request_core_sha256"]
                        and score["generated_text_sha256"] == binding["generated_text_sha256"]
                        and score["generation_receipt_sha256"] == requested_binding["generation_receipt_sha256"],
                        "mapping_original_score_binding")
            label = score["binary_prediction"]
            require(label is None or type(label) is bool, "mapping_source_label")
            projection = {**binding, "binary_prediction": label}
            scores.append(projection)
            numeric_scores[trial, horizon] = projection
        projected_row = {
            "trial_id": trial, "placement": row["placement"], "intent_frame": row["intent_frame"],
            "safe_intent": row["frame_group"] == "safe", "origin": row["origin"],
            "request_core_id": row["request_core_id"], "request_core_sha256": row["request_core_sha256"],
            "scores": scores,
        }
        require(sha(projected_row) == row["source_numeric_row_sha256"], "mapping_a155_numeric_row_hash")
    mapped = []
    seen_bindings = set()
    for alias in aliases["mappings"]:
        keys(alias, ALIAS_FIELDS, "mapping_alias_fields")
        trial = alias["trial_id"]
        require(trial in selected, "mapping_unselected_alias")
        row = selected[trial]
        require(alias["review_id"] == row["review_id"], "mapping_alias_review")
        digest(alias["prefix_id"])
        require(alias["prefix_id"] == sha({"review_id": row["review_id"], **{
            k: alias[k] for k in ("observed_token_count", "generated_token_ids_sha256",
                                 "generated_text_sha256", "right_censored")
        }}), "mapping_prefix_identity")
        original = requested[trial]["original"]
        require(alias["prompt_token_ids_sha256"] == original["prompt_token_ids_sha256"],
                "mapping_prompt_token_binding")
        require(type(alias["requested_horizons"]) is list and alias["requested_horizons"]
                and alias["requested_horizons"] == sorted(set(alias["requested_horizons"])),
                "mapping_alias_horizons")
        horizons = []
        for horizon in alias["requested_horizons"]:
            require(type(horizon) is int and horizon in HORIZONS
                    and (trial, horizon) not in seen_bindings, "mapping_duplicate_alias_binding")
            seen_bindings.add((trial, horizon))
            score = numeric_scores[trial, horizon]
            require(all(alias[k] == score[k] for k in PREFIX_FIELDS - {"requested_horizon"}),
                    "mapping_alias_score_binding")
            requested_binding = requested[trial]["prefix_bindings"][HORIZONS.index(horizon)]
            require(alias["generation_receipt_sha256"] == requested_binding["generation_receipt_sha256"],
                    "mapping_alias_generation")
            label = score["binary_prediction"]
            horizons.append({"horizon": horizon, "classifier_label": None if label is None else int(label)})
        mapped.append({"prefix_id": alias["prefix_id"], "case_index": case_indices[trial],
                       "core_index": core_indices[row["request_core_sha256"]],
                       "placement": row["placement"], "intent_frame": row["intent_frame"],
                       "stratum": row["outcome_stratum"],
                       "observed_token_count": alias["observed_token_count"],
                       "right_censored": alias["right_censored"], "horizons": horizons})
    require(len(seen_bindings) == expected_cases * 4, "mapping_all_horizons")
    value = {"schema_version": "a161-numeric-mapping-v1", "packet_sha256": packet_sha256,
             "provenance": {"selection_sha256": selection_sha256, "scoring_sha256": scoring_sha256,
                            "alias_map_sha256": aliases_sha256},
             "rows": sorted(mapped, key=lambda row: row["prefix_id"])}
    validate_mapping(value, packet_sha256, sorted(row["prefix_id"] for row in mapped))
    return value


def _file_sha(path: Path, maximum: int = 256 * 1024**2) -> str:
    path = _safe_path(path)
    require(path.is_file() and path.stat().st_size <= maximum, "mapping_file_size")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _module(path: Path, expected_sha256: str):
    digest(expected_sha256)
    require(_file_sha(path, 1024**2) == expected_sha256, "mapping_verifier_source")
    spec = importlib.util.spec_from_file_location("a161_mapping_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = read_json(args.inputs, args.inputs_sha256)
        keys(inputs, {"schema_version", "repository", "sources", "artifacts"}, "mapping_inputs")
        require(inputs["schema_version"] == "a161-mapping-inputs-v1", "mapping_inputs_version")
        repo = _safe_path(Path(inputs["repository"]))
        keys(inputs["sources"], {"verify_scoring17_a148.py", "capacity_scoring_core_a148.py"},
             "mapping_source_inventory")
        keys(inputs["artifacts"], {"selection", "aliases", "request", "completion", "scoring"},
             "mapping_artifact_inventory")
        fixed_paths = {
            "selection": "private/continuation/human-audit-29/selection.private.json",
            "aliases": "private/human-audit/a156-review/alias-map.private.json",
            "request": "private/human-audit/a158-request/request.private.json",
            "completion": "private/human-audit/a156-review/complete.private.json",
            "scoring": "private/continuation/a148-verification-17/scoring-receipts.private.zip",
        }
        documents = {}
        for key, relative in fixed_paths.items():
            digest(inputs["artifacts"][key])
            path = repo / relative
            if key == "scoring":
                require(_file_sha(path) == inputs["artifacts"][key], "mapping_scoring_zip_hash")
            else:
                documents[key] = read_json(path, inputs["artifacts"][key])
        # The pinned verifier reads numeric scoring journal records, never packets.
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            verifier = _module(repo / "private/continuation/verify_scoring17_a148.py",
                               inputs["sources"]["verify_scoring17_a148.py"])
            core = _module(repo / "private/continuation/capacity_scoring_core_a148.py",
                           inputs["sources"]["capacity_scoring_core_a148.py"])
            reader = verifier.ZipReader(repo / fixed_paths["scoring"])
            try:
                summary, _ = verifier.verify(reader, core)
                current, _, _ = verifier.chain(reader, "scoring16", summary["operation_binding_sha256"], core)
                prior, _, _ = verifier.chain(reader, "prior10", verifier.OLD_OPERATION, core, verifier.OLD_TIP)
                require(not set(prior) & set(current), "mapping_overlapping_score_journals")
                scores = {**prior, **current}
            finally:
                reader.archive.close()
        original = {}
        for row in documents["selection"]["selected_rows"]:
            if row["origin"] == "original_eos":
                trial = row["trial_id"]
                digest(trial, 24)
                original[trial] = read_json(repo / "private/runs/jlens-incremental-a142/scoring/trials"
                                            / (trial + ".json"))
        value = build_mapping(
            **documents, continued_scores=scores, original_scores=original,
            selection_sha256=inputs["artifacts"]["selection"],
            aliases_sha256=inputs["artifacts"]["aliases"],
            request_sha256=inputs["artifacts"]["request"], scoring_sha256=inputs["artifacts"]["scoring"],
        )
        raw = canonical(value)
        output = _safe_path(args.output)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": "a161_numeric_mapping_written", "selected_cases": 87,
                          "unique_prefixes": len(value["rows"]),
                          "horizon_bindings": sum(len(row["horizons"]) for row in value["rows"]),
                          "mapping_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
        return 0
    except Exception:
        print('{"status":"a161_numeric_mapping_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
