"""Private numeric utility joins and descriptive semantic/mechanical comparisons.

Reads only the pinned feature export's metadata member, never response text or
feature arrays. Model receipts must be validated by their provider adapter first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from .llm_audit_analysis import (
    HORIZONS, _safe_path, canonical, digest, keys, read_json, require, sha, validate_mapping,
)

FEATURE_EXPORT_SHA256 = "6b072eb2c825678c7f259afe50cf8a0353af92d3d5914cf4788eff326003a574"
UTILITY_FIELDS = {
    "trial_id", "requested_horizon", "observed_token_count", "right_censored",
    "prefix_token_ids_sha256", "source_generation_receipt_sha256", "generated_text_sha256",
    "refusal_detected", "utility_exact_match",
}


def build_utility_mapping(utility: list, aliases: dict, numeric_mapping: dict,
                          feature_sha256: str, alias_sha256: str) -> dict:
    """Join each alias to its exact generation, with no horizon propagation."""
    digest(feature_sha256)
    require(sha(aliases) == alias_sha256, "utility_alias_hash")
    require(numeric_mapping["provenance"]["alias_map_sha256"] == alias_sha256,
            "utility_alias_provenance")
    selection = sorted(row["prefix_id"] for row in numeric_mapping["rows"])
    mapped = validate_mapping(numeric_mapping, numeric_mapping["packet_sha256"], selection)
    require(aliases.get("schema_version") == "a156-review-alias-map-v1"
            and type(aliases.get("mappings")) is list, "utility_alias_schema")
    require(type(utility) is list and 1 <= len(utility) <= 35520, "utility_source_size")
    cells = {}
    for row in utility:
        keys(row, UTILITY_FIELDS, "utility_source_fields")
        digest(row["trial_id"], 24)
        for key in ("prefix_token_ids_sha256", "source_generation_receipt_sha256", "generated_text_sha256"):
            digest(row[key])
        horizon = row["requested_horizon"]
        require(type(horizon) is int and horizon in HORIZONS, "utility_horizon")
        require(type(row["observed_token_count"]) is int
                and 0 <= row["observed_token_count"] <= horizon
                and type(row["right_censored"]) is bool
                and (not row["right_censored"] or row["observed_token_count"] == horizon),
                "utility_endpoint")
        require(type(row["refusal_detected"]) is bool
                and (row["utility_exact_match"] is None or type(row["utility_exact_match"]) is bool),
                "utility_source_values")
        key = (row["trial_id"], horizon)
        require(key not in cells, "utility_duplicate_source")
        cells[key] = row
    rows = {}
    for alias in aliases["mappings"]:
        prefix = alias["prefix_id"]
        require(prefix in mapped and prefix not in rows, "utility_alias_selection")
        numeric = mapped[prefix]
        require(alias["requested_horizons"] == sorted(h["horizon"] for h in numeric["horizons"]),
                "utility_alias_horizons")
        require(all(alias[k] == numeric[k] for k in ("observed_token_count", "right_censored")),
                "utility_alias_endpoint")
        require(prefix == sha({"review_id": alias["review_id"], **{
            k: alias[k] for k in ("observed_token_count", "right_censored",
                                 "generated_token_ids_sha256", "generated_text_sha256")
        }}), "utility_prefix_identity")
        observed = []
        for horizon in alias["requested_horizons"]:
            require((alias["trial_id"], horizon) in cells, "utility_missing_source")
            cell = cells[alias["trial_id"], horizon]
            equal = {
                "observed_token_count": "observed_token_count", "right_censored": "right_censored",
                "prefix_token_ids_sha256": "generated_token_ids_sha256",
                "generated_text_sha256": "generated_text_sha256",
                "source_generation_receipt_sha256": "generation_receipt_sha256",
            }
            require(all(cell[k] == alias[v] for k, v in equal.items()), "utility_generation_binding")
            require((cell["utility_exact_match"] is None) == (numeric["intent_frame"] == "unsafe_direct"),
                    "utility_applicability")
            observed.append((cell["utility_exact_match"], cell["refusal_detected"]))
        require(len(set(observed)) == 1, "utility_alias_value_disagreement")
        rows[prefix] = {"prefix_id": prefix, "utility_exact_match": observed[0][0],
                        "refusal_detected": observed[0][1]}
    require(set(rows) == set(mapped), "utility_full_coverage")
    return {"schema_version": "a161-utility-mapping-v1", "packet_sha256": numeric_mapping["packet_sha256"],
            "provenance": {"feature_export_sha256": feature_sha256, "alias_map_sha256": alias_sha256,
                           "numeric_mapping_sha256": sha(numeric_mapping)},
            "rows": [rows[key] for key in sorted(rows)]}


def summarize_utility(mapping: dict, numeric_mapping: dict, receipt_lists: list[list[dict]]) -> dict:
    """Compare already-validated model receipts, retaining uncertainty and absence."""
    keys(mapping, {"schema_version", "packet_sha256", "provenance", "rows"}, "utility_mapping_keys")
    require(mapping["schema_version"] == "a161-utility-mapping-v1"
            and mapping["packet_sha256"] == numeric_mapping["packet_sha256"], "utility_mapping_packet")
    keys(mapping["provenance"], {"feature_export_sha256", "alias_map_sha256", "numeric_mapping_sha256"},
         "utility_mapping_provenance")
    for value in mapping["provenance"].values():
        digest(value)
    require(mapping["provenance"]["numeric_mapping_sha256"] == sha(numeric_mapping)
            and mapping["provenance"]["alias_map_sha256"] == numeric_mapping["provenance"]["alias_map_sha256"],
            "utility_mapping_binding")
    numeric = {row["prefix_id"]: row for row in numeric_mapping["rows"]}
    validate_mapping(numeric_mapping, mapping["packet_sha256"], sorted(numeric))
    utilities = {}
    for row in mapping["rows"]:
        keys(row, {"prefix_id", "utility_exact_match", "refusal_detected"}, "utility_rating_keys")
        require(row["prefix_id"] in numeric and row["prefix_id"] not in utilities, "utility_rating_selection")
        require(type(row["refusal_detected"]) is bool
                and (row["utility_exact_match"] is None or type(row["utility_exact_match"]) is bool),
                "utility_rating_values")
        require((row["utility_exact_match"] is None)
                == (numeric[row["prefix_id"]]["intent_frame"] == "unsafe_direct"), "utility_applicability")
        utilities[row["prefix_id"]] = row
    require(set(utilities) == set(numeric), "utility_full_coverage")

    def empty():
        return {"expected": 0, "failed_or_missing": 0, "valid": 0,
                "safe_task_table": {mechanical: dict.fromkeys(
                    ("complete", "incomplete", "uncertain", "not_applicable"), 0)
                    for mechanical in ("exact", "not_exact", "not_applicable")},
                "refusal_table": {parser: dict.fromkeys(("yes", "no", "uncertain"), 0)
                                  for parser in ("detected", "not_detected")}}

    out = []
    for receipts in receipt_lists:
        require(len({r["prefix_id"] for r in receipts}) == len(receipts), "utility_duplicate_receipt")
        require(all(r["prefix_id"] in numeric for r in receipts), "utility_receipt_selection")
        lookup = {r["prefix_id"]: r for r in receipts}
        groups = {"distinct_prefixes": empty(), **{str(h): empty() for h in HORIZONS}}
        for prefix, row in numeric.items():
            receipt = lookup.get(prefix)
            for group in ("distinct_prefixes", *(str(h["horizon"]) for h in row["horizons"])):
                counts = groups[group]
                counts["expected"] += 1
                if receipt is None or receipt["status"] != "valid":
                    counts["failed_or_missing"] += 1
                    continue
                counts["valid"] += 1
                judgment = receipt["judgment"]
                u = utilities[prefix]
                label = "not_applicable" if u["utility_exact_match"] is None else (
                    "exact" if u["utility_exact_match"] else "not_exact")
                counts["safe_task_table"][label][judgment["safe_task_completion"]] += 1
                counts["refusal_table"]["detected" if u["refusal_detected"] else "not_detected"][
                    judgment["refusal_present"]] += 1
        out.append(groups)
    return {"schema_version": "a161-utility-comparison-v1", "mechanical_ground_truth": False,
            "mapping_sha256": sha(mapping), "per_judge": out,
            "limitation": "Descriptive selected-sample comparison; exact matching and refusal parsing are mechanical measures, not semantic truth."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-export", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--aliases-sha256", required=True)
    parser.add_argument("--numeric-mapping", type=Path, required=True)
    parser.add_argument("--numeric-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = _safe_path(args.feature_export)
        require(path.is_file() and path.stat().st_size <= 256 * 1024**2, "utility_export_size")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == FEATURE_EXPORT_SHA256, "utility_export_hash")
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if info.filename == "metadata.private.json"]
            require(len(infos) == 1 and infos[0].file_size <= 64 * 1024**2, "utility_metadata_size")
            metadata = json.loads(archive.read(infos[0]))
        require(type(metadata) is dict and len(metadata.get("utility", [])) == 35520, "utility_export_coverage")
        numeric = read_json(args.numeric_mapping, args.numeric_sha256)
        aliases = read_json(args.aliases, args.aliases_sha256)
        result = build_utility_mapping(metadata["utility"], aliases, numeric,
                                       FEATURE_EXPORT_SHA256, args.aliases_sha256)
        raw = canonical(result)
        fd = os.open(_safe_path(args.output), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"status": "a161_utility_mapping_written", "distinct_prefixes": len(result["rows"]),
                          "mapping_sha256": hashlib.sha256(raw).hexdigest()}))
        return 0
    except Exception:
        print('{"status":"a161_utility_mapping_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
