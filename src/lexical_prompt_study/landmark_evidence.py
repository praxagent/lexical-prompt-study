"""Model-free, lossless matched-landmark evidence contract (V1).

This module validates declared evidence, not actual tokenization, rendering, model
execution, layer choice, label truth, or predictor fitting. A future acquisition
adapter needs separate qualification. No model, tokenizer, scheduler, fit or
metric implementation is imported. Every manifest observation x landmark slot is
retained. Predictions on fit/calibration groups are explicit not_evaluation slots.

The landmark t counts emitted non-EOS tokens. Its input is exactly prompt plus
those t tokens; its absolute position is P+t-1, including P-1 at t=0. Only FP32
source/storage with exact little-endian bytes is supported; no cast is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "matched-landmark-v1"
COMPARATORS = ("text", "text_internal")
SPLITS = ("fit", "calibration", "evaluation")
DECODER = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
COUNT_CONVENTION = "emitted_non_eos_tokens"


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    """Strict finite canonical JSON, used for all logical object hashes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def object_hash(value):
    return sha256(canonical(value))


def _keys(value, keys, code):
    _require(type(value) is dict and set(value) == set(keys.split()), code)


def _int(value, minimum=0):
    _require(type(value) is int and value >= minimum, "integer")


def _str(value):
    _require(type(value) is str and bool(value), "string")
    value.encode("utf-8")


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "sha256")


def _ids(value, vocab, *, nonempty=False):
    _require(type(value) is list and (bool(value) or not nonempty), "token_ids")
    for token in value:
        _int(token)
        _require(token < vocab, "token_range")


def _utf8(value):
    _require(type(value) in (bytes, bytearray), "utf8_bytes")
    value = bytes(value)
    value.decode("utf-8")
    return value


def _snapshot(value):
    return json.loads(canonical(value))


def validate_manifest(manifest):
    """Validate and defensively copy a prospectively fixed finite manifest."""
    _keys(
        manifest,
        "schema_version study_id provenance vocab_size eos_token_ids "
        "generation landmarks count_convention decoder capture groups observations "
        "endpoint comparators",
        "manifest_fields",
    )
    _require(manifest["schema_version"] == SCHEMA, "manifest_schema")
    _str(manifest["study_id"])
    _keys(
        manifest["provenance"],
        "model_sha256 tokenizer_sha256 chat_template_sha256 capture_code_sha256",
        "provenance_fields",
    )
    for value in manifest["provenance"].values():
        _hash(value)
    _int(manifest["vocab_size"], 1)
    eos = manifest["eos_token_ids"]
    _ids(eos, manifest["vocab_size"], nonempty=True)
    _require(len(set(eos)) == len(eos), "duplicate_eos")
    gen = manifest["generation"]
    _keys(gen, "policy_sha256 seed max_new_tokens", "generation_policy_fields")
    _hash(gen["policy_sha256"])
    _int(gen["seed"])
    _int(gen["max_new_tokens"], 1)
    landmarks = manifest["landmarks"]
    _require(type(landmarks) is list and bool(landmarks), "landmarks")
    for landmark in landmarks:
        _int(landmark)
        _require(landmark <= gen["max_new_tokens"], "landmark_beyond_cap")
    _require(landmarks == sorted(set(landmarks)), "landmark_order")
    _require(manifest["count_convention"] == COUNT_CONVENTION, "count_convention")
    _require(canonical(manifest["decoder"]) == canonical(DECODER), "decoder")
    cap = manifest["capture"]
    _keys(
        cap,
        "layer_index model_layers hidden_width site source_dtype storage_dtype "
        "byte_order attention_policy position_policy cache_policy",
        "capture_policy_fields",
    )
    _int(cap["layer_index"])
    _int(cap["model_layers"], 1)
    _int(cap["hidden_width"], 1)
    _require(cap["layer_index"] < cap["model_layers"], "layer_range")
    for key, expected in {
        "site": "residual_post",
        "source_dtype": "float32",
        "storage_dtype": "float32",
        "byte_order": "little",
        "attention_policy": "all_ones",
        "position_policy": "absolute_zero_based",
        "cache_policy": "none",
    }.items():
        _require(cap[key] == expected, "capture_policy")
    groups = manifest["groups"]
    _require(type(groups) is dict and bool(groups), "groups")
    for group, split in groups.items():
        _str(group)
        _require(split in SPLITS, "split")
    observations = manifest["observations"]
    _require(type(observations) is list and bool(observations), "observations")
    seen, used, cores = set(), set(), {}
    for obs in observations:
        _keys(obs, "observation_id core_id group_id split strata", "planned_observation_fields")
        for key in ("observation_id", "core_id", "group_id"):
            _str(obs[key])
        _require(obs["observation_id"] not in seen, "duplicate_observation")
        seen.add(obs["observation_id"])
        _require(groups.get(obs["group_id"]) == obs["split"], "group_split")
        used.add(obs["group_id"])
        _require(
            cores.setdefault(obs["core_id"], obs["group_id"]) == obs["group_id"],
            "core_group_leakage",
        )
        _require(type(obs["strata"]) is dict, "strata")
        for key, value in obs["strata"].items():
            _str(key)
            _str(value)
    _require(used == set(groups), "unused_group")
    endpoint = manifest["endpoint"]
    _keys(
        endpoint,
        "instrument_sha256 target_sha256 horizon_tokens uncertainty_policy_sha256",
        "endpoint_fields",
    )
    for key in ("instrument_sha256", "target_sha256", "uncertainty_policy_sha256"):
        _hash(endpoint[key])
    _int(endpoint["horizon_tokens"], 1)
    _keys(manifest["comparators"], "text text_internal", "comparators")
    for comp in manifest["comparators"].values():
        _keys(
            comp,
            "model_sha256 transform_sha256 train_manifest_sha256 calibration_manifest_sha256 "
            "fit_group_ids calibration_group_ids fit_budget calibration_budget",
            "comparator_fields",
        )
        for key in (
            "model_sha256",
            "transform_sha256",
            "train_manifest_sha256",
            "calibration_manifest_sha256",
        ):
            _hash(comp[key])
        for key, split in (("fit_group_ids", "fit"), ("calibration_group_ids", "calibration")):
            _require(
                type(comp[key]) is list
                and comp[key] == sorted(group for group in groups if groups[group] == split),
                "comparator_groups",
            )
        for key in ("fit_budget", "calibration_budget"):
            _int(comp[key])
    return _snapshot(manifest)


@dataclass(frozen=True)
class ValidatedBundle:
    """Immutable bytes, never a mutable caller-owned dict/tensor buffer."""

    metadata_bytes: bytes
    blob_items: tuple[tuple[str, bytes], ...]

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    @property
    def blobs(self):
        return dict(self.blob_items)

    @property
    def sha256(self):
        return sha256(self.metadata_bytes)


def _generation(manifest, generation):
    _keys(
        generation,
        "status stop_reason token_ids observed_tokens attempt_sha256 dispatch_sha256",
        "generation_fields",
    )
    ids = generation["token_ids"]
    _ids(ids, manifest["vocab_size"])
    _int(generation["observed_tokens"])
    _require(generation["observed_tokens"] == len(ids), "observed_length")
    _require(len(ids) <= manifest["generation"]["max_new_tokens"], "generation_cap")
    eos = manifest["eos_token_ids"]
    _require(not any(token in eos for token in ids[:-1]), "early_eos_in_stream")
    terminal = bool(ids and ids[-1] in eos)
    status, reason = generation["status"], generation["stop_reason"]
    _require(
        status in ("completed", "interrupted", "infrastructure_failed", "missing"),
        "generation_status",
    )
    if status == "completed":
        _require(
            (reason == "eos" and terminal)
            or (
                reason == "cap"
                and not terminal
                and len(ids) == manifest["generation"]["max_new_tokens"]
            ),
            "generation_stop",
        )
    else:
        _require(reason == status and not terminal, "generation_stop")
    for key in ("attempt_sha256", "dispatch_sha256"):
        if generation[key] is not None:
            _hash(generation[key])
    if status == "missing":
        _require(
            not ids
            and generation["attempt_sha256"] is None
            and generation["dispatch_sha256"] is None,
            "missing_generation",
        )
    else:
        _hash(generation["attempt_sha256"])
        if status != "infrastructure_failed" or ids:
            _hash(generation["dispatch_sha256"])
    if generation["dispatch_sha256"] is not None:
        _hash(generation["attempt_sha256"])
    return ids[:-1] if terminal else ids, terminal


def _capture(manifest, obs, prefix, metadata, raw, prompt, prefix_ids, blobs):
    _keys(metadata, "status reason evidence", "capture_fields")
    status = metadata["status"]
    _require(
        status in ("completed", "infrastructure_failed", "missing", "unavailable"), "capture_status"
    )
    if prefix["status"] != "available":
        _require(
            status == "unavailable" and metadata["reason"] == "prefix_unavailable",
            "capture_without_prefix",
        )
    else:
        _require(status != "unavailable", "available_prefix_capture_status")
    if status != "completed":
        _require(metadata["evidence"] is None and raw is None, "unexpected_capture")
        _require(
            metadata["reason"] in ("prefix_unavailable", "not_attempted", "capture_failed"),
            "capture_reason",
        )
        _require(
            metadata["reason"]
            == {
                "unavailable": "prefix_unavailable",
                "missing": "not_attempted",
                "infrastructure_failed": "capture_failed",
            }[status],
            "capture_reason_status",
        )
        return
    _require(metadata["reason"] is None, "completed_capture_reason")
    evidence = metadata["evidence"]
    _keys(
        evidence,
        "observation_id landmark prompt_ids_sha256 prefix_ids_sha256 "
        "prompt_utf8_sha256 prefix_utf8_sha256 input_token_ids input_ids_sha256 "
        "input_length attention_mask position_ids absolute_position policy provenance "
        "shape byte_length content_sha256 forward_receipt",
        "capture_evidence_fields",
    )
    _require(
        evidence["observation_id"] == obs["observation_id"]
        and type(evidence["landmark"]) is int
        and evidence["landmark"] == prefix["landmark"],
        "capture_identity",
    )
    expected_input = prompt + prefix_ids
    _ids(evidence["input_token_ids"], manifest["vocab_size"], nonempty=True)
    _require(evidence["input_token_ids"] == expected_input, "capture_boundary")
    for key, expected in {
        "prompt_ids_sha256": object_hash(prompt),
        "prefix_ids_sha256": object_hash(prefix_ids),
        "prompt_utf8_sha256": sha256(blobs["prompt.utf8"]),
        "prefix_utf8_sha256": sha256(blobs["prefix.utf8"]),
        "input_ids_sha256": object_hash(expected_input),
    }.items():
        _require(evidence[key] == expected, "capture_input_hash")
    _int(evidence["input_length"], 1)
    _int(evidence["absolute_position"])
    _require(
        evidence["input_length"] == len(expected_input)
        and evidence["absolute_position"] == len(expected_input) - 1,
        "capture_position",
    )
    for key, expected in (
        ("attention_mask", [1] * len(expected_input)),
        ("position_ids", list(range(len(expected_input)))),
    ):
        _require(
            type(evidence[key]) is list
            and all(type(x) is int for x in evidence[key])
            and evidence[key] == expected,
            "capture_attention_position",
        )
    _require(
        canonical(evidence["policy"]) == canonical(manifest["capture"]), "capture_policy_binding"
    )
    _require(
        canonical(evidence["provenance"]) == canonical(manifest["provenance"]), "capture_provenance"
    )
    _require(
        type(evidence["shape"]) is list
        and all(type(x) is int for x in evidence["shape"])
        and evidence["shape"] == [1, manifest["capture"]["hidden_width"]],
        "capture_shape",
    )
    _require(type(raw) in (bytes, bytearray), "capture_bytes")
    raw = bytes(raw)
    _int(evidence["byte_length"], 1)
    _require(
        len(raw) == evidence["byte_length"] == 4 * manifest["capture"]["hidden_width"],
        "capture_byte_length",
    )
    _require(sha256(raw) == evidence["content_sha256"], "capture_content_hash")
    _require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", raw)), "capture_nonfinite")
    receipt = evidence["forward_receipt"]
    _keys(
        receipt,
        "attempt_sha256 dispatch_sha256 source_sha256 input_ids_sha256",
        "forward_receipt_fields",
    )
    for value in receipt.values():
        _hash(value)
    _require(
        receipt["source_sha256"] == manifest["provenance"]["capture_code_sha256"]
        and receipt["input_ids_sha256"] == evidence["input_ids_sha256"],
        "forward_receipt_binding",
    )
    blobs["capture.fp32"] = raw


def validate_landmark(
    manifest, observation, generation, prefix, capture_metadata, capture_bytes, outcome
):
    """Validate a single slot; complete-cohort checking occurs before predictions."""
    manifest = validate_manifest(manifest)
    _keys(
        observation,
        "observation_id messages rendered_prompt_utf8 prompt_token_ids",
        "observation_fields",
    )
    planned = {obs["observation_id"]: obs for obs in manifest["observations"]}
    _require(observation["observation_id"] in planned, "unplanned_observation")
    _require(type(observation["messages"]) is list and bool(observation["messages"]), "messages")
    for message in observation["messages"]:
        _keys(message, "role content", "message_fields")
        _require(message["role"] in ("system", "user", "assistant"), "message_role")
        _require(type(message["content"]) is str, "message_content")
        message["content"].encode("utf-8")
    prompt = observation["prompt_token_ids"]
    _ids(prompt, manifest["vocab_size"], nonempty=True)
    blobs = {"prompt.utf8": _utf8(observation["rendered_prompt_utf8"])}
    _require(bool(blobs["prompt.utf8"]), "empty_prompt")
    content, terminal_eos = _generation(manifest, generation)
    _keys(prefix, "landmark status reason token_ids decoded_utf8", "prefix_fields")
    t = prefix["landmark"]
    _int(t)
    _require(t in manifest["landmarks"], "unplanned_landmark")
    expected_status = (
        "available"
        if len(content) >= t
        else "unreached"
        if generation["status"] == "completed"
        else "missing"
    )
    _require(prefix["status"] == expected_status, "prefix_status")
    if expected_status == "available":
        _ids(prefix["token_ids"], manifest["vocab_size"])
        _require(prefix["token_ids"] == content[:t] and prefix["reason"] is None, "prefix_slice")
        blobs["prefix.utf8"] = _utf8(prefix["decoded_utf8"])
        if t == 0:
            _require(not blobs["prefix.utf8"], "t0_text")
    else:
        _require(
            prefix["token_ids"] is None and prefix["decoded_utf8"] is None,
            "unavailable_prefix_data",
        )
        _require(
            prefix["reason"]
            == (
                "early_eos"
                if expected_status == "unreached"
                else "generation_" + generation["status"]
            ),
            "prefix_reason",
        )
    _capture(
        manifest,
        observation,
        prefix,
        capture_metadata,
        capture_bytes,
        prompt,
        prefix["token_ids"],
        blobs,
    )
    _keys(
        outcome,
        "status applicable label reason disagreement endpoint generation_sha256 "
        "terminal_eos capped censored",
        "outcome_fields",
    )
    _require(canonical(outcome["endpoint"]) == canonical(manifest["endpoint"]), "outcome_endpoint")
    _require(outcome["generation_sha256"] == object_hash(generation), "outcome_generation")
    _require(
        type(outcome["terminal_eos"]) is bool and outcome["terminal_eos"] == terminal_eos,
        "outcome_eos",
    )
    capped = generation["stop_reason"] == "cap"
    _require(type(outcome["capped"]) is bool and outcome["capped"] == capped, "outcome_cap")
    censored = not terminal_eos and len(content) < manifest["endpoint"]["horizon_tokens"]
    _require(
        type(outcome["censored"]) is bool and outcome["censored"] == censored, "outcome_censoring"
    )
    status = outcome["status"]
    _require(status in ("known", "unknown", "not_applicable", "missing"), "outcome_status")
    _require(
        outcome["disagreement"] is None or type(outcome["disagreement"]) is bool, "disagreement"
    )
    if status == "known":
        _require(
            outcome["applicable"] is True
            and type(outcome["label"]) is int
            and outcome["label"] in (0, 1)
            and outcome["reason"] is None,
            "known_outcome",
        )
    else:
        _require(outcome["label"] is None, "unknown_label")
        expected_reason = {
            "unknown": ("uncertain", "disagreement", "censored"),
            "not_applicable": ("not_applicable",),
            "missing": ("not_measured",),
        }[status]
        _require(outcome["reason"] in expected_reason, "outcome_reason")
        _require(
            outcome["applicable"] is False
            if status == "not_applicable"
            else outcome["applicable"] is None or outcome["applicable"] is True,
            "outcome_applicability",
        )
        if outcome["reason"] == "disagreement":
            _require(outcome["disagreement"] is True, "outcome_disagreement")
        if outcome["reason"] == "censored":
            _require(censored, "outcome_censored_reason")
    obs_json = {key: value for key, value in observation.items() if key != "rendered_prompt_utf8"}
    obs_json.update(
        prompt_utf8_sha256=sha256(blobs["prompt.utf8"]), prompt_ids_sha256=object_hash(prompt)
    )
    pre_json = {key: value for key, value in prefix.items() if key != "decoded_utf8"}
    pre_json.update(
        decoded_utf8_sha256=sha256(blobs["prefix.utf8"]) if "prefix.utf8" in blobs else None,
        token_ids_sha256=object_hash(prefix["token_ids"])
        if prefix["token_ids"] is not None
        else None,
    )
    metadata = {
        "schema_version": SCHEMA,
        "manifest_sha256": object_hash(manifest),
        "planned": planned[observation["observation_id"]],
        "observation": obs_json,
        "generation": generation,
        "prefix": pre_json,
        "capture": capture_metadata,
        "outcome": outcome,
        "blobs": {name: sha256(raw) for name, raw in sorted(blobs.items())},
    }
    return ValidatedBundle(canonical(metadata), tuple(sorted(blobs.items())))


def _revalidate(manifest, bundle):
    _require(
        type(bundle) is ValidatedBundle
        and type(bundle.metadata_bytes) is bytes
        and type(bundle.blob_items) is tuple,
        "bundle_type",
    )
    for item in bundle.blob_items:
        _require(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bytes,
            "bundle_blob_type",
        )
    _require(len(dict(bundle.blob_items)) == len(bundle.blob_items), "duplicate_blob")
    meta, blobs = bundle.metadata, bundle.blobs
    _keys(
        meta,
        "schema_version manifest_sha256 planned observation generation prefix capture outcome blobs",
        "bundle_fields",
    )
    obs = dict(meta["observation"])
    obs.pop("prompt_utf8_sha256")
    obs.pop("prompt_ids_sha256")
    obs["rendered_prompt_utf8"] = blobs["prompt.utf8"]
    pre = dict(meta["prefix"])
    pre.pop("decoded_utf8_sha256")
    pre.pop("token_ids_sha256")
    pre["decoded_utf8"] = blobs.get("prefix.utf8")
    fresh = validate_landmark(
        manifest,
        obs,
        meta["generation"],
        pre,
        meta["capture"],
        blobs.get("capture.fp32"),
        meta["outcome"],
    )
    _require(bundle == fresh, "bundle_integrity")
    return fresh


def predictor_view(bundle, comparator="text"):
    """Only feature content and its content hash; no joining or outcome envelope.

    Bundles should first come from validate_landmark/load_landmark. The writer and
    cohort validator always revalidate even manually constructed dataclass values.
    """
    _require(
        type(bundle) is ValidatedBundle
        and type(bundle.metadata_bytes) is bytes
        and type(bundle.blob_items) is tuple
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is bytes
            for item in bundle.blob_items
        )
        and comparator in COMPARATORS,
        "predictor_view",
    )
    meta, blobs = bundle.metadata, bundle.blobs
    _require(meta["prefix"]["status"] == "available", "prefix_unavailable")
    features = {
        "rendered_prompt_utf8": blobs["prompt.utf8"],
        "prompt_token_ids": tuple(meta["observation"]["prompt_token_ids"]),
        "prefix_utf8": blobs["prefix.utf8"],
        "prefix_token_ids": tuple(meta["prefix"]["token_ids"]),
    }
    if comparator == "text_internal":
        _require(meta["capture"]["status"] == "completed", "capture_unavailable")
        features["residual_fp32le"] = blobs["capture.fp32"]
        features["residual_shape"] = tuple(meta["capture"]["evidence"]["shape"])
    binding = {
        key: {"bytes_sha256": sha256(value)} if type(value) is bytes else value
        for key, value in features.items()
    }
    return {"features": features, "features_sha256": object_hash(binding)}


def _cohort(manifest, bundles):
    _require(type(bundles) is list, "bundle_cohort")
    expected = [
        (obs["observation_id"], t)
        for obs in manifest["observations"]
        for t in manifest["landmarks"]
    ]
    _require(len(bundles) == len(expected), "bundle_coverage")
    indexed, shared = {}, {}
    for key, bundle in zip(expected, bundles, strict=True):
        bundle = _revalidate(manifest, bundle)
        meta = bundle.metadata
        actual = (meta["observation"]["observation_id"], meta["prefix"]["landmark"])
        _require(key == actual, "bundle_order")
        identity = canonical(
            {name: meta[name] for name in ("observation", "generation", "outcome")}
        )
        _require(
            shared.setdefault(key[0], identity) == identity, "inconsistent_observation_landmarks"
        )
        indexed[key] = bundle
    return expected, indexed


@dataclass(frozen=True)
class PredictionPacket:
    metadata_bytes: bytes

    @property
    def metadata(self):
        return json.loads(self.metadata_bytes)

    @property
    def sha256(self):
        return sha256(self.metadata_bytes)


def validate_prediction_pairs(manifest, bundles, rows, *, keyed=False):
    """Retain both comparator slots for every planned key; compute no metrics.

    keyed=True explicitly checks a unique complete key join then restores manifest
    order. The default rejects reordered rows. Unknown labels do not erase a valid
    prediction; prefix/internal availability and fitting failures stay separate.
    """
    manifest = validate_manifest(manifest)
    _require(type(keyed) is bool, "keyed_boolean")
    expected, indexed = _cohort(manifest, bundles)
    _require(type(rows) is list and len(rows) == len(expected), "prediction_coverage")
    by_key, actual_order, fit_statuses = {}, [], {}
    for row in rows:
        _keys(
            row,
            "observation_id landmark group_id split bundle_sha256 outcome_sha256 predictions",
            "prediction_row_fields",
        )
        _str(row["observation_id"])
        _int(row["landmark"])
        key = (row["observation_id"], row["landmark"])
        _require(key in indexed and key not in by_key, "prediction_key")
        bundle = indexed[key]
        meta = bundle.metadata
        _require(
            row["group_id"] == meta["planned"]["group_id"]
            and row["split"] == meta["planned"]["split"],
            "prediction_group",
        )
        _require(
            row["bundle_sha256"] == bundle.sha256
            and row["outcome_sha256"] == object_hash(meta["outcome"]),
            "prediction_evidence_binding",
        )
        _keys(row["predictions"], "text text_internal", "prediction_slots")
        for name, slot in row["predictions"].items():
            _keys(
                slot,
                "comparator_sha256 features_sha256 eligibility probability null_reason fit_status",
                "prediction_slot_fields",
            )
            _require(
                slot["comparator_sha256"] == object_hash(manifest["comparators"][name]),
                "prediction_comparator",
            )
            availability = (
                "prefix_unavailable"
                if meta["prefix"]["status"] != "available"
                else "capture_unavailable"
                if name == "text_internal" and meta["capture"]["status"] != "completed"
                else "eligible"
            )
            eligibility = "not_evaluation" if row["split"] != "evaluation" else availability
            _require(slot["eligibility"] == eligibility, "prediction_eligibility")
            view_hash = (
                predictor_view(bundle, name)["features_sha256"]
                if availability == "eligible"
                else None
            )
            _require(slot["features_sha256"] == view_hash, "prediction_features")
            _require(slot["fit_status"] in ("fitted", "failed", "not_attempted"), "fit_status")
            _require(
                fit_statuses.setdefault(name, slot["fit_status"]) == slot["fit_status"],
                "inconsistent_comparator_fit_status",
            )
            probability = slot["probability"]
            if probability is not None:
                _require(
                    type(probability) in (int, float)
                    and math.isfinite(probability)
                    and 0 <= probability <= 1,
                    "probability",
                )
                _require(
                    eligibility == "eligible"
                    and slot["fit_status"] == "fitted"
                    and slot["null_reason"] is None,
                    "probability_availability",
                )
            else:
                reason = (
                    eligibility
                    if eligibility != "eligible"
                    else {
                        "failed": "fit_failed",
                        "not_attempted": "fit_not_attempted",
                        "fitted": "prediction_failed",
                    }[slot["fit_status"]]
                )
                _require(slot["null_reason"] == reason, "null_prediction_reason")
        by_key[key] = row
        actual_order.append(key)
    _require(keyed or actual_order == expected, "prediction_order")
    packet = {
        "schema_version": SCHEMA,
        "manifest_sha256": object_hash(manifest),
        "rows": [by_key[key] for key in expected],
        "planned_rows": len(expected),
        "planned_comparator_slots": 2 * len(expected),
    }
    return PredictionPacket(canonical(packet))


def _directory(path, *, create=False):
    path = Path(path)
    _require(path.is_absolute(), "absolute_private_directory")
    for parent in reversed([path, *path.parents]):
        if parent.exists() or parent.is_symlink():
            _require(stat.S_ISDIR(parent.lstat().st_mode), "directory_symlink_or_special")
    if create:
        path.mkdir(mode=0o700)  # Exclusive; a failed publication remains consumed.
        _fsync_dir(path.parent)
    else:
        _require(path.is_dir(), "missing_directory")
    return path


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(path, raw):
    """Publish fsynced bytes by exclusive link; interrupted temporaries are orphans.

    This is not a multi-file transaction. A concurrent reader does not establish
    that an in-progress writer has returned from its final directory fsync.
    """
    temporary = path.with_name("." + path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
    finally:
        _fsync_dir(path.parent)


def _read_file(path):
    _require(stat.S_ISREG(path.lstat().st_mode), "file_symlink_or_special")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "opened_special_file")
        return stream.read()


def _read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")),
    )
    _require(canonical(value) == raw, "noncanonical_json")
    return value


def _commit(directory, kind, manifest, children):
    directory = _directory(directory, create=True)
    claim = {"schema_version": SCHEMA, "kind": kind, "manifest_sha256": object_hash(manifest)}
    _publish(directory / "claim.json", canonical(claim))
    for name, raw in sorted(children.items()):
        _publish(directory / name, raw)
    receipt = {**claim, "files": {name: sha256(raw) for name, raw in sorted(children.items())}}
    _publish(directory / "receipt.json", canonical(receipt))
    return {"receipt_sha256": object_hash(receipt), "files": len(children), "kind": kind}


def _load(directory, kind, manifest):
    directory = _directory(directory)
    receipt = _read_json(_read_file(directory / "receipt.json"))
    _keys(receipt, "schema_version kind manifest_sha256 files", "receipt_fields")
    expected_claim = {
        "schema_version": SCHEMA,
        "kind": kind,
        "manifest_sha256": object_hash(manifest),
    }
    _require({key: receipt[key] for key in expected_claim} == expected_claim, "receipt_binding")
    _require(_read_file(directory / "claim.json") == canonical(expected_claim), "claim_binding")
    _require(
        type(receipt["files"]) is dict
        and all(
            type(name) is str
            and re.fullmatch(r"[a-z][a-z_.0-9]*", name) is not None
            and name not in ("claim.json", "receipt.json")
            for name in receipt["files"]
        ),
        "receipt_inventory",
    )
    _require(
        {p.name for p in directory.iterdir()}
        == set(receipt["files"]) | {"claim.json", "receipt.json"},
        "unexpected_file_inventory",
    )
    children = {}
    for name, digest in receipt["files"].items():
        _hash(digest)
        children[name] = _read_file(directory / name)
        _require(sha256(children[name]) == digest, "child_hash")
    return children


def commit_landmark(directory, bundle, manifest):
    """Create a new private directory; any previous/partial directory blocks reuse."""
    manifest = validate_manifest(manifest)
    bundle = _revalidate(manifest, bundle)
    children = {"bundle.json": bundle.metadata_bytes, **bundle.blobs}
    return _commit(directory, "landmark", manifest, children)


def load_landmark(directory, manifest):
    manifest = validate_manifest(manifest)
    children = _load(directory, "landmark", manifest)
    _require("bundle.json" in children, "missing_bundle")
    metadata = children.pop("bundle.json")
    _read_json(metadata)
    return _revalidate(manifest, ValidatedBundle(metadata, tuple(sorted(children.items()))))


def commit_prediction_pairs(directory, packet, manifest, bundles):
    manifest = validate_manifest(manifest)
    _require(
        type(packet) is PredictionPacket and type(packet.metadata_bytes) is bytes,
        "prediction_packet",
    )
    fresh = validate_prediction_pairs(manifest, bundles, packet.metadata["rows"])
    _require(packet == fresh, "prediction_packet_integrity")
    return _commit(directory, "predictions", manifest, {"predictions.json": packet.metadata_bytes})


def load_prediction_pairs(directory, manifest, bundles):
    manifest = validate_manifest(manifest)
    children = _load(directory, "predictions", manifest)
    _require(set(children) == {"predictions.json"}, "prediction_inventory")
    raw = children["predictions.json"]
    meta = _read_json(raw)
    fresh = validate_prediction_pairs(manifest, bundles, meta["rows"])
    _require(raw == fresh.metadata_bytes, "prediction_packet_integrity")
    return fresh
