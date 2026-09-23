"""Independent invented A183 evidence-contract adversarial qualification.

No source corpus, tokenizer, model, encoder, or fitting operation is used.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import copy
import hashlib
import json
import struct

import pytest

from lexical_prompt_study import landmark_evidence as evidence


def test_contract_module_dependencies_are_standard_library_only():
    path = Path(__file__).parents[1] / "src/lexical_prompt_study/landmark_evidence.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(name.name.split(".")[0] in sys.stdlib_module_names for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in sys.stdlib_module_names


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def object_digest(value):
    return digest(encoded(value))


def example_manifest():
    groups = {"core-fit": "fit", "core-cal": "calibration", "core-eval": "evaluation"}
    provenance = {
        key: digest(key.encode())
        for key in (
            "model_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "capture_code_sha256",
        )
    }
    comparator = {
        key: digest(key.encode())
        for key in (
            "model_sha256",
            "transform_sha256",
            "train_manifest_sha256",
            "calibration_manifest_sha256",
        )
    }
    comparator.update(
        fit_group_ids=["core-fit"],
        calibration_group_ids=["core-cal"],
        fit_budget=1,
        calibration_budget=1,
    )
    return {
        "schema_version": "matched-landmark-v1",
        "study_id": "invented-independent",
        "provenance": provenance,
        "vocab_size": 100,
        "eos_token_ids": [99],
        "generation": {"policy_sha256": digest(b"invented-policy"), "seed": 0, "max_new_tokens": 4},
        "landmarks": [0, 2],
        "count_convention": "emitted_non_eos_tokens",
        "decoder": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        "capture": {
            "layer_index": 0,
            "model_layers": 2,
            "hidden_width": 2,
            "site": "residual_post",
            "source_dtype": "float32",
            "storage_dtype": "float32",
            "byte_order": "little",
            "attention_policy": "all_ones",
            "position_policy": "absolute_zero_based",
            "cache_policy": "none",
        },
        "groups": groups,
        "observations": [
            {
                "observation_id": "row-" + str(i),
                "core_id": key,
                "group_id": key,
                "split": split,
                "strata": {"placement": "invented"},
            }
            for i, (key, split) in enumerate(groups.items())
        ],
        "endpoint": {
            "instrument_sha256": digest(b"instrument"),
            "target_sha256": digest(b"target"),
            "horizon_tokens": 4,
            "uncertainty_policy_sha256": digest(b"uncertainty"),
        },
        "comparators": {
            "text": copy.deepcopy(comparator),
            "text_internal": copy.deepcopy(comparator),
        },
    }


def example_slot(manifest, row=2, landmark=2, *, tokens=None, reason="cap", label=0, capture=True):
    """Invented declarations; construction uses no runtime or author fixture helpers."""
    tokens = [10, 11, 12, 13] if tokens is None else list(tokens)
    observation = {
        "observation_id": manifest["observations"][row]["observation_id"],
        "messages": [{"role": "user", "content": "invented π  \n"}],
        "rendered_prompt_utf8": b"rendered \xcf\x80  \n",
        "prompt_token_ids": [1, 2, 3],
    }
    status = "completed" if reason in ("cap", "eos") else reason
    generation = {
        "status": status,
        "stop_reason": reason,
        "token_ids": tokens,
        "observed_tokens": len(tokens),
        "attempt_sha256": digest(b"generation attempt"),
        "dispatch_sha256": digest(b"generation dispatch"),
    }
    if reason == "missing":
        generation["attempt_sha256"] = generation["dispatch_sha256"] = None
    eos = bool(tokens and tokens[-1] == 99)
    content = tokens[:-1] if eos else tokens
    reached = len(content) >= landmark
    prefix = {
        "landmark": landmark,
        "status": "available" if reached else "unreached" if reason == "eos" else "missing",
        "reason": None if reached else "early_eos" if reason == "eos" else "generation_" + reason,
        "token_ids": content[:landmark] if reached else None,
        "decoded_utf8": (b"" if landmark == 0 else b"\xef\xbf\xbd  \n") if reached else None,
    }
    metadata = {
        "status": "completed" if capture and reached else "missing" if reached else "unavailable",
        "reason": None
        if capture and reached
        else "not_attempted"
        if reached
        else "prefix_unavailable",
        "evidence": None,
    }
    raw = bytes.fromhex("000000800000c03f") if capture and reached else None  # -0.0 and 1.5
    if raw is not None:
        inputs = observation["prompt_token_ids"] + prefix["token_ids"]
        metadata["evidence"] = {
            "observation_id": observation["observation_id"],
            "landmark": landmark,
            "prompt_ids_sha256": object_digest(observation["prompt_token_ids"]),
            "prefix_ids_sha256": object_digest(prefix["token_ids"]),
            "prompt_utf8_sha256": digest(observation["rendered_prompt_utf8"]),
            "prefix_utf8_sha256": digest(prefix["decoded_utf8"]),
            "input_token_ids": inputs,
            "input_ids_sha256": object_digest(inputs),
            "input_length": len(inputs),
            "attention_mask": [1] * len(inputs),
            "position_ids": list(range(len(inputs))),
            "absolute_position": len(inputs) - 1,
            "policy": copy.deepcopy(manifest["capture"]),
            "provenance": copy.deepcopy(manifest["provenance"]),
            "shape": [1, 2],
            "byte_length": 8,
            "content_sha256": digest(raw),
            "forward_receipt": {
                "attempt_sha256": digest(b"capture attempt"),
                "dispatch_sha256": digest(b"capture dispatch"),
                "source_sha256": manifest["provenance"]["capture_code_sha256"],
                "input_ids_sha256": object_digest(inputs),
            },
        }
    censored = not eos and len(content) < manifest["endpoint"]["horizon_tokens"]
    outcome = {
        "status": "known" if label is not None else "unknown",
        "applicable": True,
        "label": label,
        "reason": None if label is not None else "uncertain",
        "disagreement": None,
        "endpoint": copy.deepcopy(manifest["endpoint"]),
        "generation_sha256": object_digest(generation),
        "terminal_eos": eos,
        "capped": reason == "cap",
        "censored": censored,
    }
    return [manifest, observation, generation, prefix, metadata, raw, outcome]


def bundle(*args, **kwargs):
    return evidence.validate_landmark(*example_slot(*args, **kwargs))


def examples():
    manifest = example_manifest()
    bundles = [
        bundle(manifest, row, t, capture=(row, t) != (2, 2)) for row in range(3) for t in (0, 2)
    ]
    rows = []
    for b in bundles:
        meta = b.metadata
        slots = {}
        for name in ("text", "text_internal"):
            available = name == "text" or meta["capture"]["status"] == "completed"
            eligibility = (
                "not_evaluation"
                if meta["planned"]["split"] != "evaluation"
                else ("eligible" if available else "capture_unavailable")
            )
            features = evidence.predictor_view(b, name)["features_sha256"] if available else None
            slots[name] = {
                "comparator_sha256": object_digest(manifest["comparators"][name]),
                "features_sha256": features,
                "eligibility": eligibility,
                "probability": 0.25
                if name == "text" and eligibility == "eligible"
                else 0.75
                if eligibility == "eligible"
                else None,
                "null_reason": None if eligibility == "eligible" else eligibility,
                "fit_status": "fitted",
            }
        rows.append(
            {
                "observation_id": meta["observation"]["observation_id"],
                "landmark": meta["prefix"]["landmark"],
                "group_id": meta["planned"]["group_id"],
                "split": meta["planned"]["split"],
                "bundle_sha256": b.sha256,
                "outcome_sha256": object_digest(meta["outcome"]),
                "predictions": slots,
            }
        )
    return manifest, bundles, rows


def test_future_terminal_outcome_changes_cannot_change_features():
    manifest = example_manifest()
    capped = bundle(manifest)
    eos = bundle(manifest, tokens=[10, 11, 99], reason="eos", label=1)
    unknown = bundle(manifest, tokens=[10, 11, 42, 99], reason="eos", label=None)
    assert len({b.sha256 for b in (capped, eos, unknown)}) == 3
    for comparator in ("text", "text_internal"):
        views = [evidence.predictor_view(b, comparator) for b in (capped, eos, unknown)]
        assert views[0] == views[1] == views[2]
        assert set(views[0]) == {"features", "features_sha256"}
    text = evidence.predictor_view(capped)["features"]
    assert set(text) == {
        "rendered_prompt_utf8",
        "prompt_token_ids",
        "prefix_utf8",
        "prefix_token_ids",
    }
    assert text["prefix_utf8"] == b"\xef\xbf\xbd  \n"
    assert text["prefix_token_ids"] == (10, 11)


@pytest.mark.parametrize("t,expected", [(0, 2), (2, 4)])
def test_prefix_only_site_is_actual_last_input(t, expected):
    args = example_slot(example_manifest(), landmark=t)
    checked = evidence.validate_landmark(*args)
    assert checked.metadata["capture"]["evidence"]["absolute_position"] == expected
    assert checked.metadata["capture"]["evidence"]["input_token_ids"] == [1, 2, 3] + [10, 11][:t]
    args[4]["evidence"]["input_token_ids"].append(12)
    # Rehashing the altered forward must not bypass the boundary rule.
    cap = args[4]["evidence"]
    cap["input_ids_sha256"] = object_digest(cap["input_token_ids"])
    cap["input_length"] += 1
    cap["absolute_position"] += 1
    cap["attention_mask"].append(1)
    cap["position_ids"].append(cap["position_ids"][-1] + 1)
    cap["forward_receipt"]["input_ids_sha256"] = cap["input_ids_sha256"]
    with pytest.raises(ValueError, match="capture_boundary"):
        evidence.validate_landmark(*args)


@pytest.mark.parametrize(
    "tokens,reason,status",
    [
        ([10, 99], "eos", "unreached"),
        ([10, 11, 99], "eos", "available"),
        ([10, 11, 12, 13], "cap", "available"),
        ([10], "interrupted", "missing"),
        ([10, 11], "interrupted", "available"),
        ([], "missing", "missing"),
    ],
)
def test_eos_cap_interruption_and_missing_distinctions(tokens, reason, status):
    b = bundle(example_manifest(), tokens=tokens, reason=reason, label=None, capture=False)
    assert b.metadata["prefix"]["status"] == status
    assert b.metadata["generation"]["token_ids"] == tokens
    assert b.metadata["outcome"]["terminal_eos"] is (reason == "eos")
    if status == "available":
        assert evidence.predictor_view(b)["features"]["prefix_token_ids"] == (10, 11)
        with pytest.raises(ValueError, match="capture_unavailable"):
            evidence.predictor_view(b, "text_internal")
    else:
        with pytest.raises(ValueError, match="prefix_unavailable"):
            evidence.predictor_view(b)


@pytest.mark.parametrize(
    "words", ["0000807f00000000", "000080ff00000000", "0100c07f00000000", "0000000000000000ffff"]
)
def test_raw_nonfinite_and_wrong_width_captures_rejected_even_with_matching_hash(words):
    args = example_slot(example_manifest())
    args[5] = bytes.fromhex(words)
    args[4]["evidence"]["content_sha256"] = digest(args[5])
    args[4]["evidence"]["byte_length"] = len(args[5])
    with pytest.raises(ValueError):
        evidence.validate_landmark(*args)


@pytest.mark.parametrize(
    "field,value",
    [("source_dtype", "bfloat16"), ("storage_dtype", "float16"), ("byte_order", "big")],
)
def test_unsupported_dtype_never_silently_casts(field, value):
    manifest = example_manifest()
    manifest["capture"][field] = value
    with pytest.raises(ValueError, match="capture_policy"):
        evidence.validate_manifest(manifest)


def test_native_bits_and_defensive_copies_survive_publication(tmp_path):
    args = example_slot(example_manifest())
    args[1]["rendered_prompt_utf8"] = bytearray(args[1]["rendered_prompt_utf8"])
    args[3]["decoded_utf8"] = bytearray(args[3]["decoded_utf8"])
    args[5] = bytearray(args[5])
    validated = evidence.validate_landmark(*args)
    expected = evidence.predictor_view(validated, "text_internal")
    args[1]["prompt_token_ids"][0] = 88
    args[1]["rendered_prompt_utf8"][:] = b"changed"
    args[3]["token_ids"][0] = 88
    args[3]["decoded_utf8"][:] = b"changed"
    args[5][:] = b"changed"
    visible = validated.metadata
    visible["outcome"]["label"] = 1
    assert evidence.predictor_view(validated, "text_internal") == expected
    output = tmp_path / "bundle"
    evidence.commit_landmark(output, validated, args[0])
    loaded = evidence.load_landmark(output, args[0])
    assert loaded == validated
    assert loaded.blobs["capture.fp32"] == bytes.fromhex("000000800000c03f")
    assert struct.unpack("<I", loaded.blobs["capture.fp32"][:4])[0] == 0x80000000
    with pytest.raises((ValueError, FileExistsError)):
        evidence.commit_landmark(output, validated, args[0])


@pytest.mark.parametrize(
    "change", ["same_core_two_groups", "evaluation_in_fit", "duplicate_observation"]
)
def test_manifest_group_and_fit_isolation(change):
    manifest = example_manifest()
    if change == "same_core_two_groups":
        manifest["observations"][2]["core_id"] = manifest["observations"][0]["core_id"]
    elif change == "evaluation_in_fit":
        manifest["comparators"]["text"]["fit_group_ids"].append("core-eval")
    else:
        manifest["observations"][2]["observation_id"] = manifest["observations"][0][
            "observation_id"
        ]
    with pytest.raises(ValueError):
        evidence.validate_manifest(manifest)


def test_complete_paired_retention_verified_permutation_and_explicit_missing_partner(tmp_path):
    manifest, bundles, rows = examples()
    packet = evidence.validate_prediction_pairs(manifest, bundles, rows)
    meta = packet.metadata
    assert meta["planned_rows"] == 6 and meta["planned_comparator_slots"] == 12
    assert meta["rows"] == rows
    last = meta["rows"][-1]["predictions"]
    assert last["text"]["probability"] == 0.25
    assert last["text_internal"]["probability"] is None
    assert last["text_internal"]["null_reason"] == "capture_unavailable"
    reordered = list(reversed(rows))
    with pytest.raises(ValueError, match="prediction_order"):
        evidence.validate_prediction_pairs(manifest, bundles, reordered)
    assert evidence.validate_prediction_pairs(manifest, bundles, reordered, keyed=True) == packet
    path = tmp_path / "predictions"
    evidence.commit_prediction_pairs(path, packet, manifest, bundles)
    assert evidence.load_prediction_pairs(path, manifest, bundles) == packet


@pytest.mark.parametrize(
    "change",
    [
        "missing_bundle",
        "missing_row",
        "duplicate_row",
        "partner_removed",
        "false_zero",
        "wrong_group",
        "future_field",
        "boolean",
        "nan",
    ],
)
def test_successful_subset_and_paired_prediction_tampering_rejected(change):
    manifest, bundles, rows = examples()
    if change == "missing_bundle":
        bundles.pop()
    elif change == "missing_row":
        rows.pop()
    elif change == "duplicate_row":
        rows[-1] = rows[0]
    elif change == "partner_removed":
        del rows[-1]["predictions"]["text_internal"]
    elif change == "false_zero":
        rows[-1]["predictions"]["text_internal"]["probability"] = 0.0
    elif change == "wrong_group":
        rows[-1]["group_id"] = "core-fit"
    elif change == "future_field":
        rows[-1]["predictions"]["text"]["final_answer"] = "future private-looking invented text"
    else:
        rows[-1]["predictions"]["text"]["probability"] = (
            True if change == "boolean" else float("nan")
        )
    with pytest.raises(ValueError):
        evidence.validate_prediction_pairs(manifest, bundles, rows, keyed=True)


@pytest.mark.parametrize("position", ["directory", "child", "extra"])
def test_symlink_and_unlisted_payload_rejected(tmp_path, position):
    manifest = example_manifest()
    b = bundle(manifest)
    target = tmp_path / "target"
    if position == "directory":
        target.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with pytest.raises(ValueError, match="directory_symlink_or_special"):
            evidence.commit_landmark(alias / "new", b, manifest)
        return
    evidence.commit_landmark(target, b, manifest)
    if position == "child":
        data = target / "capture.fp32"
        duplicate = tmp_path / "duplicate"
        duplicate.write_bytes(data.read_bytes())
        data.unlink()
        data.symlink_to(duplicate)
    else:
        (target / "unlisted").write_bytes(b"not in manifest")
    with pytest.raises(ValueError):
        evidence.load_landmark(target, manifest)


def test_interrupted_publication_never_becomes_complete_or_reusable(tmp_path, monkeypatch):
    manifest = example_manifest()
    b = bundle(manifest)
    publish = evidence._publish

    def stop_before_receipt(path, raw):
        if path.name == "receipt.json":
            raise OSError("invented publication interruption")
        publish(path, raw)

    monkeypatch.setattr(evidence, "_publish", stop_before_receipt)
    target = tmp_path / "interrupted"
    with pytest.raises(OSError, match="invented publication interruption"):
        evidence.commit_landmark(target, b, manifest)
    assert (target / "capture.fp32").read_bytes() == bytes.fromhex("000000800000c03f")
    assert not (target / "receipt.json").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        evidence.load_landmark(target, manifest)
    with pytest.raises((ValueError, FileExistsError)):
        evidence.commit_landmark(target, b, manifest)


def test_manually_constructed_mutable_prediction_packet_is_rejected(tmp_path):
    manifest, bundles, rows = examples()
    packet = evidence.validate_prediction_pairs(manifest, bundles, rows)
    mutable = evidence.PredictionPacket(bytearray(packet.metadata_bytes))
    with pytest.raises(ValueError):
        evidence.commit_prediction_pairs(tmp_path / "mutable-packet", mutable, manifest, bundles)
    assert not (tmp_path / "mutable-packet").exists()


def test_one_bound_comparator_cannot_have_conflicting_fit_statuses():
    manifest, bundles, rows = examples()
    # The same fixed model/transform is declared fitted on every other row.
    slot = rows[-1]["predictions"]["text"]
    slot.update(fit_status="failed", probability=None, null_reason="fit_failed")
    with pytest.raises(ValueError):
        evidence.validate_prediction_pairs(manifest, bundles, rows)


def test_known_observed_prefix_negative_is_retained_separately_from_censoring():
    manifest = example_manifest()
    manifest["endpoint"]["horizon_tokens"] = 8
    observed = bundle(manifest, label=0)
    outcome = observed.metadata["outcome"]
    assert outcome["label"] == 0 and outcome["status"] == "known"
    assert outcome["capped"] is True and outcome["censored"] is True
    assert outcome["terminal_eos"] is False
    # This is a submitted instrument label, not validator-certified label truth.
    assert set(evidence.predictor_view(observed)["features"]) == {
        "rendered_prompt_utf8",
        "prompt_token_ids",
        "prefix_utf8",
        "prefix_token_ids",
    }
