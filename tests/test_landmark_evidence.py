"""Invented evidence only: no model, tokenizer, corpus, or predictor fitting."""

import copy
import os
import struct
from dataclasses import FrozenInstanceError

import pytest

from lexical_prompt_study import landmark_evidence as a


def digest(label):
    return a.sha256(label.encode())


def manifest():
    comp = {
        "model_sha256": digest("predictor"),
        "transform_sha256": digest("transform"),
        "train_manifest_sha256": digest("train"),
        "calibration_manifest_sha256": digest("cal"),
        "fit_group_ids": [],
        "calibration_group_ids": [],
        "fit_budget": 0,
        "calibration_budget": 0,
    }
    return {
        "schema_version": a.SCHEMA,
        "study_id": "invented",
        "provenance": {
            key: digest(key)
            for key in (
                "model_sha256",
                "tokenizer_sha256",
                "chat_template_sha256",
                "capture_code_sha256",
            )
        },
        "vocab_size": 100,
        "eos_token_ids": [99],
        "generation": {"policy_sha256": digest("generation"), "seed": 0, "max_new_tokens": 10},
        "landmarks": [0, 8],
        "count_convention": a.COUNT_CONVENTION,
        "decoder": dict(a.DECODER),
        "capture": {
            "layer_index": 1,
            "model_layers": 3,
            "hidden_width": 3,
            "site": "residual_post",
            "source_dtype": "float32",
            "storage_dtype": "float32",
            "byte_order": "little",
            "attention_policy": "all_ones",
            "position_policy": "absolute_zero_based",
            "cache_policy": "none",
        },
        "groups": {"g": "evaluation"},
        "observations": [
            {
                "observation_id": "o",
                "core_id": "core",
                "group_id": "g",
                "split": "evaluation",
                "strata": {"variant": "invented"},
            }
        ],
        "endpoint": {
            "instrument_sha256": digest("instrument"),
            "target_sha256": digest("target"),
            "horizon_tokens": 10,
            "uncertainty_policy_sha256": digest("uncertainty"),
        },
        "comparators": {name: copy.deepcopy(comp) for name in a.COMPARATORS},
    }


def inputs(m=None, *, t=8, ids=None, status="completed", stop="cap", capture="completed"):
    m = manifest() if m is None else m
    ids = list(range(10, 20)) if ids is None else ids
    obs = {
        "observation_id": "o",
        "messages": [
            {"role": "system", "content": " Invented\n"},
            {"role": "user", "content": "é\t e\u0301"},
        ],
        "rendered_prompt_utf8": "<s> Invented\n é\t e\u0301  ".encode(),
        "prompt_token_ids": [1, 2, 3],
    }
    generation = {
        "status": status,
        "stop_reason": stop,
        "token_ids": ids,
        "observed_tokens": len(ids),
        "attempt_sha256": digest("attempt"),
        "dispatch_sha256": digest("dispatch"),
    }
    if status == "missing":
        generation["attempt_sha256"] = generation["dispatch_sha256"] = None
    eos = bool(ids and ids[-1] in m["eos_token_ids"])
    content = ids[:-1] if eos else ids
    prefix_status = (
        "available" if len(content) >= t else "unreached" if status == "completed" else "missing"
    )
    pre = {
        "landmark": t,
        "status": prefix_status,
        "reason": None,
        "token_ids": content[:t] if prefix_status == "available" else None,
        "decoded_utf8": (b"" if t == 0 else "  é\n\ufffd<special>".encode())
        if prefix_status == "available"
        else None,
    }
    if prefix_status != "available":
        pre["reason"] = "early_eos" if prefix_status == "unreached" else "generation_" + status
        capture = "unavailable"
    raw = struct.pack("<fff", 1.25, -0.0, -2.5) if capture == "completed" else None
    cap = {
        "status": capture,
        "reason": {
            "completed": None,
            "unavailable": "prefix_unavailable",
            "missing": "not_attempted",
            "infrastructure_failed": "capture_failed",
        }[capture],
        "evidence": None,
    }
    if capture == "completed":
        inp = obs["prompt_token_ids"] + pre["token_ids"]
        cap["evidence"] = {
            "observation_id": "o",
            "landmark": t,
            "prompt_ids_sha256": a.object_hash(obs["prompt_token_ids"]),
            "prefix_ids_sha256": a.object_hash(pre["token_ids"]),
            "prompt_utf8_sha256": a.sha256(obs["rendered_prompt_utf8"]),
            "prefix_utf8_sha256": a.sha256(pre["decoded_utf8"]),
            "input_token_ids": inp,
            "input_ids_sha256": a.object_hash(inp),
            "input_length": len(inp),
            "attention_mask": [1] * len(inp),
            "position_ids": list(range(len(inp))),
            "absolute_position": len(inp) - 1,
            "policy": copy.deepcopy(m["capture"]),
            "provenance": copy.deepcopy(m["provenance"]),
            "shape": [1, 3],
            "byte_length": len(raw),
            "content_sha256": a.sha256(raw),
            "forward_receipt": {
                "attempt_sha256": digest("capture-attempt-" + str(t)),
                "dispatch_sha256": digest("capture-dispatch-" + str(t)),
                "source_sha256": m["provenance"]["capture_code_sha256"],
                "input_ids_sha256": a.object_hash(inp),
            },
        }
    censored = not eos and len(content) < m["endpoint"]["horizon_tokens"]
    outcome = {
        "status": "unknown" if censored else "known",
        "applicable": True,
        "label": None if censored else 0,
        "reason": "censored" if censored else None,
        "disagreement": False,
        "endpoint": copy.deepcopy(m["endpoint"]),
        "generation_sha256": a.object_hash(generation),
        "terminal_eos": eos,
        "capped": stop == "cap",
        "censored": censored,
    }
    return [m, obs, generation, pre, cap, raw, outcome]


def bundle(m=None, **kwargs):
    return a.validate_landmark(*inputs(m, **kwargs))


def rows_for(m, bundles):
    rows = []
    for b in bundles:
        meta = b.metadata
        row = {
            "observation_id": meta["observation"]["observation_id"],
            "landmark": meta["prefix"]["landmark"],
            "group_id": meta["planned"]["group_id"],
            "split": meta["planned"]["split"],
            "bundle_sha256": b.sha256,
            "outcome_sha256": a.object_hash(meta["outcome"]),
            "predictions": {},
        }
        for name in a.COMPARATORS:
            availability = (
                "prefix_unavailable"
                if meta["prefix"]["status"] != "available"
                else "capture_unavailable"
                if name == "text_internal" and meta["capture"]["status"] != "completed"
                else "eligible"
            )
            eligibility = "not_evaluation" if row["split"] != "evaluation" else availability
            row["predictions"][name] = {
                "comparator_sha256": a.object_hash(m["comparators"][name]),
                "features_sha256": a.predictor_view(b, name)["features_sha256"]
                if availability == "eligible"
                else None,
                "eligibility": eligibility,
                "probability": 0.25 if eligibility == "eligible" else None,
                "null_reason": None if eligibility == "eligible" else eligibility,
                "fit_status": "fitted",
            }
        rows.append(row)
    return rows


def test_exact_bytes_t0_t8_and_special_ids():
    m = manifest()
    for t, position in ((0, 2), (8, 10)):
        b = bundle(m, t=t)
        assert b.metadata["capture"]["evidence"]["absolute_position"] == position
        assert b.blobs["prompt.utf8"].endswith("e\u0301  ".encode())
        assert b.metadata["prefix"]["token_ids"] == list(range(10, 10 + t))
        assert struct.unpack("<fff", b.blobs["capture.fp32"]) == (1.25, -0.0, -2.5)
        if t:
            assert b.blobs["prefix.utf8"] == "  é\n\ufffd<special>".encode()
        assert set(a.predictor_view(b)["features"]) == {
            "rendered_prompt_utf8",
            "prompt_token_ids",
            "prefix_utf8",
            "prefix_token_ids",
        }


def test_mutable_sources_and_returned_metadata_are_defensive():
    args = inputs()
    args[1]["rendered_prompt_utf8"] = bytearray(args[1]["rendered_prompt_utf8"])
    args[3]["decoded_utf8"] = bytearray(args[3]["decoded_utf8"])
    args[5] = bytearray(args[5])
    b = a.validate_landmark(*args)
    original = b.metadata_bytes, b.blob_items
    args[1]["prompt_token_ids"][0] = 50
    args[1]["rendered_prompt_utf8"][:] = b"changed"
    args[3]["decoded_utf8"][:] = b"changed"
    args[5][:] = b"changed"
    b.metadata["generation"]["token_ids"].append(99)
    b.blobs.clear()
    assert (b.metadata_bytes, b.blob_items) == original
    with pytest.raises(FrozenInstanceError):
        b.metadata_bytes = b"changed"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(vocab_size=True),
        lambda m: m.update(eos_token_ids=[True]),
        lambda m: m.update(eos_token_ids=[99, 99]),
        lambda m: m.update(landmarks=[False, 8]),
        lambda m: m.update(landmarks=[8, 0]),
        lambda m: m.update(landmarks=[0, 11]),
        lambda m: m["generation"].update(seed=True),
        lambda m: m["generation"].update(max_new_tokens=False),
        lambda m: m["decoder"].update(skip_special_tokens=0),
        lambda m: m["capture"].update(layer_index=3),
        lambda m: m["capture"].update(source_dtype="bfloat16"),
        lambda m: m["capture"].update(byte_order="big"),
        lambda m: m["provenance"].update(model_sha256="A" * 64),
        lambda m: m["comparators"]["text"].update(fit_group_ids=["g"]),
        lambda m: m["observations"].append(copy.deepcopy(m["observations"][0])),
    ],
)
def test_manifest_guards(mutate):
    m = manifest()
    mutate(m)
    with pytest.raises(ValueError):
        a.validate_manifest(m)


def test_shared_core_cannot_cross_groups_or_splits():
    m = manifest()
    m["groups"]["new"] = "fit"
    m["observations"].append(
        {**m["observations"][0], "observation_id": "other", "group_id": "new", "split": "fit"}
    )
    for comp in m["comparators"].values():
        comp["fit_group_ids"] = ["new"]
    with pytest.raises(ValueError, match="core_group"):
        a.validate_manifest(m)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e["input_token_ids"].append(42),
        lambda e: e["input_token_ids"].__setitem__(0, 42),
        lambda e: e.update(observation_id="other"),
        lambda e: e.update(landmark=True),
        lambda e: e.update(input_length=True),
        lambda e: e.update(absolute_position=9),
        lambda e: e["attention_mask"].__setitem__(0, True),
        lambda e: e["position_ids"].__setitem__(0, False),
        lambda e: e.update(shape=[True, 3]),
        lambda e: e.update(shape=[1, 2]),
        lambda e: e.update(byte_length=8),
        lambda e: e.update(prefix_utf8_sha256=digest("wrong")),
        lambda e: e["policy"].update(cache_policy="past"),
        lambda e: e["provenance"].update(model_sha256=digest("wrong")),
        lambda e: e["forward_receipt"].update(source_sha256=digest("wrong")),
    ],
)
def test_capture_boundary_and_provenance_guards(mutate):
    args = inputs()
    mutate(args[4]["evidence"])
    with pytest.raises(ValueError):
        a.validate_landmark(*args)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_vector_rejected_even_with_matching_hash(value):
    args = inputs()
    args[5] = struct.pack("<fff", value, 0, 1)
    args[4]["evidence"]["content_sha256"] = a.sha256(args[5])
    with pytest.raises(ValueError, match="capture_nonfinite"):
        a.validate_landmark(*args)


def test_reject_prefix_slice_and_bytes_hash_mismatch():
    for field in ("token_ids", "decoded_utf8"):
        args = inputs()
        args[3][field] = [3] * 8 if field == "token_ids" else b"other"
        with pytest.raises(ValueError):
            a.validate_landmark(*args)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"ids": [10, 99], "stop": "eos"}, "unreached"),
        ({"ids": list(range(10, 18)) + [99], "stop": "eos"}, "available"),
        ({}, "available"),
        ({"ids": [10], "status": "interrupted", "stop": "interrupted"}, "missing"),
        ({"ids": list(range(10, 18)), "status": "interrupted", "stop": "interrupted"}, "available"),
        ({"ids": [], "status": "missing", "stop": "missing"}, "missing"),
        (
            {"ids": [], "status": "infrastructure_failed", "stop": "infrastructure_failed"},
            "missing",
        ),
    ],
)
def test_availability_is_separate_from_final_outcome(kwargs, expected):
    b = bundle(**kwargs)
    assert b.metadata["prefix"]["status"] == expected
    if expected != "available":
        assert b.metadata["capture"]["status"] == "unavailable"
        assert set(b.blobs) == {"prompt.utf8"}
        with pytest.raises(ValueError, match="prefix_unavailable"):
            a.predictor_view(b)


def test_t0_available_without_any_generation():
    b = bundle(t=0, ids=[], status="missing", stop="missing", capture="missing")
    assert b.metadata["prefix"]["status"] == "available"
    assert a.predictor_view(b)["features"]["prefix_token_ids"] == ()
    with pytest.raises(ValueError, match="capture_unavailable"):
        a.predictor_view(b, "text_internal")


def test_capture_failure_preserves_text_and_unknown_outcome():
    b = bundle(
        capture="infrastructure_failed",
        ids=list(range(10, 18)),
        status="interrupted",
        stop="interrupted",
    )
    assert b.metadata["outcome"]["status"] == "unknown"
    assert a.predictor_view(b)["features"]["prefix_token_ids"] == tuple(range(10, 18))
    with pytest.raises(ValueError, match="capture_unavailable"):
        a.predictor_view(b, "text_internal")


def test_generation_guards_and_censoring_does_not_adjudicate_labels():
    for update in (
        {"token_ids": [10, 99, 11], "observed_tokens": 3},
        {"observed_tokens": True},
        {"dispatch_sha256": None},
        {"stop_reason": "eos"},
        {"token_ids": [100] * 10},
    ):
        args = inputs()
        args[2].update(update)
        args[6]["generation_sha256"] = a.object_hash(args[2])
        with pytest.raises(ValueError):
            a.validate_landmark(*args)
    args = inputs(ids=list(range(10, 18)), status="interrupted", stop="interrupted")
    args[6].update(status="known", label=0, reason=None)
    retained = a.validate_landmark(*args)
    assert retained.metadata["outcome"]["label"] == 0
    assert retained.metadata["outcome"]["censored"] is True


def test_future_and_label_changes_never_change_features():
    first = bundle()
    args = inputs(ids=list(range(10, 18)) + [99], stop="eos")
    args[6].update(label=1, disagreement=True)
    second = a.validate_landmark(*args)
    assert first.sha256 != second.sha256
    for name in a.COMPARATORS:
        assert a.predictor_view(first, name) == a.predictor_view(second, name)
    changed = inputs()
    changed[1]["rendered_prompt_utf8"] += b" "
    changed[4]["evidence"]["prompt_utf8_sha256"] = a.sha256(changed[1]["rendered_prompt_utf8"])
    assert a.predictor_view(a.validate_landmark(*changed)) != a.predictor_view(first)


def test_complete_cohort_prediction_retention_and_roundtrip(tmp_path):
    m = manifest()
    bundles = [bundle(m, t=t) for t in m["landmarks"]]
    rows = rows_for(m, bundles)
    packet = a.validate_prediction_pairs(m, bundles, rows)
    assert packet.metadata["planned_comparator_slots"] == 4
    assert packet.metadata["rows"][1]["predictions"]["text"]["probability"] == 0.25
    root = tmp_path / "predictions"
    a.commit_prediction_pairs(root, packet, m, bundles)
    assert a.load_prediction_pairs(root, m, bundles) == packet
    for index, b in enumerate(bundles):
        path = tmp_path / str(index)
        a.commit_landmark(path, b, m)
        assert a.load_landmark(path, m) == b
        assert path.stat().st_mode & 0o777 == 0o700
        assert all(child.stat().st_mode & 0o777 == 0o600 for child in path.iterdir())
        with pytest.raises(FileExistsError):
            a.commit_landmark(path, b, m)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -0.01, 1.01, "0.5"])
def test_bad_probabilities(value):
    m = manifest()
    bundles = [bundle(m, t=t) for t in m["landmarks"]]
    rows = rows_for(m, bundles)
    rows[0]["predictions"]["text"]["probability"] = value
    with pytest.raises(ValueError):
        a.validate_prediction_pairs(m, bundles, rows)


def test_null_partner_and_failed_fit_remain_in_cohort():
    m = manifest()
    bundles = [bundle(m, t=0), bundle(m, capture="missing")]
    rows = rows_for(m, bundles)
    slot = rows[0]["predictions"]["text"]
    slot.update(probability=None, null_reason="prediction_failed")
    packet = a.validate_prediction_pairs(m, bundles, rows)
    retained = packet.metadata["rows"]
    assert retained[1]["predictions"]["text"]["probability"] == 0.25
    assert retained[1]["predictions"]["text_internal"]["null_reason"] == "capture_unavailable"
    assert retained[0]["predictions"]["text"]["null_reason"] == "prediction_failed"


def test_one_global_fit_status_and_all_failed_slots_retained():
    m = manifest()
    bundles = [bundle(m, t=t) for t in m["landmarks"]]
    rows = rows_for(m, bundles)
    rows[0]["predictions"]["text"].update(
        probability=None, fit_status="failed", null_reason="fit_failed"
    )
    with pytest.raises(ValueError, match="inconsistent_comparator_fit_status"):
        a.validate_prediction_pairs(m, bundles, rows)
    rows[1]["predictions"]["text"].update(
        probability=None, fit_status="failed", null_reason="fit_failed"
    )
    retained = a.validate_prediction_pairs(m, bundles, rows).metadata["rows"]
    assert all(row["predictions"]["text"]["null_reason"] == "fit_failed" for row in retained)


def test_unattempted_slots_and_nonevaluation_slots_preserved():
    m = manifest()
    m["groups"]["g"] = m["observations"][0]["split"] = "fit"
    for comp in m["comparators"].values():
        comp["fit_group_ids"] = ["g"]
    bundles = [
        bundle(m, t=t, ids=[], status="missing", stop="missing", capture="missing")
        for t in m["landmarks"]
    ]
    rows = rows_for(m, bundles)
    packet = a.validate_prediction_pairs(m, bundles, rows)
    assert packet.metadata["planned_rows"] == 2
    assert all(
        s["null_reason"] == "not_evaluation"
        for r in packet.metadata["rows"]
        for s in r["predictions"].values()
    )


def test_prediction_order_is_explicit_and_key_join_lossless():
    m = manifest()
    bundles = [bundle(m, t=t) for t in m["landmarks"]]
    rows = rows_for(m, bundles)
    normal = a.validate_prediction_pairs(m, bundles, rows)
    with pytest.raises(ValueError, match="prediction_order"):
        a.validate_prediction_pairs(m, bundles, rows[::-1])
    assert a.validate_prediction_pairs(m, bundles, rows[::-1], keyed=True) == normal
    for invalid in (rows[:1], [rows[0], rows[0]]):
        with pytest.raises(ValueError):
            a.validate_prediction_pairs(m, bundles, invalid)
    with pytest.raises(ValueError, match="bundle_order"):
        a.validate_prediction_pairs(m, bundles[::-1], rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(group_id="other"),
        lambda r: r.update(split="fit"),
        lambda r: r.update(outcome_sha256=digest("other")),
        lambda r: r.update(bundle_sha256=digest("other")),
        lambda r: r["predictions"].pop("text_internal"),
        lambda r: r["predictions"]["text"].update(features_sha256=digest("other")),
        lambda r: r["predictions"]["text"].update(comparator_sha256=digest("other")),
        lambda r: r["predictions"]["text"].update(eligibility="capture_unavailable"),
        lambda r: r["predictions"]["text"].update(fit_status="failed"),
    ],
)
def test_prediction_bindings(mutation):
    m = manifest()
    bundles = [bundle(m, t=t) for t in m["landmarks"]]
    rows = rows_for(m, bundles)
    mutation(rows[1])
    with pytest.raises(ValueError):
        a.validate_prediction_pairs(m, bundles, rows)


@pytest.mark.parametrize("fail_at", range(1, 7))
def test_publication_interruption_consumes_directory_without_recovery(
    tmp_path, monkeypatch, fail_at
):
    m, b = manifest(), bundle()
    original, calls = a._publish, 0

    def interrupted(path, raw):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise KeyboardInterrupt()
        original(path, raw)

    path = tmp_path / "orphan"
    monkeypatch.setattr(a, "_publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        a.commit_landmark(path, b, m)
    assert path.is_dir()
    with pytest.raises((ValueError, FileNotFoundError)):
        a.load_landmark(path, m)
    monkeypatch.setattr(a, "_publish", original)
    with pytest.raises(FileExistsError):
        a.commit_landmark(path, b, m)


@pytest.mark.parametrize("filename", ["capture.fp32", "receipt.json"])
def test_interruption_after_temporary_fsync_preserves_uncommitted_orphan(
    tmp_path, monkeypatch, filename
):
    m, b = manifest(), bundle()
    original_link = a.os.link

    def interrupted_link(source, destination, **kwargs):
        if destination.name == filename:
            raise KeyboardInterrupt()
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(a.os, "link", interrupted_link)
    path = tmp_path / "interrupted-link"
    with pytest.raises(KeyboardInterrupt):
        a.commit_landmark(path, b, m)
    assert (path / ("." + filename + ".tmp")).is_file()
    assert not (path / filename).exists()
    with pytest.raises(FileNotFoundError):
        a.load_landmark(path, m)
    with pytest.raises(FileExistsError):
        a.commit_landmark(path, b, m)


def test_symlink_unknown_inventory_and_corrupt_children(tmp_path):
    m, b = manifest(), bundle()
    parent = tmp_path / "actual"
    parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError, match="directory_symlink"):
        a.commit_landmark(link / "child", b, m)
    for mode in ("symlink", "fifo", "extra", "corrupt"):
        path = tmp_path / mode
        a.commit_landmark(path, b, m)
        target = path / "capture.fp32"
        if mode == "extra":
            (path / "extra").write_bytes(b"extra")
        elif mode == "corrupt":
            target.write_bytes(b"bad")
        else:
            target.unlink()
            if mode == "symlink":
                target.symlink_to(tmp_path / "missing")
            else:
                os.mkfifo(target)
        with pytest.raises(ValueError):
            a.load_landmark(path, m)


def test_forged_dataclass_and_prediction_packet_do_not_bypass_writers(tmp_path):
    m, b = manifest(), bundle()
    meta = b.metadata
    meta["prefix"]["token_ids"][0] = 50
    forged = a.ValidatedBundle(a.canonical(meta), b.blob_items)
    with pytest.raises(ValueError):
        a.commit_landmark(tmp_path / "forged", forged, m)
    assert not (tmp_path / "forged").exists()
    bundles = [bundle(m, t=t) for t in m["landmarks"]]
    packet = a.validate_prediction_pairs(m, bundles, rows_for(m, bundles))
    meta = packet.metadata
    meta["planned_rows"] = 99
    with pytest.raises(ValueError):
        a.commit_prediction_pairs(
            tmp_path / "forged-pred", a.PredictionPacket(a.canonical(meta)), m, bundles
        )
    assert not (tmp_path / "forged-pred").exists()
