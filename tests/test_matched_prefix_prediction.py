"""Invented-data qualification of the fixed A186 fit and missingness contract."""

from __future__ import annotations

import copy
import hashlib
import itertools
import math

import numpy as np
import pytest

from lexical_prompt_study import landmark_evidence as e
from lexical_prompt_study import matched_prefix_prediction as p
from lexical_prompt_study import matched_prefix_tasks as tasks


def fixture():
    roster = tasks.prediction_roster(tasks.build_roster())
    packets, labels = [], {}
    for row in roster:
        core = row["core_index"]
        semantic = np.zeros(768, dtype=np.float32)
        semantic[core % 7] = 0.5
        semantic[7] = (core % 2) * 0.25
        internal = np.zeros(4096, dtype=np.float32)
        internal[core % 19] = 1
        internal[20] = 0.5 if core % 2 else -0.5
        packets.append(
            {
                "schema_version": "a186-prediction-features-v1",
                "observation_id": row["observation_id"],
                "common": {
                    "rendered_prompt_utf8": "Invented prompt",
                    "prompt_token_ids": [1, 2],
                    "prefix_utf8": "",
                    "prefix_token_ids": list(range(3, 11)),
                    "completed_fields": 0,
                    "prefix_known_error": False,
                },
                "text_semantic": semantic.tolist(),
                "internal": internal.tolist(),
            }
        )
        labels[row["observation_id"]] = core % 2
    return roster, packets, labels


def bind(packets, labels, reached=None):
    result = {k: "1" * 64 for k in p.BINDING_KEYS}
    result.update(
        feature_packets_sha256=e.object_hash(packets),
        labels_sha256=e.object_hash(labels),
        landmark_reached_sha256=e.object_hash(
            reached
            if reached is not None
            else {
                item["observation_id"]: True if item["common"] is not None else None
                for item in packets
            }
        ),
        fit_runtime_sha256=e.object_hash(p.runtime_descriptor()),
        prediction_source_sha256=p.source_sha256(),
    )
    return result


@pytest.fixture(scope="module")
def fitted():
    roster, packets, labels = fixture()
    bindings = bind(packets, labels)
    result = p.fit_predict(roster, packets, labels, bindings=bindings)
    return roster, packets, labels, bindings, result


def test_exact_hash_manual_overlapping_unicode_and_repetition():
    raw = "AaéAaéAa".encode()
    expected = [0] * 256
    for length in (3, 4, 5):
        for gram in [raw[i : i + length] for i in range(len(raw) + 1 - length)]:
            digest = hashlib.sha256(length.to_bytes(1, "big") + gram).digest()
            expected[int(digest.hex()[:2], 16)] += (-1) ** (digest[1] % 2)
    norm = math.sqrt(sum(n * n for n in expected))
    np.testing.assert_array_equal(p.prefix_hash(raw), np.asarray(expected) / norm)
    assert not np.array_equal(p.prefix_hash(raw), p.prefix_hash(raw.lower()))


@pytest.mark.parametrize("raw", [b"", b"a", b"ab"])
def test_short_prefix_zero(raw):
    np.testing.assert_array_equal(p.prefix_hash(raw), np.zeros(256))


@pytest.mark.parametrize("p_known,q_known,y_known", itertools.product((False, True), repeat=3))
def test_all_missingness_patterns_sharp_extrema(p_known, q_known, y_known):
    left, right, label = (
        (0.3 if p_known else None),
        (0.8 if q_known else None),
        (1 if y_known else None),
    )
    result = p.brier_bounds(left, right, label)
    values = [
        (a - y) ** 2 - (b - y) ** 2
        for a in ([left] if p_known else np.linspace(0, 1, 11))
        for b in ([right] if q_known else np.linspace(0, 1, 11))
        for y in ([label] if y_known else [0, 1])
    ]
    assert result["lower"] == pytest.approx(min(values))
    assert result["upper"] == pytest.approx(max(values))
    assert result["resolved"] is (p_known and q_known and y_known)
    assert (result["point"] is None) is not result["resolved"]


def test_collapsed_bounds_still_null_when_label_unknown():
    assert p.brier_bounds(0.7, 0.7, None) == {
        "point": None,
        "lower": 0.0,
        "upper": 0.0,
        "resolved": False,
    }


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability(value):
    with pytest.raises(ValueError):
        p.brier_bounds(value, 0.5, 1)


@pytest.mark.parametrize(
    "field", ["outcome", "generation", "stop_reason", "future_token_ids", "total_length"]
)
def test_future_keys_rejected(field):
    roster, packets, _ = fixture()
    packets[0]["common"][field] = None
    with pytest.raises(ValueError, match="common_keys"):
        p.validate_feature_packet(packets[0], roster[0])


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_id",
        "seven_tokens",
        "bool_count",
        "false_flag",
        "lossy_float",
        "nan",
        "width",
        "surrogate",
    ],
)
def test_feature_boundary_and_lossless_types(mutation):
    roster, packets, _ = fixture()
    packet = packets[0]
    if mutation == "bool_id":
        packet["common"]["prefix_token_ids"][0] = True
    if mutation == "seven_tokens":
        packet["common"]["prefix_token_ids"].pop()
    if mutation == "bool_count":
        packet["common"]["completed_fields"] = False
    if mutation == "false_flag":
        packet["common"]["prefix_known_error"] = True
    if mutation == "lossy_float":
        packet["internal"][0] = 0.1
    if mutation == "nan":
        packet["text_semantic"][0] = float("nan")
    if mutation == "width":
        packet["internal"].pop()
    if mutation == "surrogate":
        packet["common"]["prefix_utf8"] = "\ud800"
    with pytest.raises((ValueError, UnicodeError)):
        p.validate_feature_packet(packet, roster[0])


@pytest.mark.parametrize("mutation", ["duplicate", "split", "order", "drop", "unknown_format"])
def test_roster_exactness(mutation):
    roster, _, _ = fixture()
    if mutation == "duplicate":
        roster[1] = roster[0]
    if mutation == "split":
        roster[0]["split"] = "fit" if roster[0]["split"] == "evaluation" else "evaluation"
    if mutation == "order":
        roster.reverse()
    if mutation == "drop":
        roster.pop()
    if mutation == "unknown_format":
        roster[0]["format"] = "other"
    with pytest.raises(ValueError, match="roster_exact"):
        p.validate_roster(roster)


def test_pca_rank_zero_keeps16_columns_and_signs():
    vector = [0.0] * 4096
    transform = p.fit_pca([vector] * 17)
    assert transform["numerical_rank"] == 0
    assert len(transform["components"]) == 16
    axes = np.array(transform["components"])
    np.testing.assert_array_equal(axes @ axes.T, np.eye(16))
    assert all(axis[np.argmax(np.abs(axis))] > 0 for axis in axes)
    np.testing.assert_array_equal(p.transform_internal(vector, transform), np.zeros(16))


def test_actual_two_fits_shared_training_and_fixed_cohorts(fitted):
    roster, packets, labels, bindings, result = fitted
    assert result["actual_fits"] == 2 and result["terminal_error"] is None
    assert len(result["predictions"]) == 112
    assert result["fit_support"]["eligible_rows"] == 64
    ids = result["models"]["text"]["fit_observation_ids"]
    assert ids == result["models"]["text_internal"]["fit_observation_ids"]
    assert len(ids) == 64
    assert all(next(r for r in roster if r["observation_id"] == i)["core_index"] < 32 for i in ids)
    assert all(m["options"] == p.OPTIONS for m in result["models"].values())
    for row in result["predictions"]:
        assert (
            all(s["status"] == "not_evaluation" for s in row["comparators"].values())
            if row["split"] == "fit"
            else all(s["status"] == "predicted" for s in row["comparators"].values())
        )
    assert result["analysis"]["primary_csv"]["planned_rows"] == 16
    assert result["analysis"]["secondary_known_formats"]["planned_rows"] == 32
    assert p.validate_result(result, roster, packets, labels, bindings=bindings) == result


def test_retained_logits_replay_matches_actual_sklearn(fitted):
    from sklearn.linear_model import LogisticRegression

    roster, packets, _, _, result = fitted
    row_index = next(i for i, r in enumerate(roster) if r["split"] == "evaluation")
    for name in p.COMPARATORS:
        artifact = result["models"][name]
        model = LogisticRegression(**p.OPTIONS)
        model.classes_ = np.array(artifact["classes"])
        model.coef_ = np.array(artifact["coef"])
        model.intercept_ = np.array(artifact["intercept"])
        model.n_features_in_ = model.coef_.shape[1]
        x = p.text_features(packets[row_index])
        if name == "text_internal":
            x = np.r_[x, p.transform_internal(packets[row_index]["internal"], result["transform"])]
        assert p.replay_probability(x, artifact) == pytest.approx(
            model.predict_proba(x[None])[0, 1], abs=1e-15
        )


def test_missing_internal_does_not_remove_text_and_zero_gate_preserves_null(fitted):
    roster, packets, labels, _, result = copy.deepcopy(fitted)
    i = next(i for i, r in enumerate(roster) if r["split"] == "evaluation" and r["format"] == "csv")
    packets[i]["internal"] = None
    packets[i]["common"].update(prefix_utf8="invalid,", completed_fields=1, prefix_known_error=True)
    rows = p._predict_rows(roster, packets, result["models"], result["transform"])
    slots = rows[i]["comparators"]
    assert slots["text"]["probability"] == 0 and slots["text"]["override_applied"] is True
    assert (
        slots["text_internal"]["probability"] is None
        and slots["text_internal"]["override_applied"] is False
    )
    assert slots["text"]["learned_probability"] is not None
    analysis = p.analyze_predictions(roster, rows, labels)
    assert analysis["primary_csv"]["point"] is None
    assert analysis["primary_csv"]["resolved_rows"] == 15
    assert analysis["secondary_known_formats"]["point"] is not None


def test_primary_independent_of_missing_secondary_label(fitted):
    roster, _, labels, _, result = copy.deepcopy(fitted)
    row = next(r for r in roster if r["split"] == "evaluation" and r["format"] == "plain")
    labels[row["observation_id"]] = None
    analysis = p.analyze_predictions(roster, result["predictions"], labels)
    assert analysis["primary_csv"] == result["analysis"]["primary_csv"]
    assert analysis["secondary_known_formats"]["point"] is None


def test_secondary_exact_core_weight_and_analytic_bounds(fitted):
    roster, _, labels, _, result = copy.deepcopy(fitted)
    rows = result["predictions"]
    evaluation = [r for r in rows if r["split"] == "evaluation" and r["format"] != "csv"]
    totals = []
    for core in range(32, 48):
        per_core = []
        for row in [r for r in evaluation if r["core_index"] == core]:
            slots = row["comparators"]
            y = labels[row["observation_id"]]
            per_core.append(
                (slots["text"]["probability"] - y) ** 2
                - (slots["text_internal"]["probability"] - y) ** 2
            )
        totals.append(sum(per_core) / 2)
    assert result["analysis"]["secondary_known_formats"]["point"] == pytest.approx(sum(totals) / 16)


def test_class_core_guard_not_satisfied_by_repeated_rows(monkeypatch):
    roster, packets, labels = fixture()
    for row in roster:
        labels[row["observation_id"]] = int(row["core_index"] < 3)
    monkeypatch.setattr(p, "_estimator", lambda: pytest.fail("must not fit"))
    result = p.fit_predict(roster, packets, labels, bindings=bind(packets, labels))
    assert result["actual_fits"] == 0
    assert result["fit_support"]["class_rows"]["1"] == 6
    assert result["fit_support"]["class_core_counts"]["1"] == 3
    assert result["analysis"]["primary_csv"]["lower"] == -1
    assert result["analysis"]["primary_csv"]["upper"] == 1


def test_less_than17_shared_fit_rows_no_fit(monkeypatch):
    roster, packets, labels = fixture()
    kept = 0
    for row, packet in zip(roster, packets, strict=True):
        if row["split"] == "fit":
            kept += 1
            if kept > 16:
                packet["internal"] = None
    monkeypatch.setattr(p, "_estimator", lambda: pytest.fail("must not fit"))
    result = p.fit_predict(roster, packets, labels, bindings=bind(packets, labels))
    assert result["fit_support"]["eligible_rows"] == 16
    assert result["actual_fits"] == 0


def test_infrastructure_fit_failure_stops_second_attempt(monkeypatch):
    roster, packets, labels = fixture()
    calls = []

    class Broken:
        def fit(self, *_):
            calls.append(1)
            raise OSError("invented")

    monkeypatch.setattr(p, "_estimator", Broken)
    result = p.fit_predict(roster, packets, labels, bindings=bind(packets, labels))
    assert calls == [1] and result["actual_fits"] == 1
    assert result["models"]["text"]["status"] == "failed"
    assert result["models"]["text_internal"]["status"] == "unattempted"


def test_nonconvergence_unavailable_and_other_fixed_attempt_continues(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    roster, packets, labels = fixture()
    calls = []

    class Nonconverged:
        n_iter_ = np.array([1000])

        def fit(self, *_):
            calls.append(1)
            warnings.warn("invented", ConvergenceWarning)

    monkeypatch.setattr(p, "_estimator", Nonconverged)
    result = p.fit_predict(roster, packets, labels, bindings=bind(packets, labels))
    assert len(calls) == 2 and result["actual_fits"] == 2
    assert all(m["status"] == "nonconverged" for m in result["models"].values())


@pytest.mark.parametrize("field", p.BINDING_KEYS)
def test_bindings_enforced_or_declared_external(field):
    roster, packets, labels = fixture()
    bindings = bind(packets, labels)
    bindings[field] = "bad"
    with pytest.raises(ValueError):
        p.validate_inputs(roster, packets, labels, bindings)


def test_output_roundtrip_no_overwrite_symlink_or_partial(fitted, tmp_path):
    roster, packets, labels, bindings, result = fitted
    path = tmp_path / "result.json"
    digest = p.save_result(path, result, roster, packets, labels, bindings=bindings)
    assert (
        p.load_result(path, roster, packets, labels, bindings=bindings, expected_sha256=digest)
        == result
    )
    with pytest.raises(FileExistsError):
        p.save_result(path, result, roster, packets, labels, bindings=bindings)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        p.load_result(link, roster, packets, labels, bindings=bindings, expected_sha256=digest)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="file_hash"):
        p.load_result(path, roster, packets, labels, bindings=bindings, expected_sha256=digest)


@pytest.mark.parametrize(
    "mutation", ["point", "probability", "fit_count", "training_id", "pca_center", "bool_count"]
)
def test_retained_result_tampering(fitted, mutation):
    roster, packets, labels, bindings, result = copy.deepcopy(fitted)
    if mutation == "point":
        result["analysis"]["primary_csv"]["point"] += 0.1
    if mutation == "probability":
        next(r for r in result["predictions"] if r["split"] == "evaluation")["comparators"]["text"][
            "probability"
        ] = 0.123
    if mutation == "fit_count":
        result["actual_fits"] = 1
    if mutation == "training_id":
        result["models"]["text"]["fit_observation_ids"][0] = "not-a-row"
    if mutation == "pca_center":
        result["transform"]["center"][0] += 0.1
    if mutation == "bool_count":
        result["actual_fits"] = True
    with pytest.raises(ValueError):
        p.validate_result(result, roster, packets, labels, bindings=bindings)


def test_missing_membership_descriptive_not_selected_subset(fitted):
    roster, packets, labels, _, result = copy.deepcopy(fitted)
    i = next(i for i, r in enumerate(roster) if r["split"] == "evaluation")
    packets[i].update(common=None, text_semantic=None, internal=None)
    rows = p._predict_rows(roster, packets, result["models"], result["transform"])
    descriptive = p.analyze_predictions(roster, rows, labels)["unflagged_evaluation_descriptive"]
    assert descriptive["membership_unknown"] is True
    assert descriptive["point"] is None and descriptive["lower"] is None


def test_label_and_feature_content_hashes_not_interchangeable():
    roster, packets, labels = fixture()
    bindings = bind(packets, labels)
    labels[roster[0]["observation_id"]] ^= 1
    with pytest.raises(ValueError, match="labels_binding"):
        p.validate_inputs(roster, packets, labels, bindings)


def test_equal_represented_core_diagnostic_weights():
    # One included format in one core and three in another: equal-core != equal-row.
    rows = []
    labels = {}
    for core, count, left, right in [(32, 1, 1.0, 0.0), (33, 3, 0.0, 1.0)]:
        for index in range(count):
            key = f"invented_{core}_{index}"
            rows.append(
                {
                    "observation_id": key,
                    "core_index": core,
                    "comparators": {
                        "text": {"probability": left},
                        "text_internal": {"probability": right},
                    },
                }
            )
            labels[key] = 0
    result = p._cohort(rows, labels, planned_count=48)
    assert result["point"] == 0
    assert result["included_cores"] == 2 and result["included_rows"] == 4
    labels[rows[0]["observation_id"]] = None
    missing = p._cohort(rows, labels, planned_count=48)
    assert missing["point"] is None
    assert missing["lower"] == -1 and missing["upper"] == 0


def test_fit_event_is_immutable_and_callback_failure_stops_before_fit(monkeypatch):
    roster, packets, labels = fixture()
    events = []
    sentinel = OSError("invented journal failure")

    def event(raw):
        assert type(raw) is bytes
        decoded = e._read_json(raw)
        events.append(decoded)
        if decoded["kind"] == "before_fit":
            raise sentinel

    class NeverFit:
        def fit(self, *_):
            pytest.fail("callback failure must prevent entry")

    monkeypatch.setattr(p, "_estimator", NeverFit)
    with pytest.raises(OSError) as caught:
        p.fit_predict(roster, packets, labels, bindings=bind(packets, labels), event=event)
    assert caught.value is sentinel
    assert [item["kind"] for item in events] == ["transform_return", "before_fit"]
    assert events[1]["slot"] == "text" and events[1]["attempt_index"] == 1
    assert events[0]["artifact_sha256"] == e.object_hash(events[0]["artifact"])


def test_keyboard_interrupt_propagates_with_failure_event_and_no_second_fit(monkeypatch):
    roster, packets, labels = fixture()
    events, entered = [], []
    sentinel = KeyboardInterrupt("invented interruption")

    class Interrupted:
        def fit(self, *_):
            entered.append(1)
            raise sentinel

    monkeypatch.setattr(p, "_estimator", Interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        p.fit_predict(
            roster,
            packets,
            labels,
            bindings=bind(packets, labels),
            event=lambda raw: events.append(e._read_json(raw)),
        )
    assert caught.value is sentinel and entered == [1]
    assert [event["kind"] for event in events] == ["transform_return", "before_fit", "fit_failure"]
    assert events[-1]["error_type"] == "KeyboardInterrupt"


def test_callback_failure_after_return_propagates_without_second_fit(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    roster, packets, labels = fixture()
    entered = []
    sentinel = OSError("invented returned-artifact journal failure")

    class Nonconverged:
        n_iter_ = np.array([1000])

        def fit(self, *_):
            entered.append(1)
            warnings.warn("invented", ConvergenceWarning)

    def callback(raw):
        if e._read_json(raw)["kind"] == "fit_return":
            raise sentinel

    monkeypatch.setattr(p, "_estimator", Nonconverged)
    with pytest.raises(OSError) as caught:
        p.fit_predict(roster, packets, labels, bindings=bind(packets, labels), event=callback)
    assert caught.value is sentinel and entered == [1]


@pytest.mark.parametrize(
    "mutation", ["unknown_status", "false_iterations", "failure_then_fit", "error_without_failure"]
)
def test_fit_lifecycle_tamper_rejected(fitted, mutation):
    roster, packets, labels, bindings, result = copy.deepcopy(fitted)
    if mutation == "unknown_status":
        result["models"]["text"]["status"] = "invented"
    if mutation == "false_iterations":
        result["models"]["text"]["n_iter"] = [True]
    if mutation == "failure_then_fit":
        result["models"]["text"] = {"status": "failed", "error_type": "OSError"}
        result["terminal_error"] = "OSError"
    if mutation == "error_without_failure":
        result["terminal_error"] = "OSError"
    with pytest.raises(ValueError):
        p.validate_result(result, roster, packets, labels, bindings=bindings)


def test_known_early_eos_excludes_diagnostic_without_unknown_membership(fitted):
    roster, packets, labels, _, result = copy.deepcopy(fitted)
    i = next(i for i, row in enumerate(roster) if row["split"] == "evaluation")
    packets[i].update(common=None, text_semantic=None, internal=None)
    reached = {row["observation_id"]: True for row in roster}
    reached[roster[i]["observation_id"]] = False
    labels[roster[i]["observation_id"]] = 0
    bindings = bind(packets, labels, reached)
    p.validate_inputs(roster, packets, labels, bindings, landmark_reached=reached)
    rows = p._predict_rows(
        roster, packets, result["models"], result["transform"], landmark_reached=reached
    )
    descriptive = p.analyze_predictions(roster, rows, labels)["unflagged_evaluation_descriptive"]
    assert descriptive["included_rows"] == 47 and descriptive["membership_unknown"] is False
    assert descriptive["point"] is not None
    support = p.class_support(roster, packets, labels, landmark_reached=reached)
    assert support["evaluation"]["known_unreached_rows"] == 1
    assert support["evaluation"]["unknown_reachability_rows"] == 0
    assert all(slot["probability"] is None for slot in rows[i]["comparators"].values())


@pytest.mark.parametrize("mutation", ["bool_as_int", "missing", "contradictory", "wrong_hash"])
def test_analysis_reachability_binding_and_types(mutation):
    roster, packets, labels = fixture()
    reached = {row["observation_id"]: True for row in roster}
    bindings = bind(packets, labels, reached)
    if mutation == "bool_as_int":
        reached[roster[0]["observation_id"]] = 1
    if mutation == "missing":
        reached.pop(roster[0]["observation_id"])
    if mutation == "contradictory":
        reached[roster[0]["observation_id"]] = False
    if mutation == "wrong_hash":
        bindings["landmark_reached_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        p.validate_inputs(roster, packets, labels, bindings, landmark_reached=reached)


def test_all_known_unreached_descriptive_is_undefined_not_zero(fitted):
    roster, packets, labels, _, result = copy.deepcopy(fitted)
    reached = {}
    for row, packet in zip(roster, packets, strict=True):
        reached[row["observation_id"]] = row["split"] == "fit"
        if row["split"] == "evaluation":
            packet.update(common=None, text_semantic=None, internal=None)
    rows = p._predict_rows(
        roster, packets, result["models"], result["transform"], landmark_reached=reached
    )
    descriptive = p.analyze_predictions(roster, rows, labels)["unflagged_evaluation_descriptive"]
    assert descriptive["included_rows"] == 0 and descriptive["membership_unknown"] is False
    assert (
        descriptive["point"] is None
        and descriptive["lower"] is None
        and descriptive["upper"] is None
    )
