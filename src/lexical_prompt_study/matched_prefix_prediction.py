"""A186 fixed prediction pipeline, separate from acquisition and its labels.

Only invented fixtures may exercise this module before the prospective freeze.
Inputs contain verified common-time views, not generation records. Validation
checks declared bindings and finite, lossless FP32 values; it does not prove
native rendering, tokenization, capture timing, or semantic encoder execution.
No tokenizer, encoder, torch, or historical-data loader is imported here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import warnings
from pathlib import Path

from . import landmark_evidence as evidence

SCHEMA = "a186-prediction-v1"
COMPARATORS = ("text", "text_internal")
BINDING_KEYS = (
    "plan_sha256 acquisition_terminal_sha256 encoder_terminal_sha256 "
    "feature_packets_sha256 labels_sha256 landmark_reached_sha256 fit_runtime_sha256 prediction_source_sha256"
).split()
OPTIONS = {
    "C": 1.0,
    "solver": "liblinear",
    "penalty": "l2",
    "tol": 1e-6,
    "max_iter": 1000,
    "random_state": 20260915,
    "class_weight": None,
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "dual": False,
    "warm_start": False,
}


def _require(ok, code):
    if not ok:
        raise ValueError(code)


def _keys(value, keys, code):
    _require(type(value) is dict and set(value) == set(keys.split()), code)


def _same(left, right):
    return evidence.canonical(left) == evidence.canonical(right)


def _snapshot(value):
    return json.loads(evidence.canonical(value))


def runtime_descriptor():
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
    }


def source_sha256():
    return evidence.sha256(Path(__file__).read_bytes())


def _tasks():
    from . import matched_prefix_tasks

    return matched_prefix_tasks


def validate_roster(roster):
    """Exact prospective schedule, including IDs, group splits and all112 slots."""
    _require(type(roster) is list, "roster_type")
    expected = _tasks().prediction_roster(_tasks().build_roster())
    _require(_same(roster, expected), "roster_exact")
    return _snapshot(roster)


def _vector(values, width):
    import numpy as np

    _require(type(values) is list and len(values) == width, "feature_width")
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in values), "feature_finite")
    vector = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        restored = vector.astype(np.float32).astype(np.float64)
    _require(bool(np.array_equal(vector, restored)), "feature_not_lossless_fp32")
    return vector


def validate_feature_packet(packet, row):
    """Reject additional keys, labels, stop information and future-token views."""
    _keys(
        packet, "schema_version observation_id common text_semantic internal", "feature_packet_keys"
    )
    _require(packet["schema_version"] == "a186-prediction-features-v1", "feature_schema")
    _require(packet["observation_id"] == row["observation_id"], "feature_identity")
    common = packet["common"]
    if common is None:
        _require(
            packet["text_semantic"] is None and packet["internal"] is None,
            "features_without_prefix",
        )
    else:
        _keys(
            common,
            "rendered_prompt_utf8 prompt_token_ids prefix_utf8 prefix_token_ids completed_fields prefix_known_error",
            "common_keys",
        )
        for key in ("rendered_prompt_utf8", "prefix_utf8"):
            _require(type(common[key]) is str, "utf8_string")
            common[key].encode("utf-8", errors="strict")
        _require(bool(common["rendered_prompt_utf8"]), "empty_prompt")
        for key in ("prompt_token_ids", "prefix_token_ids"):
            _require(type(common[key]) is list and bool(common[key]), "id_list")
            _require(all(type(x) is int and x >= 0 for x in common[key]), "token_id")
        _require(len(common["prefix_token_ids"]) == 8, "landmark_eight")
        _require(type(common["completed_fields"]) is int, "completed_fields_type")
        _require(type(common["prefix_known_error"]) is bool, "prefix_flag_type")
        check = _tasks().prefix_features(row["core_index"], common["prefix_utf8"])
        _require(
            common["completed_fields"] == check["completed_fields"]
            and common["prefix_known_error"] is check["prefix_known_error"],
            "prefix_check",
        )
        for key, width in (("text_semantic", 768), ("internal", 4096)):
            if packet[key] is not None:
                _vector(packet[key], width)
    return _snapshot(packet)


def _landmarks(roster, packets, landmark_reached):
    if landmark_reached is None:
        landmark_reached = {
            row["observation_id"]: True if packet["common"] is not None else None
            for row, packet in zip(roster, packets, strict=True)
        }
    _require(
        type(landmark_reached) is dict
        and set(landmark_reached) == {row["observation_id"] for row in roster},
        "landmark_coverage",
    )
    _require(
        all(value is None or type(value) is bool for value in landmark_reached.values()),
        "landmark_types",
    )
    for row, packet in zip(roster, packets, strict=True):
        if packet["common"] is not None:
            _require(landmark_reached[row["observation_id"]] is True, "landmark_common_consistency")
    return _snapshot(landmark_reached)


def validate_inputs(roster, feature_packets, labels, bindings, *, landmark_reached=None):
    roster = validate_roster(roster)
    _require(
        type(feature_packets) is list and len(feature_packets) == len(roster), "feature_coverage"
    )
    _require(
        type(labels) is dict and set(labels) == {r["observation_id"] for r in roster},
        "label_coverage",
    )
    _require(
        all(y is None or (type(y) is int and y in (0, 1)) for y in labels.values()), "binary_labels"
    )
    _require(type(bindings) is dict and set(bindings) == set(BINDING_KEYS), "binding_keys")
    for value in bindings.values():
        evidence._hash(value)
    _require(
        evidence.object_hash(feature_packets) == bindings["feature_packets_sha256"],
        "feature_binding",
    )
    _require(evidence.object_hash(labels) == bindings["labels_sha256"], "labels_binding")
    reached = _landmarks(roster, feature_packets, landmark_reached)
    _require(
        evidence.object_hash(reached) == bindings["landmark_reached_sha256"], "landmark_binding"
    )
    _require(source_sha256() == bindings["prediction_source_sha256"], "source_binding")
    _require(
        evidence.object_hash(runtime_descriptor()) == bindings["fit_runtime_sha256"],
        "runtime_binding",
    )
    packets = [validate_feature_packet(p, r) for p, r in zip(feature_packets, roster, strict=True)]
    return roster, packets, _snapshot(labels), _snapshot(bindings)


def prefix_hash(prefix_utf8):
    """Exact overlapping UTF-8 byte 3/4/5grams; no case or Unicode normalization."""
    import numpy as np

    _require(type(prefix_utf8) is bytes, "prefix_bytes")
    prefix_utf8.decode("utf-8", errors="strict")
    counts = np.zeros(256, dtype=np.float64)
    for n in (3, 4, 5):
        for start in range(len(prefix_utf8) - n + 1):
            digest = hashlib.sha256(bytes([n]) + prefix_utf8[start : start + n]).digest()
            counts[digest[0]] += 1 if digest[1] & 1 == 0 else -1
    norm = float(np.linalg.norm(counts))
    return counts if norm == 0 else counts / norm


def text_features(packet):
    import numpy as np

    if packet["common"] is None or packet["text_semantic"] is None:
        return None
    common = packet["common"]
    return np.concatenate(
        (
            _vector(packet["text_semantic"], 768),
            prefix_hash(common["prefix_utf8"].encode("utf-8")),
            np.array(
                [common["completed_fields"] / 12, int(common["prefix_known_error"])],
                dtype=np.float64,
            ),
        )
    )


def normalize_internal(values):
    import numpy as np

    vector = _vector(values, 4096)
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def fit_pca(vectors):
    """One float64 centered full SVD; sixteen retained axes including null axes."""
    import numpy as np

    _require(type(vectors) is list and len(vectors) >= 17, "pca_rows")
    matrix = np.stack([normalize_internal(v) for v in vectors])
    center = matrix.mean(axis=0, dtype=np.float64)
    _, singular, components = np.linalg.svd(matrix - center, full_matrices=False)
    axes = components[:16].copy()
    for axis in axes:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis *= -1
    tolerance = float(np.finfo(np.float64).eps * max(matrix.shape) * singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    _require(
        bool(np.isfinite(center).all() and np.isfinite(axes).all() and np.isfinite(singular).all()),
        "pca_finite",
    )
    return {
        "schema_version": "a186-pca-v1",
        "dtype": "float64",
        "normalization": "l2_nonzero",
        "whiten": False,
        "n_rows": len(vectors),
        "n_features": 4096,
        "n_components": 16,
        "center": center.tolist(),
        "components": axes.tolist(),
        "singular_values": singular.tolist(),
        "rank_tolerance": tolerance,
        "numerical_rank": rank,
    }


def transform_internal(values, transform):
    import numpy as np

    return (
        normalize_internal(values) - np.asarray(transform["center"], dtype=np.float64)
    ) @ np.asarray(transform["components"], dtype=np.float64).T


def _fit_support(roster, packets, labels):
    eligibility, indices = [], []
    class_cores = {0: set(), 1: set()}
    for i, (row, packet) in enumerate(zip(roster, packets, strict=True)):
        reasons = []
        if row["split"] != "fit":
            reasons.append("not_fitting_split")
        if packet["common"] is None:
            reasons.append("prefix_unavailable")
        if packet["text_semantic"] is None:
            reasons.append("semantic_unavailable")
        if packet["internal"] is None:
            reasons.append("internal_unavailable")
        if labels[row["observation_id"]] is None:
            reasons.append("outcome_unknown")
        eligible = not reasons
        eligibility.append(
            {"observation_id": row["observation_id"], "eligible": eligible, "reasons": reasons}
        )
        if eligible:
            indices.append(i)
            class_cores[labels[row["observation_id"]]].add(row["core_index"])
    support = {
        "eligible_rows": len(indices),
        "class_rows": {
            str(y): sum(labels[roster[i]["observation_id"]] == y for i in indices) for y in (0, 1)
        },
        "class_core_counts": {str(y): len(class_cores[y]) for y in (0, 1)},
    }
    support["passes"] = len(indices) >= 17 and all(len(class_cores[y]) >= 4 for y in (0, 1))
    return eligibility, indices, support


def _estimator():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**OPTIONS)


def _model_artifact(model, fit_ids):
    import numpy as np

    _require(np.array_equal(model.classes_, [0, 1]), "model_classes")
    _require(
        bool(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()), "model_finite"
    )
    return {
        "status": "fitted",
        "options": dict(OPTIONS),
        "classes": model.classes_.tolist(),
        "coef": model.coef_.astype(np.float64).tolist(),
        "intercept": model.intercept_.astype(np.float64).tolist(),
        "n_iter": model.n_iter_.tolist(),
        "fit_observation_ids": list(fit_ids),
    }


def replay_probability(features, artifact):
    """Replay the stored binary logistic coefficients; never fit during replay."""
    import numpy as np

    _require(artifact["status"] == "fitted", "model_unavailable")
    z = float(
        np.asarray(artifact["coef"], dtype=np.float64)[0] @ features + artifact["intercept"][0]
    )
    _require(math.isfinite(z), "decision_nonfinite")
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _predict_rows(roster, packets, models, transform, *, landmark_reached=None):
    import numpy as np

    rows = []
    reached = _landmarks(roster, packets, landmark_reached)
    for row, packet in zip(roster, packets, strict=True):
        common = packet["common"]
        text = text_features(packet)
        slots = {}
        for name in COMPARATORS:
            available = text is not None and (name == "text" or packet["internal"] is not None)
            fit_status = models[name]["status"]
            slot = {
                "fit_status": fit_status,
                "feature_available": available,
                "status": None,
                "learned_probability": None,
                "probability": None,
                "override_applied": False,
            }
            if row["split"] != "evaluation":
                slot["status"] = "not_evaluation"
            elif fit_status != "fitted":
                slot["status"] = "fit_unavailable"
            elif not available:
                slot["status"] = "feature_unavailable"
            else:
                features = (
                    text
                    if name == "text"
                    else np.concatenate((text, transform_internal(packet["internal"], transform)))
                )
                p = replay_probability(features, models[name])
                slot.update(
                    status="predicted",
                    learned_probability=p,
                    probability=0.0 if common["prefix_known_error"] else p,
                    override_applied=common["prefix_known_error"],
                )
            slots[name] = slot
        rows.append(
            {
                **row,
                "features_sha256": evidence.object_hash(packet),
                "prefix_available": common is not None,
                "landmark_reached": reached[row["observation_id"]],
                "prefix_known_error": None if common is None else common["prefix_known_error"],
                "comparators": slots,
            }
        )
    return rows


def brier_bounds(p, q, y):
    """Sharp row bounds with unknown probabilities in[0,1] and binary labels."""
    for probability in (p, q):
        _require(
            probability is None
            or (
                type(probability) in (int, float)
                and math.isfinite(probability)
                and 0 <= probability <= 1
            ),
            "probability",
        )
    _require(y is None or (type(y) is int and y in (0, 1)), "binary_label")
    if p is not None and q is not None and y is not None:
        value = (p - y) ** 2 - (q - y) ** 2
        return {"point": value, "lower": value, "upper": value, "resolved": True}
    values = []
    for target in (0, 1) if y is None else (y,):
        for left in (0.0, 1.0) if p is None else (p,):
            for right in (0.0, 1.0) if q is None else (q,):
                values.append((left - target) ** 2 - (right - target) ** 2)
    return {"point": None, "lower": min(values), "upper": max(values), "resolved": False}


def _cohort(rows, labels, *, planned_count, membership_unknown=False):
    bounds = [
        brier_bounds(
            r["comparators"]["text"]["probability"],
            r["comparators"]["text_internal"]["probability"],
            labels[r["observation_id"]],
        )
        for r in rows
    ]
    n = len(bounds)
    by_core = {}
    for row, bound in zip(rows, bounds, strict=True):
        by_core.setdefault(row["core_index"], []).append(bound)

    def average(field):
        return math.fsum(
            math.fsum(b[field] for b in values) / len(values) for values in by_core.values()
        ) / len(by_core)

    resolved = sum(b["resolved"] for b in bounds)
    all_resolved = n > 0 and resolved == n and not membership_unknown
    return {
        "planned_rows": planned_count,
        "included_rows": n,
        "included_cores": len({r["core_index"] for r in rows}),
        "resolved_rows": resolved,
        "membership_unknown": membership_unknown,
        "weighting": "equal_represented_core_then_equal_included_format",
        "point": average("point") if all_resolved else None,
        "lower": average("lower") if n and not membership_unknown else None,
        "upper": average("upper") if n and not membership_unknown else None,
    }


def class_support(roster, packets, labels, *, landmark_reached=None):
    reached = _landmarks(roster, packets, landmark_reached)
    report = {}
    for split in ("fit", "evaluation"):
        subset = [(r, p) for r, p in zip(roster, packets, strict=True) if r["split"] == split]
        unflagged = [
            (r, p)
            for r, p in subset
            if p["common"] is not None and p["common"]["prefix_known_error"] is False
        ]
        report[split] = {
            "planned_rows": len(subset),
            "known_unreached_rows": sum(reached[r["observation_id"]] is False for r, _ in subset),
            "unknown_reachability_rows": sum(
                reached[r["observation_id"]] is None for r, _ in subset
            ),
            "flagged_rows": sum(
                p["common"] is not None and p["common"]["prefix_known_error"] for _, p in subset
            ),
            "unknown_prefix_rows": sum(p["common"] is None for _, p in subset),
            "unflagged_rows": len(unflagged),
            "unflagged_class_rows": {
                str(y): sum(labels[r["observation_id"]] == y for r, _ in unflagged) for y in (0, 1)
            },
            "unflagged_class_core_counts": {
                str(y): len(
                    {r["core_index"] for r, _ in unflagged if labels[r["observation_id"]] == y}
                )
                for y in (0, 1)
            },
            "unflagged_unknown_labels": sum(
                labels[r["observation_id"]] is None for r, _ in unflagged
            ),
        }
    return report


def analyze_predictions(roster, rows, labels):
    roster = validate_roster(roster)
    _require(type(rows) is list and len(rows) == len(roster), "prediction_coverage")
    _require(
        type(labels) is dict and set(labels) == {r["observation_id"] for r in roster},
        "label_coverage",
    )
    for expected, row in zip(roster, rows, strict=True):
        _keys(
            row,
            "observation_id core_index format split features_sha256 prefix_available landmark_reached prefix_known_error comparators",
            "prediction_keys",
        )
        _require(all(_same(row[k], v) for k, v in expected.items()), "prediction_identity")
        _require(type(row["prefix_available"]) is bool, "prefix_availability")
        _require(
            row["landmark_reached"] is None or type(row["landmark_reached"]) is bool,
            "landmark_type",
        )
        _require(
            not row["prefix_available"] or row["landmark_reached"] is True, "landmark_consistency"
        )
        _require(
            (type(row["prefix_known_error"]) is bool)
            if row["prefix_available"]
            else row["prefix_known_error"] is None,
            "prefix_membership",
        )
        _require(
            type(row["comparators"]) is dict and set(row["comparators"]) == set(COMPARATORS),
            "paired_slots",
        )
        for slot in row["comparators"].values():
            _keys(
                slot,
                "fit_status feature_available status learned_probability probability override_applied",
                "prediction_slot_keys",
            )
            _require(
                type(slot["feature_available"]) is bool and type(slot["override_applied"]) is bool,
                "prediction_slot_types",
            )
            _require(
                slot["status"]
                in ("predicted", "not_evaluation", "fit_unavailable", "feature_unavailable"),
                "prediction_status",
            )
            if slot["status"] == "predicted":
                _require(
                    row["split"] == "evaluation"
                    and slot["feature_available"]
                    and slot["fit_status"] == "fitted"
                    and row["prefix_available"],
                    "prediction_available",
                )
                p = slot["learned_probability"]
                _require(
                    type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1,
                    "learned_probability",
                )
                _require(
                    slot["override_applied"] is row["prefix_known_error"]
                    and _same(slot["probability"], 0.0 if slot["override_applied"] else p),
                    "probability_override",
                )
            else:
                _require(
                    slot["probability"] is None
                    and slot["learned_probability"] is None
                    and slot["override_applied"] is False,
                    "unavailable_prediction",
                )
    primary = [r for r in rows if r["split"] == "evaluation" and r["format"] == "csv"]
    secondary = [r for r in rows if r["split"] == "evaluation" and r["format"] != "csv"]
    evaluation = primary + secondary
    unflagged = [r for r in evaluation if r["prefix_known_error"] is False]
    # Every secondary core has exactly two fixed rows: equal-row and equal-core means coincide.
    return {
        "primary_csv": _cohort(primary, labels, planned_count=16),
        "secondary_known_formats": _cohort(secondary, labels, planned_count=32),
        "unflagged_evaluation_descriptive": _cohort(
            unflagged,
            labels,
            planned_count=48,
            membership_unknown=any(
                r["landmark_reached"] is None
                or (r["landmark_reached"] is True and r["prefix_known_error"] is None)
                for r in evaluation
            ),
        ),
    }


def fit_predict(roster, feature_packets, labels, *, bindings, landmark_reached=None, event=None):
    """Two planned fits, no retries/fallback. Inputs and all112 rows are retained.

    Shared support failure makes both slots unavailable without fitting. A solver
    nonconvergence is a declared unavailable fit, and the other planned fit still
    runs. An unexpected estimator/transform exception stops subsequent fits.
    ``event`` receives immutable canonical JSON bytes before each fit entry and
    after transform/fit return or failure. Callback failure always propagates;
    BaseException interruptions propagate after the attempted failure event.
    The outer caller owns durable journals, actual body counts, and no-retry.
    """
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    _require(event is None or callable(event), "event_callback")
    callback_error = None

    def emit(kind, **payload):
        nonlocal callback_error
        if event is not None:
            try:
                event(
                    evidence.canonical(
                        {"schema_version": "a186-fit-event-v1", "kind": kind, **payload}
                    )
                )
            except BaseException as error:
                callback_error = error
                raise

    roster, packets, labels, bindings = validate_inputs(
        roster, feature_packets, labels, bindings, landmark_reached=landmark_reached
    )
    reached = _landmarks(roster, packets, landmark_reached)
    eligibility, indices, support = _fit_support(roster, packets, labels)
    models = {
        name: {"status": "unavailable_support", "reason": "shared_support"} for name in COMPARATORS
    }
    transform, actual_fits, terminal_error = None, 0, None
    if support["passes"]:
        models = {
            name: {"status": "unattempted", "reason": "prior_failure"} for name in COMPARATORS
        }
        try:
            transform = fit_pca([packets[i]["internal"] for i in indices])
            emit(
                "transform_return",
                artifact=transform,
                artifact_sha256=evidence.object_hash(transform),
            )
            base = np.stack([text_features(packets[i]) for i in indices])
            internal = np.stack(
                [transform_internal(packets[i]["internal"], transform) for i in indices]
            )
            y = np.asarray([labels[roster[i]["observation_id"]] for i in indices], dtype=np.int64)
            fit_ids = [roster[i]["observation_id"] for i in indices]
            for name in COMPARATORS:
                matrix = base if name == "text" else np.concatenate((base, internal), axis=1)
                model = _estimator()
                emit(
                    "before_fit",
                    slot=name,
                    attempt_index=actual_fits + 1,
                    options=OPTIONS,
                    fit_observation_ids_sha256=evidence.object_hash(fit_ids),
                    matrix_shape=list(matrix.shape),
                    matrix_fp64le_sha256=evidence.sha256(matrix.astype("<f8").tobytes()),
                    labels_sha256=evidence.object_hash(y.tolist()),
                )
                actual_fits += 1
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(matrix, y)
                    nonconverged = any(
                        issubclass(w.category, ConvergenceWarning) for w in caught
                    ) or bool(np.any(model.n_iter_ >= OPTIONS["max_iter"]))
                    models[name] = (
                        {"status": "nonconverged", "reason": "fixed_solver_did_not_converge"}
                        if nonconverged
                        else _model_artifact(model, fit_ids)
                    )
                    emit(
                        "fit_return",
                        slot=name,
                        attempt_index=actual_fits,
                        artifact=models[name],
                        artifact_sha256=evidence.object_hash(models[name]),
                    )
                except BaseException as error:
                    if error is callback_error:
                        raise
                    emit(
                        "fit_failure",
                        slot=name,
                        attempt_index=actual_fits,
                        error_type=type(error).__name__,
                    )
                    if not isinstance(error, Exception):
                        raise
                    models[name] = {"status": "failed", "error_type": type(error).__name__}
                    terminal_error = type(error).__name__
                    break
        except Exception as error:
            if error is callback_error:
                raise
            terminal_error = type(error).__name__
    rows = _predict_rows(roster, packets, models, transform, landmark_reached=reached)
    return {
        "schema_version": SCHEMA,
        "bindings": bindings,
        "runtime": runtime_descriptor(),
        "roster_sha256": evidence.object_hash(roster),
        "landmark_reached": reached,
        "planned_fits": 2,
        "actual_fits": actual_fits,
        "terminal_error": terminal_error,
        "fit_eligibility": eligibility,
        "fit_support": support,
        "transform": transform,
        "models": models,
        "predictions": rows,
        "class_support": class_support(roster, packets, labels, landmark_reached=reached),
        "analysis": analyze_predictions(roster, rows, labels),
    }


def validate_result(result, roster, feature_packets, labels, *, bindings, landmark_reached=None):
    """No refit: replay retained models/transforms and all prediction arithmetic."""
    roster, packets, labels, bindings = validate_inputs(
        roster, feature_packets, labels, bindings, landmark_reached=landmark_reached
    )
    reached = _landmarks(roster, packets, landmark_reached)
    _keys(
        result,
        "schema_version bindings runtime roster_sha256 landmark_reached planned_fits actual_fits terminal_error fit_eligibility fit_support transform models predictions class_support analysis",
        "result_keys",
    )
    _require(
        result["schema_version"] == SCHEMA and _same(result["bindings"], bindings), "result_binding"
    )
    _require(_same(result["landmark_reached"], reached), "result_landmarks")
    _require(
        _same(result["runtime"], runtime_descriptor())
        and result["roster_sha256"] == evidence.object_hash(roster),
        "result_runtime_roster",
    )
    _require(
        type(result["actual_fits"]) is int
        and 0 <= result["actual_fits"] <= 2
        and type(result["planned_fits"]) is int
        and result["planned_fits"] == 2,
        "fit_accounting",
    )
    eligibility, indices, support = _fit_support(roster, packets, labels)
    _require(
        _same(result["fit_eligibility"], eligibility) and _same(result["fit_support"], support),
        "fit_cohort",
    )
    _require(
        type(result["models"]) is dict and set(result["models"]) == set(COMPARATORS), "model_slots"
    )
    _require(
        result["terminal_error"] is None
        or (type(result["terminal_error"]) is str and bool(result["terminal_error"])),
        "terminal_error_type",
    )
    for model in result["models"].values():
        _require(
            type(model) is dict
            and model.get("status")
            in ("fitted", "failed", "nonconverged", "unattempted", "unavailable_support"),
            "model_status",
        )
        status = model["status"]
        if status == "failed":
            _keys(model, "status error_type", "failed_model_keys")
            _require(
                type(model["error_type"]) is str
                and model["error_type"] == result["terminal_error"],
                "failed_model_error",
            )
        elif status != "fitted":
            _keys(model, "status reason", "unavailable_model_keys")
            _require(
                model["reason"]
                == {
                    "unattempted": "prior_failure",
                    "unavailable_support": "shared_support",
                    "nonconverged": "fixed_solver_did_not_converge",
                }[status],
                "model_reason",
            )
    fitted = sum(m["status"] == "fitted" for m in result["models"].values())
    attempted = sum(
        m["status"] in ("fitted", "nonconverged", "failed") for m in result["models"].values()
    )
    _require(result["actual_fits"] == attempted and fitted <= attempted, "fit_count")
    if not support["passes"]:
        _require(
            result["actual_fits"] == 0
            and result["transform"] is None
            and result["terminal_error"] is None
            and all(m["status"] == "unavailable_support" for m in result["models"].values()),
            "support_guard",
        )
    else:
        statuses = [result["models"][name]["status"] for name in COMPARATORS]
        _require("unavailable_support" not in statuses, "supported_model_status")
        if result["terminal_error"] is None:
            _require(
                all(status in ("fitted", "nonconverged") for status in statuses),
                "successful_fit_schedule",
            )
        else:
            _require(
                any(status in ("failed", "unattempted") for status in statuses),
                "failed_fit_schedule",
            )
        if statuses[0] in ("failed", "unattempted"):
            _require(statuses[1] == "unattempted", "no_fit_after_failure")
        if attempted:
            _require(result["transform"] is not None, "fit_requires_transform")
    if result["transform"] is not None:
        _require(
            _same(result["transform"], fit_pca([packets[i]["internal"] for i in indices])),
            "pca_replay",
        )
    for name, model in result["models"].items():
        if model["status"] == "fitted":
            _keys(
                model,
                "status options classes coef intercept n_iter fit_observation_ids",
                "model_keys",
            )
            _require(
                _same(model["options"], OPTIONS) and _same(model["classes"], [0, 1]),
                "model_options",
            )
            _require(
                model["fit_observation_ids"] == [roster[i]["observation_id"] for i in indices],
                "model_fit_ids",
            )
            _require(
                type(model["coef"]) is list
                and len(model["coef"]) == 1
                and len(model["coef"][0]) == (1026 if name == "text" else 1042),
                "model_width",
            )
            _require(
                type(model["intercept"]) is list
                and len(model["intercept"]) == 1
                and all(
                    type(x) in (int, float) and math.isfinite(x)
                    for x in model["coef"][0] + model["intercept"]
                ),
                "model_numeric",
            )
            _require(
                type(model["n_iter"]) is list
                and len(model["n_iter"]) == 1
                and type(model["n_iter"][0]) is int
                and 0 < model["n_iter"][0] < OPTIONS["max_iter"],
                "model_iterations",
            )
    rows = _predict_rows(
        roster, packets, result["models"], result["transform"], landmark_reached=reached
    )
    _require(_same(result["predictions"], rows), "prediction_replay")
    _require(
        _same(result["analysis"], analyze_predictions(roster, rows, labels)), "analysis_replay"
    )
    _require(
        _same(
            result["class_support"],
            class_support(roster, packets, labels, landmark_reached=reached),
        ),
        "support_replay",
    )
    return _snapshot(result)


def save_result(path, result, roster, feature_packets, labels, *, bindings, landmark_reached=None):
    """Exclusive single-file artifact. A partial write is not a valid saved result."""
    import os

    validated = validate_result(
        result,
        roster,
        feature_packets,
        labels,
        bindings=bindings,
        landmark_reached=landmark_reached,
    )
    raw = evidence.canonical(validated)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return evidence.sha256(raw)


def load_result(
    path, roster, feature_packets, labels, *, bindings, expected_sha256, landmark_reached=None
):
    import stat

    path = Path(path)
    _require(stat.S_ISREG(path.lstat().st_mode), "result_regular_file")
    raw = path.read_bytes()
    _require(evidence.sha256(raw) == expected_sha256, "result_file_hash")
    parsed = json.loads(raw)
    _require(evidence.canonical(parsed) == raw, "result_canonical")
    return validate_result(
        parsed,
        roster,
        feature_packets,
        labels,
        bindings=bindings,
        landmark_reached=landmark_reached,
    )
