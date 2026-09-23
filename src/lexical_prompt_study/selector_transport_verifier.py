"""Independent, model-free A182 evidence and finite-cohort verification.

This module imports neither the runtime nor Torch/Transformers. Saved scalar
normalizers can be checked against chosen-logit arithmetic, but cannot be
recomputed from absent full-vocabulary logits. Structural native checks do not
replace an independently bound tokenizer audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import stat
import struct
from fractions import Fraction
from pathlib import Path


TOLERANCE = 1e-4
SEPARATION = 1e-3
PREFIX_COUNT = 3
LAYER_INDEX = 15
HIDDEN_WIDTH = 4096
if not __debug__:
    raise RuntimeError("a182_verifier_requires_unoptimized_python")


ARMS = ("no_patch", "self", "match", "opposite")
OWN = ("R1", "R2")
FOUR = ("R1", "R2", "D1", "D2")
DONOR = ("D1", "D2")
RECIPIENT_PAIRS = (("s112", "s113"), ("s114", "s115"))
DONOR_PAIRS = (("s116", "s117"), ("s118", "s119"))
SELECTORS = (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"))


class VerificationError(ValueError):
    """Only fixed, non-private diagnostic codes leave this verifier."""


def require(condition, code):
    if not condition:
        raise VerificationError("a182_verify_" + code)


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        + b"\n"
    )


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def conjunction(values):
    values = list(values)
    require(
        bool(values) and all(value is None or type(value) is bool for value in values), "flag_type"
    )
    return (
        False
        if any(value is False for value in values)
        else True
        if all(value is True for value in values)
        else None
    )


def counts(values):
    values = list(values)
    require(all(value is None or type(value) is bool for value in values), "flag_type")
    return {
        "true": sum(v is True for v in values),
        "false": sum(v is False for v in values),
        "unknown": sum(v is None for v in values),
    }


def verify_measurement(value, target_count):
    """Replay saved chosen-logit subtraction and every q/EOS-inclusive sum."""
    require(type(target_count) is int and target_count > PREFIX_COUNT, "target_count")
    require(
        type(value) is dict
        and set(value) == {"chosen_logits", "log_normalizers", "token_logprobs", "scores"},
        "measurement_shape",
    )
    for key in ("chosen_logits", "log_normalizers", "token_logprobs"):
        require(
            type(value[key]) is list
            and len(value[key]) == target_count
            and all(finite(v) for v in value[key]),
            "measurement_finiteness",
        )
    lp = [a - b for a, b in zip(value["chosen_logits"], value["log_normalizers"])]
    require(all(math.isfinite(v) and v <= 0 for v in lp), "logprob_domain")
    require(canonical(lp) == canonical(value["token_logprobs"]), "logprob_replay")
    prefix, suffix = math.fsum(lp[:PREFIX_COUNT]), math.fsum(lp[PREFIX_COUNT:])
    expected = {
        "json_prefix_logprob": prefix,
        "answer_continuation_logprob": suffix,
        "joint_logprob": math.fsum((prefix, suffix)),
        "eos_logprob": lp[-1],
        "json_prefix_tokens": PREFIX_COUNT,
        "answer_continuation_tokens": target_count - PREFIX_COUNT,
        "joint_tokens": target_count,
        "eos_tokens": 1,
    }
    require(
        all(
            finite(expected[k])
            for k in (
                "json_prefix_logprob",
                "answer_continuation_logprob",
                "joint_logprob",
                "eos_logprob",
            )
        ),
        "sum_finiteness",
    )
    require(canonical(value["scores"]) == canonical(expected), "sum_replay")
    return expected


def cohort(measurements, candidates):
    """Independent complete-prefix / locally decidable repeat tri-states."""
    require(
        type(measurements) is dict and all(name in measurements for name in candidates),
        "cohort_candidates",
    )
    available = []
    drift, repeat = {}, {}
    for candidate in candidates:
        pair = measurements[candidate]
        require(type(pair) is list and len(pair) == 2, "repeat_slots")
        available.extend(item for item in pair if item is not None)
        if any(item is None for item in pair):
            drift[candidate] = repeat[candidate] = None
        else:
            scores = [item["scores"]["answer_continuation_logprob"] for item in pair]
            require(all(finite(item) for item in scores), "cohort_score_finiteness")
            drift[candidate] = abs(scores[1] - scores[0])
            repeat[candidate] = drift[candidate] <= TOLERANCE
    complete = len(available) == 2 * len(candidates)
    prefixes = [item["scores"]["json_prefix_logprob"] for item in available]
    require(all(finite(v) for v in prefixes), "cohort_prefix_finiteness")
    spread = max(prefixes) - min(prefixes) if complete else None
    prefix_guard = spread <= TOLERANCE if complete else None
    return {
        "complete": complete,
        "completed_measurements": len(available),
        "prefix_sum_range_nats": spread,
        "prefix_guard_passed": prefix_guard,
        "repeat_drift_nats": drift,
        "repeat_guard_passed": repeat,
        "numerical_valid": conjunction([prefix_guard, *repeat.values()]),
    }


def score_pair(measurements, candidate):
    return [item["scores"]["answer_continuation_logprob"] for item in measurements[candidate]]


def margin(measurements, selected, other, validity):
    if validity is not True:
        return {"mean": None, "worst": None, "best": None}
    selected_scores, other_scores = (
        score_pair(measurements, selected),
        score_pair(measurements, other),
    )
    mean = math.fsum(selected_scores) / 2 - math.fsum(other_scores) / 2
    return {
        "mean": mean,
        "worst": min(selected_scores) - max(other_scores),
        "best": max(selected_scores) - min(other_scores),
    }


def dominance(measurements, expected, competitors, validity):
    if validity is not True:
        return {"separation_nats": None, "passed": None}
    separation = min(score_pair(measurements, expected)) - max(
        score for name in competitors for score in score_pair(measurements, name)
    )
    return {"separation_nats": separation, "passed": separation > SEPARATION}


def prefix_control(measurements):
    """Never infer a prefix-range flag from an incomplete designated cohort."""
    if any(value is None for value in measurements):
        return {"range_nats": None, "passed": None}
    values = [m["scores"]["json_prefix_logprob"] for m in measurements]
    spread = max(values) - min(values)
    return {"range_nats": spread, "passed": spread <= TOLERANCE}


def self_control(baseline, patched):
    differences = {name: [None, None] for name in FOUR}
    flags = []
    for name in FOUR:
        for index in range(2):
            left, right = baseline[name][index], patched[name][index]
            difference = (
                None
                if left is None or right is None
                else abs(
                    left["scores"]["answer_continuation_logprob"]
                    - right["scores"]["answer_continuation_logprob"]
                )
            )
            differences[name][index] = difference
            flags.append(None if difference is None else difference <= TOLERANCE)
    return {"differences_nats": differences, "passed": conjunction(flags)}


def fixed_summary(values):
    known = [v for v in values if v is not None]
    require(all(finite(v) for v in known), "summary_finite")
    point = (
        float(sum((Fraction(v) for v in known), Fraction()) / len(values))
        if len(known) == len(values)
        else None
    )
    return {
        "planned": len(values),
        "resolved": len(known),
        "point": point,
        "lower": point,
        "upper": point,
        "unbounded": point is None,
    }


def verify_capture(metadata, raw, evaluation_id, attempt_sha256, prompt_count):
    require(type(raw) is bytes and len(raw) == HIDDEN_WIDTH * 4, "capture_bytes")
    expected = {
        "schema_version": "a182-capture-v1",
        "evaluation_id": evaluation_id,
        "attempt_sha256": attempt_sha256,
        "layer_index": LAYER_INDEX,
        "absolute_position": prompt_count + PREFIX_COUNT - 1,
        "shape": [1, HIDDEN_WIDTH],
        "dtype": "float32",
        "byte_order": "little",
        "bytes": HIDDEN_WIDTH * 4,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    require(canonical(metadata) == canonical(expected), "capture_binding")
    values = struct.unpack("<" + str(HIDDEN_WIDTH) + "f", raw)
    require(all(math.isfinite(v) for v in values), "capture_finiteness")
    return values


FIXED_BINDINGS = {
    "config_sha256": "1ed34956372d265be806316676da7f4b899dbe741a00e3b599bc4693c62f3745",
    "a179_helper_sha256": "af0b811da83166db5a410956c1b7160e7fde297458f07d4b531a30586f8790e8",
    "a169_helper_sha256": "1b63ddf15eae5866b444cdb94293b80dcf7bd6e580978a23873c0167cb40e69e",
    "apparatus_sha256": "1df9515417ecbc7395c23a92f128d4ef13a1969801b6c7e9fad35ada841c643b",
    "reference_script_sha256": "f704a8a223c792ca35b7f2e0e000537fc366fd2f267dbb5d3802fe9c3ec4321e",
}


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def expected_plan(bindings):
    """Independent public construction; never calls the experiment compiler."""
    require(
        type(bindings) is dict
        and set(bindings)
        == set(FIXED_BINDINGS) | {"protocol_sha256", "tests_sha256", "source_sha256"},
        "plan_bindings",
    )
    require(
        all(is_hash(value) for value in bindings.values())
        and all(bindings[key] == value for key, value in FIXED_BINDINGS.items()),
        "plan_binding_values",
    )
    contexts = []
    for kind in ("donor", "recipient"):
        for world in range(4):
            pair = world // 2
            values = dict(zip(FOUR, (*RECIPIENT_PAIRS[pair], *DONOR_PAIRS[pair])))
            own = DONOR if kind == "donor" else OWN
            reverse = world % 2 == (0 if kind == "donor" else 1)
            labels = {"A": own[int(reverse)], "B": own[1 - int(reverse)]}
            mapping = {label: values[candidate] for label, candidate in labels.items()}
            for selector_index, selector in enumerate(SELECTORS[world]):
                opposite = {"A": "B", "B": "A"}[selector]
                system = (
                    f"Select the entry labeled {selector} from the user's mapping. "
                    f'Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value. '
                    "Do not transform the value or include other fields, explanations, or Markdown fences."
                )
                user = json.dumps(mapping, separators=(",", ":"))
                if kind == "recipient":
                    user += f"\n\nSelect the entry labeled {opposite} from the mapping."
                c = {
                    "kind": kind,
                    "world_index": world,
                    "world_id": f"w{world:02d}",
                    "pair_index": pair,
                    "assignment_swapped": reverse,
                    "presentation_order": "AB",
                    "base_context_index": 2 * world + selector_index,
                    "selector": selector,
                    "user_selector": opposite if kind == "recipient" else None,
                    "selected_candidate": labels[selector],
                    "other_candidate": labels[opposite],
                    "selected_answer": mapping[selector],
                    "unselected_answer": mapping[opposite],
                    "context_index": len(contexts),
                    "mapping": mapping,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "canonical_candidates": {
                        name: json.dumps({"answer": values[name]}, separators=(",", ":"))
                        for name in (DONOR if kind == "donor" else FOUR)
                    },
                }
                c["context_id"] = digest(
                    {
                        "schema_version": "a182-context-identity-v1",
                        "bindings": bindings,
                        "context": c,
                    }
                )[:24]
                contexts.append(c)
    lookup = {(c["kind"], c["world_index"], c["selector"]): c for c in contexts}
    evaluations, references = [], {}
    for c in contexts:
        world, kind = c["world_index"], c["kind"]
        if kind == "donor":
            arms = ("baseline",)
            sequence = ("D1", "D2", "D2", "D1") if world % 2 == 0 else ("D2", "D1", "D1", "D2")
        else:
            orders = (
                ("self", "match", "opposite"),
                ("match", "opposite", "self"),
                ("opposite", "self", "match"),
            )
            patched = orders[world % 3]
            arms = ("no_patch", *(patched if c["base_context_index"] % 2 == 0 else patched[::-1]))
            sequence = (
                ("R1", "R2", "D1", "D2", "D2", "D1", "R2", "R1")
                if world % 2 == 0
                else ("D2", "D1", "R2", "R1", "R1", "R2", "D1", "D2")
            )
        for arm in arms:
            reference = None
            if arm in ("self", "match", "opposite"):
                source = (
                    c
                    if arm == "self"
                    else lookup[
                        "donor", world, c["selector"] if arm == "match" else c["user_selector"]
                    ]
                )
                reference = references[source["context_id"]]
            seen = set()
            for candidate in sequence:
                e = {
                    "context_index": c["context_index"],
                    "context_id": c["context_id"],
                    "arm": arm,
                    "candidate": candidate,
                    "repeat_index": int(candidate in seen),
                    "sequence_index": len(evaluations),
                    "source_evaluation_id": reference,
                }
                e["evaluation_id"] = digest(
                    {
                        "schema_version": "a182-evaluation-identity-v1",
                        "bindings": bindings,
                        "evaluation": e,
                    }
                )[:24]
                evaluations.append(e)
                seen.add(candidate)
                if (
                    arm in ("baseline", "no_patch")
                    and candidate == ("D1" if kind == "donor" else "R1")
                    and e["repeat_index"] == 0
                ):
                    references[c["context_id"]] = e["evaluation_id"]
    return {
        "schema_version": "a182-plan-v1",
        "partition": "fresh_public_cross_world_selector_transport",
        "bindings": dict(bindings),
        "contexts": contexts,
        "evaluations": evaluations,
        "planned_contexts": 16,
        "planned_native_candidate_paths": 48,
        "planned_scored_contexts": 40,
        "planned_candidate_paths": 144,
        "planned_forward_calls": 288,
        "planned_donor_value_pairs": 2,
        "planned_recipient_value_pairs": 2,
        "planned_world_assignments": 4,
        "prefix_tokens": 3,
        "layer_index": 15,
        "model_layers": 32,
        "hidden_width": 4096,
        "numerical_tolerance_nats": TOLERANCE,
        "functional_separation_nats": SEPARATION,
        "primary": {
            "name": "match_minus_opposite",
            "planned_pairs": 8,
            "planned_arm_margins": 16,
            "planned_measurements": 64,
            "candidate_cohort": ["R1", "R2"],
            "cross_arm_prefix_guard": True,
        },
        "source_references": references,
        "capture_equality": "finite_elementwise_exact_float32",
        "suffix_includes_eos": True,
        "equal_candidate_lengths_required": True,
        "padding_allowed": False,
        "retries": 0,
        "resume_allowed": False,
        "heldout_allowed": False,
        "interim_gate": False,
        "generation_allowed": False,
        "scaffold_allowed": False,
        "warmup_allowed": False,
    }


def verify_plan(plan, expected_bindings=None):
    require(type(plan) is dict and type(plan.get("bindings")) is dict, "plan_type")
    if expected_bindings is not None:
        require(
            canonical(plan["bindings"]) == canonical(expected_bindings), "external_plan_bindings"
        )
    require(canonical(plan) == canonical(expected_plan(plan["bindings"])), "independent_plan")
    return plan


def verify_prepared(plan, prepared):
    """Check bound native-array structure; does not claim tokenizer execution."""
    require(
        type(prepared) is dict and set(prepared) == {c["context_id"] for c in plan["contexts"]},
        "native_cohort",
    )
    prompts, prefixes = set(), set()
    keys = {
        "rendered_text",
        "prompt_token_ids",
        "chat_template_sha256",
        "shared_json_prefix_token_ids",
        "shared_json_prefix_text",
        "candidate_token_ids",
        "candidate_lengths",
        "intended_eos_token_id",
        "model_context_limit",
        "model_vocab_size",
        "hidden_width",
        "model_layers",
        "absolute_position",
    }
    for c in plan["contexts"]:
        p = prepared[c["context_id"]]
        require(type(p) is dict and set(p) == keys, "native_shape")
        require(
            type(p["rendered_text"]) is str
            and all(m["content"] in p["rendered_text"] for m in c["messages"])
            and is_hash(p["chat_template_sha256"]),
            "native_rendering_metadata",
        )
        for key, value in (
            ("model_context_limit", 131072),
            ("model_vocab_size", 128256),
            ("hidden_width", 4096),
            ("model_layers", 32),
        ):
            require(type(p[key]) is int and p[key] == value, "native_model_dimensions")
        prompt, paths, lengths = (
            p["prompt_token_ids"],
            p["candidate_token_ids"],
            p["candidate_lengths"],
        )
        require(type(prompt) is list and 0 < len(prompt) <= 2048, "native_prompt")
        require(
            type(paths) is dict
            and type(lengths) is dict
            and set(paths) == set(lengths) == set(c["canonical_candidates"]),
            "native_candidates",
        )
        prefix, text = p["shared_json_prefix_token_ids"], p["shared_json_prefix_text"]
        require(
            type(prefix) is list
            and len(prefix) == 3
            and all(type(t) is int and 0 <= t < 128256 for t in prefix)
            and type(text) is str
            and bool(text)
            and '{"answer":"'.startswith(text),
            "native_prefix",
        )
        eos = p["intended_eos_token_id"]
        require(type(eos) is int and eos in (128001, 128008, 128009), "native_eos")
        require(
            type(p["absolute_position"]) is int and p["absolute_position"] == len(prompt) + 2,
            "native_site",
        )
        for name, target in paths.items():
            require(
                type(target) is list
                and len(target) > 3
                and type(lengths[name]) is int
                and lengths[name] == len(target)
                and target[:3] == prefix
                and target[-1] == eos
                and not set(target[:-1]).intersection((128001, 128008, 128009)),
                "native_target_geometry",
            )
            require(
                all(type(t) is int and 0 <= t < 128256 for t in prompt + target)
                and len(prompt) + len(target) <= min(2048 + 64, 131072),
                "native_token_limits",
            )
        require(len(set(lengths.values())) == 1, "native_equal_lengths")
        prompts.add(digest(prompt))
        prefixes.add(digest(prefix))
    require(len(prompts) == 16 and len(prefixes) == 1, "native_unique_shared")


def source_eligibility(plan, context_id, results, captures):
    baseline = [
        e
        for e in plan["evaluations"]
        if e["context_id"] == context_id and e["arm"] in ("baseline", "no_patch")
    ]
    require(len(baseline) in (4, 8), "source_count")
    states, vectors = [], []
    for e in baseline:
        result = results.get(e["evaluation_id"])
        if result is None:
            states.append(None)
        elif result["status"] != "completed":
            states.append(False)
        else:
            raw = captures.get(e["evaluation_id"])
            require(type(raw) is bytes and len(raw) == 16384, "source_capture_missing")
            values = struct.unpack("<4096f", raw)
            require(all(math.isfinite(v) for v in values), "source_capture_nonfinite")
            vectors.append(values)
            states.append(True)
    unequal = bool(vectors) and any(v != vectors[0] for v in vectors[1:])
    equal = False if unequal else True if len(vectors) == len(baseline) else None
    return conjunction([*states, equal])


def source_binding(plan, evaluation, results, captures):
    reference = evaluation["source_evaluation_id"]
    if reference is None:
        return None
    source = next(e for e in plan["evaluations"] if e["evaluation_id"] == reference)
    group = [
        e
        for e in plan["evaluations"]
        if e["context_id"] == source["context_id"] and e["arm"] in ("baseline", "no_patch")
    ]
    require(
        all(e["sequence_index"] < evaluation["sequence_index"] for e in group), "dependency_order"
    )
    reference_result = results.get(reference)
    return {
        "source_evaluation_id": reference,
        "source_context_id": source["context_id"],
        "eligibility": source_eligibility(plan, source["context_id"], results, captures),
        "baseline_result_sha256": {
            e["evaluation_id"]: digest(results[e["evaluation_id"]])
            if e["evaluation_id"] in results
            else None
            for e in group
        },
        "capture_sha256": digest(reference_result["capture"])
        if reference_result is not None and reference_result["capture"] is not None
        else None,
    }


def _prefix_schema(values):
    return {
        "planned_measurements": len(values),
        "completed_measurements": sum(v is not None for v in values),
        **prefix_control(values),
    }


def _cohort_schema(values, candidates, selected, other):
    c = cohort(values, candidates)
    m = margin(values, selected, other, c["numerical_valid"])
    means = {
        name: math.fsum(score_pair(values, name)) / 2 if c["numerical_valid"] is True else None
        for name in candidates
    }
    return {
        "candidates": list(candidates),
        "planned_measurements": 2 * len(candidates),
        "completed_measurements": c["completed_measurements"],
        "complete": c["complete"],
        "prefix_guard": _prefix_schema([v for name in candidates for v in values[name]]),
        "repeat_drift_nats": c["repeat_drift_nats"],
        "repeat_guard_passed": c["repeat_guard_passed"],
        "numerical_valid": c["numerical_valid"],
        "candidate_mean_scores": means,
        "mean_margin_nats": m["mean"],
        "worst_margin_nats": m["worst"],
        "best_margin_nats": m["best"],
        "mean_tie": m["mean"] == 0 if m["mean"] is not None else None,
        "mean_reversed": m["mean"] < 0 if m["mean"] is not None else None,
    }


def _functional_schema(validity, separation):
    if validity is not True:
        separation = None
    passed = separation > SEPARATION if separation is not None else None
    return {
        "numerical_valid": validity,
        "separation_nats": separation,
        "functional_passed": passed,
        "threshold_boundary": separation == SEPARATION if separation is not None else None,
        "tie": separation == 0 if separation is not None else None,
        "reversed": separation < 0 if separation is not None else None,
        "qualified": conjunction([validity, passed]),
    }


def reconstruct_records(plan, prepared, results, captures):
    """Construct all fixed record fields from measurements, without runtime calls."""
    del prepared  # Geometry has been verified separately; no numeric input is taken from it.
    matrices = {}
    for c in plan["contexts"]:
        for arm in ("baseline",) if c["kind"] == "donor" else ARMS:
            matrices[c["context_id"], arm] = {
                name: [None, None] for name in c["canonical_candidates"]
            }
    for e in plan["evaluations"]:
        r = results.get(e["evaluation_id"])
        if r is not None and r["status"] == "completed":
            matrices[e["context_id"], e["arm"]][e["candidate"]][e["repeat_index"]] = r[
                "measurement"
            ]
    output = {"schema_version": "a182-records-v1", "donors": [], "recipients": []}
    metadata = (
        "context_id",
        "context_index",
        "kind",
        "world_index",
        "pair_index",
        "base_context_index",
        "selector",
        "selected_candidate",
        "other_candidate",
    )
    for c in plan["contexts"]:
        row = {key: c[key] for key in metadata}
        cid, selected, other = c["context_id"], c["selected_candidate"], c["other_candidate"]
        row["source_eligible"] = source_eligibility(plan, cid, results, captures)
        if c["kind"] == "donor":
            own = _cohort_schema(matrices[cid, "baseline"], DONOR, selected, other)
            row["own_value"] = own
            row["competence"] = _functional_schema(own["numerical_valid"], own["worst_margin_nats"])
            output["donors"].append(row)
            continue
        arm_rows = {}
        for arm in ARMS:
            matrix = matrices[cid, arm]
            own = _cohort_schema(matrix, OWN, selected, other)
            four = _cohort_schema(matrix, FOUR, selected, other)
            expected = selected if arm == "match" else other if arm == "opposite" else None
            functional = None
            if expected is not None:
                d = dominance(
                    matrix,
                    expected,
                    [name for name in FOUR if name != expected],
                    four["numerical_valid"],
                )
                functional = _functional_schema(four["numerical_valid"], d["separation_nats"])
            arm_rows[arm] = {
                "own_value": own,
                "four_path": four,
                "expected_recipient_candidate": expected,
                "recipient_content": functional,
            }
        primary_prefix = _prefix_schema(
            [m for arm in ("match", "opposite") for name in OWN for m in matrices[cid, arm][name]]
        )
        validity = conjunction(
            [
                primary_prefix["passed"],
                *[arm_rows[a]["own_value"]["numerical_valid"] for a in ("match", "opposite")],
            ]
        )
        difference = (
            arm_rows["match"]["own_value"]["mean_margin_nats"]
            - arm_rows["opposite"]["own_value"]["mean_margin_nats"]
            if validity is True
            else None
        )
        baseline = arm_rows["no_patch"]["own_value"]
        self_check = self_control(matrices[cid, "no_patch"], matrices[cid, "self"])
        row.update(
            arms=arm_rows,
            conflict=_functional_schema(
                baseline["numerical_valid"],
                -baseline["best_margin_nats"] if baseline["best_margin_nats"] is not None else None,
            ),
            primary={
                "prefix_guard": primary_prefix,
                "numerical_valid": validity,
                "difference_nats": difference,
            },
            full_cross_arm_prefix_guard=_prefix_schema(
                [m for arm in ARMS for name in FOUR for m in matrices[cid, arm][name]]
            ),
            self_no_patch={
                "drift_nats": self_check["differences_nats"],
                "passed": self_check["passed"],
            },
        )
        output["recipients"].append(row)
    return output


def reconstruct_analysis(plan, records, results, attempted, invoked):
    donors, recipients = records["donors"], records["recipients"]
    differences = [row["primary"]["difference_nats"] for row in recipients]
    point = math.fsum(differences) / 8 if all(v is not None for v in differences) else None
    primary_valid = conjunction(row["primary"]["numerical_valid"] for row in recipients)
    own = [row["arms"][arm]["own_value"] for row in recipients for arm in ("match", "opposite")]
    source = [row["source_eligible"] for row in [*donors, *recipients]]
    four = [row["arms"][arm]["four_path"]["numerical_valid"] for row in recipients for arm in ARMS]
    prefix = [row["full_cross_arm_prefix_guard"]["passed"] for row in recipients]
    identity = [row["self_no_patch"]["passed"] for row in recipients]
    apparatus = conjunction([*source, *four, *prefix, *identity])
    gate_rows = {
        "donor_competence": [r["competence"] for r in donors],
        "recipient_conflict": [r["conflict"] for r in recipients],
        "recipient_content": [
            r["arms"][a]["recipient_content"] for r in recipients for a in ("match", "opposite")
        ],
    }
    gates = {
        key: {
            "planned": len(rows),
            "numerical_valid": counts(r["numerical_valid"] for r in rows),
            "functional_passed": counts(r["functional_passed"] for r in rows),
            "qualified": conjunction(r["qualified"] for r in rows),
        }
        for key, rows in gate_rows.items()
    }
    completed = sum(r["status"] == "completed" for r in results.values())
    return {
        "schema_version": "a182-analysis-v1",
        "status": "complete" if completed == 288 else "incomplete",
        "planned_contexts": 16,
        "planned_native_candidate_paths": 48,
        "planned_scored_contexts": 40,
        "planned_candidate_paths": 144,
        "planned_forward_calls": 288,
        "coverage": {
            "attempted": len(attempted),
            "processed_slots": len(results),
            "forward_dispatch_entries": len(invoked),
            "completed": completed,
            "infrastructure_failed": sum(
                r["status"] == "infrastructure_failed" for r in results.values()
            ),
            "dependency_unavailable": sum(
                r["status"] == "dependency_unavailable" for r in results.values()
            ),
            "interrupted": len(attempted) - len(results),
            "unattempted": 288 - len(attempted),
            "missing": 288 - completed,
        },
        "primary": {
            "planned_pairs": 8,
            "resolved_pairs": sum(x is not None for x in differences),
            "planned_arm_margins": 16,
            "resolved_arm_margins": sum(c["mean_margin_nats"] is not None for c in own),
            "planned_measurements": 64,
            "completed_measurements": sum(c["completed_measurements"] for c in own),
            "numerical_valid": primary_valid,
            "point": point,
            "lower": point,
            "upper": point,
            "unbounded": point is None,
        },
        "apparatus": {
            "source_eligible": counts(source),
            "four_path_numerical_valid": counts(four),
            "full_cross_arm_prefix_guard": counts(prefix),
            "self_no_patch": counts(identity),
            "qualified": apparatus,
        },
        "functional_gates": gates,
        "qualified_transport": conjunction(
            [
                primary_valid,
                point > 0 if point is not None else None,
                apparatus,
                *[g["qualified"] for g in gates.values()],
            ]
        ),
        "numeric_records_sha256": digest(records),
        "plan_object_sha256": digest(plan),
        "claim_boundaries": {
            "canonical_path_conditional_preference_only": True,
            "suffix_includes_eos": True,
            "saved_normalizers_not_independently_recomputed": True,
            "generation_accuracy": False,
            "scaffold_mediation": False,
            "abstract_selector_transport_established": False,
            "population_inference": False,
            "confidence_intervals": False,
            "complete_case_substitution": False,
            "missing_imputed": False,
            "adaptive_tolerance": False,
        },
    }


def _error_shape(value):
    require(
        type(value) is dict and set(value) == {"exception_type", "frames", "code"}, "error_shape"
    )
    require(
        type(value["exception_type"]) is str
        and bool(value["exception_type"])
        and type(value["frames"]) is list
        and (value["code"] is None or type(value["code"]) is str),
        "error_fields",
    )
    for frame in value["frames"]:
        require(
            type(frame) is dict
            and set(frame) == {"file", "function", "line"}
            and type(frame["file"]) is str
            and "/" not in frame["file"]
            and "\\" not in frame["file"]
            and type(frame["function"]) is str
            and type(frame["line"]) is int
            and frame["line"] > 0,
            "error_frame",
        )


def verify_evidence(
    plan,
    prepared,
    results,
    attempted,
    invoked,
    captures,
    *,
    attempts,
    invocation_receipts,
    run_sha256,
    expected_records=None,
    expected_analysis=None,
    expected_bindings=None,
):
    """Verify an entire saved matrix, including honest incomplete schedules.

    Inputs are already parsed private objects/bytes. This function performs no
    filesystem, tokenizer or model operations and never prints private data.
    Orphan capture bytes are allowed only for the last unresolved invoked
    baseline slot; they are validated but never enter eligibility or records.
    """
    verify_plan(plan, expected_bindings)
    verify_prepared(plan, prepared)
    require(is_hash(run_sha256), "run_hash")
    require(
        type(attempted) is set
        and type(invoked) is set
        and all(type(x) is str for x in attempted | invoked),
        "slot_sets",
    )
    for obj in (results, captures, attempts, invocation_receipts):
        require(type(obj) is dict, "evidence_mapping")
    order = [e["evaluation_id"] for e in plan["evaluations"]]
    require(
        attempted == set(order[: len(attempted)]) and set(attempts) == attempted, "attempt_prefix"
    )
    require(
        set(results) <= attempted and invoked <= attempted and set(invocation_receipts) == invoked,
        "slot_inventories",
    )
    unresolved = attempted - set(results)
    require(not unresolved or unresolved == {order[len(attempted) - 1]}, "unresolved_final_only")
    require(set(captures) <= attempted, "capture_inventory")
    accepted, valid_captures = {}, {}
    for e in plan["evaluations"][: len(attempted)]:
        key, p = e["evaluation_id"], prepared[e["context_id"]]
        source = source_binding(plan, e, accepted, valid_captures)
        expected = {
            "schema_version": "a182-attempt-v1",
            "run_sha256": run_sha256,
            "evaluation": e,
            "prepared_sha256": digest(p),
            "prompt_token_ids": p["prompt_token_ids"],
            "target_token_ids": p["candidate_token_ids"][e["candidate"]],
            "json_prefix_token_count": 3,
            "intended_eos_token_id": p["intended_eos_token_id"],
            "source": source,
        }
        require(canonical(attempts[key]) == canonical(expected), "attempt_reconstruction")
        if key in invoked:
            require(
                canonical(invocation_receipts[key])
                == canonical(
                    {
                        "schema_version": "a182-invocation-v1",
                        "attempt_sha256": digest(expected),
                        "model_forward_entered": True,
                    }
                ),
                "invocation_binding",
            )
            require(source is None or source["eligibility"] is True, "invoked_dependency")
        baseline = e["arm"] in ("baseline", "no_patch")
        if key not in results:
            if key in captures:
                raw = captures[key]
                require(
                    key in invoked and baseline and type(raw) is bytes and len(raw) == 16384,
                    "orphan_capture",
                )
                require(
                    all(math.isfinite(v) for v in struct.unpack("<4096f", raw)),
                    "orphan_capture_finite",
                )
            continue
        result = results[key]
        require(
            type(result) is dict
            and set(result)
            == {
                "schema_version",
                "attempt_sha256",
                "status",
                "measurement",
                "capture",
                "error",
                "elapsed_seconds",
            },
            "result_shape",
        )
        require(
            result["schema_version"] == "a182-result-v1"
            and result["attempt_sha256"] == digest(expected)
            and finite(result["elapsed_seconds"])
            and result["elapsed_seconds"] >= 0,
            "result_binding",
        )
        status = result["status"]
        if status == "completed":
            require(key in invoked and result["error"] is None, "completed_dispatch")
            verify_measurement(result["measurement"], len(expected["target_token_ids"]))
            if baseline:
                require(key in captures, "completed_capture")
                verify_capture(
                    result["capture"],
                    captures[key],
                    key,
                    digest(expected),
                    len(p["prompt_token_ids"]),
                )
                valid_captures[key] = captures[key]
            else:
                require(result["capture"] is None and key not in captures, "patched_capture_absent")
        else:
            require(
                result["measurement"] is None and result["capture"] is None and key not in captures,
                "failed_payload_absent",
            )
            if status == "dependency_unavailable":
                require(
                    key not in invoked
                    and result["error"] is None
                    and source is not None
                    and source["eligibility"] is not True,
                    "dependency_skip",
                )
            else:
                require(
                    status == "infrastructure_failed"
                    and (source is None or source["eligibility"] is True),
                    "failed_status",
                )
                _error_shape(result["error"])
        accepted[key] = result
    records = reconstruct_records(plan, prepared, accepted, valid_captures)
    analysis = reconstruct_analysis(plan, records, accepted, attempted, invoked)
    if expected_records is not None:
        require(canonical(records) == canonical(expected_records), "records_replay")
    if expected_analysis is not None:
        require(canonical(analysis) == canonical(expected_analysis), "analysis_replay")
    return {
        "records": records,
        "analysis": analysis,
        "verification": {
            "schema_version": "a182-matrix-verification-v1",
            "status": "pass",
            "independent_native_contexts": 16,
            "independent_candidate_paths": 48,
            "planned_forwards": 288,
            "independently_replayed_measurements": analysis["coverage"]["completed"],
            "verified_completed_captures": len(valid_captures),
            "orphan_captures_excluded": len(captures) - len(valid_captures),
            "independent_records_and_aggregate_replayed": True,
            "model_forward_reexecuted": False,
            "saved_normalizers_not_independently_recomputed": True,
            "tokenizer_audit_performed": False,
        },
    }


def file_sha(path):
    require(path.is_file() and not path.is_symlink(), "regular_file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, *, canonical_bytes=False):
    require(path.is_file() and not path.is_symlink(), "regular_json")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    raw = path.read_bytes()
    value = json.loads(
        raw, object_pairs_hook=unique, parse_constant=lambda _: require(False, "nonfinite_json")
    )
    require(not canonical_bytes or raw == canonical(value), "canonical_json_bytes")
    return value


def model_inventory(base, expected, *, support_only=False):
    """Allow pinned snapshot file symlinks, never directory/special entries.

    Hugging Face snapshots link their named files to content-addressed blobs.
    Frozen code and execution receipts intentionally use stricter inventories.
    Native-only audit hashes every support file, without reading weight bytes.
    """
    require(
        base.is_dir() and not base.is_symlink() and type(expected) is dict and bool(expected),
        "model_inventory_shape",
    )
    for name, sha in expected.items():
        require(
            type(name) is str
            and bool(name)
            and Path(name).name == name
            and name not in (".", "..")
            and "\\" not in name
            and is_hash(sha),
            "model_inventory_entry",
        )
    paths = list(base.iterdir())
    require({p.name for p in paths} == set(expected), "model_inventory_names")
    checked = {}
    for path in paths:
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "model_regular_target")
        target = path.resolve(strict=True)
        require(target.is_file() and not target.is_symlink(), "model_resolved_target")
        if not support_only or not path.name.endswith(".safetensors"):
            require(file_sha(target) == expected[path.name], "model_content_hash")
            checked[path.name] = target
    return checked


def runtime_versions():
    import importlib.metadata
    import platform

    packages = (
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "safetensors",
        "jinja2",
        "tokenizers",
        "numpy",
        "huggingface-hub",
    )
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def inventory(base, expected):
    require(type(expected) is dict and base.is_dir() and not base.is_symlink(), "inventory_shape")
    actual = {}
    for path in base.rglob("*"):
        require(not path.is_symlink(), "inventory_symlink")
        if path.is_file():
            actual[path.relative_to(base).as_posix()] = file_sha(path)
        else:
            require(path.is_dir(), "inventory_special")
    require(canonical(actual) == canonical(expected), "inventory_mismatch")


def _bound_paths(root, protocol):
    require(
        root.is_absolute()
        and root == Path(os.path.abspath(root))
        and all(not path.is_symlink() for path in (root, *root.parents)),
        "root_path",
    )
    plan = read_json(root / "inputs/plan.json")
    verify_plan(plan)
    cfg = read_json(root / "inputs/runtime-config.json")
    require(
        file_sha(root / "inputs/runtime-config.json") == FIXED_BINDINGS["config_sha256"],
        "config_file_binding",
    )
    require(file_sha(protocol) == plan["bindings"]["protocol_sha256"], "protocol_file_binding")
    source = root / "frozen/src/lexical_prompt_study"
    require(
        file_sha(source / "selector_transport.py") == plan["bindings"]["source_sha256"],
        "source_file_binding",
    )
    require(
        file_sha(root / "frozen/tests/test_selector_transport.py")
        == plan["bindings"]["tests_sha256"],
        "tests_file_binding",
    )
    for name, field in (
        ("conditional_path_qualification.py", "a179_helper_sha256"),
        ("instruction_selection_answer_path.py", "a169_helper_sha256"),
        ("prefix_residual_intervention.py", "apparatus_sha256"),
    ):
        require(file_sha(source / name) == FIXED_BINDINGS[field], "helper_file_binding")
    require(
        file_sha(root / "frozen/reference/run_cpu_reference_audit.py")
        == FIXED_BINDINGS["reference_script_sha256"],
        "reference_file_binding",
    )
    require(
        type(cfg["max_prompt_tokens"]) is int
        and cfg["max_prompt_tokens"] == 2048
        and type(cfg["max_new_tokens"]) is int
        and cfg["max_new_tokens"] == 64
        and type(cfg["cpu_threads"]) is int
        and cfg["cpu_threads"] == 4
        and cfg["attention_implementation"] == "sdpa",
        "configured_execution",
    )
    receipt = read_json(root / "inputs/native-prepared.private.json")
    sources = receipt["source_hashes"]
    require(type(sources) is dict and bool(sources), "native_source_inventory")
    for name, sha in sources.items():
        require(
            type(name) is str
            and Path(name).name == name
            and is_hash(sha)
            and file_sha(source / name) == sha,
            "native_source_binding",
        )
    require(
        sources.get("selector_transport.py") == plan["bindings"]["source_sha256"],
        "native_runtime_binding",
    )
    expected = {
        "schema_version": "a182-native-freeze-v1",
        "plan_sha256": file_sha(root / "inputs/plan.json"),
        "config_sha256": file_sha(root / "inputs/runtime-config.json"),
        "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "tests_sha256": plan["bindings"]["tests_sha256"],
        "source_hashes": sources,
        "reference_script_sha256": FIXED_BINDINGS["reference_script_sha256"],
        "planned_contexts": 16,
        "planned_native_candidate_paths": 48,
        "planned_forward_calls": 288,
        "eos_token_ids": receipt["eos_token_ids"],
        "native_prepared_sha256": digest(receipt["native_inputs"]),
        "native_inputs": receipt["native_inputs"],
    }
    require(canonical(receipt) == canonical(expected), "native_receipt_shape")
    verify_prepared(plan, receipt["native_inputs"])
    return plan, cfg, receipt


def construct_native(plan, config, tokenizer, eos_ids):
    """Independent 16-prompt/48-path token audit; injectable public fake in tests."""
    require(
        type(eos_ids) is list
        and eos_ids == sorted(set(eos_ids))
        and all(type(t) is int and t in (128001, 128008, 128009) for t in eos_ids),
        "eos_inventory",
    )
    require(
        type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id in eos_ids, "tokenizer_eos"
    )
    template = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    require(template == config["chat_template_sha256"], "tokenizer_template")
    special = tokenizer.all_special_ids
    require(type(special) in (list, tuple) and all(type(t) is int for t in special), "special_ids")
    prepared = {}
    for context in plan["contexts"]:
        messages = context["messages"]
        kw = {"add_generation_prompt": True, "date_string": "26 Jul 2024"}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kw)
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kw)
        require(
            type(prompt) is list and prompt == tokenizer.encode(rendered, add_special_tokens=False),
            "native_joint_prompt",
        )
        require(
            rendered.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
            and rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
            and rendered.count("<|begin_of_text|>") == 1
            and rendered.count("<|start_header_id|>") == 3
            and rendered.count("<|eot_id|>") == 2,
            "native_roles",
        )
        require(
            all(
                message["content"] in rendered
                and not set(special).intersection(
                    tokenizer.encode(message["content"], add_special_tokens=False)
                )
                for message in messages
            ),
            "native_payload_special",
        )
        paths = {}
        for name, response in context["canonical_candidates"].items():
            joined = tokenizer.encode(rendered + response, add_special_tokens=False)
            require(
                type(joined) is list
                and joined[: len(prompt)] == prompt
                and len(joined) > len(prompt),
                "native_continuation_prefix",
            )
            target = joined[len(prompt) :]
            require(
                tokenizer.decode(
                    target, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                == response
                and not set(target).intersection(eos_ids),
                "native_continuation_decode",
            )
            closed_messages = [*messages, {"role": "assistant", "content": response}]
            closed_kw = {"add_generation_prompt": False, "date_string": "26 Jul 2024"}
            closed = tokenizer.apply_chat_template(
                closed_messages, tokenize=True, return_dict=False, **closed_kw
            )
            closed_text = tokenizer.apply_chat_template(
                closed_messages, tokenize=False, **closed_kw
            )
            require(
                closed == joined + [tokenizer.eos_token_id]
                and closed == tokenizer.encode(closed_text, add_special_tokens=False),
                "native_closed_eos",
            )
            paths[name] = [*target, tokenizer.eos_token_id]
        prefix = next(iter(paths.values()))[:3]
        text = tokenizer.decode(
            prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        require(
            type(text) is str
            and bool(text)
            and '{"answer":"'.startswith(text)
            and tokenizer.encode(rendered + text, add_special_tokens=False) == prompt + prefix,
            "native_pure_prefix",
        )
        prepared[context["context_id"]] = {
            "rendered_text": rendered,
            "prompt_token_ids": prompt,
            "chat_template_sha256": template,
            "shared_json_prefix_token_ids": prefix,
            "shared_json_prefix_text": text,
            "candidate_token_ids": paths,
            "candidate_lengths": {k: len(v) for k, v in paths.items()},
            "intended_eos_token_id": tokenizer.eos_token_id,
            "model_context_limit": 131072,
            "model_vocab_size": 128256,
            "hidden_width": 4096,
            "model_layers": 32,
            "absolute_position": len(prompt) + 2,
        }
    verify_prepared(plan, prepared)
    return prepared


def native_audit(root, protocol, tokenizer_loader=None):
    """Called only for separately authorized native preparation/replay."""
    plan, cfg, receipt = _bound_paths(root, protocol)
    model = Path(cfg["model_path"])
    support = model_inventory(model, cfg["model_files_sha256"], support_only=True)
    require(canonical(runtime_versions()) == canonical(cfg["runtime_versions"]), "runtime_versions")
    metadata = read_json(support["config.json"])
    for key, value in (
        ("num_hidden_layers", 32),
        ("hidden_size", 4096),
        ("max_position_embeddings", 131072),
        ("vocab_size", 128256),
    ):
        require(type(metadata[key]) is int and metadata[key] == value, "model_metadata")
    eos = read_json(support["generation_config.json"])["eos_token_id"]
    eos = sorted(set(eos if type(eos) is list else [eos]))
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model, local_files_only=True, trust_remote_code=False
        )
    else:
        tokenizer = tokenizer_loader(cfg)
    reconstructed = construct_native(plan, cfg, tokenizer, eos)
    require(
        canonical(reconstructed) == canonical(receipt["native_inputs"])
        and canonical(eos) == canonical(receipt["eos_token_ids"]),
        "native_independent_replay",
    )
    lengths = [len(t) for p in reconstructed.values() for t in p["candidate_token_ids"].values()]
    prompts = [len(p["prompt_token_ids"]) for p in reconstructed.values()]
    return {
        "schema_version": "a182-independent-native-audit-v1",
        "status": "pass",
        "target_model_calls": 0,
        "source_sha256": plan["bindings"]["source_sha256"],
        "protocol_sha256": file_sha(protocol),
        "plan_sha256": file_sha(root / "inputs/plan.json"),
        "native_prepared_file_sha256": file_sha(root / "inputs/native-prepared.private.json"),
        "native_prepared_sha256": receipt["native_prepared_sha256"],
        "independent_native_contexts": 16,
        "independent_candidate_paths": 48,
        "planned_forwards": 288,
        "planned_scored_contexts": 40,
        "planned_candidate_paths": 144,
        "recipient_value_pairs": 2,
        "donor_value_pairs": 2,
        "prefix_tokens": 3,
        "layer_index": 15,
        "hidden_width": 4096,
        "all_context_prefix_ids_identical": True,
        "equal_candidate_lengths_all_contexts": True,
        "payload_special_token_count": 0,
        "prompt_token_range": [min(prompts), max(prompts)],
        "candidate_target_length_range": [min(lengths), max(lengths)],
    }


# Process-receipt lineage adapted from the frozen independent A181 verifier.
# Assertions in this legacy shape checker are protected by the unconditional
# module-level optimization guard; no runtime aggregation is used.
def _exact(left, right):
    require(canonical(left) == canonical(right), "exact_structure")


def owned_alive(receipt):
    assert type(receipt["pid"]) is int and receipt["pid"] > 0
    assert type(receipt["start_ticks"]) is int and receipt["start_ticks"] >= 0
    p = Path("/proc") / str(receipt["pid"]) / "stat"
    if not p.exists():
        return False
    try:
        stat = p.read_text().rsplit(")", 1)[1].split()
        return int(stat[19]) == receipt["start_ticks"] and stat[0] != "Z"
    except FileNotFoundError:
        return False


def interruption_shape(interruption, pid, status):
    if interruption is None:
        assert status in ("failed", "finished_schedule")
        return
    assert type(pid) is int and pid > 0
    assert type(interruption) is dict and set(interruption) == {
        "signal_number",
        "signal_name",
        "pid",
        "exception_type",
    }
    assert type(interruption["pid"]) is int and interruption["pid"] == pid
    number = interruption["signal_number"]
    kind = interruption["exception_type"]
    assert number is None or type(number) is int
    assert type(kind) is str and kind in ("A182Interrupted", "A182Deadline", "KeyboardInterrupt")
    allowed = {
        "A182Interrupted": (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
        "A182Deadline": (signal.SIGALRM,),
        "KeyboardInterrupt": (None,),
    }
    assert number in allowed[kind]
    assert interruption["signal_name"] == (
        signal.Signals(number).name if number is not None else None
    )
    assert (
        status
        == {
            "A182Deadline": "deadline",
            "A182Interrupted": "interrupted",
            "KeyboardInterrupt": "interrupted",
        }[kind]
    )


def finished_shape(finished, header):
    assert type(header["pid"]) is int and header["pid"] > 0
    assert type(finished) is dict and set(finished) == {
        "schema_version",
        "status",
        "elapsed_seconds",
        "run_sha256",
        "interruption",
    }
    assert finished["schema_version"] == "a182-execution-finished-v1" and finished[
        "run_sha256"
    ] == digest(header)
    assert (
        type(finished["elapsed_seconds"]) in (int, float)
        and math.isfinite(finished["elapsed_seconds"])
        and finished["elapsed_seconds"] >= 0
    )
    assert finished["status"] in ("finished_schedule", "interrupted", "deadline", "failed")
    interruption_shape(finished["interruption"], header["pid"], finished["status"])


def startup_shape(attempt, failure):
    assert type(attempt) is dict and set(attempt) == {
        "schema_version",
        "pid",
        "plan_sha256",
        "config_sha256",
        "source_sha256",
    }
    assert type(attempt["pid"]) is int and attempt["pid"] > 0
    assert type(failure) is dict and set(failure) == {
        "schema_version",
        "startup_attempt_sha256",
        "status",
        "elapsed_seconds",
        "interruption",
        "target_model_calls",
        "error",
    }
    assert (
        type(failure["elapsed_seconds"]) in (int, float)
        and math.isfinite(failure["elapsed_seconds"])
        and failure["elapsed_seconds"] >= 0
    )
    assert type(failure["target_model_calls"]) is int and failure["target_model_calls"] == 0
    error = failure["error"]
    if error is not None:
        assert (
            type(error) is dict
            and set(error) == {"code", "exception_type", "frames"}
            and error["code"] == "a182_startup_failed"
        )
        assert (
            type(error["exception_type"]) is str
            and bool(error["exception_type"])
            and type(error["frames"]) is list
        )
        for frame in error["frames"]:
            assert type(frame) is dict and set(frame) == {"file", "function", "line"}
            assert type(frame["file"]) is str and Path(frame["file"]).name == frame["file"]
            assert (
                type(frame["function"]) is str and type(frame["line"]) is int and frame["line"] > 0
            )
    assert attempt["schema_version"] == "a182-startup-attempt-v1"
    assert failure["schema_version"] == "a182-startup-failure-v1" and failure[
        "startup_attempt_sha256"
    ] == digest(attempt)
    assert failure["status"] in ("failed", "interrupted", "deadline")
    interruption_shape(failure["interruption"], attempt["pid"], failure["status"])


def command_for(root, freeze):
    i, f = root / "inputs", root / "frozen"
    return [
        "flock",
        "-n",
        str(root / ".cpu-study.lock"),
        "timeout",
        "--foreground",
        "--signal=TERM",
        "--kill-after=60s",
        "7200s",
        "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python",
        "-m",
        "lexical_prompt_study.selector_transport",
        "--mode",
        "run",
        "--plan",
        str(i / "plan.json"),
        "--plan-sha256",
        freeze["input_sha256"]["plan.json"],
        "--config",
        str(i / "runtime-config.json"),
        "--config-sha256",
        freeze["input_sha256"]["runtime-config.json"],
        "--protocol",
        str(f / "docs/a182-selector-transport-protocol.md"),
        "--reference-script",
        str(f / "reference/run_cpu_reference_audit.py"),
        "--output-root",
        str(root / "execution"),
    ]


def operational_shapes(root, freeze, freeze_sha, receipts, hashes):
    start = receipts["queue-start-one-shot.json"]
    queue = receipts["resource-queue.json"]
    queue_claim = receipts["queue-one-shot.json"]
    launch = receipts["launch-one-shot.json"]
    job = receipts["job.json"]
    command = command_for(root, freeze)
    _exact(freeze["command"], command)
    queue_command = [
        "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python",
        "-B",
        str(root / "inputs/wait_for_memory_and_launch.py"),
    ]
    for obj, pid, ticks in (
        (start, "starter_pid", "starter_start_ticks"),
        (queue, "pid", "start_ticks"),
        (queue_claim, "pid", "start_ticks"),
        (launch, "queue_pid", "queue_start_ticks"),
        (job, "pid", "start_ticks"),
    ):
        assert (
            type(obj) is dict
            and type(obj[pid]) is int
            and obj[pid] > 0
            and type(obj[ticks]) is int
            and obj[ticks] >= 0
        )
        assert obj["freeze_sha256"] == freeze_sha
    _exact(
        start,
        dict(
            schema_version="a182-queue-start-one-shot-v1",
            created_at_utc=start["created_at_utc"],
            created_monotonic_ns=start["created_monotonic_ns"],
            starter_pid=start["starter_pid"],
            starter_start_ticks=start["starter_start_ticks"],
            freeze_sha256=freeze_sha,
            watcher_sha256=freeze["input_sha256"]["wait_for_memory_and_launch.py"],
            starter_sha256=freeze["input_sha256"]["start_resource_queue.py"],
            command=queue_command,
        ),
    )
    assert (
        type(start["created_at_utc"]) is str
        and type(start["created_monotonic_ns"]) is int
        and start["created_monotonic_ns"] > 0
    )
    _exact(
        queue,
        dict(
            schema_version="a182-resource-queue-v1",
            created_at_utc=start["created_at_utc"],
            created_monotonic_ns=start["created_monotonic_ns"],
            pid=queue["pid"],
            start_ticks=queue["start_ticks"],
            freeze_sha256=freeze_sha,
            watcher_sha256=start["watcher_sha256"],
            starter_sha256=start["starter_sha256"],
            command=queue_command,
            minimum_available_memory_bytes=48 * 1024**3,
            poll_seconds=60,
            required_consecutive_checks=2,
            expiry_seconds=86400,
            start_claim_sha256=hashes["queue-start-one-shot.json"],
        ),
    )
    _exact(
        queue_claim,
        dict(
            schema_version="a182-queue-one-shot-v1",
            pid=queue["pid"],
            start_ticks=queue["start_ticks"],
            started_at_utc=queue_claim["started_at_utc"],
            freeze_sha256=freeze_sha,
            resource_queue_sha256=hashes["resource-queue.json"],
        ),
    )
    _exact(
        launch,
        dict(
            schema_version="a182-launch-one-shot-v1",
            claimed_at_utc=launch["claimed_at_utc"],
            queue_pid=queue["pid"],
            queue_start_ticks=queue["start_ticks"],
            freeze_sha256=freeze_sha,
            resource_queue_sha256=hashes["resource-queue.json"],
            command=command,
            available_memory_bytes=launch["available_memory_bytes"],
            first_eligible_monotonic=launch["first_eligible_monotonic"],
            qualifying_check_monotonic=launch["qualifying_check_monotonic"],
            external_deadline_seconds=7200,
            kill_grace_seconds=60,
        ),
    )
    assert (
        type(launch["available_memory_bytes"]) is int
        and launch["available_memory_bytes"] >= 48 * 1024**3
    )
    for key in ("first_eligible_monotonic", "qualifying_check_monotonic"):
        assert type(launch[key]) in (int, float) and math.isfinite(launch[key]) and launch[key] > 0
    assert launch["qualifying_check_monotonic"] - launch["first_eligible_monotonic"] >= 60
    _exact(
        job,
        dict(
            schema_version="a182-owned-job-v1",
            launched_at_utc=job["launched_at_utc"],
            pid=job["pid"],
            start_ticks=job["start_ticks"],
            freeze_sha256=freeze_sha,
            review_sha256=freeze["review_sha256"],
            native_audit_sha256=freeze["native_audit_sha256"],
            command=command,
            available_memory_bytes_at_launch=launch["available_memory_bytes"],
            external_deadline_seconds=7200,
            kill_grace_seconds=60,
            resource_queue_sha256=hashes["resource-queue.json"],
            launch_claim_sha256=hashes["launch-one-shot.json"],
        ),
    )
    for obj, key in (
        (queue_claim, "started_at_utc"),
        (launch, "claimed_at_utc"),
        (job, "launched_at_utc"),
    ):
        assert type(obj[key]) is str and bool(obj[key])


def _fixed_header(root, plan, config, native_receipt):
    return {
        "schema_version": "a182-run-v1",
        "plan_sha256": file_sha(root / "inputs/plan.json"),
        "config_sha256": file_sha(root / "inputs/runtime-config.json"),
        "source_hashes": native_receipt["source_hashes"],
        "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "tests_sha256": plan["bindings"]["tests_sha256"],
        "reference_script_sha256": FIXED_BINDINGS["reference_script_sha256"],
        "reference_script_path": str(root / "frozen/reference/run_cpu_reference_audit.py"),
        "native_prepared_sha256": native_receipt["native_prepared_sha256"],
        "model_manifest_sha256": digest(config["model_files_sha256"]),
        "eos_token_ids": native_receipt["eos_token_ids"],
        "settings": {
            "device": "cpu",
            "dtype": "float32",
            "quantization": None,
            "attention_implementation": "sdpa",
            "cpu_threads": 4,
            "seed": config["seed"],
            "use_cache": False,
            "logits_to_keep": "target_count_plus_one",
            "padding": False,
        },
        "planned_contexts": 16,
        "planned_native_candidate_paths": 48,
        "planned_scored_contexts": 40,
        "layer_index": 15,
        "hidden_width": 4096,
        "planned_forward_calls": 288,
        "maximum_seconds": 7200,
        "prefix_tokens": 3,
        "numerical_tolerance_nats": TOLERANCE,
        "functional_separation_nats": SEPARATION,
        "suffix_includes_eos": True,
        "normalization_arithmetic": "float64_logsumexp_of_float32_logits",
        "kill_grace_seconds": 60,
        "retries": 0,
        "resume_allowed": False,
        "heldout_allowed": False,
    }


def write_exclusive(path, value):
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def freeze_shape(root, freeze, review, decision, qualification):
    require(freeze.get("schema_version") == "a182-prospective-freeze-v1", "freeze_schema")
    commit = freeze.get("source_commit")
    require(
        type(commit) is str
        and len(commit) == 40
        and all(c in "0123456789abcdef" for c in commit)
        and freeze.get("remote_commit_verified") == commit,
        "freeze_commit",
    )
    fixed = {
        "planned_worlds": 4,
        "planned_recipient_value_pairs": 2,
        "planned_donor_value_pairs": 2,
        "planned_contexts": 16,
        "planned_native_paths": 48,
        "planned_arm_contexts": 40,
        "planned_candidate_paths": 144,
        "planned_forwards": 288,
        "external_deadline_seconds": 7200,
        "kill_grace_seconds": 60,
        "minimum_available_memory_gib": 48,
        "retries": 0,
        "model_calls_before_freeze": 0,
    }
    require(
        all(type(freeze.get(key)) is int and freeze[key] == value for key, value in fixed.items())
        and freeze.get("heldout_calls_authorized") is False,
        "freeze_fixed_fields",
    )
    inputs = {
        "QUESTION.json",
        "plan.json",
        "runtime-config.json",
        "native-prepared.private.json",
        "freeze_when_pushed.py",
        "wait_for_memory_and_launch.py",
        "start_resource_queue.py",
    }
    require(set(freeze["input_sha256"]) == inputs, "freeze_seven_inputs")
    required = {
        "src/lexical_prompt_study/selector_transport.py",
        "docs/a182-selector-transport-protocol.md",
        "reference/run_cpu_reference_audit.py",
        "tests/test_selector_transport.py",
        "src/lexical_prompt_study/selector_transport_verifier.py",
        "verification/prepare_native_inputs.py",
        "src/lexical_prompt_study/prefix_residual_intervention.py",
        "tests/test_prefix_residual_intervention.py",
        "tests/test_selector_transport_verifier.py",
    }
    require(required <= set(freeze["source_files_sha256"]), "freeze_required_sources")
    _exact(
        freeze["environment"],
        {
            "PYTHONPATH": str(root / "frozen/src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONOPTIMIZE": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        },
    )
    _exact(freeze["command"], command_for(root, freeze))
    require(
        freeze["launcher_sha256"] == freeze["input_sha256"]["freeze_when_pushed.py"],
        "freeze_launcher",
    )
    require(
        review["status"] == qualification["status"] == "pass"
        and review["source_commit"] == qualification["source_commit"] == commit
        and review["root_full_code_protocol_review"]
        == review["independent_code_protocol_review"]
        == "no_blocker"
        and review["ruff"] == freeze["ruff"] == "pass"
        and review["independent_verifier_qualification"] == "pass",
        "review_status",
    )
    total = freeze["tests_passed"]
    require(
        type(total) is int
        and total > 58
        and all(
            type(obj[key]) is int and obj[key] == total
            for obj, key in (
                (review, "synthetic_tests_passed"),
                (review, "root_frozen_tests_passed"),
                (qualification, "tests_passed"),
            )
        ),
        "test_qualification",
    )
    require(
        all(
            type(obj[key]) is int and obj[key] == 0
            for obj, key in (
                (review, "model_calls_before_freeze"),
                (qualification, "target_model_calls"),
                (decision, "target_model_calls_before_decision"),
            )
        ),
        "pre_execution_zero_calls",
    )
    require(
        review["native_audit_sha256"] == freeze["native_audit_sha256"]
        and review["synthetic_qualification_sha256"] == freeze["synthetic_qualification_sha256"]
        and decision["status"] == "selected_for_one_shot_execution"
        and decision["source_commit"] == commit
        and decision["review_sha256"] == freeze["review_sha256"],
        "decision_review_binding",
    )
    approved = {
        "source_sha256": "src/lexical_prompt_study/selector_transport.py",
        "tests_sha256": "tests/test_selector_transport.py",
        "verifier_sha256": "src/lexical_prompt_study/selector_transport_verifier.py",
        "verifier_tests_sha256": "tests/test_selector_transport_verifier.py",
        "apparatus_sha256": "src/lexical_prompt_study/prefix_residual_intervention.py",
        "apparatus_tests_sha256": "tests/test_prefix_residual_intervention.py",
        "protocol_sha256": "docs/a182-selector-transport-protocol.md",
    }
    require(
        all(
            review.get(key) == freeze["source_files_sha256"][path] for key, path in approved.items()
        ),
        "review_approved_public_bytes",
    )


def _freeze_audit(root, freeze_sha):
    require(
        root.is_absolute()
        and root == Path(os.path.abspath(root))
        and all(not path.is_symlink() for path in (root, *root.parents)),
        "root_path",
    )
    require(is_hash(freeze_sha) and file_sha(root / "FREEZE.json") == freeze_sha, "freeze_binding")
    freeze = read_json(root / "FREEZE.json")
    inventory(root / "frozen", freeze["source_files_sha256"])
    inventory(root / "inputs", freeze["input_sha256"])
    for name, key in (
        ("REVIEW.json", "review_sha256"),
        ("NATIVE-AUDIT.json", "native_audit_sha256"),
        ("EXECUTION-DECISION.json", "execution_decision_sha256"),
        ("SYNTHETIC-QUALIFICATION.json", "synthetic_qualification_sha256"),
    ):
        require(file_sha(root / name) == freeze[key], "external_receipt_binding")
    freeze_shape(
        root,
        freeze,
        read_json(root / "REVIEW.json"),
        read_json(root / "EXECUTION-DECISION.json"),
        read_json(root / "SYNTHETIC-QUALIFICATION.json"),
    )
    require(
        file_sha(Path(__file__))
        == freeze["source_files_sha256"]["src/lexical_prompt_study/selector_transport_verifier.py"],
        "executing_verifier_binding",
    )
    names = (
        "queue-start-one-shot.json",
        "resource-queue.json",
        "queue-one-shot.json",
        "launch-one-shot.json",
        "job.json",
    )
    receipts = {name: read_json(root / name) for name in names}
    hashes = {name: file_sha(root / name) for name in names}
    operational_shapes(root, freeze, freeze_sha, receipts, hashes)
    require(
        not owned_alive(receipts["job.json"]) and not owned_alive(receipts["resource-queue.json"]),
        "owned_process_alive",
    )
    return freeze


def _startup_audit(root, plan, config, receipt, freeze):
    execution = root / "execution"
    attempt, failure = (
        read_json(execution / "startup-attempt.json", canonical_bytes=True),
        read_json(execution / "startup-failure.json", canonical_bytes=True),
    )
    startup_shape(attempt, failure)
    require(not Path("/proc", str(attempt["pid"])).exists(), "startup_owner_present")
    expected = {
        "schema_version": "a182-startup-attempt-v1",
        "pid": attempt["pid"],
        "plan_sha256": file_sha(root / "inputs/plan.json"),
        "config_sha256": file_sha(root / "inputs/runtime-config.json"),
        "source_sha256": plan["bindings"]["source_sha256"],
    }
    _exact(attempt, expected)
    require(
        not (execution / "evaluations").exists()
        and not (execution / "execution-finished.json").exists(),
        "startup_zero_slots",
    )
    require(
        not any(
            (execution / name).exists() or (execution / name).is_symlink()
            for name in ("records.json", "analysis.json", "error.json")
        ),
        "startup_no_post_header_exports",
    )
    checked = verify_evidence(
        plan,
        receipt["native_inputs"],
        {},
        set(),
        set(),
        {},
        attempts={},
        invocation_receipts={},
        run_sha256="0" * 64,
    )
    return {
        "schema_version": "a182-independent-startup-verification-v1",
        "status": "pass",
        "scope": "startup_failure_zero_target_calls",
        "freeze_sha256": file_sha(root / "FREEZE.json"),
        "verifier_sha256": file_sha(Path(__file__)),
        "planned_contexts": 16,
        "planned_forwards": 288,
        "target_model_calls": 0,
        "model_forward_reexecuted": False,
        "actual_model_files_rehashed": False,
        "coverage": checked["analysis"]["coverage"],
        "qualified_transport": None,
        "terminal_status": failure["status"],
        "interruption": failure["interruption"],
        "startup_attempt_sha256": file_sha(execution / "startup-attempt.json"),
        "startup_failure_sha256": file_sha(execution / "startup-failure.json"),
        "partial_pre_header_files_preserved": True,
    }


def terminal(root, protocol, freeze_sha256, tokenizer_loader=None):
    """One exclusive terminal receipt; no missing forward is ever completed."""
    import datetime
    import fcntl

    require(not (root / "VERIFICATION.json").exists(), "verification_consumed")
    freeze = _freeze_audit(root, freeze_sha256)
    plan, config, native_receipt = _bound_paths(root, protocol)
    require(
        config["model_files_sha256"] == freeze["model_files_sha256"]
        and config["chat_template_sha256"] == freeze["chat_template_sha256"],
        "model_manifest_binding",
    )
    execution = root / "execution"
    require(
        (execution / ".lock").is_file() and not (execution / ".lock").is_symlink(), "execution_lock"
    )
    with (execution / ".lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if (execution / "startup-failure.json").exists():
            out = _startup_audit(root, plan, config, native_receipt, freeze)
            write_exclusive(root / "VERIFICATION.json", out)
            return out
        model_inventory(Path(config["model_path"]), config["model_files_sha256"])
        header = read_json(execution / "run.json", canonical_bytes=True)
        require(
            type(header.get("pid")) is int
            and header["pid"] > 0
            and not Path("/proc", str(header["pid"])).exists(),
            "target_owner_present",
        )
        require(
            type(header.get("available_memory_bytes")) is int
            and header["available_memory_bytes"] >= 48 * 1024**3,
            "memory_receipt",
        )
        _exact(
            header,
            {
                **_fixed_header(root, plan, config, native_receipt),
                "pid": header["pid"],
                "available_memory_bytes": header["available_memory_bytes"],
            },
        )
        _exact(
            read_json(execution / "one-shot.json", canonical_bytes=True),
            {
                "schema_version": "a182-one-shot-v1",
                "pid": header["pid"],
                "plan_sha256": header["plan_sha256"],
            },
        )
        _exact(
            read_json(execution / "startup-attempt.json", canonical_bytes=True),
            {
                "schema_version": "a182-startup-attempt-v1",
                "pid": header["pid"],
                "plan_sha256": header["plan_sha256"],
                "config_sha256": header["config_sha256"],
                "source_sha256": plan["bindings"]["source_sha256"],
            },
        )
        prepared = read_json(execution / "native-inputs.json", canonical_bytes=True)
        _exact(prepared, native_receipt["native_inputs"])
        audit = native_audit(root, protocol, tokenizer_loader)
        old_audit = read_json(root / "NATIVE-AUDIT.json")
        require(
            all(canonical(old_audit.get(k)) == canonical(value) for k, value in audit.items()),
            "native_audit_replay",
        )
        evaluations = {e["evaluation_id"]: e for e in plan["evaluations"]}
        folders = execution / "evaluations"
        results, attempts, invocations, captures = {}, {}, {}, {}
        file_hashes = {}
        if folders.exists():
            require(folders.is_dir() and not folders.is_symlink(), "evaluation_directory")
            for folder in folders.iterdir():
                require(
                    folder.name in evaluations and folder.is_dir() and not folder.is_symlink(),
                    "unknown_evaluation_directory",
                )
                files = {p.name for p in folder.iterdir()}
                require(
                    files <= {"attempt.json", "result.json", "capture.fp32", "invocation.json"}
                    and "attempt.json" in files
                    and all(p.is_file() and not p.is_symlink() for p in folder.iterdir()),
                    "evaluation_files",
                )
                key = folder.name
                attempts[key] = read_json(folder / "attempt.json", canonical_bytes=True)
                if "result.json" in files:
                    results[key] = read_json(folder / "result.json", canonical_bytes=True)
                    file_hashes[key] = file_sha(folder / "result.json")
                if "invocation.json" in files:
                    invocations[key] = read_json(folder / "invocation.json", canonical_bytes=True)
                if "capture.fp32" in files:
                    captures[key] = (folder / "capture.fp32").read_bytes()
        checked = verify_evidence(
            plan,
            prepared,
            results,
            set(attempts),
            set(invocations),
            captures,
            attempts=attempts,
            invocation_receipts=invocations,
            run_sha256=digest(header),
        )
        records, analysis = checked["records"], checked["analysis"]
        finished = (
            read_json(execution / "execution-finished.json", canonical_bytes=True)
            if (execution / "execution-finished.json").exists()
            else None
        )
        if finished is not None:
            finished_shape(finished, header)
            require(
                finished["status"] != "finished_schedule" or len(results) == 288,
                "finished_coverage",
            )
        analysis["execution_status"] = (
            finished["status"] if finished is not None else "receipt_missing"
        )
        analysis["provenance"] = {
            key: header[key]
            for key in (
                "plan_sha256",
                "config_sha256",
                "source_hashes",
                "tests_sha256",
                "protocol_sha256",
                "native_prepared_sha256",
                "model_manifest_sha256",
            )
        }
        analysis["provenance"]["run_sha256"] = digest(header)
        for name, value in (("records.json", records), ("analysis.json", analysis)):
            path = execution / name
            if path.exists():
                _exact(read_json(path, canonical_bytes=True), value)
            else:
                write_exclusive(path, value)
        out = {
            **checked["verification"],
            "schema_version": "a182-independent-verification-v1",
            "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "freeze_sha256": freeze_sha256,
            "verifier_sha256": file_sha(Path(__file__)),
            "source_files": len(freeze["source_files_sha256"]),
            "input_files": len(freeze["input_sha256"]),
            "model_files": len(config["model_files_sha256"]),
            "tokenizer_audit_performed": True,
            "coverage": analysis["coverage"],
            "primary": analysis["primary"],
            "apparatus": analysis["apparatus"],
            "functional_gates": analysis["functional_gates"],
            "qualified_transport": analysis["qualified_transport"],
            "terminal_status": analysis["execution_status"],
            "elapsed_seconds": finished["elapsed_seconds"] if finished else None,
            "interruption": finished["interruption"] if finished else None,
            "analysis_sha256": file_sha(execution / "analysis.json"),
            "records_sha256": file_sha(execution / "records.json"),
            "run_sha256": file_sha(execution / "run.json"),
            "execution_finished_sha256": file_sha(execution / "execution-finished.json")
            if finished
            else None,
            "result_sha256": file_hashes,
        }
        write_exclusive(root / "VERIFICATION.json", out)
        return out


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "terminal"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.mode == "native":
            require(not (args.root / "NATIVE-AUDIT.json").exists(), "native_audit_consumed")
            out = native_audit(args.root, args.protocol)
            out["verifier_sha256"] = file_sha(Path(__file__))
            write_exclusive(args.root / "NATIVE-AUDIT.json", out)
        else:
            out = terminal(args.root, args.protocol, args.freeze_sha256)
        print(
            json.dumps(
                {key: value for key, value in out.items() if key != "result_sha256"},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "verification_failed", "exception_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
