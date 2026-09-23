"""Independent public A182 matrix, arithmetic, dependency and tamper fixtures.

No tokenizer, pretrained model, old run or restricted material is used.
"""

import copy
import hashlib
import itertools
import math
import json
import re
import struct
from pathlib import Path
from fractions import Fraction

import pytest

from lexical_prompt_study import selector_transport_verifier as v


def measurement(score=-2.0, prefix=-3.0):
    lp = [prefix / 3] * 3 + [-0.25, -0.25, score + 0.75, -0.25]
    p, s = math.fsum(lp[:3]), math.fsum(lp[3:])
    return {
        "chosen_logits": [0.0] * 7,
        "log_normalizers": [-x for x in lp],
        "token_logprobs": lp,
        "scores": {
            "json_prefix_logprob": p,
            "answer_continuation_logprob": s,
            "joint_logprob": math.fsum((p, s)),
            "eos_logprob": lp[-1],
            "json_prefix_tokens": 3,
            "answer_continuation_tokens": 4,
            "joint_tokens": 7,
            "eos_tokens": 1,
        },
    }


def bindings():
    return {
        **v.FIXED_BINDINGS,
        "source_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "tests_sha256": "c" * 64,
    }


def native(plan):
    output = {}
    for c in plan["contexts"]:
        prompt = [1000 + c["context_index"], 2000, 3000]
        paths = {
            name: [3, 4, 5, 10 + list(v.FOUR).index(name), 6, 7, 128009]
            for name in c["canonical_candidates"]
        }
        output[c["context_id"]] = {
            "rendered_text": "\n".join(m["content"] for m in c["messages"]),
            "prompt_token_ids": prompt,
            "chat_template_sha256": "d" * 64,
            "shared_json_prefix_token_ids": [3, 4, 5],
            "shared_json_prefix_text": '{"answer":"',
            "candidate_token_ids": paths,
            "candidate_lengths": dict.fromkeys(paths, 7),
            "intended_eos_token_id": 128009,
            "model_context_limit": 131072,
            "model_vocab_size": 128256,
            "hidden_width": 4096,
            "model_layers": 32,
            "absolute_position": 5,
        }
    return output


def fixture(
    *,
    limit=288,
    failure=None,
    unequal=None,
    diagnostic_failure=None,
    bad_repeat=None,
    baseline_functional_failure=False,
    interrupted=False,
    plan=None,
    prepared=None,
    run_hash="e" * 64,
):
    """Build evidence independently, with optional honest scheduled failures."""
    plan = v.expected_plan(bindings()) if plan is None else plan
    prepared = native(plan) if prepared is None else prepared
    results, captures, attempts, invocations = {}, {}, {}, {}
    contexts = {c["context_id"]: c for c in plan["contexts"]}
    for e in plan["evaluations"][:limit]:
        key, c = e["evaluation_id"], contexts[e["context_id"]]
        p = prepared[e["context_id"]]
        source = v.source_binding(plan, e, results, captures)
        attempt = {
            "schema_version": "a182-attempt-v1",
            "run_sha256": run_hash,
            "evaluation": e,
            "prepared_sha256": v.digest(p),
            "prompt_token_ids": p["prompt_token_ids"],
            "target_token_ids": p["candidate_token_ids"][e["candidate"]],
            "json_prefix_token_count": 3,
            "intended_eos_token_id": 128009,
            "source": source,
        }
        attempts[key] = attempt
        if source is not None and source["eligibility"] is not True:
            results[key] = {
                "schema_version": "a182-result-v1",
                "attempt_sha256": v.digest(attempt),
                "status": "dependency_unavailable",
                "measurement": None,
                "capture": None,
                "error": None,
                "elapsed_seconds": 0.0,
            }
            continue
        invocations[key] = {
            "schema_version": "a182-invocation-v1",
            "attempt_sha256": v.digest(attempt),
            "model_forward_entered": True,
        }
        if interrupted and e["sequence_index"] == limit - 1:
            if e["arm"] in ("baseline", "no_patch"):
                captures[key] = struct.pack("<4096f", *([float(c["context_index"])] * 4096))
            break
        failed = e["sequence_index"] == failure or (
            diagnostic_failure is not None
            and e["context_index"] == diagnostic_failure
            and e["arm"] == "match"
            and e["candidate"] == "D1"
            and e["repeat_index"] == 0
        )
        if failed:
            results[key] = {
                "schema_version": "a182-result-v1",
                "attempt_sha256": v.digest(attempt),
                "status": "infrastructure_failed",
                "measurement": None,
                "capture": None,
                "error": {"exception_type": "RuntimeError", "frames": [], "code": "synthetic"},
                "elapsed_seconds": 0.1,
            }
            continue
        expected = (
            c["selected_candidate"]
            if c["kind"] == "donor" or e["arm"] == "match"
            else c["other_candidate"]
        )
        if baseline_functional_failure and e["arm"] == "baseline":
            expected = c["other_candidate"]
        score = (
            -1.0
            if e["candidate"] == expected
            else -4.0
            if e["candidate"] in (v.DONOR if c["kind"] == "donor" else v.OWN)
            else -6.0
        )
        if bad_repeat == e["sequence_index"]:
            score -= 0.01
        result = {
            "schema_version": "a182-result-v1",
            "attempt_sha256": v.digest(attempt),
            "status": "completed",
            "measurement": measurement(score),
            "capture": None,
            "error": None,
            "elapsed_seconds": 0.1,
        }
        if e["arm"] in ("baseline", "no_patch"):
            raw = struct.pack(
                "<4096f",
                *([float(c["context_index"]) + int(e["sequence_index"] == unequal)] * 4096),
            )
            captures[key] = raw
            result["capture"] = {
                "schema_version": "a182-capture-v1",
                "evaluation_id": key,
                "attempt_sha256": v.digest(attempt),
                "layer_index": 15,
                "absolute_position": len(p["prompt_token_ids"]) + 2,
                "shape": [1, 4096],
                "dtype": "float32",
                "byte_order": "little",
                "bytes": 16384,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        results[key] = result
    return dict(
        plan=plan,
        prepared=prepared,
        results=results,
        attempted=set(attempts),
        invoked=set(invocations),
        captures=captures,
        attempts=attempts,
        invocation_receipts=invocations,
        run_sha256=run_hash,
    )


def replay(data):
    return v.verify_evidence(**data)


def test_independent_schedule_counts_orders_and_identity_oracles():
    plan = v.expected_plan(bindings())
    assert len(plan["contexts"]) == 16 and len(plan["evaluations"]) == 288
    assert len({e["evaluation_id"] for e in plan["evaluations"]}) == 288
    assert [e["sequence_index"] for e in plan["evaluations"]] == list(range(288))
    assert sum(len(c["canonical_candidates"]) for c in plan["contexts"]) == 48
    for world in range(4):
        donors = [c for c in plan["contexts"] if c["kind"] == "donor" and c["world_index"] == world]
        recipients = [
            c for c in plan["contexts"] if c["kind"] == "recipient" and c["world_index"] == world
        ]
        assert donors[0]["mapping"] == donors[1]["mapping"]
        assert donors[0]["assignment_swapped"] is not recipients[0]["assignment_swapped"]
        for c in donors + recipients:
            assert c["selected_answer"] == c["mapping"][c["selector"]]
            es = [e for e in plan["evaluations"] if e["context_id"] == c["context_id"]]
            candidates = list(c["canonical_candidates"])
            unique = candidates if world % 2 == 0 else candidates[::-1]
            size = 2 * len(candidates)
            assert all(
                [e["candidate"] for e in es[start : start + size]] == unique + unique[::-1]
                for start in range(0, len(es), size)
            )
            ref = next(
                e for e in es if e["evaluation_id"] == plan["source_references"][c["context_id"]]
            )
            assert (
                ref["candidate"] == ("D1" if c["kind"] == "donor" else "R1")
                and ref["repeat_index"] == 0
            )
    order_counts = []
    for c in plan["contexts"][8:]:
        arms = [e["arm"] for e in plan["evaluations"] if e["context_id"] == c["context_id"]]
        order_counts.append(arms.index("match") < arms.index("opposite"))
    assert sum(order_counts) == 4


@pytest.mark.parametrize(
    "values,expected",
    [([True, True], True), ([True, None], None), ([False, None], False), ([False, True], False)],
)
def test_tristate(values, expected):
    assert v.conjunction(values) is expected


@pytest.mark.parametrize("values", [[], [1], [0], ["true"], [float("nan")]])
def test_tristate_malformed(values):
    with pytest.raises(v.VerificationError):
        v.conjunction(values)


def test_complete_matrix_all_gates_and_exact_fraction_primary():
    data = fixture()
    out = replay(data)
    a = out["analysis"]
    assert a["coverage"]["completed"] == a["coverage"]["forward_dispatch_entries"] == 288
    assert a["primary"]["completed_measurements"] == 64 and a["primary"]["point"] == 6.0
    assert a["qualified_transport"] is True
    assert a["apparatus"]["qualified"] is True
    assert [g["planned"] for g in a["functional_gates"].values()] == [8, 8, 16]
    rational = Fraction()
    ctx = {c["context_id"]: c for c in data["plan"]["contexts"]}
    for e in data["plan"]["evaluations"]:
        if e["arm"] in ("match", "opposite") and e["candidate"] in ("R1", "R2"):
            sign = (1 if e["arm"] == "match" else -1) * (
                1 if e["candidate"] == ctx[e["context_id"]]["selected_candidate"] else -1
            )
            rational += (
                sign
                * Fraction(
                    data["results"][e["evaluation_id"]]["measurement"]["scores"][
                        "answer_continuation_logprob"
                    ]
                )
                / 16
            )
    assert float(rational) == a["primary"]["point"]
    replay({**data, "expected_records": out["records"], "expected_analysis": a})


def test_all_missing_has_no_known_failure_or_imputation():
    a = replay(fixture(limit=0))["analysis"]
    assert a["coverage"]["missing"] == 288 and a["primary"]["point"] is None
    assert a["primary"]["numerical_valid"] is None and a["primary"]["unbounded"] is True
    assert a["qualified_transport"] is None


def test_missing_donor_content_patched_score_leaves_primary_complete():
    a = replay(fixture(diagnostic_failure=8))["analysis"]
    assert a["coverage"]["completed"] == 287 and a["primary"]["completed_measurements"] == 64
    assert a["primary"]["point"] == 6.0 and a["primary"]["numerical_valid"] is True
    assert a["apparatus"]["qualified"] is None and a["qualified_transport"] is None


def test_bad_self_source_does_not_block_donor_arms():
    a = replay(fixture(failure=32))["analysis"]
    assert a["coverage"]["dependency_unavailable"] == 8
    assert a["coverage"]["forward_dispatch_entries"] == 280
    assert a["primary"]["point"] == 6.0 and a["qualified_transport"] is False


@pytest.mark.parametrize("change", ["failed", "unequal"])
def test_bad_donor_source_only_blocks_dependent_arms(change):
    a = replay(fixture(**({"failure": 0} if change == "failed" else {"unequal": 0})))["analysis"]
    assert a["coverage"]["dependency_unavailable"] == 16
    assert a["coverage"]["forward_dispatch_entries"] == 272
    assert a["primary"]["point"] is None and a["primary"]["resolved_pairs"] == 6
    assert a["apparatus"]["source_eligible"]["false"] == 1


def test_functional_failure_does_not_gate_any_intervention():
    a = replay(fixture(baseline_functional_failure=True))["analysis"]
    assert a["coverage"]["completed"] == 288 and a["coverage"]["dependency_unavailable"] == 0
    assert a["functional_gates"]["donor_competence"]["qualified"] is False
    assert a["primary"]["point"] == 6.0 and a["qualified_transport"] is False


def test_score_repeat_failure_does_not_make_source_ineligible():
    a = replay(fixture(bad_repeat=0))["analysis"]
    assert a["coverage"]["completed"] == 288 and a["apparatus"]["source_eligible"]["true"] == 16
    assert a["functional_gates"]["donor_competence"]["numerical_valid"]["false"] == 1
    assert a["primary"]["point"] == 6.0


def test_interrupted_capture_publication_is_preserved_but_ineligible():
    out = replay(fixture(limit=3, interrupted=True))
    assert out["verification"]["orphan_captures_excluded"] == 1
    assert out["analysis"]["coverage"]["interrupted"] == 1
    assert out["records"]["donors"][0]["source_eligible"] is None


def test_complete_prefix_rule_and_early_known_repeat_failure():
    matrix = {"R1": [measurement(-1), measurement(-1.01)], "R2": [None, None]}
    c = v.cohort(matrix, v.OWN)
    assert c["prefix_guard_passed"] is None and c["numerical_valid"] is False
    assert (
        v.prefix_control([measurement(prefix=-3), measurement(prefix=-5), None])["passed"] is None
    )


def test_partial_self_control_failure_dominates_unknowns():
    left = dict.fromkeys(v.FOUR, [None, None])
    right = copy.deepcopy(left)
    left["R1"], right["R1"] = [measurement(-1), None], [measurement(-2), None]
    assert v.self_control(left, right)["passed"] is False


def test_strong_conflict_uses_best_authoritative_not_worst():
    # A negative worst authoritative difference is insufficient: one repeat still favors it.
    matrix = {"R1": [measurement(-1), measurement(-4)], "R2": [measurement(-2), measurement(-2)]}
    m = v.margin(matrix, "R1", "R2", True)
    assert m["worst"] < 0 and m["best"] > 0
    assert v.dominance(matrix, "R2", ("R1",), True)["passed"] is False


@pytest.mark.parametrize(
    "separation,expected", [(0.0, False), (0.001, False), (0.0010000000001, True), (-1.0, False)]
)
def test_strict_functional_boundary(separation, expected):
    # Direct synthetic score dictionaries isolate the exact threshold, without rounding sums.
    def m(score):
        return {"scores": {"answer_continuation_logprob": score, "json_prefix_logprob": -3.0}}

    matrix = {"R1": [m(0.0), m(0.0)], "R2": [m(-separation), m(-separation)]}
    assert v.dominance(matrix, "R1", ("R2",), True)["passed"] is expected


def test_four_way_gate_checks_both_donor_values():
    matrix = {name: [measurement(-4), measurement(-4)] for name in v.FOUR}
    matrix["R1"] = [measurement(-1), measurement(-1)]
    assert v.dominance(matrix, "R1", ("R2", "D1", "D2"), True)["passed"] is True
    matrix["D2"] = [measurement(-0.9), measurement(-0.9)]
    assert v.dominance(matrix, "R1", ("R2", "D1", "D2"), True)["passed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_index", 14),
        ("absolute_position", 6),
        ("shape", [4096]),
        ("dtype", "float64"),
        ("byte_order", "big"),
        ("bytes", True),
        ("evaluation_id", "wrong"),
        ("attempt_sha256", "0" * 64),
        ("sha256", "0" * 64),
    ],
)
def test_capture_metadata_tamper(field, value):
    data = fixture(limit=1)
    key = next(iter(data["results"]))
    data["results"][key]["capture"][field] = value
    with pytest.raises(v.VerificationError):
        replay(data)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"x" * 16383,
        struct.pack("<4096f", *([float("nan")] * 4096)),
        struct.pack("<4096f", *([float("inf")] * 4096)),
    ],
)
def test_capture_payload_tamper(raw):
    data = fixture(limit=1)
    key = next(iter(data["results"]))
    data["captures"][key] = raw
    data["results"][key]["capture"]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(v.VerificationError):
        replay(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["results"][next(iter(d["results"]))]["measurement"][
            "token_logprobs"
        ].__setitem__(3, -0.7),
        lambda d: d["results"][next(iter(d["results"]))]["measurement"]["scores"].__setitem__(
            "eos_logprob", -0.7
        ),
        lambda d: d["results"][next(iter(d["results"]))].__setitem__("elapsed_seconds", True),
        lambda d: d["invocation_receipts"][next(iter(d["invocation_receipts"]))].__setitem__(
            "model_forward_entered", 1
        ),
        lambda d: d["plan"]["evaluations"][0].__setitem__("repeat_index", False),
        lambda d: d["plan"]["contexts"][0].__setitem__("selector", "B"),
        lambda d: d["prepared"][next(iter(d["prepared"]))][
            "shared_json_prefix_token_ids"
        ].__setitem__(0, True),
    ],
)
def test_typed_arithmetic_identity_and_invocation_tamper(mutate):
    data = fixture(limit=1)
    mutate(data)
    with pytest.raises(v.VerificationError):
        replay(data)


@pytest.mark.parametrize(
    "key", ["primary", "apparatus", "functional_gates", "coverage", "numeric_records_sha256"]
)
def test_saved_aggregate_tamper(key):
    data = fixture()
    actual = replay(data)
    altered = copy.deepcopy(actual["analysis"])
    altered[key] = None
    with pytest.raises(v.VerificationError, match="analysis_replay"):
        replay({**data, "expected_analysis": altered})


def test_every_subset_of_unknown_fixed_effects_is_unbounded():
    values = [Fraction(1, 2), Fraction(-3, 4), Fraction(5, 8), Fraction(7, 16)]
    for flags in itertools.product((False, True), repeat=4):
        supplied = [float(x) if present else None for x, present in zip(values, flags)]
        result = v.fixed_summary(supplied)
        assert result["unbounded"] is (not all(flags))
        assert result["point"] == (float(sum(values) / 4) if all(flags) else None)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"limit": 0},
        {"diagnostic_failure": 8},
        {"failure": 32},
        {"failure": 0},
        {"unequal": 0},
        {"bad_repeat": 0},
        {"baseline_functional_failure": True},
        {"limit": 3, "interrupted": True},
    ],
)
def test_runtime_comparison_on_full_public_fixtures(options):
    # Comparison only: the verifier never imports or calls this implementation.
    from lexical_prompt_study import selector_transport as runtime

    data = fixture(**options)
    checked = replay(data)
    accepted_captures = {
        key: raw for key, raw in data["captures"].items() if key in data["results"]
    }
    records = runtime._records(data["plan"], data["prepared"], data["results"], accepted_captures)
    aggregate = runtime._aggregate(
        data["plan"], records, data["results"], data["attempted"], data["invoked"]
    )
    assert v.canonical(records) == v.canonical(checked["records"])
    assert v.canonical(aggregate) == v.canonical(checked["analysis"])


def test_independent_constructor_matches_runtime_current_public_source():
    from lexical_prompt_study import selector_transport as runtime

    plan = runtime.compile_plan(v.FIXED_BINDINGS["config_sha256"], "b" * 64, "c" * 64)
    assert v.canonical(plan) == v.canonical(v.expected_plan(plan["bindings"]))


class PublicTokenizer:
    """Reversible synthetic pieces; no learned tokenizer or model involved."""

    eos_token_id = 128009
    specials = {
        "<|begin_of_text|>": 128000,
        "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007,
        "<|eot_id|>": 128009,
    }
    pieces = {'{"': 10, "answer": 11, '":"': 12}
    all_special_ids = list(specials.values())

    def get_chat_template(self):
        return "public synthetic template"

    def encode(self, text, **kwargs):
        out = []
        for piece in re.findall(r'<\|[^>]+\|>|\{"|answer|":"|s\d+|[\s\S]', text):
            if piece in self.specials:
                out.append(self.specials[piece])
            elif piece in self.pieces:
                out.append(self.pieces[piece])
            elif re.fullmatch(r"s\d+", piece):
                out.append(10000 + int(piece[1:]))
            else:
                out.append(1000 + ord(piece))
        return out

    def decode(self, tokens, **kwargs):
        pieces = {v: k for k, v in {**self.specials, **self.pieces}.items()}
        return "".join(
            pieces[t]
            if t in pieces
            else "s" + str(t - 10000)
            if 10000 <= t < 20000
            else chr(t - 1000)
            for t in tokens
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "<|begin_of_text|>" + "".join(
            "<|start_header_id|>"
            + m["role"]
            + "<|end_header_id|>\n\n"
            + m["content"]
            + "<|eot_id|>"
            for m in messages
        )
        if add_generation_prompt:
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return self.encode(text) if tokenize else text


def fake_config(model):
    return {
        "max_prompt_tokens": 2048,
        "max_new_tokens": 64,
        "cpu_threads": 4,
        "attention_implementation": "sdpa",
        "seed": 17,
        "model_path": str(model),
        "runtime_versions": {"python": "public fixture"},
        "chat_template_sha256": hashlib.sha256(
            PublicTokenizer().get_chat_template().encode()
        ).hexdigest(),
    }


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v.canonical(obj))


def package(tmp_path, monkeypatch, **options):
    """Self-contained public snapshot and evidence, never a real run root."""
    root = tmp_path / "synthetic-run"
    source = root / "frozen/src/lexical_prompt_study"
    source.mkdir(parents=True)
    pin_files = {
        "a179_helper_sha256": "conditional_path_qualification.py",
        "a169_helper_sha256": "instruction_selection_answer_path.py",
        "apparatus_sha256": "prefix_residual_intervention.py",
    }
    pins = dict(v.FIXED_BINDINGS)
    for key, name in pin_files.items():
        path = source / name
        path.write_text("# public synthetic dependency\n")
        pins[key] = v.file_sha(path)
    ref = root / "frozen/reference/run_cpu_reference_audit.py"
    ref.parent.mkdir(parents=True)
    ref.write_text("# public synthetic reference\n")
    pins["reference_script_sha256"] = v.file_sha(ref)
    src = source / "selector_transport.py"
    src.write_text("# public synthetic runtime bytes\n")
    (source / "selector_transport_verifier.py").write_bytes(Path(v.__file__).read_bytes())
    for name in (
        "tests/test_selector_transport.py",
        "tests/test_selector_transport_verifier.py",
        "tests/test_prefix_residual_intervention.py",
        "verification/prepare_native_inputs.py",
    ):
        path = root / "frozen" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public synthetic bound file\n")
    protocol = root / "frozen/docs/a182-selector-transport-protocol.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("Public synthetic protocol\n")
    model = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    model.mkdir()
    blobs.mkdir()
    objects = {
        "config.json": {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "max_position_embeddings": 131072,
            "vocab_size": 128256,
        },
        "generation_config.json": {"eos_token_id": [128001, 128008, 128009]},
        "tokenizer.json": {},
        "tokenizer_config.json": {},
        "special_tokens_map.json": {},
        "model.safetensors.index.json": {},
    }
    for name, obj in objects.items():
        put(blobs / name, obj)
    for name in (
        "chat_template.jinja",
        "part1.safetensors",
        "part2.safetensors",
        "part3.safetensors",
    ):
        (blobs / name).write_bytes(b"public synthetic support or weight placeholder\n")
    for path in blobs.iterdir():
        (model / path.name).symlink_to(path)
    cfg = fake_config(model)
    cfg["model_files_sha256"] = {p.name: v.file_sha(p) for p in blobs.iterdir()}
    inputs = root / "inputs"
    put(inputs / "runtime-config.json", cfg)
    pins["config_sha256"] = v.file_sha(inputs / "runtime-config.json")
    monkeypatch.setattr(v, "FIXED_BINDINGS", pins)
    monkeypatch.setattr(v, "runtime_versions", lambda: cfg["runtime_versions"])
    plan = v.expected_plan(
        {
            **pins,
            "source_sha256": v.file_sha(src),
            "protocol_sha256": v.file_sha(protocol),
            "tests_sha256": v.file_sha(root / "frozen/tests/test_selector_transport.py"),
        }
    )
    put(inputs / "plan.json", plan)
    tokenizer = PublicTokenizer()
    prepared = v.construct_native(plan, cfg, tokenizer, [128001, 128008, 128009])
    receipt = {
        "schema_version": "a182-native-freeze-v1",
        "plan_sha256": v.file_sha(inputs / "plan.json"),
        "config_sha256": pins["config_sha256"],
        "protocol_sha256": plan["bindings"]["protocol_sha256"],
        "tests_sha256": plan["bindings"]["tests_sha256"],
        "source_hashes": {p.name: v.file_sha(p) for p in source.iterdir()},
        "reference_script_sha256": pins["reference_script_sha256"],
        "planned_contexts": 16,
        "planned_native_candidate_paths": 48,
        "planned_forward_calls": 288,
        "eos_token_ids": [128001, 128008, 128009],
        "native_prepared_sha256": v.digest(prepared),
        "native_inputs": prepared,
    }
    put(inputs / "native-prepared.private.json", receipt)
    put(inputs / "QUESTION.json", {"public": "synthetic"})
    for name in (
        "freeze_when_pushed.py",
        "wait_for_memory_and_launch.py",
        "start_resource_queue.py",
    ):
        (inputs / name).write_text("# public synthetic operational helper\n")
    audit = v.native_audit(root, protocol, lambda _: tokenizer)
    put(root / "NATIVE-AUDIT.json", audit)
    commit = "a" * 40
    qualification = {
        "status": "pass",
        "source_commit": commit,
        "target_model_calls": 0,
        "tests_passed": 200,
    }
    put(root / "SYNTHETIC-QUALIFICATION.json", qualification)
    review = {
        "status": "pass",
        "source_commit": commit,
        "root_full_code_protocol_review": "no_blocker",
        "independent_code_protocol_review": "no_blocker",
        "ruff": "pass",
        "independent_verifier_qualification": "pass",
        "synthetic_tests_passed": 200,
        "root_frozen_tests_passed": 200,
        "model_calls_before_freeze": 0,
        "native_audit_sha256": v.file_sha(root / "NATIVE-AUDIT.json"),
        "synthetic_qualification_sha256": v.file_sha(root / "SYNTHETIC-QUALIFICATION.json"),
    }
    for key, path in {
        "source_sha256": "src/lexical_prompt_study/selector_transport.py",
        "tests_sha256": "tests/test_selector_transport.py",
        "verifier_sha256": "src/lexical_prompt_study/selector_transport_verifier.py",
        "verifier_tests_sha256": "tests/test_selector_transport_verifier.py",
        "apparatus_sha256": "src/lexical_prompt_study/prefix_residual_intervention.py",
        "apparatus_tests_sha256": "tests/test_prefix_residual_intervention.py",
        "protocol_sha256": "docs/a182-selector-transport-protocol.md",
    }.items():
        review[key] = v.file_sha(root / "frozen" / path)
    put(root / "REVIEW.json", review)
    decision = {
        "status": "selected_for_one_shot_execution",
        "source_commit": commit,
        "review_sha256": v.file_sha(root / "REVIEW.json"),
        "target_model_calls_before_decision": 0,
    }
    put(root / "EXECUTION-DECISION.json", decision)
    freeze = {
        "schema_version": "a182-prospective-freeze-v1",
        "source_commit": commit,
        "remote_commit_verified": commit,
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
        "heldout_calls_authorized": False,
        "tests_passed": 200,
        "ruff": "pass",
        "model_files_sha256": cfg["model_files_sha256"],
        "chat_template_sha256": cfg["chat_template_sha256"],
        "source_files_sha256": {
            p.relative_to(root / "frozen").as_posix(): v.file_sha(p)
            for p in (root / "frozen").rglob("*")
            if p.is_file()
        },
        "input_sha256": {p.name: v.file_sha(p) for p in inputs.iterdir()},
        "environment": {
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
    }
    freeze["command"] = v.command_for(root, freeze)
    freeze["launcher_sha256"] = freeze["input_sha256"]["freeze_when_pushed.py"]
    for name, key in (
        ("REVIEW.json", "review_sha256"),
        ("NATIVE-AUDIT.json", "native_audit_sha256"),
        ("EXECUTION-DECISION.json", "execution_decision_sha256"),
        ("SYNTHETIC-QUALIFICATION.json", "synthetic_qualification_sha256"),
    ):
        freeze[key] = v.file_sha(root / name)
    put(root / "FREEZE.json", freeze)
    freeze_sha = v.file_sha(root / "FREEZE.json")
    operational_receipts(root, freeze, freeze_sha)
    execution = root / "execution"
    execution.mkdir()
    (execution / ".lock").touch()
    pid = 2147483647
    assert not Path("/proc", str(pid)).exists()
    header = {
        **v._fixed_header(root, plan, cfg, receipt),
        "pid": pid,
        "available_memory_bytes": 48 * 1024**3,
    }
    put(execution / "run.json", header)
    put(
        execution / "one-shot.json",
        {"schema_version": "a182-one-shot-v1", "pid": pid, "plan_sha256": header["plan_sha256"]},
    )
    put(
        execution / "startup-attempt.json",
        {
            "schema_version": "a182-startup-attempt-v1",
            "pid": pid,
            "plan_sha256": header["plan_sha256"],
            "config_sha256": header["config_sha256"],
            "source_sha256": plan["bindings"]["source_sha256"],
        },
    )
    put(execution / "native-inputs.json", prepared)
    data = fixture(plan=plan, prepared=prepared, run_hash=v.digest(header), **options)
    for key, attempt in data["attempts"].items():
        folder = execution / "evaluations" / key
        put(folder / "attempt.json", attempt)
        for filename, mapping in (
            ("result.json", "results"),
            ("invocation.json", "invocation_receipts"),
        ):
            if key in data[mapping]:
                put(folder / filename, data[mapping][key])
        if key in data["captures"]:
            (folder / "capture.fp32").write_bytes(data["captures"][key])
    interrupted = options.get("interrupted", False)
    finished = {
        "schema_version": "a182-execution-finished-v1",
        "run_sha256": v.digest(header),
        "status": "interrupted" if interrupted else "finished_schedule",
        "elapsed_seconds": 1.5,
        "interruption": {
            "signal_number": 15,
            "signal_name": "SIGTERM",
            "pid": pid,
            "exception_type": "A182Interrupted",
        }
        if interrupted
        else None,
    }
    put(execution / "execution-finished.json", finished)
    return root, protocol, freeze_sha, cfg, data


def operational_receipts(root, freeze, freeze_sha):
    command = v.command_for(root, freeze)
    queue_command = [
        "/data2/PRAX/lexical-prompt-study-data/runtime/a163-local311/bin/python",
        "-B",
        str(root / "inputs/wait_for_memory_and_launch.py"),
    ]
    pid = 2147483647
    start = {
        "schema_version": "a182-queue-start-one-shot-v1",
        "created_at_utc": "public-fixture",
        "created_monotonic_ns": 1,
        "starter_pid": pid,
        "starter_start_ticks": 0,
        "freeze_sha256": freeze_sha,
        "watcher_sha256": freeze["input_sha256"]["wait_for_memory_and_launch.py"],
        "starter_sha256": freeze["input_sha256"]["start_resource_queue.py"],
        "command": queue_command,
    }
    put(root / "queue-start-one-shot.json", start)
    queue = {
        "schema_version": "a182-resource-queue-v1",
        "created_at_utc": start["created_at_utc"],
        "created_monotonic_ns": 1,
        "pid": pid,
        "start_ticks": 0,
        "freeze_sha256": freeze_sha,
        "watcher_sha256": start["watcher_sha256"],
        "starter_sha256": start["starter_sha256"],
        "command": queue_command,
        "minimum_available_memory_bytes": 48 * 1024**3,
        "poll_seconds": 60,
        "required_consecutive_checks": 2,
        "expiry_seconds": 86400,
        "start_claim_sha256": v.file_sha(root / "queue-start-one-shot.json"),
    }
    put(root / "resource-queue.json", queue)
    put(
        root / "queue-one-shot.json",
        {
            "schema_version": "a182-queue-one-shot-v1",
            "pid": pid,
            "start_ticks": 0,
            "started_at_utc": "public-fixture",
            "freeze_sha256": freeze_sha,
            "resource_queue_sha256": v.file_sha(root / "resource-queue.json"),
        },
    )
    launch = {
        "schema_version": "a182-launch-one-shot-v1",
        "claimed_at_utc": "public-fixture",
        "queue_pid": pid,
        "queue_start_ticks": 0,
        "freeze_sha256": freeze_sha,
        "resource_queue_sha256": v.file_sha(root / "resource-queue.json"),
        "command": command,
        "available_memory_bytes": 48 * 1024**3,
        "first_eligible_monotonic": 1.0,
        "qualifying_check_monotonic": 61.0,
        "external_deadline_seconds": 7200,
        "kill_grace_seconds": 60,
    }
    put(root / "launch-one-shot.json", launch)
    put(
        root / "job.json",
        {
            "schema_version": "a182-owned-job-v1",
            "launched_at_utc": "public-fixture",
            "pid": pid,
            "start_ticks": 0,
            "freeze_sha256": freeze_sha,
            "review_sha256": freeze["review_sha256"],
            "native_audit_sha256": freeze["native_audit_sha256"],
            "command": command,
            "available_memory_bytes_at_launch": 48 * 1024**3,
            "external_deadline_seconds": 7200,
            "kill_grace_seconds": 60,
            "resource_queue_sha256": v.file_sha(root / "resource-queue.json"),
            "launch_claim_sha256": v.file_sha(root / "launch-one-shot.json"),
        },
    )


def test_native_fake_tokenizer_exact_roundtrip_and_runtime_parity():
    from lexical_prompt_study import selector_transport as runtime

    plan, tokenizer = v.expected_plan(bindings()), PublicTokenizer()
    cfg = fake_config("/public/unopened")
    native = v.construct_native(plan, cfg, tokenizer, [128001, 128008, 128009])
    for context in plan["contexts"]:
        independent = native[context["context_id"]]
        emitted = runtime.prepare_context(tokenizer, context, cfg, [128001, 128008, 128009])
        assert all(
            v.canonical(independent[key]) == v.canonical(value) for key, value in emitted.items()
        )
    assert len(native) == 16
    assert sum(len(row["candidate_token_ids"]) for row in native.values()) == 48


@pytest.mark.parametrize(
    "options", [{}, {"failure": 0}, {"diagnostic_failure": 8}, {"limit": 3, "interrupted": True}]
)
def test_temporary_terminal_symlink_snapshot_and_no_forward(tmp_path, monkeypatch, options):
    root, protocol, freeze_sha, cfg, data = package(tmp_path, monkeypatch, **options)
    out = v.terminal(root, protocol, freeze_sha, lambda _: PublicTokenizer())
    assert out["status"] == "pass" and out["model_forward_reexecuted"] is False
    assert out["model_files"] == 10 and out["tokenizer_audit_performed"] is True
    assert out["coverage"]["completed"] == sum(
        r["status"] == "completed" for r in data["results"].values()
    )
    assert out["primary"]["point"] == replay(data)["analysis"]["primary"]["point"]
    with pytest.raises(v.VerificationError, match="verification_consumed"):
        v.terminal(root, protocol, freeze_sha, lambda _: PublicTokenizer())


def test_native_hashes_support_and_versions_before_loading(tmp_path, monkeypatch):
    root, protocol, _, cfg, _ = package(tmp_path, monkeypatch)
    entered = []

    def loader(_):
        return entered.append(True)

    (Path(cfg["model_path"]) / "tokenizer.json").resolve().write_bytes(b"changed")
    with pytest.raises(v.VerificationError, match="model_content_hash"):
        v.native_audit(root, protocol, loader)
    assert entered == []


def test_native_versions_bound_before_loading(tmp_path, monkeypatch):
    root, protocol, _, cfg, _ = package(tmp_path, monkeypatch)
    monkeypatch.setattr(v, "runtime_versions", lambda: {"python": "wrong"})
    with pytest.raises(v.VerificationError, match="runtime_versions"):
        v.native_audit(root, protocol, lambda _: pytest.fail("unqualified loader entered"))


@pytest.mark.parametrize(
    "change", ["extra", "missing", "directory", "dangling", "special", "digest"]
)
def test_model_manifest_rejects_unlisted_or_invalid_snapshot_entries(tmp_path, change):
    import os

    model, blob = tmp_path / "snapshot", tmp_path / "blob"
    model.mkdir()
    blob.write_bytes(b"public model placeholder")
    link = model / "named.json"
    link.symlink_to(blob)
    expected = {"named.json": v.file_sha(blob)}
    assert v.model_inventory(model, expected)["named.json"] == blob
    if change == "extra":
        (model / "unlisted").write_bytes(b"x")
    elif change == "missing":
        link.unlink()
    elif change in ("directory", "dangling", "special"):
        blob.unlink()
        if change == "directory":
            blob.mkdir()
        elif change == "special":
            os.mkfifo(blob)
    else:
        blob.write_bytes(b"drift")
    with pytest.raises(v.VerificationError):
        v.model_inventory(model, expected)


@pytest.mark.parametrize(
    "name",
    [
        "run.json",
        "startup-attempt.json",
        "one-shot.json",
        "native-inputs.json",
        "execution-finished.json",
    ],
)
def test_noncanonical_execution_bytes_rejected(tmp_path, monkeypatch, name):
    root, protocol, freeze_sha, _, _ = package(tmp_path, monkeypatch)
    path = root / "execution" / name
    path.write_text(json.dumps(v.read_json(path), indent=2) + "\n")
    with pytest.raises(v.VerificationError, match="canonical_json_bytes"):
        v.terminal(root, protocol, freeze_sha, lambda _: PublicTokenizer())
    assert not (root / "VERIFICATION.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("planned_forwards", 289),
        ("planned_worlds", True),
        ("external_deadline_seconds", 3600),
        ("heldout_calls_authorized", True),
        ("minimum_available_memory_gib", 47),
    ],
)
def test_fixed_freeze_tamper(tmp_path, monkeypatch, key, value):
    root, _, _, _, _ = package(tmp_path, monkeypatch)
    freeze = v.read_json(root / "FREEZE.json")
    freeze[key] = value
    with pytest.raises(v.VerificationError):
        v.freeze_shape(
            root,
            freeze,
            v.read_json(root / "REVIEW.json"),
            v.read_json(root / "EXECUTION-DECISION.json"),
            v.read_json(root / "SYNTHETIC-QUALIFICATION.json"),
        )


@pytest.mark.parametrize(
    "kind", ["environment", "decision", "review", "qualification", "approved_bytes"]
)
def test_freeze_review_and_execution_selection_tamper(tmp_path, monkeypatch, kind):
    root, _, _, _, _ = package(tmp_path, monkeypatch)
    freeze, review, decision, qualification = [
        v.read_json(root / name)
        for name in (
            "FREEZE.json",
            "REVIEW.json",
            "EXECUTION-DECISION.json",
            "SYNTHETIC-QUALIFICATION.json",
        )
    ]
    if kind == "environment":
        freeze["environment"]["PYTHONOPTIMIZE"] = "1"
    elif kind == "decision":
        decision["review_sha256"] = "0" * 64
    elif kind == "review":
        review["status"] = "pending"
    elif kind == "approved_bytes":
        review["verifier_tests_sha256"] = "0" * 64
    else:
        qualification["target_model_calls"] = True
    with pytest.raises(v.VerificationError):
        v.freeze_shape(root, freeze, review, decision, qualification)


@pytest.mark.parametrize("leftover", [None, "records.json", "analysis.json", "error.json"])
def test_startup_failure_zero_slot_receipt_without_tokenizer(tmp_path, monkeypatch, leftover):
    import shutil

    root, protocol, freeze_sha, _, _ = package(tmp_path, monkeypatch)
    execution = root / "execution"
    shutil.rmtree(execution / "evaluations")
    (execution / "execution-finished.json").unlink()
    startup = v.read_json(execution / "startup-attempt.json")
    put(
        execution / "startup-failure.json",
        {
            "schema_version": "a182-startup-failure-v1",
            "startup_attempt_sha256": v.digest(startup),
            "status": "failed",
            "elapsed_seconds": 0.1,
            "interruption": None,
            "target_model_calls": 0,
            "error": {
                "code": "a182_startup_failed",
                "exception_type": "RuntimeError",
                "frames": [],
            },
        },
    )
    for name in ("records.json", "analysis.json", "error.json"):
        assert not (execution / name).exists()
    if leftover is not None:
        put(execution / leftover, {"public": "contradictory post-header evidence"})
        with pytest.raises(v.VerificationError, match="startup_no_post_header_exports"):
            v.terminal(
                root, protocol, freeze_sha, lambda _: pytest.fail("startup invokes tokenizer")
            )
        assert not (root / "VERIFICATION.json").exists()
        return
    out = v.terminal(root, protocol, freeze_sha, lambda _: pytest.fail("startup invokes tokenizer"))
    assert out["status"] == "pass" and out["scope"] == "startup_failure_zero_target_calls"
    assert out["target_model_calls"] == 0 and out["actual_model_files_rehashed"] is False


@pytest.mark.parametrize("field", ["pid", "signal_number"])
def test_boolean_terminal_signal_pid_rejected(field):
    interruption = {
        "signal_number": 1,
        "signal_name": "SIGHUP",
        "pid": 10,
        "exception_type": "A182Interrupted",
    }
    interruption[field] = True
    with pytest.raises(AssertionError):
        v.interruption_shape(interruption, 10, "interrupted")


def test_json_duplicate_nonfinite_and_symlink_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    for raw in (b'{"k":1,"k":2}', b'{"k":NaN}'):
        path.write_bytes(raw)
        with pytest.raises(v.VerificationError):
            v.read_json(path)
    other = tmp_path / "other.json"
    other.write_bytes(b"{}")
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(v.VerificationError):
        v.read_json(path)


def test_optimization_guard_refuses_before_operations():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-B", "-O", str(Path(v.__file__)), "--help"], capture_output=True
    )
    assert result.returncode != 0
    assert b"a182_verifier_requires_unoptimized_python" in result.stderr
    assert result.stdout == b""


@pytest.mark.parametrize(
    "receipt,field,value",
    [
        ("job.json", "pid", True),
        ("job.json", "command", []),
        ("resource-queue.json", "start_ticks", False),
        ("queue-start-one-shot.json", "watcher_sha256", "0" * 64),
        ("queue-one-shot.json", "resource_queue_sha256", "0" * 64),
        ("launch-one-shot.json", "qualifying_check_monotonic", 60.0),
    ],
)
def test_operational_receipt_lineage_tamper(tmp_path, monkeypatch, receipt, field, value):
    root, _, freeze_sha, _, _ = package(tmp_path, monkeypatch)
    obj = v.read_json(root / receipt)
    obj[field] = value
    put(root / receipt, obj)
    with pytest.raises((v.VerificationError, AssertionError)):
        v._freeze_audit(root, freeze_sha)


@pytest.mark.parametrize(
    "kind", ["blocked_as_infrastructure", "missing_invocation", "sequence_hole"]
)
def test_dependency_and_dispatch_evidence_rejection(kind):
    data = fixture(failure=0 if kind == "blocked_as_infrastructure" else None)
    if kind == "blocked_as_infrastructure":
        key = next(k for k, r in data["results"].items() if r["status"] == "dependency_unavailable")
        data["results"][key]["status"] = "infrastructure_failed"
        data["results"][key]["error"] = {
            "exception_type": "RuntimeError",
            "frames": [],
            "code": "synthetic",
        }
    else:
        key = data["plan"]["evaluations"][0]["evaluation_id"]
        if kind == "missing_invocation":
            data["invoked"].remove(key)
            del data["invocation_receipts"][key]
        else:
            data["attempted"].remove(key)
            del data["attempts"][key]
    with pytest.raises(v.VerificationError):
        replay(data)


@pytest.mark.parametrize("kind", ["closed_eos", "pure_prefix", "payload_special"])
def test_fake_native_alignment_rejection(kind):
    class Malformed(PublicTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            out = super().apply_chat_template(messages, **kwargs)
            if (
                kind == "closed_eos"
                and kwargs.get("tokenize")
                and not kwargs.get("add_generation_prompt")
            ):
                return out[:-1]
            return out

        def decode(self, tokens, **kwargs):
            if kind == "pure_prefix" and tokens == [10, 11, 12]:
                return '{"answer":"s'
            return super().decode(tokens, **kwargs)

    tokenizer = Malformed()
    if kind == "payload_special":
        tokenizer.all_special_ids = [*tokenizer.all_special_ids, 1000 + ord("S")]
    with pytest.raises(v.VerificationError):
        v.construct_native(
            v.expected_plan(bindings()),
            fake_config("/unopened"),
            tokenizer,
            [128001, 128008, 128009],
        )
