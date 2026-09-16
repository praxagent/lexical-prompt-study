"""Public synthetic A169 contracts; never load any checkpoint or private data."""
import copy
import importlib.util
import json
from pathlib import Path
import signal
import types

import pytest

from lexical_prompt_study import instruction_selection_answer_path as a

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a169_public_helpers", ROOT / "tests/test_instruction_selection_label_rename.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class Tokenizer(helper.Tokenizer):
    eos_token_id = 1

    def encode(self, text, **kwargs):
        if text.endswith("<EOS>"):
            return super().encode(text[:-5], **kwargs) + [1]
        return super().encode(text, **kwargs)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        if kwargs["add_generation_prompt"]:
            return super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
        assert len(messages) == 3 and messages[-1]["role"] == "assistant"
        text = super().apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        text += messages[-1]["content"] + "<EOS>"
        return self.encode(text) if tokenize else text


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prior, old_args = helper.fixture.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    protocol = tmp_path / "a169.md"
    protocol.write_text("Public synthetic teacher-forced protocol")
    plan = a.compile_plan(prior["a167_plan"], prior["a167_run_header"], a.native.engine.file_digest(protocol))
    path = tmp_path / "a169-plan.json"
    path.write_bytes(a.tasks.canonical(plan))
    args = {**old_args, "plan_path": path, "plan_sha256": a.native.engine.file_digest(path),
            "protocol_path": protocol, "output_root": tmp_path / "a169-output", "tokenizer_loader": lambda _: Tokenizer()}
    return plan, args


def prepare(fixture):
    return a.prepare_inputs(**{k: v for k, v in fixture[1].items() if k != "output_root"})


class FakeScorer:
    def __init__(self, plan, mode="complete", interrupt_after=None):
        self.plan, self.mode, self.interrupt_after, self.calls = plan, mode, interrupt_after, 0

    def __call__(self, model, torch, prompt, target, prefix_count, device):
        if self.calls == self.interrupt_after:
            a._interrupt(signal.SIGTERM, None)
        evaluation = self.plan["evaluations"][self.calls]
        self.calls += 1
        trial = self.plan["trials"][evaluation["context_index"]]
        kind = evaluation["candidate_kind"]
        assert target[-1] == 1 and device == "cpu"
        assert Tokenizer().decode(target[:-1]) == trial["canonical_candidates"][kind]
        if self.mode == "infrastructure" and self.calls == 40:
            raise RuntimeError("synthetic private error text")
        if self.mode == "baseline_missing" and self.calls == 1:
            raise RuntimeError("synthetic baseline failure")
        desired = {"selected_value": -2., "other_value": -5., "selected_label": -6., "other_label": -7.}[kind]
        if trial["diagnostic_phase"] == "full" and kind == "selected_value":
            desired = -1. - trial["world_index"] / 10
        if self.mode == "tie" and evaluation["context_index"] == 0 and kind == "other_value":
            desired = -2.
        if self.mode == "wrong" and evaluation["context_index"] == 0 and kind == "other_value":
            desired = -1.
        values = [-.25] * prefix_count + [desired / (len(target) - prefix_count)] * (len(target) - prefix_count)
        if self.mode == "prefix_drift" and self.calls == 1:
            values[0] -= 2 * a.PREFIX_ABSOLUTE_TOLERANCE_NATS
        return {"chosen_logits": values.copy(), "log_normalizers": [0.] * len(values),
                "token_logprobs": values, "scores": a._scores(values, prefix_count)}


def execute(fixture, mode="complete", interrupt_after=None, loader_error=None, memory=None):
    scorer = FakeScorer(fixture[0], mode, interrupt_after)
    loaded = []
    def load(config, engine):
        loaded.append(True)
        if loader_error:
            raise loader_error
        return types.SimpleNamespace(tokenizer=Tokenizer(), eos_ids=[1], model=object(), torch=object(), device="cpu")
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES if memory is None else memory,
                                      load_cpu_reference=load)
    result = a.run_plan(**fixture[1], reference_loader=lambda _: reference, candidate_scorer=scorer)
    assert len(loaded) == 1
    return result, scorer


def test_fixed72_parent_contexts_and_all_candidates(fixture):
    plan, _ = fixture
    assert len(plan["trials"]) == 72 and len(plan["evaluations"]) == 288
    assert len({a.object_sha(t["messages"]) for t in plan["trials"]}) == 72
    assert [t["diagnostic_phase"] for t in plan["trials"][:8]] == ["baseline"] * 8
    assert {p: sum(t["diagnostic_phase"] == p for t in plan["trials"]) for p in a.PHASES} == {
        "baseline": 8, "full": 16, "sham": 16, "inert": 16, "prose": 16}
    for t in plan["trials"]:
        old = plan["a167_plan"] if t["source_study"] == "a167" else plan["a167_plan"]["a165_plan"]
        original = next(row for row in old["trials"] if row["trial_id"] == t["source_trial_id"])
        assert t["messages"] == original["messages"]
        assert t["world"] == original["world"] and t["selector"] == original["selector"]
        assert list(t["canonical_candidates"]) == list(a.KINDS)
    groups = [plan["trials"][i:i + 4] for i in range(8, 72, 4)]
    orders = [tuple(t["diagnostic_phase"] for t in group) for group in groups]
    assert all(orders.count(order) == 4 for order in a.MATERIAL_ORDERS)
    assert sum(o.index("full") < o.index("sham") for o in orders) == 8
    assert sum(o.index("prose") < o.index("inert") for o in orders) == 8
    for group, order in zip(groups, orders):
        t = group[0]
        assert len({(t["world_index"], t["selector_index"], t["placement"]) for t in group}) == 1
        assert order == a.MATERIAL_ORDERS[(t["world_index"] + t["selector_index"] + (t["placement"] == "after")) % 4]


def test_full_joint_tokenization_prefix_eos_and_private_return(fixture):
    result = prepare(fixture)
    assert len(result["native_inputs"]) == 72
    assert result["native_prepared_sha256"] == a.object_sha(result["native_inputs"])
    for value in result["native_inputs"].values():
        assert value["shared_json_prefix_text"] == a.JSON_PREFIX
        for kind in a.KINDS:
            tokens = value["candidate_token_ids"][kind]
            assert tokens[-1] == 1 and 1 not in tokens[:-1]
            assert value["candidate_lengths"][kind]["joint_tokens"] == len(tokens)
            assert value["candidate_lengths"][kind]["answer_continuation_tokens"] == len(tokens) - len(a.JSON_PREFIX)
    assert not a._PROCESS_CONSUMED and not fixture[1]["output_root"].exists()


@pytest.mark.parametrize("mutation", ["message", "candidate", "order", "world", "parent_header", "missing"])
def test_plan_drift_rejected(fixture, mutation):
    plan = copy.deepcopy(fixture[0])
    if mutation == "message":
        plan["trials"][8]["messages"][1]["content"] += "changed"
    elif mutation == "candidate":
        plan["trials"][0]["canonical_candidates"]["selected_value"] += " "
    elif mutation == "order":
        plan["trials"][8:10] = reversed(plan["trials"][8:10])
    elif mutation == "world":
        plan["trials"][0]["world_index"] = 3
    elif mutation == "parent_header":
        plan["a167_run_header"]["native_prepared_sha256"] = "0" * 64
    else:
        plan["evaluations"].pop()
    with pytest.raises(ValueError):
        a.validate_plan(plan)


@pytest.mark.parametrize("mutation", ["prompt_boundary", "eos", "common_syntax", "source_identity"])
def test_native_boundary_failures_prevent_any_model_call(fixture, monkeypatch, mutation):
    original = a.native.prepare_generation
    if mutation == "source_identity":
        def drift(tokenizer, trial, config):
            result = original(tokenizer, trial, config)
            if "canonical_candidates" in trial:
                result["prompt_token_ids"][0] += 1
            return result
        monkeypatch.setattr(a.native, "prepare_generation", drift)
    else:
        class BadTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                result = super().encode(text, **kwargs)
                if mutation == "prompt_boundary" and a.JSON_PREFIX in text:
                    result[0] += 1
                return result
            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
                if mutation == "eos" and not kwargs["add_generation_prompt"]:
                    return result + [1] if tokenize else result + "<EOS>"
                return result
            def decode(self, tokens, **kwargs):
                result = super().decode(tokens, **kwargs)
                return "unexpected" if mutation == "common_syntax" and result == a.JSON_PREFIX else result
        fixture[1]["tokenizer_loader"] = lambda _: BadTokenizer()
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture[1]["output_root"].exists()


def test_shifted_logits_eos_normalization_and_fresh_no_cache():
    torch = pytest.importorskip("torch")
    class Model:
        def __init__(self):
            self.calls = 0
        def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
            self.calls += 1
            assert input_ids.tolist() == [[7, 8, 3, 4, 5, 1]]
            assert not use_cache and logits_to_keep == 5
            assert attention_mask.tolist() == [[1] * 6]
            logits = torch.zeros(1, 6, 10, dtype=torch.float32)
            for position, target in enumerate([8, 3, 4, 5, 1]):
                logits[0, position, target] = position + 1
            logits[0, -1, :] = float("nan")  # Last logit is outside this path.
            return types.SimpleNamespace(logits=logits[:, -logits_to_keep:, :])
    model = Model()
    result = a.teacher_force(model, torch, [7, 8], [3, 4, 5, 1], 2)
    expected = [v - math_log_exp(v, 9) for v in (2., 3., 4., 5.)]
    assert result["token_logprobs"] == pytest.approx(expected, abs=1e-12)
    assert result["chosen_logits"] == [2., 3., 4., 5.]
    assert result["scores"]["eos_logprob"] == pytest.approx(expected[-1])
    assert result["scores"]["joint_logprob"] == pytest.approx(sum(expected))
    assert result["scores"]["answer_continuation_logprob"] == pytest.approx(sum(expected[2:]))
    assert result["scores"]["joint_tokens"] == 4 and model.calls == 1


def math_log_exp(value, other):
    import math
    return math.log(math.exp(value) + other)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_and_scores_rejected(value):
    torch = pytest.importorskip("torch")
    def model(**kw):
        return types.SimpleNamespace(logits=torch.full((1, 4, 8), value))
    with pytest.raises(ValueError, match="finite_logits"):
        a.teacher_force(model, torch, [5], [2, 3, 1], 1)
    with pytest.raises(ValueError, match="finite_token_logprobs"):
        a._scores([-.1, value], 1)


def test_complete_schedule_replays_scores_and_fixed_world_contrasts(fixture):
    result, scorer = execute(fixture)
    assert scorer.calls == 288 and result["status"] == "complete"
    assert result["baseline_gate"]["gate_passed"]
    for placement in ("before", "after"):
        contrast = result["primary_full_minus_sham"][placement]
        assert contrast["planned_pairs"] == contrast["identified_pairs"] == 8
        assert contrast["world_mean_difference_nats"] == pytest.approx([1., .9, .8, .7])
        assert contrast["equal_world_mean_difference_nats"] == pytest.approx(.85)
        assert result["secondary_prose_minus_inert"][placement]["equal_world_mean_difference_nats"] == pytest.approx(0.)
        pairwise = result["secondary_pairwise_margins"]["sham"][placement]
        assert pairwise["selected_minus_other_value_nats"]["equal_world_mean_nats"] == pytest.approx(3.)
        assert pairwise["selected_minus_selected_label_nats"]["equal_world_mean_nats"] == pytest.approx(4.)
    replay = a.export_run(**fixture[1])
    assert replay["aggregate"] == result and len(replay["records"]) == 72
    text = json.dumps(result)
    assert all(t["trial_id"] not in text and t["world_id"] not in text for t in fixture[0]["trials"])
    assert "token_logprobs" not in text and "prompt_token_ids" not in text


@pytest.mark.parametrize("mode", ["tie", "wrong", "baseline_missing", "prefix_drift"])
def test_baseline_gate_stops_exactly32_and_keeps_all_missing(mode, fixture):
    result, scorer = execute(fixture, mode)
    assert scorer.calls == 32 and not result["baseline_gate"]["gate_passed"]
    assert result["status"] == "baseline_unqualified" and result["coverage"]["unattempted"] == 256
    assert result["primary_full_minus_sham"]["before"]["equal_world_mean_difference_nats"] is None
    assert result["primary_full_minus_sham"]["before"]["missing_pairs"] == 8
    assert result["arms"]["full"]["selected_preference_missing"] == 16
    assert a.export_run(**fixture[1])["aggregate"] == result


def test_material_failure_no_retry_and_continuous_missingness(fixture):
    result, scorer = execute(fixture, "infrastructure")
    assert scorer.calls == 288
    assert result["coverage"]["infrastructure_failed"] == 1
    assert result["coverage"]["completed"] == 287
    assert result["primary_full_minus_sham"]["before"]["equal_world_mean_difference_nats"] is None
    assert result["primary_full_minus_sham"]["before"]["identified_pairs"] == 7
    assert result["primary_full_minus_sham"]["before"]["continuous_missingness_bounds"] == "unbounded"
    assert result["primary_full_minus_sham"]["after"]["identified_pairs"] == 8


def test_interruption_and_load_deadline_preserve_missingness(fixture):
    result, scorer = execute(fixture, interrupt_after=40)
    assert scorer.calls == 40 and result["coverage"]["interrupted"] == 1
    assert result["coverage"]["unattempted"] == 247
    assert a.export_run(**fixture[1])["aggregate"] == result


def test_deadline_during_load_and_no_resume(fixture, monkeypatch):
    result, scorer = execute(fixture, loader_error=a.A169Deadline())
    assert scorer.calls == 0 and result["coverage"]["unattempted"] == 288
    with pytest.raises(ValueError, match="fresh_process"):
        execute(fixture)
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        execute(fixture)


@pytest.mark.parametrize("mutation", ["normalizer", "sum", "native", "order", "orphan", "analysis", "finished"])
def test_receipt_replay_rejects_changed_arithmetic_and_lineage(fixture, mutation):
    execute(fixture)
    plan, args = fixture
    root = args["output_root"]
    folder = root / "evaluations" / plan["evaluations"][0]["evaluation_id"]
    if mutation in ("normalizer", "sum"):
        path = folder / "result.json"
        value = json.loads(path.read_bytes())
        if mutation == "normalizer":
            value["measurement"]["log_normalizers"][0] += 1
        else:
            value["measurement"]["scores"]["joint_logprob"] += 1
    elif mutation == "native":
        path = root / "native-inputs.json"
        value = json.loads(path.read_bytes())
        value[plan["trials"][0]["trial_id"]]["prompt_token_ids"][0] += 1
    elif mutation == "order":
        path = folder / "attempt.json"
        value = json.loads(path.read_bytes())
        value["evaluation"]["sequence_index"] += 1
    elif mutation == "orphan":
        (folder / "attempt.json").unlink()
        with pytest.raises(ValueError):
            a.export_run(**args)
        return
    elif mutation == "finished":
        path = root / "execution-finished.json"
        value = json.loads(path.read_bytes())
        value["status"] = "baseline_unqualified"
    else:
        path = root / "analysis.json"
        value = json.loads(path.read_bytes())
        value["coverage"]["completed"] = 1
    path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**args)


def test_memory_gate_before_model_and_one_shot(fixture):
    with pytest.raises(ValueError, match="preload_memory_gate"):
        execute(fixture, memory=a.MIN_AVAILABLE_BYTES - 1)
    assert not (fixture[1]["output_root"] / "one-shot.json").exists()


def test_signals_timer_and_baseexceptions():
    old = {n: signal.getsignal(n) for n in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    with pytest.raises(a.A169Interrupted), a.bounded_signals():
        assert 7190 < signal.getitimer(signal.ITIMER_REAL)[0] <= 7200
        a._interrupt(signal.SIGTERM, None)
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert {n: signal.getsignal(n) for n in old} == old
