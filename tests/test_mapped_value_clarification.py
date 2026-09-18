"""A171 independent public synthetic qualification; no target-model calls."""

import copy
import itertools
import json
import os
import signal
import types
from collections import Counter
from pathlib import Path

import pytest

from lexical_prompt_study import mapped_value_clarification as a


EXPECTED_WORLDS = (
    ("a171_v47", "a171_v83", "AB"),
    ("a171_v26", "a171_v91", "BA"),
    ("a171_v68", "a171_v14", "BA"),
    ("a171_v35", "a171_v72", "AB"),
    ("a171_v89", "a171_v20", "AB"),
    ("a171_v56", "a171_v03", "BA"),
    ("a171_v11", "a171_v64", "BA"),
    ("a171_v97", "a171_v42", "AB"),
)

EXPECTED_BASE = (
    ("none", "standard", "A"),
    ("before", "clarified", "B"),
    ("after", "standard", "A"),
    ("none", "clarified", "B"),
    ("before", "standard", "A"),
    ("after", "clarified", "B"),
    ("none", "standard", "B"),
    ("before", "clarified", "A"),
    ("after", "standard", "B"),
    ("none", "clarified", "A"),
    ("before", "standard", "B"),
    ("after", "clarified", "A"),
)

EXPECTED_PROSE = (
    "Beyond the window, a narrow garden path curved around a patch of low shrubs. "
    "Rain from the previous evening remained on the broad leaves, while the gravel "
    "had begun to dry. A wooden bench stood beneath the nearest tree. Its surface "
    "was smooth where visitors usually sat, and a few fallen leaves rested at one "
    "end. Farther along the path, small flowers grew beside a shallow stone basin. "
    "Water moved gently when a breeze crossed the garden. The branches above made "
    "shifting patterns of light on the ground. Near the wall, a climbing plant "
    "reached toward a sheltered corner. The air felt cool in the shade and warmer "
    "beside the open lawn. From time to time a bird landed on the fence, paused, "
    "and flew toward the neighboring trees. The scene changed slowly as the sun "
    "rose. Shadows shortened, the leaves became less damp, and the quiet path "
    "remained open between the plants."
)


class Tokenizer:
    """Character coding with explicit EOS IDs, independent of target tokenization."""

    eos_token_id, pad_token_id, bos_token_id = 1, None, 0

    def __len__(self):
        return 512

    def get_chat_template(self):
        return "a171 public synthetic native template"

    def encode(self, text, **kwargs):
        result = []
        while text:
            if text.startswith("<EOS>"):
                result.append(1)
                text = text[5:]
            else:
                result.append(ord(text[0]) + 10)
                text = text[1:]
        return result

    def decode(self, tokens, **kwargs):
        return "".join("<EOS>" if token in (1, 2) else chr(token - 10) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        text = messages[0]["content"] + "\nUSER\n" + messages[1]["content"] + "\nASSISTANT\n"
        if not kwargs["add_generation_prompt"]:
            assert len(messages) == 3 and messages[2]["role"] == "assistant"
            text += messages[2]["content"] + "<EOS>"
        return self.encode(text) if tokenize else text


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    model = tmp_path / "synthetic-model"
    model.mkdir()
    generation = model / "generation_config.json"
    generation.write_text('{"eos_token_id":[1,2]}')
    model_config = model / "config.json"
    model_config.write_text('{"max_position_embeddings":8192,"vocab_size":512}')
    digest = a.native.engine.file_digest
    config = {
        "schema_version": "a163-runtime-v1", "model_path": str(model), "model_revision": "a" * 40,
        "model_files_sha256": {
            **{name: "b" * 64 for name in
               ("tokenizer_config.json", "tokenizer.json", "model.safetensors")},
            "config.json": digest(model_config),
            "generation_config.json": digest(generation),
        },
        "chat_template_sha256": a.tasks.sha(Tokenizer().get_chat_template().encode()),
        "runtime_versions": {name: "synthetic" for name in ("python", *a.native.engine.VERSION_PACKAGES)},
        "quantization": "nf4", "dtype": "bfloat16", "attention_implementation": "sdpa", "device": "cuda:0",
        "max_prompt_tokens": 4096, "max_new_tokens": 64, "seed": 1, "cpu_threads": 4, "gpu_memory_gib": 10,
    }
    config_path = tmp_path / "config.json"
    protocol = tmp_path / "protocol.md"
    reference = tmp_path / "reference.py"
    config_path.write_bytes(a.tasks.canonical(config))
    protocol.write_text("Public synthetic A171 protocol fixture")
    reference.write_text("Public synthetic reference binding fixture")
    monkeypatch.setattr(a, "CONFIG_SHA", digest(config_path))
    monkeypatch.setattr(a.native.engine, "file_digest", lambda path: a.REFERENCE_SHA if Path(path) == reference else digest(path))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: config["model_files_sha256"])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    monkeypatch.setattr(a, "PRIVATE_RUNS_ROOT", tmp_path)
    plan = a.compile_plan(digest(config_path), digest(protocol), digest(Path(__file__)))
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(a.tasks.canonical(plan))
    args = {
        "plan_path": plan_path, "plan_sha256": digest(plan_path), "config_path": config_path,
        "config_sha256": digest(config_path), "protocol_path": protocol, "reference_script": reference,
        "output_root": tmp_path / "output", "tokenizer_loader": lambda config: Tokenizer(),
    }
    return plan, args


def prepare(fixture):
    return a.prepare_inputs(**{key: value for key, value in fixture[1].items() if key != "output_root"})


def encoded_answer(trial, field="selected_answer", eos=1):
    return Tokenizer().encode(json.dumps({"answer": trial[field]}, separators=(",", ":"))) + [eos]


def test_exact_public_stimuli_and_all_order_balances(fixture):
    trials = fixture[0]["trials"]
    assert tuple(a.WORLDS) == EXPECTED_WORLDS
    assert tuple(a.BASE_SCHEDULE) == EXPECTED_BASE
    assert a.PROSE == EXPECTED_PROSE
    assert len(trials) == 96 and len({trial["trial_id"] for trial in trials}) == 96
    assert [trial["sequence_index"] for trial in trials] == list(range(96))
    positions = []
    for world, (value_a, value_b, order) in enumerate(EXPECTED_WORLDS):
        cells = [trial for trial in trials if trial["world_index"] == world]
        assert len(cells) == 12
        shift = 3 * (world // 2)
        expected_order = EXPECTED_BASE[shift:] + EXPECTED_BASE[:shift]
        if world % 2:
            expected_order = expected_order[::-1]
        actual_order = tuple((trial["context"], trial["wording"], trial["selector"]) for trial in cells)
        assert actual_order == expected_order
        positions.append({cell: index for index, cell in enumerate(actual_order)})
        mapping = {"A": value_a, "B": value_b}
        core = "Mapping:\n" + json.dumps({label: mapping[label] for label in order}, separators=(",", ":"))
        for trial in cells:
            selector = trial["selector"]
            assert trial["world_id"] == f"w{world:02d}"
            assert trial["presentation_order"] == order
            assert trial["selected_answer"] == mapping[selector]
            assert trial["unselected_answer"] == mapping["B" if selector == "A" else "A"]
            system = (
                f"Select the entry labeled {selector} from the user's mapping. Return that entry's mapped value "
                'as exactly one JSON object with the single key "answer" and a string value. '
                "Do not return any other text."
            )
            if trial["wording"] == "clarified":
                system += " The selector names a mapping entry; the answer is its mapped value, not the selector label."
            user = {"none": core, "before": EXPECTED_PROSE + "\n\n" + core,
                    "after": core + "\n\n" + EXPECTED_PROSE}[trial["context"]]
            assert trial["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for context in ("none", "before", "after"):
            assert len({trial["messages"][1]["content"] for trial in cells if trial["context"] == context}) == 1
    for left, right in itertools.combinations(EXPECTED_BASE, 2):
        assert sum(position[left] < position[right] for position in positions) == 4
    assert Counter(world[2] for world in EXPECTED_WORLDS) == {"AB": 4, "BA": 4}
    assert Counter(world[2] for world in EXPECTED_WORLDS[::2]) == {"AB": 2, "BA": 2}
    assert Counter(world[2] for world in EXPECTED_WORLDS[1::2]) == {"AB": 2, "BA": 2}
    assert len({value for world in EXPECTED_WORLDS for value in world[:2]}) == 16


@pytest.mark.parametrize("change", ["message", "oracle", "world", "schedule", "duplicate", "config", "protocol", "tests"])
def test_plan_drift_rejected(fixture, change):
    plan = copy.deepcopy(fixture[0])
    if change == "message":
        plan["trials"][0]["messages"][1]["content"] += " "
    elif change == "oracle":
        plan["trials"][0]["selected_answer"] = "synthetic incorrect oracle"
    elif change == "world":
        plan["trials"][0]["world_index"] = 7
    elif change == "schedule":
        plan["trials"][0:2] = reversed(plan["trials"][0:2])
    elif change == "duplicate":
        plan["trials"].append(copy.deepcopy(plan["trials"][-1]))
    else:
        plan["bindings"][f"{change}_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.validate_plan(plan)


@pytest.mark.parametrize("answer_kind,category", [
    ("selected_answer", "correct"), ("unselected_answer", "other_mapped_value"),
    ("selector", "selected_label"), ("other_selector", "other_label"),
    ("unrecognized", "other_or_format"),
])
@pytest.mark.parametrize("selector", ["A", "B"])
def test_exhaustive_valid_answer_categories(fixture, answer_kind, category, selector):
    trial = next(trial for trial in fixture[0]["trials"] if trial["selector"] == selector)
    answer = trial.get(answer_kind, "B" if selector == "A" else "A")
    if answer_kind == "unrecognized":
        answer = "synthetic-other"
    tokens = Tokenizer().encode(json.dumps({"answer": answer}, separators=(",", ":"))) + [2]
    score = a.score_tokens(Tokenizer(), tokens, trial, [1, 2])
    assert score["category"] == category
    assert score["strict_correct"] is (category == "correct")
    assert score["selected_label"] is (category == "selected_label")
    assert score["format_valid"] and score["eos_valid"] and not score["capped"]
    assert score["generated_token_count"] == len(tokens)
    assert score["attempted"] is True


@pytest.mark.parametrize("text", [
    '{"answer":"A","answer":"A"}', '{"answer":"A","extra":0}',
    '{"answer":0}', '{"answer":null}', '{"ANSWER":"A"}', '[{"answer":"A"}]',
    '"A"', '```json\n{"answer":"A"}\n```', '{"answer":"A"} trailing',
    '{"answer":"A"}{"answer":"A"}', '{"answer":', '',
])
def test_strict_format_rejections(fixture, text):
    trial = fixture[0]["trials"][0]
    score = a.score_tokens(Tokenizer(), Tokenizer().encode(text) + [1], trial, [1, 2])
    assert score["category"] == "other_or_format"
    assert not score["strict_correct"] and not score["selected_label"] and not score["format_valid"]
    assert score["eos_valid"] and not score["capped"]


def test_whitespace_terminal_eos_and_cap_precedence(fixture):
    trial = fixture[0]["trials"][0]
    body = Tokenizer().encode(json.dumps({"answer": trial["selected_answer"]}, separators=(",", ":")))
    last_slot_eos = body + [ord(" ") + 10] * (63 - len(body)) + [1]
    score = a.score_tokens(Tokenizer(), last_slot_eos, trial, [1, 2])
    assert len(last_slot_eos) == 64
    assert score["category"] == "correct" and score["eos_valid"] and not score["capped"]
    capped = last_slot_eos[:-1] + [ord(" ") + 10]
    score = a.score_tokens(Tokenizer(), capped, trial, [1, 2])
    assert score["category"] == "cap" and score["capped"] and not score["strict_correct"]
    short = a.score_tokens(Tokenizer(), body, trial, [1, 2])
    assert short["category"] == "other_or_format" and not short["eos_valid"] and not short["capped"]
    premature = a.score_tokens(Tokenizer(), [1] + body + [2], trial, [1, 2])
    assert premature["category"] == "other_or_format" and not premature["eos_valid"]


@pytest.mark.parametrize("tail", [[ord("x") + 10], [1, ord("x") + 10, 2], [ord("x") + 10, 2]])
def test_generation_adapter_preserves_explicit_settings_and_observable_termination(tail):
    torch = pytest.importorskip("torch")
    calls = []

    class Model:
        config = types.SimpleNamespace(vocab_size=512)

        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.tensor([kwargs["input_ids"][0].tolist() + tail], dtype=torch.long)

    runtime = types.SimpleNamespace(model=Model(), tokenizer=Tokenizer(), torch=torch,
                                    device="cpu", eos_ids=[1, 2], config={"max_new_tokens": 64})
    result = a.generate_tokens(runtime, {"prompt_token_ids": [20, 21, 22]})
    assert result == tail and len(calls) == 1
    call = calls[0]
    assert call["input_ids"].tolist() == [[20, 21, 22]]
    assert call["attention_mask"].tolist() == [[1, 1, 1]]
    assert call["logits_to_keep"] == 1
    generation = call["generation_config"]
    assert generation.max_new_tokens == 64
    assert generation.do_sample is False and generation.num_beams == 1 and generation.use_cache is True
    assert generation.eos_token_id == [1, 2] and generation.pad_token_id == 1
    assert generation.bos_token_id == 0


@pytest.mark.parametrize("change", ["prompt", "empty", "overlong", "negative", "out_of_range"])
def test_generation_adapter_rejects_malformed_arrays(change):
    torch = pytest.importorskip("torch")

    class Model:
        config = types.SimpleNamespace(vocab_size=512)

        def generate(self, **kwargs):
            prefix = kwargs["input_ids"][0].tolist()
            tail = [30, 1]
            if change == "prompt":
                prefix[0] += 1
            elif change == "empty":
                tail = []
            elif change == "overlong":
                tail = [30] * 65
            elif change == "negative":
                tail = [-1, 1]
            else:
                tail = [512, 1]
            return torch.tensor([prefix + tail], dtype=torch.long)

    runtime = types.SimpleNamespace(model=Model(), tokenizer=Tokenizer(), torch=torch,
                                    device="cpu", eos_ids=[1, 2], config={"max_new_tokens": 64})
    with pytest.raises(ValueError):
        a.generate_tokens(runtime, {"prompt_token_ids": [20, 21, 22]})


def test_native_preparation_replays_every_public_cell_without_consumption(fixture):
    prepared = prepare(fixture)
    assert prepared["planned_cells"] == prepared["maximum_generation_calls"] == 96
    assert prepared["eos_token_ids"] == [1, 2]
    assert prepared["native_prepared_sha256"] == a.object_sha(prepared["native_inputs"])
    tokenizer = Tokenizer()
    for trial in fixture[0]["trials"]:
        receipt = prepared["native_inputs"][trial["trial_id"]]
        text = tokenizer.apply_chat_template(trial["messages"], tokenize=False, add_generation_prompt=True)
        assert receipt["rendered_text"] == text
        assert receipt["prompt_token_ids"] == tokenizer.encode(text)
        assert receipt["intended_eos_token_id"] == 1
        assert receipt["assistant_probe_token_ids"] == encoded_answer(trial)
    assert not fixture[1]["output_root"].exists() and a._PROCESS_CONSUMED is False


@pytest.mark.parametrize("change", ["template", "rendering", "boundary", "decode", "assistant_eos", "native_eos", "context"])
def test_native_preflight_drift_stops_without_consumption(fixture, change):
    class BadTokenizer(Tokenizer):
        eos_token_id = 99 if change == "native_eos" else 1

        def get_chat_template(self):
            return super().get_chat_template() + (" changed" if change == "template" else "")

        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if change == "boundary" and text.endswith('"}'):
                result[0] += 1
            return result

        def decode(self, tokens, **kwargs):
            return super().decode(tokens, **kwargs) + (" altered" if change == "decode" else "")

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            if change == "rendering" and tokenize and kwargs["add_generation_prompt"]:
                result[0] += 1
            if change == "assistant_eos" and not kwargs["add_generation_prompt"]:
                result = result + [1] if tokenize else result + "<EOS>"
            if change == "context":
                result = result + [ord(" ") + 10] * 5000 if tokenize else result + " " * 5000
            return result

    fixture[1]["tokenizer_loader"] = lambda config: BadTokenizer()
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture[1]["output_root"].exists() and a._PROCESS_CONSUMED is False


def make_records(plan, assignment):
    records = []
    for trial in plan["trials"]:
        outcome = assignment(trial)
        if outcome is None:
            result = None
        elif outcome == "infrastructure":
            result = {"status": "infrastructure_failed"}
        else:
            answer = trial["selector"] if outcome == "label" else trial["selected_answer"] if outcome else trial["unselected_answer"]
            tokens = Tokenizer().encode(json.dumps({"answer": answer}, separators=(",", ":"))) + [1]
            result = {"status": "completed", "score": a.score_tokens(Tokenizer(), tokens, trial, [1, 2])}
        records.append(a._record(trial, result, attempted=outcome is not None))
    return records


def independent_contrast(records, field, weights):
    """Literal within-world, within-selector paired arithmetic from the protocol."""
    total = 0.0
    for world in range(8):
        for selector in ("A", "B"):
            for context, weight in weights.items():
                pair = {row["wording"]: row[field] for row in records
                        if row["world_index"] == world and row["selector"] == selector and row["context"] == context}
                total += weight * (int(pair["clarified"]) - int(pair["standard"])) / 16
    return total


def test_paired_weights_placements_and_interaction_signs(fixture):
    limits = {"none": {"standard": 11, "clarified": 9},
              "before": {"standard": 3, "clarified": 10}, "after": {"standard": 7, "clarified": 12}}

    def assignment(trial):
        unit = 2 * trial["world_index"] + (trial["selector"] == "B")
        return unit < limits[trial["context"]][trial["wording"]]

    records = make_records(fixture[0], assignment)
    contrasts = a._contrasts(records, "strict_correct")
    for context, expected in (("none", -2 / 16), ("before", 7 / 16), ("after", 5 / 16)):
        result = contrasts["wording_by_context"][context]
        assert result == {"point": expected, "lower": expected, "upper": expected,
                          "planned_outcomes": 32, "resolved_outcomes": 32, "planned_pairs": 16, "resolved_pairs": 16}
        assert result["point"] == independent_contrast(records, "strict_correct", {context: 1})
    assert contrasts["primary_prose_wording"]["point"] == 6 / 16
    assert contrasts["primary_prose_wording"]["planned_pairs"] == 32
    assert contrasts["prose_minus_none_interaction"]["point"] == 8 / 16
    assert contrasts["prose_minus_none_interaction"]["planned_pairs"] == 48
    assert contrasts["placement_minus_none_interaction"]["before"]["point"] == 9 / 16
    assert contrasts["placement_minus_none_interaction"]["after"]["point"] == 7 / 16
    labels = make_records(fixture[0], lambda trial: "label" if assignment(trial) else False)
    assert a._contrasts(labels, "selected_label") == contrasts


@pytest.mark.parametrize("direction", [-1, 1])
def test_complete_interactions_can_reach_plus_or_minus_two(fixture, direction):
    def assignment(trial):
        positive = (trial["wording"] == "clarified") != (trial["context"] == "none")
        return positive if direction == 1 else not positive

    result = a._contrasts(make_records(fixture[0], assignment), "strict_correct")
    for contrast in [result["prose_minus_none_interaction"], *result["placement_minus_none_interaction"].values()]:
        assert contrast["point"] == contrast["lower"] == contrast["upper"] == 2 * direction


def test_all_unresolved_interaction_bounds_are_not_clipped(fixture):
    records = make_records(fixture[0], lambda trial: None)
    result = a._contrasts(records, "strict_correct")
    for contrast in result["wording_by_context"].values():
        assert contrast == {"point": None, "lower": -1, "upper": 1, "planned_outcomes": 32,
                            "resolved_outcomes": 0, "planned_pairs": 16, "resolved_pairs": 0}
    interaction = result["prose_minus_none_interaction"]
    assert interaction == {"point": None, "lower": -2, "upper": 2, "planned_outcomes": 96,
                           "resolved_outcomes": 0, "planned_pairs": 48, "resolved_pairs": 0}


def test_partial_bounds_equal_independent_exhaustive_assignments(fixture):
    missing = {(0, "A", "none", "standard"), (3, "B", "before", "clarified"), (7, "A", "after", "standard")}

    def assignment(trial):
        key = (trial["world_index"], trial["selector"], trial["context"], trial["wording"])
        if key in missing:
            return None if trial["context"] != "before" else "infrastructure"
        return (trial["world_index"] + (trial["selector"] == "B")) % 3 == 0

    records = make_records(fixture[0], assignment)
    unknown = [row for row in records if row["strict_correct"] is None]
    assert len(unknown) == 3
    weights = {"none": -1, "before": .5, "after": .5}
    values = []
    for outcomes in itertools.product((False, True), repeat=3):
        candidate = copy.deepcopy(records)
        replacements = dict(zip((row["trial_id"] for row in unknown), outcomes, strict=True))
        for row in candidate:
            if row["trial_id"] in replacements:
                row["strict_correct"] = replacements[row["trial_id"]]
        values.append(independent_contrast(candidate, "strict_correct", weights))
    result = a._contrast(records, "strict_correct", weights)
    assert result["point"] is None and result["lower"] == min(values) and result["upper"] == max(values)
    assert result["planned_outcomes"] == 96 and result["resolved_outcomes"] == 93
    assert result["planned_pairs"] == 48 and result["resolved_pairs"] == 45
    counts = a._counts(records)
    assert sum(counts["categories"].values()) == 96
    assert counts["categories"]["missing"] == 2 and counts["categories"]["infrastructure"] == 1
    assert counts["strict_accuracy"]["point"] is None


def test_deeply_nested_decoded_json_is_behavioral_format_failure(fixture):
    class NestedTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return "[" * 2000 + "0" + "]" * 2000

    score = a.score_tokens(NestedTokenizer(), [30, 1], fixture[0]["trials"][0], [1, 2])
    assert score["category"] == "other_or_format" and not score["format_valid"]
    assert score["eos_valid"] and not score["strict_correct"]


def execute(fixture, *, fail_at=None, interrupt_at=None, signal_number=signal.SIGTERM,
            loader_error=None, response=None):
    plan, args = fixture
    calls, loaded = [], []

    def generate(runtime, prepared):
        index = len(calls)
        trial = plan["trials"][index]
        calls.append(trial["trial_id"])
        expected = Tokenizer().apply_chat_template(trial["messages"], tokenize=False, add_generation_prompt=True)
        assert prepared["rendered_text"] == expected
        assert prepared["prompt_token_ids"] == Tokenizer().encode(expected)
        if index == interrupt_at:
            if signal_number is None:
                raise KeyboardInterrupt
            if signal_number == signal.SIGALRM:
                a._deadline(signal_number, None)
            a._interrupt(signal_number, None)
        if index == fail_at:
            raise RuntimeError("synthetic confidential exception must not escape")
        return response(trial) if response else encoded_answer(trial)

    def load(config, engine):
        loaded.append(True)
        assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
        if loader_error:
            raise loader_error
        return types.SimpleNamespace(tokenizer=Tokenizer(), eos_ids=[1, 2], device="cpu")

    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    aggregate = a.run_plan(**args, reference_loader=lambda path: reference, generation_runner=generate)
    assert len(loaded) == 1
    return aggregate, calls


def test_complete_synthetic_generation_schedule_and_export(fixture):
    aggregate, calls = execute(fixture)
    assert calls == [trial["trial_id"] for trial in fixture[0]["trials"]]
    assert aggregate["status"] == "complete" and aggregate["execution_status"] == "finished_schedule"
    assert aggregate["coverage"] == {"attempted": 96, "completed": 96, "infrastructure_failed": 0,
                                     "interrupted": 0, "unattempted": 0, "missing": 0}
    assert aggregate["overall"]["categories"]["correct"] == 96
    replay = a.export_run(**fixture[1])
    assert replay["aggregate"] == aggregate and len(replay["records"]) == 96
    assert all(row["strict_correct"] is True for row in replay["records"])
    assert aggregate["contrasts"]["strict_accuracy"]["primary_prose_wording"]["point"] == 0
    serialized = json.dumps(aggregate)
    assert "prompt_token_ids" not in serialized and "response_text" not in serialized
    assert all(trial["trial_id"] not in serialized for trial in fixture[0]["trials"])
    assert (fixture[1]["output_root"] / "native-inputs.json").stat().st_mode & 0o777 == 0o600


def test_all_baseline_failures_and_caps_do_not_gate_remaining_calls(fixture):
    def response(trial):
        if trial["context"] == "none":
            return [ord("x") + 10] * 64
        return encoded_answer(trial, field="selector")

    aggregate, calls = execute(fixture, response=response)
    assert len(calls) == 96 and aggregate["coverage"]["completed"] == 96
    assert aggregate["overall"]["categories"]["cap"] == 32
    assert aggregate["overall"]["categories"]["selected_label"] == 64
    assert aggregate["overall"]["strict_accuracy"]["point"] == 0


def test_infrastructure_failure_is_unresolved_and_never_retried(fixture):
    aggregate, calls = execute(fixture, fail_at=2)
    assert len(calls) == len(set(calls)) == 96
    assert aggregate["status"] == "incomplete" and aggregate["execution_status"] == "finished_schedule"
    assert aggregate["coverage"]["infrastructure_failed"] == 1
    assert aggregate["overall"]["categories"]["infrastructure"] == 1
    assert aggregate["overall"]["strict_accuracy"] == {
        "point": None, "lower": 95 / 96, "upper": 1., "planned_outcomes": 96, "resolved_outcomes": 95}
    row = a.export_run(**fixture[1])["records"][2]
    assert row["strict_correct"] is None and row["selected_label"] is None and row["attempted"] is True
    result = json.loads((fixture[1]["output_root"] / "trials" / calls[2] / "result.json").read_text())
    assert "confidential" not in json.dumps(result)


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_signal_deadline_and_keyboard_interrupt_preserve_coverage(fixture, number):
    aggregate, calls = execute(fixture, interrupt_at=3, signal_number=number)
    assert len(calls) == 4
    assert aggregate["coverage"] == {"attempted": 4, "completed": 3, "infrastructure_failed": 0,
                                     "interrupted": 1, "unattempted": 92, "missing": 93}
    receipt = json.loads((fixture[1]["output_root"] / "execution-finished.json").read_text())
    assert receipt["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert receipt["interruption"] == {
        "signal_number": int(number) if number is not None else None,
        "signal_name": signal.Signals(number).name if number is not None else None,
        "pid": os.getpid(), "exception_type": "KeyboardInterrupt" if number is None else
        "A171Deadline" if number == signal.SIGALRM else "A171Interrupted"}
    exported = a.export_run(**fixture[1])
    assert exported["aggregate"] == aggregate
    assert exported["records"][3]["attempted"] is True and exported["records"][4]["attempted"] is False
    assert exported["records"][3]["category"] == exported["records"][4]["category"] == "missing"


def test_signal_context_restores_handlers_and_uses_two_hour_limit():
    old = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM)}
    with a.bounded_signals():
        remaining, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 7199 < remaining <= 7200 and interval == 0
        assert signal.getsignal(signal.SIGALRM) is a._deadline
        assert signal.getsignal(signal.SIGTERM) is a._interrupt
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert {number: signal.getsignal(number) for number in old} == old


def test_loader_failure_consumes_one_shot_and_cannot_change_output_path(fixture, monkeypatch):
    aggregate, calls = execute(fixture, loader_error=RuntimeError("synthetic loader failure"))
    assert calls == [] and aggregate["execution_status"] == "failed"
    assert aggregate["coverage"]["unattempted"] == aggregate["coverage"]["missing"] == 96
    original_root = fixture[1]["output_root"]
    fixture[1]["output_root"] = original_root.parent / "different-output"
    with pytest.raises(ValueError, match="fresh_process"):
        execute(fixture)
    fixture[1]["output_root"] = original_root
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        execute(fixture)


def test_system_exit_is_not_a_recoverable_evaluation_error(fixture):
    with pytest.raises(SystemExit):
        execute(fixture, loader_error=SystemExit(7))
    root = fixture[1]["output_root"]
    assert (root / "one-shot.json").exists()
    assert json.loads((root / "execution-finished.json").read_text())["status"] == "failed"
    assert a.export_run(**fixture[1])["aggregate"]["coverage"]["unattempted"] == 96


@pytest.mark.parametrize("tamper", ["native", "attempt", "tokens", "score", "orphan", "order", "receipt", "signal", "records", "aggregate"])
def test_export_rejects_tampered_evidence(fixture, tamper):
    execute(fixture)
    plan, args = fixture
    root = args["output_root"]
    folder = root / "trials" / plan["trials"][1]["trial_id"]
    if tamper == "orphan":
        (root / "trials" / "unplanned").mkdir()
    elif tamper == "order":
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
    else:
        path = {"native": root / "native-inputs.json", "attempt": folder / "attempt.json",
                "tokens": folder / "result.json", "score": folder / "result.json",
                "receipt": root / "execution-finished.json", "signal": root / "execution-finished.json",
                "records": root / "records.json", "aggregate": root / "analysis.json"}[tamper]
        value = json.loads(path.read_text())
        if tamper == "native":
            value[plan["trials"][0]["trial_id"]]["prompt_token_ids"][0] += 1
        elif tamper == "attempt":
            value["sequence_index"] += 1
        elif tamper == "tokens":
            value["generated_token_ids"][0] += 1
        elif tamper == "score":
            value["score"]["strict_correct"] = False
        elif tamper == "receipt":
            value["run_sha256"] = "0" * 64
        elif tamper == "signal":
            value["interruption"] = {"signal_number": 15, "signal_name": "SIGTERM", "pid": os.getpid(),
                                     "exception_type": "A171Interrupted"}
        elif tamper == "records":
            value[0]["category"] = "other_mapped_value"
        else:
            value["contrasts"]["strict_accuracy"]["primary_prose_wording"]["point"] = .5
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**args)


def test_exclusive_lock_prevents_execution(fixture):
    import fcntl
    root = fixture[1]["output_root"]
    root.mkdir()
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a.run_plan(**fixture[1])


def rebind_model_metadata(fixture, monkeypatch, **changes):
    """Build a new synthetic bound plan, never relax production metadata guards."""
    _, args = fixture
    config = json.loads(args["config_path"].read_text())
    path = Path(config["model_path"]) / "config.json"
    metadata = json.loads(path.read_text())
    metadata.update(changes)
    path.write_bytes(a.tasks.canonical(metadata))
    config["model_files_sha256"]["config.json"] = a.native.engine.file_digest(path)
    args["config_path"].write_bytes(a.tasks.canonical(config))
    args["config_sha256"] = a.native.engine.file_digest(args["config_path"])
    monkeypatch.setattr(a, "CONFIG_SHA", args["config_sha256"])
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: config["model_files_sha256"])
    plan = a.compile_plan(args["config_sha256"], a.native.engine.file_digest(args["protocol_path"]),
                          a.native.engine.file_digest(Path(__file__)))
    args["plan_path"].write_bytes(a.tasks.canonical(plan))
    args["plan_sha256"] = a.native.engine.file_digest(args["plan_path"])
    return plan, args


def test_model_context_limit_includes_all_64_generation_tokens(fixture, monkeypatch):
    largest = max(len(value["prompt_token_ids"]) for value in prepare(fixture)["native_inputs"].values())
    exact = rebind_model_metadata(fixture, monkeypatch, max_position_embeddings=largest + 64)
    native = prepare(exact)
    assert all(value["model_context_limit"] == largest + 64 for value in native["native_inputs"].values())
    too_short = rebind_model_metadata(exact, monkeypatch, max_position_embeddings=largest + 63)
    with pytest.raises(ValueError, match="generation_context_fit"):
        prepare(too_short)
    assert a._PROCESS_CONSUMED is False


def test_native_vocabulary_bound_is_not_inferred_from_decoder_acceptance(fixture, monkeypatch):
    native = prepare(fixture)["native_inputs"]
    highest = max(token for value in native.values() for token in value["prompt_token_ids"] + value["assistant_probe_token_ids"])
    exact = rebind_model_metadata(fixture, monkeypatch, vocab_size=highest + 1)
    assert len(prepare(exact)["native_inputs"]) == 96
    too_small = rebind_model_metadata(exact, monkeypatch, vocab_size=highest)
    with pytest.raises(ValueError, match="native_token_vocabulary"):
        prepare(too_small)


def test_model_metadata_content_must_match_bound_hash(fixture):
    config = json.loads(fixture[1]["config_path"].read_text())
    (Path(config["model_path"]) / "config.json").write_text('{"max_position_embeddings":9999,"vocab_size":512}')
    with pytest.raises(ValueError, match="model_config_binding"):
        prepare(fixture)


def test_out_of_vocabulary_injected_output_is_infrastructure_and_no_retry(fixture):
    target = fixture[0]["trials"][2]["trial_id"]

    def response(trial):
        return [512, 1] if trial["trial_id"] == target else encoded_answer(trial)

    aggregate, calls = execute(fixture, response=response)
    assert len(calls) == len(set(calls)) == 96
    assert aggregate["coverage"]["infrastructure_failed"] == 1
    assert a.export_run(**fixture[1])["records"][2]["category"] == "infrastructure"


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_preflight_interrupt_preserves_startup_receipt_and_consumes_claim(fixture, monkeypatch, number):
    def interrupted(*args):
        if number is None:
            raise KeyboardInterrupt
        if number == signal.SIGALRM:
            a._deadline(number, None)
        a._interrupt(number, None)

    monkeypatch.setattr(a, "_inputs", interrupted)
    exception = KeyboardInterrupt if number is None else a.A171Deadline if number == signal.SIGALRM else a.A171Interrupted
    with pytest.raises(exception):
        a.run_plan(**fixture[1])
    root = fixture[1]["output_root"]
    startup = json.loads((root / "startup-attempt.json").read_text())
    failed = json.loads((root / "startup-failure.json").read_text())
    assert failed["startup_attempt_sha256"] == a.object_sha(startup)
    assert failed["target_model_calls"] == 0 and not (root / "one-shot.json").exists()
    assert failed["status"] == ("deadline" if number == signal.SIGALRM else "interrupted")
    assert failed["interruption"]["signal_number"] == (int(number) if number is not None else None)
    assert failed["interruption"]["signal_name"] == (signal.Signals(number).name if number is not None else None)
    assert failed["interruption"]["pid"] == os.getpid()
    assert a._PROCESS_CONSUMED is True
    with pytest.raises(ValueError, match="fresh_process"):
        a.run_plan(**fixture[1])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[1])


def test_preload_memory_failure_records_zero_calls_and_consumes_startup(fixture, monkeypatch):
    calls = []

    def forbidden(*args):
        calls.append(True)
        pytest.fail("model loader called below the 48 GiB memory gate")

    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES - 1, load_cpu_reference=forbidden)
    with pytest.raises(ValueError, match="preload_memory_gate"):
        a.run_plan(**fixture[1], reference_loader=lambda path: reference)
    root = fixture[1]["output_root"]
    assert calls == [] and not (root / "one-shot.json").exists()
    failed = json.loads((root / "startup-failure.json").read_text())
    assert failed["target_model_calls"] == 0 and failed["status"] == "failed" and failed["interruption"] is None
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    with pytest.raises(ValueError, match="consumed_run"):
        a.run_plan(**fixture[1], reference_loader=lambda path: reference)


def test_cli_exposes_only_safe_status_coverage_and_private_log(fixture, monkeypatch, capsys):
    def fake_run(**kwargs):
        print("synthetic private token and loader output")
        return {"status": "complete", "execution_status": "finished_schedule",
                "coverage": {"attempted": 96, "completed": 96}, "response_text": "synthetic private response"}

    monkeypatch.setattr(a, "run_plan", fake_run)
    args = fixture[1]
    argv = []
    for flag, key in (("plan", "plan_path"), ("plan-sha256", "plan_sha256"), ("config", "config_path"),
                      ("config-sha256", "config_sha256"), ("protocol", "protocol_path"),
                      ("reference-script", "reference_script"), ("output-root", "output_root")):
        argv.extend(["--" + flag, str(args[key])])
    previous = os.umask(0o077)
    try:
        assert a.main(argv) == 0
    finally:
        os.umask(previous)
    captured = capsys.readouterr()
    assert set(json.loads(captured.out)) == {"status", "execution_status", "coverage"}
    assert "synthetic private" not in captured.out + captured.err
    log = args["output_root"] / "execution.log"
    assert "synthetic private" in log.read_text() and log.stat().st_mode & 0o777 == 0o600
