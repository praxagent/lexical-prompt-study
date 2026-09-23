"""A181 public synthetic qualification; no target tokenizer or pretrained weights."""

import contextlib
import copy
import itertools
import json
import math
import os
import signal
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from lexical_prompt_study import user_selector_conflict as a


SELECTION = "Select the entry labeled {selector} from the user's mapping."
FORMAT = ('Return the mapped value of the entry labeled {selector} as exactly one JSON object '
          'with the single key "answer" and a string value.')
CLOSING = 'Do not transform the value or include other fields, explanations, or Markdown fences.'
JSON_PREFIX = '{"answer":"'
PAIRS = (('s104', 's105'), ('s106', 's107'))
PACKAGES = ('clean', 'full', 'sham')
USER_STATES = ('none', 'concordant', 'contradictory')
CONDITIONS = tuple(f'{package}_{state}' for package in PACKAGES for state in USER_STATES)
DIRECTIVE = 'Select the entry labeled {user_selector} from the mapping.'
MATERIALS = {'full': 'Public full material.', 'sham': 'Public sham fixture with a distinct length.'}
SELECTOR_ORDERS = (('A', 'B'), ('B', 'A'), ('B', 'A'), ('A', 'B'))


class Tokenizer:
    """Public character coding, with a fixed three-token canonical JSON opener."""

    eos_token_id, pad_token_id, bos_token_id = 1, None, 0
    all_special_ids = [0, 1, 2]
    syntax = {7: '{"', 8: 'answer', 9: '":"'}

    def __len__(self):
        return 512

    def get_chat_template(self):
        return 'a181 public synthetic native template'

    def encode(self, text, **kwargs):
        result = []
        while text:
            if text.startswith('<EOS>'):
                result.append(1)
                text = text[5:]
            elif text.startswith(JSON_PREFIX):
                result.extend((7, 8, 9))
                text = text[len(JSON_PREFIX):]
            else:
                result.append(ord(text[0]) + 20)
                text = text[1:]
        return result

    def decode(self, tokens, **kwargs):
        return ''.join('<EOS>' if token in (1, 2) else self.syntax[token]
                       if token in self.syntax else chr(token - 20) for token in tokens)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        assert messages[0]['role'] == 'system' and messages[1]['role'] == 'user'
        text = messages[0]['content'] + '\nUSER\n' + messages[1]['content'] + '\nASSISTANT\n'
        if not kwargs['add_generation_prompt']:
            assert len(messages) == 3 and messages[2]['role'] == 'assistant'
            text += messages[2]['content'] + '<EOS>'
        return self.encode(text) if tokenize else text


def test_teacher_force_uses_preceding_logits_including_eos_and_discards_final_row():
    import torch

    class Model:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
            self.calls += 1
            assert input_ids.tolist() == [[7, 8, 3, 4, 5, 6, 1]]
            assert attention_mask.tolist() == [[1] * 7]
            assert use_cache is False and logits_to_keep == 6
            logits = torch.zeros(1, 7, 10, dtype=torch.float32)
            for position, target in enumerate([8, 3, 4, 5, 6, 1]):
                logits[0, position, target] = position + 1
            logits[0, -1, :] = float('nan')
            return types.SimpleNamespace(logits=logits[:, -logits_to_keep:, :])

    model = Model()
    result = a.teacher_force(model, torch, [7, 8], [3, 4, 5, 6, 1], 3)
    expected = [value - math.log(math.exp(value) + 9) for value in (2., 3., 4., 5., 6.)]
    assert model.calls == 1 and result['chosen_logits'] == [2., 3., 4., 5., 6.]
    assert result['token_logprobs'] == pytest.approx(expected, abs=1e-12)
    assert result['scores']['json_prefix_logprob'] == pytest.approx(math.fsum(expected[:3]), abs=1e-12)
    assert result['scores']['answer_continuation_logprob'] == pytest.approx(math.fsum(expected[3:]), abs=1e-12)
    assert result['scores']['eos_logprob'] == pytest.approx(expected[-1], abs=1e-12)
    assert result['scores']['joint_logprob'] == pytest.approx(math.fsum(expected), abs=1e-12)
    assert result['scores']['json_prefix_tokens'] == 3 and result['scores']['answer_continuation_tokens'] == 2
    assert result['scores']['eos_tokens'] == 1 and result['scores']['joint_tokens'] == 5


def test_tiny_untrained_torch_causal_model_matches_independent_full_forward():
    import torch

    class TinyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 7)
            self.readout = torch.nn.Linear(7, 16, bias=False)
            self.calls = []

        def all_logits(self, ids):
            return self.readout(torch.cumsum(self.embedding(ids), dim=1))

        def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
            assert use_cache is False and bool((attention_mask == 1).all())
            self.calls.append(input_ids.tolist())
            return types.SimpleNamespace(logits=self.all_logits(input_ids)[:, -logits_to_keep:])

    with torch.random.fork_rng():
        torch.manual_seed(20260918)
        model = TinyCausal().float().eval()
        prompt, target, alternate = [10, 11], [7, 8, 9, 12, 1], [7, 8, 9, 13, 1]
        one = a.teacher_force(model, torch, prompt, target, 3)
        two = a.teacher_force(model, torch, prompt, target, 3)
        changed = a.teacher_force(model, torch, prompt, alternate, 3)
        with torch.no_grad():
            logits = model.all_logits(torch.tensor([prompt + target]))[0].double()
        expected = [torch.log_softmax(logits[len(prompt) - 1 + j], dim=-1)[token].item()
                    for j, token in enumerate(target)]
    assert one == two and len(model.calls) == 3
    assert model.calls == [[prompt + target], [prompt + target], [prompt + alternate]]
    assert one['token_logprobs'] == pytest.approx(expected, abs=1e-12)
    assert one['token_logprobs'][:3] == changed['token_logprobs'][:3]
    assert one['scores']['answer_continuation_logprob'] == pytest.approx(sum(expected[3:]), abs=1e-12)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_used_logits_are_rejected(value):
    import torch

    def model(**kwargs):
        return types.SimpleNamespace(logits=torch.full((1, 5, 16), value))

    with pytest.raises(ValueError, match='finite_logits'):
        a.teacher_force(model, torch, [10], [7, 8, 9, 1], 3)


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
        "max_prompt_tokens": 4048, "max_new_tokens": 64, "seed": 1, "cpu_threads": 4, "gpu_memory_gib": 10,
    }
    config_path = tmp_path / "config.json"
    protocol = tmp_path / "protocol.md"
    reference = tmp_path / "reference.py"
    config_path.write_bytes(a.tasks.canonical(config))
    protocol.write_text("Public synthetic A181 protocol fixture")
    reference.write_text("Public synthetic reference binding fixture")
    monkeypatch.setattr(a, "CONFIG_SHA", digest(config_path))
    monkeypatch.setattr(a.native.engine, "file_digest", lambda path: a.REFERENCE_SHA if Path(path) == reference else digest(path))
    monkeypatch.setattr(a.native.engine, "snapshot_manifest", lambda path: config["model_files_sha256"])
    monkeypatch.setattr(a, "_PROCESS_CONSUMED", False)
    monkeypatch.setattr(a, "PRIVATE_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(a, "MATERIAL_RECEIPTS", {key: {"sha256": a.tasks.sha(value.encode()), "bytes": len(value.encode())}
                                               for key, value in MATERIALS.items()})
    plan = a.compile_plan(digest(config_path), digest(protocol), digest(Path(__file__)), MATERIALS)
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


def measurement(target_count, suffix=-.01, prefix=0., eos=0.):
    values = [0.] * target_count
    values[0], values[3], values[-1] = prefix, suffix - eos, eos
    prefix_sum = math.fsum(values[:3])
    suffix_sum = math.fsum(values[3:])
    return {'chosen_logits': list(values), 'log_normalizers': [0.] * target_count, 'token_logprobs': values,
            'scores': {'json_prefix_logprob': prefix_sum, 'answer_continuation_logprob': suffix_sum,
                       'joint_logprob': math.fsum((prefix_sum, suffix_sum)), 'eos_logprob': values[-1],
                       'json_prefix_tokens': 3, 'answer_continuation_tokens': target_count - 3,
                       'joint_tokens': target_count, 'eos_tokens': 1}}


def synthetic_results(plan, prepared, *, selected=(-.01, -.01), other=(-.02, -.02),
                      missing=(), prefix_values=None, context_index=0):
    results = {}
    for evaluation in plan['evaluations']:
        if evaluation['sequence_index'] in missing:
            continue
        context = plan['contexts'][evaluation['context_index']]
        candidate, repeat = evaluation['candidate'], evaluation['repeat_index']
        if evaluation['context_index'] == context_index:
            score = (selected if candidate == context['selected_candidate'] else other)[repeat]
            prefix = (prefix_values or {}).get(evaluation['sequence_index'], 0.)
        else:
            score = -.01 if candidate == context['selected_candidate'] else -.02
            prefix = 0.
        count = len(prepared[context['context_id']]['candidate_token_ids'][candidate])
        results[evaluation['evaluation_id']] = {'status': 'completed', 'measurement': measurement(count, score, prefix)}
    return results


def summarize(fixture, **kwargs):
    plan = fixture[0]
    prepared = prepare(fixture)['native_inputs']
    results = synthetic_results(plan, prepared, **kwargs)
    records = a._records(plan, prepared, results)
    aggregate = a._aggregate(plan, records, results, set(results))
    return records, aggregate


def execute(fixture, *, fail_at=None, interrupt_at=None, signal_number=signal.SIGTERM,
            loader_error=None, mode='pass', scorer_transform=None):
    plan, args = fixture
    calls, loaded = [], []

    def scorer(model, torch, prompt, target, prefix_count, device):
        index = len(calls)
        evaluation = plan['evaluations'][index]
        context = plan['contexts'][evaluation['context_index']]
        calls.append(evaluation['evaluation_id'])
        assert prefix_count == 3 and device == 'cpu' and target[-1] == 1
        assert Tokenizer().decode(target[:-1]) == context['canonical_candidates'][evaluation['candidate']]
        assert prompt == Tokenizer().apply_chat_template(context['messages'], tokenize=True, add_generation_prompt=True)
        if index == interrupt_at:
            if signal_number is None:
                raise KeyboardInterrupt
            if signal_number == signal.SIGALRM:
                a._deadline(signal_number, None)
            a._interrupt(signal_number, None)
        if index == fail_at:
            raise RuntimeError('synthetic confidential exception must not escape')
        selected = evaluation['candidate'] == context['selected_candidate']
        suffix = -.01 if selected else -.02
        if mode == 'tie' and evaluation['context_index'] == 0:
            suffix = -.01
        if mode == 'reverse' and evaluation['context_index'] == 0:
            suffix = -.02 if selected else -.01
        if mode == 'repeat' and index == 3:
            suffix -= .0002
        value = measurement(len(target), suffix, -.0002 if mode == 'prefix' and index == 0 else 0.)
        return scorer_transform(value, index) if scorer_transform else value

    def load(config, engine):
        loaded.append(True)
        assert os.environ['HF_HUB_OFFLINE'] == '1' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
        if loader_error:
            raise loader_error
        return types.SimpleNamespace(tokenizer=Tokenizer(), eos_ids=[1, 2], device='cpu', model=object(), torch=object())

    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES, load_cpu_reference=load)
    aggregate = a.run_plan(**args, reference_loader=lambda path: reference, forward_scorer=scorer)
    assert len(loaded) == 1
    return aggregate, calls




def test_exact_four_protocol_templates_and_original_teacher_force_alias():
    import re
    text = Path('plans/user_selector_conflict_a181.md').read_text()
    assert re.findall(r'```text\n([^\n]+)\n```', text) == [SELECTION, FORMAT, CLOSING, DIRECTIVE]
    assert a.teacher_force is a.a169.teacher_force
    assert a.native.engine.file_digest(Path(a.a169.__file__)) == a.A169_SHA
    assert a.native.engine.file_digest(Path(a.previous.__file__)) == a.A179_SHA
    assert a.PREFIX_COUNT == 3 and a.NUMERICAL_TOLERANCE_NATS == 1e-4 and a.FUNCTIONAL_SEPARATION_NATS == 1e-3


@pytest.mark.parametrize('change', ['context', 'candidate', 'selected', 'assignment', 'order', 'repeat', 'count', 'threshold', 'config', 'protocol', 'tests'])
def test_fixed_plan_drift_rejected(fixture, change):
    plan = copy.deepcopy(fixture[0])
    if change == 'context':
        plan['contexts'][0]['messages'][1]['content'] += ' '
    elif change == 'candidate':
        plan['contexts'][0]['canonical_candidates']['value_1'] += ' '
    elif change == 'selected':
        plan['contexts'][0]['selected_candidate'] = 'value_2'
    elif change == 'assignment':
        plan['contexts'][0]['assignment_swapped'] = True
    elif change == 'order':
        plan['evaluations'][:2] = plan['evaluations'][:2][::-1]
    elif change == 'repeat':
        plan['evaluations'][0]['repeat_index'] = 1
    elif change == 'count':
        plan['evaluations'].pop()
    elif change == 'threshold':
        plan['functional_separation_nats'] /= 2
    else:
        plan['bindings'][change + '_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def test_native_paths_share_exact_fixed_three_tokens_not_longest_prefix(fixture):
    native = prepare(fixture)
    assert native['planned_contexts'] == 72 and native['planned_forward_calls'] == 288
    assert native['paired_user_selector_contexts'] == 24
    assert len({a.object_sha(v['prompt_token_ids']) for v in native['native_inputs'].values()}) == 72
    assert native['native_prepared_sha256'] == a.object_sha(native['native_inputs'])
    for context in fixture[0]['contexts']:
        value = native['native_inputs'][context['context_id']]
        assert value['shared_json_prefix_token_ids'] == [7, 8, 9]
        assert value['shared_json_prefix_text'] == JSON_PREFIX
        paths = value['candidate_token_ids']
        assert paths['value_1'][:4] == paths['value_2'][:4]  # Common value stem is NOT added to q.
        assert len(paths['value_1']) == len(paths['value_2']) == 10
        for candidate, path in paths.items():
            assert path[-1] == 1 and 1 not in path[:-1]
            assert Tokenizer().decode(path[:-1]) == context['canonical_candidates'][candidate]
            closed = Tokenizer().apply_chat_template(context['messages'] + [{'role': 'assistant', 'content': context['canonical_candidates'][candidate]}],
                                                      tokenize=True, add_generation_prompt=False)
            assert closed == value['prompt_token_ids'] + path
    assert a._PROCESS_CONSUMED is False and not fixture[1]['output_root'].exists()


@pytest.mark.parametrize('change', ['template', 'rendering', 'boundary', 'decode', 'eos', 'native_eos', 'syntax', 'length', 'context'])
def test_native_construction_drift_rejected_before_consumption(fixture, change):
    class BadTokenizer(Tokenizer):
        eos_token_id = 99 if change == 'native_eos' else 1

        def get_chat_template(self):
            return super().get_chat_template() + (' drift' if change == 'template' else '')

        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if change == 'boundary' and text.endswith('"}') and JSON_PREFIX in text:
                result[0] += 1
            if change == 'length' and '"answer":"s105"' in text:
                result.append(30)
            return result

        def decode(self, tokens, **kwargs):
            result = super().decode(tokens, **kwargs)
            if change == 'decode' and len(tokens) > 3:
                result += 'x'
            if change == 'syntax' and tokens == [7, 8, 9]:
                result = 'not JSON syntax'
            return result

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            if change == 'rendering' and tokenize and kwargs['add_generation_prompt']:
                result[0] += 1
            if change == 'eos' and not kwargs['add_generation_prompt']:
                result = result + [1] if tokenize else result + '<EOS>'
            if change == 'context':
                result = result + [40] * 5000 if tokenize else result + ' ' * 5000
            return result

    fixture[1]['tokenizer_loader'] = lambda config: BadTokenizer()
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture[1]['output_root'].exists() and a._PROCESS_CONSUMED is False


@pytest.mark.parametrize('prefix,difference,valid', [(0., 1e-4, True), (0., math.nextafter(1e-4, math.inf), False)])
def test_prefix_guard_inclusive_exact_boundary(fixture, prefix, difference, valid):
    records, aggregate = summarize(fixture, prefix_values={0: prefix - difference})
    row = records[0]
    assert row['prefix_sum_range_nats'] == difference and row['prefix_guard_passed'] is valid
    assert row['numerical_valid'] is valid
    assert aggregate['clean_qualification']['qualified'] is valid
    if not valid:
        assert row['mean_margin_nats'] is None and row['worst_margin_nats'] is None and row['margins_unbounded']
        assert all(row[key] is None for key in a.FUNCTIONAL_FLAGS)


@pytest.mark.parametrize('difference,valid', [(1e-4, True), (math.nextafter(1e-4, math.inf), False)])
def test_repeat_guard_inclusive_exact_boundary(fixture, difference, valid):
    records, _ = summarize(fixture, selected=(0., -difference))
    row = records[0]
    assert row['continuation_repeat_drift_nats']['value_1'] == difference
    assert row['repeat_guard_passed']['value_1'] is valid and row['numerical_valid'] is valid
    assert row['complete'] and row['completed_evaluations'] == 4


@pytest.mark.parametrize('selected,other,status,flags', [
    ((0., 0.), (0., 0.), 'worst_tie', {'mean_tie', 'worst_tie'}),
    ((-.00004, 0.), (-.00004, -.00008), 'worst_tie', {'worst_tie'}),
    ((-.002, -.002), (0., 0.), 'worst_reversed', {'mean_reversed', 'worst_reversed'}),
    ((0., -.00009), (-.00006, -.00006), 'worst_reversed', {'worst_reversed'}),
    ((0., 0.), (-.0005, -.0005), 'positive_low_separation', {'positive_low_separation'}),
    ((0., 0.), (-.001, -.001), 'threshold_boundary', {'positive_low_separation', 'threshold_boundary'}),
    ((0., 0.), (-math.nextafter(.001, math.inf), -math.nextafter(.001, math.inf)), 'pass', {'functional_passed'}),
])
def test_functional_threshold_ties_reversals_and_positive_low_separation(fixture, selected, other, status, flags):
    records, aggregate = summarize(fixture, selected=selected, other=other)
    row = records[0]
    expected_mean = math.fsum(selected) / 2 - math.fsum(other) / 2
    expected_worst = min(selected) - max(other)
    assert row['numerical_valid'] is True and row['functional_status'] == status
    assert {key for key in a.FUNCTIONAL_FLAGS if row[key]} == flags
    assert row['mean_margin_nats'] == expected_mean and row['worst_margin_nats'] == expected_worst
    assert row['mean_margin_bounds_nats'] == {'lower': expected_mean, 'upper': expected_mean}
    assert row['worst_margin_bounds_nats'] == {'lower': expected_worst, 'upper': expected_worst}
    assert row['margins_unbounded'] is False and aggregate['clean_qualification']['qualified'] is (status == 'pass')


@pytest.mark.parametrize('missing_mask', range(16))
def test_all_missingness_patterns_keep_margins_unknown_until_all_four_complete(fixture, missing_mask):
    missing = [index for index in range(4) if missing_mask & (1 << index)]
    records, aggregate = summarize(fixture, missing=missing)
    row = records[0]
    assert row['completed_evaluations'] == 4 - len(missing)
    assert row['complete'] is (not missing)
    assert row['prefix_guard_passed'] is (None if missing else True)
    for candidate, indices in (('value_1', {0, 3}), ('value_2', {1, 2})):
        expected = None if set(missing).intersection(indices) else True
        assert row['repeat_guard_passed'][candidate] is expected
    if missing:
        assert row['numerical_valid'] is None and aggregate['clean_qualification']['qualified'] is None
        assert row['mean_margin_nats'] is None and row['worst_margin_nats'] is None
        assert row['mean_margin_bounds_nats'] == row['worst_margin_bounds_nats'] == {'lower': None, 'upper': None}
        assert all(row[key] is None for key in a.FUNCTIONAL_FLAGS)
    else:
        assert row['numerical_valid'] is True and aggregate['clean_qualification']['qualified'] is True


def test_partial_known_repeat_failure_disqualifies_without_observed_subset_margin(fixture):
    records, aggregate = summarize(fixture, selected=(0., -.0002), missing=(1,))
    row = records[0]
    assert row['complete'] is False and row['prefix_guard_passed'] is None
    assert row['repeat_guard_passed'] == {'value_1': False, 'value_2': None}
    assert row['numerical_valid'] is False and aggregate['clean_qualification']['qualified'] is False
    assert row['mean_margin_nats'] is None and row['worst_margin_nats'] is None and row['margins_unbounded']
    assert all(row[key] is None for key in a.FUNCTIONAL_FLAGS)




def test_complete_schedule_replays_all_288_fresh_measurements_privately(fixture):
    aggregate, calls = execute(fixture)
    assert calls == [row['evaluation_id'] for row in fixture[0]['evaluations']]
    assert aggregate['clean_qualification']['qualified'] is True and aggregate['status'] == 'complete'
    assert aggregate['coverage'] == {'attempted': 288, 'completed': 288, 'infrastructure_failed': 0,
                                     'interrupted': 0, 'unattempted': 0, 'missing': 0}
    assert aggregate['complete_contexts'] == 72 and aggregate['numerical_valid'] == {'true': 72, 'false': 0, 'unknown': 0}
    replay = a.export_run(**fixture[1])
    assert replay['aggregate'] == aggregate and len(replay['records']) == 72
    text = json.dumps(aggregate)
    for key in ('prompt_token_ids', 'target_token_ids', 'chosen_logits', 'log_normalizers', 'token_logprobs', 'canonical_candidates'):
        assert key not in text
    assert all(row['context_id'] not in text for row in fixture[0]['contexts'])
    assert all(row['evaluation_id'] not in text for row in fixture[0]['evaluations'])


@pytest.mark.parametrize('mode', ['prefix', 'repeat', 'tie', 'reverse'])
def test_numerical_or_functional_failure_never_gates_or_retries(fixture, mode):
    aggregate, calls = execute(fixture, mode=mode)
    assert len(calls) == len(set(calls)) == 288 and aggregate['status'] == 'complete'
    assert aggregate['clean_qualification']['qualified'] is False and aggregate['execution_status'] == 'finished_schedule'
    records = a.export_run(**fixture[1])['records']
    if mode in ('prefix', 'repeat'):
        assert records[0]['numerical_valid'] is False and records[0]['mean_margin_nats'] is None
    else:
        assert records[0]['numerical_valid'] is True and records[0]['mean_margin_nats'] is not None


def test_infrastructure_failure_is_unknown_and_all_remaining_slots_are_attempted(fixture):
    aggregate, calls = execute(fixture, fail_at=1)
    assert len(calls) == 288 and aggregate['status'] == 'incomplete' and aggregate['clean_qualification']['qualified'] is None
    assert aggregate['coverage'] == {'attempted': 288, 'completed': 287, 'infrastructure_failed': 1,
                                     'interrupted': 0, 'unattempted': 0, 'missing': 1}
    row = a.export_run(**fixture[1])['records'][0]
    assert row['completed_evaluations'] == 3 and row['numerical_valid'] is None
    path = fixture[1]['output_root'] / 'evaluations' / calls[1] / 'result.json'
    assert 'confidential' not in path.read_text()


@pytest.mark.parametrize('change', ['nonfinite', 'positive_logprob', 'length', 'normalization', 'suffix', 'eos', 'boolean'])
def test_invalid_forward_measurement_is_infrastructure_not_a_repaired_score(fixture, change):
    def transform(value, index):
        if index != 0:
            return value
        if change == 'nonfinite':
            value['chosen_logits'][0] = float('nan')
        elif change == 'positive_logprob':
            value['chosen_logits'][0] = value['token_logprobs'][0] = .1
        elif change == 'length':
            value['token_logprobs'].pop()
        elif change == 'normalization':
            value['log_normalizers'][0] += 1
        elif change == 'suffix':
            value['scores']['answer_continuation_logprob'] += 1
        elif change == 'eos':
            value['scores']['eos_logprob'] -= .1
        else:
            value['chosen_logits'][0] = True
        return value

    aggregate, calls = execute(fixture, scorer_transform=transform)
    assert len(calls) == 288 and aggregate['coverage']['infrastructure_failed'] == 1 and aggregate['clean_qualification']['qualified'] is None


@pytest.mark.parametrize('tamper', ['native', 'attempt', 'chosen', 'normalizer', 'logprob', 'score', 'numeric_bool',
                                   'orphan', 'order', 'receipt', 'signal', 'records', 'aggregate'])
def test_export_rejects_tampered_bindings_arithmetic_and_coverage(fixture, tamper):
    execute(fixture)
    plan, args = fixture
    root = args['output_root']
    folder = root / 'evaluations' / plan['evaluations'][1]['evaluation_id']
    if tamper == 'orphan':
        (root / 'evaluations' / 'unplanned').mkdir()
    elif tamper == 'order':
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
    else:
        path = {'native': root / 'native-inputs.json', 'attempt': folder / 'attempt.json',
                'chosen': folder / 'result.json', 'normalizer': folder / 'result.json', 'logprob': folder / 'result.json',
                'score': folder / 'result.json', 'numeric_bool': folder / 'result.json',
                'receipt': root / 'execution-finished.json', 'signal': root / 'execution-finished.json',
                'records': root / 'records.json', 'aggregate': root / 'analysis.json'}[tamper]
        value = json.loads(path.read_text())
        if tamper == 'native':
            value[plan['contexts'][0]['context_id']]['model_vocab_size'] += 1
        elif tamper == 'attempt':
            value['evaluation']['repeat_index'] += 1
        elif tamper in ('chosen', 'normalizer', 'logprob'):
            key = {'chosen': 'chosen_logits', 'normalizer': 'log_normalizers', 'logprob': 'token_logprobs'}[tamper]
            value['measurement'][key][0] -= .1
        elif tamper == 'score':
            value['measurement']['scores']['answer_continuation_logprob'] += .5
        elif tamper == 'numeric_bool':
            value['measurement']['scores']['eos_tokens'] = True
        elif tamper == 'receipt':
            value['run_sha256'] = '0' * 64
        elif tamper == 'signal':
            value['interruption'] = {'signal_number': 15, 'signal_name': 'SIGTERM', 'pid': os.getpid(), 'exception_type': 'A181Interrupted'}
        elif tamper == 'records':
            value[0]['functional_passed'] = False
        else:
            value['clean_qualification']['qualified'] = False
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**args)


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
    assert aggregate["coverage"]["unattempted"] == aggregate["coverage"]["missing"] == 288
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
    assert a.export_run(**fixture[1])["aggregate"]["coverage"]["unattempted"] == 288



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
                          a.native.engine.file_digest(Path(__file__)), MATERIALS)
    args["plan_path"].write_bytes(a.tasks.canonical(plan))
    args["plan_sha256"] = a.native.engine.file_digest(args["plan_path"])
    return plan, args



def test_model_metadata_content_must_match_bound_hash(fixture):
    config = json.loads(fixture[1]["config_path"].read_text())
    (Path(config["model_path"]) / "config.json").write_text('{"max_position_embeddings":9999,"vocab_size":512}')
    with pytest.raises(ValueError, match="model_config_binding"):
        prepare(fixture)



@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_preflight_interrupt_preserves_startup_receipt_and_consumes_claim(fixture, monkeypatch, number):
    def interrupted(*args):
        if number is None:
            raise KeyboardInterrupt
        if number == signal.SIGALRM:
            a._deadline(number, None)
        a._interrupt(number, None)

    monkeypatch.setattr(a, "_inputs", interrupted)
    exception = KeyboardInterrupt if number is None else a.A181Deadline if number == signal.SIGALRM else a.A181Interrupted
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
                "coverage": {"attempted": 288, "completed": 288}, "response_text": "synthetic private response"}

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



def test_timer_entry_failure_is_durable_and_cannot_restart(fixture, monkeypatch):
    @contextlib.contextmanager
    def failed_timer():
        raise RuntimeError('synthetic timer entry failure')
        yield

    monkeypatch.setattr(a, 'bounded_signals', failed_timer)
    with pytest.raises(RuntimeError, match='timer entry'):
        a.run_plan(**fixture[1])
    root = fixture[1]['output_root']
    failed = json.loads((root / 'startup-failure.json').read_text())
    startup = json.loads((root / 'startup-attempt.json').read_text())
    assert failed['startup_attempt_sha256'] == a.object_sha(startup)
    assert failed['target_model_calls'] == 0 and failed['status'] == 'failed'
    assert not (root / 'run.json').exists() and a._PROCESS_CONSUMED
    monkeypatch.setattr(a, '_PROCESS_CONSUMED', False)
    with pytest.raises(ValueError, match='consumed_run'):
        a.run_plan(**fixture[1])



def test_signal_after_result_publication_preserves_durable_completed_count(fixture, monkeypatch):
    original_write = a.native.engine.write_private
    triggered = []

    def interrupted_write(path, value):
        original_write(path, value)
        if Path(path).name == 'result.json' and not triggered:
            triggered.append(True)
            a._interrupt(signal.SIGTERM, None)

    monkeypatch.setattr(a.native.engine, 'write_private', interrupted_write)
    aggregate, calls = execute(fixture)
    assert len(calls) == 1 and aggregate['execution_status'] == 'interrupted'
    assert aggregate['coverage'] == {'attempted': 1, 'completed': 1, 'infrastructure_failed': 0,
                                     'interrupted': 0, 'unattempted': 287, 'missing': 287}
    assert a.export_run(**fixture[1])['aggregate'] == aggregate



def test_boolean_signal_number_cannot_impersonate_sighup(fixture):
    execute(fixture, interrupt_at=1, signal_number=signal.SIGHUP)
    path = fixture[1]['output_root'] / 'execution-finished.json'
    receipt = json.loads(path.read_text())
    assert receipt['interruption']['signal_number'] == 1
    receipt['interruption']['signal_number'] = True
    path.write_bytes(a.tasks.canonical(receipt))
    with pytest.raises(ValueError):
        a.export_run(**fixture[1])



@pytest.mark.parametrize('number', [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, None])
def test_runtime_signal_and_deadline_preserve_attempts_and_null_margins(fixture, number):
    aggregate, calls = execute(fixture, interrupt_at=3, signal_number=number)
    assert len(calls) == 4
    assert aggregate['coverage'] == {'attempted': 4, 'completed': 3, 'infrastructure_failed': 0,
                                     'interrupted': 1, 'unattempted': 284, 'missing': 285}
    assert aggregate['clean_qualification']['qualified'] is None and aggregate['complete_contexts'] == 0
    receipt = json.loads((fixture[1]['output_root'] / 'execution-finished.json').read_text())
    assert receipt['status'] == ('deadline' if number == signal.SIGALRM else 'interrupted')
    assert receipt['interruption'] == {
        'signal_number': int(number) if number is not None else None,
        'signal_name': signal.Signals(number).name if number is not None else None,
        'pid': os.getpid(), 'exception_type': 'KeyboardInterrupt' if number is None else
        'A181Deadline' if number == signal.SIGALRM else 'A181Interrupted'}
    replay = a.export_run(**fixture[1])
    assert replay['aggregate'] == aggregate and len(replay['records']) == 72
    assert all(row['mean_margin_nats'] is None and row['margins_unbounded'] for row in replay['records'])


def test_model_context_guard_uses_actual_complete_candidate_path(fixture, monkeypatch):
    entries = prepare(fixture)['native_inputs'].values()
    largest = max(len(value['prompt_token_ids']) + max(value['candidate_lengths'].values()) for value in entries)
    exact = rebind_model_metadata(fixture, monkeypatch, max_position_embeddings=largest)
    assert all(value['model_context_limit'] == largest for value in prepare(exact)['native_inputs'].values())
    short = rebind_model_metadata(exact, monkeypatch, max_position_embeddings=largest - 1)
    with pytest.raises(ValueError, match='model_context_fit'):
        prepare(short)
    assert a._PROCESS_CONSUMED is False


def test_native_vocab_bound_includes_candidates_and_eos(fixture, monkeypatch):
    entries = prepare(fixture)['native_inputs'].values()
    highest = max(token for value in entries for token in value['prompt_token_ids']
                  + list(itertools.chain.from_iterable(value['candidate_token_ids'].values())))
    exact = rebind_model_metadata(fixture, monkeypatch, vocab_size=highest + 1)
    assert len(prepare(exact)['native_inputs']) == 72
    short = rebind_model_metadata(exact, monkeypatch, vocab_size=highest)
    with pytest.raises(ValueError, match='native_token_vocabulary'):
        prepare(short)


def test_equal_candidate_lengths_enforced_after_valid_roundtrips(fixture):
    class UnequalTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if '"answer":"s105"' in text:
                position = result.index(1) if 1 in result else len(result)
                result.insert(position, 6)
            return result

        def decode(self, tokens, **kwargs):
            return super().decode([token for token in tokens if token != 6], **kwargs)

    fixture[1]['tokenizer_loader'] = lambda config: UnequalTokenizer()
    with pytest.raises(ValueError, match='equal_candidate_lengths'):
        prepare(fixture)
    assert not fixture[1]['output_root'].exists()


def test_prefix_ids_must_be_identical_across_all_seventy_two_contexts(fixture):
    class ShiftedTokenizer(Tokenizer):
        syntax = {**Tokenizer.syntax, 17: '{"', 18: 'answer', 19: '":"'}

        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if 's106' in text:
                result = [{7: 17, 8: 18, 9: 19}.get(value, value) for value in result]
            return result

    fixture[1]['tokenizer_loader'] = lambda config: ShiftedTokenizer()
    with pytest.raises(ValueError, match='fixed_common_prefix_boundary'):
        prepare(fixture)


def test_wrong_current_numeric_flags_cannot_be_saved_as_boolean_counts(fixture):
    execute(fixture)
    path = fixture[1]['output_root'] / 'analysis.json'
    value = json.loads(path.read_text())
    value['numerical_valid']['false'] = False
    path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        a.export_run(**fixture[1])



# A181 schedule and native-pair qualification.

def test_exact_nine_conditions_and_288_forward_schedule(fixture):
    plan = fixture[0]
    assert (plan['planned_value_pairs'], plan['planned_world_assignments'], plan['planned_base_contexts'],
            plan['planned_contexts'], plan['planned_candidate_paths'], plan['planned_forward_calls']) == (2, 4, 8, 72, 144, 288)
    assert len(plan['contexts']) == len({c['context_id'] for c in plan['contexts']}) == 72
    assert len(plan['evaluations']) == len({e['evaluation_id'] for e in plan['evaluations']}) == 288
    assert Counter(c['condition'] for c in plan['contexts']) == dict.fromkeys(CONDITIONS, 8)
    assert set(plan['materials']) == {'full', 'sham'} and plan['materials'] == MATERIALS
    assert plan['condition_order_rotations'] == [0, 2, 4, 6]
    schedules = []
    for base in range(8):
        world, selector_position = divmod(base, 2)
        selector = SELECTOR_ORDERS[world][selector_position]
        values = PAIRS[world // 2]
        displayed = values[::-1] if world % 2 else values
        mapping = dict(zip(('A', 'B'), displayed))
        mapping_text = json.dumps(mapping, separators=(',', ':'))
        expected_order = CONDITIONS[2 * world:] + CONDITIONS[:2 * world]
        if selector_position:
            expected_order = expected_order[::-1]
        contexts = plan['contexts'][base * 9:(base + 1) * 9]
        assert [c['condition'] for c in contexts] == list(expected_order)
        schedules.append(expected_order)
        for context in contexts:
            assert context['base_context_index'] == base and context['world_index'] == world
            assert context['pair_index'] == world // 2 and context['assignment_swapped'] is bool(world % 2)
            assert context['presentation_order'] == 'AB' and context['selector'] == selector
            assert context['messages'][0] == {'role': 'system', 'content': ' '.join(
                template.format(selector=selector) for template in (SELECTION, FORMAT, CLOSING))}
            package, user_state = context['condition'].split('_', 1)
            assert (context['package'], context['user_selector_state']) == (package, user_state)
            user_selector = None if user_state == 'none' else selector if user_state == 'concordant' else ('B' if selector == 'A' else 'A')
            user = mapping_text if user_selector is None else mapping_text + '\n\n' + DIRECTIVE.format(user_selector=user_selector)
            if package != 'clean':
                user += '\n\n' + MATERIALS[package]
            assert context['messages'][1] == {'role': 'user', 'content': user}
            assert context['user_selector'] == user_selector
            assert context['scaffold_kind'] == ('none' if package == 'clean' else package)
            assert context['placement'] == ('none' if package == 'clean' else 'after')
            selected = 'value_1' if mapping[selector] == values[0] else 'value_2'
            assert context['selected_candidate'] == selected
            assert context['other_candidate'] == ('value_2' if selected == 'value_1' else 'value_1')
            assert context['selected_answer'] == mapping[selector]
            assert context['unselected_answer'] == mapping['B' if selector == 'A' else 'A']
            assert context['canonical_candidates'] == {
                f'value_{i + 1}': json.dumps({'answer': value}, separators=(',', ':')) for i, value in enumerate(values)}
            ci = context['context_index']
            evaluations = plan['evaluations'][4 * ci:4 * ci + 4]
            expected_candidates = ['value_1', 'value_2', 'value_2', 'value_1'] if world % 2 == 0 else ['value_2', 'value_1', 'value_1', 'value_2']
            assert [e['candidate'] for e in evaluations] == expected_candidates
            assert [e['repeat_index'] for e in evaluations] == [0, 0, 1, 1]
            assert [e['sequence_index'] for e in evaluations] == list(range(4 * ci, 4 * ci + 4))
            assert all(e['context_id'] == context['context_id'] and e['context_index'] == ci for e in evaluations)
    pairs = list(itertools.combinations(CONDITIONS, 2))
    assert len(pairs) == 36
    assert all(sum(order.index(left) < order.index(right) for order in schedules) == 4 for left, right in pairs)
    assert [c['context_index'] for c in plan['contexts']] == list(range(72))
    assert plan['clean_gate_condition'] == 'clean_none' and plan['clean_gate_interpretation_only'] is True
    assert plan['after_only'] is True and plan['paired_user_selector_native_length_required'] is True
    assert plan['retries'] == 0
    assert all(plan[key] is False for key in ('user_selector_changes_oracle', 'noncontrol_functional_filter',
               'resume_allowed', 'heldout_allowed', 'interim_gate', 'generation_allowed', 'warmup_allowed'))


def test_all_24_native_pairs_change_only_the_user_label_with_equal_length(fixture):
    plan = fixture[0]
    prepared = prepare(fixture)['native_inputs']
    cells = {(c['base_context_index'], c['condition']): c for c in plan['contexts']}
    changed_labels = Counter()
    for base, package in itertools.product(range(8), PACKAGES):
        group = [cells[base, f'{package}_{state}'] for state in USER_STATES]
        none, concordant, contradictory = group
        assert all(c['messages'][0] == none['messages'][0] and c['selected_answer'] == none['selected_answer']
                   and c['selected_candidate'] == none['selected_candidate'] for c in group)
        left, right = (c['messages'][1]['content'] for c in (concordant, contradictory))
        differences = [(i, x, y) for i, (x, y) in enumerate(zip(left, right)) if x != y]
        assert len(left) == len(right) and len(differences) == 1
        offset, source, target = differences[0]
        assert left[:offset].endswith('\n\nSelect the entry labeled ')
        assert right[offset + 1:].startswith(' from the mapping.') and {source, target} == {'A', 'B'}
        assert source == none['selector'] and target != none['selector']
        changed_labels[source, target] += 1
        assert len(prepared[concordant['context_id']]['prompt_token_ids']) == len(prepared[contradictory['context_id']]['prompt_token_ids'])
        assert len(prepared[none['context_id']]['prompt_token_ids']) < len(prepared[concordant['context_id']]['prompt_token_ids'])
    assert changed_labels == {('A', 'B'): 12, ('B', 'A'): 12}
    assert a._audit_user_selector_pairs(plan, prepared) == 24


@pytest.mark.parametrize('change,code', [('system', 'paired_system_oracle'), ('oracle', 'paired_system_oracle'),
    ('label', 'paired_label_only_substitution'), ('material_position', 'paired_label_only_substitution'),
    ('metadata', 'paired_user_selector'), ('native_length', 'paired_user_selector_native_length'),
    ('matrix', 'paired_context_matrix')])
def test_paired_audit_rejects_nonlabel_changes(fixture, change, code):
    plan = copy.deepcopy(fixture[0])
    prepared = prepare(fixture)['native_inputs']
    row = next(c for c in plan['contexts'] if c['base_context_index'] == 0 and c['condition'] == 'full_contradictory')
    if change == 'system':
        row['messages'][0]['content'] += ' '
    elif change == 'oracle':
        row['selected_answer'] = row['unselected_answer']
    elif change == 'label':
        row['messages'][1]['content'] = row['messages'][1]['content'].replace('labeled B from', 'labeled A from')
    elif change == 'material_position':
        mapping, directive, material = row['messages'][1]['content'].split('\n\n')
        row['messages'][1]['content'] = '\n\n'.join((mapping, material, directive))
    elif change == 'metadata':
        row['user_selector'] = row['selector']
    elif change == 'native_length':
        prepared[row['context_id']]['prompt_token_ids'].append(6)
    else:
        plan['contexts'].remove(row)
    with pytest.raises(ValueError, match=code):
        a._audit_user_selector_pairs(plan, prepared)


def test_valid_native_roundtrips_with_unequal_user_label_lengths_are_rejected(fixture):
    class UnequalLabelTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if '\n\nSelect the entry labeled B from the mapping.' in text:
                result.insert(0, 6)
            return result

        def decode(self, tokens, **kwargs):
            return super().decode([token for token in tokens if token != 6], **kwargs)

    fixture[1]['tokenizer_loader'] = lambda config: UnequalLabelTokenizer()
    with pytest.raises(ValueError, match='paired_user_selector_native_length'):
        prepare(fixture)
    assert a._PROCESS_CONSUMED is False and not fixture[1]['output_root'].exists()


@pytest.mark.parametrize('inventory,code', [([0, 1, 2, ord('S') + 20], 'payload_special_tokens'),
    ([0, True], 'payload_special_inventory'), (None, 'payload_special_inventory')])
def test_payload_special_tokens_and_malformed_inventory_are_rejected(fixture, inventory, code):
    tokenizer = Tokenizer()
    tokenizer.all_special_ids = inventory
    fixture[1]['tokenizer_loader'] = lambda config: tokenizer
    with pytest.raises(ValueError, match=code):
        prepare(fixture)
    assert a._PROCESS_CONSUMED is False and not fixture[1]['output_root'].exists()


def test_duplicated_complete_native_prompts_are_rejected(fixture, monkeypatch):
    prepare_context = a.prepare_context
    shared_prompt = []

    def duplicate_prepared_prompt(*args, **kwargs):
        prepared = prepare_context(*args, **kwargs)
        if not shared_prompt:
            shared_prompt.extend(prepared['prompt_token_ids'])
        prepared['prompt_token_ids'] = list(shared_prompt)
        return prepared

    # Corrupt the prepared boundary after the separately tested rendering guards.
    monkeypatch.setattr(a, 'prepare_context', duplicate_prepared_prompt)
    with pytest.raises(ValueError, match='unique_native_contexts'):
        prepare(fixture)
    assert a._PROCESS_CONSUMED is False and not fixture[1]['output_root'].exists()


@pytest.mark.parametrize('change', ['extra', 'missing', 'nonstring', 'whitespace', 'unicode', 'exchange', 'bytes_pin', 'digest_pin'])
def test_restricted_material_receipts_reject_public_fixture_tampering(fixture, monkeypatch, change):
    materials = dict(MATERIALS)
    if change == 'extra':
        materials['replacement'] = 'Unplanned material'
    elif change == 'missing':
        materials.pop('full')
    elif change == 'nonstring':
        materials['full'] = 123
    elif change == 'whitespace':
        materials['full'] += '\n'
    elif change == 'unicode':
        materials['full'] = materials['full'].replace(' ', '\u00a0', 1)
    elif change == 'exchange':
        materials['full'], materials['sham'] = materials['sham'], materials['full']
    else:
        receipts = copy.deepcopy(a.MATERIAL_RECEIPTS)
        receipts['full']['bytes' if change == 'bytes_pin' else 'sha256'] = 0 if change == 'bytes_pin' else '0' * 64
        monkeypatch.setattr(a, 'MATERIAL_RECEIPTS', receipts)
    with pytest.raises(ValueError):
        a.compile_plan(a.CONFIG_SHA, 'a' * 64, 'b' * 64, materials)


@pytest.mark.parametrize('change', ['plan_material', 'object_binding', 'source_binding', 'condition', 'placement', 'base_context'])
def test_material_provenance_and_condition_identity_tampering_is_rejected(fixture, change):
    plan = copy.deepcopy(fixture[0])
    if change == 'plan_material':
        plan['materials']['full'] += ' changed'
    elif change == 'object_binding':
        plan['bindings']['materials_object_sha256'] = '0' * 64
    elif change == 'source_binding':
        plan['bindings']['material_source_sha256'] = '0' * 64
    else:
        row = plan['contexts'][1]
        key = {'condition': 'condition', 'placement': 'placement', 'base_context': 'base_context_index'}[change]
        row[key] = {'condition': 'clean_none', 'placement': 'before', 'base_context': 1}[change]
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def signed_fixture(fixture, margin_fn, *, missing_contexts=(), numeric_failed_contexts=()):
    """Generate public nonpositive token scores from independently chosen margins."""
    plan = fixture[0]
    prepared = prepare(fixture)['native_inputs']
    results = {}
    for evaluation in plan['evaluations']:
        ci = evaluation['context_index']
        if ci in missing_contexts:
            continue
        context = plan['contexts'][ci]
        margin = float(margin_fn(context['base_context_index'], context['condition']))
        selected, other = (0., -margin) if margin >= 0. else (margin, 0.)
        suffix = selected if evaluation['candidate'] == context['selected_candidate'] else other
        prefix = -.001 if ci in numeric_failed_contexts and evaluation['repeat_index'] == 1 else 0.
        count = prepared[context['context_id']]['candidate_lengths'][evaluation['candidate']]
        results[evaluation['evaluation_id']] = {'status': 'completed', 'measurement': measurement(count, suffix, prefix)}
    records = a._records(plan, prepared, results)
    return records, a._aggregate(plan, records, results, set(results))


def independent_margin(base, condition):
    index = CONDITIONS.index(condition)
    return Fraction((base + 1) * (index + 2) + (base + 3) ** 2 * (index % 3 - 1) - index ** 3, 64)


def assert_resolved(estimate, expected, contexts, pairs=None, quartets=None):
    assert estimate['planned_contexts'] == estimate['resolved_contexts'] == contexts
    assert estimate['point'] == estimate['lower'] == estimate['upper'] == float(expected)
    assert estimate['unbounded'] is False
    if pairs is not None:
        assert estimate['planned_pairs'] == estimate['resolved_pairs'] == pairs
    if quartets is not None:
        assert estimate['planned_quartets'] == estimate['resolved_quartets'] == quartets


def test_primary_components_secondaries_and_means_match_independent_fraction_arithmetic(fixture):
    records, aggregate = signed_fixture(fixture, independent_margin)
    assert all(row['mean_margin_nats'] == float(independent_margin(row['base_context_index'], row['condition'])) for row in records)
    primary = aggregate['contrasts']['mean_margin_nats']
    assert set(primary) == {'package_by_user_selector_alignment'}
    expected = sum((independent_margin(i, 'full_contradictory') - independent_margin(i, 'full_concordant'))
                   - (independent_margin(i, 'sham_contradictory') - independent_margin(i, 'sham_concordant'))
                   for i in range(8)) / 8
    assert_resolved(primary['package_by_user_selector_alignment'], expected, 32, quartets=8)
    components = aggregate['primary_components']['mean_margin_nats']
    assert set(components) == {'full_contradictory_minus_concordant', 'sham_contradictory_minus_concordant'}
    for package in ('full', 'sham'):
        expected = sum(independent_margin(i, f'{package}_contradictory') - independent_margin(i, f'{package}_concordant') for i in range(8)) / 8
        assert_resolved(components[f'{package}_contradictory_minus_concordant'], expected, 16, pairs=8)
    secondaries = aggregate['secondary_contrasts']['mean_margin_nats']
    assert set(secondaries) == {f'{package}_{state}_minus_none' for package in PACKAGES for state in USER_STATES[1:]}
    for package, state in itertools.product(PACKAGES, USER_STATES[1:]):
        expected = sum(independent_margin(i, f'{package}_{state}') - independent_margin(i, f'{package}_none') for i in range(8)) / 8
        assert_resolved(secondaries[f'{package}_{state}_minus_none'], expected, 16, pairs=8)
    assert set(aggregate['by_condition']) == set(CONDITIONS)
    for condition in CONDITIONS:
        expected = sum(independent_margin(i, condition) for i in range(8)) / 8
        assert_resolved(aggregate['by_condition'][condition], expected, 8)
    plan = fixture[0]
    assert plan['primary_contrasts'] == {'package_by_user_selector_alignment': {
        'condition_coefficients': {'full_contradictory': 1 / 8, 'full_concordant': -1 / 8,
                                   'sham_contradictory': -1 / 8, 'sham_concordant': 1 / 8},
        'planned_contexts': 32, 'planned_quartets': 8}}
    for key, actual in (('primary_components', components), ('secondary_contrasts', secondaries)):
        assert set(plan[key]) == set(actual)
        for definition in plan[key].values():
            assert (definition['planned_contexts'], definition['planned_pairs'], definition['cell_coefficient_magnitude']) == (16, 8, 1 / 8)
    assert plan['condition_mean_cohorts'] == dict.fromkeys(CONDITIONS, {'planned_contexts': 8})


@pytest.mark.parametrize('condition', CONDITIONS)
@pytest.mark.parametrize('failure', ['missing', 'numeric'])
def test_missing_or_invalid_margin_only_nulls_its_prespecified_cohorts(fixture, condition, failure):
    ci = next(c['context_index'] for c in fixture[0]['contexts'] if c['base_context_index'] == 3 and c['condition'] == condition)
    records, aggregate = signed_fixture(fixture, independent_margin,
        **{'missing_contexts' if failure == 'missing' else 'numeric_failed_contexts': (ci,)})
    row = records[ci]
    assert row['mean_margin_nats'] is None and row['margins_unbounded'] is True
    assert row['numerical_valid'] is (None if failure == 'missing' else False)
    cohorts = [('contrasts', 'package_by_user_selector_alignment',
                ('full_contradictory', 'full_concordant', 'sham_contradictory', 'sham_concordant'))]
    cohorts += [('primary_components', f'{package}_contradictory_minus_concordant',
                 (f'{package}_contradictory', f'{package}_concordant')) for package in ('full', 'sham')]
    cohorts += [('secondary_contrasts', f'{package}_{state}_minus_none',
                 (f'{package}_{state}', f'{package}_none')) for package in PACKAGES for state in USER_STATES[1:]]
    for section, name, cohort in cohorts:
        estimate = aggregate[section]['mean_margin_nats'][name]
        affected = condition in cohort
        assert estimate['planned_contexts'] == 8 * len(cohort)
        assert estimate['resolved_contexts'] == 8 * len(cohort) - int(affected)
        group = 'quartets' if section == 'contrasts' else 'pairs'
        assert estimate[f'planned_{group}'] == 8 and estimate[f'resolved_{group}'] == 8 - int(affected)
        assert estimate['unbounded'] is affected
        if affected:
            assert estimate['point'] is estimate['lower'] is estimate['upper'] is None
        else:
            assert estimate['point'] is not None and estimate['point'] == estimate['lower'] == estimate['upper']
    for name, estimate in aggregate['by_condition'].items():
        assert estimate['resolved_contexts'] == 8 - int(name == condition)
        assert estimate['unbounded'] is (name == condition)
        if name == condition:
            assert estimate['point'] is estimate['lower'] is estimate['upper'] is None
    assert len(records) == 72 and aggregate['complete_contexts'] == (71 if failure == 'missing' else 72)


def test_all_none_controls_can_be_missing_without_erasing_primary_or_direct_components(fixture):
    missing = [c['context_index'] for c in fixture[0]['contexts'] if c['user_selector_state'] == 'none']
    records, aggregate = signed_fixture(fixture, independent_margin, missing_contexts=missing)
    assert len(missing) == 24 and sum(r['mean_margin_nats'] is None for r in records) == 24
    assert aggregate['clean_qualification']['qualified'] is None
    primary = aggregate['contrasts']['mean_margin_nats']['package_by_user_selector_alignment']
    assert primary['resolved_contexts'] == 32 and primary['resolved_quartets'] == 8 and primary['point'] is not None
    for estimate in aggregate['primary_components']['mean_margin_nats'].values():
        assert estimate['resolved_contexts'] == 16 and estimate['resolved_pairs'] == 8 and estimate['point'] is not None
    for estimate in aggregate['secondary_contrasts']['mean_margin_nats'].values():
        assert estimate['resolved_contexts'] == 8 and estimate['resolved_pairs'] == 0
        assert estimate['point'] is estimate['lower'] is estimate['upper'] is None and estimate['unbounded'] is True
    assert aggregate['claim_boundaries']['shared_none_observations_per_package'] == 8


def test_two_missing_primary_cells_in_same_quartet_count_unique_contexts_once(fixture):
    missing = [c['context_index'] for c in fixture[0]['contexts'] if c['base_context_index'] == 5
               and c['condition'] in ('full_concordant', 'full_contradictory')]
    _, aggregate = signed_fixture(fixture, independent_margin, missing_contexts=missing)
    primary = aggregate['contrasts']['mean_margin_nats']['package_by_user_selector_alignment']
    component = aggregate['primary_components']['mean_margin_nats']['full_contradictory_minus_concordant']
    assert (primary['resolved_contexts'], primary['resolved_quartets']) == (30, 7)
    assert (component['resolved_contexts'], component['resolved_pairs']) == (14, 7)
    assert primary['point'] is component['point'] is None


@pytest.mark.parametrize('margin', [-2., 0., .0005, .001, .002])
def test_all_64_noncontrol_margins_are_diagnostic_and_never_functionally_filtered(fixture, margin):
    records, aggregate = signed_fixture(fixture, lambda base, condition: .01 if condition == 'clean_none' else margin)
    assert aggregate['clean_qualification']['qualified'] is True
    assert aggregate['numerical_valid'] == {'true': 72, 'false': 0, 'unknown': 0}
    applicable = [row for row in records if row['functional_applicable']]
    assert len(applicable) == 8 and all(row['condition'] == 'clean_none' and row['functional_passed'] is True for row in applicable)
    for row in records:
        if row['condition'] == 'clean_none':
            continue
        assert row['functional_applicable'] is False and row['functional_status'] == 'not_applicable'
        assert row['functional_passed'] is row['positive_low_separation'] is row['threshold_boundary'] is None
        assert row['mean_margin_nats'] == row['worst_margin_nats'] == margin and row['margins_unbounded'] is False
        assert row['mean_reversed'] is row['worst_reversed'] is (margin < 0)
        assert row['mean_tie'] is row['worst_tie'] is (margin == 0)
    for section in ('contrasts', 'primary_components', 'secondary_contrasts'):
        assert all(value['point'] is not None for value in aggregate[section]['mean_margin_nats'].values())


@pytest.mark.parametrize('margin', [-2., 0., .0005, .001])
def test_clean_gate_failure_preserves_all_numeric_margin_estimands(fixture, margin):
    records, aggregate = signed_fixture(fixture, lambda base, condition: margin if condition == 'clean_none' else .01)
    assert aggregate['clean_qualification']['qualified'] is False
    assert aggregate['clean_qualification']['planned_contexts'] == 8
    assert all(row['numerical_valid'] is True and row['mean_margin_nats'] is not None for row in records)
    for section in ('contrasts', 'primary_components', 'secondary_contrasts'):
        assert all(value['point'] is not None and value['unbounded'] is False for value in aggregate[section]['mean_margin_nats'].values())


def test_negative_interaction_does_not_require_negative_changes_or_reversed_margins(fixture):
    values = dict.fromkeys(CONDITIONS, 1.)
    values.update(full_concordant=2., full_contradictory=4., sham_concordant=3., sham_contradictory=8.)
    records, aggregate = signed_fixture(fixture, lambda base, condition: values[condition])
    assert all(row['mean_margin_nats'] > 0 for row in records)
    assert_resolved(aggregate['contrasts']['mean_margin_nats']['package_by_user_selector_alignment'], -3, 32, quartets=8)
    assert_resolved(aggregate['primary_components']['mean_margin_nats']['full_contradictory_minus_concordant'], 2, 16, pairs=8)
    assert_resolved(aggregate['primary_components']['mean_margin_nats']['sham_contradictory_minus_concordant'], 5, 16, pairs=8)


def test_fixed_candidate_preference_reverses_with_system_oracle_not_user_selector(fixture):
    def margin(base, condition):
        world, selector_position = divmod(base, 2)
        selector = SELECTOR_ORDERS[world][selector_position]
        first_label = 'B' if world % 2 else 'A'
        return 2 if selector == first_label else -2

    records, aggregate = signed_fixture(fixture, margin)
    assert Counter(row['mean_margin_nats'] for row in records) == {2.: 36, -2.: 36}
    for row in records:
        assert row['mean_margin_nats'] == (2 if row['selected_candidate'] == 'value_1' else -2)
    assert aggregate['clean_qualification']['qualified'] is False
    assert all(value['point'] == 0 for value in aggregate['by_condition'].values())
    for section in ('contrasts', 'primary_components', 'secondary_contrasts'):
        assert all(value['point'] == 0 for value in aggregate[section]['mean_margin_nats'].values())


def test_full_schedule_keeps_all_negative_noncontrol_outcomes(fixture):
    def transform(value, index):
        evaluation = fixture[0]['evaluations'][index]
        context = fixture[0]['contexts'][evaluation['context_index']]
        if context['condition'] == 'clean_none':
            return value
        suffix = -.02 if evaluation['candidate'] == context['selected_candidate'] else -.01
        return measurement(value['scores']['joint_tokens'], suffix)

    aggregate, calls = execute(fixture, scorer_transform=transform)
    assert len(calls) == len(set(calls)) == 288
    assert aggregate['clean_qualification']['qualified'] is True and aggregate['status'] == 'complete'
    assert aggregate['mean_margin_signs'] == {'positive': 8, 'zero': 0, 'negative': 64, 'unknown': 0}
    assert all(value['point'] is not None for value in aggregate['by_condition'].values())


def test_plan_mutable_materials_receipts_and_coefficients_are_defensive_copies(fixture):
    materials = dict(MATERIALS)
    plan = a.compile_plan(a.CONFIG_SHA, 'a' * 64, 'b' * 64, materials)
    materials['full'] += ' externally changed'
    assert plan['materials'] == MATERIALS
    original_receipts = copy.deepcopy(a.MATERIAL_RECEIPTS)
    original_coefficients = dict(a.PRIMARY_COEFFICIENTS)
    plan['materials']['sham'] += ' changed in plan'
    plan['material_receipts']['full']['bytes'] = -1
    plan['primary_contrasts']['package_by_user_selector_alignment']['condition_coefficients']['full_contradictory'] = -1
    assert a.MATERIAL_RECEIPTS == original_receipts and a.PRIMARY_COEFFICIENTS == original_coefficients
    fresh = a.compile_plan(a.CONFIG_SHA, 'a' * 64, 'b' * 64, MATERIALS)
    assert fresh['materials'] == MATERIALS and fresh['material_receipts'] == original_receipts
    assert fresh['primary_contrasts']['package_by_user_selector_alignment']['condition_coefficients'] == original_coefficients



def test_boolean_interruption_pid_is_rejected(fixture):
    execute(fixture, interrupt_at=1, signal_number=signal.SIGHUP)
    path = fixture[1]['output_root'] / 'execution-finished.json'
    receipt = json.loads(path.read_text())
    receipt['interruption']['pid'] = True
    path.write_bytes(a.tasks.canonical(receipt))
    with pytest.raises(ValueError):
        a.export_run(**fixture[1])
