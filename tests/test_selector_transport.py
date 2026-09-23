"""A182 synthetic qualification. No target weights, tokenizer or private data."""
import copy
import itertools
import json
import math
import signal
import struct
import types
from fractions import Fraction

import pytest

from lexical_prompt_study import selector_transport as a


@pytest.fixture
def plan():
    return a.compile_plan(a.CONFIG_SHA, '1' * 64, '2' * 64)


class Tokenizer:
    eos_token_id = 1
    all_special_ids = [0, 1, 2]
    syntax = {7: '{"', 8: 'answer', 9: '":"'}

    def get_chat_template(self):
        return 'public synthetic A182 template'

    def encode(self, text, **kwargs):
        result = []
        while text:
            if text.startswith('<EOS>'):
                result.append(1)
                text = text[5:]
            elif text.startswith('{"answer":"'):
                result.extend((7, 8, 9))
                text = text[11:]
            else:
                result.append(ord(text[0]) + 20)
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        return ''.join('<EOS>' if token == 1 else self.syntax[token] if token in self.syntax else chr(token - 20) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = messages[0]['content'] + '\nUSER\n' + messages[1]['content'] + '\nASSISTANT\n'
        if not kwargs['add_generation_prompt']:
            text += messages[2]['content'] + '<EOS>'
        return self.encode(text) if tokenize else text


def prepared_for(plan):
    tokenizer = Tokenizer()
    config = {'chat_template_sha256': a.tasks.sha(tokenizer.get_chat_template().encode()),
              'max_prompt_tokens': 4096, 'max_new_tokens': 64}
    prepared = {c['context_id']: a.prepare_context(tokenizer, c, config, [1]) for c in plan['contexts']}
    for value in prepared.values():
        value.update(model_context_limit=8192, model_vocab_size=512, hidden_width=4096, model_layers=32,
                     absolute_position=len(value['prompt_token_ids']) + 2)
    return prepared


def measurement(score, length, prefix=-3.):
    logprobs = [prefix / 3] * 3 + [score] + [0.] * (length - 4)
    return {'chosen_logits': logprobs.copy(), 'log_normalizers': [0.] * length,
            'token_logprobs': logprobs, 'scores': a.a169._scores(logprobs, 3)}


def score_for(context, evaluation):
    candidate, arm = evaluation['candidate'], evaluation['arm']
    if context['kind'] == 'donor':
        return -1. if candidate == context['selected_candidate'] else -5.
    if candidate in ('D1', 'D2'):
        return -10.
    expected = context['selected_candidate'] if arm == 'match' else context['other_candidate']
    return -1. if candidate == expected else -5.


def raw_for(context):
    return struct.pack('<4096f', *([float(context['context_index'] + 1)] * 4096))


def evidence(plan, prepared=None):
    prepared = prepared_for(plan) if prepared is None else prepared
    results, captures, attempts = {}, {}, {}
    header = {'public': 'synthetic'}
    for evaluation in plan['evaluations']:
        key = evaluation['evaluation_id']
        context = plan['contexts'][evaluation['context_index']]
        source = a._source_binding(plan, evaluation, results, captures)
        attempt = a._attempt(header, evaluation, prepared, source)
        attempts[key] = attempt
        raw = raw_for(context)
        baseline = evaluation['arm'] in ('baseline', 'no_patch')
        result = {'schema_version': 'a182-result-v1', 'attempt_sha256': a.object_sha(attempt),
            'status': 'completed', 'measurement': measurement(score_for(context, evaluation), len(attempt['target_token_ids'])),
            'capture': a._capture_metadata(attempt, raw) if baseline else None, 'error': None, 'elapsed_seconds': 0.}
        a._validate_result(result, attempt)
        results[key] = result
        if baseline:
            captures[key] = raw
    return prepared, results, captures, attempts


def replay_numeric(plan, results, captures, prepared):
    records = a._records(plan, prepared, results, captures)
    return records, a._aggregate(plan, records, results, set(results), {key for key, value in results.items() if value['status'] != 'dependency_unavailable'})


def replace_score(result, score=None, prefix=None):
    value = result['measurement']
    old = value['scores']
    result['measurement'] = measurement(old['answer_continuation_logprob'] if score is None else score,
        len(value['token_logprobs']), old['json_prefix_logprob'] if prefix is None else prefix)


def test_exact_public_constructor_and_oracles(plan):
    recipients = [('s112', 's113'), ('s113', 's112'), ('s114', 's115'), ('s115', 's114')]
    donors = [('s117', 's116'), ('s116', 's117'), ('s119', 's118'), ('s118', 's119')]
    selectors = [('A', 'B'), ('B', 'A'), ('B', 'A'), ('A', 'B')]
    for index, context in enumerate(plan['contexts']):
        kind = 'donor' if index < 8 else 'recipient'
        world = (index % 8) // 2
        selector = selectors[world][index % 2]
        mapping = dict(zip(('A', 'B'), (donors if kind == 'donor' else recipients)[world]))
        assert context['mapping'] == mapping and context['selector'] == selector and context['kind'] == kind
        assert context['assignment_swapped'] is bool((world % 2) != (kind == 'donor'))
        assert context['selected_answer'] == mapping[selector]
        assert context['messages'][0]['content'] == (
            f"Select the entry labeled {selector} from the user's mapping. "
            f'Return the mapped value of the entry labeled {selector} as exactly one JSON object with the single key "answer" and a string value. '
            'Do not transform the value or include other fields, explanations, or Markdown fences.')
        payload = json.dumps(mapping, separators=(',', ':'))
        if kind == 'recipient':
            payload += '\n\nSelect the entry labeled ' + ('B' if selector == 'A' else 'A') + ' from the mapping.'
        assert context['messages'][1]['content'] == payload
    assert len(plan['contexts']) == 16 and len(plan['evaluations']) == 288
    assert sum(len(c['canonical_candidates']) for c in plan['contexts']) == 48
    assert len({e['evaluation_id'] for e in plan['evaluations']}) == 288


def test_exact_schedule_and_fixed_source_references(plan):
    match_first = 0
    for context in plan['contexts']:
        evaluations = [e for e in plan['evaluations'] if e['context_id'] == context['context_id']]
        world, base = context['world_index'], context['base_context_index']
        names = ('D1', 'D2') if context['kind'] == 'donor' else ('R1', 'R2', 'D1', 'D2')
        unique = names if world % 2 == 0 else names[::-1]
        candidate_order = unique + unique[::-1]
        if context['kind'] == 'donor':
            arms = ('baseline',)
        else:
            patch = ('self', 'match', 'opposite')
            patch = patch[world % 3:] + patch[:world % 3]
            patch = patch if base % 2 == 0 else patch[::-1]
            arms = ('no_patch',) + patch
            match_first += arms.index('match') < arms.index('opposite')
        assert [(e['arm'], e['candidate']) for e in evaluations] == list(itertools.product(arms, candidate_order))
        for arm in arms:
            subset = [e for e in evaluations if e['arm'] == arm]
            for name in names:
                assert [e['repeat_index'] for e in subset if e['candidate'] == name] == [0, 1]
        for e in evaluations:
            if e['source_evaluation_id'] is not None:
                source = next(x for x in plan['evaluations'] if x['evaluation_id'] == e['source_evaluation_id'])
                source_context = plan['contexts'][source['context_index']]
                assert source['repeat_index'] == 0 and source['candidate'] == ('R1' if e['arm'] == 'self' else 'D1')
                assert source['sequence_index'] < e['sequence_index'] and source_context['world_index'] == world
                assert (source_context['selector'] == context['selector']) is (e['arm'] != 'opposite')
    assert match_first == 4
    assert [e['sequence_index'] for e in plan['evaluations']] == list(range(288))


@pytest.mark.parametrize('field,value', [('layer_index', 14), ('hidden_width', 2048), ('prefix_tokens', 4),
    ('planned_forward_calls', 289), ('interim_gate', True), ('padding_allowed', True), ('source_references', {})])
def test_plan_tamper_rejected(plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError):
        a.validate_plan(plan)


def test_48_native_paths_and_q_not_longest_common_prefix(plan):
    prepared = prepared_for(plan)
    assert len(prepared) == 16
    assert sum(len(v['candidate_token_ids']) for v in prepared.values()) == 48
    for value in prepared.values():
        assert value['shared_json_prefix_token_ids'] == [7, 8, 9]
        assert value['shared_json_prefix_text'] == '{"answer":"'
        assert len(set(value['candidate_lengths'].values())) == 1
        assert all(tokens[3] == ord('s') + 20 for tokens in value['candidate_token_ids'].values())
        assert all(tokens[-1] == 1 for tokens in value['candidate_token_ids'].values())


@pytest.mark.parametrize('failure', ['unequal', 'wrong_prefix', 'special_eos', 'bad_closed', 'prefix_shift', 'template', 'bool_eos'])
def test_native_audit_rejects_mutations(plan, failure):
    tokenizer = Tokenizer()
    context = copy.deepcopy(plan['contexts'][8])
    config = {'chat_template_sha256': a.tasks.sha(tokenizer.get_chat_template().encode()), 'max_prompt_tokens': 4096, 'max_new_tokens': 64}
    if failure == 'unequal':
        context['canonical_candidates']['D2'] = '{"answer":"longer"}'
    elif failure == 'wrong_prefix':
        context['canonical_candidates']['D2'] = '{"other":"s117"}'
    elif failure == 'special_eos':
        context['canonical_candidates']['D2'] = '{"answer":"<EOS>"}'
    elif failure == 'template':
        config['chat_template_sha256'] = '0' * 64
    elif failure == 'bool_eos':
        tokenizer.eos_token_id = True
    elif failure == 'bad_closed':
        original = tokenizer.apply_chat_template
        def bad(messages, **kwargs):
            result = original(messages, **kwargs)
            return result + ([2] if kwargs['tokenize'] else 'x') if not kwargs['add_generation_prompt'] else result
        tokenizer.apply_chat_template = bad
    else:
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kwargs: ([10] + original(text, **kwargs)) if text.endswith('s112"}') else original(text, **kwargs)
    with pytest.raises(ValueError):
        a.prepare_context(tokenizer, context, config, [1])


@pytest.fixture
def complete(plan):
    return evidence(plan)


def test_complete_primary_and_qualified_controls(plan, complete):
    prepared, results, captures, _ = complete
    records, aggregate = replay_numeric(plan, results, captures, prepared)
    assert aggregate['primary'] == {'planned_pairs': 8, 'resolved_pairs': 8, 'planned_arm_margins': 16,
        'resolved_arm_margins': 16, 'planned_measurements': 64, 'completed_measurements': 64,
        'numerical_valid': True, 'point': 8., 'lower': 8., 'upper': 8., 'unbounded': False}
    assert aggregate['qualified_transport'] is True and aggregate['apparatus']['qualified'] is True
    assert all(g['qualified'] is True for g in aggregate['functional_gates'].values())
    assert aggregate['coverage']['forward_dispatch_entries'] == 288
    assert all(r['source_eligible'] is True for r in records['donors'] + records['recipients'])


def test_primary_independent_fraction_oracle(plan, complete):
    prepared, results, captures, _ = complete
    exact = Fraction(0)
    for e in plan['evaluations']:
        context = plan['contexts'][e['context_index']]
        if e['arm'] in ('match', 'opposite') and e['candidate'] in ('R1', 'R2'):
            score = -Fraction((e['context_index'] + 1) * (1 if e['candidate'] == context['selected_candidate'] else 3) * (2 if e['arm'] == 'match' else 1), 8)
            replace_score(results[e['evaluation_id']], float(score))
            sign = (1 if e['candidate'] == context['selected_candidate'] else -1) * (1 if e['arm'] == 'match' else -1)
            exact += sign * score / 16
    _, aggregate = replay_numeric(plan, results, captures, prepared)
    assert aggregate['primary']['point'] == float(exact)


@pytest.mark.parametrize('arm,candidate', [('no_patch', 'R1'), ('self', 'R2'), ('match', 'D1'), ('opposite', 'D2')])
def test_unrelated_missing_does_not_erase_primary(plan, complete, arm, candidate):
    prepared, results, captures, _ = complete
    e = next(e for e in plan['evaluations'] if e['arm'] == arm and e['candidate'] == candidate)
    del results[e['evaluation_id']]
    captures.pop(e['evaluation_id'], None)
    _, aggregate = replay_numeric(plan, results, captures, prepared)
    assert aggregate['primary']['point'] == 8.
    assert aggregate['apparatus']['qualified'] is None and aggregate['qualified_transport'] is None


@pytest.mark.parametrize('arm,candidate', itertools.product(('match', 'opposite'), ('R1', 'R2')))
def test_primary_missing_is_unbounded_not_subset(plan, complete, arm, candidate):
    prepared, results, captures, _ = complete
    e = next(e for e in plan['evaluations'] if e['arm'] == arm and e['candidate'] == candidate)
    del results[e['evaluation_id']]
    _, aggregate = replay_numeric(plan, results, captures, prepared)
    primary = aggregate['primary']
    assert primary['point'] is primary['lower'] is primary['upper'] is None
    assert primary['unbounded'] is True and primary['completed_measurements'] == 63
    assert primary['resolved_pairs'] == 7 and primary['resolved_arm_margins'] == 15


def test_known_repeat_failure_dominates_other_unknown(plan, complete):
    prepared, results, captures, _ = complete
    es = [e for e in plan['evaluations'] if e['context_index'] == 8 and e['arm'] == 'match']
    for e in es:
        if e['candidate'] == 'R2':
            del results[e['evaluation_id']]
        elif e['candidate'] == 'R1' and e['repeat_index'] == 1:
            replace_score(results[e['evaluation_id']], -3.)
    records, aggregate = replay_numeric(plan, results, captures, prepared)
    cohort = records['recipients'][0]['arms']['match']['own_value']
    assert cohort['prefix_guard']['passed'] is None and cohort['numerical_valid'] is False
    assert aggregate['primary']['numerical_valid'] is False and aggregate['qualified_transport'] is False


def test_primary_cross_arm_prefix_guard_distinct_from_local_guards(plan, complete):
    prepared, results, captures, _ = complete
    for e in plan['evaluations']:
        if e['context_index'] == 8 and e['arm'] == 'match' and e['candidate'] in ('R1', 'R2'):
            replace_score(results[e['evaluation_id']], prefix=-2.)
    records, aggregate = replay_numeric(plan, results, captures, prepared)
    row = records['recipients'][0]
    assert row['arms']['match']['own_value']['numerical_valid'] is True
    assert row['primary']['numerical_valid'] is False and aggregate['primary']['point'] is None


def test_fourway_donor_content_defeats_interpretation_not_primary(plan, complete):
    prepared, results, captures, _ = complete
    for e in plan['evaluations']:
        if e['arm'] in ('match', 'opposite') and e['candidate'] in ('D1', 'D2'):
            replace_score(results[e['evaluation_id']], 0.)
    _, aggregate = replay_numeric(plan, results, captures, prepared)
    assert aggregate['primary']['point'] == 8. and aggregate['apparatus']['qualified'] is True
    assert aggregate['functional_gates']['recipient_content']['qualified'] is False
    assert aggregate['qualified_transport'] is False


@pytest.mark.parametrize('separation,expected', [(0., False), (-1., False), (0.001, False), (0.0010000001, True), (None, None)])
def test_strict_functional_threshold(separation, expected):
    cohort = {'numerical_valid': True if separation is not None else None}
    result = a._functional(cohort, separation)
    assert result['functional_passed'] is expected
    assert result['threshold_boundary'] is (separation == .001) if separation is not None else result['threshold_boundary'] is None


@pytest.mark.parametrize('bad', [True, 1, 0, 'true'])
def test_tristate_rejects_non_booleans(bad):
    if bad is True:
        assert a._tri([bad, None]) is None
    else:
        with pytest.raises(ValueError):
            a._tri([bad, None])


@pytest.mark.parametrize('change', ['numeric_failure', 'functional_failure', 'missing', 'unequal', 'signed_zero'])
def test_source_eligibility_is_capture_not_score_gate(plan, complete, change):
    _, results, captures, _ = complete
    context = plan['contexts'][0]
    es = a._baseline_evaluations(plan, context['context_id'])
    if change in ('numeric_failure', 'functional_failure'):
        replace_score(results[es[0]['evaluation_id']], -100.)
    elif change == 'missing':
        del results[es[0]['evaluation_id']]
    else:
        for index, e in enumerate(es):
            raw = struct.pack('<4096f', *([(-0. if index else 0.) if change == 'signed_zero' else float(index)] * 4096))
            captures[e['evaluation_id']] = raw
            results[e['evaluation_id']]['capture']['sha256'] = a.tasks.sha(raw)
    expected = None if change == 'missing' else False if change == 'unequal' else True
    assert a._source_eligibility(plan, context['context_id'], results, captures) is expected


@pytest.mark.parametrize('bad', [b'', bytes(16383), struct.pack('<4096f', *([float('nan')] * 4096)),
    struct.pack('<4096f', *([float('inf')] * 4096))])
def test_capture_bytes_reject_shape_or_nonfinite(bad):
    with pytest.raises(ValueError):
        a._capture_values(bad)
@pytest.fixture
def runtime_fixture(plan, tmp_path, monkeypatch):
    prepared = prepared_for(plan)
    tokenizer = Tokenizer()
    config = {'model_path': str(tmp_path / 'model'), 'model_files_sha256': {'config.json': 'f' * 64},
              'seed': 0, 'chat_template_sha256': a.tasks.sha(tokenizer.get_chat_template().encode())}
    sources = a.validate_sources()
    monkeypatch.setattr(a, '_inputs', lambda *args: (plan, config, sources, tokenizer, [1], prepared))
    monkeypatch.setattr(a.native.engine, 'snapshot_manifest', lambda path: config['model_files_sha256'])
    monkeypatch.setattr(a, 'PRIVATE_RUNS_ROOT', tmp_path)
    monkeypatch.setattr(a, '_PROCESS_CONSUMED', False)
    runtime = types.SimpleNamespace(eos_ids=[1], device='cpu', tokenizer=tokenizer)
    reference = types.SimpleNamespace(require_memory=lambda: a.MIN_AVAILABLE_BYTES,
        load_cpu_reference=lambda *args: runtime)
    calls = []
    def runner(runtime, attempt, callback, **kwargs):
        callback()
        e = attempt['evaluation']
        calls.append(e)
        c = plan['contexts'][e['context_index']]
        if e['source_evaluation_id'] is not None:
            source = next(x for x in plan['evaluations'] if x['evaluation_id'] == e['source_evaluation_id'])
            assert kwargs['replacement'] == raw_for(plan['contexts'][source['context_index']])
        else:
            assert kwargs['replacement'] is None
        return measurement(score_for(c, e), len(attempt['target_token_ids'])), raw_for(c)
    kwargs = {'plan_path': tmp_path / 'plan.json', 'plan_sha256': '3' * 64,
        'config_path': tmp_path / 'config.json', 'config_sha256': a.CONFIG_SHA,
        'protocol_path': tmp_path / 'protocol.md', 'reference_script': tmp_path / 'reference.py',
        'output_root': tmp_path / 'execution', 'reference_loader': lambda path: reference, 'measurement_runner': runner}
    return types.SimpleNamespace(plan=plan, prepared=prepared, config=config, kwargs=kwargs,
                                 runner=runner, reference=reference, calls=calls, root=kwargs['output_root'])


def export_fixture(fixture):
    kwargs = {k: v for k, v in fixture.kwargs.items() if k not in ('reference_loader', 'measurement_runner')}
    return a.export_run(**kwargs)


def test_full_schedule_run_export_and_one_shot(runtime_fixture):
    f = runtime_fixture
    aggregate = a.run_plan(**f.kwargs)
    assert len(f.calls) == 288 and aggregate['qualified_transport'] is True
    assert aggregate['coverage'] == {'attempted': 288, 'processed_slots': 288, 'forward_dispatch_entries': 288,
        'completed': 288, 'infrastructure_failed': 0, 'dependency_unavailable': 0, 'interrupted': 0, 'unattempted': 0, 'missing': 0}
    assert export_fixture(f)['aggregate'] == aggregate
    assert len(list((f.root / 'evaluations').glob('*/capture.fp32'))) == 96
    with pytest.raises(ValueError, match='fresh_process'):
        a.run_plan(**f.kwargs)


@pytest.mark.parametrize('failed_kind,blocked', [('donor', 16), ('recipient', 8)])
def test_failed_source_blocks_only_exact_dependencies(runtime_fixture, failed_kind, blocked):
    f = runtime_fixture
    failure = next(e['evaluation_id'] for e in f.plan['evaluations']
                   if f.plan['contexts'][e['context_index']]['kind'] == failed_kind)
    def runner(runtime, attempt, callback, **kwargs):
        result = f.runner(runtime, attempt, callback, **kwargs)
        if attempt['evaluation']['evaluation_id'] == failure:
            raise RuntimeError('synthetic failure after capture before successful readout')
        return result
    aggregate = a.run_plan(**{**f.kwargs, 'measurement_runner': runner})
    assert aggregate['execution_status'] == 'finished_schedule'
    assert len(f.calls) == 288 - blocked
    assert aggregate['coverage']['dependency_unavailable'] == blocked
    assert aggregate['coverage']['infrastructure_failed'] == 1
    assert aggregate['coverage']['completed'] == 287 - blocked
    assert not (f.root / 'evaluations' / failure / 'capture.fp32').exists()
    if failed_kind == 'recipient':
        assert aggregate['primary']['point'] == 8.
    else:
        assert aggregate['primary']['point'] is None
    assert export_fixture(f)['aggregate'] == aggregate


@pytest.mark.parametrize('failure', ['numeric', 'functional', 'unequal_capture'])
def test_source_score_failures_do_not_block_but_unequal_capture_does(runtime_fixture, failure):
    f = runtime_fixture
    def runner(runtime, attempt, callback, **kwargs):
        result, raw = f.runner(runtime, attempt, callback, **kwargs)
        e = attempt['evaluation']
        if e['context_index'] == 0:
            if failure == 'numeric' and e['sequence_index'] == 0:
                result = measurement(-100., len(attempt['target_token_ids']))
            elif failure == 'functional':
                result = measurement(-5. if e['candidate'] == 'D2' else -1., len(attempt['target_token_ids']))
            elif failure == 'unequal_capture' and e['sequence_index'] == 0:
                raw = struct.pack('<4096f', *([99.] * 4096))
        return result, raw
    aggregate = a.run_plan(**{**f.kwargs, 'measurement_runner': runner})
    assert aggregate['coverage']['forward_dispatch_entries'] == (272 if failure == 'unequal_capture' else 288)
    assert aggregate['functional_gates']['donor_competence']['qualified'] is False if failure != 'unequal_capture' else True
    assert aggregate['primary']['point'] is None if failure == 'unequal_capture' else aggregate['primary']['point'] == 8.


@pytest.mark.parametrize('exception', [lambda: a.A182Interrupted(signal.SIGTERM), lambda: a.A182Deadline(signal.SIGALRM), KeyboardInterrupt])
def test_signal_after_dispatch_retains_actual_receipts(runtime_fixture, exception):
    f = runtime_fixture
    def runner(runtime, attempt, callback, **kwargs):
        callback()
        raise exception()
    aggregate = a.run_plan(**{**f.kwargs, 'measurement_runner': runner})
    assert aggregate['coverage']['attempted'] == aggregate['coverage']['forward_dispatch_entries'] == 1
    assert aggregate['coverage']['interrupted'] == 1 and aggregate['coverage']['completed'] == 0
    terminal = json.loads((f.root / 'execution-finished.json').read_text())
    assert type(terminal['interruption']['pid']) is int
    if terminal['interruption']['signal_number'] is not None:
        assert type(terminal['interruption']['signal_number']) is int
    assert export_fixture(f)['aggregate'] == aggregate


@pytest.mark.parametrize('boundary', ['capture', 'result'])
def test_signal_after_durable_publication_reconciles_without_reuse(runtime_fixture, monkeypatch, boundary):
    f = runtime_fixture
    if boundary == 'capture':
        original = a._write_capture
        def write(path, raw):
            original(path, raw)
            raise a.A182Interrupted(signal.SIGTERM)
        monkeypatch.setattr(a, '_write_capture', write)
    else:
        original = a.native.engine.write_private
        def write(path, value):
            original(path, value)
            if path.name == 'result.json':
                raise a.A182Interrupted(signal.SIGTERM)
        monkeypatch.setattr(a.native.engine, 'write_private', write)
    aggregate = a.run_plan(**f.kwargs)
    assert aggregate['coverage']['completed'] == (0 if boundary == 'capture' else 1)
    assert aggregate['coverage']['forward_dispatch_entries'] == 1
    assert aggregate['apparatus']['source_eligible']['true'] == 0
    assert export_fixture(f)['aggregate'] == aggregate


@pytest.mark.parametrize('failure', ['inputs', 'memory', 'timer', 'load'])
def test_startup_or_model_load_failure_is_durable(runtime_fixture, monkeypatch, failure):
    f = runtime_fixture
    def fail(*args):
        raise RuntimeError('public synthetic setup failure')
    if failure == 'inputs':
        monkeypatch.setattr(a, '_inputs', fail)
    elif failure == 'memory':
        f.reference.require_memory = lambda: a.MIN_AVAILABLE_BYTES - 1
    elif failure == 'timer':
        monkeypatch.setattr(a, 'bounded_signals', fail)
    else:
        f.reference.load_cpu_reference = fail
    if failure == 'load':
        aggregate = a.run_plan(**f.kwargs)
        assert aggregate['execution_status'] == 'failed' and aggregate['coverage']['attempted'] == 0
    else:
        with pytest.raises((ValueError, RuntimeError)):
            a.run_plan(**f.kwargs)
        receipt = json.loads((f.root / 'startup-failure.json').read_text())
        assert receipt['target_model_calls'] == 0
        assert (f.root / 'startup-attempt.json').exists()
    with pytest.raises(ValueError, match='fresh_process'):
        a.run_plan(**f.kwargs)


def test_system_exit_cleanup_retains_terminal_attempt(runtime_fixture):
    f = runtime_fixture
    def runner(runtime, attempt, callback, **kwargs):
        callback()
        raise SystemExit(7)
    with pytest.raises(SystemExit) as exc:
        a.run_plan(**{**f.kwargs, 'measurement_runner': runner})
    assert exc.value.code == 7
    assert (f.root / 'execution-finished.json').exists()
    aggregate = export_fixture(f)['aggregate']
    assert aggregate['execution_status'] == 'failed' and aggregate['coverage']['interrupted'] == 1


@pytest.mark.parametrize('mutation', ['measurement', 'capture', 'source', 'invocation', 'bool_pid', 'unplanned', 'primary'])
def test_export_tamper_rejected(runtime_fixture, mutation):
    f = runtime_fixture
    a.run_plan(**f.kwargs)
    first = f.root / 'evaluations' / f.plan['evaluations'][0]['evaluation_id']
    if mutation == 'capture':
        path = first / 'capture.fp32'
        path.write_bytes(bytes(16384))
    elif mutation == 'unplanned':
        (f.root / 'evaluations' / 'unexpected').mkdir()
    else:
        path = first / ('invocation.json' if mutation == 'invocation' else 'result.json')
        if mutation == 'source':
            e = next(e for e in f.plan['evaluations'] if e['arm'] == 'self')
            path = f.root / 'evaluations' / e['evaluation_id'] / 'attempt.json'
        elif mutation == 'bool_pid':
            path = f.root / 'run.json'
        elif mutation == 'primary':
            path = f.root / 'analysis.json'
        value = json.loads(path.read_text())
        if mutation == 'measurement':
            value['measurement']['token_logprobs'][0] += 1
        elif mutation == 'source':
            value['source']['eligibility'] = None
        elif mutation == 'invocation':
            value['model_forward_entered'] = 1
        elif mutation == 'bool_pid':
            value['pid'] = True
        else:
            value['primary']['point'] += 1
        path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError):
        export_fixture(f)
@pytest.fixture
def tiny_runtime(monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(a, 'HIDDEN_WIDTH', 16)
    monkeypatch.setattr(a, 'MODEL_LAYERS', 4)
    monkeypatch.setattr(a, 'LAYER_INDEX', 1)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng():
        torch.manual_seed(20260923)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, attention_dropout=0., use_cache=False)
        config._attn_implementation = 'sdpa'
        model = LlamaForCausalLM(config).to(device='cpu', dtype=torch.float32).eval()
    runtime = types.SimpleNamespace(model=model, torch=torch, device='cpu')
    try:
        yield runtime
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)


def tiny_attempt(target=None):
    return {'prompt_token_ids': [2, 3, 4, 5], 'target_token_ids': target or [6, 7, 8, 9, 10, 1]}


def test_actual_measure_dispatch_capture_self_and_causal_prefix(tiny_runtime):
    runtime, attempt = tiny_runtime, tiny_attempt()
    baseline = a.teacher_force(runtime.model, runtime.torch, attempt['prompt_token_ids'], attempt['target_token_ids'], 3)
    entries = []
    captured, raw = a.measure(runtime, attempt, lambda: entries.append('entered'))
    identity, identity_raw = a.measure(runtime, attempt, lambda: entries.append('entered'), replacement=raw)
    assert entries == ['entered', 'entered']
    assert baseline == captured == identity and raw == identity_raw
    vector = a._capture_values(raw)
    changed = struct.pack('<16f', *(v + (0.5 if i % 2 else -0.5) for i, v in enumerate(vector)))
    patched, prepatch_capture = a.measure(runtime, attempt, lambda: entries.append('entered'), replacement=changed)
    assert prepatch_capture == raw
    assert patched['token_logprobs'][:3] == baseline['token_logprobs'][:3]
    assert patched['token_logprobs'][3:] != baseline['token_logprobs'][3:]
    assert a.teacher_force(runtime.model, runtime.torch, attempt['prompt_token_ids'], attempt['target_token_ids'], 3) == baseline
    a._hook_audit(runtime.model, runtime.torch)


def test_actual_measure_four_future_paths_share_exact_capture(tiny_runtime):
    captures, entries = [], []
    for pair in ((9, 10), (11, 12), (13, 14), (15, 16)):
        for _ in range(2):
            _, raw = a.measure(tiny_runtime, tiny_attempt([6, 7, 8, *pair, 1]), lambda: entries.append(1))
            captures.append(raw)
    assert len(entries) == 8 and len(set(captures)) == 1


@pytest.mark.parametrize('failure', ['callback', 'readout', 'system_exit', 'keyboard', 'foreign_signal'])
def test_actual_measure_baseexception_cleanup_and_identity(tiny_runtime, failure):
    runtime, attempt = tiny_runtime, tiny_attempt()
    foreign = []
    original = SystemExit(9) if failure == 'system_exit' else KeyboardInterrupt() if failure == 'keyboard' else a.A182Interrupted(signal.SIGTERM) if failure == 'foreign_signal' else RuntimeError('synthetic')
    def scorer(*args):
        result = a.teacher_force(*args)
        if failure == 'foreign_signal':
            foreign.append(runtime.model.model.layers[0].register_forward_hook(lambda *args: None))
        if failure != 'callback':
            raise original
        return result
    def callback():
        if failure == 'callback':
            raise original
    try:
        with pytest.raises(BaseException) as exc:
            a.measure(runtime, attempt, callback, forward_scorer=scorer)
        assert exc.value is original
        assert not runtime.model._forward_pre_hooks and not runtime.model.model.layers[1]._forward_hooks
    finally:
        for handle in foreign:
            handle.remove()
    a._hook_audit(runtime.model, runtime.torch)


@pytest.mark.parametrize('location', ['model_pre', 'other_block', 'global'])
def test_actual_measure_rejects_preexisting_model_or_global_hooks(tiny_runtime, location):
    runtime = tiny_runtime
    if location == 'model_pre':
        handle = runtime.model.register_forward_pre_hook(lambda *args: None)
    elif location == 'other_block':
        handle = runtime.model.model.layers[3].register_forward_hook(lambda *args: None)
    else:
        handle = runtime.torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    entries = []
    try:
        with pytest.raises(ValueError):
            a.measure(runtime, tiny_attempt(), lambda: entries.append(1))
        assert not entries
    finally:
        handle.remove()
    a._hook_audit(runtime.model, runtime.torch)


def test_actual_measure_rejects_repeat_dispatch(tiny_runtime):
    entries = []
    def twice(*args):
        a.teacher_force(*args)
        return a.teacher_force(*args)
    with pytest.raises(ValueError, match='dispatch_count'):
        a.measure(tiny_runtime, tiny_attempt(), lambda: entries.append(1), forward_scorer=twice)
    assert entries == [1]
    a._hook_audit(tiny_runtime.model, tiny_runtime.torch)


def test_actual_measure_never_uses_final_eos_next_logit(tiny_runtime):
    runtime = tiny_runtime
    original = runtime.model.forward
    def with_bad_last(*args, **kwargs):
        output = original(*args, **kwargs)
        output.logits[:, -1, :] = float('nan')
        return output
    runtime.model.forward = with_bad_last
    try:
        result, _ = a.measure(runtime, tiny_attempt(), lambda: None)
        assert all(math.isfinite(v) for v in result['token_logprobs']) and len(result['token_logprobs']) == 6
        assert result['scores']['eos_logprob'] == result['token_logprobs'][-1]
    finally:
        runtime.model.forward = original
@pytest.mark.parametrize('point', [-8., 0., 8.])
def test_primary_negative_zero_positive_kept_without_relabeling(plan, complete, point):
    prepared, results, captures, _ = complete
    for e in plan['evaluations']:
        if e['arm'] == 'match' and e['candidate'] in ('R1', 'R2'):
            c = plan['contexts'][e['context_index']]
            score = -5. + (point - 4.) if e['candidate'] == c['selected_candidate'] else -5.
            replace_score(results[e['evaluation_id']], score)
    _, aggregate = replay_numeric(plan, results, captures, prepared)
    assert aggregate['primary']['point'] == point and aggregate['primary']['unbounded'] is False
    assert aggregate['qualified_transport'] is (point > 0)


def test_full_prefix_unknown_even_when_available_values_differ(plan, complete):
    prepared, results, captures, _ = complete
    changed = next(e for e in plan['evaluations'] if e['context_index'] == 8 and e['arm'] == 'self' and e['candidate'] == 'D1')
    missing = next(e for e in plan['evaluations'] if e['context_index'] == 8 and e['arm'] == 'no_patch' and e['candidate'] == 'D2')
    replace_score(results[changed['evaluation_id']], prefix=-30.)
    del results[missing['evaluation_id']]
    captures.pop(missing['evaluation_id'])
    records, aggregate = replay_numeric(plan, results, captures, prepared)
    assert records['recipients'][0]['full_cross_arm_prefix_guard']['passed'] is None
    assert aggregate['primary']['point'] == 8.


def test_self_known_drift_failure_dominates_missing_comparison(plan, complete):
    prepared, results, captures, _ = complete
    changed = next(e for e in plan['evaluations'] if e['context_index'] == 8 and e['arm'] == 'self' and e['candidate'] == 'D1')
    missing = next(e for e in plan['evaluations'] if e['context_index'] == 8 and e['arm'] == 'self' and e['candidate'] == 'D2')
    replace_score(results[changed['evaluation_id']], -20.)
    del results[missing['evaluation_id']]
    records, aggregate = replay_numeric(plan, results, captures, prepared)
    assert records['recipients'][0]['self_no_patch']['passed'] is False
    assert aggregate['apparatus']['qualified'] is False and aggregate['primary']['point'] == 8.
    assert aggregate['qualified_transport'] is False


def test_four_path_guard_failure_does_not_null_own_primary(plan, complete):
    prepared, results, captures, _ = complete
    e = next(e for e in plan['evaluations'] if e['context_index'] == 8 and e['arm'] == 'match' and e['candidate'] == 'D1')
    replace_score(results[e['evaluation_id']], -15.)
    records, aggregate = replay_numeric(plan, results, captures, prepared)
    row = records['recipients'][0]['arms']['match']
    assert row['four_path']['numerical_valid'] is False and row['recipient_content']['functional_passed'] is None
    assert row['own_value']['numerical_valid'] is True and aggregate['primary']['point'] == 8.
    assert aggregate['functional_gates']['recipient_content']['qualified'] is False


def test_capture_known_inequality_dominates_missing_observation(plan, complete):
    _, results, captures, _ = complete
    es = a._baseline_evaluations(plan, plan['contexts'][0]['context_id'])
    raw = struct.pack('<4096f', *([99.] * 4096))
    captures[es[0]['evaluation_id']] = raw
    results[es[0]['evaluation_id']]['capture']['sha256'] = a.tasks.sha(raw)
    del results[es[-1]['evaluation_id']]
    assert a._source_eligibility(plan, plan['contexts'][0]['context_id'], results, captures) is False


@pytest.mark.parametrize('bad_field', ['chosen_logits', 'log_normalizers', 'token_logprobs', 'scores'])
def test_invalid_readout_never_publishes_capture(runtime_fixture, bad_field):
    f = runtime_fixture
    first = f.plan['evaluations'][0]['evaluation_id']
    def runner(runtime, attempt, callback, **kwargs):
        result, raw = f.runner(runtime, attempt, callback, **kwargs)
        if attempt['evaluation']['evaluation_id'] == first:
            if bad_field == 'scores':
                result[bad_field]['eos_logprob'] = 100.
            else:
                result[bad_field][0] = float('nan')
        return result, raw
    aggregate = a.run_plan(**{**f.kwargs, 'measurement_runner': runner})
    assert aggregate['coverage']['infrastructure_failed'] == 1 and aggregate['coverage']['dependency_unavailable'] == 16
    assert not (f.root / 'evaluations' / first / 'capture.fp32').exists()


@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM])
def test_preheader_signal_receipt(runtime_fixture, monkeypatch, signum):
    f = runtime_fixture
    exc = a.A182Deadline(signum) if signum == signal.SIGALRM else a.A182Interrupted(signum)
    def fail(*args):
        raise exc
    monkeypatch.setattr(a, '_inputs', fail)
    with pytest.raises(a.A182Interrupted) as caught:
        a.run_plan(**f.kwargs)
    assert caught.value is exc
    receipt = json.loads((f.root / 'startup-failure.json').read_text())
    assert receipt['interruption']['signal_number'] == signum
    assert receipt['interruption']['signal_name'] == signal.Signals(signum).name
    assert type(receipt['interruption']['pid']) is int and receipt['target_model_calls'] == 0


@pytest.mark.parametrize('field,value', [('hidden_size', 2048), ('hidden_size', True), ('num_hidden_layers', 31),
    ('num_hidden_layers', True), ('max_position_embeddings', 1), ('max_position_embeddings', True), ('vocab_size', 20)])
def test_inputs_bind_fixed_model_dimensions(plan, tmp_path, monkeypatch, field, value):
    tokenizer = Tokenizer()
    metadata = {'hidden_size': 4096, 'num_hidden_layers': 32, 'max_position_embeddings': 8192, 'vocab_size': 512}
    metadata[field] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(metadata))
    config = {'model_path': str(tmp_path), 'model_files_sha256': {'config.json': a.native.engine.file_digest(path)},
        'cpu_threads': 4, 'attention_implementation': 'sdpa', 'max_prompt_tokens': 4096, 'max_new_tokens': 64,
        'chat_template_sha256': a.tasks.sha(tokenizer.get_chat_template().encode())}
    monkeypatch.setattr(a.native, '_read_bound', lambda p, digest: plan if p.name == 'plan' else config)
    monkeypatch.setattr(a.native.engine, 'validate_config', lambda config: None)
    monkeypatch.setattr(a.native, 'pinned_eos_ids', lambda config: [1])
    protocol, reference = tmp_path / 'protocol', tmp_path / 'reference'
    original = a.native.engine.file_digest
    monkeypatch.setattr(a.native.engine, 'file_digest', lambda p: plan['bindings']['protocol_sha256'] if p == protocol else a.REFERENCE_SHA if p == reference else original(p))
    with pytest.raises(ValueError):
        a._inputs(tmp_path / 'plan', '3' * 64, tmp_path / 'config', a.CONFIG_SHA, protocol, reference, lambda config: tokenizer)
def test_blocked_source_cannot_masquerade_as_infrastructure_failure(plan, complete):
    _, results, _, attempts = complete
    key = next(e['evaluation_id'] for e in plan['evaluations'] if e['arm'] == 'self')
    attempt = copy.deepcopy(attempts[key])
    attempt['source']['eligibility'] = False
    result = copy.deepcopy(results[key])
    result.update(status='infrastructure_failed', measurement=None, capture=None,
        attempt_sha256=a.object_sha(attempt), error={'exception_type': 'ValueError', 'frames': [], 'code': 'synthetic'})
    with pytest.raises(ValueError, match='blocked_source_status'):
        a._validate_result(result, attempt)
    result.update(status='dependency_unavailable', error=None)
    a._validate_result(result, attempt)


def test_boolean_signal_number_is_not_sighup(runtime_fixture):
    f = runtime_fixture
    def runner(runtime, attempt, callback, **kwargs):
        callback()
        raise a.A182Interrupted(signal.SIGHUP)
    a.run_plan(**{**f.kwargs, 'measurement_runner': runner})
    path = f.root / 'execution-finished.json'
    value = json.loads(path.read_text())
    value['interruption']['signal_number'] = True
    path.write_bytes(a.tasks.canonical(value))
    with pytest.raises(ValueError, match='actual_signal_receipt'):
        export_fixture(f)


def test_claim_on_disk_prevents_fresh_process_reuse(runtime_fixture, monkeypatch):
    f = runtime_fixture
    a.run_plan(**f.kwargs)
    monkeypatch.setattr(a, '_PROCESS_CONSUMED', False)
    with pytest.raises(ValueError, match='consumed_run'):
        a.run_plan(**f.kwargs)


def test_cli_prints_only_safe_failure_type(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(a, 'PRIVATE_RUNS_ROOT', tmp_path)
    def fail(**kwargs):
        print('private synthetic stdout marker')
        raise ValueError('private synthetic exception marker')
    monkeypatch.setattr(a, 'run_plan', fail)
    args = ['--plan', str(tmp_path / 'plan'), '--plan-sha256', '1' * 64, '--config', str(tmp_path / 'config'),
        '--config-sha256', '2' * 64, '--protocol', str(tmp_path / 'protocol'), '--reference-script', str(tmp_path / 'reference'),
        '--output-root', str(tmp_path / 'execution')]
    assert a.main(args) == 1
    assert json.loads(capsys.readouterr().out) == {'status': 'a182_failed', 'error_type': 'ValueError'}
    assert 'private synthetic stdout marker' in (tmp_path / 'execution' / 'execution.log').read_text()
