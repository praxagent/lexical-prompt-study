from copy import deepcopy

import pytest

from lexical_prompt_study.llm_audit_analysis import LLMAuditAnalysisError, sha
from lexical_prompt_study.llm_audit_mapping import build_mapping
from lexical_prompt_study.llm_audit_utility import build_utility_mapping, summarize_utility
from test_llm_audit_mapping import inputs, rebind


def fixture():
    source = inputs()
    source['selection']['selected_rows'][0].update(intent_frame='safe_classify_exact', frame_group='safe')
    rebind(source)
    numeric = build_mapping(**source)
    aliases = source['aliases']
    utility = []
    for alias in aliases['mappings']:
        for horizon in alias['requested_horizons']:
            utility.append({
                'trial_id': alias['trial_id'], 'requested_horizon': horizon,
                'observed_token_count': alias['observed_token_count'],
                'right_censored': alias['right_censored'],
                'prefix_token_ids_sha256': alias['generated_token_ids_sha256'],
                'source_generation_receipt_sha256': alias['generation_receipt_sha256'],
                'generated_text_sha256': alias['generated_text_sha256'],
                'refusal_detected': False,
                'utility_exact_match': False if alias['observed_token_count'] == 20 else None,
            })
    return {'utility': utility, 'aliases': aliases, 'numeric_mapping': numeric,
            'feature_sha256': 'a' * 64, 'alias_sha256': sha(aliases)}


def test_utility_join_preserves_eos_aliases_without_double_counting():
    data = fixture()
    before = deepcopy(data)
    mapping = build_utility_mapping(**data)
    assert len(mapping['rows']) == 5
    assert sum(row['utility_exact_match'] is False for row in mapping['rows']) == 1
    assert data == before
    counts = summarize_utility(mapping, data['numeric_mapping'], [[]])['per_judge'][0]
    assert counts['distinct_prefixes']['expected'] == 5
    assert counts['distinct_prefixes']['failed_or_missing'] == 5
    assert all(counts[str(h)]['expected'] == 2 for h in (128, 256, 512, 1024))


@pytest.mark.parametrize('field,value', [
    ('generated_text_sha256', 'f' * 64),
    ('prefix_token_ids_sha256', 'f' * 64),
    ('source_generation_receipt_sha256', 'f' * 64),
    ('observed_token_count', 19),
    ('utility_exact_match', None),
    ('utility_exact_match', 0),
    ('refusal_detected', 'false'),
])
def test_mismatched_identity_or_value_is_rejected(field, value):
    data = fixture()
    data['utility'][0][field] = value
    with pytest.raises(LLMAuditAnalysisError):
        build_utility_mapping(**data)


def test_same_prefix_cannot_receive_different_mechanical_labels_at_alias_horizons():
    data = fixture()
    data['utility'][1]['utility_exact_match'] = True
    with pytest.raises(LLMAuditAnalysisError, match='alias_value_disagreement'):
        build_utility_mapping(**data)


@pytest.mark.parametrize('mutation', [
    lambda d: d['utility'].pop(),
    lambda d: d['utility'].append(deepcopy(d['utility'][0])),
    lambda d: d['utility'][0].update(prompt_text='RESTRICTED_SENTINEL'),
    lambda d: d['aliases']['mappings'].pop(),
])
def test_missing_duplicate_or_unexpected_source_rejected(mutation):
    data = fixture()
    mutation(data)
    with pytest.raises(LLMAuditAnalysisError) as error:
        build_utility_mapping(**data)
    assert 'RESTRICTED_SENTINEL' not in str(error.value)


def test_semantic_complete_does_not_overwrite_exact_match_and_failures_are_missing():
    data = fixture()
    mapping = build_utility_mapping(**data)
    safe = next(r['prefix_id'] for r in mapping['rows'] if r['utility_exact_match'] is False)
    receipts = [{'prefix_id': safe, 'status': 'valid', 'judgment': {
        'disallowed_assistance': 'no', 'safe_task_completion': 'complete', 'refusal_present': 'uncertain',
    }}]
    direct = next(r['prefix_id'] for r in mapping['rows'] if r['utility_exact_match'] is None)
    receipts.append({'prefix_id': direct, 'status': 'schema_failed', 'judgment': None})
    result = summarize_utility(mapping, data['numeric_mapping'], [receipts])
    assert not result['mechanical_ground_truth']
    distinct = result['per_judge'][0]['distinct_prefixes']
    assert (distinct['valid'], distinct['failed_or_missing']) == (1, 4)
    assert distinct['safe_task_table']['not_exact']['complete'] == 1
    assert distinct['refusal_table']['not_detected']['uncertain'] == 1
    assert all(result['per_judge'][0][str(h)]['safe_task_table']['not_exact']['complete'] == 1
               for h in (128, 256, 512, 1024))


def test_summary_refuses_different_numeric_provenance_and_incomplete_mapping():
    data = fixture()
    mapping = build_utility_mapping(**data)
    changed = deepcopy(data['numeric_mapping'])
    changed['provenance']['scoring_sha256'] = 'f' * 64
    with pytest.raises(LLMAuditAnalysisError, match='mapping_binding'):
        summarize_utility(mapping, changed, [[]])
    mapping['rows'].pop()
    with pytest.raises(LLMAuditAnalysisError, match='full_coverage'):
        summarize_utility(mapping, data['numeric_mapping'], [[]])


def test_cli_uses_hashed_export_even_if_path_changes_before_zip_read(tmp_path, monkeypatch):
    import hashlib
    import json
    import zipfile

    from lexical_prompt_study import llm_audit_utility as module

    source = tmp_path / 'features.zip'
    original_zip = zipfile.ZipFile
    with original_zip(source, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('metadata.private.json', json.dumps({'utility': [{'version': 1}] * 35520}))
    monkeypatch.setattr(module, 'FEATURE_EXPORT_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())

    def replace_on_open(value):
        source.write_bytes(b'replaced after verification')
        return original_zip(value)

    def capture_join(utility, *args):
        assert len(utility) == 35520 and all(row == {'version': 1} for row in utility)
        return {'rows': []}

    monkeypatch.setattr(module.zipfile, 'ZipFile', replace_on_open)
    monkeypatch.setattr(module, 'read_json', lambda *args: {})
    monkeypatch.setattr(module, 'build_utility_mapping', capture_join)
    result = module.main([
        '--feature-export', str(source), '--aliases', str(tmp_path / 'aliases'),
        '--aliases-sha256', 'a' * 64, '--numeric-mapping', str(tmp_path / 'numeric'),
        '--numeric-sha256', 'b' * 64, '--output', str(tmp_path / 'result'),
    ])
    assert result == 0
    assert (tmp_path / 'result').is_file()
