"""Invented filesystem controls: never read a saved research response."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lexical_prompt_study.gpu3b_arithmetic_independent_audit import reconstruct_roster


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "response_reader", ROOT / "scripts/diagnose_a189_completed_responses.py"
)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return sha(raw)


@pytest.fixture
def synthetic(tmp_path):
    source, fresh = tmp_path / "sealed", tmp_path / "analysis"
    fresh.mkdir()
    rows = reconstruct_roster()
    inventory = {}

    def sealed(relative, value):
        inventory[relative] = write(source / relative, value)

    sealed("prepared/freeze.json", {"plan": {"rows": rows}})
    records = []
    for index, row in enumerate(rows):
        records.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "observation_id",
                        "sequence_index",
                        "core_index",
                        "item_count",
                        "magnitude",
                    )
                },
                "status": "completed" if index < 5 else "unattempted",
                "generation_status": "completed" if index < 5 else "missing",
                "label": 0 if index < 5 else None,
                "terminal_eos": True if index < 5 else None,
            }
        )
        if index < 5:
            base = f"run-001/worker/acquisition/rows/{index:03d}/generation/"
            sealed(base + "result.json", {"status": "completed", "stop_reason": "eos"})
            sealed(base + "response.utf8", b"invented prose with no extracted answer")
    sealed("run-001/worker/acquisition/failure.json", {"records": records, "accepted_rows": 5})
    # Unused placeholders model a sealed whole-run inventory, not response data.
    for index in range(1102 - len(inventory)):
        inventory[f"unused-fixture-{index}"] = "0" * 64
    inv_sha = write(tmp_path / "inventory.json", inventory)
    ver_sha = write(
        tmp_path / "verification.json",
        {
            "status": "verified_incomplete_resource_stop",
            "terminal_verification_closed": True,
            "complete_rows": 5,
            "independent_outcomes": {
                "overall": {"labels": {"success": 0, "failure": 5, "unknown": 27}}
            },
        },
    )
    receipt_sha = write(
        tmp_path / "archive.json",
        {
            "status": "archive_verified",
            "source_unchanged": True,
            "every_member_content_verified": True,
            "unique_members_verified": True,
            "source_inventory_sha256": inv_sha,
            "verification_sha256": ver_sha,
        },
    )
    classifier = ROOT / "src/lexical_prompt_study/arithmetic_response_diagnosis.py"
    freeze = {
        "schema": "a190-five-completed-diagnosis-v1",
        "output_path": str(fresh / "output"),
        "sealed_source_root": str(source),
        "inventory_path": str(tmp_path / "inventory.json"),
        "inventory_sha256": inv_sha,
        "verification_path": str(tmp_path / "verification.json"),
        "verification_sha256": ver_sha,
        "archive_receipt_path": str(tmp_path / "archive.json"),
        "archive_receipt_sha256": receipt_sha,
        "classifier_path": str(classifier),
        "classifier_sha256": sha(classifier.read_bytes()),
        "runner_sha256": sha(Path(reader.__file__).read_bytes()),
    }
    for key in ("protocol", "qualification", "source_review", "scientific_review", "code_backup"):
        path = fresh / (key + ".json")
        freeze[key + "_path"] = str(path)
        freeze[key + "_sha256"] = write(path, {"invented_fixture": True})
    freeze_path = fresh / "freeze.json"
    freeze_sha = write(freeze_path, freeze)
    return source, fresh, freeze_path, freeze_sha


def test_exact_five_read_once_and_aggregate_only(synthetic):
    source, fresh, freeze_path, frozen_sha = synthetic
    before = {str(p): sha(p.read_bytes()) for p in source.rglob("*") if p.is_file()}
    result = reader.run(freeze_path, frozen_sha, fresh / "output")
    assert result["aggregates"] == {
        "completed_responses": 5,
        "category_counts": {
            "parseable_wrong_values": 0,
            "wrong_count": 0,
            "wrapper_correct_values": 0,
            "wrapper_wrong_values": 0,
            "unresolved": 5,
        },
        "wrong_count_style_counts": {"bare": 0, "bracketed": 0, "fenced": 0},
    }
    assert result["partial_response_reads"] == result["model_tokenizer_encoder_fit_calls"] == 0
    assert result["all_read_artifacts_unchanged"] is True
    assert "invented prose" not in json.dumps(result)
    assert before == {str(p): sha(p.read_bytes()) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(reader.EvidenceError, match="analysis_already_claimed"):
        reader.run(freeze_path, frozen_sha, fresh / "output")


def test_changed_response_is_failed_without_qualified_result(synthetic):
    source, fresh, freeze_path, frozen_sha = synthetic
    (source / "run-001/worker/acquisition/rows/000/generation/response.utf8").write_bytes(
        b"changed private fixture"
    )
    with pytest.raises(reader.EvidenceError, match="artifact_hash_mismatch"):
        reader.run(freeze_path, frozen_sha, fresh / "output")
    assert not (fresh / "output/RESULT.json").exists()
    assert "changed private" not in (fresh / "output/FAILURE.json").read_text()


def test_wrong_freeze_digest_never_claims(synthetic):
    _, fresh, freeze_path, _ = synthetic
    with pytest.raises(reader.EvidenceError, match="artifact_hash_mismatch"):
        reader.run(freeze_path, "0" * 64, fresh / "output")
    assert not (fresh / "output").exists()


def test_alternate_output_cannot_replay_frozen_analysis(synthetic):
    _, fresh, freeze_path, frozen_sha = synthetic
    reader.run(freeze_path, frozen_sha, fresh / "output")
    with pytest.raises(reader.EvidenceError, match="sole_frozen_output"):
        reader.run(freeze_path, frozen_sha, fresh / "output-002")
    assert not (fresh / "output-002").exists()


def test_aggregate_rejects_unallowlisted_or_invalid_fields():
    value = {
        "completed_responses": 5,
        "category_counts": {
            "parseable_wrong_values": 0,
            "wrong_count": 0,
            "wrapper_correct_values": 0,
            "wrapper_wrong_values": 0,
            "unresolved": 5,
        },
        "wrong_count_style_counts": {"bare": 0, "bracketed": 0, "fenced": 0},
    }
    reader.check_aggregates(value)
    for key, bad in (
        ("private_response", "secret"),
        ("completed_responses", True),
        ("category_counts", {}),
    ):
        changed = {**value, key: bad}
        with pytest.raises(reader.EvidenceError):
            reader.check_aggregates(changed)
    value["wrong_count_style_counts"]["bare"] = 1
    with pytest.raises(reader.EvidenceError, match="aggregate_style_total"):
        reader.check_aggregates(value)


def test_output_cannot_enter_source(synthetic):
    source, fresh, freeze_path, frozen_sha = synthetic
    with pytest.raises(reader.EvidenceError):
        reader.run(freeze_path, frozen_sha, source / "new-output")
    assert not (source / "new-output").exists()


def test_symlinked_source_file_rejected(synthetic):
    source, fresh, freeze_path, frozen_sha = synthetic
    response = source / "run-001/worker/acquisition/rows/000/generation/response.utf8"
    other = fresh / "copy"
    other.write_bytes(response.read_bytes())
    response.unlink()
    response.symlink_to(other)
    with pytest.raises(reader.EvidenceError, match="regular_file_required"):
        reader.run(freeze_path, frozen_sha, fresh / "output")


def test_duplicate_and_nonfinite_json_rejected():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e9999}'):
        with pytest.raises(ValueError):
            reader.json_value(raw)


def test_model_import_guard():
    with pytest.raises(reader.EvidenceError, match="model_or_numerical_import_forbidden"):
        reader.NoModelImports().find_spec("torch")


def test_independent_expected_answers_agree_with_public_input_constructor():
    for row in reconstruct_roster():
        assert reader.expected_answers(row) == row["expected_sums"]
    row = reconstruct_roster()[0]
    row["expected_sums"][0] += 1
    with pytest.raises(reader.EvidenceError, match="independent_answer_binding"):
        reader.expected_answers(row)
