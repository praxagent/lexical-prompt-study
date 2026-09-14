"""Check publication hazards in Git objects, including deleted history."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_git_payload.py"
SPEC = importlib.util.spec_from_file_location("git_payload", SCRIPT)
payload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(payload)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    payload.git("config", "user.name", "Payload Test")
    payload.git("config", "user.email", "test@example.invalid")
    payload.git("commit", "--allow-empty", "-qm", "baseline")
    base = payload.git("rev-parse", "HEAD").decode().strip()
    return tmp_path, base


def test_reads_staged_bytes_not_clean_working_copy(repository):
    root, _ = repository
    path = root / "config.py"
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    path.write_text(marker)
    payload.git("add", "config.py")
    path.write_text("# harmless working copy\n")
    assert payload.audit(payload.staged_entries()) == [
        ("config.py", "possible credential or private key")
    ]


def test_removed_private_file_still_blocks_outgoing_history(repository):
    root, base = repository
    (root / "raw").mkdir()
    (root / "raw" / "sample.txt").write_text("synthetic data")
    payload.git("add", "raw/sample.txt")
    payload.git("commit", "-qm", "accidental data")
    payload.git("rm", "-q", "raw/sample.txt")
    payload.git("commit", "-qm", "remove data")
    assert payload.audit(payload.outgoing_entries(base)) == [
        ("raw/sample.txt", "private or runtime-data directory")
    ]


def test_new_large_blob_blocks_but_unchanged_baseline_is_not_rescanned(repository):
    root, _ = repository
    (root / "aggregate.json").write_bytes(b" " * (payload.MAX_BYTES + 1))
    payload.git("add", "aggregate.json")
    assert payload.audit(payload.staged_entries()) == [
        ("aggregate.json", "exceeds 5 MiB publication limit")
    ]
    payload.git("commit", "-qm", "historical large aggregate")
    base = payload.git("rev-parse", "HEAD").decode().strip()
    (root / "code.py").write_text("answer = 42\n")
    payload.git("add", "code.py")
    payload.git("commit", "-qm", "code only")
    assert payload.audit(payload.outgoing_entries(base)) == []


def test_regular_code_allowed_but_data_symlink_rejected(repository):
    root, _ = repository
    (root / "code.py").write_text("answer = 42\n")
    (root / ".env.example").write_text("API_KEY=replace-me\n")
    (root / "evidence").symlink_to("../lexical-prompt-study-data")
    payload.git("add", "code.py", ".env.example", "evidence")
    assert payload.audit(payload.staged_entries()) == [
        ("evidence", "symlink or non-regular Git entry requires separate review")
    ]


def test_unmerged_index_refuses_even_without_staged_regular_changes(repository):
    root, _ = repository
    (root / "code.py").write_text("baseline\n")
    payload.git("add", "code.py")
    payload.git("commit", "-qm", "base code")
    payload.git("branch", "side")
    (root / "code.py").write_text("main change\n")
    payload.git("commit", "-qam", "main change")
    payload.git("switch", "-q", "side")
    (root / "code.py").write_text("side change\n")
    payload.git("commit", "-qam", "side change")
    result = subprocess.run(
        ["git", "merge", "--no-edit", "@{-1}"], capture_output=True, check=False,
    )
    assert result.returncode == 1
    with pytest.raises(ValueError, match="Unmerged"):
        payload.staged_entries()


def test_merged_side_branch_deleted_data_is_still_audited(repository):
    root, base = repository
    primary = payload.git("branch", "--show-current").decode().strip()
    payload.git("switch", "-qc", "side")
    (root / "raw").mkdir()
    (root / "raw" / "sample.txt").write_text("synthetic only")
    payload.git("add", "raw/sample.txt")
    payload.git("commit", "-qm", "add private data")
    payload.git("rm", "-q", "raw/sample.txt")
    payload.git("commit", "-qm", "remove private data")
    payload.git("switch", "-q", primary)
    payload.git("merge", "--no-ff", "-qm", "merge side history", "side")
    assert payload.audit(payload.outgoing_entries(base)) == [
        ("raw/sample.txt", "private or runtime-data directory")
    ]
