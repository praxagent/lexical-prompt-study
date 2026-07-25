from pathlib import Path

from lexical_prompt_study.hashing import git_blob_oid_file


def test_git_blob_oid_matches_git_object_format(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    example = tmp_path / "example"
    example.write_bytes(b"test content\n")

    assert git_blob_oid_file(empty) == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert git_blob_oid_file(example) == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"
