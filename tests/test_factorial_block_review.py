from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexical_prompt_study.factorial_block_review import (
    BOUNDARY_MARKER,
    MATERIAL_FILES,
    auto_align_factorial_block_review,
    compile_factorial_block_review,
    write_factorial_block_review_aid,
)
from lexical_prompt_study.hashing import sha256_text


class SafeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }


def _manifest(tmp_path: Path, *, mismatched: bool = False) -> Path:
    rows = {}
    for material, filename in MATERIAL_FILES.items():
        original = "AAAABBBBCCCC"
        marked = f"AAAA{BOUNDARY_MARKER}BBBB{BOUNDARY_MARKER}CCCC"
        if mismatched and material == "structural_sham":
            original = "AAAAZZBBBCCCC"
            marked = f"AAAA{BOUNDARY_MARKER}ZZBBB{BOUNDARY_MARKER}CCCC"
        path = tmp_path / filename
        path.write_text(marked)
        rows[material] = {
            "path": str(path),
            "original_text_sha256": sha256_text(original),
            "canonical_token_count": len(original),
            "review_status": "insert_boundaries_then_compile",
        }
    manifest = {
        "schema_version": "1.0",
        "status": "human_semantic_block_review_required",
        "tokenizer_revision": "safe",
        "tokenizer_chat_template_sha256": "1" * 64,
        "boundary_marker": BOUNDARY_MARKER,
        "instructions": "safe",
        "agent_plaintext_inspection_forbidden": True,
        "materials": rows,
    }
    path = tmp_path / "manifest.private.json"
    path.write_text(json.dumps(manifest))
    return path


def _whitespace_manifest(tmp_path: Path) -> Path:
    rows = {}
    original = "aa bb cc dd ee ff gg hh"
    for material, filename in MATERIAL_FILES.items():
        path = tmp_path / filename
        path.write_text(f"aa bb{BOUNDARY_MARKER} cc dd ee ff gg hh")
        rows[material] = {
            "path": str(path),
            "original_text_sha256": sha256_text(original),
            "canonical_token_count": len(original),
            "review_status": "insert_boundaries_then_compile",
        }
    manifest = {
        "schema_version": "1.0",
        "status": "human_semantic_block_review_required",
        "tokenizer_revision": "safe",
        "tokenizer_chat_template_sha256": "1" * 64,
        "boundary_marker": BOUNDARY_MARKER,
        "instructions": "safe",
        "agent_plaintext_inspection_forbidden": True,
        "materials": rows,
    }
    path = tmp_path / "manifest.private.json"
    path.write_text(json.dumps(manifest))
    return path


def test_human_block_compiler_preserves_text_and_exact_prefix_counts(
    tmp_path: Path,
) -> None:
    result = compile_factorial_block_review(
        manifest_path=_manifest(tmp_path),
        tokenizer=SafeTokenizer(),
        output_path=tmp_path / "blocks.private.json",
    )
    assert result["block_count"] == 3
    assert result["cumulative_token_counts"] == [4, 8, 12]
    assert result["raw_text_returned"] is False


def test_human_block_compiler_rejects_cumulative_token_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cumulative-token mismatch"):
        compile_factorial_block_review(
            manifest_path=_manifest(tmp_path, mismatched=True),
            tokenizer=SafeTokenizer(),
            output_path=tmp_path / "blocks.private.json",
        )


def test_human_block_compiler_rejects_non_boundary_edits(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    path = Path(manifest["materials"]["full_scaffold"]["path"])
    path.write_text(path.read_text().replace("AAAA", "AAAX", 1))
    with pytest.raises(ValueError, match="non-boundary bytes changed"):
        compile_factorial_block_review(
            manifest_path=manifest_path,
            tokenizer=SafeTokenizer(),
            output_path=tmp_path / "blocks.private.json",
        )


def test_human_block_review_aid_is_private_and_returns_only_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "review.private.html"
    result = write_factorial_block_review_aid(
        manifest_path=_manifest(tmp_path),
        tokenizer=SafeTokenizer(),
        output_path=output,
    )
    assert result["status"] == "human_only_factorial_block_review_aid_written"
    assert result["raw_text_returned"] is False
    assert result["materials"]["full_scaffold"] == {
        "marker_count": 2,
        "current_cumulative_token_counts": [4, 8, 12],
        "canonical_token_count": 12,
    }
    assert output.stat().st_mode & 0o777 == 0o600
    rendered = output.read_text()
    assert "CURRENT B1" in rendered
    assert "T012" in rendered


def test_auto_align_uses_shared_whitespace_boundaries_and_private_backups(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocks.private.json"
    result = auto_align_factorial_block_review(
        manifest_path=_whitespace_manifest(tmp_path),
        tokenizer=SafeTokenizer(),
        output_path=output,
    )
    assert result["status"] == (
        "machine_whitespace_blocks_compiled_exact_token_match"
    )
    assert result["raw_text_returned"] is False
    assert result["block_count"] == 4
    assert result["cumulative_token_counts"][-1] == len(
        "aa bb cc dd ee ff gg hh"
    )
    assert result["cumulative_token_counts"] == sorted(
        set(result["cumulative_token_counts"])
    )
    assert output.stat().st_mode & 0o777 == 0o600
    for material, filename in MATERIAL_FILES.items():
        marked = (tmp_path / filename).read_text()
        assert marked.count(BOUNDARY_MARKER) == 3
        assert marked.replace(BOUNDARY_MARKER, "") == (
            "aa bb cc dd ee ff gg hh"
        )
        backup = Path(result["backup_receipts"][material]["backup_path"])
        assert backup.stat().st_mode & 0o777 == 0o600
