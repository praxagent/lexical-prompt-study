from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lexical_prompt_study.factorial_private import build_factorial_private_plan
from lexical_prompt_study.hashing import sha256_file, sha256_text

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PLAN = ROOT / "plans" / "factorial_8b_v1.public.json"


class SafeTokenizer:
    chat_template = "safe-factorial-template-v1"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt
        text = "[USER]" + messages[0]["content"] + "[ASSISTANT]"
        return [ord(character) for character in text] if tokenize else text


def _source() -> dict:
    panels = {}
    for level_index, level in enumerate(
        (
            "minimal_neutral_carrier",
            "ordinary_benign_request",
            "harmful_request",
        )
    ):
        panels[level] = []
        for index in range(20):
            request = f"safe request class {level_index} item {index:02d}"
            panels[level].append(
                {
                    "request_id": f"R{level_index}-{index:02d}",
                    "prompt_family_id": f"F{level_index}-{index:02d}",
                    "request": request,
                    "request_sha256": sha256_text(request),
                    "source_sha256": sha256_text(f"source-{level_index}-{index}"),
                }
            )
    materials = {}
    for material, prefix in (
        ("inert_length", "I"),
        ("structural_sham", "S"),
        ("full_scaffold", "F"),
    ):
        blocks = []
        for index in range(4):
            text = f"{prefix}{index:03d}"
            blocks.append(
                {
                    "block_id": f"{material}-{index}",
                    "text": text,
                    "text_sha256": sha256_text(text),
                }
            )
        materials[material] = {"blocks": blocks}
    sentinel = "safe minimal carrier"
    joiner = "|"
    return {
        "schema_version": "1.0",
        "study_id": "lexical-scaffold-8b-factorial-v1",
        "public_plan_sha256": sha256_file(PUBLIC_PLAN),
        "provenance": {
            "human_reviewed_restricted_scaffold": True,
            "new_harmful_panel_disjointness_attested": True,
            "outcome_independent_material_construction": True,
        },
        "request_panels": panels,
        "material_block_joiner": joiner,
        "material_block_joiner_sha256": sha256_text(joiner),
        "scaffold_materials": materials,
        "literal_sentinel": {
            "minimal_carrier": sentinel,
            "minimal_carrier_sha256": sha256_text(sentinel),
        },
    }


def _write_source(tmp_path: Path, source: dict) -> Path:
    path = tmp_path / "factorial-materials.private.json"
    path.write_text(json.dumps(source))
    return path


def _build(tmp_path: Path, source: dict) -> dict:
    return build_factorial_private_plan(
        public_plan_path=PUBLIC_PLAN,
        material_source_path=_write_source(tmp_path, source),
        tokenizer=SafeTokenizer(),
        tokenizer_revision="safe-pinned",
        private_output_path=tmp_path / "factorial.private.json",
        public_receipt_path=tmp_path / "factorial.public-receipt.json",
    )


def test_factorial_private_builder_realizes_exact_full_matrix(tmp_path: Path) -> None:
    result = _build(tmp_path, _source())
    assert result["canonical_observations"] == 422
    assert result["additional_dose_observations"] == 540
    assert result["total_observations"] == 962
    private_path = Path(result["private_plan_path"])
    public_path = Path(result["public_receipt_path"])
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert public_path.stat().st_mode & 0o777 == 0o644

    private = json.loads(private_path.read_text())
    public = json.loads(public_path.read_text())
    assert len(private["doses"]) == 4
    assert public["exact_size_matching_passed"] is True
    assert public["target_generation_performed"] is False
    assert "observations" not in public
    assert "request_panels" not in public
    assert "prompt_token_ids" not in public_path.read_text()


def test_factorial_private_builder_rejects_token_size_mismatch(
    tmp_path: Path,
) -> None:
    source = _source()
    block = source["scaffold_materials"]["structural_sham"]["blocks"][0]
    block["text"] += "x"
    block["text_sha256"] = sha256_text(block["text"])
    with pytest.raises(ValueError, match="cumulative injected-token counts"):
        _build(tmp_path, source)


def test_factorial_private_builder_rejects_prompt_family_reuse(
    tmp_path: Path,
) -> None:
    source = _source()
    rows = source["request_panels"]["ordinary_benign_request"]
    rows[1]["prompt_family_id"] = rows[0]["prompt_family_id"]
    with pytest.raises(ValueError, match="pseudoreplication"):
        _build(tmp_path, source)


def test_factorial_private_builder_merges_duplicate_realized_doses(
    tmp_path: Path,
) -> None:
    source = _source()
    for material in source["scaffold_materials"].values():
        material["blocks"] = material["blocks"][:2]
    result = _build(tmp_path, source)
    private = json.loads(Path(result["private_plan_path"]).read_text())
    assert len(private["doses"]) < 4
    prefix_counts = [dose["prefix_block_count"] for dose in private["doses"]]
    assert len(prefix_counts) == len(set(prefix_counts))
    assert result["additional_dose_observations"] < 540


def test_factorial_private_builder_rejects_cross_class_request_reuse(
    tmp_path: Path,
) -> None:
    source = _source()
    first = source["request_panels"]["minimal_neutral_carrier"][0]
    duplicate = source["request_panels"]["ordinary_benign_request"][0]
    duplicate["request"] = first["request"]
    duplicate["request_sha256"] = first["request_sha256"]
    with pytest.raises(ValueError, match="reused across request classes"):
        _build(tmp_path, source)


def test_factorial_private_source_fixture_mutations_are_isolated() -> None:
    source = _source()
    mutated = copy.deepcopy(source)
    mutated["provenance"]["outcome_independent_material_construction"] = False
    assert source["provenance"]["outcome_independent_material_construction"] is True
