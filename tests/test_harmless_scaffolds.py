from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from lexical_prompt_study.harmless_scaffolds import (
    TARGET_CUMULATIVE_TOKEN_COUNTS,
    compile_harmless_wrappers,
    validate_harmless_library,
    validate_human_review_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = ROOT / "plans" / "harmless_scaffold_families_v1.public.json"


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"\w+|[^\w\s]", text))))


def _library() -> dict:
    return json.loads(LIBRARY_PATH.read_text())


def test_harmless_library_passes_strict_validation() -> None:
    validate_harmless_library(_library())


def test_harmless_library_rejects_continue_after_refusal() -> None:
    library = copy.deepcopy(_library())
    library["safety_contract"]["continue_after_refusal_forbidden"] = False
    with pytest.raises(ValueError, match="safety contract"):
        validate_harmless_library(library)


def test_harmless_library_rejects_family_count_drift() -> None:
    library = copy.deepcopy(_library())
    library["families"].pop()
    with pytest.raises(ValueError, match="30 semantic families"):
        validate_harmless_library(library)


def test_human_review_receipt_binds_exact_library() -> None:
    receipt = json.loads(
        (ROOT / "validation" / "harmless-scaffolds.human-review.public.json").read_text()
    )
    validate_human_review_receipt(
        receipt,
        library_sha256="46d5697068e3de551cd4a1ef06d2112248e4e66c4b613365d72f61340a8e0276",
    )
    receipt["library_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="boundary drift"):
        validate_human_review_receipt(
            receipt,
            library_sha256="46d5697068e3de551cd4a1ef06d2112248e4e66c4b613365d72f61340a8e0276",
        )


def test_harmless_compiler_makes_disjoint_exact_size_partitions() -> None:
    compiled = compile_harmless_wrappers(library=_library(), tokenizer=WordTokenizer())
    wrappers = compiled["wrappers"]
    assert len(wrappers) == 600
    assert len({row["text_sha256"] for row in wrappers}) == 600
    calibration = [row for row in wrappers if row["partition"] == "detector_calibration"]
    confirmation = [row for row in wrappers if row["partition"] == "detector_confirmation"]
    assert len(calibration) == 300
    assert len(confirmation) == 300
    assert {row["family_id"] for row in calibration}.isdisjoint(
        {row["family_id"] for row in confirmation}
    )
    for row in wrappers:
        assert row["canonical_token_count"] == 252
        assert [block["cumulative_token_count"] for block in row["blocks"]] == list(
            TARGET_CUMULATIVE_TOKEN_COUNTS
        )
        assert all(block["text"] for block in row["blocks"])
