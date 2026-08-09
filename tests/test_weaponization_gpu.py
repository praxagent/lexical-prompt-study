from __future__ import annotations

from lexical_prompt_study.weaponization_gpu import structural_prompt_metrics


def test_structural_prompt_metrics_are_nonsemantic_and_deterministic() -> None:
    prompt = "### Stage\n[check]\n---\n===\n"
    assert structural_prompt_metrics(prompt) == {
        "prompt_bytes": 26.0,
        "line_count": 5.0,
        "dash_divider_count": 1.0,
        "equals_divider_count": 1.0,
        "markdown_heading_count": 1.0,
        "bracket_count": 2.0,
    }
