from lexical_prompt_study.behavior import REFUSAL_PATTERN


def test_refusal_pattern() -> None:
    assert REFUSAL_PATTERN.search("I'm sorry, but I can't help with that.")
    assert not REFUSAL_PATTERN.search("Here is a harmless synthetic answer.")
