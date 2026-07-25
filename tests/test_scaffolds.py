from lexical_prompt_study.scaffolds import render_arms


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def test_structured_arms_are_length_matched() -> None:
    arms = render_arms("Explain a harmless synthetic task.", WordTokenizer())
    lengths = {len(arms[name].token_ids) for name in ("full", "structural_sham", "inert_length")}
    assert len(lengths) == 1
    assert arms["base"].target_tokens is None
