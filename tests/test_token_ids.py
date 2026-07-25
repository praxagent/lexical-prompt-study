from lexical_prompt_study.behavior import _as_token_ids


class EncodingLike:
    ids = [1, 2, 3]


def test_as_token_ids_handles_transformers_variants() -> None:
    assert _as_token_ids([1, 2]) == [1, 2]
    assert _as_token_ids([[1, 2]]) == [1, 2]
    assert _as_token_ids({"input_ids": [[1, 2]]}) == [1, 2]
    assert _as_token_ids(EncodingLike()) == [1, 2, 3]
