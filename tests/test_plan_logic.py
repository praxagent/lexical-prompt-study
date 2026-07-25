from lexical_prompt_study.plans import CATEGORIES, _harmful_splits


def test_category_balanced_split_is_deterministic() -> None:
    rows = [
        {
            "Index": str(category_index * 10 + row_index),
            "Category": category,
            "Goal": f"safe synthetic goal {category_index} {row_index}",
            "Target": "safe target",
            "Behavior": "safe behavior",
            "Source": "synthetic",
        }
        for category_index, category in enumerate(CATEGORIES)
        for row_index in range(10)
    ]
    first = _harmful_splits(rows)
    second = _harmful_splits(list(reversed(rows)))
    assert [(r["Index"], s) for r, s in first] == [(r["Index"], s) for r, s in second]
    for category in CATEGORIES:
        splits = [split for row, split in first if row["Category"] == category]
        assert splits.count("discovery") == 2
        assert splits.count("confirmatory") == 4
        assert splits.count("reserve") == 4
