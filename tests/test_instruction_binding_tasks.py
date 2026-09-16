import hashlib
import json
import random
from dataclasses import replace

import pytest

from lexical_prompt_study.instruction_binding_tasks import (
    RENDERERS,
    SCAFFOLD_KINDS,
    SYMBOLS,
    LookupWorld,
    build_conditions,
    counterbalanced_query_orders,
    counterbalanced_presentations,
    generate_worlds,
    make_world,
    oracle,
    render_world,
    score_response,
)


@pytest.mark.parametrize("depth", [1, 2])
def test_seeded_cohort_has_unique_worlds_and_distinct_actual_depth_answers(depth):
    worlds = generate_worlds(seed=20260914, count=100, depth=depth)
    assert worlds == generate_worlds(seed=20260914, count=100, depth=depth)
    assert len({world.world_id for world in worlds}) == 100
    assert worlds[0] == make_world(seed=20260914, depth=depth)
    assert worlds != generate_worlds(seed=20260915, count=100, depth=depth)
    for world in worlds:
        assert oracle(world, "A") != oracle(world, "B")
        for selector, key in zip(("A", "B"), world.query_keys, strict=True):
            # Independently materialize maps and traverse them, without oracle
            # indexing, to catch wrong-table or wrong-query implementation.
            answer = key
            for outputs in world.tables:
                mapping = dict(zip(SYMBOLS, outputs, strict=True))
                answer = mapping[answer]
            assert oracle(world, selector) == answer


def test_two_hop_oracle_uses_second_table_not_first_hop_or_wrong_composition():
    first = tuple(SYMBOLS[(index + 1) % 8] for index in range(8))
    second = tuple(reversed(SYMBOLS))
    world = LookupWorld(depth=2, tables=(first, second), query_keys=("s0", "s3"))
    assert oracle(world, "A") == "s6"
    assert oracle(world, "B") == "s3"
    assert oracle(world, "A") != first[0]
    assert oracle(world, "A") != first[7]  # table_1(table_2(s0))


def test_world_hash_identifies_content_without_selector_or_provenance():
    world = make_world(seed=91, depth=2)
    payload = {"schema_version": "instruction-binding-world-v1", **world.payload()}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    assert world.world_id == expected
    assert replace(world).world_id == expected
    assert replace(world, query_keys=tuple(reversed(world.query_keys))).world_id != expected
    assert "selector" not in payload and "seed" not in payload


def test_generation_does_not_mutate_global_random_state():
    before = random.getstate()
    generate_worlds(seed=4, count=5, depth=1)
    assert random.getstate() == before


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("renderer", RENDERERS)
def test_all_eighteen_cells_share_world_and_counterfactual_user_bytes(depth, renderer):
    world = make_world(seed=81, depth=depth)
    scaffolds = {kind: f"  Synthetic {kind} text.\n" for kind in SCAFFOLD_KINDS}
    cells = build_conditions(world, renderer=renderer, scaffolds=scaffolds)
    assert len(cells) == len({cell.condition_id for cell in cells}) == 18
    assert {cell.world_id for cell in cells} == {world.world_id}
    expected = {("none", "none", selector) for selector in ("A", "B")}
    expected |= {(kind, placement, selector) for kind in SCAFFOLD_KINDS
                 for placement in ("before", "after") for selector in ("A", "B")}
    assert {(cell.scaffold_kind, cell.placement, cell.selector) for cell in cells} == expected
    for a, b in zip(cells[::2], cells[1::2], strict=True):
        assert a.selector == "A" and b.selector == "B"
        assert a.user_message.encode() == b.user_message.encode()
        assert a.system_message.replace("selector is A", "selector is B") == b.system_message
        assert a.messages[0]["role"] == "system"
        assert a.messages[1] == {"role": "user", "content": a.user_message}
        base = render_world(world, renderer)
        if a.placement == "before":
            assert a.user_message == scaffolds[a.scaffold_kind] + "\n\n" + base
        elif a.placement == "after":
            assert a.user_message == base + "\n\n" + scaffolds[a.scaffold_kind]
        else:
            assert a.user_message == base


def test_both_renderers_encode_identical_query_and_table_data():
    world = make_world(seed=6, depth=2)
    assert json.loads(render_world(world, "json")) == world.payload()
    lines = render_world(world, "line_table").splitlines()
    assert lines[:3] == ["depth: 2", f"query A: {world.query_keys[0]}",
                         f"query B: {world.query_keys[1]}"]
    for hop, table in enumerate(world.tables):
        start = 3 + hop * 9
        assert lines[start] == f"table_{hop + 1} (input -> output):"
        recovered = dict(line.split(" -> ") for line in lines[start + 1:start + 9])
        assert recovered == dict(zip(SYMBOLS, table, strict=True))


@pytest.mark.parametrize("renderer", RENDERERS)
def test_query_order_changes_presentation_only_and_all_cells_remain_counterfactual(renderer):
    world = make_world(seed=6, depth=2)
    before = world.world_id
    a = render_world(world, renderer, query_order="AB")
    b = render_world(world, renderer, query_order="BA")
    assert a != b and world.world_id == before
    if renderer == "json":
        assert json.loads(a) == json.loads(b) == world.payload()
        assert list(json.loads(a)["queries"]) == ["A", "B"]
        assert list(json.loads(b)["queries"]) == ["B", "A"]
    else:
        assert a.splitlines()[1:3] == list(reversed(b.splitlines()[1:3]))
        assert a.splitlines()[3:] == b.splitlines()[3:]
    scaffolds = dict.fromkeys(SCAFFOLD_KINDS, "Harmless synthetic context.")
    ab = build_conditions(world, renderer=renderer, scaffolds=scaffolds, query_order="AB")
    ba = build_conditions(world, renderer=renderer, scaffolds=scaffolds, query_order="BA")
    assert not ({cell.condition_id for cell in ab} & {cell.condition_id for cell in ba})
    assert {cell.query_order for cell in ba} == {"BA"}
    for first, second in zip(ba[::2], ba[1::2], strict=True):
        assert first.user_message == second.user_message
        assert first.system_message.replace("selector is A", "selector is B") == second.system_message


def test_query_order_assignment_balanced_reproducible_and_not_input_order_dependent():
    worlds = generate_worlds(seed=40, count=8, depth=1) + generate_worlds(seed=41, count=8, depth=2)
    state = random.getstate()
    assignments = counterbalanced_query_orders(worlds, seed=16)
    assert random.getstate() == state
    assert assignments == counterbalanced_query_orders(tuple(reversed(worlds)), seed=16)
    for depth in (1, 2):
        orders = [assignments[world.world_id] for world in worlds if world.depth == depth]
        assert orders.count("AB") == orders.count("BA") == 4
    with pytest.raises(ValueError):
        counterbalanced_query_orders(worlds[:-1], seed=16)
    with pytest.raises(ValueError):
        counterbalanced_query_orders(worlds + (worlds[0],), seed=16)
    with pytest.raises(ValueError):
        render_world(worlds[0], "json", query_order="AA")


@pytest.mark.parametrize("renderer", RENDERERS)
def test_reversed_table_presentation_preserves_named_composition(renderer):
    world = make_world(seed=4, depth=2)
    forward = render_world(world, renderer, table_order="forward")
    reverse = render_world(world, renderer, table_order="reverse")
    assert forward.index("table_1") < forward.index("table_2")
    assert reverse.index("table_2") < reverse.index("table_1")
    if renderer == "json":
        assert json.loads(forward) == json.loads(reverse) == world.payload()
    scaffolds = dict.fromkeys(SCAFFOLD_KINDS, "Harmless synthetic context.")
    cells = build_conditions(world, renderer=renderer, scaffolds=scaffolds,
                             query_order="BA", table_order="reverse")
    assert len(cells) == 18 and all(cell.table_order == "reverse" for cell in cells)
    assert all("table_1" in cell.system_message and "table_2" in cell.system_message for cell in cells)
    for a, b in zip(cells[::2], cells[1::2], strict=True):
        assert a.user_message == b.user_message


def test_joint_presentation_assignment_exactly_balances_each_family():
    worlds = generate_worlds(seed=8, count=8, depth=1) + generate_worlds(seed=9, count=8, depth=2)
    assignments = counterbalanced_presentations(worlds, seed=10)
    assert assignments == counterbalanced_presentations(tuple(reversed(worlds)), seed=10)
    for depth in (1, 2):
        cells = [tuple(assignments[w.world_id].values()) for w in worlds if w.depth == depth]
        combinations = {(q, t) for q in ("AB", "BA")
                        for t in (("forward", "reverse") if depth == 2 else ("forward",))}
        assert set(cells) == combinations
        assert all(cells.count(cell) == 8 // len(combinations) for cell in combinations)
    with pytest.raises(ValueError):
        counterbalanced_presentations(worlds[:-1], seed=10)
    with pytest.raises(ValueError):
        render_world(worlds[0], "json", table_order="reverse")


@pytest.mark.parametrize("selector", ["A", "B"])
def test_scoring_correct_other_selector_other_symbol_and_external_string(selector):
    world = make_world(seed=17, depth=2)
    alternate = "B" if selector == "A" else "A"
    assert score_response(json.dumps({"answer": oracle(world, selector)}), world, selector) == "exact"
    assert score_response(json.dumps({"answer": oracle(world, alternate)}), world,
                          selector) == "other_selector"
    remaining = set(SYMBOLS) - {oracle(world, "A"), oracle(world, "B")}
    for answer in remaining | {"not-a-symbol", "", oracle(world, selector) + " "}:
        assert score_response(json.dumps({"answer": answer}), world, selector) == "other_answer"


@pytest.mark.parametrize("answer_source", ["A", "B"])
def test_format_failure_precedes_any_semantic_answer(answer_source):
    world = make_world(seed=17, depth=1)
    answer = oracle(world, answer_source)
    valid = json.dumps({"answer": answer})
    failures = [
        json.dumps({"wrong_key": answer}),
        json.dumps({"answer": answer, "extra": "text"}),
        valid + " trailing", valid + valid, f"```json\n{valid}\n```",
        answer, json.dumps(answer), json.dumps([{"answer": answer}]),
        '{"answer":"' + answer + '","answer":"' + answer + '"}',
        '{"answer":NaN}', '{"answer":Infinity}', '{"answer":null}',
        '{"answer":1}', '{"answer":true}', '{"answer":[]}',
        '{"answer":{}}', '{"answer":"s0",}', "", None,
    ]
    for response in failures:
        assert score_response(response, world, "A") == "format"
    expected = "exact" if answer_source == "A" else "other_selector"
    assert score_response(" \n\t" + valid + "\r\n", world, "A") == expected


@pytest.mark.parametrize("kwargs", [
    {"seed": True, "count": 1, "depth": 1},
    {"seed": 1, "count": 0, "depth": 1},
    {"seed": 1, "count": True, "depth": 1},
    {"seed": 1, "count": 1, "depth": 3},
    {"seed": 1, "count": 1, "depth": True},
])
def test_invalid_generation_arguments_fail(kwargs):
    with pytest.raises(ValueError):
        generate_worlds(**kwargs)


def test_invalid_worlds_selectors_renderers_and_scaffolds_fail():
    world = make_world(seed=1, depth=1)
    with pytest.raises(ValueError):
        replace(world, tables=(("s0",) * 8,))
    with pytest.raises(ValueError):
        replace(world, query_keys=("s0", "s0"))
    with pytest.raises(ValueError):
        replace(world, tables=[SYMBOLS])
    with pytest.raises(ValueError):
        oracle(world, "C")
    with pytest.raises(ValueError):
        render_world(world, "xml")
    for scaffolds in ({}, {kind: "" for kind in SCAFFOLD_KINDS},
                      {**dict.fromkeys(SCAFFOLD_KINDS, "text"), "extra": "text"}):
        with pytest.raises(ValueError):
            build_conditions(world, renderer="json", scaffolds=scaffolds)
