import json

import pytest

from lexical_prompt_study.instruction_binding_plan import (
    assemble_materials, compile_plan, compile_qualification, digest, load_materials, world_cohort,
)


def synthetic_materials():
    return {"material_block_joiner": "\n--\n", "scaffold_materials": {
        kind: {"blocks": [{"text": f"{kind} synthetic block {i}"} for i in range(4)]}
        for kind in ("full_scaffold", "structural_sham", "inert_length")}}


def test_material_recipe_and_hash_boundary(tmp_path):
    payload = synthetic_materials()
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload))
    material = load_materials(path, digest(path.read_bytes()))
    assert material["replacement"].split("\n--\n") == [
        "full_scaffold synthetic block 0", "full_scaffold synthetic block 1",
        "full_scaffold synthetic block 2", "inert_length synthetic block 3"]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_materials(path, "0" * 64)


@pytest.mark.parametrize("stage,worlds,trials,diagnostics", [
    ("development_pilot", 8, 160, 16), ("development", 8, 160, 16),
    ("heldout", 64, 1152, 0),
])
def test_matrix_and_primary_support(stage, worlds, trials, diagnostics):
    plan = compile_plan(stage=stage, scaffolds=assemble_materials(synthetic_materials()), bindings={})
    assert plan["world_count"] == worlds
    assert len(plan["trials"]) == trials
    assert sum(row["diagnostic"] for row in plan["trials"]) == diagnostics
    main = [r for r in plan["trials"] if not r["diagnostic"]]
    assert all(r["selected_answer"] != r["unselected_answer"] for r in main)
    assert len({r["trial_id"] for r in plan["trials"]}) == trials
    for world_id in {r["world_id"] for r in main}:
        cells = [r for r in main if r["world_id"] == world_id]
        assert len(cells) == 18
        for scaffold, placement in {(r["scaffold_kind"], r["placement"]) for r in cells}:
            a, b = [r for r in cells if (r["scaffold_kind"], r["placement"]) ==
                    (scaffold, placement)]
            assert a["messages"][1] == b["messages"][1]
            assert a["selected_answer"] == b["unselected_answer"]


def test_partitions_and_pilot_are_disjoint_and_presentation_balanced():
    scaffolds = assemble_materials(synthetic_materials())
    plans = [compile_plan(stage=s, scaffolds=scaffolds, bindings={})
             for s in ("development_pilot", "development", "heldout")]
    ids = [{r["world_id"] for r in p["trials"]} for p in plans]
    assert not ids[0] & ids[1] and not (ids[0] | ids[1]) & ids[2]
    assert ids[0] | ids[1] == {w.world_id for w in world_cohort("development")}
    for plan in plans:
        representatives = {r["world_id"]: r for r in plan["trials"]}
        for depth in (1, 2):
            rows = [r for r in representatives.values() if r["depth"] == depth]
            assert sum(r["query_order"] == "AB" for r in rows) == len(rows) // 2
            if depth == 2:
                for order in ("AB", "BA"):
                    assert sum(r["query_order"] == order and r["table_order"] == "reverse"
                               for r in rows) == len(rows) // 4


def test_qualification_is_disjoint_harmless_repeated_input():
    plan = compile_qualification(bindings={})
    assert plan["partition"] == "synthetic" and len(plan["trials"]) == 16
    assert len({r["trial_id"] for r in plan["trials"]}) == 16
    protected = {w.world_id for p in ("development", "heldout") for w in world_cohort(p)}
    assert not protected & {r["world_id"] for r in plan["trials"]}
    seen = {}
    for row in plan["trials"]:
        assert row["scaffold_kind"] == "none"
        key = json.dumps(row["messages"], sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
    assert len(seen) == 8 and set(seen.values()) == {2}
