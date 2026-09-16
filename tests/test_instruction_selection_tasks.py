import copy
import json
import random

import pytest

from lexical_prompt_study import instruction_selection_tasks as tasks


def materials():
    return {kind: "Public synthetic " + kind for kind in tasks.SCAFFOLDS}


def plan(partition="development"):
    return tasks.compile_plan(tasks.make_cohort_manifest(), partition=partition,
                              scaffolds=materials(), bindings={"protocol_sha256": "a" * 64,
                                                               "materials_sha256": "b" * 64})


def test_manifest_reproducible_balanced_disjoint_and_does_not_mutate_rng():
    state = random.getstate()
    manifest = tasks.make_cohort_manifest()
    assert random.getstate() == state
    assert manifest == tasks.make_cohort_manifest()
    assert manifest != tasks.make_cohort_manifest(tasks.COHORT_SEED + 1)
    tasks.validate_cohort_manifest(manifest)
    dev = {r["world_id"] for r in manifest["development"]}
    test = {r["world_id"] for r in manifest["heldout"]}
    assert len(dev) == 16 and len(test) == 64 and not dev & test
    for partition, multiplier in (("development", 1), ("heldout", 4)):
        rows = manifest[partition]
        for label in ("A", "B"):
            assert all(sum(row["world"]["answers"][label] == symbol for row in rows) == multiplier
                       for symbol in tasks.SYMBOLS)
        assert sum(row["query_order"] == "AB" for row in rows) == len(rows) // 2


@pytest.mark.parametrize("partition,count", [("development", 32), ("heldout", 1152)])
def test_plans_have_exact_cells_and_user_identical_selector_pairs(partition, count):
    value = plan(partition)
    tasks.validate_plan(value)
    assert len(value["trials"]) == len({r["trial_id"] for r in value["trials"]}) == count
    for a, b in zip(value["trials"][::2], value["trials"][1::2], strict=True):
        assert a["messages"][1] == b["messages"][1]
        assert a["messages"][0]["content"].replace("selector is A", "selector is B") == b["messages"][0]["content"]
        assert a["selected_answer"] == b["unselected_answer"]
        assert a["selected_answer"] != b["selected_answer"]
        assert "depth" not in a["world"] and "task_depth" not in a
        assert a["selected_answer"] == a["world"]["answers"][a["selector"]]
    if partition == "development":
        assert {r["scaffold_kind"] for r in value["trials"]} == {"none"}


def test_world_identity_excludes_active_selector_and_presentation():
    world = tasks.AnswerWorld("s02", "s14")
    assert tasks.render_world(world, "AB") != tasks.render_world(world, "BA")
    assert json.loads(tasks.render_world(world, "BA")) == world.payload()["answers"]
    assert world.world_id == tasks.AnswerWorld("s02", "s14").world_id
    assert world.world_id != tasks.AnswerWorld("s14", "s02").world_id
    with pytest.raises(ValueError):
        tasks.AnswerWorld("s02", "s02")


@pytest.mark.parametrize("mutation", [
    lambda p: p["trials"].pop(),
    lambda p: p["trials"][0].update(selected_answer="s99"),
    lambda p: p["trials"][0]["messages"][0].update(role="user"),
    lambda p: p["trials"][0]["messages"][1].update(content="changed"),
    lambda p: p["trials"][0].update(query_order="BA" if p["trials"][0]["query_order"] == "AB" else "AB"),
    lambda p: p["bindings"].update(task_source_sha256="f" * 64),
])
def test_plan_tampering_rejected(mutation):
    value = plan()
    mutation(value)
    with pytest.raises(ValueError):
        tasks.validate_plan(value)


def test_heldout_scaffold_actual_bytes_are_verified_against_material_receipt():
    value = plan("heldout")
    row = next(r for r in value["trials"] if r["scaffold_kind"] == "full")
    row["messages"][1]["content"] = "altered" + row["messages"][1]["content"]
    with pytest.raises(ValueError, match="scaffold_binding"):
        tasks.validate_plan(value)


def test_manifest_overlap_or_balance_drift_rejected():
    manifest = tasks.make_cohort_manifest()
    bad = copy.deepcopy(manifest)
    bad["heldout"][0] = copy.deepcopy(bad["development"][0])
    with pytest.raises(ValueError):
        tasks.validate_cohort_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["development"][0]["query_order"] = "AB" if bad["development"][0]["query_order"] == "BA" else "BA"
    with pytest.raises(ValueError, match="presentation_balance"):
        tasks.validate_cohort_manifest(bad)


@pytest.mark.parametrize("response,expected", [
    ('{"answer":"s02"}', "exact"), ('{"answer":"s14"}', "other_selector"),
    ('{"answer":"s05"}', "other_answer"), ('{"answer":"s99"}', "other_answer"),
    ('{"answer":"s02","extra":1}', "format"), ('{"answer":"s02"} trailing', "format"),
    ('{"answer":"s02","answer":"s14"}', "format"), ('{"answer":2}', "format"),
    ('{"answer":NaN}', "format"), ('```json\n{"answer":"s02"}\n```', "format"),
])
def test_strict_scorer_preserves_format_precedence(response, expected):
    assert tasks.score_response(response, "s02", "s14") == expected
