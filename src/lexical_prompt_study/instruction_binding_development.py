"""One fixed, baseline-only A163 development amendment; no model execution.

Inputs are outcome-free development plans and receipt-validated numeric records.
Restricted source messages stay inside returned private plans. Summaries contain
only fixed labels, counts and hashes. Existing instruments are imported unchanged.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from . import instruction_binding_analysis as analysis
from . import instruction_binding_plan as compiler
from . import instruction_binding_runtime as runtime
from . import instruction_binding_tasks as tasks

AMENDMENT = "a163-development-v2"
PHASES = {"initial": "development_pilot", "remaining": "development"}
EXAMPLES = (
    tasks.LookupWorld(1, (("s4", "s2", "s7", "s0", "s6", "s1", "s5", "s3"),),
                      ("s1", "s6")),
    tasks.LookupWorld(2, (("s2", "s5", "s1", "s7", "s0", "s6", "s3", "s4"),
                         ("s6", "s3", "s5", "s0", "s7", "s2", "s4", "s1")),
                      ("s0", "s3")),
)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode() + b"\n"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def source_hashes() -> dict[str, str]:
    modules = (analysis, compiler, runtime, tasks)
    paths = [Path(__file__), *(Path(module.__file__) for module in modules)]
    return {path.name: sha(path.read_bytes()) for path in paths}


def worked_examples(renderer: str) -> str:
    """The same two worlds, with both selector solutions, in a fixed syntax."""
    require(renderer in tasks.RENDERERS, "example_renderer")
    sections = [
        "Worked examples follow. They are demonstrations, not the live lookup world. "
        "Each example shows the answer for both possible selectors. For the live task, "
        "use only the authoritative selector stated above and the world in the user message."
    ]
    for world in EXAMPLES:
        sections.append(f"Example at depth {world.depth}:\n" + tasks.render_world(
            world, renderer, query_order="AB", table_order="forward"))
        for selector, key in zip(tasks.SELECTORS, world.query_keys, strict=True):
            chain, value = [key], key
            for table in world.tables:
                value = table[tasks.SYMBOLS.index(value)]
                chain.append(value)
            sections.append(
                f"If this example's selector is {selector}, the lookup chain is "
                + " -> ".join(chain) + ". The entire required output is:\n"
                + json.dumps({"answer": value}, separators=(",", ":"))
            )
    sections.append(
        "Now solve the separate live world from the user message. Do not copy an example answer "
        "or show the lookup chain. Return only the live answer JSON object."
    )
    return "\n\n".join(sections)


def _sources(raw_plans: list[bytes], expected_sha256: list[str]) -> dict[str, dict]:
    require(type(raw_plans) is list and type(expected_sha256) is list
            and len(raw_plans) == len(expected_sha256) == 2, "two_development_halfplans_required")
    by_stage, worlds, identities = {}, set(), set()
    pins = source_hashes()
    for raw, expected in zip(raw_plans, expected_sha256, strict=True):
        require(type(raw) is bytes and sha(raw) == expected, "source_plan_hash")
        plan = json.loads(raw)
        require(plan.get("partition") == "development" and plan.get("status") == "outcome_free"
                and plan.get("stage") in PHASES.values(), "development_only_source")
        require(plan["stage"] not in by_stage, "duplicate_source_stage")
        runtime.validate_plan(plan)
        require(plan.get("world_count") == 8 and plan.get("trial_count") == 160
                and len(plan["trials"]) == 160, "complete_source_halfplan_required")
        bindings = plan["bindings"]
        require(bindings.get("compiler_sha256") == pins["instruction_binding_plan.py"]
                and bindings.get("generator_sha256") == pins["instruction_binding_tasks.py"],
                "source_compiler_binding")
        groups = {}
        for trial in plan["trials"]:
            require(trial["trial_id"] not in identities, "duplicate_source_trial")
            identities.add(trial["trial_id"])
            groups.setdefault(trial["world_id"], []).append(trial)
            require(trial["messages"][0]["content"] == tasks.system_instruction(trial["selector"]),
                    "unmodified_system_instruction_required")
        require(len(groups) == 8 and not worlds.intersection(groups), "cross_half_world_overlap")
        worlds.update(groups)
        require(Counter(rows[0]["depth"] for rows in groups.values()) == {1: 4, 2: 4},
                "source_depth_balance")
        expected_cells = {("json", "none", "none", selector, False) for selector in tasks.SELECTORS}
        expected_cells |= {("json", kind, placement, selector, False)
                           for kind in tasks.SCAFFOLD_KINDS for placement in ("before", "after")
                           for selector in tasks.SELECTORS}
        expected_cells |= {("line_table", "none", "none", selector, True)
                           for selector in tasks.SELECTORS}
        for rows in groups.values():
            observed = {(r["renderer"], r["scaffold_kind"], r["placement"], r["selector"],
                         r["diagnostic"]) for r in rows}
            require(len(rows) == 20 and observed == expected_cells, "source_matrix_incomplete")
            require(len({(r["query_order"], r["table_order"], r["depth"]) for r in rows}) == 1,
                    "source_presentation_inconsistency")
            for renderer in tasks.RENDERERS:
                pair = [r for r in rows if r["renderer"] == renderer and r["scaffold_kind"] == "none"]
                require(pair[0]["messages"][1] == pair[1]["messages"][1], "source_user_pair_mismatch")
        by_stage[plan["stage"]] = {"plan": plan, "sha256": expected}
    require(set(by_stage) == set(PHASES.values()), "both_development_halves_required")
    first, second = (by_stage[stage]["plan"] for stage in PHASES.values())
    require(first["bindings"] == second["bindings"]
            and first["material_receipts"] == second["material_receipts"],
            "source_halves_different_instrument")
    require(not worlds.intersection(example.world_id for example in EXAMPLES), "example_world_overlap")
    return by_stage


def _construct(sources: dict, *, phase: str, protocol_sha256: str,
               other_known_world_ids: list[str]) -> dict:
    source = sources[PHASES[phase]]
    example_ids = [world.world_id for world in EXAMPLES]
    require(all(analysis._digest(value) for value in other_known_world_ids)
            and len(other_known_world_ids) == len(set(other_known_world_ids)), "external_world_ids")
    require(not set(example_ids).intersection(other_known_world_ids), "example_world_overlap")
    pins = source_hashes()
    identity = {"amendment": AMENDMENT, "phase": phase,
                "source_plans": {stage: value["sha256"] for stage, value in sources.items()},
                "source_hashes": pins, "protocol_sha256": protocol_sha256,
                "example_world_ids": example_ids,
                "external_world_ids_sha256": sha(canonical(sorted(other_known_world_ids)))}
    plan = copy.deepcopy(source["plan"])
    trials = []
    for original in source["plan"]["trials"]:
        if original["scaffold_kind"] != "none":
            continue
        trial = copy.deepcopy(original)
        trial["source_trial_id"] = original["trial_id"]
        trial["messages"][0]["content"] += "\n\n" + worked_examples(trial["renderer"])
        trial["trial_id"] = sha(canonical({"identity": identity, "source_trial_id": original["trial_id"],
                                           "messages": trial["messages"]}))[:24]
        trials.append(trial)
    require(len(trials) == 32 and len({row["trial_id"] for row in trials}) == 32, "baseline_matrix")
    plan.update(trials=trials, trial_count=32)
    plan["bindings"].update(amendment_protocol_sha256=protocol_sha256,
                            amendment_source_sha256=pins[Path(__file__).name],
                            source_plan_sha256=source["sha256"])
    plan["development_amendment"] = {
        **identity, "example_world_payloads_sha256": sha(canonical([w.payload() for w in EXAMPLES])),
        "external_world_count_checked": len(other_known_world_ids),
        "root_full_cohort_disjointness_check_required": True,
        "launch_requires_failed_v1_competence_gate": True,
        "launch_requires_review_freeze_and_source_backup": True,
        "initial_gate": None,
    }
    return plan


def build_baseline_plan(raw_plans: list[bytes], *, expected_sha256: list[str], phase: str,
                        protocol_sha256: str, other_known_world_ids: list[str],
                        initial_records: list[dict] | None = None) -> dict:
    """Prepare exactly one wording amendment from the two original dev halves.

    The caller supplies any known qualification/held-out IDs for a hash-only
    disjointness check. This module never opens or generates held-out material;
    root must additionally verify that the supplied ID inventory is complete.
    Remaining-half construction requires receipt-backed initial-half gate data.
    """
    require(phase in PHASES and analysis._digest(protocol_sha256), "amendment_identity")
    sources = _sources(raw_plans, expected_sha256)
    plan = _construct(sources, phase=phase, protocol_sha256=protocol_sha256,
                      other_known_world_ids=other_known_world_ids)
    if phase == "initial":
        require(initial_records is None, "initial_phase_must_be_outcome_free")
    else:
        require(initial_records is not None, "initial_gate_required")
        initial = _construct(sources, phase="initial", protocol_sha256=protocol_sha256,
                             other_known_world_ids=other_known_world_ids)
        summary = summarize_baseline_records(initial, initial_records)
        require(summary["gate_passed"], "initial_competence_gate_failed")
        plan["development_amendment"]["initial_gate"] = {
            "summary_sha256": sha(canonical(summary)),
            "records_sha256": sha(canonical(initial_records)),
            "initial_plan_sha256": sha(canonical(initial)),
        }
    return plan


def summarize_baseline_records(plan: dict, records: list[dict]) -> dict:
    """Numeric baseline denominators only; no selection on scaffold responses."""
    require(plan.get("partition") == "development" and plan.get("world_count") == 8
            and plan.get("trial_count") == 32 and len(plan.get("trials", [])) == 32,
            "baseline_plan_shape")
    amendment = plan.get("development_amendment", {})
    require(amendment.get("amendment") == AMENDMENT and amendment.get("phase") in PHASES,
            "baseline_amendment_identity")
    require(amendment.get("source_hashes") == source_hashes(), "amendment_source_drift")
    runtime.validate_plan(plan)
    identity_keys = ("amendment", "phase", "source_plans", "source_hashes", "protocol_sha256",
                     "example_world_ids", "external_world_ids_sha256")
    identity = {key: amendment[key] for key in identity_keys}
    require(amendment["example_world_ids"] == [world.world_id for world in EXAMPLES]
            and plan["stage"] == PHASES[amendment["phase"]], "amendment_example_or_phase_drift")
    expected = []
    for trial in plan["trials"]:
        require(trial["scaffold_kind"] == "none" and trial["placement"] == "none", "baseline_only")
        require(trial["messages"][0]["content"] == tasks.system_instruction(trial["selector"])
                + "\n\n" + worked_examples(trial["renderer"]), "fixed_amended_system_drift")
        require(trial["messages"][1]["content"] == tasks.render_world(
            runtime._world(trial), trial["renderer"], query_order=trial["query_order"],
            table_order=trial["table_order"]), "baseline_user_payload_drift")
        require(trial["trial_id"] == sha(canonical({"identity": identity,
                    "source_trial_id": trial["source_trial_id"], "messages": trial["messages"]}))[:24],
                "amended_trial_id_drift")
        expected.append({field: runtime.numeric_record(plan, trial)[field] for field in analysis.META_FIELDS})
    require(len({(r["world_id"], r["renderer"], r["selector"]) for r in expected}) == 32,
            "duplicate_baseline_cell")
    rows = analysis._validate_records({"expected_cells": expected}, records)
    indexed = {record["trial_id"]: record for record in rows.values()}

    def counts(cells):
        found = [indexed[row["trial_id"]] for row in cells if row["trial_id"] in indexed]
        categories = Counter(row["score"]["category"] for row in found
                             if row["generation_status"] == "completed")
        worlds = {row["world_id"] for row in cells}
        pairs, pair_completed = 0, 0
        for world in worlds:
            pair = [indexed.get(row["trial_id"]) for row in cells if row["world_id"] == world]
            require(len(pair) == 2, "paired_selector_denominator")
            pair_completed += all(row is not None and row["generation_status"] == "completed" for row in pair)
            pairs += all(row is not None and row["score"]["strict_correct"] is True for row in pair)
        correct = sum(row["score"]["strict_correct"] is True for row in found)
        completed = sum(row["generation_status"] == "completed" for row in found)
        ll_ok = sum(row["teacher_forced"]["status"] == "ok" for row in found)
        return {
            "planned_cells": len(cells), "recorded_cells": len(found), "missing_cells": len(cells) - len(found),
            "generation_completed": completed, "infrastructure_failed": len(found) - completed,
            "strict_correct": correct, "strict_correct_fraction": correct / len(cells),
            "length_terminated": sum(row["finish_reason"] == "length" for row in found),
            "categories": {category: categories[category] for category in analysis.CATEGORIES},
            "planned_pairs": len(worlds), "completed_pairs": pair_completed,
            "both_selectors_correct": pairs, "pair_correct_fraction": pairs / len(worlds),
            "likelihood_ok": ll_ok, "likelihood_failed": len(found) - ll_ok,
            "likelihood_missing": len(cells) - len(found),
            "gate_passed": completed == len(cells) and correct * 10 >= 9 * len(cells)
                           and pairs * 10 >= 9 * len(worlds),
        }

    renderers = {}
    for renderer in tasks.RENDERERS:
        cells = [row for row in expected if row["renderer"] == renderer]
        require(len(cells) == 16 and Counter(row["task_depth"] for row in cells) == {1: 8, 2: 8},
                "baseline_renderer_depth_balance")
        renderers[renderer] = {**counts(cells), "by_depth": {
            str(depth): counts([row for row in cells if row["task_depth"] == depth]) for depth in (1, 2)
        }}
    return {
        "schema_version": "a163-development-v2-summary-v1", "phase": amendment["phase"],
        "plan_sha256": sha(canonical(plan)), "records_sha256": sha(canonical(records)),
        "source_hashes": source_hashes(), "planned_cells": 32, "recorded_cells": len(records),
        "gate_passed": all(row["gate_passed"] for row in renderers.values()),
        "renderers": renderers, "scaffold_effects_used": False, "heldout_used": False,
        "likelihood_is_gate": False, "additional_wording_repairs_allowed": 0,
    }
