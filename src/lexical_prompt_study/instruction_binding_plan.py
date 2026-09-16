"""Compile private, outcome-free A163 plans without exposing restricted material."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .instruction_binding_tasks import build_conditions, generate_worlds, oracle

MATERIAL_SHA256 = "ee899a47ceca1b53e90787c5279174412e8d01e5872db9bba44a486921eba345"
SCHEMA_VERSION = "a163-plan-v1"
DEV_SEED = 2026091600
TEST_SEED = 2026091700


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def assemble_materials(payload: dict) -> dict[str, str]:
    """Reuse existing blocks; never reconstruct or emit their contents to logs."""
    joiner = payload["material_block_joiner"]
    if not isinstance(joiner, str):
        raise ValueError("invalid material joiner")
    variants = payload["scaffold_materials"]
    blocks = {}
    for name in ("full_scaffold", "structural_sham", "inert_length"):
        rows = variants[name]["blocks"]
        if len(rows) != 4 or any(not isinstance(row.get("text"), str) for row in rows):
            raise ValueError("material block topology mismatch")
        blocks[name] = [row["text"] for row in rows]
    return {
        "full": joiner.join(blocks["full_scaffold"]),
        "sham": joiner.join(blocks["structural_sham"]),
        "replacement": joiner.join(blocks["full_scaffold"][:3] + blocks["inert_length"][3:]),
        "inert": joiner.join(blocks["inert_length"]),
    }


def load_materials(path: Path, expected_sha256: str = MATERIAL_SHA256) -> dict[str, str]:
    data = path.read_bytes()
    if digest(data) != expected_sha256:
        raise ValueError("material source hash mismatch")
    return assemble_materials(json.loads(data))


def world_cohort(partition: str) -> tuple:
    if partition == "development":
        seed, count = DEV_SEED, 8
    elif partition == "heldout":
        seed, count = TEST_SEED, 32
    else:
        raise ValueError("unknown partition")
    result = tuple(world for depth in (1, 2)
                   for world in generate_worlds(seed=seed + depth, count=count, depth=depth))
    if len({world.world_id for world in result}) != len(result):
        raise ValueError("duplicate world in cohort")
    return result


def compile_plan(*, stage: str, scaffolds: dict[str, str], bindings: dict) -> dict:
    """Pilot and remainder are disjoint halves of a fixed development cohort."""
    if stage not in ("development_pilot", "development", "heldout"):
        raise ValueError("unknown stage")
    partition = "heldout" if stage == "heldout" else "development"
    all_worlds = world_cohort(partition)
    if partition == "heldout":
        dev_ids = {world.world_id for world in world_cohort("development")}
        if dev_ids & {world.world_id for world in all_worlds}:
            raise ValueError("development/heldout world overlap")
    by_depth = {depth: [w for w in all_worlds if w.depth == depth] for depth in (1, 2)}
    selected = []
    for depth in (1, 2):
        for index, world in enumerate(by_depth[depth]):
            if stage == "development_pilot" and index >= 4:
                continue
            if stage == "development" and index < 4:
                continue
            selected.append((index, world))
    main_renderer = "line_table" if partition == "heldout" else "json"
    trials = []
    for index, world in selected:
        query_order = "AB" if index % 2 == 0 else "BA"
        table_order = "reverse" if world.depth == 2 and (index // 2) % 2 else "forward"
        renderers = [main_renderer] if partition == "heldout" else [main_renderer, "line_table"]
        for renderer in renderers:
            diagnostic = renderer != main_renderer
            conditions = build_conditions(world, renderer=renderer, scaffolds=scaffolds,
                                          query_order=query_order, table_order=table_order)
            for cell in conditions:
                if diagnostic and cell.scaffold_kind != "none":
                    continue
                trials.append({
                    "trial_id": digest((stage + ":" + cell.condition_id).encode())[:24],
                    "world_id": world.world_id, "depth": world.depth,
                    "selector": cell.selector, "renderer": renderer,
                    "query_order": query_order, "table_order": table_order,
                    "scaffold_kind": cell.scaffold_kind, "placement": cell.placement,
                    "messages": list(cell.messages), "diagnostic": diagnostic,
                    "selected_answer": oracle(world, cell.selector),
                    "unselected_answer": oracle(world, "B" if cell.selector == "A" else "A"),
                    "world": world.payload(),
                })
    if len({row["trial_id"] for row in trials}) != len(trials):
        raise ValueError("duplicate planned trial")
    return {
        "schema_version": SCHEMA_VERSION, "stage": stage, "partition": partition,
        "status": "outcome_free", "world_count": len(selected),
        "trial_count": len(trials), "trials": trials,
        "bindings": {**bindings, "cohort_sha256": digest(canonical([
            {"world_id": world.world_id, **world.payload()} for world in all_worlds
        ]))},
        "material_receipts": {
            name: {"sha256": digest(value.encode()), "bytes": len(value.encode())}
            for name, value in scaffolds.items()
        },
        "policy": {"max_new_tokens": 64, "automatic_retries": 0,
                   "max_explicit_infrastructure_retries": 2,
                   "heldout_worlds": 64, "bootstrap_worlds": True,
                   "fixed_sample_size": True, "raw_data_private": True},
    }


def compile_qualification(*, bindings: dict) -> dict:
    """Sixteen harmless calls: eight unique native inputs, each repeated twice."""
    worlds = tuple(generate_worlds(seed=2026091550 + depth, count=1, depth=depth)[0]
                   for depth in (1, 2))
    protected = {w.world_id for partition in ("development", "heldout")
                 for w in world_cohort(partition)}
    if protected & {w.world_id for w in worlds}:
        raise ValueError("qualification world overlaps experimental cohort")
    materials = {kind: "Synthetic unused fixture." for kind in
                 ("full", "sham", "replacement", "inert")}
    trials = []
    for world in worlds:
        for renderer in ("json", "line_table"):
            for cell in build_conditions(world, renderer=renderer, scaffolds=materials):
                if cell.scaffold_kind != "none":
                    continue
                for repeat in (0, 1):
                    trials.append({
                        "trial_id": digest(("qualification:" + cell.condition_id +
                                            f":{repeat}").encode())[:24],
                        "world_id": world.world_id, "depth": world.depth,
                        "selector": cell.selector, "renderer": renderer,
                        "query_order": "AB", "table_order": "forward",
                        "scaffold_kind": "none", "placement": "none",
                        "messages": list(cell.messages), "diagnostic": True,
                        "selected_answer": oracle(world, cell.selector),
                        "unselected_answer": oracle(world, "B" if cell.selector == "A" else "A"),
                        "world": world.payload(), "repeat_index": repeat,
                    })
    return {"schema_version": SCHEMA_VERSION, "stage": "synthetic_qualification",
            "partition": "synthetic", "status": "outcome_free",
            "world_count": 2, "trial_count": len(trials), "trials": trials,
            "bindings": {**bindings, "cohort_sha256": digest(canonical([
                {"world_id": world.world_id, **world.payload()} for world in worlds
            ]))},
            "policy": {"max_new_tokens": 64, "automatic_retries": 0,
                       "max_explicit_infrastructure_retries": 2,
                       "repeat_likelihood_absolute_tolerance": 1e-5,
                       "require_exact_repeated_generated_tokens": True,
                       "accuracy_is_descriptive_not_runtime_integrity": True}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development_pilot", "development", "heldout"),
                        required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    source_dir = Path(__file__).parent
    bindings = {"materials_sha256": MATERIAL_SHA256,
                "protocol_sha256": digest(args.protocol.read_bytes()),
                "generator_sha256": digest((source_dir / "instruction_binding_tasks.py").read_bytes()),
                "compiler_sha256": digest(Path(__file__).read_bytes())}
    plan = compile_plan(stage=args.stage, scaffolds=load_materials(args.materials), bindings=bindings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(plan) + b"\n"
    with args.output.open("xb") as stream:
        stream.write(content)
    print(json.dumps({"status": "plan_compiled", "stage": args.stage,
                      "worlds": plan["world_count"], "trials": plan["trial_count"],
                      "plan_sha256": digest(content), "output": str(args.output)}))


if __name__ == "__main__":
    main()
