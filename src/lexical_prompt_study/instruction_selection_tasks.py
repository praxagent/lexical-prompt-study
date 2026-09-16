"""A164 direct answer selection: new worlds, balanced cohorts and private plans."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

from .instruction_binding_tasks import _reject_constant, _unique_object

SYMBOLS = tuple(f"s{index:02d}" for index in range(16))
SCAFFOLDS = ("full", "sham", "replacement", "inert")
FAMILY = "direct_answer_selection"
COHORT_SEED = 2026091800


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode() + b"\n"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def is_hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


@dataclass(frozen=True)
class AnswerWorld:
    value_a: str
    value_b: str

    def __post_init__(self):
        require(self.value_a in SYMBOLS and self.value_b in SYMBOLS
                and self.value_a != self.value_b, "distinct_symbol_values_required")

    def payload(self):
        return {"answers": {"A": self.value_a, "B": self.value_b}}

    @property
    def world_id(self):
        return sha(canonical({"schema_version": "a164-world-v1", **self.payload()}))


def world_from_payload(payload):
    require(type(payload) is dict and set(payload) == {"answers"}
            and type(payload["answers"]) is dict and set(payload["answers"]) == {"A", "B"},
            "world_schema")
    return AnswerWorld(payload["answers"]["A"], payload["answers"]["B"])


def oracle(world: AnswerWorld, selector: str) -> str:
    require(selector in ("A", "B"), "selector")
    return world.value_a if selector == "A" else world.value_b


def render_world(world: AnswerWorld, query_order: str) -> str:
    require(query_order in ("AB", "BA"), "query_order")
    return json.dumps({selector: oracle(world, selector) for selector in query_order},
                      separators=(",", ":"))


def system_instruction(selector: str) -> str:
    require(selector in ("A", "B"), "selector")
    return (
        f"The authoritative query selector is {selector}. The user supplies two explicit "
        "answer values labeled A and B. Copy the value with the authoritative label. "
        'Output exactly one JSON object with the single field "answer" containing that '
        "value as a string. Do not transform the value or include other fields, "
        "explanations, or Markdown fences."
    )


def make_cohort_manifest(seed: int = COHORT_SEED) -> dict:
    """Five disjoint derangements: one dev and four test; no outcome selection.

    This restricted randomization balances both label marginals. It does not
    produce independent draws from a general population of instruction tasks.
    """
    require(type(seed) is int, "cohort_seed")
    rng, used, rounds = random.Random(seed), set(), []
    for _ in range(5):
        for _attempt in range(100_000):
            outputs = list(SYMBOLS)
            rng.shuffle(outputs)
            pairs = tuple(zip(SYMBOLS, outputs, strict=True))
            if all(a != b and (a, b) not in used for a, b in pairs):
                used.update(pairs)
                rounds.append([AnswerWorld(a, b) for a, b in pairs])
                break
        else:
            raise RuntimeError("balanced_cohort_construction_failed")
    partitions = {}
    for name, worlds in (("development", rounds[0]),
                         ("heldout", [world for group in rounds[1:] for world in group])):
        rng.shuffle(worlds)
        partitions[name] = [{"world_id": world.world_id, "world": world.payload(),
                             "query_order": "AB" if index % 2 == 0 else "BA"}
                            for index, world in enumerate(worlds)]
    manifest = {"schema_version": "a164-cohort-v1", "seed": seed, "symbols": list(SYMBOLS),
                "sampling": "five_disjoint_seeded_derangements_then_shuffled_balanced_presentation",
                **partitions}
    validate_cohort_manifest(manifest)
    return manifest


def validate_cohort_manifest(manifest: dict) -> None:
    require(set(manifest) == {"schema_version", "seed", "symbols", "sampling", "development", "heldout"}
            and manifest["schema_version"] == "a164-cohort-v1"
            and manifest["symbols"] == list(SYMBOLS) and type(manifest["seed"]) is int
            and manifest["sampling"] == "five_disjoint_seeded_derangements_then_shuffled_balanced_presentation",
            "cohort_schema")
    all_ids = set()
    for partition, count, frequency in (("development", 16, 1), ("heldout", 64, 4)):
        rows = manifest[partition]
        require(type(rows) is list and len(rows) == count, "cohort_size")
        values_a, values_b, orders = Counter(), Counter(), Counter()
        for row in rows:
            require(set(row) == {"world_id", "world", "query_order"}, "cohort_row_schema")
            world = world_from_payload(row["world"])
            require(row["world_id"] == world.world_id and world.world_id not in all_ids,
                    "cohort_identity_or_overlap")
            all_ids.add(world.world_id)
            values_a[world.value_a] += 1
            values_b[world.value_b] += 1
            orders[row["query_order"]] += 1
        require(values_a == values_b == Counter(dict.fromkeys(SYMBOLS, frequency)), "symbol_balance")
        require(orders == {"AB": count // 2, "BA": count // 2}, "presentation_balance")


def compile_plan(manifest: dict, *, partition: str, scaffolds: dict[str, str], bindings: dict) -> dict:
    validate_cohort_manifest(manifest)
    require(partition in ("development", "heldout"), "partition")
    require(set(scaffolds) == set(SCAFFOLDS) and all(type(v) is str and v.strip() for v in scaffolds.values()),
            "four_caller_supplied_scaffolds_required")
    require(set(bindings) == {"protocol_sha256", "materials_sha256"}
            and all(is_hash(value) for value in bindings.values()), "plan_bindings")
    cohort_sha = sha(canonical(manifest))
    bound = {**bindings, "task_source_sha256": sha(Path(__file__).read_bytes())}
    material_receipts = {name: {"sha256": sha(value.encode()), "bytes": len(value.encode())}
                         for name, value in scaffolds.items()}
    trials = []
    for row in manifest[partition]:
        world = world_from_payload(row["world"])
        user = render_world(world, row["query_order"])
        variants = [("none", "none", user)]
        if partition == "heldout":
            variants.extend((kind, placement, scaffolds[kind] + "\n\n" + user if placement == "before"
                             else user + "\n\n" + scaffolds[kind])
                            for kind in SCAFFOLDS for placement in ("before", "after"))
        for kind, placement, content in variants:
            for selector in ("A", "B"):
                trial = {"world_id": world.world_id, "partition": partition, "task_family": FAMILY,
                         "renderer": "json", "query_order": row["query_order"],
                         "scaffold_kind": kind, "placement": placement, "selector": selector,
                         "messages": [{"role": "system", "content": system_instruction(selector)},
                                      {"role": "user", "content": content}],
                         "world": world.payload(), "selected_answer": oracle(world, selector),
                         "unselected_answer": oracle(world, "B" if selector == "A" else "A")}
                trial["trial_id"] = sha(canonical({"cohort_sha256": cohort_sha, "bindings": bound,
                                                   "material_receipts": material_receipts,
                                                   "trial": trial}))[:24]
                trials.append(trial)
    plan = {"schema_version": "a164-plan-v1", "stage": "development_baseline" if partition == "development" else "heldout",
            "partition": partition, "task_family": FAMILY, "renderer": "json",
            "world_count": len(manifest[partition]), "trial_count": len(trials),
            "cohort_sha256": cohort_sha, "cohort_manifest": manifest, "bindings": bound,
            "material_receipts": material_receipts, "trials": trials, "likelihood_collected": False}
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    require(set(plan) == {"schema_version", "stage", "partition", "task_family", "renderer", "world_count",
                         "trial_count", "cohort_sha256", "cohort_manifest", "bindings", "material_receipts",
                         "trials", "likelihood_collected"}, "plan_schema")
    partition = plan["partition"]
    require(plan["schema_version"] == "a164-plan-v1" and partition in ("development", "heldout")
            and plan["task_family"] == FAMILY and plan["renderer"] == "json"
            and plan["likelihood_collected"] is False, "plan_identity")
    require(plan["stage"] == ("development_baseline" if partition == "development" else "heldout"), "plan_stage")
    validate_cohort_manifest(plan["cohort_manifest"])
    require(plan["cohort_sha256"] == sha(canonical(plan["cohort_manifest"])), "cohort_binding")
    require(set(plan["bindings"]) == {"protocol_sha256", "materials_sha256", "task_source_sha256"}
            and all(is_hash(value) for value in plan["bindings"].values())
            and plan["bindings"]["task_source_sha256"] == sha(Path(__file__).read_bytes()), "source_binding")
    require(set(plan["material_receipts"]) == set(SCAFFOLDS), "material_receipts")
    for receipt in plan["material_receipts"].values():
        require(set(receipt) == {"sha256", "bytes"} and is_hash(receipt["sha256"])
                and type(receipt["bytes"]) is int and receipt["bytes"] > 0, "material_receipt")
    worlds = {row["world_id"]: row for row in plan["cohort_manifest"][partition]}
    variants = {("none", "none")}
    if partition == "heldout":
        variants |= {(kind, placement) for kind in SCAFFOLDS for placement in ("before", "after")}
    required = {(world, kind, placement, selector) for world in worlds
                for kind, placement in variants for selector in ("A", "B")}
    actual, ids, pairs = set(), set(), {}
    for trial in plan["trials"]:
        require(set(trial) == {"trial_id", "world_id", "partition", "task_family", "renderer", "query_order",
                              "scaffold_kind", "placement", "selector", "messages", "world",
                              "selected_answer", "unselected_answer"}, "trial_schema")
        world = world_from_payload(trial["world"])
        key = (world.world_id, trial["scaffold_kind"], trial["placement"], trial["selector"])
        require(key in required and key not in actual and trial["world_id"] == world.world_id, "trial_matrix")
        actual.add(key)
        require(trial["partition"] == partition and trial["task_family"] == FAMILY
                and trial["renderer"] == "json" and trial["query_order"] == worlds[world.world_id]["query_order"],
                "trial_metadata")
        require(trial["selected_answer"] == oracle(world, trial["selector"])
                and trial["unselected_answer"] == oracle(world, "B" if trial["selector"] == "A" else "A"), "oracle_binding")
        messages = trial["messages"]
        require(type(messages) is list and len(messages) == 2
                and messages[0] == {"role": "system", "content": system_instruction(trial["selector"])}
                and set(messages[1]) == {"role", "content"} and messages[1]["role"] == "user"
                and type(messages[1]["content"]) is str, "system_user_roles")
        content = messages[1]["content"]
        base = render_world(world, trial["query_order"])
        if trial["scaffold_kind"] == "none":
            require(content == base, "baseline_payload")
        else:
            before = trial["placement"] == "before"
            edge = "\n\n" + base if before else base + "\n\n"
            require(content.endswith(edge) if before else content.startswith(edge), "scaffold_placement")
            scaffold = content[:-len(edge)] if before else content[len(edge):]
            receipt = plan["material_receipts"][trial["scaffold_kind"]]
            require(sha(scaffold.encode()) == receipt["sha256"] and len(scaffold.encode()) == receipt["bytes"], "scaffold_binding")
        require(pairs.setdefault(key[:3], content) == content, "selector_user_invariance")
        identity = {k: value for k, value in trial.items() if k != "trial_id"}
        expected_id = sha(canonical({"cohort_sha256": plan["cohort_sha256"], "bindings": plan["bindings"],
                                    "material_receipts": plan["material_receipts"], "trial": identity}))[:24]
        require(trial["trial_id"] == expected_id and expected_id not in ids, "trial_id_binding")
        ids.add(expected_id)
    require(actual == required and plan["world_count"] == len(worlds)
            and plan["trial_count"] == len(required) == len(plan["trials"]), "complete_planned_matrix")


def score_response(response, selected: str, alternative: str) -> str:
    require(selected in SYMBOLS and alternative in SYMBOLS and selected != alternative, "answer_pair")
    if type(response) is not str:
        return "format"
    try:
        result = json.loads(response, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return "format"
    if type(result) is not dict or set(result) != {"answer"} or type(result["answer"]) is not str:
        return "format"
    return "exact" if result["answer"] == selected else (
        "other_selector" if result["answer"] == alternative else "other_answer")
