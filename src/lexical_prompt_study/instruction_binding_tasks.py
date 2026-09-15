"""Deterministic harmless lookup tasks for a prospective instruction-binding study.

This module generates no model responses and reads no artifacts. A task world is
the independent sampling unit; its selector, renderer, and scaffold conditions
are repeated observations. Renderer competence must be qualified separately on
development worlds before interpreting an experiment.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

SYMBOLS = tuple(f"s{index}" for index in range(8))
SELECTORS = ("A", "B")
RENDERERS = ("json", "line_table")
SCAFFOLD_KINDS = ("full", "sham", "replacement", "inert")
Selector = Literal["A", "B"]
Renderer = Literal["json", "line_table"]
ScoreCategory = Literal["exact", "other_selector", "other_answer", "format"]


def _selector(value: str) -> None:
    if type(value) is not str or value not in SELECTORS:
        raise ValueError("selector must be A or B")


@dataclass(frozen=True)
class LookupWorld:
    """Each table stores its eight outputs in the fixed SYMBOLS input order."""

    depth: int
    tables: tuple[tuple[str, ...], ...]
    query_keys: tuple[str, str]

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        if type(self.tables) is not tuple or len(self.tables) != self.depth:
            raise ValueError("one immutable permutation table is required per hop")
        for table in self.tables:
            if (
                type(table) is not tuple
                or len(table) != len(SYMBOLS)
                or any(type(symbol) is not str for symbol in table)
                or set(table) != set(SYMBOLS)
            ):
                raise ValueError("each table must permute all eight symbols")
        if (
            type(self.query_keys) is not tuple
            or len(self.query_keys) != 2
            or any(type(key) is not str or key not in SYMBOLS for key in self.query_keys)
            or self.query_keys[0] == self.query_keys[1]
        ):
            raise ValueError("queries A and B require distinct symbol keys")

    def payload(self) -> dict:
        """Only task content; no active selector, renderer, scaffold, or seed."""
        return {
            "depth": self.depth,
            "queries": dict(zip(SELECTORS, self.query_keys, strict=True)),
            "tables": {
                f"table_{hop}": dict(zip(SYMBOLS, table, strict=True))
                for hop, table in enumerate(self.tables, start=1)
            },
        }

    @property
    def world_id(self) -> str:
        content = {"schema_version": "instruction-binding-world-v1", **self.payload()}
        raw = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _draw_world(rng: random.Random, depth: int) -> LookupWorld:
    tables = []
    for _ in range(depth):
        outputs = list(SYMBOLS)
        rng.shuffle(outputs)
        tables.append(tuple(outputs))
    keys = rng.sample(SYMBOLS, 2)
    return LookupWorld(depth=depth, tables=tuple(tables), query_keys=tuple(keys))


def generate_worlds(*, seed: int, count: int, depth: int) -> tuple[LookupWorld, ...]:
    """Generate a reproducible cohort with distinct content-derived world IDs.

    Uniqueness is checked on content, not seed or a counter. Collisions are
    resampled, and exhaustion fails instead of returning a smaller cohort.
    Random state is local and does not affect the caller's global generator.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(depth) is not int or depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    rng = random.Random(seed)
    worlds: dict[str, LookupWorld] = {}
    for _ in range(count * 100):
        world = _draw_world(rng, depth)
        worlds.setdefault(world.world_id, world)
        if len(worlds) == count:
            return tuple(worlds.values())
    raise RuntimeError("could not construct the requested distinct-world cohort")


def make_world(*, seed: int, depth: int) -> LookupWorld:
    return generate_worlds(seed=seed, count=1, depth=depth)[0]


def oracle(world: LookupWorld, selector: Selector) -> str:
    """Compute the selected answer at the world's actual depth."""
    _selector(selector)
    value = world.query_keys[SELECTORS.index(selector)]
    for table in world.tables:
        value = table[SYMBOLS.index(value)]
    return value


def render_world(world: LookupWorld, renderer: Renderer) -> str:
    """Render both candidate queries with no active-selector instruction."""
    if type(renderer) is not str or renderer not in RENDERERS:
        raise ValueError("unknown renderer")
    if renderer == "json":
        return json.dumps(world.payload(), indent=2, sort_keys=True)
    lines = [f"depth: {world.depth}", f"query A: {world.query_keys[0]}",
             f"query B: {world.query_keys[1]}"]
    for hop, table in enumerate(world.tables, start=1):
        lines.append(f"table_{hop} (input -> output):")
        lines.extend(f"{key} -> {value}" for key, value in zip(SYMBOLS, table, strict=True))
    return "\n".join(lines)


def system_instruction(selector: Selector) -> str:
    _selector(selector)
    return (
        f"The authoritative query selector is {selector}. Evaluate only that query "
        "in the user-supplied lookup world. Start with its query key and apply "
        "table_1. At depth 2, use that result as the input to table_2. The result "
        'after the final table is the answer. Output exactly one JSON object with '
        'the single field "answer" containing that symbol as a string. '
        'Do not include other fields, explanations, or Markdown fences.'
    )


@dataclass(frozen=True)
class Condition:
    world_id: str
    renderer: Renderer
    scaffold_kind: str
    placement: str
    selector: Selector
    system_message: str
    user_message: str

    @property
    def condition_id(self) -> str:
        return ":".join((self.world_id, self.renderer, self.scaffold_kind,
                         self.placement, self.selector))

    @property
    def messages(self) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
        )


def build_conditions(
    world: LookupWorld, *, renderer: Renderer, scaffolds: Mapping[str, str]
) -> tuple[Condition, ...]:
    """Build 18 cells for one renderer, preserving supplied scaffold bytes.

    The caller supplies all four scaffold texts. This code neither loads nor
    reconstructs study scaffolds. A/B counterparts have identical user messages;
    only their authoritative system selector changes. The no-scaffold control
    has one placement ('none') rather than two duplicated control cells.
    """
    if not isinstance(scaffolds, Mapping) or set(scaffolds) != set(SCAFFOLD_KINDS):
        raise ValueError("provide exactly full, sham, replacement, and inert scaffolds")
    if any(type(text) is not str or not text.strip() for text in scaffolds.values()):
        raise ValueError("each supplied scaffold must be a nonempty string")
    rendered = render_world(world, renderer)
    cells = []
    variants = [("none", "none", rendered)]
    for kind in SCAFFOLD_KINDS:
        variants.extend(((kind, "before", scaffolds[kind] + "\n\n" + rendered),
                         (kind, "after", rendered + "\n\n" + scaffolds[kind])))
    for kind, placement, user in variants:
        for selector in SELECTORS:
            cells.append(Condition(
                world_id=world.world_id, renderer=renderer, scaffold_kind=kind,
                placement=placement, selector=selector,
                system_message=system_instruction(selector), user_message=user,
            ))
    return tuple(cells)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonstandard JSON constant")


def score_response(response: str, world: LookupWorld, selector: Selector) -> ScoreCategory:
    """Score with precedence format > exact > other_selector > other_answer.

    Format requires one complete JSON object with exactly one string field
    named 'answer'. Surrounding JSON whitespace is permitted; duplicate keys,
    extra fields, nonstandard constants, fences, or trailing content are not.
    A well-formed other string (including a symbol outside the world) is an
    'other_answer'. No semantic repair is attempted: even a correct or alternate
    answer inside a malformed response stays 'format', never selector rebinding.
    Any separate semantic diagnosis must retain this primary category.
    """
    _selector(selector)
    if type(response) is not str:
        return "format"
    try:
        value = json.loads(response, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return "format"
    if type(value) is not dict or set(value) != {"answer"} or type(value["answer"]) is not str:
        return "format"
    if value["answer"] == oracle(world, selector):
        return "exact"
    other_selector = "B" if selector == "A" else "A"
    if value["answer"] == oracle(world, other_selector):
        return "other_selector"
    return "other_answer"
