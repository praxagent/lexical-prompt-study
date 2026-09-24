"""Lossless special-token metadata inventory, independent of tokenizer execution.

``all_special_ids`` describes named/extra attributes; it need not include every
backend AddedToken marked special. This module snapshots both metadata sources.
It never constructs a tokenizer, encodes/decodes text, reads assets or loads a
model. Metadata validation does not attest the tokenizer object's provenance.
Returned JSON-compatible dictionaries are defensive copies, not frozen objects.
"""

from __future__ import annotations

import hashlib
import json

SCHEMA = "special-token-inventory-v1"
TERMINAL_SCHEMA = "special-terminal-token-v1"


def _require(ok, code):
    if not ok:
        raise ValueError("special_token_inventory_" + code)


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, "integer")


def _ids(values, vocab_size, *, nonempty=False):
    _require(type(values) is list and (bool(values) or not nonempty), "id_list")
    for token_id in values:
        _integer(token_id)
        _require(token_id < vocab_size, "id_range")
    _require(len(values) == len(set(values)), "duplicate_id")
    return sorted(values)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _build(vocab_size, declared_ids, entries):
    _integer(vocab_size, 1)
    declared = _ids(declared_ids, vocab_size)
    _require(type(entries) is list, "added_entries")
    added = []
    for entry in entries:
        _require(
            type(entry) is dict and set(entry) == {"token_id", "special"}, "added_entry_fields"
        )
        _integer(entry["token_id"])
        _require(entry["token_id"] < vocab_size, "added_id_range")
        _require(type(entry["special"]) is bool, "special_flag")
        added.append({"token_id": entry["token_id"], "special": entry["special"]})
    _require(len(added) == len({entry["token_id"] for entry in added}), "duplicate_added_id")
    added.sort(key=lambda entry: entry["token_id"])
    backend_special = [entry["token_id"] for entry in added if entry["special"]]
    return {
        "schema_version": SCHEMA,
        "vocab_size": vocab_size,
        "declared_special_ids": declared,
        "added_tokens": added,
        "added_special_ids": backend_special,
        "special_ids": sorted(set(declared) | set(backend_special)),
    }


def collect_special_token_inventory(tokenizer, *, vocab_size):
    """Read actual API attributes once; never interpret raw tokenizer JSON.

    ``added_tokens_decoder`` must be an exact dict keyed by integer token IDs.
    Each value must expose an exact Boolean ``special`` attribute, as the native
    AddedToken API does. No token strings are copied. The interface is checked
    structurally so invented objects can be qualified without importing tokenizers.
    """
    _integer(vocab_size, 1)
    declared = tokenizer.all_special_ids
    _ids(declared, vocab_size)
    declared = list(declared)
    decoder = tokenizer.added_tokens_decoder
    _require(type(decoder) is dict, "added_decoder_type")
    entries = []
    for token_id, token in decoder.items():
        _integer(token_id)
        _require(token_id < vocab_size, "added_id_range")
        flag = getattr(token, "special", None)
        _require(type(flag) is bool, "added_token_special_attribute")
        entries.append({"token_id": token_id, "special": flag})
    return _build(vocab_size, declared, entries)


def validate_inventory(inventory):
    """Reconstruct all derived fields, requiring canonical order and exact types."""
    _require(
        type(inventory) is dict
        and set(inventory)
        == {
            "schema_version",
            "vocab_size",
            "declared_special_ids",
            "added_tokens",
            "added_special_ids",
            "special_ids",
        },
        "inventory_fields",
    )
    expected = _build(
        inventory["vocab_size"], inventory["declared_special_ids"], inventory["added_tokens"]
    )
    _require(type(inventory["schema_version"]) is str, "schema_type")
    _ids(inventory["added_special_ids"], inventory["vocab_size"])
    _ids(inventory["special_ids"], inventory["vocab_size"])
    _require(_canonical(inventory) == _canonical(expected), "inventory_binding")
    return expected


def inventory_sha256(inventory):
    return hashlib.sha256(_canonical(validate_inventory(inventory))).hexdigest()


def validate_terminal_token(inventory, *, terminal_token_id, stop_token_ids):
    """Check IDs after the caller establishes the explicit terminal token identity.

    Every declared generation stop and the supplied terminal ID must be in the
    complete union. This does not assert that a token ID denotes a particular
    spelling: the prospective native adapter must independently establish that.
    """
    inventory = validate_inventory(inventory)
    stops = _ids(stop_token_ids, inventory["vocab_size"], nonempty=True)
    _integer(terminal_token_id)
    _require(terminal_token_id < inventory["vocab_size"], "terminal_range")
    _require(set(stops) <= set(inventory["special_ids"]), "stops_not_special")
    _require(terminal_token_id in stops, "terminal_not_stop")
    return {
        "schema_version": TERMINAL_SCHEMA,
        "inventory_sha256": inventory_sha256(inventory),
        "terminal_token_id": terminal_token_id,
        "stop_token_ids": stops,
    }
