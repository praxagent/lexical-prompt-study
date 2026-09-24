"""Invented metadata interfaces only: no tokenizer library, assets or encoding."""

import copy
import unittest
from types import SimpleNamespace

from lexical_prompt_study import special_token_inventory as inventory


def fixture():
    # Deliberately invented IDs: two named attributes, three generation stops.
    return SimpleNamespace(
        all_special_ids=[2, 7],
        added_tokens_decoder={
            2: SimpleNamespace(special=True),
            7: SimpleNamespace(special=True),
            11: SimpleNamespace(special=True),
            13: SimpleNamespace(special=True),
            17: SimpleNamespace(special=True),
            19: SimpleNamespace(special=False),
        },
    )


class InventoryTests(unittest.TestCase):
    def test_two_declared_three_stops_and_extra_backend_special(self):
        result = inventory.collect_special_token_inventory(fixture(), vocab_size=32)
        self.assertEqual(result["declared_special_ids"], [2, 7])
        self.assertEqual(result["added_special_ids"], [2, 7, 11, 13, 17])
        self.assertEqual(result["special_ids"], [2, 7, 11, 13, 17])
        checked = inventory.validate_terminal_token(
            result, terminal_token_id=13, stop_token_ids=[13, 7, 11]
        )
        self.assertEqual(checked["stop_token_ids"], [7, 11, 13])
        self.assertEqual(checked["inventory_sha256"], inventory.inventory_sha256(result))
        self.assertNotIn(19, result["special_ids"])

    def test_declared_special_need_not_have_added_entry(self):
        tokenizer = fixture()
        tokenizer.all_special_ids.append(23)
        result = inventory.collect_special_token_inventory(tokenizer, vocab_size=32)
        self.assertIn(23, result["special_ids"])
        self.assertNotIn(23, result["added_special_ids"])

    def test_false_backend_flag_does_not_erase_declared_provenance(self):
        tokenizer = fixture()
        tokenizer.added_tokens_decoder[7].special = False
        result = inventory.collect_special_token_inventory(tokenizer, vocab_size=32)
        self.assertIn(7, result["special_ids"])
        self.assertNotIn(7, result["added_special_ids"])

    def test_snapshot_has_no_aliases(self):
        tokenizer = fixture()
        result = inventory.collect_special_token_inventory(tokenizer, vocab_size=32)
        saved = copy.deepcopy(result)
        tokenizer.all_special_ids.append(23)
        tokenizer.added_tokens_decoder[11].special = False
        tokenizer.added_tokens_decoder.clear()
        self.assertEqual(result, saved)
        cloned = inventory.validate_inventory(result)
        cloned["added_tokens"][0]["special"] = False
        self.assertEqual(result, saved)

    def test_deterministic_order_and_hash(self):
        a = fixture()
        b = fixture()
        b.all_special_ids.reverse()
        b.added_tokens_decoder = dict(reversed(list(b.added_tokens_decoder.items())))
        left = inventory.collect_special_token_inventory(a, vocab_size=32)
        right = inventory.collect_special_token_inventory(b, vocab_size=32)
        self.assertEqual(left, right)
        self.assertEqual(inventory.inventory_sha256(left), inventory.inventory_sha256(right))

    def test_no_token_strings_or_runtime_methods_accessed(self):
        class MetadataOnly:
            all_special_ids = [2]
            added_tokens_decoder = {2: SimpleNamespace(special=True)}

            def __getattr__(self, key):
                raise AssertionError("unexpected access")

        result = inventory.collect_special_token_inventory(MetadataOnly(), vocab_size=32)
        self.assertEqual(set(result["added_tokens"][0]), {"token_id", "special"})

    def test_empty_inventory_is_structural_not_terminal_qualification(self):
        result = inventory.collect_special_token_inventory(
            SimpleNamespace(all_special_ids=[], added_tokens_decoder={}), vocab_size=32
        )
        self.assertEqual(result["special_ids"], [])
        with self.assertRaises(ValueError):
            inventory.validate_terminal_token(result, terminal_token_id=1, stop_token_ids=[1])


def invalid_collect(change):
    def test(self):
        tokenizer = fixture()
        vocab = 32
        if change == "bool_vocab":
            vocab = True
        elif change == "zero_vocab":
            vocab = 0
        elif change == "float_vocab":
            vocab = 32.0
        elif change == "declared_bool":
            tokenizer.all_special_ids = [True]
        elif change == "declared_negative":
            tokenizer.all_special_ids = [-1]
        elif change == "declared_range":
            tokenizer.all_special_ids = [32]
        elif change == "declared_float":
            tokenizer.all_special_ids = [2.0]
        elif change == "declared_tuple":
            tokenizer.all_special_ids = (2, 7)
        elif change == "declared_duplicate":
            tokenizer.all_special_ids = [2, 2]
        elif change == "decoder_list":
            tokenizer.added_tokens_decoder = []
        elif change == "decoder_string_id":
            tokenizer.added_tokens_decoder = {"2": SimpleNamespace(special=True)}
        elif change == "decoder_bool_id":
            tokenizer.added_tokens_decoder = {True: SimpleNamespace(special=True)}
        elif change == "decoder_range":
            tokenizer.added_tokens_decoder = {32: SimpleNamespace(special=True)}
        elif change == "flag_integer":
            tokenizer.added_tokens_decoder = {2: SimpleNamespace(special=1)}
        elif change == "flag_string":
            tokenizer.added_tokens_decoder = {2: SimpleNamespace(special="true")}
        elif change == "raw_json_dict":
            tokenizer.added_tokens_decoder = {2: {"special": True}}
        elif change == "flag_missing":
            tokenizer.added_tokens_decoder = {2: object()}
        with self.assertRaises(ValueError):
            inventory.collect_special_token_inventory(tokenizer, vocab_size=vocab)

    return test


for name in (
    "bool_vocab",
    "zero_vocab",
    "float_vocab",
    "declared_bool",
    "declared_negative",
    "declared_range",
    "declared_float",
    "declared_tuple",
    "declared_duplicate",
    "decoder_list",
    "decoder_string_id",
    "decoder_bool_id",
    "decoder_range",
    "flag_integer",
    "flag_string",
    "raw_json_dict",
    "flag_missing",
):
    setattr(InventoryTests, "test_collect_reject_" + name, invalid_collect(name))


def invalid_snapshot(change):
    def test(self):
        result = inventory.collect_special_token_inventory(fixture(), vocab_size=32)
        if change == "union_missing":
            result["special_ids"].remove(17)
        elif change == "union_extra":
            result["special_ids"].append(23)
        elif change == "union_bool":
            result["special_ids"][0] = True
        elif change == "union_tuple":
            result["special_ids"] = tuple(result["special_ids"])
        elif change == "backend_tuple":
            result["added_special_ids"] = tuple(result["added_special_ids"])
        elif change == "schema_subclass":

            class StringSubclass(str):
                pass

            result["schema_version"] = StringSubclass(inventory.SCHEMA)
        elif change == "order":
            result["added_tokens"].reverse()
        elif change == "duplicate":
            result["added_tokens"].append(copy.deepcopy(result["added_tokens"][0]))
        elif change == "flag_integer":
            result["added_tokens"][0]["special"] = 1
        elif change == "backend_list":
            result["added_special_ids"].remove(11)
        elif change == "unknown_field":
            result["token_content"] = "invented"
        elif change == "schema":
            result["schema_version"] = "different"
        with self.assertRaises(ValueError):
            inventory.validate_inventory(result)

    return test


for name in (
    "union_missing",
    "union_extra",
    "union_bool",
    "union_tuple",
    "backend_tuple",
    "schema_subclass",
    "order",
    "duplicate",
    "flag_integer",
    "backend_list",
    "unknown_field",
    "schema",
):
    setattr(InventoryTests, "test_snapshot_reject_" + name, invalid_snapshot(name))


def invalid_terminal(terminal, stops):
    def test(self):
        result = inventory.collect_special_token_inventory(fixture(), vocab_size=32)
        with self.assertRaises(ValueError):
            inventory.validate_terminal_token(
                result, terminal_token_id=terminal, stop_token_ids=stops
            )

    return test


for name, terminal, stops in (
    ("bool_terminal", True, [7]),
    ("float_terminal", 7.0, [7]),
    ("out_of_range", 32, [7]),
    ("not_stop", 13, [7]),
    ("ordinary_stop", 13, [13, 19]),
    ("empty_stops", 13, []),
    ("bool_stop", 13, [True, 13]),
    ("duplicate_stop", 13, [13, 13]),
    ("float_stop", 13, [13.0]),
):
    setattr(InventoryTests, "test_terminal_reject_" + name, invalid_terminal(terminal, stops))


if __name__ == "__main__":
    unittest.main()
