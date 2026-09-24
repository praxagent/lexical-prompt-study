"""One frozen, model-free descriptive analysis of five sealed A189 responses.

The existing terminal verification supplies native provenance. This reader does
not replay it, inspect partial responses, import a model, or change old scores.
Only aggregate diagnostics are emitted. All paths and whole-file hashes arrive
through a private, reviewed freeze, not through public source code.
"""

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import re
import sys
import time
import types


class EvidenceError(ValueError):
    """A fixed, content-free evidence or binding failure."""


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "regular_file_required")
    require(all(not p.is_symlink() for p in path.parents), "symlink_ancestor")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "file_changed_during_read",
    )
    return raw


def json_value(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def finite(_):
        raise EvidenceError("nonfinite_json")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    json.dumps(value, allow_nan=False)
    return value


def bound(path, sha256):
    require(type(sha256) is str and re.fullmatch(r"[0-9a-f]{64}", sha256), "invalid_hash")
    raw = read_regular(path)
    require(digest(raw) == sha256, "artifact_hash_mismatch")
    return raw


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_aggregates(value):
    require(
        type(value) is dict
        and set(value) == {"category_counts", "wrong_count_style_counts", "completed_responses"},
        "aggregate_fields",
    )
    require(
        type(value["completed_responses"]) is int and value["completed_responses"] == 5,
        "aggregate_denominator",
    )
    categories = {
        "parseable_wrong_values",
        "wrong_count",
        "wrapper_correct_values",
        "wrapper_wrong_values",
        "unresolved",
    }
    for key, names in (
        ("category_counts", categories),
        ("wrong_count_style_counts", {"bare", "bracketed", "fenced"}),
    ):
        counts = value[key]
        require(type(counts) is dict and set(counts) == names, "aggregate_count_fields")
        require(
            all(type(x) is int and 0 <= x <= 5 for x in counts.values()), "aggregate_count_type"
        )
    require(sum(value["category_counts"].values()) == 5, "aggregate_category_total")
    require(
        sum(value["wrong_count_style_counts"].values()) == value["category_counts"]["wrong_count"],
        "aggregate_style_total",
    )


def expected_answers(row):
    """Independent public input construction, without producer/auditor imports."""
    core, count, magnitude = row["core_index"], row["item_count"], row["magnitude"]
    require(type(core) is int and 0 <= core < 8, "core_range")
    require(type(count) is int and count in (2, 4), "count_range")
    require(magnitude in ("one_digit", "two_digit"), "magnitude_range")
    start, modulus = (1, 9) if magnitude == "one_digit" else (10, 90)
    answers = []
    for item in range(count):
        pair = []
        for side in range(2):
            raw = hashlib.sha256(
                f"gpu3b-arithmetic-load-v1|{core}|{item}|{side}".encode("ascii")
            ).digest()
            absolute = start + int.from_bytes(raw, "big") % modulus
            pair.append(-absolute if raw[0] % 2 else absolute)
        answers.append(sum(pair))
    require(
        type(row["expected_sums"]) is list
        and all(type(x) is int for x in row["expected_sums"])
        and row["expected_sums"] == answers,
        "independent_answer_binding",
    )
    return answers


class NoModelImports(importlib.abc.MetaPathFinder):
    roots = frozenset(
        {"torch", "transformers", "tokenizers", "numpy", "scipy", "sklearn", "safetensors"}
    )

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots:
            raise EvidenceError("model_or_numerical_import_forbidden")
        return None


def run(freeze_path, freeze_sha256, output):
    started = time.monotonic()
    require(
        not any(x.split(".")[0] in NoModelImports.roots for x in sys.modules),
        "cold_import_required",
    )
    guard = NoModelImports()
    sys.meta_path.insert(0, guard)
    try:
        return _run(freeze_path, freeze_sha256, output, started)
    finally:
        sys.meta_path.remove(guard)


def _run(freeze_path, freeze_sha256, output, started):
    freeze_path, output = Path(freeze_path), Path(output)
    raw_freeze = bound(freeze_path, freeze_sha256)
    freeze = json_value(raw_freeze)
    require(freeze["schema"] == "a190-five-completed-diagnosis-v1", "freeze_schema")
    require(output.absolute() == Path(freeze["output_path"]), "sole_frozen_output")
    source_root = Path(freeze["sealed_source_root"])
    inventory_path = Path(freeze["inventory_path"])
    require(not output.exists(), "analysis_already_claimed")
    require(output.parent.resolve() == freeze_path.parent.resolve(), "output_parent")
    require(not output.resolve().is_relative_to(source_root.resolve()), "output_in_sealed_source")
    tracked = {freeze_path: freeze_sha256}
    for key in (
        "inventory",
        "verification",
        "archive_receipt",
        "protocol",
        "qualification",
        "source_review",
        "scientific_review",
        "code_backup",
    ):
        path = Path(freeze[key + "_path"])
        wanted = freeze[key + "_sha256"]
        bound(path, wanted)
        tracked[path] = wanted
    require(inventory_path == Path(freeze["inventory_path"]), "inventory_path")
    inventory = json_value(bound(inventory_path, freeze["inventory_sha256"]))
    require(type(inventory) is dict and len(inventory) == 1102, "sealed_inventory_count")
    receipt = json_value(bound(freeze["archive_receipt_path"], freeze["archive_receipt_sha256"]))
    require(
        receipt["status"] == "archive_verified"
        and receipt["source_unchanged"] is True
        and receipt["every_member_content_verified"] is True
        and receipt["unique_members_verified"] is True
        and receipt["source_inventory_sha256"] == freeze["inventory_sha256"]
        and receipt["verification_sha256"] == freeze["verification_sha256"],
        "sealed_archive_binding",
    )
    verifier = json_value(bound(freeze["verification_path"], freeze["verification_sha256"]))
    require(
        verifier["status"] == "verified_incomplete_resource_stop"
        and verifier["terminal_verification_closed"] is True
        and type(verifier["complete_rows"]) is int
        and verifier["complete_rows"] == 5
        and verifier["independent_outcomes"]["overall"]["labels"]
        == {"success": 0, "failure": 5, "unknown": 27},
        "completed_cohort_verification",
    )
    module_path = Path(freeze["classifier_path"])
    module_raw = bound(module_path, freeze["classifier_sha256"])
    tracked[module_path] = freeze["classifier_sha256"]
    runner_path = Path(__file__)
    bound(runner_path, freeze["runner_sha256"])
    tracked[runner_path] = freeze["runner_sha256"]
    output.mkdir(mode=0o700)
    save_new(
        output / "CLAIM.json",
        {"schema": freeze["schema"], "freeze_sha256": freeze_sha256, "pid": os.getpid()},
    )
    sync_directory(output)
    sync_directory(output.parent)
    # The exclusive claim is durable before any response is read. No retry after
    # a claimed failure, and no data-dependent traceback or error text is emitted.
    try:
        classifier = types.ModuleType("frozen_response_diagnosis")
        exec(compile(module_raw, str(module_path), "exec"), classifier.__dict__)

        def sealed(relative):
            rel = Path(relative)
            require(not rel.is_absolute() and ".." not in rel.parts, "sealed_relative_path")
            require(relative in inventory, "sealed_inventory_entry")
            path = source_root / rel
            raw = bound(path, inventory[relative])
            tracked[path] = inventory[relative]
            return raw

        prepared = json_value(sealed("prepared/freeze.json"))
        plan = prepared["plan"]
        rows = plan["rows"]
        failure = json_value(sealed("run-001/worker/acquisition/failure.json"))
        records = failure["records"]
        require(
            type(rows) is list and type(records) is list and len(rows) == len(records) == 32,
            "fixed_roster",
        )
        diagnoses = []
        completions = [i for i, record in enumerate(records) if record["status"] == "completed"]
        require(len(completions) == 5 and failure["accepted_rows"] == 5, "five_completions")
        for index in completions:
            row, record = rows[index], records[index]
            for key in (
                "observation_id",
                "sequence_index",
                "core_index",
                "item_count",
                "magnitude",
            ):
                require(
                    type(row[key]) is type(record[key]) and row[key] == record[key],
                    "row_record_binding",
                )
            require(
                type(row["sequence_index"]) is int
                and row["sequence_index"] == index
                and record["generation_status"] == "completed"
                and type(record["label"]) is int
                and record["label"] == 0
                and record["terminal_eos"] is True,
                "verified_completed_failure",
            )
            answers = expected_answers(row)
            base = f"run-001/worker/acquisition/rows/{index:03d}/generation/"
            result = json_value(sealed(base + "result.json"))
            require(
                result["status"] == "completed" and result["stop_reason"] == "eos",
                "completed_eos_receipt",
            )
            response = sealed(base + "response.utf8")
            require(len(response) <= 8192, "completed_response_size")
            diagnoses.append(classifier.classify_response(response.decode("utf-8"), answers))
        aggregates = classifier.aggregate_diagnoses(diagnoses)
        check_aggregates(aggregates)
        for path, wanted in tracked.items():
            bound(path, wanted)
        result = {
            "status": "completed_model_free_diagnosis",
            "schema": freeze["schema"],
            "aggregates": aggregates,
            "freeze_sha256": freeze_sha256,
            "classifier_sha256": freeze["classifier_sha256"],
            "runner_sha256": freeze["runner_sha256"],
            "source_inventory_sha256": freeze["inventory_sha256"],
            "source_verification_sha256": freeze["verification_sha256"],
            "all_read_artifacts_unchanged": True,
            "model_tokenizer_encoder_fit_calls": 0,
            "old_scores_changed": False,
            "partial_response_reads": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        save_new(output / "RESULT.json", result)
        return result
    except BaseException:
        save_new(
            output / "FAILURE.json",
            {
                "status": "failed_no_qualified_diagnosis",
                "freeze_sha256": freeze_sha256,
                "retry_allowed": False,
            },
        )
        raise


def main():
    os.umask(0o077)
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(args.freeze, args.freeze_sha256, args.output)
    except BaseException:
        print('{"status":"diagnosis_failed_no_qualified_result"}')
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
