"""One authenticated A186 fitting subprocess; no native model or encoder imports.

The outer frozen supervisor owns the process group, executable binding, resource
limits and normal subprocess exit. Importing this file uses only the stdlib.
The request and fitted artifacts are private; console output is aggregate only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

SCHEMA = "a186-fit-worker-v1"
REQUEST_KEYS = {"roster", "feature_packets", "labels", "bindings", "landmark_reached"}


class FitPipelineFailed(RuntimeError):
    """The prediction function returned a retained, explicitly failed partial result."""


def require(ok, code):
    if not ok:
        raise ValueError("a186_fit_worker_" + code)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def object_hash(value):
    return sha256(canonical(value))


def source_sha256():
    return sha256(Path(__file__).read_bytes())


def _hash(value):
    require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None, "sha256")


def _path_without_links(path):
    path = Path(path).absolute()
    require(all(not ancestor.is_symlink() for ancestor in (path, *path.parents)), "symlink")
    return path


def _read(path, *, expected_sha256=None):
    path = _path_without_links(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= 64 * 1024**2, "regular_bounded_file")
    raw = path.read_bytes()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "file_changed",
    )
    if expected_sha256 is not None:
        _hash(expected_sha256)
        require(sha256(raw) == expected_sha256, "file_hash")
    return raw


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def finite(value):
        parsed = float(value)
        require(math.isfinite(parsed), "nonfinite_json")
        return parsed

    value = json.loads(raw, object_pairs_hook=pairs, parse_float=finite, parse_constant=finite)
    require(canonical(value) == raw, "canonical_json")
    return value


def _publish(path, raw):
    """Fsync a unique temporary, exclusively link final, then fsync the directory."""
    require(type(raw) is bytes, "immutable_publication")
    path = _path_without_links(path)
    temporary = path.with_name(".pending-" + path.name)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256(raw)


def _dependencies():
    require(sys.version_info[:2] == (3, 12) and sys.prefix != sys.base_prefix, "python312_venv")
    source_root = Path(__file__).absolute().parents[1] / "src"
    _path_without_links(source_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from lexical_prompt_study import matched_prefix_prediction as prediction
    from sklearn.linear_model import LogisticRegression

    expected = source_root / "lexical_prompt_study/matched_prefix_prediction.py"
    require(Path(prediction.__file__).absolute() == expected, "imported_prediction_path")
    _read(expected)
    code = inspect.unwrap(LogisticRegression.fit).__code__
    require(code.co_name == "fit", "fit_code")
    return prediction, code


def _load_request(path, expected_sha256, prediction):
    raw = _read(path, expected_sha256=expected_sha256)
    request = _parse(raw)
    require(type(request) is dict and set(request) == REQUEST_KEYS, "request_keys")
    require(type(request["landmark_reached"]) is dict, "explicit_landmark_map")
    prediction.validate_inputs(
        request["roster"],
        request["feature_packets"],
        request["labels"],
        request["bindings"],
        landmark_reached=request["landmark_reached"],
    )
    return request, raw


def _aggregate(state, *, status, **hashes):
    return {
        "schema_version": SCHEMA,
        "status": status,
        "fit_claims": len(state["claims"]),
        "actual_fit_entries": state["entries"],
        "fit_body_returns": state["body_returns"],
        "event_count": len(state["event_hashes"]),
        **hashes,
    }


def run_worker(request_path, request_sha256, output_directory):
    """Run once; BaseException and cleanup failures never become completed results."""
    if not __debug__:
        raise RuntimeError("a186_fit_worker_optimization_forbidden")
    state = {"claims": [], "entries": 0, "body_returns": 0, "event_hashes": [], "entry_hashes": []}
    root = None
    run_sha = None
    result_sha = normal_sha = None
    profile_owned = False
    previous_profile = None
    cleanup_error = None
    prediction = None
    begun = time.monotonic()
    try:
        _hash(request_sha256)
        prediction, fit_code = _dependencies()
        request, raw = _load_request(request_path, request_sha256, prediction)
        previous_profile = sys.getprofile()
        require(previous_profile is None, "existing_profiler")
        output = _path_without_links(output_directory)
        require(output.parent.is_dir(), "output_parent")
        os.mkdir(output, 0o700)  # The only ownership claim; never adopt an existing directory.
        root = output
        os.mkdir(root / "events", 0o700)
        os.mkdir(root / "entries", 0o700)
        header = {
            "schema_version": SCHEMA,
            "request_sha256": request_sha256,
            "worker_source_sha256": source_sha256(),
            "prediction_source_sha256": prediction.source_sha256(),
            "fit_runtime": prediction.runtime_descriptor(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "maximum_fit_entries": 2,
        }
        run_sha = _publish(root / "run.json", canonical(header))
        _publish(root / "request.json", raw)
        active_frames = {}
        returned_slots = set()

        def journal(event_raw):
            require(type(event_raw) is bytes, "event_bytes")
            value = _parse(event_raw)
            require(
                type(value) is dict and value.get("schema_version") == "a186-fit-event-v1",
                "event_schema",
            )
            kind = value.get("kind")
            require(
                kind in ("transform_return", "before_fit", "fit_return", "fit_failure"),
                "event_kind",
            )
            if kind == "before_fit":
                ordinal = len(state["claims"]) + 1
                require(
                    ordinal <= 2
                    and type(value.get("attempt_index")) is int
                    and value["attempt_index"] == ordinal,
                    "claim_count",
                )
                require(value.get("slot") == ("text", "text_internal")[ordinal - 1], "claim_order")
                require(
                    canonical(value.get("options")) == canonical(prediction.OPTIONS),
                    "claim_options",
                )
                require(ordinal == 1 or "text" in returned_slots, "claim_after_failure")
            elif kind in ("fit_return", "fit_failure"):
                require(
                    bool(state["claims"])
                    and value.get("slot") == state["claims"][-1]["slot"]
                    and type(value.get("attempt_index")) is int
                    and value["attempt_index"] == len(state["claims"]),
                    "fit_event_identity",
                )
                require(state["entries"] == len(state["claims"]), "fit_event_entry")
                if kind == "fit_return":
                    require(state["body_returns"] == state["entries"], "fit_return_body")
            digest = _publish(
                root / "events" / f"event_{len(state['event_hashes']):03d}.json", event_raw
            )
            state["event_hashes"].append(digest)
            if kind == "before_fit":
                state["claims"].append({"slot": value["slot"], "event_sha256": digest})
            elif kind == "fit_return":
                returned_slots.add(value["slot"])

        def observe(frame, event, arg):
            if frame.f_code is not fit_code:
                return
            if event == "call":
                # This is the observed code-entry boundary, even if its durable
                # publication fails before the first estimator statement runs.
                state["entries"] += 1
                ordinal = state["entries"]
                require(
                    ordinal <= 2 and ordinal == len(state["claims"]),
                    "actual_entry_ceiling_or_claim",
                )
                require(not active_frames, "recursive_fit")
                entry = {
                    "schema_version": "a186-fit-entry-v1",
                    "run_sha256": run_sha,
                    "request_sha256": request_sha256,
                    "entry_number": ordinal,
                    "slot": state["claims"][-1]["slot"],
                    "claim_sha256": state["claims"][-1]["event_sha256"],
                    "code_boundary": "unwrapped_LogisticRegression.fit_call_before_body",
                }
                digest = _publish(root / "entries" / f"fit_{ordinal:02d}.json", canonical(entry))
                state["entry_hashes"].append(digest)
                active_frames[id(frame)] = True
            elif event == "return" and id(frame) in active_frames:
                del active_frames[id(frame)]
                # Profile 'return' also fires on exceptions; fit's successful
                # return is specifically its original self object.
                if arg is frame.f_locals.get("self"):
                    state["body_returns"] += 1

        original = None
        try:
            sys.setprofile(observe)
            profile_owned = True
            result = prediction.fit_predict(
                request["roster"],
                request["feature_packets"],
                request["labels"],
                bindings=request["bindings"],
                landmark_reached=request["landmark_reached"],
                event=journal,
            )
            normal = {
                "schema_version": SCHEMA,
                "request_sha256": request_sha256,
                "run_sha256": run_sha,
                "fit_predict_returned": True,
                "returned_result_sha256": object_hash(result),
                "terminal_error": result["terminal_error"],
                "actual_fit_entries": state["entries"],
            }
            normal_sha = _publish(root / "normal-return.json", canonical(normal))
            result_sha = prediction.save_result(
                root / "result.json",
                result,
                request["roster"],
                request["feature_packets"],
                request["labels"],
                bindings=request["bindings"],
                landmark_reached=request["landmark_reached"],
            )
            require(
                result["actual_fits"] == state["entries"] == len(state["claims"]) <= 2,
                "reported_actual_fit_count",
            )
            if result["terminal_error"] is not None:
                raise FitPipelineFailed("a186_fit_worker_retained_fit_failure")
        except BaseException as error:
            original = error
        finally:
            try:
                sys.setprofile(previous_profile)
                profile_owned = False
                require(sys.getprofile() is previous_profile, "profiler_cleanup")
            except BaseException as error:
                cleanup_error = error
        if original is not None:
            raise original
        if cleanup_error is not None:
            raise cleanup_error
        terminal = _aggregate(
            state,
            status="completed",
            request_sha256=request_sha256,
            run_sha256=run_sha,
            result_sha256=result_sha,
            normal_return_sha256=normal_sha,
            event_sha256=state["event_hashes"],
            entry_sha256=state["entry_hashes"],
            model_free_replay=True,
            profile_restored=True,
        )
        terminal_sha = _publish(root / "terminal.json", canonical(terminal))
        loaded = load_worker(root, expected_request_sha256=request_sha256)
        return {
            key: value for key, value in loaded.items() if key not in ("result", "terminal")
        } | {"terminal_sha256": terminal_sha}
    except BaseException as error:
        if root is not None:
            failure = _aggregate(
                state,
                status="failed",
                request_sha256=request_sha256,
                run_sha256=run_sha,
                result_sha256=result_sha,
                normal_return_sha256=normal_sha,
                failure_type=type(error).__name__,
                cleanup_failure_type=None
                if cleanup_error is None
                else type(cleanup_error).__name__,
                profile_restored=not profile_owned,
                elapsed_seconds=time.monotonic() - begun,
                event_sha256=state["event_hashes"],
                entry_sha256=state["entry_hashes"],
            )
            try:
                failure_sha = _publish(root / "failure.json", canonical(failure))
            except BaseException:
                failure_sha = None
        else:
            failure_sha = None
        try:
            error.fit_worker_summary = _aggregate(
                state,
                status="failed",
                failure_type=type(error).__name__,
                failure_sha256=failure_sha,
            )
        except BaseException:
            pass
        raise


def load_worker(directory, *, expected_request_sha256):
    """Authenticate complete writer evidence and replay coefficients; never refit."""
    root = _path_without_links(directory)
    require(root.is_dir(), "output_directory")
    require(
        {path.name for path in root.iterdir()}
        == {
            "run.json",
            "request.json",
            "events",
            "entries",
            "result.json",
            "normal-return.json",
            "terminal.json",
        },
        "completed_root_inventory",
    )
    prediction, _ = _dependencies()
    request, _ = _load_request(root / "request.json", expected_request_sha256, prediction)
    header = _parse(_read(root / "run.json"))
    require(
        type(header) is dict
        and set(header)
        == {
            "schema_version",
            "request_sha256",
            "worker_source_sha256",
            "prediction_source_sha256",
            "fit_runtime",
            "pid",
            "parent_pid",
            "maximum_fit_entries",
        },
        "header_keys",
    )
    require(
        header["schema_version"] == SCHEMA
        and header["request_sha256"] == expected_request_sha256
        and header["worker_source_sha256"] == source_sha256()
        and header["prediction_source_sha256"] == prediction.source_sha256()
        and canonical(header["fit_runtime"]) == canonical(prediction.runtime_descriptor()),
        "header_binding",
    )
    require(
        all(type(header[key]) is int and header[key] > 0 for key in ("pid", "parent_pid"))
        and type(header["maximum_fit_entries"]) is int
        and header["maximum_fit_entries"] == 2,
        "header_integers",
    )
    terminal = _parse(_read(root / "terminal.json"))
    require(
        type(terminal) is dict
        and set(terminal)
        == {
            "schema_version",
            "status",
            "fit_claims",
            "actual_fit_entries",
            "fit_body_returns",
            "event_count",
            "request_sha256",
            "run_sha256",
            "result_sha256",
            "normal_return_sha256",
            "event_sha256",
            "entry_sha256",
            "model_free_replay",
            "profile_restored",
        },
        "terminal_keys",
    )
    for key in ("fit_claims", "actual_fit_entries", "fit_body_returns", "event_count"):
        require(type(terminal[key]) is int and terminal[key] >= 0, "terminal_counter")
    require(
        terminal["schema_version"] == SCHEMA
        and terminal["status"] == "completed"
        and terminal["request_sha256"] == expected_request_sha256
        and terminal["run_sha256"] == object_hash(header)
        and terminal["model_free_replay"] is True
        and terminal["profile_restored"] is True,
        "terminal_binding",
    )
    normal = _parse(
        _read(root / "normal-return.json", expected_sha256=terminal["normal_return_sha256"])
    )
    result = prediction.load_result(
        root / "result.json",
        request["roster"],
        request["feature_packets"],
        request["labels"],
        bindings=request["bindings"],
        landmark_reached=request["landmark_reached"],
        expected_sha256=terminal["result_sha256"],
    )
    expected_normal = {
        "schema_version": SCHEMA,
        "request_sha256": expected_request_sha256,
        "run_sha256": object_hash(header),
        "fit_predict_returned": True,
        "returned_result_sha256": terminal["result_sha256"],
        "terminal_error": None,
        "actual_fit_entries": terminal["actual_fit_entries"],
    }
    require(
        canonical(normal) == canonical(expected_normal) and result["terminal_error"] is None,
        "normal_return",
    )
    count = result["actual_fits"]
    require(
        count
        == terminal["actual_fit_entries"]
        == terminal["fit_claims"]
        == terminal["fit_body_returns"]
        <= 2,
        "terminal_fits",
    )
    events_dir, entries_dir = (
        _path_without_links(root / "events"),
        _path_without_links(root / "entries"),
    )
    require(events_dir.is_dir() and entries_dir.is_dir(), "journal_directories")
    require(
        type(terminal["event_sha256"]) is list
        and len(terminal["event_sha256"]) == terminal["event_count"],
        "event_hashes",
    )
    require(
        type(terminal["entry_sha256"]) is list and len(terminal["entry_sha256"]) == count,
        "entry_hashes",
    )
    require(
        {path.name for path in events_dir.iterdir()}
        == {f"event_{i:03d}.json" for i in range(terminal["event_count"])},
        "event_inventory",
    )
    require(
        {path.name for path in entries_dir.iterdir()}
        == {f"fit_{i:02d}.json" for i in range(1, count + 1)},
        "entry_inventory",
    )
    events = [
        _parse(_read(events_dir / f"event_{i:03d}.json", expected_sha256=sha))
        for i, sha in enumerate(terminal["event_sha256"])
    ]
    expected_kinds = (
        []
        if count == 0
        else ["transform_return"]
        + [kind for _ in range(count) for kind in ("before_fit", "fit_return")]
    )
    require([item.get("kind") for item in events] == expected_kinds, "event_sequence")
    if count:
        import numpy as np

        indices = [i for i, row in enumerate(result["fit_eligibility"]) if row["eligible"]]
        fit_ids = [request["roster"][i]["observation_id"] for i in indices]
        base = np.stack([prediction.text_features(request["feature_packets"][i]) for i in indices])
        internal = np.stack(
            [
                prediction.transform_internal(
                    request["feature_packets"][i]["internal"], result["transform"]
                )
                for i in indices
            ]
        )
        fit_labels = [request["labels"][key] for key in fit_ids]
        first = {
            "schema_version": "a186-fit-event-v1",
            "kind": "transform_return",
            "artifact": result["transform"],
            "artifact_sha256": object_hash(result["transform"]),
        }
        require(canonical(events[0]) == canonical(first), "transform_event")
        for i in range(count):
            claim, returned = events[1 + 2 * i : 3 + 2 * i]
            slot = ("text", "text_internal")[i]
            matrix = base if slot == "text" else np.concatenate((base, internal), axis=1)
            expected_claim = {
                "schema_version": "a186-fit-event-v1",
                "kind": "before_fit",
                "slot": slot,
                "attempt_index": i + 1,
                "options": prediction.OPTIONS,
                "fit_observation_ids_sha256": object_hash(fit_ids),
                "matrix_shape": list(matrix.shape),
                "matrix_fp64le_sha256": sha256(matrix.astype("<f8").tobytes()),
                "labels_sha256": object_hash(fit_labels),
            }
            require(canonical(claim) == canonical(expected_claim), "claim_inputs")
            require(
                claim.get("schema_version") == "a186-fit-event-v1"
                and claim.get("slot") == slot
                and type(claim.get("attempt_index")) is int
                and claim["attempt_index"] == i + 1
                and canonical(claim.get("options")) == canonical(prediction.OPTIONS),
                "claim_identity",
            )
            require(
                claim.get("fit_observation_ids_sha256")
                == object_hash(
                    result["models"][slot].get(
                        "fit_observation_ids",
                        [
                            row["observation_id"]
                            for row in result["fit_eligibility"]
                            if row["eligible"]
                        ],
                    )
                )
                and claim.get("matrix_shape")
                == [result["fit_support"]["eligible_rows"], 1026 if slot == "text" else 1042],
                "claim_cohort",
            )
            expected_return = {
                "schema_version": "a186-fit-event-v1",
                "kind": "fit_return",
                "slot": slot,
                "attempt_index": i + 1,
                "artifact": result["models"][slot],
                "artifact_sha256": object_hash(result["models"][slot]),
            }
            require(canonical(returned) == canonical(expected_return), "returned_model")
            entry = _parse(
                _read(
                    entries_dir / f"fit_{i + 1:02d}.json",
                    expected_sha256=terminal["entry_sha256"][i],
                )
            )
            expected_entry = {
                "schema_version": "a186-fit-entry-v1",
                "run_sha256": object_hash(header),
                "request_sha256": expected_request_sha256,
                "entry_number": i + 1,
                "slot": slot,
                "claim_sha256": terminal["event_sha256"][1 + 2 * i],
                "code_boundary": "unwrapped_LogisticRegression.fit_call_before_body",
            }
            require(canonical(entry) == canonical(expected_entry), "entry_binding")
    return {
        "schema_version": SCHEMA,
        "status": "completed",
        "request_sha256": expected_request_sha256,
        "terminal_sha256": object_hash(terminal),
        "result_sha256": terminal["result_sha256"],
        "normal_return_sha256": terminal["normal_return_sha256"],
        "fit_claims": terminal["fit_claims"],
        "actual_fit_entries": count,
        "fit_body_returns": terminal["fit_body_returns"],
        "event_count": terminal["event_count"],
        "model_free_replay": True,
        "result": result,
        "terminal": terminal,
    }


def _lifecycle():
    path = Path(__file__).absolute().with_name("run_native_prefix_qualification.py")
    spec = importlib.util.spec_from_file_location("a186_parent_lifecycle", path)
    require(spec is not None and spec.loader is not None, "lifecycle_loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if not __debug__:
            raise RuntimeError("a186_fit_worker_optimization_forbidden")
        lifecycle = _lifecycle()
        with lifecycle.owned_interruptions():
            if args.parent is not None:
                require(args.parent > 0, "parent_pid")
                lifecycle.bind_parent_lifetime(args.parent)
            summary = run_worker(args.request, args.request_sha256, args.output)
    except BaseException as error:
        summary = getattr(
            error,
            "fit_worker_summary",
            {
                "schema_version": SCHEMA,
                "status": "failed",
                "failure_type": type(error).__name__,
                "actual_fit_entries": None,
            },
        )
        print(json.dumps(summary, sort_keys=True))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
