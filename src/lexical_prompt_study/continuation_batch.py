"""A148 fixed-four-slot continuation, with checkpoint-before-diagnostics.

The batch layout is part of the numerical path. Finished slots remain present
and are forced to EOS; the final short cohort pads physical slots with copies
of its first member, also forced to EOS. These slots are not additional trials.
Exact token snapshots are restricted artifacts and must never be printed.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .continuation import CHECKPOINTS, STUDY_ID, _hash, _identifier
from .continuation_gpu import PrefixGenerator, _token_hash, _validate_ids
from .hashing import canonical_json_bytes, sha256_bytes

BATCH_SIZE = 4


def _processors(torch: Any, inactive: Sequence[int], eos_id: int) -> Any:
    from transformers import LogitsProcessor, LogitsProcessorList

    class FixedSlots(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if not bool(torch.isfinite(scores).all().item()):
                raise ValueError("batch continuation nonfinite logits")
            for index in inactive:
                scores[index, :] = -torch.inf
                scores[index, eos_id] = 0
            return scores

    return LogitsProcessorList([FixedSlots()])


def _deadline_criteria(deadline: float | None, clock: Callable[[], float]) -> Any:
    from transformers import StoppingCriteria, StoppingCriteriaList

    class Deadline(StoppingCriteria):
        def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
            return deadline is not None and clock() >= deadline

    return StoppingCriteriaList([Deadline()])


class BatchPrefixGenerator:
    def __init__(
        self,
        runtime: Any,
        members: Sequence[Mapping[str, Any]],
        *,
        cohort_id: str,
        runtime_sha256: str,
        plan_sha256: str,
        residual_root: Path | None = None,
        deadline_monotonic: float | None = None,
        preserved_readouts: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        readout_sink: Callable[[str, dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= len(members) <= BATCH_SIZE:
            raise ValueError("batch cohort must have one to four real members")
        self.runtime, self.torch = runtime, runtime.torch
        self.model = runtime.model
        self.device = next(self.model.parameters()).device
        self.cohort_id = _identifier(cohort_id)
        self.runtime_sha256, self.plan_sha256 = _hash(runtime_sha256), _hash(plan_sha256)
        self.deadline_monotonic, self.clock = deadline_monotonic, clock
        self.started = clock()
        self.horizon = 128
        self.helpers: dict[str, PrefixGenerator] = {}
        self.members: list[dict[str, Any]] = []
        self.generation_seconds = 0.0
        self.diagnostic_forward_seconds = 0.0
        self.generation_calls = 0
        self.generation_step_count = 0
        self.diagnostic_forward_calls = 0
        self.new_tokens_by_trial: dict[str, int] = {}
        for row in members:
            trial_id = _identifier(row["trial_id"])
            if trial_id in self.helpers:
                raise ValueError("duplicate batch member")
            helper = PrefixGenerator(
                runtime,
                row["prompt_token_ids"],
                residual_root=residual_root / trial_id if residual_root is not None else None,
                deadline_monotonic=deadline_monotonic,
                preserved_readouts=(preserved_readouts or {}).get(trial_id),
                readout_sink=(
                    (lambda value, identity=trial_id: readout_sink(identity, value))
                    if readout_sink is not None
                    else None
                ),
                clock=clock,
            )
            original = _validate_ids(
                row["original_token_ids"], vocabulary_size=helper.vocabulary_size
            )
            if len(original) != 128 or any(token in helper.eos_token_ids for token in original):
                raise ValueError("batch requires an exact non-EOS 128-token original prefix")
            if len(helper.prompt_token_ids) + 1024 > self.model.config.max_position_embeddings:
                raise ValueError("batch continuation exceeds model context window")
            self.helpers[trial_id] = helper
            self.members.append(
                {
                    "trial_id": trial_id,
                    "prompt_token_ids": list(helper.prompt_token_ids),
                    "original_token_ids": original,
                    "generated_token_ids": list(original),
                    "eos": False,
                }
            )
            self.new_tokens_by_trial[trial_id] = 0
        if set(preserved_readouts or {}) - set(self.helpers):
            raise ValueError("readouts contain an unplanned batch member")
        self.eos_ids = sorted(next(iter(self.helpers.values())).eos_token_ids)
        self.pad_id = int(runtime.tokenizer.pad_token_id)
        self.member_ids = [row["trial_id"] for row in self.members]
        self.layout_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "cohort_id": self.cohort_id,
                    "batch_size": BATCH_SIZE,
                    "generation_padding": "left",
                    "diagnostic_padding": "right_valid_position",
                    "inactive_policy": "force_eos_keep_physical_slot",
                    "tail_policy": "copy_first_real_member_force_eos",
                    "members": [self._identity(row) for row in self.members],
                }
            )
        )

    @staticmethod
    def _identity(row: Mapping[str, Any]) -> dict[str, str]:
        return {
            "trial_id": row["trial_id"],
            "prompt_token_ids_sha256": _token_hash(row["prompt_token_ids"]),
            "original_prefix_token_ids_sha256": _token_hash(row["original_token_ids"]),
        }

    def _check_deadline(self) -> None:
        if self.deadline_monotonic is not None and self.clock() >= self.deadline_monotonic:
            raise TimeoutError("batch continuation deadline reached")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "status": "batch_horizon_tokens_complete",
            "cohort_id": self.cohort_id,
            "runtime_sha256": self.runtime_sha256,
            "plan_sha256": self.plan_sha256,
            "layout_sha256": self.layout_sha256,
            "batch_size": BATCH_SIZE,
            "member_ids": list(self.member_ids),
            "requested_horizon": self.horizon,
            "members": [
                {
                    **self._identity(row),
                    "generated_token_ids": list(row["generated_token_ids"]),
                    "eos": row["eos"],
                }
                for row in self.members
            ],
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        expected = self.snapshot()
        if set(snapshot) != set(expected) or any(
            snapshot[key] != expected[key]
            for key in expected
            if key not in {"members", "requested_horizon"}
        ):
            raise ValueError("batch snapshot layout or provenance drift")
        horizon = snapshot["requested_horizon"]
        if type(horizon) is not int or horizon not in CHECKPOINTS or horizon < self.horizon:
            raise ValueError("batch snapshot horizon drift")
        if len(snapshot["members"]) != len(self.members):
            raise ValueError("batch snapshot membership drift")
        validated = []
        for original, saved in zip(self.members, snapshot["members"], strict=True):
            if set(saved) != set(self._identity(original)) | {"generated_token_ids", "eos"} or any(
                saved[key] != value for key, value in self._identity(original).items()
            ):
                raise ValueError("batch snapshot original prefix binding drift")
            prefix = _validate_ids(
                saved["generated_token_ids"], vocabulary_size=self.model.config.vocab_size
            )
            if (
                type(saved["eos"]) is not bool
                or prefix[:128] != original["original_token_ids"]
                or prefix[: len(original["generated_token_ids"])] != original["generated_token_ids"]
                or any(token in self.eos_ids for token in prefix)
                or not 128 <= len(prefix) <= horizon
                or (saved["eos"] and (horizon == 128 or len(prefix) >= horizon))
                or (not saved["eos"] and len(prefix) != horizon)
                or (
                    original["eos"]
                    and (not saved["eos"] or prefix != original["generated_token_ids"])
                )
            ):
                raise ValueError("batch snapshot token/EOS state drift")
            validated.append((prefix, saved["eos"]))
        for row, (prefix, eos) in zip(self.members, validated, strict=True):
            row["generated_token_ids"], row["eos"] = prefix, eos
        self.horizon = horizon

    def _inputs(self) -> tuple[Any, Any, list[int]]:
        torch = self.torch
        sequences = [
            [*row["prompt_token_ids"], *row["generated_token_ids"]] for row in self.members
        ]
        inactive = [index for index, row in enumerate(self.members) if row["eos"]]
        while len(sequences) < BATCH_SIZE:
            inactive.append(len(sequences))
            sequences.append(list(sequences[0]))
        width = max(map(len, sequences))
        inputs = torch.tensor(
            [[self.pad_id] * (width - len(row)) + row for row in sequences],
            device=self.device,
            dtype=torch.long,
        )
        masks = torch.tensor(
            [[0] * (width - len(row)) + [1] * len(row) for row in sequences],
            device=self.device,
            dtype=torch.long,
        )
        return inputs, masks, inactive

    def _generate(self, new_tokens: int) -> list[dict[str, Any]]:
        self._check_deadline()
        inputs, masks, inactive = self._inputs()
        self._sync()
        started = time.perf_counter()
        with self.torch.inference_mode():
            sequences = self.model.generate(
                input_ids=inputs,
                attention_mask=masks,
                do_sample=False,
                max_new_tokens=new_tokens,
                pad_token_id=self.pad_id,
                eos_token_id=self.eos_ids,
                use_cache=True,
                return_dict_in_generate=False,
                logits_processor=_processors(self.torch, inactive, self.eos_ids[0]),
                stopping_criteria=_deadline_criteria(self.deadline_monotonic, self.clock),
            )
        self._sync()
        self.generation_seconds += time.perf_counter() - started
        self.generation_calls += 1
        self.generation_step_count += int(sequences.shape[1] - inputs.shape[1])
        if sequences.shape[0] != BATCH_SIZE or not self.torch.equal(
            sequences[:, : inputs.shape[1]], inputs
        ):
            raise ValueError("batch generation altered exact conditioning IDs")
        result = []
        for index, row in enumerate(self.members):
            if row["eos"]:
                result.append(
                    {"generated_token_ids": list(row["generated_token_ids"]), "eos": True}
                )
                continue
            raw = [int(token) for token in sequences[index, inputs.shape[1] :].tolist()]
            first_eos = next(
                (position for position, token in enumerate(raw) if token in self.eos_ids), None
            )
            delta = raw if first_eos is None else raw[:first_eos]
            if first_eos is None and len(delta) != new_tokens:
                self._check_deadline()
                raise ValueError("batch generation stopped short without EOS")
            result.append(
                {
                    "generated_token_ids": [*row["generated_token_ids"], *delta],
                    "eos": first_eos is not None,
                }
            )
        return result

    def _capture_sites(self, counts: Sequence[Sequence[int]]) -> dict[int, Any]:
        """Same residual-post hooks/valid-position indexing as retained runtime.

        Device-neutral implementation exercises the identical capture on CPU.
        Four physical rows are retained even when only one needs a readout.
        """
        self._check_deadline()
        torch = self.torch
        sequences = [
            [*row["prompt_token_ids"], *row["generated_token_ids"][: max(sites)]]
            for row, sites in zip(self.members, counts, strict=True)
        ]
        site_positions = [
            [len(row["prompt_token_ids"]) + count - 1 for count in sites]
            for row, sites in zip(self.members, counts, strict=True)
        ]
        sequences.extend([list(sequences[0]) for _ in range(BATCH_SIZE - len(sequences))])
        site_positions.extend(
            [list(site_positions[0]) for _ in range(BATCH_SIZE - len(site_positions))]
        )
        width = max(map(len, sequences))
        inputs = torch.tensor(
            [row + [self.pad_id] * (width - len(row)) for row in sequences],
            dtype=torch.long,
            device=self.device,
        )
        masks = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in sequences],
            dtype=torch.long,
            device=self.device,
        )
        indices = torch.arange(BATCH_SIZE, device=self.device)[:, None]
        positions = torch.tensor(site_positions, device=self.device)
        states, handles = {}, []
        for layer in self.runtime.source_layers:

            def hook(_module, _inputs, output, *, index=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                states[index] = hidden[indices, positions].detach().clone()

            handles.append(self.model.model.layers[layer].register_forward_hook(hook))
        self._sync()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                self.model(input_ids=inputs, attention_mask=masks, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        self._sync()
        self.diagnostic_forward_seconds += time.perf_counter() - started
        self.diagnostic_forward_calls += 1
        if sorted(states) != list(range(31)):
            raise ValueError("batch readout capture incomplete")
        return states

    def _diagnose_sites(self, nominal_counts: Sequence[int]) -> None:
        for row in self.members:
            helper = self.helpers[row["trial_id"]]
            helper._prefix = list(row["generated_token_ids"])
            helper._restore_readouts(helper._prefix)
        counts = [
            [min(count, len(row["generated_token_ids"])) for count in nominal_counts]
            for row in self.members
        ]
        if all(
            count in self.helpers[row["trial_id"]]._readouts
            for row, sites in zip(self.members, counts, strict=True)
            for count in sites
        ):
            return
        before = self.diagnostic_forward_seconds
        states = self._capture_sites(counts)
        elapsed_ms = (
            (self.diagnostic_forward_seconds - before) * 1000 / BATCH_SIZE / len(nominal_counts)
        )
        for index, (row, sites) in enumerate(zip(self.members, counts, strict=True)):
            helper = self.helpers[row["trial_id"]]
            for column, count in enumerate(sites):
                if count not in helper._readouts:
                    helper._record_readout(
                        count,
                        {
                            layer: value[index : index + 1, column, :]
                            for layer, value in states.items()
                        },
                        elapsed_ms,
                    )

    def advance_to_horizon(
        self, horizon: int, checkpoint_sink: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        if type(horizon) is not int or horizon not in CHECKPOINTS:
            raise ValueError("unplanned batch horizon")
        if horizon != self.horizon:
            if self.horizon == 1024 or horizon != CHECKPOINTS[CHECKPOINTS.index(self.horizon) + 1]:
                raise ValueError("batch horizons must be consecutive")
            if any(not row["eos"] for row in self.members):
                completed = self._generate(horizon - self.horizon)
                for row, value in zip(self.members, completed, strict=True):
                    self.new_tokens_by_trial[row["trial_id"]] += len(
                        value["generated_token_ids"]
                    ) - len(row["generated_token_ids"])
                    row.update(value)
            self.horizon = horizon
        snapshot = self.snapshot()
        # One durable group checkpoint precedes every per-row diagnostic. A
        # failure in a later hook never obliges the runner to regenerate tokens.
        checkpoint_sink(copy.deepcopy(snapshot))
        # Historical sites share one causal prompt+128 replay. Keeping this
        # replay length fixed also makes orphan-diagnostic reconstruction use
        # the same numerical shape after a later-horizon crash.
        self._diagnose_sites((16, 32, 64, 128))
        # On a restored later snapshot, replay each available frozen horizon.
        for count in (256, 512, 1024):
            if count <= horizon:
                self._diagnose_sites((count,))
        return {
            "snapshot": snapshot,
            "readouts": {
                trial: helper.readouts_by_generated_tokens for trial, helper in self.helpers.items()
            },
        }

    @property
    def timing_summary(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": max(0.0, self.clock() - self.started),
            "generated_new_token_count": sum(self.new_tokens_by_trial.values()),
            "new_tokens_by_trial": dict(self.new_tokens_by_trial),
            "generation_calls": self.generation_calls,
            "generation_step_count": self.generation_step_count,
            "model_forward_count": self.generation_step_count + self.diagnostic_forward_calls,
            "generation_elapsed_seconds": self.generation_seconds,
            "diagnostic_forward_calls": self.diagnostic_forward_calls,
            "diagnostic_forward_elapsed_seconds": self.diagnostic_forward_seconds,
            "detector_readout_elapsed_seconds": sum(
                helper._readout_elapsed_seconds for helper in self.helpers.values()
            ),
        }


def verify_batch_conditional_parity(
    runtime: Any,
    members: Sequence[Mapping[str, Any]],
    *,
    extension_tokens: int = 8,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Independent tensor assembly checks batch layout and EOS bookkeeping.

    Both use the same trusted HF generator, so this is an adapter wiring gate,
    not independent validation of HF or bitwise singleton equivalence.
    """
    if type(extension_tokens) is not int or not 2 <= extension_tokens <= 16:
        raise ValueError("batch parity extension outside frozen technical range")
    adapter = BatchPrefixGenerator(
        runtime,
        members,
        cohort_id="technical-parity",
        runtime_sha256="0" * 64,
        plan_sha256="0" * 64,
        deadline_monotonic=deadline_monotonic,
    )
    torch = runtime.torch

    def trusted_step(
        current: Sequence[Mapping[str, Any]], count: int
    ) -> tuple[list[dict[str, Any]], bool]:
        # Deliberately independent of adapter._inputs and its trim/update code.
        private_rows = [
            list(member["prompt_token_ids"]) + list(value["generated_token_ids"])
            for member, value in zip(members, current, strict=True)
        ]
        inactive = [index for index, value in enumerate(current) if value["eos"]]
        inactive.extend(range(len(members), BATCH_SIZE))
        private_rows += [list(private_rows[0]) for _ in range(BATCH_SIZE - len(private_rows))]
        width = max(map(len, private_rows))
        ids = torch.full(
            (BATCH_SIZE, width), adapter.pad_id, dtype=torch.long, device=adapter.device
        )
        mask = torch.zeros_like(ids)
        for index, row in enumerate(private_rows):
            ids[index, -len(row) :] = torch.tensor(row, dtype=torch.long, device=adapter.device)
            mask[index, -len(row) :] = 1
        adapter._check_deadline()
        with torch.inference_mode():
            reference = runtime.model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=count,
                pad_token_id=adapter.pad_id,
                eos_token_id=adapter.eos_ids,
                use_cache=True,
                return_dict_in_generate=False,
                logits_processor=_processors(torch, inactive, adapter.eos_ids[0]),
                stopping_criteria=_deadline_criteria(deadline_monotonic, time.monotonic),
            )
        adapter._check_deadline()
        input_matches = bool(torch.equal(reference[:, :width], ids))
        result = []
        for index, value in enumerate(current):
            prefix, eos = list(value["generated_token_ids"]), bool(value["eos"])
            if not eos:
                for token in reference[index, width:].tolist():
                    if token in adapter.eos_ids:
                        eos = True
                        break
                    prefix.append(int(token))
                if not eos and len(prefix) != len(value["generated_token_ids"]) + count:
                    raise ValueError("trusted batch parity stopped short without EOS")
            result.append({"generated_token_ids": prefix, "eos": eos})
        return result, input_matches

    def compare(expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]):
        return [
            {
                "trial_id": row["trial_id"],
                "conditional_token_ids_match": reference["generated_token_ids"]
                == value["generated_token_ids"],
                "eos_match": reference["eos"] == value["eos"],
                "reference_token_ids_sha256": _token_hash(reference["generated_token_ids"]),
                "adapter_token_ids_sha256": _token_hash(value["generated_token_ids"]),
                "continued_token_count": len(reference["generated_token_ids"]) - 128,
            }
            for row, reference, value in zip(members, expected, observed, strict=True)
        ]

    initial = [
        {"generated_token_ids": list(row["original_token_ids"]), "eos": False} for row in members
    ]
    observed = adapter._generate(extension_tokens)
    expected, input_matches = trusted_step(initial, extension_tokens)
    comparisons = compare(expected, observed)

    # A segment boundary replays its entire saved conditioning with a fresh
    # cache. Compare like-for-like refills, not refill versus uninterrupted
    # generation (which can differ under floating-point kernels).
    split_expected = initial
    split_input_matches = True
    boundary_comparisons = []
    for count in (extension_tokens // 2, extension_tokens - extension_tokens // 2):
        split_observed = adapter._generate(count)
        split_expected, input_ok = trusted_step(split_expected, count)
        split_input_matches = split_input_matches and input_ok
        boundary_comparisons.append(compare(split_expected, split_observed))
        for row, value in zip(adapter.members, split_observed, strict=True):
            row.update(value)
    split_comparisons = boundary_comparisons[-1]
    passed = (
        input_matches
        and split_input_matches
        and all(
            row["conditional_token_ids_match"] and row["eos_match"]
            for row in [*comparisons, *boundary_comparisons[0], *split_comparisons]
        )
    )
    return {
        "status": "pass" if passed else "stop_batch_conditional_parity_mismatch",
        "parity_scope": "fixed_four_slot_saved_prefix_conditional",
        "batch_size": BATCH_SIZE,
        "observation_count": len(members),
        "layout_sha256": adapter.layout_sha256,
        "trusted_input_prefix_unchanged": input_matches,
        "split_boundary_input_prefix_unchanged": split_input_matches,
        "split_boundary_members": split_comparisons,
        "members": comparisons,
    }
