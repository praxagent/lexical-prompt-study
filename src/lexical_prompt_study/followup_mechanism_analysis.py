from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .followup_design import PLACEMENTS
from .followup_plan import validate_followup_plan
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic
from .mechanism_runner import _torch_margin_batch
from .mechanisms import RANDOM_TRANSPORT_SEED, deterministic_transport_seed
from .models import FollowupTrialReceipt


SAE_HOOK_LAYER = 19
TRANSPORTS = ("jacobian_lens", "identity", "random_gaussian")
ARMS = ("base", "inert_length", "structural_sham", "full")
STRUCTURED_ARMS = ("inert_length", "structural_sham", "full")
PARTITIONS = ("discovery", "calibration")
MINIMUM_DISCOVERY_FULL_PREVALENCE = 0.10
MAXIMUM_SUBSPACE_FEATURES = 8
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BASE_SEED = 20260801
RANDOM_NORM_RELATIVE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Candidate:
    kind: str
    feature_ids: tuple[int, ...]
    weights: tuple[float, ...]

    @property
    def candidate_id(self) -> str:
        joined = "-".join(str(item) for item in self.feature_ids)
        return f"{self.kind}:{joined}"


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def standardized_paired_effect(delta: Any) -> tuple[float, float, float]:
    values = _finite_array(delta, name="paired delta", ndim=1)
    if values.size < 2:
        raise ValueError("paired delta requires at least two independent units")
    mean = float(values.mean())
    rms = float(np.sqrt(np.mean(np.square(values))))
    standardized = mean / rms if rms > np.finfo(np.float64).eps else 0.0
    return mean, rms, standardized


def stable_bootstrap_seed(
    *,
    base_seed: int,
    partition: str,
    placement: str,
    layer: int,
    transport: str,
    statistic: str,
) -> int:
    payload = {
        "base_seed": int(base_seed),
        "partition": partition,
        "placement": placement,
        "layer": int(layer),
        "transport": transport,
        "statistic": statistic,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def paired_bootstrap_interval(
    values: Any,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    array = _finite_array(values, name="bootstrap values", ndim=1)
    if array.size < 2 or replicates < 1:
        raise ValueError("bootstrap requires at least two rows and one replicate")
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, array.size, size=(replicates, array.size))]
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def discover_candidates(
    *,
    full_by_placement: dict[str, np.ndarray],
    sham_by_placement: dict[str, np.ndarray],
    decoder_norms: np.ndarray,
    minimum_full_prevalence: float = MINIMUM_DISCOVERY_FULL_PREVALENCE,
    maximum_subspace_features: int = MAXIMUM_SUBSPACE_FEATURES,
) -> tuple[list[dict[str, Any]], Candidate, Candidate]:
    if tuple(full_by_placement) != tuple(PLACEMENTS):
        raise ValueError("full placement order drift")
    if tuple(sham_by_placement) != tuple(PLACEMENTS):
        raise ValueError("sham placement order drift")
    norms = _finite_array(decoder_norms, name="decoder norms", ndim=1)
    if (norms < 0).any():
        raise ValueError("decoder norms must be non-negative")
    if not 1 <= maximum_subspace_features <= 8:
        raise ValueError("subspace feature limit must be in [1, 8]")

    per_order: dict[str, dict[str, np.ndarray]] = {}
    feature_count: int | None = None
    for placement in PLACEMENTS:
        full = _finite_array(
            full_by_placement[placement],
            name=f"{placement} full",
            ndim=2,
        )
        sham = _finite_array(
            sham_by_placement[placement],
            name=f"{placement} sham",
            ndim=2,
        )
        if full.shape != sham.shape or full.shape[0] < 2:
            raise ValueError("paired full/sham topology drift")
        if (full < 0).any() or (sham < 0).any():
            raise ValueError("SAE activations must be non-negative")
        feature_count = full.shape[1] if feature_count is None else feature_count
        if full.shape[1] != feature_count or norms.shape != (feature_count,):
            raise ValueError("SAE feature-width drift")
        delta = full - sham
        mean = delta.mean(axis=0)
        rms = np.sqrt(np.mean(np.square(delta), axis=0))
        standardized = np.divide(
            mean,
            rms,
            out=np.zeros_like(mean),
            where=rms > np.finfo(np.float64).eps,
        )
        per_order[placement] = {
            "mean": mean,
            "rms": rms,
            "standardized": standardized,
            "full_prevalence": np.mean(full > 0, axis=0),
            "sham_prevalence": np.mean(sham > 0, axis=0),
        }

    assert feature_count is not None
    eligible = np.ones(feature_count, dtype=bool)
    for placement in PLACEMENTS:
        order = per_order[placement]
        eligible &= order["mean"] > 0
        eligible &= order["rms"] > np.finfo(np.float64).eps
        eligible &= order["standardized"] > 0
        eligible &= order["full_prevalence"] >= minimum_full_prevalence
    eligible &= norms > 0
    feature_ids = np.flatnonzero(eligible)
    if feature_ids.size == 0:
        raise ValueError("no discovery feature is eligible in both placements")

    rows: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        order_metrics = {
            placement: {
                key: float(per_order[placement][key][feature_id])
                for key in (
                    "mean",
                    "rms",
                    "standardized",
                    "full_prevalence",
                    "sham_prevalence",
                )
            }
            for placement in PLACEMENTS
        }
        rows.append(
            {
                "feature_id": int(feature_id),
                "decoder_norm": float(norms[feature_id]),
                "minimum_standardized_effect": min(
                    order_metrics[placement]["standardized"]
                    for placement in PLACEMENTS
                ),
                "minimum_raw_mean": min(
                    order_metrics[placement]["mean"] for placement in PLACEMENTS
                ),
                "ordering_results": order_metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["minimum_standardized_effect"],
            -row["minimum_raw_mean"],
            row["feature_id"],
        )
    )
    single_id = int(rows[0]["feature_id"])
    single = Candidate("single_feature", (single_id,), (1.0,))

    ranked_ids = [int(row["feature_id"]) for row in rows[:maximum_subspace_features]]
    ascending_ids = tuple(sorted(ranked_ids))
    worst_order_rms = np.asarray(
        [
            max(
                float(per_order[placement]["rms"][feature_id])
                for placement in PLACEMENTS
            )
            for feature_id in ascending_ids
        ],
        dtype=np.float64,
    )
    inverse_scale = 1.0 / worst_order_rms
    normalized = inverse_scale / np.linalg.norm(inverse_scale)
    subspace = Candidate(
        "linear_subspace",
        ascending_ids,
        tuple(float(value) for value in normalized),
    )
    return rows, single, subspace


def candidate_scores(activations: np.ndarray, candidate: Candidate) -> np.ndarray:
    values = _finite_array(activations, name="candidate activations", ndim=2)
    ids = np.asarray(candidate.feature_ids, dtype=np.int64)
    weights = _finite_array(candidate.weights, name="candidate weights", ndim=1)
    if ids.size == 0 or weights.shape != ids.shape:
        raise ValueError("candidate feature/weight topology drift")
    if ids.min() < 0 or ids.max() >= values.shape[1]:
        raise ValueError("candidate feature outside SAE width")
    return values[:, ids] @ weights


def rank_calibration_candidates(
    candidate_deltas: dict[str, dict[str, np.ndarray]],
) -> tuple[Candidate | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates_by_id: dict[str, Candidate] = {}
    for candidate_id, payload in candidate_deltas.items():
        candidate = payload["candidate"]
        if not isinstance(candidate, Candidate) or candidate.candidate_id != candidate_id:
            raise ValueError("candidate identity drift")
        candidates_by_id[candidate_id] = candidate
        ordering: dict[str, dict[str, float]] = {}
        eligible = True
        for placement in PLACEMENTS:
            mean, rms, standardized = standardized_paired_effect(payload[placement])
            if rms <= np.finfo(np.float64).eps or mean <= 0 or standardized <= 0:
                eligible = False
            ordering[placement] = {
                "mean": mean,
                "rms": rms,
                "standardized": standardized,
            }
        rows.append(
            {
                "candidate_id": candidate_id,
                "kind": candidate.kind,
                "feature_ids": list(candidate.feature_ids),
                "weights": list(candidate.weights),
                "eligible": eligible,
                "minimum_standardized_effect": min(
                    ordering[placement]["standardized"] for placement in PLACEMENTS
                ),
                "minimum_raw_mean": min(
                    ordering[placement]["mean"] for placement in PLACEMENTS
                ),
                "ordering_results": ordering,
            }
        )
    rows.sort(
        key=lambda row: (
            not row["eligible"],
            -row["minimum_standardized_effect"],
            -row["minimum_raw_mean"],
            0 if row["kind"] == "single_feature" else 1,
            tuple(row["feature_ids"]),
        )
    )
    selected = candidates_by_id[rows[0]["candidate_id"]] if rows and rows[0]["eligible"] else None
    return selected, rows


def fit_common_dense_projection(
    full_by_placement: dict[str, np.ndarray],
    sham_by_placement: dict[str, np.ndarray],
) -> np.ndarray:
    unit_directions = []
    for placement in PLACEMENTS:
        full = _finite_array(full_by_placement[placement], name="dense full", ndim=2)
        sham = _finite_array(sham_by_placement[placement], name="dense sham", ndim=2)
        if full.shape != sham.shape:
            raise ValueError("dense projection topology drift")
        direction = (full - sham).mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError(f"{placement}: zero dense mean-difference direction")
        unit_directions.append(direction / norm)
    common = np.mean(np.stack(unit_directions), axis=0)
    norm = float(np.linalg.norm(common))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("ordering-specific dense directions cancel")
    return common / norm


def _atomic_torch_save(torch, path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)
    return sha256_file(path)


def _receipt_rows(root: Path) -> list[tuple[Path, FollowupTrialReceipt]]:
    paths = sorted((root / "receipts" / "trials").glob("*.json"))
    return [
        (path, FollowupTrialReceipt.model_validate_json(path.read_text()))
        for path in paths
    ]


def validate_state_payload(
    *,
    state_payload: dict[str, Any],
    receipt: FollowupTrialReceipt,
    expected_layers: tuple[int, ...] | None,
) -> tuple[dict[int, Any], tuple[int, ...]]:
    if set(state_payload) != {"provenance", "states"}:
        raise ValueError("unexpected state-bundle keys")
    expected_provenance = {
        "trial_id": receipt.trial_id,
        "public_plan_sha256": receipt.plan_sha256,
        "private_plan_sha256": receipt.private_plan_sha256,
        "source_commit": receipt.source_commit,
        "run_id": receipt.run_id,
        "model_revision": receipt.model_revision,
        "prompt_token_ids_sha256": receipt.prompt_token_ids_sha256,
        "lens_sha256": receipt.lens_sha256,
        "sae_sha256": receipt.sae_sha256,
    }
    if state_payload["provenance"] != expected_provenance:
        raise ValueError("state provenance drift")
    layer_states = {int(key): value for key, value in state_payload["states"].items()}
    layers = tuple(sorted(layer_states))
    if expected_layers is not None and layers != expected_layers:
        raise ValueError("capture layer topology drift")
    if SAE_HOOK_LAYER not in layers:
        raise ValueError("SAE hook layer missing from capture")
    for value in layer_states.values():
        if tuple(value.shape) != (4096,) or str(value.dtype) != "torch.bfloat16":
            raise ValueError("state shape/dtype drift")
    return layer_states, layers


def _validate_input_partition(
    *,
    torch,
    public_plan: dict[str, Any],
    root: Path,
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, Any]], str]:
    run_key = f"g2_{partition}" if partition == "discovery" else "g2_calibration_generation"
    binding = public_plan["compute"]["scientific_runs"][run_key]["result_binding"]
    receipt_rows = _receipt_rows(root)
    if len(receipt_rows) != binding["receipt_count"] or len(receipt_rows) != 140:
        raise ValueError(f"{partition}: expected 140 generation receipts")
    records: list[dict[str, Any]] = []
    states: dict[str, dict[int, Any]] = {}
    manifest_rows = []
    seen_trials: set[str] = set()
    expected_layers: tuple[int, ...] | None = None
    for receipt_path, receipt in receipt_rows:
        if receipt.trial_id in seen_trials or receipt_path.stem != receipt.trial_id:
            raise ValueError(f"{partition}: duplicate or misnamed trial receipt")
        seen_trials.add(receipt.trial_id)
        if (
            receipt.partition != partition
            or receipt.source_commit != binding["source_commit"]
            or receipt.plan_sha256 != binding["public_plan_sha256"]
            or receipt.private_plan_sha256 != binding["private_plan_sha256"]
            or receipt.run_id != binding["run_id"]
            or receipt.lens_sha256 != public_plan["artifacts"]["llama31_lens"]["sha256"]
            or receipt.sae_sha256 != public_plan["artifacts"]["llama31_sae"]["sha256"]
            or receipt.model_revision
            != public_plan["artifacts"]["llama31_model"]["revision"]
            or receipt.tokenizer_revision
            != public_plan["artifacts"]["llama31_model"]["revision"]
        ):
            raise ValueError(f"{partition}: generation receipt provenance drift")
        state_path = root / "states" / f"{receipt.trial_id}.pt"
        if not state_path.exists() or sha256_file(state_path) != receipt.state_bundle_sha256:
            raise ValueError(f"{partition}: state bundle hash drift")
        state_payload = torch.load(state_path, map_location="cpu", weights_only=True)
        layer_states, layers = validate_state_payload(
            state_payload=state_payload,
            receipt=receipt,
            expected_layers=expected_layers,
        )
        expected_layers = layers if expected_layers is None else expected_layers
        states[receipt.trial_id] = layer_states
        records.append(receipt.model_dump(mode="json"))
        manifest_rows.append(
            {
                "trial_id": receipt.trial_id,
                "receipt_sha256": sha256_file(receipt_path),
                "state_sha256": receipt.state_bundle_sha256,
            }
        )
    topology = {}
    for arm in ARMS:
        for placement in (None, *PLACEMENTS):
            count = sum(
                row["arm"] == arm and row["placement"] == placement for row in records
            )
            if count:
                topology[f"{arm}:{placement or 'shared'}"] = count
    expected_topology = {"base:shared": 20}
    expected_topology.update(
        {
            f"{arm}:{placement}": 20
            for arm in STRUCTURED_ARMS
            for placement in PLACEMENTS
        }
    )
    if topology != expected_topology:
        raise ValueError(f"{partition}: arm/placement topology drift")
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest_rows))
    return records, states, manifest_sha


def _trial_index(records: list[dict[str, Any]]) -> dict[tuple[str, str, str | None], int]:
    index = {}
    for row_index, row in enumerate(records):
        key = (row["behavior_id"], row["arm"], row["placement"])
        if key in index:
            raise ValueError("duplicate behavior/arm/placement row")
        index[key] = row_index
    return index


def _ordered_behavior_ids(
    records: list[dict[str, Any]],
    *,
    arm: str,
    placement: str | None,
) -> list[str]:
    return sorted(
        row["behavior_id"]
        for row in records
        if row["arm"] == arm and row["placement"] == placement
    )


def _paired_rows(
    values: np.ndarray,
    records: list[dict[str, Any]],
    *,
    placement: str,
    full_arm: str = "full",
    sham_arm: str = "structural_sham",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    index = _trial_index(records)
    behavior_ids = _ordered_behavior_ids(records, arm=full_arm, placement=placement)
    if behavior_ids != _ordered_behavior_ids(records, arm=sham_arm, placement=placement):
        raise ValueError("paired behavior IDs drift")
    full = np.stack([values[index[(item, full_arm, placement)]] for item in behavior_ids])
    sham = np.stack([values[index[(item, sham_arm, placement)]] for item in behavior_ids])
    return full, sham, behavior_ids


def _arm_summary(
    values: np.ndarray,
    records: list[dict[str, Any]],
    *,
    arm: str,
    placement: str | None,
) -> dict[str, Any]:
    index = _trial_index(records)
    behavior_ids = _ordered_behavior_ids(records, arm=arm, placement=placement)
    selected = np.asarray(
        [values[index[(item, arm, placement)]] for item in behavior_ids],
        dtype=np.float64,
    )
    return {
        "n": int(selected.size),
        "mean": float(selected.mean()),
        "prevalence_positive": float(np.mean(selected > 0)),
    }


def _candidate_public_summary(
    scores: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {"shared_base": _arm_summary(scores, records, arm="base", placement=None)}
    for placement in PLACEMENTS:
        full, sham, behavior_ids = _paired_rows(scores, records, placement=placement)
        delta = full - sham
        mean, rms, standardized = standardized_paired_effect(delta)
        result[placement] = {
            "full": _arm_summary(scores, records, arm="full", placement=placement),
            "structural_sham": _arm_summary(
                scores, records, arm="structural_sham", placement=placement
            ),
            "inert_length": _arm_summary(
                scores, records, arm="inert_length", placement=placement
            ),
            "full_minus_structural_sham": {
                "n": len(behavior_ids),
                "mean": mean,
                "rms": rms,
                "standardized": standardized,
            },
        }
    return result


def _load_model_readout(torch, transformers, model_path: str, public_plan: dict[str, Any]):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    if (
        int(model.config.hidden_size) != 4096
        or int(model.config.num_hidden_layers) != 32
        or int(model.config.vocab_size) != 128256
    ):
        raise ValueError("Llama 3.1 8B topology drift")
    expected_revision = public_plan["artifacts"]["llama31_model"]["revision"]
    verify_local_snapshot_revision(
        model_path=model_path,
        observed_revision=getattr(model.config, "_commit_hash", None),
        expected_revision=expected_revision,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def _sae_encode(
    *,
    torch,
    sae_path: Path,
    hidden: Any,
    device: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    state = torch.load(sae_path, map_location="cpu", weights_only=True)
    required = {
        "encoder_linear.weight",
        "encoder_linear.bias",
        "decoder_linear.weight",
    }
    if not required.issubset(state):
        raise ValueError("SAE state-dict keys drift")
    encoder = state["encoder_linear.weight"].to(device=device, dtype=torch.float32)
    bias = state["encoder_linear.bias"].to(device=device, dtype=torch.float32)
    decoder = state["decoder_linear.weight"].to(device=device, dtype=torch.float32)
    if tuple(encoder.shape) != (65536, 4096) or tuple(decoder.shape) != (4096, 65536):
        raise ValueError("SAE topology drift")
    with torch.inference_mode():
        activations = torch.relu(hidden.to(device=device, dtype=torch.float32) @ encoder.T + bias)
        decoder_norms = decoder.norm(dim=0)
    diagnostics = {
        "feature_count": int(activations.shape[1]),
        "zero_decoder_norm_count": int((decoder_norms == 0).sum().item()),
        "storage_dtype": str(state["encoder_linear.weight"].dtype),
        "computation_dtype": str(activations.dtype),
    }
    return activations.cpu(), decoder_norms.cpu(), diagnostics


def _verify_probe_tokens(tokenizer, probe_rows: list[dict[str, Any]]) -> list[int]:
    ids = []
    for row in probe_rows:
        token_id = int(row["token_id"])
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if hashlib.sha256(token_text.encode("utf-8")).hexdigest() != row["text_sha256"]:
            raise ValueError(f"probe token semantic hash drift for ID {token_id}")
        ids.append(token_id)
    return ids


def verify_local_snapshot_revision(
    *,
    model_path: str,
    observed_revision: str | None,
    expected_revision: str,
) -> str:
    snapshot = Path(model_path)
    if not snapshot.is_dir() or snapshot.resolve().name != expected_revision:
        raise ValueError("local model snapshot revision path drift")
    if observed_revision not in (None, expected_revision):
        raise ValueError("loaded object revision drift")
    return expected_revision


def verify_source_probe_plan(
    *,
    source_probe_plan_path: Path,
    mechanism_probe: dict[str, Any],
) -> str:
    expected_sha = mechanism_probe["source_plan_sha256"]
    actual_sha = sha256_file(source_probe_plan_path)
    if actual_sha != expected_sha:
        raise ValueError("source probe plan hash drift")
    source = json.loads(source_probe_plan_path.read_text())

    def projected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "token_id": int(row["token_id"]),
                "text_sha256": row["text_sha256"],
            }
            for row in rows
        ]

    if projected(source["probes"]["primary_refusal"]) != mechanism_probe["refusal"]:
        raise ValueError("source refusal probe rows drift")
    if (
        projected(source["probes"]["primary_compliance"])
        != mechanism_probe["compliance"]
    ):
        raise ValueError("source compliance probe rows drift")
    return actual_sha


def _transport_layer(
    *,
    torch,
    hidden: Any,
    jacobian: Any,
    layer: int,
    readout: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = hidden.device
    matrix = jacobian.to(device=device, dtype=torch.float32)
    fitted_norm = float(matrix.norm().item())
    seed = deterministic_transport_seed(RANDOM_TRANSPORT_SEED, layer)
    generator = torch.Generator(device=device).manual_seed(seed)
    random_matrix = torch.randn(
        matrix.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    random_matrix.mul_(fitted_norm / random_matrix.norm())
    realized_norm = float(random_matrix.norm().item())
    relative_error = abs(realized_norm - fitted_norm) / fitted_norm
    if relative_error > RANDOM_NORM_RELATIVE_TOLERANCE:
        raise ValueError("random transport Frobenius norm mismatch")
    vectors = {
        "jacobian_lens": hidden @ matrix.T,
        "identity": hidden,
        "random_gaussian": hidden @ random_matrix.T,
    }
    margins = {}
    with torch.inference_mode():
        for transport in TRANSPORTS:
            means, stds, probe_z = _torch_margin_batch(
                torch,
                vectors[transport],
                readout["mean_embedding"],
                readout["second_moment"],
                readout["probe_embeddings"],
                readout["norm_epsilon"],
            )
            refusal = probe_z[:, : readout["n_refusal"]].mean(dim=1)
            compliance = probe_z[:, readout["n_refusal"] :].mean(dim=1)
            margin = refusal - compliance
            if not (
                torch.isfinite(means).all()
                and torch.isfinite(stds).all()
                and torch.isfinite(margin).all()
            ):
                raise ValueError("non-finite J-lens readout")
            margins[transport] = margin.cpu().numpy().astype(np.float64)
    metadata = {
        "layer": layer,
        "random_seed": seed,
        "fitted_frobenius_norm": fitted_norm,
        "realized_frobenius_norm": realized_norm,
        "relative_norm_error": relative_error,
    }
    return margins, metadata


def _jlens_public_rows(
    *,
    values: dict[str, dict[int, dict[str, np.ndarray]]],
    records_by_partition: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for partition in PARTITIONS:
        records = records_by_partition[partition]
        for placement in PLACEMENTS:
            for layer in sorted(values[partition]):
                for transport in TRANSPORTS:
                    scores = values[partition][layer][transport]
                    full, sham, behavior_ids = _paired_rows(
                        scores,
                        records,
                        placement=placement,
                    )
                    delta = full - sham
                    seed = stable_bootstrap_seed(
                        base_seed=BOOTSTRAP_BASE_SEED,
                        partition=partition,
                        placement=placement,
                        layer=layer,
                        transport=transport,
                        statistic="full_minus_structural_sham",
                    )
                    lower, upper = paired_bootstrap_interval(
                        delta,
                        replicates=BOOTSTRAP_REPLICATES,
                        seed=seed,
                    )
                    output.append(
                        {
                            "partition": partition,
                            "placement": placement,
                            "layer": layer,
                            "transport": transport,
                            "n": len(behavior_ids),
                            "full_mean": float(full.mean()),
                            "structural_sham_mean": float(sham.mean()),
                            "inert_length": _arm_summary(
                                scores,
                                records,
                                arm="inert_length",
                                placement=placement,
                            ),
                            "shared_base": _arm_summary(
                                scores,
                                records,
                                arm="base",
                                placement=None,
                            ),
                            "full_minus_structural_sham": float(delta.mean()),
                            "bootstrap_95": [lower, upper],
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "bootstrap_seed": seed,
                            "pooled_placement_estimate": None,
                        }
                    )
    return output


def run_followup_mechanism_analysis(
    *,
    public_plan_path: Path,
    probe_plan_path: Path,
    discovery_root: Path,
    calibration_root: Path,
    model_path: str,
    lens_path: Path,
    sae_path: Path,
    output_root: Path,
    run_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    import jlens
    import torch
    import transformers

    public_plan = json.loads(public_plan_path.read_text())
    validate_followup_plan(public_plan)
    public_plan_sha = sha256_file(public_plan_path)
    source_commit = _source_commit()
    source_probe_plan_sha = verify_source_probe_plan(
        source_probe_plan_path=probe_plan_path,
        mechanism_probe=public_plan["mechanism_analysis"]["probe"],
    )
    if sha256_file(lens_path) != public_plan["artifacts"]["llama31_lens"]["sha256"]:
        raise ValueError("Jacobian-lens artifact hash drift")
    if sha256_file(sae_path) != public_plan["artifacts"]["llama31_sae"]["sha256"]:
        raise ValueError("SAE artifact hash drift")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)

    records_by_partition = {}
    states_by_partition = {}
    input_manifest_sha = {}
    for partition, root in (
        ("discovery", discovery_root),
        ("calibration", calibration_root),
    ):
        records, states, manifest_sha = _validate_input_partition(
            torch=torch,
            public_plan=public_plan,
            root=root,
            partition=partition,
        )
        records_by_partition[partition] = records
        states_by_partition[partition] = states
        input_manifest_sha[partition] = manifest_sha

    provenance = {
        "study_id": public_plan["study_id"],
        "run_id": run_id,
        "source_commit": source_commit,
        "public_plan_sha256": public_plan_sha,
        "source_probe_plan_sha256": source_probe_plan_sha,
        "discovery_input_manifest_sha256": input_manifest_sha["discovery"],
        "calibration_input_manifest_sha256": input_manifest_sha["calibration"],
        "lens_sha256": sha256_file(lens_path),
        "sae_sha256": sha256_file(sae_path),
        "model_revision": public_plan["artifacts"]["llama31_model"]["revision"],
    }
    summary_path = output_root / "summary.json"
    result_path = output_root / "followup-mechanism.public.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key, value in provenance.items():
            if summary.get(key) != value:
                raise ValueError(f"completed mechanism resume provenance drift: {key}")
        if (
            summary.get("status") != "complete"
            or summary.get("result_path") != str(result_path)
            or not result_path.exists()
            or sha256_file(result_path) != summary.get("result_sha256")
        ):
            raise ValueError("completed mechanism resume result drift")
        summary["model_loaded_this_call"] = False
        return summary

    all_records = records_by_partition["discovery"] + records_by_partition["calibration"]
    all_states = states_by_partition["discovery"] | states_by_partition["calibration"]
    trial_ids = [row["trial_id"] for row in all_records]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("trial IDs collide across discovery/calibration")
    activation_path = output_root / "private" / "sae-activations.pt"
    if activation_path.exists():
        activation_payload = torch.load(
            activation_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            activation_payload.get("provenance") != provenance
            or activation_payload.get("trial_ids") != trial_ids
        ):
            raise ValueError("SAE activation checkpoint provenance drift")
        activations = activation_payload["activations"]
        decoder_norms = activation_payload["decoder_norms"]
        sae_runtime = activation_payload["runtime"]
        if (
            tuple(activations.shape) != (280, 65536)
            or activations.dtype != torch.float32
            or tuple(decoder_norms.shape) != (65536,)
            or decoder_norms.dtype != torch.float32
            or not torch.isfinite(activations).all()
            or not torch.isfinite(decoder_norms).all()
            or (activations < 0).any()
            or (decoder_norms < 0).any()
        ):
            raise ValueError("SAE activation checkpoint tensor drift")
        activation_sha = sha256_file(activation_path)
    else:
        hidden_l19 = torch.stack(
            [all_states[trial_id][SAE_HOOK_LAYER] for trial_id in trial_ids]
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        activations, decoder_norms, sae_runtime = _sae_encode(
            torch=torch,
            sae_path=sae_path,
            hidden=hidden_l19,
            device=device,
        )
        activation_sha = _atomic_torch_save(
            torch,
            activation_path,
            {
                "provenance": provenance,
                "trial_ids": trial_ids,
                "activations": activations,
                "decoder_norms": decoder_norms,
                "runtime": sae_runtime,
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    split_at = len(records_by_partition["discovery"])
    activations_by_partition = {
        "discovery": activations[:split_at].numpy(),
        "calibration": activations[split_at:].numpy(),
    }

    discovery_full = {}
    discovery_sham = {}
    for placement in PLACEMENTS:
        full, sham, _ = _paired_rows(
            activations_by_partition["discovery"],
            records_by_partition["discovery"],
            placement=placement,
        )
        discovery_full[placement] = full
        discovery_sham[placement] = sham
    eligible_rows, single, subspace = discover_candidates(
        full_by_placement=discovery_full,
        sham_by_placement=discovery_sham,
        decoder_norms=decoder_norms.numpy(),
        minimum_full_prevalence=MINIMUM_DISCOVERY_FULL_PREVALENCE,
        maximum_subspace_features=MAXIMUM_SUBSPACE_FEATURES,
    )
    candidates = (single, subspace)
    candidate_scores_by_partition = {
        partition: {
            candidate.candidate_id: candidate_scores(
                activations_by_partition[partition], candidate
            )
            for candidate in candidates
        }
        for partition in PARTITIONS
    }
    calibration_deltas = {}
    for candidate in candidates:
        payload: dict[str, Any] = {"candidate": candidate}
        for placement in PLACEMENTS:
            full, sham, _ = _paired_rows(
                candidate_scores_by_partition["calibration"][candidate.candidate_id],
                records_by_partition["calibration"],
                placement=placement,
            )
            payload[placement] = full - sham
        calibration_deltas[candidate.candidate_id] = payload
    selected, calibration_ranking = rank_calibration_candidates(calibration_deltas)

    diagnostics_path = output_root / "private" / "sae-discovery-diagnostics.json"
    diagnostics_payload = {
        **provenance,
        "activation_artifact_sha256": activation_sha,
        "selection_rule": public_plan["mechanism_analysis"]["sae_candidate_construction"],
        "eligible_feature_count": len(eligible_rows),
        "eligible_feature_rows": eligible_rows,
        "discovery_candidates": [asdict(candidate) for candidate in candidates],
        "calibration_ranking": calibration_ranking,
        "selected_candidate": asdict(selected) if selected is not None else None,
    }
    write_json_atomic(diagnostics_path, diagnostics_payload)
    diagnostics_path.chmod(0o600)

    sae_public = {
        "status": (
            "candidate_selected_threshold_pending_utility_calibration"
            if selected is not None
            else "no_candidate_advances"
        ),
        "hook_layer": SAE_HOOK_LAYER,
        "activation_artifact_sha256": activation_sha,
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "eligible_feature_count": len(eligible_rows),
        "discovery_candidates": [asdict(candidate) for candidate in candidates],
        "calibration_ranking": calibration_ranking,
        "selected_candidate": asdict(selected) if selected is not None else None,
        "candidate_arm_matrix": {
            partition: {
                candidate.candidate_id: _candidate_public_summary(
                    candidate_scores_by_partition[partition][candidate.candidate_id],
                    records_by_partition[partition],
                )
                for candidate in candidates
            }
            for partition in PARTITIONS
        },
        "threshold": {
            "status": "not_fit",
            "reason": (
                "ordinary_benign and structured_benign utility-calibration "
                "negative strata have not been generated"
            ),
            "detector_or_circuit_breaker_claim": False,
        },
    }

    dense_full = {}
    dense_sham = {}
    discovery_l19 = np.stack(
        [
            states_by_partition["discovery"][row["trial_id"]][SAE_HOOK_LAYER]
            .float()
            .numpy()
            for row in records_by_partition["discovery"]
        ]
    )
    for placement in PLACEMENTS:
        full, sham, _ = _paired_rows(
            discovery_l19,
            records_by_partition["discovery"],
            placement=placement,
        )
        dense_full[placement] = full
        dense_sham[placement] = sham
    dense_direction = fit_common_dense_projection(dense_full, dense_sham)
    dense_path = output_root / "private" / "dense-mean-difference.pt"
    if dense_path.exists():
        dense_payload = torch.load(dense_path, map_location="cpu", weights_only=True)
        if dense_payload.get("provenance") != provenance:
            raise ValueError("dense comparator checkpoint provenance drift")
        cached_direction = dense_payload["direction"].numpy().astype(np.float64)
        if not np.array_equal(
            cached_direction.astype(np.float32),
            dense_direction.astype(np.float32),
        ):
            raise ValueError("dense comparator checkpoint value drift")
        dense_direction_sha = sha256_file(dense_path)
    else:
        dense_direction_sha = _atomic_torch_save(
            torch,
            dense_path,
            {
                "provenance": provenance,
                "direction": torch.from_numpy(dense_direction.astype(np.float32)),
            },
        )
    dense_summary = {
        "direction_artifact_sha256": dense_direction_sha,
        "partitions": {},
    }
    for partition in PARTITIONS:
        hidden = np.stack(
            [
                states_by_partition[partition][row["trial_id"]][SAE_HOOK_LAYER]
                .float()
                .numpy()
                for row in records_by_partition[partition]
            ]
        )
        scores = hidden @ dense_direction
        dense_summary["partitions"][partition] = _candidate_public_summary(
            scores,
            records_by_partition[partition],
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    verify_local_snapshot_revision(
        model_path=model_path,
        observed_revision=getattr(tokenizer, "_commit_hash", None),
        expected_revision=public_plan["artifacts"]["llama31_model"]["revision"],
    )
    mechanism_plan = public_plan["mechanism_analysis"]
    refusal_ids = _verify_probe_tokens(
        tokenizer,
        mechanism_plan["probe"]["refusal"],
    )
    compliance_ids = _verify_probe_tokens(
        tokenizer,
        mechanism_plan["probe"]["compliance"],
    )
    model = _load_model_readout(torch, transformers, model_path, public_plan)
    output_weight = model.lm_head.weight.detach().float()
    norm_weight = model.model.norm.weight.detach().float()
    effective = output_weight * norm_weight[None, :]
    probe_ids = refusal_ids + compliance_ids
    readout = {
        "mean_embedding": effective.mean(dim=0),
        "second_moment": effective.T @ effective / effective.shape[0],
        "probe_embeddings": effective[probe_ids],
        "norm_epsilon": float(model.model.norm.variance_epsilon),
        "n_refusal": len(refusal_ids),
    }
    fixture = (
        states_by_partition["discovery"][
            records_by_partition["discovery"][0]["trial_id"]
        ][30]
        .to(device=model.lm_head.weight.device, dtype=torch.float32)
    )
    with torch.inference_mode():
        normalized = fixture * torch.rsqrt(
            fixture.square().mean() + readout["norm_epsilon"]
        )
        full_logits = model.lm_head.weight.float() @ (normalized * norm_weight)
        full_mean = full_logits.mean()
        full_std = full_logits.std(unbiased=False)
        full_probe_z = (full_logits[probe_ids] - full_mean) / full_std
        module_logits = model.lm_head(
            model.model.norm(fixture.to(dtype=model.model.norm.weight.dtype))
        ).float()
        module_probe_z = (
            module_logits[probe_ids] - module_logits.mean()
        ) / module_logits.std(unbiased=False)
        moment_mean, moment_std, moment_probe_z = _torch_margin_batch(
            torch,
            fixture[None, :],
            readout["mean_embedding"],
            readout["second_moment"],
            readout["probe_embeddings"],
            readout["norm_epsilon"],
        )
    moment_equivalence = {
        "mean_abs_error": float(abs(full_mean - moment_mean[0]).item()),
        "std_abs_error": float(abs(full_std - moment_std[0]).item()),
        "probe_max_abs_error": float(
            (full_probe_z - moment_probe_z[0]).abs().max().item()
        ),
        "module_probe_max_abs_error": float(
            (module_probe_z - moment_probe_z[0]).abs().max().item()
        ),
        "analytic_tolerance": 2e-3,
        "module_tolerance": 2e-2,
    }
    if (
        max(
            moment_equivalence["mean_abs_error"],
            moment_equivalence["std_abs_error"],
            moment_equivalence["probe_max_abs_error"],
        )
        > moment_equivalence["analytic_tolerance"]
        or moment_equivalence["module_probe_max_abs_error"]
        > moment_equivalence["module_tolerance"]
    ):
        raise ValueError("vocabulary-moment equivalence gate failed")
    del effective, output_weight
    lens = jlens.JacobianLens.load(lens_path)
    source_layers = tuple(int(layer) for layer in lens.source_layers)
    if source_layers != tuple(range(31)):
        raise ValueError("Jacobian-lens source-layer topology drift")

    jlens_values: dict[str, dict[int, dict[str, np.ndarray]]] = {
        partition: {} for partition in PARTITIONS
    }
    layer_metadata = []
    for layer in source_layers:
        private_layer = output_root / "private" / "jlens" / f"layer-{layer:03d}.json"
        if private_layer.exists():
            layer_payload = json.loads(private_layer.read_text())
            for key, value in provenance.items():
                if layer_payload.get(key) != value:
                    raise ValueError(f"layer {layer}: resume provenance drift for {key}")
            if (
                layer_payload.get("layer") != layer
                or layer_payload.get("trial_ids") != trial_ids
                or set(layer_payload.get("margins", {})) != set(TRANSPORTS)
            ):
                raise ValueError(f"layer {layer}: resume topology drift")
            margins = {
                transport: _finite_array(
                    layer_payload["margins"][transport],
                    name=f"layer {layer} {transport}",
                    ndim=1,
                )
                for transport in TRANSPORTS
            }
            if any(values.shape != (280,) for values in margins.values()):
                raise ValueError(f"layer {layer}: resume row-count drift")
            metadata = {
                key: layer_payload[key]
                for key in (
                    "layer",
                    "random_seed",
                    "fitted_frobenius_norm",
                    "realized_frobenius_norm",
                    "relative_norm_error",
                )
            }
            if (
                metadata["random_seed"]
                != deterministic_transport_seed(RANDOM_TRANSPORT_SEED, layer)
                or metadata["relative_norm_error"] > RANDOM_NORM_RELATIVE_TOLERANCE
            ):
                raise ValueError(f"layer {layer}: random transport resume drift")
        else:
            hidden = torch.stack(
                [
                    states_by_partition[partition][row["trial_id"]][layer]
                    for partition in PARTITIONS
                    for row in records_by_partition[partition]
                ]
            ).to(device=model.lm_head.weight.device, dtype=torch.float32)
            margins, metadata = _transport_layer(
                torch=torch,
                hidden=hidden,
                jacobian=lens.jacobians[layer],
                layer=layer,
                readout=readout,
            )
            write_json_atomic(
                private_layer,
                {
                    **provenance,
                    **metadata,
                    "trial_ids": trial_ids,
                    "margins": {
                        transport: values.tolist()
                        for transport, values in margins.items()
                    },
                },
            )
            private_layer.chmod(0o600)
        layer_metadata.append(metadata)
        offset = 0
        for partition in PARTITIONS:
            count = len(records_by_partition[partition])
            jlens_values[partition][layer] = {
                transport: values[offset : offset + count]
                for transport, values in margins.items()
            }
            offset += count
        print(f"completed follow-up mechanism layer {layer}", flush=True)

    jlens_rows = _jlens_public_rows(
        values=jlens_values,
        records_by_partition=records_by_partition,
    )
    public_result = {
        "schema_version": "1.0",
        **provenance,
        "status": "complete",
        "placement_orderings": list(PLACEMENTS),
        "pooled_placement_estimate_reported": False,
        "sae": sae_public,
        "dense_mean_difference_comparator": dense_summary,
        "jlens": {
            "probe": mechanism_plan["probe"],
            "transports": list(TRANSPORTS),
            "source_layers": list(source_layers),
            "layer_transport_metadata": layer_metadata,
            "rows": jlens_rows,
            "trajectory_role": "secondary_descriptive",
            "equivalence_or_no_moderation_claim": False,
            "vocabulary_moment_equivalence": moment_equivalence,
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "jlens": getattr(jlens, "__version__", "unreported"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "state_storage_dtype": "torch.bfloat16",
            "transport_computation_dtype": "torch.float32",
            "sae_computation_dtype": sae_runtime["computation_dtype"],
        },
        "raw_data_policy": {
            "raw_prompts_opened": False,
            "raw_generations_opened": False,
            "reconstructive_token_ids_opened": False,
            "state_tensors": "private",
            "per_observation_sae_and_jlens_values": "private",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(result_path, public_result)
    summary = {
        "status": "complete",
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "source_commit": source_commit,
        "public_plan_sha256": public_plan_sha,
        "selected_candidate": asdict(selected) if selected is not None else None,
        "threshold_status": sae_public["threshold"]["status"],
        "elapsed_seconds": public_result["elapsed_seconds"],
        **provenance,
        "model_loaded_this_call": True,
    }
    write_json_atomic(summary_path, summary)
    return summary
