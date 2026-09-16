"""Pinned local A163 inference, continuation likelihoods, and private receipts.

Imports are inert: torch/transformers load only when explicitly requested. This
module never downloads models, reads credentials, or prints model responses.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time
import traceback
import uuid

from .instruction_binding_tasks import LookupWorld, SYMBOLS, oracle, score_response

ANSWER_PREFIX = '{"answer":"'
TEMPLATE_DATE = "26 Jul 2024"
CONFIG_KEYS = {
    "schema_version", "model_path", "model_revision", "model_files_sha256",
    "chat_template_sha256", "runtime_versions", "quantization", "dtype",
    "attention_implementation", "device", "max_prompt_tokens", "max_new_tokens",
    "seed", "cpu_threads", "gpu_memory_gib",
}
VERSION_PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "safetensors",
                    "jinja2", "tokenizers", "numpy", "huggingface-hub")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode() + b"\n"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def snapshot_manifest(path: Path) -> dict[str, str]:
    """Hash the actual local config/tokenizer/weight files; no network access."""
    paths = sorted({*path.glob("*.json"), *path.glob("*.safetensors"),
                    *path.glob("*.jinja"), *path.glob("*.model")})
    return {item.name: file_digest(item) for item in paths}


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in VERSION_PACKAGES}}


def validate_config(config: dict) -> None:
    if set(config) != CONFIG_KEYS or config.get("schema_version") != "a163-runtime-v1":
        raise ValueError("invalid runtime config schema")
    if not Path(config["model_path"]).is_absolute():
        raise ValueError("model path must be absolute")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("model revision must be a pinned commit")
    files = config["model_files_sha256"]
    if not isinstance(files, dict) or not {"config.json", "tokenizer_config.json",
                                         "tokenizer.json"}.issubset(files):
        raise ValueError("model/tokenizer file bindings required")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("local safetensors weights required")
    if any(Path(name).name != name or not _is_hash(value) for name, value in files.items()):
        raise ValueError("invalid model file binding")
    if not _is_hash(config["chat_template_sha256"]):
        raise ValueError("chat template hash required")
    if set(config["runtime_versions"]) != {"python", *VERSION_PACKAGES} or any(
        not isinstance(value, str) or not value for value in config["runtime_versions"].values()
    ):
        raise ValueError("complete runtime version pins required")
    if config["quantization"] != "nf4" or config["dtype"] != "bfloat16":
        raise ValueError("this instrument requires NF4 with BF16 compute")
    if config["attention_implementation"] not in ("eager", "sdpa"):
        raise ValueError("unsupported attention implementation")
    if config["device"] != "cuda:0" or config["max_new_tokens"] != 64:
        raise ValueError("instrument requires CUDA device zero and 64-token cap")
    for name, lower, upper in (("max_prompt_tokens", 64, 8192), ("seed", 0, 2**32 - 1),
                               ("cpu_threads", 1, 4), ("gpu_memory_gib", 1, 10)):
        if type(config[name]) is not int or not lower <= config[name] <= upper:
            raise ValueError("invalid bounded runtime setting")


def _world(trial: dict) -> LookupWorld:
    payload = trial["world"]
    return LookupWorld(
        depth=payload["depth"],
        tables=tuple(tuple(payload["tables"][f"table_{hop}"][symbol] for symbol in SYMBOLS)
                     for hop in range(1, payload["depth"] + 1)),
        query_keys=(payload["queries"]["A"], payload["queries"]["B"]),
    )


def validate_plan(plan: dict) -> None:
    if plan.get("schema_version") != "a163-plan-v1" or plan.get("partition") not in (
        "development", "heldout", "synthetic"
    ):
        raise ValueError("invalid plan identity")
    trials = plan.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("nonempty planned trials required")
    ids = set()
    for trial in trials:
        if not re.fullmatch(r"[0-9a-f]{24}", trial["trial_id"]) or trial["trial_id"] in ids:
            raise ValueError("invalid or repeated trial identity")
        ids.add(trial["trial_id"])
        world = _world(trial)
        if world.world_id != trial["world_id"] or world.depth != trial["depth"]:
            raise ValueError("world identity mismatch")
        if trial["selected_answer"] != oracle(world, trial["selector"]) or (
            trial["unselected_answer"] != oracle(world, "B" if trial["selector"] == "A" else "A")
        ):
            raise ValueError("planned oracle mismatch")
        if type(trial["diagnostic"]) is not bool:
            raise ValueError("diagnostic flag required")
        messages = trial["messages"]
        if not isinstance(messages, list) or len(messages) != 2 or [
            message.get("role") for message in messages
        ] != ["system", "user"] or any(
            set(message) != {"role", "content"} or type(message["content"]) is not str
            for message in messages
        ):
            raise ValueError("exact system and user messages required")


def _ids(values: object) -> list[int]:
    if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values):
        raise ValueError("tokenizer must return a nonempty flat integer token list")
    return values


def prepare_trial(tokenizer, trial: dict, config: dict) -> dict:
    """Verify native tokenization and both exact continuation boundaries."""
    template = tokenizer.get_chat_template()
    if digest(template.encode()) != config["chat_template_sha256"]:
        raise ValueError("native chat template drift")
    kwargs = dict(add_generation_prompt=True, date_string=TEMPLATE_DATE)
    rendered = tokenizer.apply_chat_template(trial["messages"], tokenize=False, **kwargs)
    prompt = _ids(tokenizer.apply_chat_template(
        trial["messages"], tokenize=True, return_dict=False, **kwargs))
    if prompt != _ids(tokenizer.encode(rendered, add_special_tokens=False)):
        raise ValueError("native rendered token mismatch")
    if any(message["content"] not in rendered for message in trial["messages"]):
        raise ValueError("native template omitted message content")
    if len(prompt) > config["max_prompt_tokens"]:
        raise ValueError("prompt exceeds frozen limit")
    prepared = {"rendered_text": rendered, "prompt_token_ids": prompt,
                "forced_prefix_token_ids": None, "candidate_token_ids": None,
                "chat_template_sha256": config["chat_template_sha256"],
                "likelihood_error": None}
    try:
        prepared.update(_prepare_candidates(tokenizer, rendered, prompt, trial, config))
    except Exception as exc:
        prepared["likelihood_error"] = safe_error(exc, "likelihood_tokenization_failed")
    return prepared


def _prepare_candidates(tokenizer, rendered: str, prompt: list[int], trial: dict, config: dict) -> dict:
    forced = _ids(tokenizer.encode(rendered + ANSWER_PREFIX, add_special_tokens=False))
    if forced[:len(prompt)] != prompt or len(forced) <= len(prompt):
        raise ValueError("assistant prefix changes native prompt tokenization")
    candidates = {}
    for key in ("selected", "unselected"):
        suffix = trial[key + "_answer"] + '"}'
        full = _ids(tokenizer.encode(rendered + ANSWER_PREFIX + suffix, add_special_tokens=False))
        if full[:len(forced)] != forced or len(full) <= len(forced):
            raise ValueError("candidate changes fixed prefix tokenization")
        candidate = full[len(forced):]
        if tokenizer.decode(candidate, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != suffix:
            raise ValueError("candidate suffix does not round trip exactly")
        candidates[key] = candidate
    if max(
        len(forced) + len(candidate) for candidate in candidates.values()
    ) > config["max_prompt_tokens"] + config["max_new_tokens"]:
        raise ValueError("prompt or likelihood context exceeds frozen limit")
    return {"forced_prefix_token_ids": forced, "candidate_token_ids": candidates}


def continuation_logprob(model, torch, prefix: list[int], continuation: list[int], device) -> float:
    """Sum full-suffix nats with the causal shift; includes JSON closure, no EOS.

    Each candidate has a fresh full forward pass, so mutable KV state is never
    shared between selected and alternative branches. Only suffix logits are
    materialized by the Llama output head.
    """
    ids = torch.tensor([prefix + continuation], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(continuation) + 1)
        logits = result.logits
        if tuple(logits.shape[:2]) != (1, len(continuation) + 1):
            raise ValueError("unexpected likelihood logits shape")
        targets = torch.tensor(continuation, dtype=torch.long, device=logits.device)
        log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        value = float(log_probs.gather(1, targets[:, None]).sum().item())
    if not math.isfinite(value) or value > 0:
        raise ValueError("invalid candidate log probability")
    return value


class LocalRuntime:
    def __init__(self, model, tokenizer, torch, config: dict):
        self.model, self.tokenizer, self.torch, self.config = model, tokenizer, torch, config
        self.device = model.get_input_embeddings().weight.device
        eos = model.generation_config.eos_token_id
        self.eos_ids = sorted(set(eos if isinstance(eos, list) else [eos]))
        if not self.eos_ids or any(type(value) is not int for value in self.eos_ids):
            raise ValueError("explicit model EOS tokens required")

    def generate(self, prepared: dict) -> tuple[list[int], str, str]:
        from transformers import GenerationConfig
        torch = self.torch
        ids = torch.tensor([prepared["prompt_token_ids"]], dtype=torch.long, device=self.device)
        generation = GenerationConfig(
            max_new_tokens=self.config["max_new_tokens"], do_sample=False, num_beams=1,
            use_cache=True, eos_token_id=self.eos_ids, pad_token_id=self.eos_ids[0],
            bos_token_id=self.tokenizer.bos_token_id,
        )
        with torch.inference_mode():
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=generation, logits_to_keep=1)
        all_ids = result[0].tolist()
        if all_ids[:ids.shape[1]] != prepared["prompt_token_ids"]:
            raise ValueError("generation did not preserve prompt tokens")
        generated = all_ids[ids.shape[1]:]
        if not generated or len(generated) > self.config["max_new_tokens"]:
            raise ValueError("invalid generated token count")
        eos = generated[-1] in self.eos_ids
        if any(token in self.eos_ids for token in generated[:-1]) or (
            not eos and len(generated) != self.config["max_new_tokens"]
        ):
            raise ValueError("unexpected generation termination")
        text = self.tokenizer.decode(generated[:-1] if eos else generated,
                                     skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        return generated, text, "eos" if eos else "length"

    def likelihoods(self, prepared: dict) -> dict:
        if prepared["likelihood_error"] is not None:
            raise ValueError("likelihood token boundary qualification failed")
        values = {}
        for key in ("selected", "unselected"):
            tokens = prepared["candidate_token_ids"][key]
            values[key] = (continuation_logprob(self.model, self.torch,
                                               prepared["forced_prefix_token_ids"],
                                               tokens, self.device), len(tokens))
        return {"status": "ok", "selected_logprob": values["selected"][0],
                "alternative_logprob": values["unselected"][0],
                "selected_token_count": values["selected"][1],
                "alternative_token_count": values["unselected"][1]}


def load_runtime(config: dict) -> LocalRuntime:
    validate_config(config)
    if runtime_versions() != config["runtime_versions"]:
        raise ValueError("runtime version drift")
    path = Path(config["model_path"])
    if snapshot_manifest(path) != config["model_files_sha256"]:
        raise ValueError("model file hash drift")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info(0)
    limit = config["gpu_memory_gib"] * 1024**3
    if free < limit + 512 * 1024**2:
        raise RuntimeError("insufficient shared GPU headroom")
    torch.cuda.set_per_process_memory_fraction(limit / total, 0)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if digest(tokenizer.get_chat_template().encode()) != config["chat_template_sha256"]:
        raise ValueError("native chat template drift")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, device_map={"": config["device"]},
        quantization_config=quantization,
        attn_implementation=config["attention_implementation"],
    ).eval()
    if not getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("NF4 model loading was not applied")
    if model.config.max_position_embeddings < config["max_prompt_tokens"] + 64:
        raise ValueError("frozen context exceeds model capacity")
    return LocalRuntime(model, tokenizer, torch, config)


def numeric_record(plan: dict, trial: dict) -> dict:
    return {
        "schema_version": "a163-cell-v1", "trial_id": trial["trial_id"],
        "world_id": trial["world_id"], "task_depth": trial["depth"],
        "partition": plan["partition"], "renderer": trial["renderer"],
        "query_order": trial["query_order"], "table_order": trial["table_order"],
        "scaffold": trial["scaffold_kind"], "placement": trial["placement"],
        "selector": trial["selector"], "diagnostic": trial["diagnostic"],
        "generation_status": "infrastructure_failed", "finish_reason": None,
        "score": {"category": None, "strict_correct": None},
        "teacher_forced": {"status": "failed", "selected_logprob": None,
                           "alternative_logprob": None, "selected_token_count": None,
                           "alternative_token_count": None},
    }


def safe_error(exc: Exception, code: str) -> dict:
    """Diagnostic locations without exception text, locals, or source lines."""
    return {"code": code, "exception_type": type(exc).__name__, "frames": [
        {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in traceback.extract_tb(exc.__traceback__)
    ]}


def evaluate_trial(runtime, plan: dict, trial: dict, prepared: dict,
                   previous_result: dict | None = None) -> tuple[dict, dict]:
    record = numeric_record(plan, trial)
    private = {"generated_token_ids": None, "response_text": None, "errors": [],
               "generation_reused": False, "likelihood_reused": False}
    if previous_result and previous_result["record"]["generation_status"] == "completed":
        for key in ("generation_status", "finish_reason", "score"):
            record[key] = previous_result["record"][key]
        for key in ("generated_token_ids", "response_text"):
            private[key] = previous_result["private"][key]
        private["generation_reused"] = True
    else:
        try:
            tokens, text, finish = runtime.generate(prepared)
            category = score_response(text, _world(trial), trial["selector"])
            record.update(generation_status="completed", finish_reason=finish,
                          score={"category": category,
                                 "strict_correct": category == "exact" and finish == "eos"})
            private.update(generated_token_ids=tokens, response_text=text)
        except Exception as exc:
            private["errors"].append(safe_error(exc, "generation_failed"))
    if previous_result and previous_result["record"]["teacher_forced"]["status"] == "ok":
        record["teacher_forced"] = previous_result["record"]["teacher_forced"]
        private["likelihood_reused"] = True
    elif prepared["likelihood_error"] is not None:
        private["errors"].append(prepared["likelihood_error"])
    else:
        try:
            record["teacher_forced"] = runtime.likelihoods(prepared)
        except Exception as exc:
            private["errors"].append(safe_error(exc, "likelihood_failed"))
    return record, private


def validate_result(result: dict, attempt: dict, trial: dict, plan: dict,
                    attempt_sha256: str) -> dict:
    """Recheck numeric metadata/scoring against the retained private response."""
    if result.get("schema_version") != "a163-result-v1" or (
        result.get("attempt_sha256") != attempt_sha256
    ):
        raise ValueError("receipt attempt binding drift")
    record, private = result["record"], result["private"]
    expected = numeric_record(plan, trial)
    if set(record) != set(expected) or any(
        record[key] != value for key, value in expected.items()
        if key not in {"generation_status", "finish_reason", "score", "teacher_forced"}
    ):
        raise ValueError("receipt numeric metadata drift")
    if record["generation_status"] == "completed":
        tokens = _ids(private["generated_token_ids"])
        eos_ids = attempt["eos_token_ids"]
        eos = tokens[-1] in eos_ids
        if len(tokens) > 64 or any(token in eos_ids for token in tokens[:-1]) or (
            not eos and len(tokens) != 64
        ):
            raise ValueError("receipt generation termination mismatch")
        finish = "eos" if eos else "length"
        category = score_response(private["response_text"], _world(trial), trial["selector"])
        score = {"category": category, "strict_correct": category == "exact" and finish == "eos"}
        if record["finish_reason"] != finish or record["score"] != score:
            raise ValueError("receipt score does not replay")
    elif record["generation_status"] == "infrastructure_failed":
        if (record["finish_reason"] is not None or record["score"] != expected["score"]
                or private["response_text"] is not None or private["generated_token_ids"] is not None):
            raise ValueError("failed generation must retain missing fields")
    else:
        raise ValueError("unknown generation status")
    forced = record["teacher_forced"]
    if forced["status"] == "failed":
        if forced != expected["teacher_forced"]:
            raise ValueError("failed likelihood must retain missing fields")
    elif forced["status"] == "ok":
        if set(forced) != set(expected["teacher_forced"]):
            raise ValueError("invalid likelihood fields")
        for label, candidate in (("selected", "selected"), ("alternative", "unselected")):
            value = forced[label + "_logprob"]
            if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
                raise ValueError("invalid likelihood value")
            count = forced[label + "_token_count"]
            if type(count) is not int or count != len(attempt["prepared"]["candidate_token_ids"][candidate]):
                raise ValueError("candidate length mismatch")
    else:
        raise ValueError("unknown likelihood status")
    return record


def write_private(path: Path, value: dict) -> None:
    """Atomic, immutable, owner-only JSON; a partial write is never a receipt."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = canonical(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("immutable private artifact mismatch")
        return
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _read_bound(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if not _is_hash(expected) or digest(raw) != expected:
        raise ValueError("input hash drift")
    return json.loads(raw)


def _history(root: Path, trial: dict, plan: dict, header: dict) -> list[tuple]:
    history, previous, previous_sha = [], None, None
    for index, path in enumerate(sorted(root.glob("attempt-*/attempt.json")), start=1):
        if path.parent.name != f"attempt-{index:02d}" or index > 3:
            raise ValueError("invalid attempt sequence")
        attempt = json.loads(path.read_bytes())
        if (attempt["trial_sha256"] != digest(canonical(trial)) or attempt["run"] != header
                or attempt["previous_result_sha256"] != previous_sha):
            raise ValueError("attempt input or retry lineage drift")
        result_path = path.with_name("result.json")
        result = None
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            validate_result(result, attempt, trial, plan, file_digest(path))
            if previous:
                if previous["record"]["generation_status"] == "completed":
                    if any(result["record"][key] != previous["record"][key] for key in (
                        "generation_status", "finish_reason", "score"
                    )) or any(result["private"][key] != previous["private"][key] for key in (
                        "generated_token_ids", "response_text"
                    )):
                        raise ValueError("retry changed completed generation")
                if previous["record"]["teacher_forced"]["status"] == "ok" and (
                    result["record"]["teacher_forced"] != previous["record"]["teacher_forced"]
                ):
                    raise ValueError("retry changed completed likelihood")
            previous, previous_sha = result, file_digest(result_path)
        history.append((path, attempt, result))
    return history


def export_records(*, plan_path: Path, config_path: Path, output_root: Path,
                   expected_plan_sha256: str, expected_config_sha256: str) -> list[dict]:
    """Replay finished durable receipts without loading a model or emitting text."""
    plan = _read_bound(plan_path, expected_plan_sha256)
    _read_bound(config_path, expected_config_sha256)
    validate_plan(plan)
    with (output_root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        header = json.loads((output_root / "run.json").read_bytes())
        if header != _header(expected_plan_sha256, expected_config_sha256):
            raise ValueError("run source or input lineage drift")
        records = []
        for trial in plan["trials"]:
            history = _history(output_root / "trials" / trial["trial_id"], trial, plan, header)
            if not history:
                continue
            result = history[-1][2]
            if result is None:
                # An interrupted attempt is unresolved, never a fabricated fail.
                continue
            records.append(result["record"])
        return records


def _header(plan_sha: str, config_sha: str) -> dict:
    return {"schema_version": "a163-run-v1", "plan_sha256": plan_sha,
            "config_sha256": config_sha, "source_sha256": file_digest(Path(__file__)),
            "generator_sha256": file_digest(Path(__file__).with_name("instruction_binding_tasks.py"))}


def run_plan(*, plan_path: Path, config_path: Path, output_root: Path,
             expected_plan_sha256: str, expected_config_sha256: str,
             max_trials: int | None = None, retry_failed: bool = False,
             runtime_loader=load_runtime) -> dict:
    plan = _read_bound(plan_path, expected_plan_sha256)
    config = _read_bound(config_path, expected_config_sha256)
    validate_plan(plan)
    validate_config(config)
    if max_trials is not None and (type(max_trials) is not int or max_trials < 1):
        raise ValueError("max_trials must be a positive integer")
    root = output_root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("private output must remain outside the Git worktree")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    header = _header(expected_plan_sha256, expected_config_sha256)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_private(root / "run.json", header)
        records, runtime, launched = [], None, 0
        for trial in plan["trials"]:
            trial_root = root / "trials" / trial["trial_id"]
            existing = _history(trial_root, trial, plan, header)
            result = None
            if existing:
                result = existing[-1][2]
                if result is None and not retry_failed:
                    raise RuntimeError("interrupted attempt requires explicit retry")
                if result and (not retry_failed or (
                    result["record"]["generation_status"] == "completed"
                    and result["record"]["teacher_forced"]["status"] == "ok"
                )):
                    records.append(result["record"])
                    continue
            if max_trials is not None and launched >= max_trials:
                if result:
                    records.append(result["record"])
                continue
            if len(existing) >= 3:
                raise RuntimeError("explicit infrastructure retry allowance exhausted")
            if runtime is None:
                runtime = runtime_loader(config)
            prepared = prepare_trial(runtime.tokenizer, trial, config)
            attempt_dir = trial_root / f"attempt-{len(existing) + 1:02d}"
            attempt_path = attempt_dir / "attempt.json"
            previous = next(((path, value) for path, _, value in reversed(existing)
                             if value is not None), None)
            attempt = {"schema_version": "a163-attempt-v1", "run": header,
                       "trial_sha256": digest(canonical(trial)), "prepared": prepared,
                       "eos_token_ids": runtime.eos_ids,
                       "previous_result_sha256": file_digest(previous[0].with_name("result.json"))
                       if previous else None}
            write_private(attempt_path, attempt)
            started = time.monotonic()
            record, private = evaluate_trial(runtime, plan, trial, prepared,
                                             previous_result=previous[1] if previous else None)
            result = {"schema_version": "a163-result-v1", "attempt_sha256": file_digest(attempt_path),
                      "record": record, "private": private,
                      "elapsed_seconds": time.monotonic() - started}
            validate_result(result, attempt, trial, plan, file_digest(attempt_path))
            write_private(attempt_dir / "result.json", result)
            records.append(record)
            launched += 1
        write_private(root / f"records-{digest(canonical(records))}.json",
                      {"schema_version": "a163-numeric-records-v1", "run": header, "records": records})
        return {"schema_version": "a163-progress-v1", "planned": len(plan["trials"]),
                "recorded": len(records), "launched_this_call": launched,
                "generation_completed": sum(r["generation_status"] == "completed" for r in records),
                "likelihood_completed": sum(r["teacher_forced"]["status"] == "ok" for r in records),
                "strict_correct": sum(r["score"]["strict_correct"] is True for r in records),
                "length_terminated": sum(r["finish_reason"] == "length" for r in records),
                "all_cells_recorded": len(records) == len(plan["trials"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "config", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), \
                contextlib.redirect_stderr(quiet):
            summary = run_plan(plan_path=args.plan, config_path=args.config,
                               output_root=args.output_root,
                               expected_plan_sha256=args.plan_sha256,
                               expected_config_sha256=args.config_sha256,
                               max_trials=args.max_trials, retry_failed=args.retry_failed)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
