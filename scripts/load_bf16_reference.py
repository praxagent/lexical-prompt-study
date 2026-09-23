"""Direct original-checkpoint CPU BF16 loader; inert until explicitly called.

This loader never first loads FP32 parameters and never casts a loaded model.
Buffer precision is reported by the separate BF16 primitive, not inferred from
parameter precision. No warmup, generation or reference forward is performed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace


def require(ok, reason):
    if not ok:
        raise ValueError("a188_loader_" + reason)


def load_cpu_bf16(config, engine):
    if not __debug__:
        raise RuntimeError("a188_optimization_forbidden")
    require(engine.runtime_versions() == config["runtime_versions"], "runtime_versions")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "cuda_hidden")
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "offline",
    )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    require(not torch.cuda.is_initialized(), "cuda_initialized")
    require(not torch.is_autocast_enabled("cpu"), "cpu_autocast")
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    path = Path(config["model_path"])
    original = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    require(getattr(original, "quantization_config", None) is None, "unquantized_checkpoint")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        attn_implementation="sdpa",
    ).eval()
    require(
        not getattr(model, "is_loaded_in_4bit", False)
        and not getattr(model, "is_loaded_in_8bit", False),
        "quantized_model",
    )
    parameters = list(model.parameters())
    require(
        bool(parameters)
        and all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in parameters),
        "parameter_device_dtype",
    )
    require(all(b.device.type == "cpu" for b in model.buffers()), "buffer_device")
    require(
        all(not type(item).__module__.startswith("bitsandbytes") for item in model.modules()),
        "quantized_module",
    )
    require(model.config._attn_implementation == "sdpa" and model.training is False, "model_mode")
    require(
        not torch.cuda.is_initialized() and not torch.is_autocast_enabled("cpu"), "device_after"
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    require(
        hashlib.sha256(tokenizer.get_chat_template().encode("utf-8")).hexdigest()
        == config["chat_template_sha256"],
        "native_template",
    )
    eos = model.generation_config.eos_token_id
    eos = eos if type(eos) is list else [eos]
    require(
        bool(eos)
        and all(type(value) is int and 0 <= value < model.config.vocab_size for value in eos),
        "eos",
    )
    return SimpleNamespace(model=model, tokenizer=tokenizer, eos_ids=sorted(set(eos)))
