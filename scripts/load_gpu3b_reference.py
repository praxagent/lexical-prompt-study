"""Direct pinned 3B CUDA BF16 loader, inert until explicitly called.

Streaming from_pretrained uses a CUDA device map without a full CPU model replica.
No post-load model cast, offload, quantization, warmup or generation is performed.
Actual buffers may retain FP32; the separate primitive records their native types.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace


def require(ok, reason):
    if not ok:
        raise ValueError("a189_loader_" + reason)


def load_cuda_bf16(config, engine):
    if not __debug__:
        raise RuntimeError("a189_optimization_forbidden")
    require(engine.runtime_versions() == config["runtime_versions"], "runtime_versions")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu_uuid"], "selected_gpu")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "cublas_workspace")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "device_order")
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "offline",
    )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    require(not torch.cuda.is_initialized(), "prior_cuda_context")
    require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"), "autocast"
    )
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "single_visible_gpu")
    require(torch.cuda.is_bf16_supported(including_emulation=False), "native_bf16_support")
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    path = Path(config["model_path"])
    original = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    require(getattr(original, "quantization_config", None) is None, "unquantized_checkpoint")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
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
        and all(str(p.device) == "cuda:0" and p.dtype == torch.bfloat16 for p in parameters),
        "parameter_device_dtype",
    )
    require(all(str(b.device) == "cuda:0" for b in model.buffers()), "buffer_device")
    require(
        all(not type(item).__module__.startswith("bitsandbytes") for item in model.modules()),
        "quantized_module",
    )
    require(model.config._attn_implementation == "sdpa" and model.training is False, "model_mode")
    require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "autocast_after",
    )
    torch.cuda.synchronize(0)
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
