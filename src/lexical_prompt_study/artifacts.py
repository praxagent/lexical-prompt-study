from __future__ import annotations

import os
import re
from pathlib import Path

from huggingface_hub import HfApi

from .hashing import write_json_atomic
from .models import Artifact, ArtifactFile

MODEL_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
SAE_REVISION = "128ee921ecd1b8b3a87d776cbcc357c0855da134"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
EVALUATOR_REVISION = "bda705349d1144fa618770bea64d99ce54e3835b"
DATASET_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"
ATTACK_REVISION = "64960b783249d36f76a48a33103cc4b168332b9b"

LENS_PATH = (
    "llama3.3-70b-it/jlens/Salesforce-wikitext/"
    "Llama-3.3-70B-Instruct_jacobian_lens.pt"
)


def artifact_manifest() -> dict:
    artifacts = [
        Artifact(
            role="target_model",
            repository="meta-llama/Llama-3.3-70B-Instruct",
            revision=MODEL_REVISION,
            files=[
                ArtifactFile(
                    path="transformers safetensor shards + tokenizer/config allowlist",
                    size_bytes=141_124_823_968,
                    sha256=None,
                )
            ],
            license="Llama 3.3 Community License",
            source_url="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
            notes="Only Transformers safetensors, tokenizer, and config files; excludes duplicate original weights.",
        ),
        Artifact(
            role="tokenizer",
            repository="meta-llama/Llama-3.3-70B-Instruct",
            revision=MODEL_REVISION,
            files=[
                ArtifactFile(
                    path="tokenizer.json",
                    size_bytes=17_209_920,
                    git_blob_oid="1c1d8d5c9024994f1d3b00f9662b8dd89ca13cf2",
                ),
                ArtifactFile(
                    path="tokenizer_config.json",
                    size_bytes=55_425,
                    git_blob_oid="41cd8d1ca52c3fc5feaf8445ca922b27f8e8ea8b",
                ),
            ],
            license="Llama 3.3 Community License",
            source_url="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
        ),
        Artifact(
            role="jacobian_lens",
            repository="neuronpedia/jacobian-lens",
            revision=LENS_REVISION,
            files=[
                ArtifactFile(
                    path=LENS_PATH,
                    size_bytes=10_603_226_027,
                    git_blob_oid="96fa055abe78505fdd4fe8e945949310c87d228f",
                )
            ],
            license="MIT",
            source_url=f"https://huggingface.co/neuronpedia/jacobian-lens/blob/{LENS_REVISION}/{LENS_PATH}",
            expected_model="meta-llama/Llama-3.3-70B-Instruct",
        ),
        Artifact(
            role="sae",
            repository="Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
            revision=SAE_REVISION,
            files=[
                ArtifactFile(
                    path="Llama-3.3-70B-Instruct-SAE-l50.pt",
                    size_bytes=4_295_264_404,
                    git_blob_oid="7b300d6c13dd6f9a454e0facb859c2d883f68bce",
                )
            ],
            license="Llama 3.3 Community License",
            source_url="https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
            expected_model="meta-llama/Llama-3.3-70B-Instruct",
        ),
        Artifact(
            role="evaluator",
            repository="cais/HarmBench-Llama-2-13b-cls",
            revision=EVALUATOR_REVISION,
            files=[
                ArtifactFile(path="six safetensor shards + config/tokenizer", size_bytes=26_032_276_403)
            ],
            license="MIT",
            source_url="https://huggingface.co/cais/HarmBench-Llama-2-13b-cls",
        ),
        Artifact(
            role="dataset",
            repository="JailbreakBench/JBB-Behaviors",
            revision=DATASET_REVISION,
            files=[
                ArtifactFile(
                    path="data/harmful-behaviors.csv",
                    size_bytes=23_116,
                    git_blob_oid="5a7549cd9de9bb327914de0800606a1af1ec8849",
                ),
                ArtifactFile(
                    path="data/benign-behaviors.csv",
                    size_bytes=20_570,
                    git_blob_oid="659647d082ec98df45fdeb3f334ebd7c04e61638",
                ),
                ArtifactFile(
                    path="data/judge-comparison.csv",
                    size_bytes=363_132,
                    git_blob_oid="569575fe3a090dcfcf73d0a85f7f7198c4424c46",
                ),
            ],
            license="MIT",
            source_url="https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        ),
        Artifact(
            role="attack",
            repository="elder-plinius/L1B3RT4S",
            revision=ATTACK_REVISION,
            files=[
                ArtifactFile(
                    path="META.mkd",
                    size_bytes=3_834,
                    git_blob_oid="d5f87609877a7f24dc0b29d000dc97daf4406cbd",
                )
            ],
            license="GNU Affero General Public License v3.0",
            source_url=f"https://github.com/elder-plinius/L1B3RT4S/blob/{ATTACK_REVISION}/META.mkd",
        ),
    ]
    return {
        "schema_version": "1.0",
        "outcome_status": "outcome-free",
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
    }


def _resolved_files(repo_id: str, revision: str, repo_type: str, selected: set[str]) -> list[ArtifactFile]:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = (
        api.dataset_info(repo_id, revision=revision, files_metadata=True)
        if repo_type == "dataset"
        else api.model_info(repo_id, revision=revision, files_metadata=True)
    )
    if info.sha != revision:
        raise ValueError(f"{repo_id}: resolved {info.sha}, expected {revision}")
    siblings = {item.rfilename: item for item in info.siblings}
    missing = selected - set(siblings)
    if missing:
        raise ValueError(f"{repo_id}: missing files {sorted(missing)}")
    files = []
    for filename in sorted(selected):
        item = siblings[filename]
        lfs = getattr(item, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs else None
        files.append(
            ArtifactFile(
                path=filename,
                size_bytes=int(item.size or 0),
                git_blob_oid=item.blob_id,
                sha256=sha256,
            )
        )
    return files


def resolved_artifact_manifest() -> dict:
    manifest = artifact_manifest()
    artifacts = manifest["artifacts"]
    model_selected = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    model_selected.update(f"model-{index:05d}-of-00030.safetensors" for index in range(1, 31))
    artifact_by_role = {item["role"]: item for item in artifacts}
    model_files = _resolved_files(
        "meta-llama/Llama-3.3-70B-Instruct", MODEL_REVISION, "model", model_selected
    )
    artifact_by_role["target_model"]["files"] = [item.model_dump(mode="json") for item in model_files]
    artifact_by_role["tokenizer"]["files"] = [
        item.model_dump(mode="json")
        for item in model_files
        if re.search(r"tokenizer|special_tokens", item.path)
    ]
    resolutions = [
        (
            "jacobian_lens",
            "neuronpedia/jacobian-lens",
            LENS_REVISION,
            "model",
            {LENS_PATH},
        ),
        (
            "sae",
            "Goodfire/Llama-3.3-70B-Instruct-SAE-l50",
            SAE_REVISION,
            "model",
            {"Llama-3.3-70B-Instruct-SAE-l50.pt"},
        ),
        (
            "evaluator",
            "cais/HarmBench-Llama-2-13b-cls",
            EVALUATOR_REVISION,
            "model",
            {
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "special_tokens_map.json",
                "tokenizer.model",
                "tokenizer_config.json",
                *(f"model-{index:05d}-of-00006.safetensors" for index in range(1, 7)),
            },
        ),
        (
            "dataset",
            "JailbreakBench/JBB-Behaviors",
            DATASET_REVISION,
            "dataset",
            {
                "data/harmful-behaviors.csv",
                "data/benign-behaviors.csv",
                "data/judge-comparison.csv",
            },
        ),
    ]
    for role, repo, revision, repo_type, selected in resolutions:
        artifact_by_role[role]["files"] = [
            item.model_dump(mode="json")
            for item in _resolved_files(repo, revision, repo_type, selected)
        ]
    return manifest


def write_artifact_manifest(path: Path) -> str:
    return write_json_atomic(path, resolved_artifact_manifest())
