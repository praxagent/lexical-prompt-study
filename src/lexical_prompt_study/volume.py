from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .hashing import git_blob_oid_file, sha256_file, write_json_atomic


def _artifact(manifest: dict, role: str) -> dict:
    matches = [item for item in manifest["artifacts"] if item["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"expected one {role} artifact, got {len(matches)}")
    return matches[0]


def prepare_volume(manifest_path: Path, root: Path, include_evaluator: bool = True) -> dict:
    manifest = json.loads(manifest_path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    roles = ["target_model", "jacobian_lens", "sae"]
    if include_evaluator:
        roles.append("evaluator")
    for role in roles:
        artifact = _artifact(manifest, role)
        repo_dir = root / role
        files = [item["path"] for item in artifact["files"]]
        if len(files) == 1:
            downloaded = [
                Path(
                    hf_hub_download(
                        repo_id=artifact["repository"],
                        revision=artifact["revision"],
                        filename=files[0],
                        local_dir=repo_dir,
                        token=os.environ.get("HF_TOKEN"),
                    )
                )
            ]
        else:
            snapshot_download(
                repo_id=artifact["repository"],
                revision=artifact["revision"],
                allow_patterns=files,
                local_dir=repo_dir,
                token=os.environ.get("HF_TOKEN"),
            )
            downloaded = [repo_dir / filename for filename in files]
        expected = {item["path"]: item for item in artifact["files"]}
        for path in downloaded:
            relative = str(path.relative_to(repo_dir))
            if not path.is_file():
                raise ValueError(f"{role}: missing downloaded file {relative}")
            actual_sha256 = sha256_file(path)
            expected_sha256 = expected[relative].get("sha256")
            expected_git_blob_oid = expected[relative].get("git_blob_oid")
            actual_git_blob_oid = (
                git_blob_oid_file(path) if expected_sha256 is None else None
            )
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{role}/{relative}: sha256 {actual_sha256}, expected {expected_sha256}"
                )
            if (
                expected_sha256 is None
                and expected_git_blob_oid
                and actual_git_blob_oid != expected_git_blob_oid
            ):
                raise ValueError(
                    f"{role}/{relative}: git blob {actual_git_blob_oid}, "
                    f"expected {expected_git_blob_oid}"
                )
            verified = (
                actual_sha256 == expected_sha256
                if expected_sha256
                else actual_git_blob_oid == expected_git_blob_oid
            )
            receipts.append(
                {
                    "role": role,
                    "repository": artifact["repository"],
                    "revision": artifact["revision"],
                    "path": str(path),
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": actual_sha256,
                    "expected_sha256": expected_sha256,
                    "git_blob_oid": actual_git_blob_oid,
                    "expected_git_blob_oid": expected_git_blob_oid,
                    "verified": verified,
                }
            )
    result = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "root": str(root),
        "files": receipts,
        "total_bytes": sum(item["size_bytes"] for item in receipts),
        "software": {"python": sys.version, "platform": platform.platform()},
    }
    write_json_atomic(root / "download-receipt.json", result)
    return {
        "status": "verified",
        "file_count": len(receipts),
        "total_bytes": result["total_bytes"],
        "receipt": str(root / "download-receipt.json"),
    }
