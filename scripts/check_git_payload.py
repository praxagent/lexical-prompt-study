"""Audit staged files or every changed blob in outgoing Git history.

This is a guardrail, not proof that arbitrary text is safe to publish. Review the
diff and research release scope as well. Findings never print file contents.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import PurePosixPath

MAX_BYTES = 5 * 1024 * 1024
PRIVATE_DIRS = {
    "private", "raw", "restricted", "datasets", "outputs", "runs", "checkpoints",
    "models", ".cache", ".venv", "local-runs", "lexical-prompt-study-data",
}
BULK_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".gguf", ".bin", ".npy", ".npz", ".parquet",
    ".arrow", ".h5", ".hdf5", ".jsonl", ".zip", ".tar", ".gz", ".tgz", ".7z",
    ".pem", ".key", ".p12", ".log",
}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    re.compile(rb"\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}|hf_[A-Za-z0-9]{30,})\b"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "--literal-pathspecs", *args])


def path_reason(path: str) -> str | None:
    item = PurePosixPath(path)
    if any(part.lower() in PRIVATE_DIRS for part in item.parts):
        return "private or runtime-data directory"
    name = item.name.lower()
    if name != ".env.example" and (name == ".env" or name.startswith(".env.")):
        return "environment/credential file"
    if name == "checkpoint.md":
        return "private continuity record"
    if item.suffix.lower() in BULK_SUFFIXES:
        return "bulk data, archive, log, or credential format"
    if re.search(r"\.(?:sqlite\w*|db|bak)(?:$|[.\-\d])", name):
        return "database or backup file"
    return None


def staged_entries() -> list[tuple[str, str, str]]:
    if git("ls-files", "--unmerged", "-z"):
        raise ValueError("Unmerged index entries cannot be audited")
    paths = git("diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z")
    entries = []
    for raw_path in paths.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", "surrogateescape")
        for row in git("ls-files", "--stage", "-z", "--", path).split(b"\0"):
            if not row:
                continue
            metadata, _ = row.split(b"\t", 1)
            mode, oid, stage = metadata.decode().split()
            if stage != "0":
                raise ValueError("Unmerged index entries cannot be audited")
            entries.append((mode, oid, path))
    return entries


def outgoing_entries(base: str) -> list[tuple[str, str, str]]:
    base_oid = git("rev-parse", "--verify", base + "^{commit}").decode().strip()
    commits = git("rev-list", "--reverse", base_oid + "..HEAD").decode().splitlines()
    entries = set()
    for commit in commits:
        paths = git(
            "diff-tree", "--no-commit-id", "--root", "-r", "-m", "--name-only",
            "--diff-filter=ACMRT", "-z", commit,
        )
        for raw_path in paths.split(b"\0"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8", "surrogateescape")
            for row in git("ls-tree", "-z", commit, "--", path).split(b"\0"):
                if not row:
                    continue
                metadata, _ = row.split(b"\t", 1)
                mode, _, oid = metadata.decode().split()
                entries.add((mode, oid, path))
    return sorted(entries)


def audit(entries: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    findings = []
    for mode, oid, path in entries:
        reason = path_reason(path)
        if reason:
            findings.append((path, reason))
            continue
        if mode not in {"100644", "100755"}:
            findings.append((path, "symlink or non-regular Git entry requires separate review"))
            continue
        size = int(git("cat-file", "-s", oid))
        if size > MAX_BYTES:
            findings.append((path, "exceeds 5 MiB publication limit"))
            continue
        content = git("cat-file", "blob", oid)
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            findings.append((path, "possible credential or private key"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--staged", action="store_true")
    group.add_argument("--base", help="Audit changed blobs in every commit between base and HEAD")
    args = parser.parse_args()
    entries = staged_entries() if args.staged else outgoing_entries(args.base)
    findings = audit(entries)
    for path, reason in findings:
        print(f"BLOCK {path!r}: {reason}")
    print(f"Audited {len(entries)} file versions; {len(findings)} blocked.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
