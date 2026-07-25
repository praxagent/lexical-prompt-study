from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_artifact_manifest
from .plans import build_plan, validate_plan
from .synthetic import run_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(prog="lexical-study")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("write-artifacts")
    manifest.add_argument("--out", type=Path, default=Path("plans/artifacts.v1.json"))
    synthetic = sub.add_parser("synthetic")
    synthetic.add_argument("--out", type=Path, default=Path("local-runs/gate0"))
    synthetic.add_argument("--n", type=int, default=8)
    synthetic.add_argument("--stop-after", type=int)
    build = sub.add_parser("build-plan")
    build.add_argument("--public", type=Path, default=Path("plans/study_v1.public.json"))
    build.add_argument("--private-root", type=Path, default=Path("private"))
    build.add_argument("--artifacts", type=Path, default=Path("plans/artifacts.v1.json"))
    validate = sub.add_parser("validate-plan")
    validate.add_argument("--public", type=Path, default=Path("plans/study_v1.public.json"))
    validate.add_argument("--private", type=Path, default=Path("private/plans/study_v1.private.json"))
    validate.add_argument("--artifacts", type=Path, default=Path("plans/artifacts.v1.json"))
    args = parser.parse_args()
    if args.command == "write-artifacts":
        digest = write_artifact_manifest(args.out)
        print(json.dumps({"path": str(args.out), "sha256": digest}))
    elif args.command == "synthetic":
        print(json.dumps(run_synthetic(args.out, args.n, args.stop_after), sort_keys=True))
    elif args.command == "build-plan":
        print(json.dumps(build_plan(args.public, args.private_root, args.artifacts), sort_keys=True))
    elif args.command == "validate-plan":
        print(json.dumps(validate_plan(args.public, args.private, args.artifacts), sort_keys=True))
