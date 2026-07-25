from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_artifact_manifest
from .analyze import analyze_behavior_gate
from .behavior import run_behavior
from .evaluate import score_behavior_receipts
from .figures import generate_behavior_figures, verify_behavior_figures
from .evaluator_validation import validate_published_judges
from .models import StudyPlan, TrialReceipt
from .plans import build_plan, validate_plan
from .hashing import write_json_atomic
from .synthetic import build_engineering_fixture, run_synthetic
from .volume import prepare_volume


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
    judges = sub.add_parser("validate-published-judges")
    judges.add_argument(
        "--source",
        type=Path,
        default=Path("private/source/JBB-Behaviors/data/judge-comparison.csv"),
    )
    judges.add_argument(
        "--out", type=Path, default=Path("validation/evaluator_validation.v1.json")
    )
    schemas = sub.add_parser("write-schemas")
    schemas.add_argument("--out", type=Path, default=Path("schemas"))
    volume = sub.add_parser("prepare-volume")
    volume.add_argument("--manifest", type=Path, default=Path("plans/artifacts.v1.json"))
    volume.add_argument("--root", type=Path, default=Path("/workspace/artifacts"))
    volume.add_argument("--skip-evaluator", action="store_true")
    behavior = sub.add_parser("run-behavior")
    behavior.add_argument("--private-plan", type=Path, required=True)
    behavior.add_argument("--public-plan", type=Path, default=Path("plans/study_v1.public.json"))
    behavior.add_argument("--model-path", required=True)
    behavior.add_argument("--out", type=Path, required=True)
    behavior.add_argument("--split", choices=["discovery", "confirmatory"], required=True)
    behavior.add_argument("--max-behaviors", type=int)
    behavior.add_argument("--max-new-tokens", type=int)
    behavior.add_argument("--run-id", required=True)
    fixture = sub.add_parser("build-engineering-fixture")
    fixture.add_argument("--public-plan", type=Path, default=Path("plans/study_v1.public.json"))
    fixture.add_argument(
        "--out", type=Path, default=Path("private/fixtures/engineering.private.json")
    )
    fixture.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    score = sub.add_parser("score-behavior")
    score.add_argument("--private-plan", type=Path, required=True)
    score.add_argument("--generation-root", type=Path, required=True)
    score.add_argument("--evaluator-path", required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=4)
    analyze = sub.add_parser("analyze-gate")
    analyze.add_argument("--public-plan", type=Path, default=Path("plans/study_v1.public.json"))
    analyze.add_argument("--generation-root", type=Path, required=True)
    analyze.add_argument("--score-root", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--split", choices=["discovery", "confirmatory"], required=True)
    figures = sub.add_parser("figures-behavior")
    figures.add_argument("--gate", type=Path, required=True)
    figures.add_argument("--out", type=Path, required=True)
    verify_figures = sub.add_parser("verify-figures")
    verify_figures.add_argument("--gate", type=Path, required=True)
    verify_figures.add_argument("--out", type=Path, required=True)
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
    elif args.command == "validate-published-judges":
        print(json.dumps(validate_published_judges(args.source, args.out), sort_keys=True))
    elif args.command == "write-schemas":
        outputs = {}
        for name, model in (("study-plan", StudyPlan), ("trial-receipt", TrialReceipt)):
            path = args.out / f"{name}.schema.json"
            outputs[name] = write_json_atomic(path, model.model_json_schema())
        print(json.dumps(outputs, sort_keys=True))
    elif args.command == "prepare-volume":
        print(
            json.dumps(
                prepare_volume(args.manifest, args.root, not args.skip_evaluator),
                sort_keys=True,
            )
        )
    elif args.command == "run-behavior":
        print(
            json.dumps(
                run_behavior(
                    private_plan_path=args.private_plan,
                    public_plan_path=args.public_plan,
                    model_path=args.model_path,
                    output_root=args.out,
                    split=args.split,
                    max_behaviors=args.max_behaviors,
                    max_new_tokens=args.max_new_tokens,
                    run_id=args.run_id,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "build-engineering-fixture":
        print(
            json.dumps(
                build_engineering_fixture(args.public_plan, args.out, args.tokenizer),
                sort_keys=True,
            )
        )
    elif args.command == "score-behavior":
        print(
            json.dumps(
                score_behavior_receipts(
                    private_plan_path=args.private_plan,
                    generation_root=args.generation_root,
                    evaluator_path=args.evaluator_path,
                    output_root=args.out,
                    batch_size=args.batch_size,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "analyze-gate":
        print(
            json.dumps(
                analyze_behavior_gate(
                    public_plan_path=args.public_plan,
                    generation_root=args.generation_root,
                    score_root=args.score_root,
                    output_path=args.out,
                    split=args.split,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "figures-behavior":
        print(json.dumps(generate_behavior_figures(args.gate, args.out), sort_keys=True))
    elif args.command == "verify-figures":
        print(json.dumps(verify_behavior_figures(args.gate, args.out), sort_keys=True))
