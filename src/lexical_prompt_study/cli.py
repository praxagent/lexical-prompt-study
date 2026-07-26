from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_artifact_manifest
from .analyze import analyze_behavior_gate
from .behavior import run_behavior
from .evaluate import score_behavior_receipts
from .evaluator_validation import validate_published_judges
from .models import (
    FollowupQualificationReceipt,
    FollowupTrialReceipt,
    InterventionReceipt,
    MechanismReceipt,
    StudyPlan,
    TrialReceipt,
)
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
    validate_factorial = sub.add_parser("validate-factorial-plan")
    validate_factorial.add_argument(
        "--plan",
        type=Path,
        default=Path("plans/factorial_8b_v1.public.json"),
    )
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
    mechanism = sub.add_parser("run-mechanism-discovery")
    mechanism.add_argument("--private-plan", type=Path, required=True)
    mechanism.add_argument(
        "--public-plan", type=Path, default=Path("plans/study_v1.public.json")
    )
    mechanism.add_argument(
        "--artifacts", type=Path, default=Path("plans/artifacts.v1.json")
    )
    mechanism.add_argument("--generation-root", type=Path, required=True)
    mechanism.add_argument("--model-path", required=True)
    mechanism.add_argument("--lens-path", type=Path, required=True)
    mechanism.add_argument("--sae-path", type=Path, required=True)
    mechanism.add_argument("--out", type=Path, required=True)
    mechanism.add_argument("--run-id", required=True)
    mechanism.add_argument("--max-behaviors", type=int)
    analyze_mechanism = sub.add_parser("analyze-mechanisms")
    analyze_mechanism.add_argument("--input", type=Path, required=True)
    analyze_mechanism.add_argument("--out", type=Path, required=True)
    mechanism_figures = sub.add_parser("figures-mechanisms")
    mechanism_figures.add_argument("--analysis", type=Path, required=True)
    mechanism_figures.add_argument("--out", type=Path, required=True)
    verify_mechanism_figures = sub.add_parser("verify-mechanism-figures")
    verify_mechanism_figures.add_argument("--analysis", type=Path, required=True)
    verify_mechanism_figures.add_argument("--out", type=Path, required=True)
    replay_figures = sub.add_parser("figures-replay")
    replay_figures.add_argument("--result", type=Path, required=True)
    replay_figures.add_argument("--out", type=Path, required=True)
    verify_replay_figures = sub.add_parser("verify-replay-figures")
    verify_replay_figures.add_argument("--result", type=Path, required=True)
    verify_replay_figures.add_argument("--out", type=Path, required=True)
    validate_intervention = sub.add_parser("validate-intervention-plan")
    validate_intervention.add_argument("--plan", type=Path, required=True)
    validate_intervention.add_argument("--public", type=Path, required=True)
    validate_intervention.add_argument("--analysis", type=Path, required=True)
    validate_intervention.add_argument("--private", type=Path)
    intervention_calibration = sub.add_parser("run-intervention-calibration")
    intervention_calibration.add_argument("--private-plan", type=Path, required=True)
    intervention_calibration.add_argument("--public-plan", type=Path, required=True)
    intervention_calibration.add_argument(
        "--intervention-plan", type=Path, required=True
    )
    intervention_calibration.add_argument(
        "--gate3-analysis", type=Path, required=True
    )
    intervention_calibration.add_argument(
        "--generation-root", type=Path, required=True
    )
    intervention_calibration.add_argument("--model-path", required=True)
    intervention_calibration.add_argument("--sae-path", type=Path, required=True)
    intervention_calibration.add_argument("--out", type=Path, required=True)
    intervention_calibration.add_argument("--run-id", required=True)
    intervention_calibration.add_argument("--max-behaviors", type=int)
    intervention_calibration.add_argument("--max-rhos", type=int)
    intervention_calibration.add_argument("--max-new-tokens", type=int)
    analyze_calibration = sub.add_parser("analyze-intervention-calibration")
    analyze_calibration.add_argument("--intervention-plan", type=Path, required=True)
    analyze_calibration.add_argument("--public-plan", type=Path, required=True)
    analyze_calibration.add_argument("--gate3-analysis", type=Path, required=True)
    analyze_calibration.add_argument("--generation-root", type=Path, required=True)
    analyze_calibration.add_argument("--score-root", type=Path, required=True)
    analyze_calibration.add_argument("--out", type=Path, required=True)
    intervention_figures = sub.add_parser("figures-interventions")
    intervention_figures.add_argument("--analysis", type=Path, required=True)
    intervention_figures.add_argument("--plan", type=Path, required=True)
    intervention_figures.add_argument("--out", type=Path, required=True)
    verify_intervention_figures = sub.add_parser("verify-intervention-figures")
    verify_intervention_figures.add_argument("--analysis", type=Path, required=True)
    verify_intervention_figures.add_argument("--plan", type=Path, required=True)
    verify_intervention_figures.add_argument("--out", type=Path, required=True)
    followup_qualification = sub.add_parser("run-followup-qualification")
    followup_qualification.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_qualification.add_argument("--model-path", required=True)
    followup_qualification.add_argument("--lens-path", type=Path, required=True)
    followup_qualification.add_argument("--sae-path", type=Path, required=True)
    followup_qualification.add_argument("--out", type=Path, required=True)
    followup_qualification.add_argument("--run-id", required=True)
    followup_private = sub.add_parser("build-followup-private-plan")
    followup_private.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_private.add_argument(
        "--source-private",
        type=Path,
        default=Path("private/plans/study_v1.private.json"),
    )
    followup_private.add_argument(
        "--source-public",
        type=Path,
        default=Path("plans/study_v1.public.json"),
    )
    followup_private.add_argument(
        "--benign-csv",
        type=Path,
        default=Path("private/source/JBB-Behaviors/data/benign-behaviors.csv"),
    )
    followup_private.add_argument(
        "--out", type=Path, default=Path("private/plans/followup_v2.private.json")
    )
    followup_generation = sub.add_parser("run-followup-generation")
    followup_generation.add_argument("--private-plan", type=Path, required=True)
    followup_generation.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_generation.add_argument("--model-path", required=True)
    followup_generation.add_argument("--lens-path", type=Path, required=True)
    followup_generation.add_argument("--sae-path", type=Path, required=True)
    followup_generation.add_argument("--out", type=Path, required=True)
    followup_generation.add_argument(
        "--partition",
        choices=[
            "discovery",
            "calibration",
            "confirmatory",
            "adaptive_stress",
            "utility_calibration",
            "utility_confirmatory",
        ],
        required=True,
    )
    followup_generation.add_argument("--run-id", required=True)
    followup_behavior_analysis = sub.add_parser("analyze-followup-behavior")
    followup_behavior_analysis.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_behavior_analysis.add_argument(
        "--generation-root", type=Path, required=True
    )
    followup_behavior_analysis.add_argument("--score-root", type=Path, required=True)
    followup_behavior_analysis.add_argument("--out", type=Path, required=True)
    followup_behavior_analysis.add_argument(
        "--partition", choices=["discovery", "calibration"], required=True
    )
    followup_mechanism = sub.add_parser("run-followup-mechanism-analysis")
    followup_mechanism.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_mechanism.add_argument(
        "--probe-plan", type=Path, default=Path("plans/study_v1.public.json")
    )
    followup_mechanism.add_argument("--discovery-root", type=Path, required=True)
    followup_mechanism.add_argument("--calibration-root", type=Path, required=True)
    followup_mechanism.add_argument("--model-path", required=True)
    followup_mechanism.add_argument("--lens-path", type=Path, required=True)
    followup_mechanism.add_argument("--sae-path", type=Path, required=True)
    followup_mechanism.add_argument("--out", type=Path, required=True)
    followup_mechanism.add_argument("--run-id", required=True)
    followup_patch_private = sub.add_parser(
        "build-followup-patch-private-plan"
    )
    followup_patch_private.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_patch_private.add_argument("--tokenizer-path", required=True)
    followup_patch_private.add_argument("--out", type=Path, required=True)
    followup_coarse_patch = sub.add_parser("run-followup-coarse-patch")
    followup_coarse_patch.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_coarse_patch.add_argument("--patch-private-plan", type=Path, required=True)
    followup_coarse_patch.add_argument("--generation-root", type=Path, required=True)
    followup_coarse_patch.add_argument("--model-path", required=True)
    followup_coarse_patch.add_argument("--out", type=Path, required=True)
    followup_coarse_patch.add_argument(
        "--partition", choices=["discovery", "calibration"], required=True
    )
    followup_coarse_patch.add_argument("--run-id", required=True)
    followup_coarse_patch.add_argument("--selected-layer", type=int)
    followup_coarse_patch.add_argument("--qualification-only", action="store_true")
    followup_coarse_patch.add_argument(
        "--safe-positive-control-result", type=Path, required=True
    )
    followup_patch_analysis = sub.add_parser("analyze-followup-coarse-patch")
    followup_patch_analysis.add_argument(
        "--public-plan", type=Path, default=Path("plans/followup_v2.public.json")
    )
    followup_patch_analysis.add_argument("--patch-root", type=Path, required=True)
    followup_patch_analysis.add_argument("--patch-score-root", type=Path, required=True)
    followup_patch_analysis.add_argument(
        "--baseline-generation-root", type=Path, required=True
    )
    followup_patch_analysis.add_argument(
        "--baseline-score-root", type=Path, required=True
    )
    followup_patch_analysis.add_argument(
        "--partition", choices=["discovery", "calibration"], required=True
    )
    followup_patch_analysis.add_argument("--public-out", type=Path, required=True)
    followup_patch_analysis.add_argument("--private-out", type=Path, required=True)
    followup_patch_figures = sub.add_parser("figures-followup-patch")
    followup_patch_figures.add_argument("--result", type=Path, required=True)
    followup_patch_figures.add_argument("--plan", type=Path, required=True)
    followup_patch_figures.add_argument("--out", type=Path, required=True)
    verify_followup_patch_figures_parser = sub.add_parser(
        "verify-followup-patch-figures"
    )
    verify_followup_patch_figures_parser.add_argument(
        "--result", type=Path, required=True
    )
    verify_followup_patch_figures_parser.add_argument("--plan", type=Path, required=True)
    verify_followup_patch_figures_parser.add_argument("--out", type=Path, required=True)
    followup_mechanism_figures = sub.add_parser("figures-followup-mechanisms")
    followup_mechanism_figures.add_argument("--result", type=Path, required=True)
    followup_mechanism_figures.add_argument("--out", type=Path, required=True)
    verify_followup_mechanism_figures = sub.add_parser(
        "verify-followup-mechanism-figures"
    )
    verify_followup_mechanism_figures.add_argument("--result", type=Path, required=True)
    verify_followup_mechanism_figures.add_argument("--out", type=Path, required=True)
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
    elif args.command == "validate-factorial-plan":
        from .factorial_plan import load_factorial_plan, validate_factorial_plan

        plan = load_factorial_plan(args.plan)
        validate_factorial_plan(plan)
        print(
            json.dumps(
                {
                    "plan": str(args.plan),
                    "status": "factorial_plan_valid",
                    "study_id": plan["study_id"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "validate-published-judges":
        print(json.dumps(validate_published_judges(args.source, args.out), sort_keys=True))
    elif args.command == "write-schemas":
        outputs = {}
        for name, model in (
            ("study-plan", StudyPlan),
            ("trial-receipt", TrialReceipt),
            ("mechanism-receipt", MechanismReceipt),
            ("intervention-receipt", InterventionReceipt),
            ("followup-qualification-receipt", FollowupQualificationReceipt),
            ("followup-trial-receipt", FollowupTrialReceipt),
        ):
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
        from .figures import generate_behavior_figures

        print(json.dumps(generate_behavior_figures(args.gate, args.out), sort_keys=True))
    elif args.command == "verify-figures":
        from .figures import verify_behavior_figures

        print(json.dumps(verify_behavior_figures(args.gate, args.out), sort_keys=True))
    elif args.command == "run-mechanism-discovery":
        from .mechanism_runner import run_mechanism_discovery

        print(
            json.dumps(
                run_mechanism_discovery(
                    private_plan_path=args.private_plan,
                    public_plan_path=args.public_plan,
                    artifacts_manifest_path=args.artifacts,
                    generation_root=args.generation_root,
                    model_path=args.model_path,
                    lens_path=args.lens_path,
                    sae_path=args.sae_path,
                    output_root=args.out,
                    run_id=args.run_id,
                    max_behaviors=args.max_behaviors,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "analyze-mechanisms":
        from .analyze_mechanisms import analyze_mechanisms

        result = analyze_mechanisms(args.input, args.out)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "gate3_artifact_sha256": result["gate3_artifact_sha256"],
                    "output": str(args.out),
                },
                sort_keys=True,
            )
        )
    elif args.command == "figures-mechanisms":
        from .mechanism_figures import generate_mechanism_figures

        print(
            json.dumps(
                generate_mechanism_figures(args.analysis, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "verify-mechanism-figures":
        from .mechanism_figures import verify_mechanism_figures

        print(
            json.dumps(
                verify_mechanism_figures(args.analysis, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "figures-replay":
        from .replay_figures import generate_replay_figure

        print(
            json.dumps(
                generate_replay_figure(args.result, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "verify-replay-figures":
        from .replay_figures import verify_replay_figure

        print(
            json.dumps(
                verify_replay_figure(args.result, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "validate-intervention-plan":
        from .intervention_plan import validate_intervention_plan

        print(
            json.dumps(
                validate_intervention_plan(
                    args.plan,
                    args.public,
                    args.analysis,
                    private_study_path=args.private,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "run-intervention-calibration":
        from .intervention_runner import run_intervention_calibration

        print(
            json.dumps(
                run_intervention_calibration(
                    private_plan_path=args.private_plan,
                    public_plan_path=args.public_plan,
                    intervention_plan_path=args.intervention_plan,
                    gate3_analysis_path=args.gate3_analysis,
                    generation_root=args.generation_root,
                    model_path=args.model_path,
                    sae_path=args.sae_path,
                    output_root=args.out,
                    run_id=args.run_id,
                    max_behaviors=args.max_behaviors,
                    max_rhos=args.max_rhos,
                    max_new_tokens=args.max_new_tokens,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "analyze-intervention-calibration":
        from .intervention_analysis import analyze_alpha_calibration_receipts

        result = analyze_alpha_calibration_receipts(
            intervention_plan_path=args.intervention_plan,
            public_plan_path=args.public_plan,
            gate3_analysis_path=args.gate3_analysis,
            generation_root=args.generation_root,
            score_root=args.score_root,
            output_path=args.out,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "selection": result["selection"],
                    "output": str(args.out),
                },
                sort_keys=True,
            )
        )
    elif args.command == "figures-interventions":
        from .intervention_figures import generate_intervention_figures

        print(
            json.dumps(
                generate_intervention_figures(
                    args.analysis,
                    args.plan,
                    args.out,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "verify-intervention-figures":
        from .intervention_figures import verify_intervention_figures

        print(
            json.dumps(
                verify_intervention_figures(
                    args.analysis,
                    args.plan,
                    args.out,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "run-followup-qualification":
        from .followup_runner import run_followup_qualification

        print(
            json.dumps(
                run_followup_qualification(
                    public_plan_path=args.public_plan,
                    model_path=args.model_path,
                    lens_path=args.lens_path,
                    sae_path=args.sae_path,
                    output_root=args.out,
                    run_id=args.run_id,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "build-followup-private-plan":
        from .followup_private import build_followup_private_plan

        print(
            json.dumps(
                build_followup_private_plan(
                    public_plan_path=args.public_plan,
                    source_private_path=args.source_private,
                    source_public_path=args.source_public,
                    benign_csv_path=args.benign_csv,
                    output_path=args.out,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "run-followup-generation":
        from .followup_generation import run_followup_generation

        print(
            json.dumps(
                run_followup_generation(
                    private_plan_path=args.private_plan,
                    public_plan_path=args.public_plan,
                    model_path=args.model_path,
                    lens_path=args.lens_path,
                    sae_path=args.sae_path,
                    output_root=args.out,
                    partition=args.partition,
                    run_id=args.run_id,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "analyze-followup-behavior":
        from .followup_behavior_analysis import (
            analyze_followup_behavior_calibration,
            analyze_followup_behavior_discovery,
        )

        analyzer = (
            analyze_followup_behavior_discovery
            if args.partition == "discovery"
            else analyze_followup_behavior_calibration
        )
        print(
            json.dumps(
                analyzer(
                    public_plan_path=args.public_plan,
                    generation_root=args.generation_root,
                    score_root=args.score_root,
                    output_path=args.out,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "run-followup-mechanism-analysis":
        from .followup_mechanism_analysis import run_followup_mechanism_analysis

        print(
            json.dumps(
                run_followup_mechanism_analysis(
                    public_plan_path=args.public_plan,
                    probe_plan_path=args.probe_plan,
                    discovery_root=args.discovery_root,
                    calibration_root=args.calibration_root,
                    model_path=args.model_path,
                    lens_path=args.lens_path,
                    sae_path=args.sae_path,
                    output_root=args.out,
                    run_id=args.run_id,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "build-followup-patch-private-plan":
        from .followup_patch_private import build_followup_patch_private_plan

        print(
            json.dumps(
                build_followup_patch_private_plan(
                    public_plan_path=args.public_plan,
                    tokenizer_path=args.tokenizer_path,
                    output_path=args.out,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "run-followup-coarse-patch":
        from .followup_patch_runner import run_followup_coarse_patch_generation

        print(
            json.dumps(
                run_followup_coarse_patch_generation(
                    public_plan_path=args.public_plan,
                    patch_private_plan_path=args.patch_private_plan,
                    generation_root=args.generation_root,
                    model_path=args.model_path,
                    output_root=args.out,
                    partition=args.partition,
                    run_id=args.run_id,
                    selected_layer=args.selected_layer,
                    qualification_only=args.qualification_only,
                    safe_positive_control_result_path=args.safe_positive_control_result,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "analyze-followup-coarse-patch":
        from .followup_patch_analysis import analyze_followup_coarse_patch

        print(
            json.dumps(
                analyze_followup_coarse_patch(
                    public_plan_path=args.public_plan,
                    patch_root=args.patch_root,
                    patch_score_root=args.patch_score_root,
                    baseline_generation_root=args.baseline_generation_root,
                    baseline_score_root=args.baseline_score_root,
                    partition=args.partition,
                    public_output_path=args.public_out,
                    private_output_path=args.private_out,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "figures-followup-patch":
        from .followup_patch_figures import generate_followup_patch_figures

        print(
            json.dumps(
                generate_followup_patch_figures(args.result, args.plan, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "verify-followup-patch-figures":
        from .followup_patch_figures import verify_followup_patch_figures

        print(
            json.dumps(
                verify_followup_patch_figures(args.result, args.plan, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "figures-followup-mechanisms":
        from .followup_mechanism_figures import generate_followup_mechanism_figures

        print(
            json.dumps(
                generate_followup_mechanism_figures(args.result, args.out),
                sort_keys=True,
            )
        )
    elif args.command == "verify-followup-mechanism-figures":
        from .followup_mechanism_figures import verify_followup_mechanism_figures

        print(
            json.dumps(
                verify_followup_mechanism_figures(args.result, args.out),
                sort_keys=True,
            )
        )
