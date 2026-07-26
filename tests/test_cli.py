from __future__ import annotations

import sys
from pathlib import Path

from lexical_prompt_study import cli


def test_factorial_plan_cli_validates_frozen_plan(
    monkeypatch, capsys
) -> None:
    argv = [
        "lexical-study",
        "validate-factorial-plan",
        "--plan",
        "plans/factorial_8b_v1.public.json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    output = capsys.readouterr().out
    assert '"status": "factorial_plan_valid"' in output
    assert '"study_id": "lexical-scaffold-8b-factorial-v1"' in output


def test_factorial_private_cli_is_local_only_and_wires_exact_paths(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured = {}
    tokenizer = object()

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return {"status": "safe"}

    import transformers

    import lexical_prompt_study.factorial_private as private

    monkeypatch.setattr(private, "build_factorial_private_plan", fake_builder)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda path, *, local_files_only: (
            tokenizer
            if path == "local-tokenizer" and local_files_only is True
            else None
        ),
    )
    argv = [
        "lexical-study",
        "build-factorial-private-plan",
        "--public-plan",
        str(tmp_path / "public.json"),
        "--material-source",
        str(tmp_path / "materials.private.json"),
        "--tokenizer-path",
        "local-tokenizer",
        "--tokenizer-revision",
        "pinned-revision",
        "--private-out",
        str(tmp_path / "factorial.private.json"),
        "--public-receipt",
        str(tmp_path / "receipt.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["tokenizer"] is tokenizer
    assert captured["tokenizer_revision"] == "pinned-revision"
    assert captured["private_output_path"] == tmp_path / "factorial.private.json"
    assert captured["public_receipt_path"] == tmp_path / "receipt.json"
    assert '"status": "safe"' in capsys.readouterr().out


def test_mechanism_cli_wires_artifact_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "safe"}

    import lexical_prompt_study.mechanism_runner as runner

    monkeypatch.setattr(runner, "run_mechanism_discovery", fake_runner)
    argv = [
        "lexical-study",
        "run-mechanism-discovery",
        "--private-plan",
        str(tmp_path / "private.json"),
        "--public-plan",
        str(tmp_path / "public.json"),
        "--artifacts",
        str(tmp_path / "artifacts.json"),
        "--generation-root",
        str(tmp_path / "generation"),
        "--model-path",
        str(tmp_path / "model"),
        "--lens-path",
        str(tmp_path / "lens.pt"),
        "--sae-path",
        str(tmp_path / "sae.pt"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "safe-run",
        "--max-behaviors",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["artifacts_manifest_path"] == tmp_path / "artifacts.json"
    assert captured["max_behaviors"] == 2
    assert '"status": "safe"' in capsys.readouterr().out


def test_followup_mechanism_cli_wires_both_partitions(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "safe"}

    import lexical_prompt_study.followup_mechanism_analysis as runner

    monkeypatch.setattr(runner, "run_followup_mechanism_analysis", fake_runner)
    argv = [
        "lexical-study",
        "run-followup-mechanism-analysis",
        "--public-plan",
        str(tmp_path / "public.json"),
        "--probe-plan",
        str(tmp_path / "probe.json"),
        "--discovery-root",
        str(tmp_path / "discovery"),
        "--calibration-root",
        str(tmp_path / "calibration"),
        "--model-path",
        str(tmp_path / "model"),
        "--lens-path",
        str(tmp_path / "lens.pt"),
        "--sae-path",
        str(tmp_path / "sae.pt"),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "safe-followup-mechanism",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert captured["discovery_root"] == tmp_path / "discovery"
    assert captured["calibration_root"] == tmp_path / "calibration"
    assert captured["probe_plan_path"] == tmp_path / "probe.json"
    assert captured["run_id"] == "safe-followup-mechanism"
    assert '"status": "safe"' in capsys.readouterr().out
