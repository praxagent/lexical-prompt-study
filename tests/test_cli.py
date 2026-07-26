from __future__ import annotations

import sys
from pathlib import Path

from lexical_prompt_study import cli


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
