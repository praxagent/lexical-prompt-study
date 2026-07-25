from pathlib import Path

from lexical_prompt_study.synthetic import run_synthetic


def test_resume_after_forced_interruption(tmp_path: Path) -> None:
    first = run_synthetic(tmp_path, n=4, stop_after=5)
    assert first["status"] == "interrupted"
    second = run_synthetic(tmp_path, n=4)
    assert second == {"status": "complete", "expected": 16, "written_this_call": 11}
    third = run_synthetic(tmp_path, n=4)
    assert third == {"status": "complete", "expected": 16, "written_this_call": 0}
    assert len(list((tmp_path / "trials").glob("*.json"))) == 16


def test_corrupt_receipt_is_not_completed(tmp_path: Path) -> None:
    run_synthetic(tmp_path, n=1)
    receipt = next((tmp_path / "trials").glob("*.json"))
    receipt.write_text("{}")
    resumed = run_synthetic(tmp_path, n=1)
    assert resumed["written_this_call"] == 1
