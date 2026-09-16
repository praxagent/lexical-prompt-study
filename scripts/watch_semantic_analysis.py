"""Wait for one exact pipeline identity, then perform one frozen aggregate readout.

This script never launches fitting or encoding. It only reads process metadata
while waiting. A durable attempt marker prevents repeating an interrupted
analysis invocation. Completion receipts contain an exit code and output hash.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PIPELINE_PID = 3019310
PIPELINE_START_TICKS = 18913423
PROJECT_PYTHON = Path("/data2/PRAX/lexical-prompt-study/.venv/bin/python")
RUN_ROOT = Path("/data2/PRAX/lexical-prompt-study-data/runs/text-semantic/run-002")
LEXICAL_MANIFEST = Path("/data2/PRAX/lexical-prompt-study-data/runs/text-challenger/run-001/full-matrix/manifest.json")
LEXICAL_MANIFEST_SHA = "ebac675bca76d70509a09a011e7a79b83883a47d4047b93320bdd3a8abc25be3"
SEMANTIC_LAUNCH = RUN_ROOT / "matrix/launch.json"
SEMANTIC_LAUNCH_SHA = "fe87ec0a3820e753e2a46924d625aa255453a1632c4c6eedadeec31c4160f6d1"
OUTPUT = RUN_ROOT / "MATRIX-ANALYSIS.json"
SOURCE_PINS = {
    "lexical_prompt_study/__init__.py": "38df51bcc0f0179554da19316a656daaa4ba07dd998a704399b29e3085bce948",
    "lexical_prompt_study/text_challenger_analysis.py": "335d4bd8dc4c2afc7377cd320d0da05c16e850ff6246cebdedbb0cceb86d239b",
    "lexical_prompt_study/text_challenger_semantic_analysis.py": "1af547a9feb95743898e8532c609ff395e360f6b6cb08912de81cb0156b2b3a4",
}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def require(value):
    if not value:
        raise ValueError("watcher_integrity_check_failed")


def sha_file(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file())
    require(path.stat().st_size <= 16 * 1024**2)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable(path, value):
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(raw):
    """Linux stat fields: command may itself contain spaces or parentheses."""
    fields = raw.rsplit(")", 1)[1].split()
    require(len(fields) >= 20 and len(fields[0]) == 1)
    ticks = int(fields[19])
    require(ticks >= 0)
    return fields[0], ticks


def original_process_finished(*, proc_root=Path("/proc")):
    try:
        raw = (proc_root / str(PIPELINE_PID) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    state, ticks = process_identity(raw)
    return ticks != PIPELINE_START_TICKS or state in ("Z", "X", "x")


def wait_for_pipeline(seconds, *, clock=time.monotonic, sleep=time.sleep, check=original_process_finished):
    deadline = clock() + seconds
    while True:
        if check():
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(15.0, remaining))


def verify_sources(source_root):
    source_root = source_root.absolute()
    require(source_root.is_dir() and not any(p.is_symlink() for p in (source_root, *source_root.parents)))
    require({str(p.relative_to(source_root)) for p in source_root.rglob("*.py")} == set(SOURCE_PINS))
    require(not any(source_root.rglob("*.pyc")))
    for relative, digest in SOURCE_PINS.items():
        require(sha_file(source_root / relative) == digest)
    return source_root


def command(source_root):
    # source_root is bound separately and supplied through PYTHONPATH below.
    require(source_root.is_absolute())
    return [str(PROJECT_PYTHON), "-B", "-m", "lexical_prompt_study.text_challenger_semantic_analysis",
        "--lexical-execution-manifest", str(LEXICAL_MANIFEST), "--lexical-manifest-sha256", LEXICAL_MANIFEST_SHA,
        "--semantic-launch", str(SEMANTIC_LAUNCH), "--semantic-launch-sha256", SEMANTIC_LAUNCH_SHA,
        "--encoder-config", str(RUN_ROOT / "encoder-config.json"), "--output", str(OUTPUT)]


def terminate_owned_group(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def run_analysis(args, source_root):
    require(sha_file(LEXICAL_MANIFEST) == LEXICAL_MANIFEST_SHA)
    require(sha_file(SEMANTIC_LAUNCH) == SEMANTIC_LAUNCH_SHA)
    verify_sources(source_root)
    require(not OUTPUT.exists())
    invocation = command(source_root)
    immutable(RUN_ROOT / "analysis-watch.analysis-attempt.json", {
        "pipeline_pid": PIPELINE_PID, "pipeline_start_ticks": PIPELINE_START_TICKS,
        "command_sha256": hashlib.sha256(canonical(invocation)).hexdigest(),
        "source_root": str(source_root), "source_pins": SOURCE_PINS,
        "analysis_hard_limit_seconds": args.analysis_seconds,
        "watcher_source_sha256": sha_file(__file__),
    })
    env = dict(os.environ)
    env.update(PYTHONPATH=str(source_root), PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONHOME", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        env[name] = "4"
    out_fd = os.open(RUN_ROOT / "analysis-watch.stdout.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    err_fd = os.open(RUN_ROOT / "analysis-watch.stderr.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(out_fd, "wb") as stdout, os.fdopen(err_fd, "wb") as stderr:
        process = subprocess.Popen(invocation, cwd=source_root, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            # Reserve five seconds inside the analysis limit for graceful termination.
            process.wait(timeout=args.analysis_seconds - 5)
            return process.returncode
        except subprocess.TimeoutExpired:
            terminate_owned_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1)
            return 124
        except BaseException:
            terminate_owned_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=29700)
    parser.add_argument("--analysis-seconds", type=int, default=290)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        require(1 <= args.wait_seconds <= 30000 and 6 <= args.analysis_seconds <= 295)
        require(args.wait_seconds + args.analysis_seconds + 1 <= 30000)
        source_root = verify_sources(args.source_root)
        require(PROJECT_PYTHON.is_file())
        if args.check_only:
            print('{"status":"source_snapshot_verified_no_wait_or_analysis"}')
            return 0
        require(RUN_ROOT.is_dir() and not any(p.is_symlink() for p in (RUN_ROOT, *RUN_ROOT.parents)))
        fd = os.open(RUN_ROOT / "analysis-watch.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('{"status":"watcher_already_owned"}')
                return 3
            completion = RUN_ROOT / "analysis-watch.completion.json"
            require(not completion.exists() and not (RUN_ROOT / "analysis-watch.analysis-attempt.json").exists()
                    and not OUTPUT.exists())
            code = 2
            try:
                if wait_for_pipeline(args.wait_seconds):
                    code = run_analysis(args, source_root)
                else:
                    code = 124
            except (ValueError, OSError, subprocess.SubprocessError, IndexError):
                code = 2
            if code == 0 and not OUTPUT.is_file():
                code = 2
            output_sha = sha_file(OUTPUT) if OUTPUT.exists() else None
            immutable(completion, {"returncode": code, "output_sha256": output_sha})
            print(canonical({"returncode": code, "output_sha256": output_sha}).decode().strip())
            return 0 if code == 0 else 1
    except (ValueError, OSError, IndexError):
        print('{"status":"watcher_preflight_rejected"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
