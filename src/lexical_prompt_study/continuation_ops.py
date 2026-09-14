"""Exact-owner, deadline-only RunPod guard for the continuation repair.

The API credential stays on the operator machine.  The provider-side
``terminateAfter`` requested at allocation is a separate backstop; this guard
does not assume that a client process will survive a host outage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def budget_limits(authority_id: str | None = None) -> tuple[float, float]:
    """Explicit user authority; absence preserves the historical A146 envelope."""
    if authority_id is None:
        return 15.0, 2.0
    if authority_id == "a148-user-70":
        return 70.0, 8.0
    raise ValueError("unknown continuation budget authority")


def epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must be timezone-aware")
    return parsed.timestamp()


def validate_lease(lease: dict[str, Any]) -> None:
    if lease.get("task_id") != "lexical-a142-censored-continuation-v1":
        raise ValueError("wrong task owner")
    if lease.get("volume_id") != "u85xfo0aue":
        raise ValueError("wrong retained volume")
    if not str(lease.get("pod_name", "")).startswith("lexical-continuation-"):
        raise ValueError("wrong pod name")
    pod_id = lease.get("pod_id")
    if not isinstance(pod_id, str) or not pod_id.isalnum():
        raise ValueError("invalid exact pod id")
    duration = epoch(lease["deadline_utc"]) - epoch(lease["created_utc"])
    rate = lease["upper_hourly_usd"]
    limit = lease["allocation_cap_usd"]
    reserve = lease.get("reserve_usd", 0)
    spent = lease.get("round_spent_upper_usd")
    pilot_spent = lease.get("pilot_spent_upper_usd")
    round_cap = lease.get("round_hard_cap_usd")
    authorized_round, authorized_pilot = budget_limits(lease.get("budget_authority_id"))
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (rate, limit, reserve, spent, pilot_spent, round_cap)
    ):
        raise ValueError("budget bounds must be finite numeric values")
    if reserve < 0 or spent < 0 or round_cap != authorized_round or spent + limit > round_cap:
        raise ValueError("allocation exceeds remaining round authorization")
    if pilot_spent < 0 or pilot_spent > spent:
        raise ValueError("pilot spending must be a non-negative part of round spending")
    if not 0 < duration <= 3600 * 12 or not 0 < rate <= 10 or not 0 < limit <= authorized_round:
        raise ValueError("invalid allocation bounds")
    if lease.get("scope") == "pilot" and pilot_spent + limit > authorized_pilot:
        raise ValueError("pilot cap exceeds remaining pilot authorization")
    if lease.get("scope") not in ("pilot", "full"):
        raise ValueError("invalid scope")
    if rate * duration / 3600 + reserve > limit:
        raise ValueError("allocation worst case exceeds cap")
    if lease.get("provider_terminate_after_requested") != lease["deadline_utc"]:
        raise ValueError("missing provider deadline request")


def validate_owner(lease: dict[str, Any], pod: dict[str, Any]) -> None:
    validate_lease(lease)
    if (pod.get("id"), pod.get("name"), pod.get("networkVolumeId")) != (
        lease["pod_id"],
        lease["pod_name"],
        lease["volume_id"],
    ):
        raise ValueError("exact pod ownership mismatch; no mutation permitted")


def credential(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            value = line.split("=", 1)[1].strip().strip("\"'")
            if value:
                return value
    raise ValueError("RunPod credential not found")


def api(key: str, method: str, path: str) -> Any:
    request = urllib.request.Request(
        "https://rest.runpod.io/v1/" + path,
        method=method,
        headers={"Authorization": "Bearer " + key, "User-Agent": "runpodctl"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read()
            return json.loads(body) if body else {"http_status": response.status}
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        # Never surface an HTTP body or credential-bearing request object.
        raise RuntimeError(f"RunPod {method} status {error.code}") from None


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.is_symlink():
        raise ValueError("guard event path must not be a symbolic link")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **event}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    args = parser.parse_args()
    lease = json.loads(args.lease.read_text())
    validate_lease(lease)
    key = credential(args.env_file)
    deadline = epoch(lease["deadline_utc"])
    append_event(args.events, {"event": "guard_started", "pod_id": lease["pod_id"],
                               "guard_pid": os.getpid(), "deadline_utc": lease["deadline_utc"]})
    last_heartbeat = 0.0
    while True:
        try:
            pod = api(key, "GET", "pods/" + lease["pod_id"])
            if pod is None:
                append_event(args.events, {"event": "pod_absent_confirmed"})
                return
            validate_owner(lease, pod)
            if time.monotonic() - last_heartbeat >= 45:
                append_event(args.events, {"event": "guard_heartbeat", "guard_pid": os.getpid(),
                                           "pod_id": lease["pod_id"]})
                last_heartbeat = time.monotonic()
            if time.time() >= deadline:
                result = api(key, "DELETE", "pods/" + lease["pod_id"])
                # API bodies can include provider metadata; log only the status,
                # then establish teardown by observing the exact pod's 404.
                status = result.get("http_status") if isinstance(result, dict) else None
                append_event(args.events, {"event": "deadline_delete", "http_status": status})
            else:
                time.sleep(min(15, max(0.1, deadline - time.time())))
        except ValueError:
            append_event(args.events, {"event": "ownership_mismatch_no_mutation"})
            raise SystemExit("ownership check failed; exact pod requires operator review") from None
        except Exception as error:
            append_event(args.events, {"event": "api_retry", "error_type": type(error).__name__})
            time.sleep(5)


if __name__ == "__main__":
    main()
