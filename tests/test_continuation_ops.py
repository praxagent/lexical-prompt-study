from copy import deepcopy

import pytest

from lexical_prompt_study.continuation_ops import validate_lease, validate_owner


def lease():
    return {
        "task_id": "lexical-a142-censored-continuation-v1",
        "pod_id": "owned123",
        "pod_name": "lexical-continuation-pilot",
        "volume_id": "u85xfo0aue",
        "scope": "pilot",
        "created_utc": "2026-09-04T21:00:00Z",
        "deadline_utc": "2026-09-04T22:00:00Z",
        "provider_terminate_after_requested": "2026-09-04T22:00:00Z",
        "upper_hourly_usd": 1.61,
        "allocation_cap_usd": 2,
        "reserve_usd": 0.25,
        "round_spent_upper_usd": 0,
        "pilot_spent_upper_usd": 0,
        "round_hard_cap_usd": 15,
    }


def test_exact_owner_only():
    value = lease()
    validate_owner(
        value, {"id": "owned123", "name": value["pod_name"], "networkVolumeId": "u85xfo0aue"}
    )
    for field in ("id", "name", "networkVolumeId"):
        pod = {"id": "owned123", "name": value["pod_name"], "networkVolumeId": "u85xfo0aue"}
        pod[field] = "other-project"
        with pytest.raises(ValueError):
            validate_owner(value, pod)


@pytest.mark.parametrize(
    "field,value",
    [
        ("allocation_cap_usd", 2.01),
        ("upper_hourly_usd", 3),
        ("reserve_usd", 1),
        ("volume_id", "other"),
        ("pod_id", "../other"),
        ("pod_name", "other-project"),
        ("provider_terminate_after_requested", "2026-09-04T23:00:00Z"),
        ("deadline_utc", "2026-09-04T20:00:00Z"),
        ("reserve_usd", -100),
        ("reserve_usd", float("nan")),
        ("reserve_usd", float("inf")),
        ("upper_hourly_usd", True),
        ("round_spent_upper_usd", -1),
        ("round_spent_upper_usd", 14),
        ("round_hard_cap_usd", 16),
        ("round_spent_upper_usd", None),
        ("pilot_spent_upper_usd", None),
        ("pilot_spent_upper_usd", -1),
        ("pilot_spent_upper_usd", float("nan")),
        ("pilot_spent_upper_usd", float("inf")),
        ("pilot_spent_upper_usd", True),
        ("pilot_spent_upper_usd", 0.01),
    ],
)
def test_invalid_or_over_budget_lease_rejected(field, value):
    record = deepcopy(lease())
    record[field] = value
    with pytest.raises(ValueError):
        validate_lease(record)


def test_full_allocation_includes_already_spent_pilot():
    record = lease()
    record.update(
        scope="full", allocation_cap_usd=13, round_spent_upper_usd=2, pilot_spent_upper_usd=2
    )
    validate_lease(record)
    record["allocation_cap_usd"] = 13.01
    with pytest.raises(ValueError, match="remaining round"):
        validate_lease(record)


def test_reallocated_pilot_includes_prior_failed_pod_cost():
    record = lease()
    record.update(round_spent_upper_usd=0.1, pilot_spent_upper_usd=0.1, allocation_cap_usd=1.9)
    validate_lease(record)
    record["allocation_cap_usd"] = 1.91
    with pytest.raises(ValueError, match="remaining pilot"):
        validate_lease(record)


def test_full_allocation_still_rejects_pilot_spend_greater_than_round_spend():
    record = lease()
    record.update(
        scope="full", allocation_cap_usd=3, round_spent_upper_usd=1, pilot_spent_upper_usd=1.01
    )
    with pytest.raises(ValueError, match="part of round spending"):
        validate_lease(record)


def test_event_log_has_restricted_permissions(tmp_path):
    import json
    import stat

    from lexical_prompt_study.continuation_ops import append_event

    path = tmp_path / "private" / "events.jsonl"
    append_event(path, {"event": "synthetic"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert json.loads(path.read_text())["event"] == "synthetic"


def test_named_a148_authority_is_inclusive_and_bounded():
    record = lease()
    record.update(budget_authority_id="a148-user-70", round_hard_cap_usd=70,
                  round_spent_upper_usd=2, pilot_spent_upper_usd=2,
                  allocation_cap_usd=6)
    validate_lease(record)
    record["allocation_cap_usd"] = 6.01
    with pytest.raises(ValueError, match="remaining pilot"):
        validate_lease(record)
    record.update(scope="full", allocation_cap_usd=68)
    validate_lease(record)
    record["allocation_cap_usd"] = 68.01
    with pytest.raises(ValueError, match="remaining round"):
        validate_lease(record)


@pytest.mark.parametrize("authority,cap", [(None, 70), ("a148-user-70", 71),
                                         ("arbitrary", 70), ("a148-user-70", 15)])
def test_budget_authority_cannot_be_forged_by_changing_only_caps(authority, cap):
    record = lease()
    record.update(budget_authority_id=authority, round_hard_cap_usd=cap)
    with pytest.raises(ValueError):
        validate_lease(record)
