from __future__ import annotations

import hashlib
import json

import pytest

from lexical_prompt_study.breaker import (
    BreakerMode,
    BreakerThresholds,
    Placement,
    RestrictedExactMatcher,
    TwoKeyCircuitBreaker,
)


def _thresholds() -> dict[Placement, BreakerThresholds]:
    return {
        Placement.BEFORE_REQUEST: BreakerThresholds(structure=0.5, weaponization=0.7),
        Placement.AFTER_REQUEST: BreakerThresholds(structure=0.6, weaponization=0.8),
    }


def test_restricted_matcher_receipt_contains_no_plaintext() -> None:
    pattern = b"synthetic restricted scaffold"
    prompt = b"prefix synthetic restricted scaffold suffix"
    matcher = RestrictedExactMatcher([pattern])
    receipt = matcher.inspect(prompt)
    serialized = json.dumps(receipt.__dict__, sort_keys=True)

    assert receipt.matched is True
    assert receipt.matched_pattern_sha256 == hashlib.sha256(pattern).hexdigest()
    assert receipt.prompt_sha256 == hashlib.sha256(prompt).hexdigest()
    assert pattern.decode() not in serialized
    assert prompt.decode() not in serialized
    assert pattern.decode() not in repr(matcher)


def test_restricted_matcher_rejects_hash_drift(tmp_path) -> None:
    source = tmp_path / "patterns.json"
    source.write_text(
        json.dumps({"patterns": [{"text": "safe synthetic", "text_sha256": "0" * 64}]})
    )
    with pytest.raises(ValueError, match="hash drift"):
        RestrictedExactMatcher.from_private_json(source)


def test_shadow_two_key_records_but_does_not_cancel() -> None:
    breaker = TwoKeyCircuitBreaker(mode=BreakerMode.SHADOW, thresholds=_thresholds())
    decision = breaker.evaluate_prefill(
        placement=Placement.BEFORE_REQUEST,
        prompt_sha256="1" * 64,
        exact_match=False,
        structure_score=0.9,
        weaponization_score=0.9,
    )
    assert decision.would_cancel is True
    assert decision.cancel_request is False
    assert decision.reason_codes == ("learned_two_key",)
    assert decision.released_tokens_before_break == 0


def test_active_canary_requires_both_learned_keys() -> None:
    breaker = TwoKeyCircuitBreaker(mode=BreakerMode.ACTIVE_CANARY, thresholds=_thresholds())
    harmless_structure = breaker.evaluate_prefill(
        placement=Placement.BEFORE_REQUEST,
        prompt_sha256="2" * 64,
        exact_match=False,
        structure_score=0.9,
        weaponization_score=0.2,
    )
    harmful_without_structure = breaker.evaluate_prefill(
        placement=Placement.BEFORE_REQUEST,
        prompt_sha256="3" * 64,
        exact_match=False,
        structure_score=0.2,
        weaponization_score=0.9,
    )
    weaponized = breaker.evaluate_prefill(
        placement=Placement.BEFORE_REQUEST,
        prompt_sha256="4" * 64,
        exact_match=False,
        structure_score=0.9,
        weaponization_score=0.9,
    )
    assert harmless_structure.cancel_request is False
    assert harmful_without_structure.cancel_request is False
    assert weaponized.cancel_request is True


def test_exact_match_still_requires_weaponization_key() -> None:
    breaker = TwoKeyCircuitBreaker(mode=BreakerMode.ACTIVE_CANARY, thresholds=_thresholds())
    benign = breaker.evaluate_prefill(
        placement=Placement.AFTER_REQUEST,
        prompt_sha256="5" * 64,
        exact_match=True,
        structure_score=0.0,
        weaponization_score=0.2,
    )
    harmful = breaker.evaluate_prefill(
        placement=Placement.AFTER_REQUEST,
        prompt_sha256="6" * 64,
        exact_match=True,
        structure_score=0.0,
        weaponization_score=0.9,
    )
    assert benign.cancel_request is False
    assert harmful.cancel_request is True
    assert harmful.reason_codes == ("restricted_exact_two_key",)


def test_mid_generation_fallback_waits_for_divider() -> None:
    breaker = TwoKeyCircuitBreaker(mode=BreakerMode.ACTIVE_CANARY, thresholds=_thresholds())
    before = breaker.evaluate_mid_generation(
        placement=Placement.BEFORE_REQUEST,
        prompt_sha256="7" * 64,
        exact_match=False,
        structure_score=0.0,
        weaponization_score=0.9,
        divider_complete=False,
        released_tokens_before_break=12,
    )
    after = breaker.evaluate_mid_generation(
        placement=Placement.BEFORE_REQUEST,
        prompt_sha256="7" * 64,
        exact_match=False,
        structure_score=0.0,
        weaponization_score=0.9,
        divider_complete=True,
        released_tokens_before_break=12,
    )
    assert before.cancel_request is False
    assert after.cancel_request is True
    assert after.reason_codes == ("post_divider_fallback",)


def test_breaker_requires_thresholds_for_both_placements() -> None:
    with pytest.raises(ValueError, match="each placement"):
        TwoKeyCircuitBreaker(
            mode=BreakerMode.SHADOW,
            thresholds={Placement.BEFORE_REQUEST: BreakerThresholds(0.5, 0.7)},
        )
