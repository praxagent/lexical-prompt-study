from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping


class BreakerMode(StrEnum):
    OFFLINE_REPLAY = "offline_replay"
    SHADOW = "non_enforcing_shadow"
    ACTIVE_CANARY = "local_active_canary"


class Placement(StrEnum):
    BEFORE_REQUEST = "scaffold_before_request"
    AFTER_REQUEST = "scaffold_after_request"


@dataclass(frozen=True)
class ExactMatchReceipt:
    matched: bool
    matched_pattern_sha256: str | None
    match_start: int | None
    match_end: int | None
    prompt_sha256: str
    prompt_bytes: int


@dataclass(frozen=True)
class BreakerThresholds:
    structure: float
    weaponization: float

    def __post_init__(self) -> None:
        for name, value in (("structure", self.structure), ("weaponization", self.weaponization)):
            if not isinstance(value, (int, float)) or value != value:
                raise ValueError(f"{name} threshold must be finite")


@dataclass(frozen=True)
class BreakerDecision:
    mode: BreakerMode
    placement: Placement
    stage: str
    would_cancel: bool
    cancel_request: bool
    reason_codes: tuple[str, ...]
    exact_match: bool
    structure_key: bool
    weaponization_key: bool
    prompt_sha256: str
    released_tokens_before_break: int

    def as_public_receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "placement": self.placement.value,
            "stage": self.stage,
            "would_cancel": self.would_cancel,
            "cancel_request": self.cancel_request,
            "reason_codes": list(self.reason_codes),
            "exact_match": self.exact_match,
            "structure_key": self.structure_key,
            "weaponization_key": self.weaponization_key,
            "prompt_sha256": self.prompt_sha256,
            "released_tokens_before_break": self.released_tokens_before_break,
        }


class RestrictedExactMatcher:
    """Match private byte strings without returning or logging their contents."""

    def __init__(self, patterns: Iterable[bytes]) -> None:
        normalized = tuple(bytes(pattern) for pattern in patterns)
        if not normalized or any(not pattern for pattern in normalized):
            raise ValueError("at least one non-empty restricted pattern is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("restricted patterns must be unique")
        self._patterns = normalized
        self._hashes = tuple(hashlib.sha256(pattern).hexdigest() for pattern in normalized)

    def __repr__(self) -> str:
        return f"RestrictedExactMatcher(pattern_count={len(self._patterns)})"

    @classmethod
    def from_private_json(cls, path: Path) -> RestrictedExactMatcher:
        payload = json.loads(path.read_text())
        rows = payload["patterns"]
        patterns: list[bytes] = []
        for row in rows:
            text = str(row["text"])
            raw = text.encode("utf-8")
            expected = str(row["text_sha256"])
            actual = hashlib.sha256(raw).hexdigest()
            if not hmac.compare_digest(actual, expected):
                raise ValueError("restricted pattern hash drift")
            patterns.append(raw)
        return cls(patterns)

    @property
    def pattern_hashes(self) -> tuple[str, ...]:
        return self._hashes

    def inspect(self, prompt: str | bytes) -> ExactMatchReceipt:
        raw = prompt.encode("utf-8") if isinstance(prompt, str) else bytes(prompt)
        prompt_hash = hashlib.sha256(raw).hexdigest()
        matches: list[tuple[int, int, int]] = []
        for index, pattern in enumerate(self._patterns):
            start = raw.find(pattern)
            if start >= 0:
                matches.append((start, -(len(pattern)), index))
        if not matches:
            return ExactMatchReceipt(
                matched=False,
                matched_pattern_sha256=None,
                match_start=None,
                match_end=None,
                prompt_sha256=prompt_hash,
                prompt_bytes=len(raw),
            )
        start, negative_length, index = min(matches)
        length = -negative_length
        return ExactMatchReceipt(
            matched=True,
            matched_pattern_sha256=self._hashes[index],
            match_start=start,
            match_end=start + length,
            prompt_sha256=prompt_hash,
            prompt_bytes=len(raw),
        )


class TwoKeyCircuitBreaker:
    """Policy breaker that never emits raw prompt or generation content."""

    def __init__(
        self,
        *,
        mode: BreakerMode,
        thresholds: Mapping[Placement, BreakerThresholds],
        exact_match_requires_weaponization_key: bool = True,
    ) -> None:
        if set(thresholds) != set(Placement):
            raise ValueError("one threshold pair is required for each placement")
        self.mode = mode
        self.thresholds = dict(thresholds)
        self.exact_match_requires_weaponization_key = exact_match_requires_weaponization_key

    def _decision(
        self,
        *,
        placement: Placement,
        stage: str,
        prompt_sha256: str,
        exact_match: bool,
        structure_score: float,
        weaponization_score: float,
        released_tokens_before_break: int,
        divider_complete: bool,
    ) -> BreakerDecision:
        if released_tokens_before_break < 0:
            raise ValueError("released token count cannot be negative")
        threshold = self.thresholds[placement]
        structure_key = structure_score >= threshold.structure
        weaponization_key = weaponization_score >= threshold.weaponization
        learned_two_key = structure_key and weaponization_key
        exact_two_key = exact_match and (
            weaponization_key or not self.exact_match_requires_weaponization_key
        )
        mid_generation_key = stage == "mid_generation" and divider_complete and weaponization_key
        would_cancel = learned_two_key or exact_two_key or mid_generation_key
        reasons: list[str] = []
        if learned_two_key:
            reasons.append("learned_two_key")
        if exact_two_key:
            reasons.append("restricted_exact_two_key")
        if mid_generation_key:
            reasons.append("post_divider_fallback")
        return BreakerDecision(
            mode=self.mode,
            placement=placement,
            stage=stage,
            would_cancel=would_cancel,
            cancel_request=would_cancel and self.mode is BreakerMode.ACTIVE_CANARY,
            reason_codes=tuple(reasons),
            exact_match=exact_match,
            structure_key=structure_key,
            weaponization_key=weaponization_key,
            prompt_sha256=prompt_sha256,
            released_tokens_before_break=released_tokens_before_break,
        )

    def evaluate_prefill(
        self,
        *,
        placement: Placement,
        prompt_sha256: str,
        exact_match: bool,
        structure_score: float,
        weaponization_score: float,
    ) -> BreakerDecision:
        return self._decision(
            placement=placement,
            stage="prefill",
            prompt_sha256=prompt_sha256,
            exact_match=exact_match,
            structure_score=structure_score,
            weaponization_score=weaponization_score,
            released_tokens_before_break=0,
            divider_complete=False,
        )

    def evaluate_mid_generation(
        self,
        *,
        placement: Placement,
        prompt_sha256: str,
        exact_match: bool,
        structure_score: float,
        weaponization_score: float,
        divider_complete: bool,
        released_tokens_before_break: int,
    ) -> BreakerDecision:
        return self._decision(
            placement=placement,
            stage="mid_generation",
            prompt_sha256=prompt_sha256,
            exact_match=exact_match,
            structure_score=structure_score,
            weaponization_score=weaponization_score,
            released_tokens_before_break=released_tokens_before_break,
            divider_complete=divider_complete,
        )
