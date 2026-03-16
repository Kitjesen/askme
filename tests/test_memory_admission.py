"""Tests for memory admission control and Ebbinghaus decay."""

from __future__ import annotations

import math

from askme.brain.episode import decay_importance
from askme.brain.memory_admission import AdmissionScore, MemoryAdmissionControl


# ── decay_importance ──────────────────────────────────────


def test_decay_fresh_event_full_importance():
    """Fresh event (age=0) returns full importance."""
    assert decay_importance(0.8, age_hours=0.0) == 0.8


def test_decay_negative_age_full_importance():
    """Negative age returns full importance (guard)."""
    assert decay_importance(0.8, age_hours=-5.0) == 0.8


def test_decay_3day_halves():
    """After 3 days (default half-life=72h), importance halves."""
    result = decay_importance(1.0, age_hours=72.0)
    assert abs(result - 0.5) < 0.01


def test_decay_10day_very_low():
    """After 10 days, importance is very low."""
    result = decay_importance(1.0, age_hours=240.0)
    # 2^(-240/72) = 2^(-3.33) ~ 0.1
    assert result < 0.15


def test_decay_custom_half_life():
    """Custom half-life works correctly."""
    result = decay_importance(1.0, age_hours=24.0, half_life_hours=24.0)
    assert abs(result - 0.5) < 0.01


def test_decay_zero_importance():
    """Zero importance stays zero regardless of age."""
    assert decay_importance(0.0, age_hours=100.0) == 0.0


# ── AdmissionScore ────────────────────────────────────────


def test_admission_score_total():
    """Score total is weighted sum: 0.3*u + 0.15*c + 0.25*n + 0.1*r + 0.2*t."""
    score = AdmissionScore(
        utility=1.0, confidence=1.0, novelty=1.0, recency=1.0, type_prior=1.0
    )
    assert abs(score.total - 1.0) < 0.001


def test_admission_score_breakdown():
    """Verify individual factor contributions."""
    score = AdmissionScore(
        utility=0.5, confidence=0.8, novelty=0.6, recency=1.0, type_prior=0.9
    )
    expected = 0.3 * 0.5 + 0.15 * 0.8 + 0.25 * 0.6 + 0.1 * 1.0 + 0.2 * 0.9
    assert abs(score.total - expected) < 0.001


# ── MemoryAdmissionControl ───────────────────────────────


def test_error_always_admitted():
    """Error events always admitted (type_prior=0.9)."""
    mac = MemoryAdmissionControl(threshold=0.4)
    admitted, score = mac.should_admit("error", "Motor overcurrent", importance=0.0)
    assert admitted
    assert score.type_prior == 0.9


def test_anomaly_always_admitted():
    """Anomaly events always admitted (type_prior=0.9)."""
    mac = MemoryAdmissionControl(threshold=0.4)
    admitted, score = mac.should_admit("anomaly", "Sensor spike detected", importance=0.0)
    assert admitted
    assert score.type_prior == 0.9


def test_duplicate_text_rejected():
    """Duplicate text has low novelty and gets rejected."""
    mac = MemoryAdmissionControl(threshold=0.4)
    # First occurrence: novel
    admitted1, _ = mac.should_admit("perception", "Battery at 80%", importance=0.1)
    # Second occurrence: same text, low novelty
    admitted2, score2 = mac.should_admit("perception", "Battery at 80%", importance=0.1)
    assert admitted1  # first is novel
    assert not admitted2  # duplicate rejected
    assert score2.novelty < 0.3  # low novelty


def test_novel_text_admitted():
    """Novel text is admitted even for low-priority types."""
    mac = MemoryAdmissionControl(threshold=0.4)
    admitted, score = mac.should_admit("action", "Navigated to warehouse B", importance=0.5)
    assert admitted
    assert score.novelty == 1.0  # first text is fully novel


def test_perception_low_importance_rejected():
    """Perception events with very low importance get rejected."""
    mac = MemoryAdmissionControl(threshold=0.5)
    # Seed recent texts so novelty drops
    mac.should_admit("perception", "ABCDEFGHIJKLMNOP routine scan", importance=0.0)
    admitted, score = mac.should_admit(
        "perception", "ABCDEFGHIJKLMNOP routine scan 2", importance=0.0
    )
    # type_prior=0.3, utility=0.0, novelty~low => should be rejected at 0.5 threshold
    assert not admitted


def test_threshold_adjustable():
    """Threshold can be changed after construction."""
    mac = MemoryAdmissionControl(threshold=0.9)
    admitted, _ = mac.should_admit("action", "Walk forward", importance=0.3)
    assert not admitted  # high threshold rejects

    mac.threshold = 0.1
    admitted, _ = mac.should_admit("action", "Walk forward again", importance=0.3)
    assert admitted  # low threshold admits


def test_command_high_confidence():
    """Command events get high confidence score (direct observation)."""
    mac = MemoryAdmissionControl(threshold=0.4)
    _, score = mac.should_admit("command", "User said: go to gate", importance=0.7)
    assert score.confidence == 0.9


def test_unknown_type_uses_default_prior():
    """Unknown event types use default type_prior of 0.4."""
    mac = MemoryAdmissionControl(threshold=0.4)
    _, score = mac.should_admit("custom_event", "Something happened", importance=0.5)
    assert score.type_prior == 0.4


def test_empty_text_novelty():
    """Empty text still computes novelty without error."""
    mac = MemoryAdmissionControl(threshold=0.1)
    admitted, score = mac.should_admit("system", "", importance=0.5)
    # Should not crash; novelty of empty text = 1.0 (no recent to compare)
    assert score.novelty == 1.0
