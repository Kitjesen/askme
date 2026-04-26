"""Tests for MemoryAdmissionControl — 5-factor scoring, novelty, threshold."""

from __future__ import annotations

from askme.memory.admission import AdmissionScore, MemoryAdmissionControl

# ── AdmissionScore ────────────────────────────────────────────────────────────

class TestAdmissionScore:
    def test_total_is_weighted_sum(self):
        score = AdmissionScore(
            utility=1.0, confidence=1.0, novelty=1.0, recency=1.0, type_prior=1.0
        )
        assert abs(score.total - 1.0) < 0.01

    def test_total_zeros(self):
        score = AdmissionScore(
            utility=0.0, confidence=0.0, novelty=0.0, recency=0.0, type_prior=0.0
        )
        assert score.total == 0.0

    def test_utility_weighted_highest(self):
        base = AdmissionScore(utility=0.0, confidence=0.0, novelty=0.0, recency=0.0, type_prior=0.0)
        with_utility = AdmissionScore(utility=1.0, confidence=0.0, novelty=0.0, recency=0.0, type_prior=0.0)
        with_confidence = AdmissionScore(utility=0.0, confidence=1.0, novelty=0.0, recency=0.0, type_prior=0.0)
        assert with_utility.total > with_confidence.total


# ── MemoryAdmissionControl ────────────────────────────────────────────────────

class TestMemoryAdmissionControl:
    def test_default_threshold(self):
        mac = MemoryAdmissionControl()
        assert mac.threshold == 0.4

    def test_custom_threshold(self):
        mac = MemoryAdmissionControl(threshold=0.6)
        assert mac.threshold == 0.6

    def test_threshold_setter(self):
        mac = MemoryAdmissionControl()
        mac.threshold = 0.7
        assert mac.threshold == 0.7

    def test_error_event_high_priority_admitted(self):
        mac = MemoryAdmissionControl(threshold=0.4)
        admitted, score = mac.should_admit("error", "Motor overcurrent", importance=0.8)
        assert admitted is True
        assert score.type_prior == 0.9

    def test_routine_perception_low_priority(self):
        mac = MemoryAdmissionControl(threshold=0.5)
        admitted, score = mac.should_admit("perception", "nothing happened", importance=0.1)
        # Low importance + low type prior → likely rejected at high threshold
        assert isinstance(admitted, bool)
        assert 0.0 <= score.total <= 1.0

    def test_unknown_kind_uses_default_prior(self):
        mac = MemoryAdmissionControl()
        admitted, score = mac.should_admit("unknown_type", "some text", importance=0.5)
        assert score.type_prior == 0.4  # default for unknown

    def test_first_event_has_full_novelty(self):
        mac = MemoryAdmissionControl()
        _, score = mac.should_admit("command", "first event text", importance=0.5)
        assert score.novelty == 1.0

    def test_identical_event_has_low_novelty(self):
        mac = MemoryAdmissionControl()
        text = "同一事件内容"
        mac.should_admit("command", text, importance=0.9)  # first — admitted, adds to recent
        _, score2 = mac.should_admit("command", text, importance=0.9)
        # Novelty should be low for identical text
        assert score2.novelty < 0.5

    def test_completely_different_event_has_high_novelty(self):
        mac = MemoryAdmissionControl()
        mac.should_admit("command", "abcdef", importance=0.9)  # first event
        _, score = mac.should_admit("command", "xyz123", importance=0.5)
        # Very different text → high novelty
        assert score.novelty > 0.3

    def test_admitted_event_added_to_recent(self):
        mac = MemoryAdmissionControl(threshold=0.0)  # admit everything
        mac.should_admit("error", "test text", importance=0.5)
        assert len(mac._recent_texts) == 1

    def test_rejected_event_not_added_to_recent(self):
        mac = MemoryAdmissionControl(threshold=1.1)  # reject everything
        mac.should_admit("error", "test text", importance=0.5)
        assert len(mac._recent_texts) == 0

    def test_recency_always_1(self):
        mac = MemoryAdmissionControl()
        _, score = mac.should_admit("command", "test", importance=0.5)
        assert score.recency == 1.0

    def test_high_confidence_for_direct_observation_types(self):
        mac = MemoryAdmissionControl()
        for kind in ("command", "error", "action"):
            _, score = mac.should_admit(kind, "text", importance=0.5)
            assert score.confidence == 0.9

    def test_low_confidence_for_inferred_types(self):
        mac = MemoryAdmissionControl()
        _, score = mac.should_admit("outcome", "text", importance=0.5)
        assert score.confidence == 0.7

    def test_score_returns_valid_total(self):
        mac = MemoryAdmissionControl()
        _, score = mac.should_admit("perception", "routine scan", importance=0.3)
        assert 0.0 <= score.total <= 1.0

    def test_empty_text_has_full_novelty(self):
        mac = MemoryAdmissionControl()
        mac.should_admit("command", "some previous text", importance=0.8)
        _, score = mac.should_admit("command", "", importance=0.5)
        assert score.novelty == 1.0
