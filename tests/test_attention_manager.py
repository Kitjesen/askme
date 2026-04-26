"""Tests for AttentionManager — should_alert, should_investigate, cooldowns."""

from __future__ import annotations

import time

from askme.perception.attention_manager import AttentionConfig, AttentionManager
from askme.schemas.events import ChangeEvent, ChangeEventType

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_event(
    event_type: ChangeEventType = ChangeEventType.PERSON_APPEARED,
    importance: float | None = None,
) -> ChangeEvent:
    """Create a minimal ChangeEvent."""
    evt = ChangeEvent(
        event_type=event_type,
        timestamp=time.time(),
        subject_class="person",
    )
    if importance is not None:
        evt.importance = importance
    return evt


def _make_manager(*, alert_threshold: float = 0.5, investigate_threshold: float = 0.7) -> AttentionManager:
    """Create an AttentionManager with custom thresholds (no config.yaml)."""
    mgr = AttentionManager()
    mgr._cfg.alert_threshold = alert_threshold
    mgr._cfg.investigate_threshold = investigate_threshold
    return mgr


# ── AttentionConfig ───────────────────────────────────────────────────────────

class TestAttentionConfig:
    def test_default_thresholds(self):
        cfg = AttentionConfig()
        assert cfg.alert_threshold == 0.5
        assert cfg.investigate_threshold == 0.7

    def test_from_config_defaults(self):
        cfg = AttentionConfig.from_config({})
        assert cfg.alert_threshold == 0.5
        assert cfg.investigate_threshold == 0.7

    def test_from_config_custom_thresholds(self):
        cfg = AttentionConfig.from_config({
            "proactive": {"attention": {"alert_threshold": 0.3, "investigate_threshold": 0.8}}
        })
        assert cfg.alert_threshold == 0.3
        assert cfg.investigate_threshold == 0.8

    def test_from_config_unknown_event_type_ignored(self):
        cfg = AttentionConfig.from_config({
            "proactive": {"attention": {"cooldowns": {"unknown_event_type": 99}}}
        })
        # Should not raise
        assert isinstance(cfg, AttentionConfig)


# ── should_alert ──────────────────────────────────────────────────────────────

class TestShouldAlert:
    def test_high_importance_triggers_alert(self):
        mgr = _make_manager()
        event = _make_event(importance=0.9)
        assert mgr.should_alert(event) is True

    def test_low_importance_suppressed(self):
        mgr = _make_manager(alert_threshold=0.5)
        event = _make_event(importance=0.3)
        assert mgr.should_alert(event) is False

    def test_exact_threshold_triggers(self):
        mgr = _make_manager(alert_threshold=0.5)
        event = _make_event(importance=0.5)
        assert mgr.should_alert(event) is True

    def test_second_call_on_cooldown(self):
        mgr = _make_manager()
        event = _make_event(importance=0.9)
        assert mgr.should_alert(event) is True
        # Second immediate call should be blocked by cooldown
        assert mgr.should_alert(event) is False

    def test_cooldown_expires(self):
        mgr = _make_manager()
        event = _make_event(importance=0.9)
        # Manually set last alert to far in the past
        mgr._last_alert[event.event_type] = time.time() - 9999
        assert mgr.should_alert(event) is True

    def test_different_event_types_independent_cooldowns(self):
        mgr = _make_manager()
        person = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.9)
        obj = _make_event(ChangeEventType.OBJECT_APPEARED, importance=0.6)
        assert mgr.should_alert(person) is True
        # Person is on cooldown, but object is not
        assert mgr.should_alert(obj) is True


# ── should_investigate ────────────────────────────────────────────────────────

class TestShouldInvestigate:
    def test_high_importance_triggers_investigate(self):
        mgr = _make_manager()
        event = _make_event(importance=0.9)
        assert mgr.should_investigate(event) is True

    def test_low_importance_not_investigated(self):
        mgr = _make_manager(investigate_threshold=0.7)
        event = _make_event(importance=0.5)
        assert mgr.should_investigate(event) is False

    def test_investigate_uses_double_cooldown(self):
        mgr = _make_manager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.9)
        # Alert cooldown = 10s, investigate cooldown = 20s
        mgr._last_investigate[event.event_type] = time.time() - 15  # 15s ago
        # 15s < 20s → still on cooldown
        assert mgr.should_investigate(event) is False

    def test_investigate_after_double_cooldown(self):
        mgr = _make_manager()
        event = _make_event(ChangeEventType.PERSON_APPEARED, importance=0.9)
        mgr._last_investigate[event.event_type] = time.time() - 25  # 25s ago > 20s
        assert mgr.should_investigate(event) is True


# ── cooldown_remaining ────────────────────────────────────────────────────────

class TestCooldownRemaining:
    def test_no_alert_yet_returns_zero(self):
        mgr = _make_manager()
        remaining = mgr.cooldown_remaining(ChangeEventType.PERSON_APPEARED)
        assert remaining == 0.0

    def test_just_alerted_returns_full_cooldown(self):
        mgr = _make_manager()
        event = _make_event(importance=0.9)
        mgr.should_alert(event)
        remaining = mgr.cooldown_remaining(ChangeEventType.PERSON_APPEARED)
        # Should be close to 10s (PERSON_APPEARED cooldown)
        assert 9.0 < remaining <= 10.0

    def test_expired_cooldown_returns_zero(self):
        mgr = _make_manager()
        mgr._last_alert[ChangeEventType.PERSON_APPEARED] = time.time() - 9999
        remaining = mgr.cooldown_remaining(ChangeEventType.PERSON_APPEARED)
        assert remaining == 0.0


# ── reset_cooldown / reset_all_cooldowns ──────────────────────────────────────

class TestResetCooldowns:
    def test_reset_single_cooldown(self):
        mgr = _make_manager()
        event = _make_event(importance=0.9)
        mgr.should_alert(event)  # sets cooldown
        mgr.reset_cooldown(event.event_type)
        # Cooldown gone — should alert again
        assert mgr.should_alert(event) is True

    def test_reset_all_cooldowns(self):
        mgr = _make_manager()
        for et in ChangeEventType:
            mgr._last_alert[et] = time.time()
        mgr.reset_all_cooldowns()
        assert mgr._last_alert == {}
        assert mgr._last_investigate == {}


# ── status ────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_returns_dict(self):
        mgr = _make_manager()
        s = mgr.status()
        assert isinstance(s, dict)

    def test_has_thresholds(self):
        mgr = _make_manager(alert_threshold=0.4, investigate_threshold=0.8)
        s = mgr.status()
        assert s["alert_threshold"] == 0.4
        assert s["investigate_threshold"] == 0.8

    def test_has_cooldowns_remaining(self):
        mgr = _make_manager()
        s = mgr.status()
        assert "cooldowns_remaining" in s
        assert isinstance(s["cooldowns_remaining"], dict)

    def test_cooldowns_remaining_are_floats(self):
        mgr = _make_manager()
        s = mgr.status()
        for val in s["cooldowns_remaining"].values():
            assert isinstance(val, float)
