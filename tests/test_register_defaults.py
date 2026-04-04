"""Tests for register_defaults — backend registration completeness."""

from __future__ import annotations

import pytest

# Import module to trigger all registrations
import askme.interfaces.register_defaults  # noqa: F401

from askme.interfaces.llm import llm_registry
from askme.interfaces.asr import asr_registry
from askme.interfaces.tts import tts_registry
from askme.interfaces.bus import bus_registry
from askme.interfaces.reaction import reaction_registry


class TestLLMRegistry:
    def test_minimax_registered(self):
        from askme.llm.client import LLMClient
        cls = llm_registry._backends.get("minimax")
        assert cls is LLMClient


class TestASRRegistry:
    def test_sherpa_registered(self):
        from askme.voice.asr import ASREngine
        cls = asr_registry._backends.get("sherpa")
        assert cls is ASREngine


class TestTTSRegistry:
    def test_minimax_registered(self):
        from askme.voice.tts import TTSEngine
        cls = tts_registry._backends.get("minimax")
        assert cls is TTSEngine


class TestBusRegistry:
    def test_pulse_registered(self):
        from askme.robot.pulse import Pulse
        cls = bus_registry._backends.get("pulse")
        assert cls is Pulse

    def test_mock_pulse_registered(self):
        from askme.robot.mock_pulse import MockPulse
        cls = bus_registry._backends.get("mock")
        assert cls is MockPulse


class TestReactionRegistry:
    def test_hybrid_registered(self):
        from askme.pipeline.reaction_engine import HybridReaction
        cls = reaction_registry._backends.get("hybrid")
        assert cls is HybridReaction

    def test_rules_registered(self):
        from askme.pipeline.reaction_engine import RuleBasedReaction
        cls = reaction_registry._backends.get("rules")
        assert cls is RuleBasedReaction


class TestRegistryCompleteness:
    def test_llm_registry_has_minimax(self):
        assert llm_registry._backends.get("minimax") is not None

    def test_asr_registry_has_sherpa(self):
        assert asr_registry._backends.get("sherpa") is not None

    def test_tts_registry_has_minimax(self):
        assert tts_registry._backends.get("minimax") is not None

    def test_bus_registry_has_pulse_and_mock(self):
        assert bus_registry._backends.get("pulse") is not None
        assert bus_registry._backends.get("mock") is not None

    def test_reaction_registry_has_both_types(self):
        assert reaction_registry._backends.get("hybrid") is not None
        assert reaction_registry._backends.get("rules") is not None
