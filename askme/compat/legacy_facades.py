"""Explicit registry of compatibility facades kept during package migration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyFacade:
    """One historical import path and the canonical owner it forwards to."""

    legacy_path: str
    canonical_path: str
    new_code_import: str
    owner: str
    reason: str


LEGACY_FACADES: tuple[LegacyFacade, ...] = (
    LegacyFacade(
        legacy_path="askme.voice.runtime_bridge",
        canonical_path="askme.voice_gateway.runtime_bridge",
        new_code_import=(
            "askme.providers.build_voice_runtime_bridge for construction; "
            "askme.voice_gateway.VoiceGatewayService for turn handling"
        ),
        owner="voice_gateway",
        reason="Historical voice bridge import path kept for callers and monkeypatch tests.",
    ),
    LegacyFacade(
        legacy_path="askme.voice.orchestration.runtime_bridge",
        canonical_path="askme.voice_gateway.runtime_bridge",
        new_code_import=(
            "askme.providers.build_voice_runtime_bridge for construction; "
            "askme.voice_gateway.VoiceGatewayService for turn handling"
        ),
        owner="voice_gateway",
        reason="Runtime bridge moved out of voice orchestration into the gateway boundary.",
    ),
    LegacyFacade(
        legacy_path="askme.voice_gateway.runtime_bridge",
        canonical_path="askme.providers.voice_runtime",
        new_code_import=(
            "askme.providers.build_voice_runtime_bridge for construction; "
            "askme.voice_gateway.VoiceGatewayService for turn handling"
        ),
        owner="providers",
        reason="Thin facade over provider-owned HTTP bridge implementation.",
    ),
    LegacyFacade(
        legacy_path="askme.voice.input.address_detector",
        canonical_path="askme.robot_interaction.address_detector",
        new_code_import="askme.robot_interaction.AddressDetector",
        owner="robot_interaction",
        reason="Address detection is robot interaction policy, not audio input.",
    ),
    LegacyFacade(
        legacy_path="askme.interaction.intent_router",
        canonical_path="askme.robot_interaction.intent_router",
        new_code_import="askme.robot_interaction.IntentRouter",
        owner="robot_interaction",
        reason="Old interaction package remains only as a router facade.",
    ),
    LegacyFacade(
        legacy_path="askme.interaction.routing_policy",
        canonical_path="askme.robot_interaction.routing_policy",
        new_code_import="askme.robot_interaction.routing_policy",
        owner="robot_interaction",
        reason="Old interaction package remains only as a policy facade.",
    ),
    LegacyFacade(
        legacy_path="askme.interaction.observability",
        canonical_path="askme.robot_interaction.observability",
        new_code_import="askme.robot_interaction.observability",
        owner="robot_interaction",
        reason="Old interaction package remains only as an observability facade.",
    ),
    LegacyFacade(
        legacy_path="askme.interaction.scenario_intents",
        canonical_path="askme.robot_interaction.scenario_intents",
        new_code_import="askme.robot_interaction.scenario_intents",
        owner="robot_interaction",
        reason="Scenario intent rules are robot interaction policy.",
    ),
    LegacyFacade(
        legacy_path="askme.pipeline.reactions.state_led_bridge",
        canonical_path="askme.providers.led",
        new_code_import="askme.providers.build_status_led",
        owner="providers",
        reason="Pipeline keeps the historical bridge import while runtime uses provider assembly.",
    ),
    LegacyFacade(
        legacy_path="askme.robot.telemetry.ota_bridge",
        canonical_path="askme.telemetry.ota_bridge",
        new_code_import="askme.telemetry.ota_bridge",
        owner="telemetry",
        reason="Shared OTA metrics are not robot hardware control.",
    ),
    LegacyFacade(
        legacy_path="askme.robot.telemetry.pubsub",
        canonical_path="askme.interfaces.bus",
        new_code_import="askme.interfaces.bus.BusBackend",
        owner="interfaces",
        reason="Pub/sub is a backend contract, not a robot implementation module.",
    ),
)

LEGACY_FACADE_BY_PATH: dict[str, LegacyFacade] = {
    item.legacy_path: item for item in LEGACY_FACADES
}


def legacy_facade_for(path: str) -> LegacyFacade | None:
    """Return facade metadata for a historical import path."""

    return LEGACY_FACADE_BY_PATH.get(path)


__all__ = ["LEGACY_FACADES", "LEGACY_FACADE_BY_PATH", "LegacyFacade", "legacy_facade_for"]
