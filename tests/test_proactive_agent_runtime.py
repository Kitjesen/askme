"""Regression tests for ProactiveAgent runtime decisions and event loops."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from askme.pipeline.proactive_agent import ProactiveAgent

from askme.schemas.events import ChangeEvent, ChangeEventType


def _make_agent(
    *,
    vision: MagicMock | None = None,
    change_events_enabled: bool = False,
    event_file: Path | None = None,
    auto_solve: bool = True,
    telemetry_hub_url: str | None = None,
) -> tuple[ProactiveAgent, MagicMock, MagicMock]:
    llm = MagicMock()
    llm.chat = AsyncMock()

    audio = MagicMock()
    audio.is_busy = False

    episodic = MagicMock()
    episodic.log.return_value = None

    agent = ProactiveAgent(
        vision=vision,
        audio=audio,
        episodic=episodic,
        llm=llm,
        config={
            "proactive": {
                "enabled": True,
                "patrol_interval": 60,
                "alert_cooldown": 0,
                "auto_solve": auto_solve,
                "telemetry_hub_url": telemetry_hub_url,
                "change_detector": {
                    "enabled": change_events_enabled,
                    "event_file": str(event_file) if event_file else "/tmp/askme_events.jsonl",
                },
            },
        },
    )
    return agent, llm, episodic


def _vision_with_scene(scene: str) -> MagicMock:
    vision = MagicMock()
    vision.available = True
    vision.describe_scene = AsyncMock(return_value=scene)
    vision.save_snapshot = AsyncMock(return_value="/tmp/snapshot.jpg")
    return vision


@pytest.mark.asyncio
async def test_detect_anomaly_skips_llm_without_baseline():
    agent, llm, _episodic = _make_agent()

    result = await agent._detect_anomaly("当前画面")

    assert result is None
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_detect_anomaly_returns_description_from_llm_response():
    agent, llm, _episodic = _make_agent()
    agent._scene_history.append("上一帧：无人")
    llm.chat.return_value = "ANOMALY|有人进入"

    result = await agent._detect_anomaly("当前帧：有人")

    assert result == "有人进入"
    llm.chat.assert_awaited_once()
    _messages, kwargs = llm.chat.call_args
    assert kwargs["model"] == agent._judge_model
    assert kwargs["temperature"] == 0.1


@pytest.mark.asyncio
async def test_patrol_tick_successful_baseline_logs_without_alert():
    vision = _vision_with_scene("货架正常")
    agent, _llm, episodic = _make_agent(vision=vision)
    agent._speak_alert = AsyncMock()

    await agent._patrol_tick()

    assert agent._tick_count == 1
    assert list(agent._scene_history) == ["货架正常"]
    episodic.log.assert_called_once_with("perception", "巡检扫描: 货架正常")
    agent._speak_alert.assert_not_awaited()
    vision.save_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_patrol_tick_retries_once_when_scene_capture_is_empty(monkeypatch):
    vision = MagicMock()
    vision.available = True
    vision.describe_scene = AsyncMock(side_effect=["", "复扫成功"])
    vision.save_snapshot = AsyncMock()
    sleep = AsyncMock()
    monkeypatch.setattr("askme.pipeline.proactive_agent.asyncio.sleep", sleep)

    agent, _llm, _episodic = _make_agent(vision=vision)
    agent._speak_alert = AsyncMock()

    await agent._patrol_tick()

    assert vision.describe_scene.await_count == 2
    sleep.assert_awaited_once_with(2.0)
    assert list(agent._scene_history) == ["复扫成功"]


@pytest.mark.asyncio
async def test_patrol_tick_anomaly_speaks_logs_snapshot_and_solves():
    vision = _vision_with_scene("检测到有人")
    agent, _llm, episodic = _make_agent(vision=vision)
    agent._scene_history.append("上一帧：无人")
    agent._detect_anomaly = AsyncMock(return_value="有人进入")
    agent._speak_alert = AsyncMock()
    solve_callback = AsyncMock()
    agent.set_solve_callback(solve_callback)

    await agent._patrol_tick()

    assert agent._consecutive_normal == 0
    assert agent._last_anomaly_time > 0
    vision.save_snapshot.assert_awaited_once_with(label="anomaly_1")
    assert episodic.log.call_args_list[0].args == ("perception", "巡检扫描: 检测到有人")
    assert episodic.log.call_args_list[1].args[:2] == ("perception", "异常: 有人进入")
    agent._speak_alert.assert_awaited_once()
    message = agent._speak_alert.await_args.args[0]
    assert message == "巡检异常：有人进入"
    assert agent._speak_alert.await_args.kwargs["severity"] == "warning"
    assert agent._speak_alert.await_args.kwargs["topic"] == "patrol.anomaly"
    solve_callback.assert_awaited_once()
    assert "有人进入" in solve_callback.await_args.args[0]
    assert list(agent._scene_history)[-1] == "检测到有人"


@pytest.mark.asyncio
async def test_change_event_loop_processes_new_events_until_stop(tmp_path):
    event_file = tmp_path / "events.jsonl"
    agent, _llm, _episodic = _make_agent(
        change_events_enabled=True,
        event_file=event_file,
    )
    stop_event = asyncio.Event()
    event = object()

    async def handle_event(_event: object) -> None:
        stop_event.set()

    agent._read_change_events = MagicMock(return_value=(99, [event]))
    agent._handle_change_event = AsyncMock(side_effect=handle_event)

    await agent._change_event_loop(stop_event)

    agent._read_change_events.assert_called_once_with(0)
    agent._handle_change_event.assert_awaited_once_with(event)


def test_read_change_events_ignores_invalid_lines_and_advances_position(tmp_path):
    event_file = tmp_path / "events.jsonl"
    event = ChangeEvent(
        event_type=ChangeEventType.PERSON_APPEARED,
        timestamp=123.0,
        subject_class="person",
        confidence=0.9,
    )
    event_file.write_text(
        json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        "not-json\n"
        "\n",
        encoding="utf-8",
    )
    agent, _llm, _episodic = _make_agent(
        change_events_enabled=True,
        event_file=event_file,
    )

    result = agent._read_change_events(0)

    assert result is not None
    new_pos, parsed = result
    assert new_pos == event_file.stat().st_size
    assert len(parsed) == 1
    assert parsed[0].event_type == ChangeEventType.PERSON_APPEARED


@pytest.mark.asyncio
async def test_handle_change_event_person_alerts_and_auto_solves():
    agent, _llm, episodic = _make_agent()
    agent._speak_alert = AsyncMock()
    solve_callback = AsyncMock()
    agent.set_solve_callback(solve_callback)
    event = ChangeEvent(
        event_type=ChangeEventType.PERSON_APPEARED,
        timestamp=123.0,
        subject_class="person",
        confidence=0.95,
        importance=0.8,
        distance_m=1.5,
    )

    await agent._handle_change_event(event)

    description = "检测到有人出现，距离1.5米"
    episodic.log.assert_called_once_with("perception", f"感知事件: {description}")
    agent._speak_alert.assert_awaited_once()
    assert agent._speak_alert.await_args.args[0] == description
    assert agent._speak_alert.await_args.kwargs["severity"] == "warning"
    assert agent._speak_alert.await_args.kwargs["topic"] == "change.person"
    assert agent._speak_alert.await_args.kwargs["payload"]["event"] == event.to_dict()
    solve_callback.assert_awaited_once()
    assert description in solve_callback.await_args.args[0]


@pytest.mark.asyncio
async def test_handle_change_event_low_importance_object_only_logs():
    agent, _llm, episodic = _make_agent()
    agent._speak_alert = AsyncMock()
    solve_callback = AsyncMock()
    agent.set_solve_callback(solve_callback)
    event = ChangeEvent(
        event_type=ChangeEventType.OBJECT_APPEARED,
        timestamp=123.0,
        subject_class="箱子",
        confidence=0.7,
        importance=0.4,
    )

    await agent._handle_change_event(event)

    episodic.log.assert_called_once_with("perception", "感知事件: 检测到新物体：箱子")
    agent._speak_alert.assert_not_awaited()
    solve_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_telemetry_events_speaks_recognized_navigation_alert(monkeypatch):
    agent, _llm, episodic = _make_agent(telemetry_hub_url="http://telemetry.local")
    agent._fetch_events = MagicMock(
        return_value=[
            {
                "event_id": "evt-1",
                "topic": "navigation.stall_detected",
                "severity": "warning",
                "payload": {"stall_duration_s": 12.0},
            }
        ]
    )
    agent._speak_alert = AsyncMock()
    monkeypatch.setattr("askme.pipeline.proactive_agent.time.monotonic", lambda: 100.0)

    await agent._poll_telemetry_events()

    agent._speak_alert.assert_awaited_once()
    assert "12秒" in agent._speak_alert.await_args.args[0]
    assert agent._speak_alert.await_args.kwargs["severity"] == "warning"
    assert agent._speak_alert.await_args.kwargs["topic"] == "navigation.stall_detected"
    episodic.log.assert_called_once()
    assert "主动告警[warning]" in episodic.log.call_args.args[1]
    assert agent._last_event_id == "evt-1"
    assert "evt-1" in agent._seen_event_ids


@pytest.mark.asyncio
async def test_poll_telemetry_events_dedupes_filters_and_applies_topic_cooldown(monkeypatch):
    agent, _llm, episodic = _make_agent(telemetry_hub_url="http://telemetry.local")
    agent._seen_event_ids.add("evt-seen")
    agent._topic_last_spoken["navigation.stall_detected"] = 95.0
    agent._fetch_events = MagicMock(
        return_value=[
            {
                "event_id": "evt-seen",
                "topic": "navigation.stall_detected",
                "payload": {"stall_duration_s": 30.0},
            },
            {
                "event_id": "evt-unknown",
                "topic": "diagnostics.heartbeat",
                "payload": {},
            },
            {
                "event_id": "evt-cooldown",
                "topic": "navigation.stall_detected",
                "payload": {"stall_duration_s": 31.0},
            },
        ]
    )
    agent._speak_alert = AsyncMock()
    monkeypatch.setattr("askme.pipeline.proactive_agent.time.monotonic", lambda: 100.0)

    await agent._poll_telemetry_events()

    agent._speak_alert.assert_not_awaited()
    episodic.log.assert_not_called()
    assert agent._last_event_id == "evt-cooldown"
    assert {"evt-seen", "evt-unknown", "evt-cooldown"} <= agent._seen_event_ids
