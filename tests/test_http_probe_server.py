from __future__ import annotations

import asyncio

import aiohttp
import pytest

from askme.health_server import AskmeHealthServer


@pytest.mark.asyncio
async def test_health_http_server_serves_health_and_metrics() -> None:
    health_payload = {
        "status": "ok",
        "uptime_seconds": 12.0,
        "model_name": "claude-haiku",
    }
    metrics_payload = {
        "status": "ok",
        "last_llm_latency_ms": 123.0,
        "active_skills": ["daily_summary"],
        "voice_pipeline_status": {"pipeline_ok": True},
    }
    server = AskmeHealthServer(
        {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 0,
        },
        health_provider=lambda: dict(health_payload),
        metrics_provider=lambda: dict(metrics_payload),
    )
    task = asyncio.create_task(server.serve())

    try:
        await server.wait_started(task)
        base_url = f"http://127.0.0.1:{server.bound_port}"
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert await response.json() == health_payload

            async with session.get(f"{base_url}/metrics") as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert await response.json() == metrics_payload
    finally:
        await server.stop()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_health_http_server_reports_provider_failures() -> None:
    server = AskmeHealthServer(
        {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 0,
        },
        health_provider=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        metrics_provider=lambda: {"status": "ok"},
    )
    task = asyncio.create_task(server.serve())

    try:
        await server.wait_started(task)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{server.bound_port}/health") as response:
                assert response.status == 500
                payload = await response.json()
                assert payload["status"] == "error"
                assert payload["error"] == "boom"
    finally:
        await server.stop()
        await asyncio.gather(task, return_exceptions=True)
