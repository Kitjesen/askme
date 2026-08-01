from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bench import bench_ttft


def _chunk(content: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        value = self._chunks.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class _FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    async def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        stream = self._owner.streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        return _FakeStream(stream)


class _FakeChat:
    def __init__(self, owner):
        self.completions = _FakeCompletions(owner)


class _FakeClient:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []
        self.closed = False
        self.chat = _FakeChat(self)

    async def close(self):
        self.closed = True


def test_module_does_not_load_root_dotenv() -> None:
    source = Path(bench_ttft.__file__).read_text(encoding="utf-8")
    assert "dotenv" not in source
    assert "load_dotenv" not in source


def test_env_requires_litellm_only_credentials() -> None:
    with pytest.raises(bench_ttft.BenchConfigError, match="LITELLM_BASE_URL"):
        bench_ttft.load_config_from_env({"LITELLM_VIRTUAL_KEY": "sk-scoped"})

    with pytest.raises(bench_ttft.BenchConfigError, match="LITELLM_VIRTUAL_KEY"):
        bench_ttft.load_config_from_env({"LITELLM_BASE_URL": "http://litellm:4000/v1"})

    with pytest.raises(bench_ttft.BenchConfigError, match="LLM_API_KEY"):
        bench_ttft.load_config_from_env(
            {
                "LITELLM_BASE_URL": "http://litellm:4000/v1",
                "LITELLM_VIRTUAL_KEY": "sk-shared",
                "LLM_API_KEY": "sk-shared",
            }
        )


def test_release_mode_requires_at_least_50_samples() -> None:
    with pytest.raises(bench_ttft.BenchConfigError, match="at least 50 samples"):
        bench_ttft.load_config_from_env(
            {
                "LITELLM_BASE_URL": "http://litellm:4000/v1",
                "LITELLM_VIRTUAL_KEY": "sk-scoped",
            },
            mode="release",
            samples=49,
        )

    config = bench_ttft.load_config_from_env(
        {
            "LITELLM_BASE_URL": "http://litellm:4000/v1",
            "LITELLM_VIRTUAL_KEY": "sk-scoped",
        },
        mode="release",
        samples=50,
    )
    assert config.samples == 50


def test_plan_samples_randomizes_cold_and_warm_strata() -> None:
    planned = bench_ttft.plan_samples(12, seed=7)

    strata = [case.stratum for case in planned]
    assert strata.count("cold") == 6
    assert strata.count("warm") == 6
    assert strata != ["cold"] * 6 + ["warm"] * 6
    assert [case.prompt_id for case in planned[:3]] == [
        "time_short_zh",
        "robot_ack_zh",
        "brief_help_zh",
    ]


@pytest.mark.asyncio
async def test_semantic_ttft_ignores_empty_chunks_until_non_empty_payload() -> None:
    client = _FakeClient([[_chunk(None), _chunk(""), _chunk("   "), _chunk("收到"), _chunk("。")]])
    sample = bench_ttft.SampleCase(
        sample_id="sample-001",
        stratum="warm",
        prompt_id="time_short_zh",
        prompt_index=0,
    )

    result = await bench_ttft.measure_semantic_ttft(
        client,
        model="voice-fast",
        messages=bench_ttft.PROMPTS[0]["messages"],
        sample=sample,
        max_tokens=12,
    )

    assert result.status == "ok"
    assert result.semantic_ttft_ms is not None
    assert result.semantic_chunks == 2
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["model"] == "voice-fast"


@pytest.mark.asyncio
async def test_empty_stream_is_not_counted_as_successful_semantic_ttft() -> None:
    client = _FakeClient([[_chunk(""), _chunk("   ")]])
    sample = bench_ttft.SampleCase("sample-001", "cold", "time_short_zh", 0)

    result = await bench_ttft.measure_semantic_ttft(
        client,
        model="voice-fast",
        messages=bench_ttft.PROMPTS[0]["messages"],
        sample=sample,
        max_tokens=12,
    )

    assert result.status == "no_semantic_payload"
    assert result.semantic_ttft_ms is None
    assert result.semantic_chunks == 0


@pytest.mark.asyncio
async def test_run_benchmark_keeps_prompt_response_and_secrets_out_of_evidence() -> None:
    secret = "sk-sensitive-virtual-key"

    def factory(base_url: str, api_key: str, timeout_s: float) -> _FakeClient:
        assert base_url == "http://litellm:4000/v1"
        assert api_key == secret
        assert timeout_s == 30.0
        return _FakeClient([[_chunk("秘密响应正文")] for _ in range(5)])

    config = bench_ttft.BenchConfig(
        base_url="http://litellm:4000/v1",
        api_key=secret,
        model="voice-fast",
        samples=4,
        seed=3,
    )

    evidence = await bench_ttft.run_benchmark(config, client_factory=factory)
    evidence_text = str(evidence)

    assert evidence["product_gate_usable"] is False
    assert evidence["benchmark"] == "litellm_semantic_ttft"
    assert evidence["model_alias"] == "voice-fast"
    assert evidence["litellm_endpoint_host"] == "litellm:4000"
    assert evidence["sample_count_requested"] == 4
    assert evidence["sample_count_ok"] == 4
    assert len(evidence["samples"]) == 4
    assert "warm-primer" not in evidence_text
    assert evidence["summary"]["semantic_ttft_ms"]["aggregate"]["p50"] is not None
    assert "秘密响应正文" not in evidence_text
    assert "用一句自然中文回答" not in evidence_text
    assert secret not in evidence_text
    assert "Authorization" not in evidence_text


@pytest.mark.asyncio
async def test_warm_samples_are_recorded_only_after_successful_same_client_primer() -> None:
    created_clients: list[_FakeClient] = []

    def factory(base_url: str, api_key: str, timeout_s: float) -> _FakeClient:
        client = _FakeClient([[_chunk("primer ok")], [_chunk("warm ok")], [_chunk("warm ok")]])
        created_clients.append(client)
        return client

    config = bench_ttft.BenchConfig(
        base_url="http://litellm:4000/v1",
        api_key="sk-scoped",
        model="voice-fast",
        samples=3,
        seed=5,
    )

    evidence = await bench_ttft.run_benchmark(config, client_factory=factory)
    warm_samples = [sample for sample in evidence["samples"] if sample["stratum"] == "warm"]
    warm_clients = [client for client in created_clients if len(client.calls) > 1]

    assert evidence["sample_count_requested"] == 3
    assert len(evidence["samples"]) == 3
    assert len(warm_clients) == 1
    assert len(warm_clients[0].calls) == len(warm_samples) + 1
    assert all(sample["status"] == "ok" for sample in warm_samples)
    assert "warm-primer" not in str(evidence)


@pytest.mark.asyncio
async def test_failed_warm_primer_marks_warm_samples_without_recording_ttft() -> None:
    def factory(base_url: str, api_key: str, timeout_s: float) -> _FakeClient:
        return _FakeClient([[_chunk(""), _chunk("   ")], [_chunk("should not be called")]])

    config = bench_ttft.BenchConfig(
        base_url="http://litellm:4000/v1",
        api_key="sk-scoped",
        model="voice-fast",
        samples=1,
        seed=1,
    )

    evidence = await bench_ttft.run_benchmark(config, client_factory=factory)

    assert evidence["sample_count_requested"] == 1
    assert evidence["sample_count_ok"] == 0
    assert evidence["samples"] == [
        {
            "sample_id": "sample-001",
            "stratum": "warm",
            "prompt_id": "time_short_zh",
            "status": "warm_primer_failed",
            "model_alias": "voice-fast",
            "semantic_ttft_ms": None,
            "total_ms": 0.0,
            "semantic_chunks": 0,
            "error_type": "no_semantic_payload",
        }
    ]


def test_evidence_statistics_include_aggregate_and_stratified_percentiles() -> None:
    config = bench_ttft.BenchConfig(
        base_url="http://127.0.0.1:4000/v1",
        api_key="sk-scoped",
        model="voice-fast",
        samples=4,
    )
    evidence = bench_ttft.build_evidence(
        config,
        [
            bench_ttft.SampleResult("s1", "cold", "p1", "ok", "voice-fast", 100.0, 150.0, 1),
            bench_ttft.SampleResult("s2", "warm", "p2", "ok", "voice-fast", 200.0, 250.0, 1),
            bench_ttft.SampleResult("s3", "cold", "p3", "ok", "voice-fast", 300.0, 350.0, 1),
            bench_ttft.SampleResult(
                "s4", "warm", "p1", "error", "voice-fast", None, 50.0, 0, "TimeoutError"
            ),
        ],
    )

    stats = evidence["summary"]["semantic_ttft_ms"]
    assert evidence["product_gate_usable"] is False
    assert stats["aggregate"] == {"p50": 200.0, "p95": 290.0, "p99": 298.0}
    assert stats["cold"] == {"p50": 200.0, "p95": 290.0, "p99": 298.0}
    assert stats["warm"] == {"p50": 200.0, "p95": 200.0, "p99": 200.0}
    assert evidence["sample_count_ok"] == 3
    assert evidence["samples"][3]["error_type"] == "TimeoutError"
