"""Queue, lifecycle, idempotency, and target safety for product speech playback."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
import uuid
import wave
from array import array
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.ports.speech_playback import (
    PlaybackTarget,
    SpeechAudioArtifactFile,
    SpeechDelivery,
    SpeechPlaybackError,
    SpeechPlaybackJob,
    SpeechPlaybackRequest,
    SpeechPlaybackState,
    SpeechPlaybackTimestamps,
    SpeechPriority,
)
from askme.telemetry.metrics import increment_counter, record_metric

logger = logging.getLogger(__name__)

_PRIORITY_RANK = {
    SpeechPriority.SAFETY: 0,
    SpeechPriority.OPERATOR: 10,
    SpeechPriority.NORMAL: 20,
    SpeechPriority.LOW: 30,
}
_BUSY_STATES = {
    SpeechPlaybackState.QUEUED,
    SpeechPlaybackState.SYNTHESIZING,
    SpeechPlaybackState.PLAYING,
}
_MARKUP = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>|\[\[[^\]]+\]\]")
_CUSTOMER_MESSAGES = {
    SpeechPlaybackState.QUEUED: "已加入播报队列。",
    SpeechPlaybackState.SYNTHESIZING: "正在准备播报。",
    SpeechPlaybackState.PLAYING: "正在播报。",
    SpeechPlaybackState.COMPLETED: "播报完成。",
    SpeechPlaybackState.CANCELLED: "已停止播报。",
    SpeechPlaybackState.FAILED: "播报失败。",
}


class SpeechPlaybackService:
    """Single-writer coordinator for one robot's physical speaker."""

    def __init__(
        self,
        *,
        audio: Any,
        robot_id: str,
        device_id: str,
        site_id: str = "",
        max_queue_size: int = 32,
        max_text_chars: int = 500,
        allowed_voice_profiles: set[str] | None = None,
        artifact_dir: str | Path = "artifacts/voice/playback",
        ledger_path: str | Path | None = None,
        max_history: int = 500,
    ) -> None:
        self._audio = audio
        self._target = PlaybackTarget(
            robot_id=str(robot_id).strip(),
            device_id=str(device_id).strip(),
            site_id=str(site_id).strip(),
        )
        self._max_queue_size = max(1, int(max_queue_size))
        self._max_text_chars = max(1, int(max_text_chars))
        self._allowed_voice_profiles = set(allowed_voice_profiles or ())
        self._artifact_dir = Path(artifact_dir).expanduser()
        self._artifact_files: dict[str, SpeechAudioArtifactFile] = {}
        self._ledger_path = Path(ledger_path).expanduser() if ledger_path else None
        self._max_history = max(10, int(max_history))
        self._queue: asyncio.PriorityQueue[tuple[int, int, str]] = asyncio.PriorityQueue()
        self._jobs: dict[str, SpeechPlaybackJob] = {}
        self._requests: dict[str, SpeechPlaybackRequest] = {}
        self._deadlines: dict[str, float] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._sequence = 0
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._active_id: str | None = None
        self._closing = False
        self._load_ledger()

    async def start(self) -> None:
        if self._worker is not None and not self._worker.done():
            return
        self._closing = False
        self._worker = asyncio.create_task(self._run(), name="speech-playback")

    async def submit(self, request: SpeechPlaybackRequest) -> SpeechPlaybackJob:
        request = self._normalize_request(request)
        self._validate(request)
        request_hash = _request_hash(request)
        async with self._lock:
            if request.idempotency_key:
                replay = self._idempotency.get(request.idempotency_key)
                if replay is not None:
                    old_hash, playback_id = replay
                    if old_hash != request_hash:
                        raise SpeechPlaybackError(
                            "idempotency_conflict",
                            "Idempotency-Key was already used with a different request.",
                            status_code=409,
                        )
                    return self._jobs[playback_id]

            busy = self._active_id is not None or self._queue.qsize() > 0
            if request.queue_policy == "reject_if_busy" and busy:
                raise SpeechPlaybackError(
                    "speaker_busy",
                    "The target speaker is busy.",
                    status_code=409,
                )
            if self._queue.qsize() >= self._max_queue_size and request.priority is not SpeechPriority.SAFETY:
                raise SpeechPlaybackError(
                    "queue_full",
                    "当前播报队列已满，请稍后再试。",
                    status_code=429,
                )

            playback_id = f"spk_{uuid.uuid4().hex}"
            now = _utc_now()
            actor_id = request.actor.operator_id if request.actor is not None else ""
            job = SpeechPlaybackJob(
                playback_id=playback_id,
                state=SpeechPlaybackState.QUEUED,
                target=request.target,
                delivery=request.delivery,
                priority=request.priority,
                text_chars=len(request.text),
                request_hash=request_hash,
                idempotency_key=request.idempotency_key,
                timestamps=SpeechPlaybackTimestamps(queued_at=now),
                operator_id=actor_id,
                customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.QUEUED],
            )
            self._jobs[playback_id] = job
            self._requests[playback_id] = request
            self._deadlines[playback_id] = time.monotonic() + request.ttl_s
            if request.idempotency_key:
                self._idempotency[request.idempotency_key] = (request_hash, playback_id)
            try:
                # Durably reserve the idempotency key before the job can run.
                # A caller must never receive an accepted response for a job
                # that a process restart could forget and replay.
                self._persist_ledger()
            except Exception as exc:
                self._jobs.pop(playback_id, None)
                self._requests.pop(playback_id, None)
                self._deadlines.pop(playback_id, None)
                if request.idempotency_key:
                    self._idempotency.pop(request.idempotency_key, None)
                raise SpeechPlaybackError(
                    "playback_ledger_unavailable",
                    "Playback could not be accepted because its durable ledger is unavailable.",
                    status_code=503,
                ) from exc
            self._sequence += 1
            self._queue.put_nowait(
                (_PRIORITY_RANK[request.priority], self._sequence, playback_id)
            )
            preempt_id = self._preemptible_active(request)

        if preempt_id is not None:
            await self.cancel(
                preempt_id,
                reason=f"superseded_by:{playback_id}",
                actor=request.actor,
            )
        increment_counter("askme_voice_playback_submitted_total")
        return job

    async def status(self, playback_id: str) -> SpeechPlaybackJob:
        async with self._lock:
            job = self._jobs.get(str(playback_id).strip())
            if job is None:
                raise SpeechPlaybackError(
                    "playback_not_found",
                    "Playback was not found.",
                    status_code=404,
                )
            return job

    async def cancel(
        self,
        playback_id: str,
        *,
        reason: str,
        actor: Any | None = None,
    ) -> SpeechPlaybackJob:
        clean_id = str(playback_id).strip()
        should_stop = False
        async with self._lock:
            job = self._jobs.get(clean_id)
            if job is None:
                raise SpeechPlaybackError(
                    "playback_not_found",
                    "Playback was not found.",
                    status_code=404,
                )
            if job.state.terminal:
                return job
            actor_id = str(getattr(actor, "operator_id", "") or "").strip()
            actor_roles = set(getattr(actor, "roles", ()) or ())
            if (
                actor is not None
                and job.operator_id
                and actor_id != job.operator_id
                and not actor_roles.intersection({"supervisor", "admin", "system"})
            ):
                raise SpeechPlaybackError(
                    "cancel_not_allowed",
                    "Operators may cancel only their own playback jobs.",
                    status_code=403,
                )
            should_stop = self._active_id == clean_id
            job = replace(
                job,
                state=SpeechPlaybackState.CANCELLED,
                timestamps=replace(job.timestamps, cancelled_at=_utc_now()),
                error={"code": "cancelled", "message": _safe_reason(reason)},
                customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.CANCELLED],
            )
            self._jobs[clean_id] = job
            self._try_persist_ledger()
        if should_stop:
            stop = getattr(self._audio, "stop_immediately", None)
            if callable(stop):
                stop()
        increment_counter("askme_voice_playback_cancelled_total")
        return job

    async def artifact_file(self, playback_id: str) -> SpeechAudioArtifactFile:
        async with self._lock:
            job = self._jobs.get(str(playback_id).strip())
            if job is None:
                raise SpeechPlaybackError(
                    "playback_not_found",
                    "Playback was not found.",
                    status_code=404,
                )
            artifact = self._artifact_files.get(job.playback_id)
            if artifact is None:
                raise SpeechPlaybackError(
                    "audio_artifact_not_ready",
                    "The synthesized audio artifact is not ready.",
                    status_code=409,
                )
            return artifact

    async def shutdown(self) -> None:
        self._closing = True
        worker = self._worker
        self._worker = None
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        active_id = self._active_id
        if active_id is not None:
            await self.cancel(active_id, reason="service_shutdown")

    def _load_ledger(self) -> None:
        path = self._ledger_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("jobs") if isinstance(payload, dict) else []
            if not isinstance(records, list):
                raise ValueError("jobs must be a list")
            recovered = False
            for raw in records[-self._max_history :]:
                if not isinstance(raw, dict):
                    continue
                try:
                    job = _job_from_record(raw)
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Ignoring invalid speech playback ledger record: %r",
                        raw.get("playback_id"),
                    )
                    continue
                if not job.state.terminal:
                    job = self._failed_job(
                        job,
                        "service_restarted",
                        "Playback did not complete before the service restarted.",
                    )
                    recovered = True
                self._jobs[job.playback_id] = job
                if job.idempotency_key:
                    self._idempotency[job.idempotency_key] = (
                        job.request_hash,
                        job.playback_id,
                    )
                artifact = job.artifact or {}
                filename = str(artifact.get("filename") or "")
                artifact_path = (
                    self._artifact_dir / filename
                    if filename and Path(filename).name == filename
                    else None
                )
                if artifact_path is not None and artifact_path.is_file():
                    self._artifact_files[job.playback_id] = SpeechAudioArtifactFile(
                        path=artifact_path.resolve(),
                        filename=filename,
                        media_type=str(artifact.get("media_type") or "audio/wav"),
                        size_bytes=int(artifact.get("size_bytes") or artifact_path.stat().st_size),
                        sha256=str(artifact.get("sha256") or ""),
                    )
            if recovered:
                self._try_persist_ledger()
        except Exception:
            logger.exception("Speech playback ledger could not be loaded: %s", path)

    def _persist_ledger(self) -> None:
        path = self._ledger_path
        if path is None:
            return
        self._prune_history()
        payload = {
            "schema_version": "1.0",
            "updated_at": _utc_now(),
            "jobs": [_job_record(job) for job in self._jobs.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _try_persist_ledger(self) -> None:
        try:
            self._persist_ledger()
        except OSError:
            logger.exception("Speech playback ledger update failed")

    def _prune_history(self) -> None:
        if len(self._jobs) <= self._max_history:
            return
        ordered = sorted(
            self._jobs.values(),
            key=lambda item: item.timestamps.queued_at,
        )
        removable = [job for job in ordered if job.state.terminal]
        while len(self._jobs) > self._max_history and removable:
            job = removable.pop(0)
            self._jobs.pop(job.playback_id, None)
            self._requests.pop(job.playback_id, None)
            self._deadlines.pop(job.playback_id, None)
            self._artifact_files.pop(job.playback_id, None)
            if job.idempotency_key:
                current = self._idempotency.get(job.idempotency_key)
                if current is not None and current[1] == job.playback_id:
                    self._idempotency.pop(job.idempotency_key, None)

    def snapshot(self) -> dict[str, Any]:
        jobs = list(self._jobs.values())
        return {
            "status": "stopping" if self._closing else "ready",
            "target": {
                "robot_id": self._target.robot_id,
                "device_id": self._target.device_id,
                "site_id": self._target.site_id,
            },
            "queue_depth": sum(job.state is SpeechPlaybackState.QUEUED for job in jobs),
            "active_playback_id": self._active_id,
            "counts": {
                state.value: sum(job.state is state for job in jobs)
                for state in SpeechPlaybackState
            },
        }

    async def _run(self) -> None:
        while True:
            _rank, _sequence, playback_id = await self._queue.get()
            try:
                await self._execute(playback_id)
            finally:
                self._queue.task_done()

    async def _execute(self, playback_id: str) -> None:
        async with self._lock:
            job = self._jobs[playback_id]
            if job.state.terminal:
                return
            if time.monotonic() > self._deadlines[playback_id]:
                self._jobs[playback_id] = self._failed_job(
                    job,
                    "request_expired",
                    "Playback expired before it reached the speaker.",
                )
                self._try_persist_ledger()
                return
            self._active_id = playback_id
            request = self._requests[playback_id]
            job = replace(
                job,
                state=SpeechPlaybackState.SYNTHESIZING,
                timestamps=replace(job.timestamps, synthesis_started_at=_utc_now()),
                customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.SYNTHESIZING],
            )
            self._jobs[playback_id] = job
            self._try_persist_ledger()

        started = time.monotonic()
        controls: dict[str, Any] = {}
        try:
            controls = self._apply_audio_controls(request)
            if request.delivery is SpeechDelivery.SYNTHESIZE_ONLY:
                artifact, public_artifact = await self._synthesize(playback_id, request)
                async with self._lock:
                    current = self._jobs[playback_id]
                    if current.state.terminal:
                        return
                    self._artifact_files[playback_id] = artifact
                    self._jobs[playback_id] = replace(
                        current,
                        state=SpeechPlaybackState.COMPLETED,
                        artifact=public_artifact,
                        timestamps=replace(current.timestamps, completed_at=_utc_now()),
                        customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.COMPLETED],
                    )
                    self._try_persist_ledger()
                increment_counter("askme_voice_synthesis_completed_total")
                return
            async with self._lock:
                current = self._jobs[playback_id]
                if current.state.terminal:
                    return
                current = replace(
                    current,
                    state=SpeechPlaybackState.PLAYING,
                    timestamps=replace(current.timestamps, playback_started_at=_utc_now()),
                    customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.PLAYING],
                )
                self._jobs[playback_id] = current
                self._try_persist_ledger()

            cache_key = _cache_key(request)
            cached = False
            cached_speak = getattr(self._audio, "speak_cached_and_wait", None)
            if callable(cached_speak):
                cached = bool(await _invoke_audio(cached_speak, request.text, cache_key=cache_key))
            if not cached:
                await _invoke_audio(self._audio.speak_and_wait, request.text)

            async with self._lock:
                current = self._jobs[playback_id]
                if current.state.terminal:
                    return
                current = replace(
                    current,
                    state=SpeechPlaybackState.COMPLETED,
                    cache_hit=cached,
                    timestamps=replace(current.timestamps, completed_at=_utc_now()),
                    customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.COMPLETED],
                )
                self._jobs[playback_id] = current
                self._try_persist_ledger()
            record_metric(
                "askme_voice_playback_latency_ms",
                (time.monotonic() - started) * 1000.0,
            )
            increment_counter("askme_voice_playback_completed_total")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._lock:
                current = self._jobs[playback_id]
                if not current.state.terminal:
                    code = exc.code if isinstance(exc, SpeechPlaybackError) else "playback_failed"
                    self._jobs[playback_id] = self._failed_job(
                        current,
                        code,
                        _public_failure_message(exc),
                    )
                    self._try_persist_ledger()
            increment_counter("askme_voice_playback_failed_total")
        finally:
            self._restore_audio_controls(controls)
            async with self._lock:
                if self._active_id == playback_id:
                    self._active_id = None

    def _apply_audio_controls(self, request: SpeechPlaybackRequest) -> dict[str, Any]:
        tts = getattr(self._audio, "tts", None)
        snapshot: dict[str, Any] = {"tts": tts}
        if tts is not None:
            snapshot["speed"] = getattr(tts, "speed", None)
            snapshot["pitch"] = getattr(tts, "pitch", None)
            snapshot["volume"] = getattr(tts, "volume", None)
            profiles = getattr(tts, "voice_profiles_payload", None)
            if callable(profiles):
                profile_payload = profiles()
                if isinstance(profile_payload, dict):
                    snapshot["voice_profile_id"] = profile_payload.get("active_profile")

        if request.voice_profile_id:
            set_profile = getattr(tts, "set_voice_profile_payload", None)
            if not callable(set_profile):
                raise SpeechPlaybackError(
                    "voice_profile_not_supported",
                    "This TTS provider does not support controlled voice profiles.",
                    status_code=422,
                )
            result = set_profile({"profile_id": request.voice_profile_id})
            if not isinstance(result, dict) or not result.get("updated"):
                raise SpeechPlaybackError(
                    "voice_profile_not_allowed",
                    "The requested voice profile is unavailable.",
                    status_code=422,
                )

        self._set_optional_control("set_speed", request.speed)
        self._set_optional_control("set_pitch", request.pitch)
        self._set_optional_control("set_volume", request.volume)
        return snapshot

    def _set_optional_control(self, name: str, value: float | None) -> None:
        if value is None:
            return
        setter = getattr(self._audio, name, None)
        if not callable(setter):
            setter = getattr(getattr(self._audio, "tts", None), name, None)
        if not callable(setter):
            raise SpeechPlaybackError(
                "audio_control_not_supported",
                f"This TTS provider does not support {name.removeprefix('set_')}.",
                status_code=422,
            )
        setter(value)

    def _restore_audio_controls(self, snapshot: dict[str, Any]) -> None:
        if not snapshot:
            return
        tts = snapshot.get("tts")
        try:
            old_profile = snapshot.get("voice_profile_id")
            set_profile = getattr(tts, "set_voice_profile_payload", None)
            if old_profile and callable(set_profile):
                set_profile({"profile_id": old_profile})
            for key, setter_name in (
                ("speed", "set_speed"),
                ("pitch", "set_pitch"),
                ("volume", "set_volume"),
            ):
                value = snapshot.get(key)
                setter = getattr(tts, setter_name, None)
                if value is not None and callable(setter):
                    setter(value)
        except Exception:
            logger.exception("Failed to restore per-playback TTS controls")

    async def _synthesize(
        self,
        playback_id: str,
        request: SpeechPlaybackRequest,
    ) -> tuple[SpeechAudioArtifactFile, dict[str, Any]]:
        tts = getattr(self._audio, "tts", None)
        prime = getattr(tts, "prime_cached_phrase", None)
        cached_pcm = getattr(tts, "cached_phrase_pcm", None)
        if not callable(prime) or not callable(cached_pcm):
            raise SpeechPlaybackError(
                "synthesize_only_unavailable",
                "This audio frontend cannot return synthesized PCM.",
                status_code=503,
            )
        cache_key = _cache_key(request)
        result = await asyncio.to_thread(prime, request.text, cache_key=cache_key)
        if not isinstance(result, dict) or not result.get("cached"):
            reason = result.get("reason") if isinstance(result, dict) else "synthesis_failed"
            raise SpeechPlaybackError(
                "synthesis_failed",
                f"Speech synthesis failed: {reason or 'empty_audio'}.",
                status_code=503,
            )
        pcm = await asyncio.to_thread(
            cached_pcm,
            request.text,
            cache_key=cache_key,
        )
        if not isinstance(pcm, tuple) or len(pcm) != 2:
            raise SpeechPlaybackError(
                "synthesis_empty",
                "Speech synthesis returned no audio.",
                status_code=503,
            )
        samples, sample_rate = pcm
        filename = f"{playback_id}.wav"
        artifact = await asyncio.to_thread(
            _write_wav_artifact,
            self._artifact_dir / filename,
            filename,
            samples,
            int(sample_rate),
        )
        sample_count = len(samples)
        public = {
            "format": "wav",
            "media_type": artifact.media_type,
            "filename": artifact.filename,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "sample_rate": int(sample_rate),
            "duration_ms": round(sample_count * 1000.0 / max(1, int(sample_rate))),
            "download_url": f"/api/voice/playbacks/{playback_id}/audio",
        }
        return artifact, public

    def _normalize_request(self, request: SpeechPlaybackRequest) -> SpeechPlaybackRequest:
        target = request.target
        if not target.site_id and self._target.site_id:
            target = replace(target, site_id=self._target.site_id)
        return replace(request, target=target)

    def _validate(self, request: SpeechPlaybackRequest) -> None:
        if not self._target.robot_id or not self._target.device_id:
            raise SpeechPlaybackError(
                "playback_not_configured",
                "Local robot_id and device_id are not configured.",
                status_code=503,
            )
        if not request.target.robot_id or not request.target.device_id:
            raise SpeechPlaybackError(
                "target_required",
                "robot_id and device_id are required.",
                status_code=422,
            )
        if request.target != self._target:
            raise SpeechPlaybackError(
                "target_not_local",
                "The requested robot/device is not controlled by this service.",
                status_code=403,
            )
        if request.target.channel != "speaker":
            raise SpeechPlaybackError(
                "invalid_channel",
                "Only the controlled speaker channel is supported.",
                status_code=422,
            )
        if not request.text or request.text != request.text.strip():
            raise SpeechPlaybackError(
                "text_not_speakable",
                "Text must be non-empty and may not contain leading or trailing whitespace.",
                status_code=422,
            )
        if len(request.text) > self._max_text_chars:
            raise SpeechPlaybackError(
                "text_too_long",
                f"Text exceeds the {self._max_text_chars} character limit.",
                status_code=413,
            )
        if any(ord(char) < 32 or ord(char) == 127 for char in request.text):
            raise SpeechPlaybackError(
                "control_characters_rejected",
                "Control characters are not allowed.",
                status_code=422,
            )
        if _MARKUP.search(request.text):
            raise SpeechPlaybackError(
                "verbatim_text_not_plain",
                "SSML, HTML, and internal control markers are not allowed.",
                status_code=422,
            )
        if request.queue_policy not in {"enqueue", "reject_if_busy", "replace_noncritical"}:
            raise SpeechPlaybackError("invalid_queue_policy", "Unknown queue policy.", status_code=422)
        if request.queue_policy == "replace_noncritical" and request.priority not in {
            SpeechPriority.SAFETY,
            SpeechPriority.OPERATOR,
        }:
            raise SpeechPlaybackError(
                "override_not_allowed",
                "Only safety or operator priority may replace playback.",
                status_code=403,
            )
        if request.speed is not None and not 0.75 <= request.speed <= 1.5:
            raise SpeechPlaybackError("invalid_speed", "speed must be between 0.75 and 1.5.", status_code=422)
        if request.pitch is not None and not -12.0 <= request.pitch <= 12.0:
            raise SpeechPlaybackError("invalid_pitch", "pitch must be between -12 and 12 semitones.", status_code=422)
        if request.volume is not None and not 0.05 <= request.volume <= 1.0:
            raise SpeechPlaybackError("invalid_volume", "volume must be between 0.05 and 1.", status_code=422)
        if not 1.0 <= request.ttl_s <= 300.0:
            raise SpeechPlaybackError("invalid_ttl", "ttl_s must be between 1 and 300 seconds.", status_code=422)
        if (
            request.voice_profile_id
            and self._allowed_voice_profiles
            and request.voice_profile_id not in self._allowed_voice_profiles
        ):
            raise SpeechPlaybackError(
                "voice_profile_not_allowed",
                "The requested voice profile is not allowed.",
                status_code=422,
            )

    def _preemptible_active(self, request: SpeechPlaybackRequest) -> str | None:
        if request.queue_policy != "replace_noncritical" or self._active_id is None:
            return None
        active = self._jobs.get(self._active_id)
        if active is None or active.state not in _BUSY_STATES:
            return None
        if _PRIORITY_RANK[request.priority] < _PRIORITY_RANK[active.priority]:
            return active.playback_id
        return None

    @staticmethod
    def _failed_job(job: SpeechPlaybackJob, code: str, message: str) -> SpeechPlaybackJob:
        return replace(
            job,
            state=SpeechPlaybackState.FAILED,
            timestamps=replace(job.timestamps, failed_at=_utc_now()),
            error={"code": code, "message": message},
            customer_message=_CUSTOMER_MESSAGES[SpeechPlaybackState.FAILED],
        )


def _job_record(job: SpeechPlaybackJob) -> dict[str, Any]:
    payload = job.to_payload()
    payload["request_hash"] = job.request_hash
    payload["operator_id"] = job.operator_id
    return payload


def _job_from_record(raw: dict[str, Any]) -> SpeechPlaybackJob:
    target_raw = raw.get("target")
    timestamps_raw = raw.get("timestamps")
    if not isinstance(target_raw, dict) or not isinstance(timestamps_raw, dict):
        raise TypeError("target and timestamps must be objects")
    artifact = raw.get("artifact")
    error = raw.get("error")
    return SpeechPlaybackJob(
        playback_id=str(raw["playback_id"]),
        state=SpeechPlaybackState(str(raw["state"])),
        target=PlaybackTarget(
            robot_id=str(target_raw["robot_id"]),
            device_id=str(target_raw["device_id"]),
            site_id=str(target_raw.get("site_id") or ""),
            channel=str(target_raw.get("channel") or "speaker"),
        ),
        delivery=SpeechDelivery(str(raw["delivery"])),
        priority=SpeechPriority(str(raw["priority"])),
        text_chars=int(raw["text_chars"]),
        request_hash=str(raw["request_hash"]),
        idempotency_key=str(raw.get("idempotency_key") or ""),
        timestamps=SpeechPlaybackTimestamps(
            queued_at=str(timestamps_raw["queued_at"]),
            synthesis_started_at=_optional_timestamp(
                timestamps_raw, "synthesis_started_at"
            ),
            playback_started_at=_optional_timestamp(
                timestamps_raw, "playback_started_at"
            ),
            completed_at=_optional_timestamp(timestamps_raw, "completed_at"),
            cancelled_at=_optional_timestamp(timestamps_raw, "cancelled_at"),
            failed_at=_optional_timestamp(timestamps_raw, "failed_at"),
        ),
        operator_id=str(raw.get("operator_id") or ""),
        cache_hit=bool(raw.get("cache_hit", False)),
        artifact=dict(artifact) if isinstance(artifact, dict) else None,
        error=dict(error) if isinstance(error, dict) else None,
        customer_message=str(raw.get("customer_message") or ""),
    )


def _optional_timestamp(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    return str(value) if value else None


def _request_hash(request: SpeechPlaybackRequest) -> str:
    payload = {
        "text": request.text,
        "target": {
            "robot_id": request.target.robot_id,
            "device_id": request.target.device_id,
            "site_id": request.target.site_id,
            "channel": request.target.channel,
        },
        "delivery": request.delivery.value,
        "priority": request.priority.value,
        "queue_policy": request.queue_policy,
        "voice_profile_id": request.voice_profile_id,
        "speed": request.speed,
        "pitch": request.pitch,
        "volume": request.volume,
        "ttl_s": request.ttl_s,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_key(request: SpeechPlaybackRequest) -> str:
    value = "|".join(
        (
            request.text,
            request.voice_profile_id,
            str(request.speed),
            str(request.pitch),
            str(request.volume),
        )
    )
    return f"product:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_reason(reason: str) -> str:
    clean = " ".join(str(reason or "operator_cancelled").split())
    return clean[:160]


def _public_failure_message(exc: Exception) -> str:
    if isinstance(exc, SpeechPlaybackError):
        return str(exc)
    lowered = str(exc).lower()
    if "busy" in lowered:
        return "The audio device is busy."
    if "timeout" in lowered:
        return "Speech synthesis or playback timed out."
    return "Speech synthesis or playback failed."


def _write_wav_artifact(
    path: Path,
    filename: str,
    samples: Any,
    sample_rate: int,
) -> SpeechAudioArtifactFile:
    if sample_rate <= 0:
        raise SpeechPlaybackError(
            "invalid_sample_rate",
            "Synthesized audio has an invalid sample rate.",
            status_code=503,
        )
    pcm = array(
        "h",
        (
            int(max(-1.0, min(1.0, float(sample))) * 32767.0)
            for sample in samples
        ),
    )
    if not pcm:
        raise SpeechPlaybackError(
            "synthesis_empty",
            "Speech synthesis returned no samples.",
            status_code=503,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    data = path.read_bytes()
    return SpeechAudioArtifactFile(
        path=path.resolve(),
        filename=filename,
        media_type="audio/wav",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


async def _invoke_audio(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(callable_obj):
        return await callable_obj(*args, **kwargs)
    return await asyncio.to_thread(callable_obj, *args, **kwargs)
