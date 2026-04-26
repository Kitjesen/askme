"""TelegramModule — Telegram Bot interface for askme.

Runs a Telegram bot alongside the existing voice/text loops.
Routes incoming messages through BrainPipeline and sends replies back.

Supports:
- Text messages → BrainPipeline.process(text) → reply
- Voice messages (.ogg) → sherpa-onnx offline ASR → pipeline → reply
- /list command → query LingTu TaggedLocations and list known places
- /cancel command → send cancel signal to LingTu navigation
- User allowlist via config.platforms.telegram.allowed_users

Configuration (config.yaml):
  platforms:
    telegram:
      enabled: true
      token_env: TELEGRAM_BOT_TOKEN   # env var name holding the bot token
      allowed_users: []               # empty = anyone; list of int user_ids = whitelist
      voice_transcribe: true          # transcribe voice messages via ASR
      lingtu_url: ""                  # override NAV_GATEWAY_URL for /list and /cancel
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from askme.pipeline.brain_pipeline import BrainPipeline
from askme.runtime.module import In, Module, ModuleRegistry

logger = logging.getLogger(__name__)


class TelegramModule(Module):
    """Telegram Bot interface — routes messages through BrainPipeline."""

    name = "telegram"
    depends_on = ("pipeline",)
    provides = ()

    # Auto-wired from PipelineModule
    pipeline: In[BrainPipeline]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        tg_cfg = cfg.get("platforms", {}).get("telegram", {})
        self._enabled = tg_cfg.get("enabled", False)

        # Resolve bot token
        token_env = tg_cfg.get("token_env", "TELEGRAM_BOT_TOKEN")
        self._token = os.environ.get(token_env, "")

        self._allowed_users: list[int] = [
            int(u) for u in tg_cfg.get("allowed_users", [])
        ]
        self._voice_transcribe: bool = tg_cfg.get("voice_transcribe", True)
        self._lingtu_url: str = tg_cfg.get("lingtu_url", "") or os.environ.get(
            "NAV_GATEWAY_URL", ""
        )

        # Pipeline wired by runtime; may be None if pipeline module absent
        pipeline_mod = self.pipeline
        self._pipeline: BrainPipeline | None = getattr(pipeline_mod, "brain_pipeline", None)

        self._app: Any = None  # telegram.ext.Application, built in start()
        self._poll_task: asyncio.Task[None] | None = None

        if self._enabled and not self._token:
            logger.warning(
                "TelegramModule: enabled=true but %s is not set — bot will not start",
                token_env,
            )
            self._enabled = False

        logger.info("TelegramModule: built (enabled=%s)", self._enabled)

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if not self._enabled:
            return

        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "TelegramModule: python-telegram-bot not installed. "
                "Run: pip install 'python-telegram-bot>=20.0'"
            )
            return

        self._app = Application.builder().token(self._token).build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("list", self._cmd_list))
        self._app.add_handler(CommandHandler("cancel", self._cmd_cancel))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        if self._voice_transcribe:
            self._app.add_handler(
                MessageHandler(filters.VOICE, self._handle_voice)
            )

        # Initialize and start polling in background
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        logger.info("TelegramModule: polling started")

    async def stop(self) -> None:
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as exc:
            logger.warning("TelegramModule stop error: %s", exc)
        logger.info("TelegramModule: stopped")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._enabled else "disabled",
            "bot_running": self._app is not None,
        }

    # ------------------------------------------------------------------
    # Auth guard
    # ------------------------------------------------------------------

    def _is_allowed(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    # ------------------------------------------------------------------
    # Telegram command handlers
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Any, context: Any) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        await update.message.reply_text(
            "你好！我是机器人助手。\n"
            "发送文字指令（如「去厨房」）即可控制导航。\n"
            "命令：/list 查看已知地点  /cancel 取消当前导航"
        )

    async def _cmd_list(self, update: Any, context: Any) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        locations = await self._fetch_lingtu_locations()
        if not locations:
            await update.message.reply_text("暂无已标注地点，或 LingTu 未运行。")
            return
        lines = "\n".join(f"• {name}" for name in sorted(locations))
        await update.message.reply_text(f"已知地点：\n{lines}")

    async def _cmd_cancel(self, update: Any, context: Any) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        result = await self._post_lingtu("/api/v1/stop", {})
        if result is None:
            await update.message.reply_text("LingTu 不可达，取消失败。")
        else:
            await update.message.reply_text("已发送取消导航指令。")

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_text(self, update: Any, context: Any) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        text = (update.message.text or "").strip()
        if not text:
            return
        await self._process_and_reply(update, text)

    async def _handle_voice(self, update: Any, context: Any) -> None:
        if not self._is_allowed(update.effective_user.id):
            return
        await update.message.reply_text("正在识别语音…")
        text = await self._transcribe_voice(update, context)
        if not text:
            await update.message.reply_text("语音识别失败，请直接发文字。")
            return
        await update.message.reply_text(f"识别结果：{text}")
        await self._process_and_reply(update, text)

    # ------------------------------------------------------------------
    # Core: feed text through BrainPipeline
    # ------------------------------------------------------------------

    async def _process_and_reply(self, update: Any, text: str) -> None:
        if self._pipeline is None:
            await update.message.reply_text("[错误] 大脑管线未就绪")
            return
        try:
            response = await self._pipeline.process(text, source="telegram")
            reply = response if isinstance(response, str) else str(response)
        except Exception as exc:
            logger.exception("TelegramModule: pipeline error")
            reply = f"[错误] {exc}"
        await update.message.reply_text(reply or "（无回复）")

    # ------------------------------------------------------------------
    # Voice: OGG → WAV → sherpa-onnx offline decode
    # ------------------------------------------------------------------

    async def _transcribe_voice(self, update: Any, context: Any) -> str:
        """Download Telegram voice file and transcribe with sherpa-onnx."""
        try:
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            with tempfile.TemporaryDirectory() as tmp:
                ogg_path = Path(tmp) / "voice.ogg"
                wav_path = Path(tmp) / "voice.wav"
                await voice_file.download_to_drive(str(ogg_path))

                # Convert OGG → 16kHz mono WAV via ffmpeg
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", str(ogg_path),
                    "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                if not wav_path.exists():
                    logger.warning("TelegramModule: ffmpeg conversion failed")
                    return ""

                return self._asr_decode_wav(wav_path)
        except Exception as exc:
            logger.warning("TelegramModule: voice transcription error: %s", exc)
            return ""

    def _asr_decode_wav(self, wav_path: Path) -> str:
        """Decode a WAV file using sherpa-onnx offline recognizer."""
        try:
            import numpy as np
            import sherpa_onnx
            import soundfile as sf
        except ImportError:
            logger.warning("TelegramModule: sherpa_onnx or soundfile not installed")
            return ""

        # Reuse the streaming ASR config from environment / askme convention
        encoder = os.environ.get(
            "ASR_ENCODER_MODEL",
            "models/asr/encoder.onnx",
        )
        decoder = os.environ.get("ASR_DECODER_MODEL", "models/asr/decoder.onnx")
        joiner = os.environ.get("ASR_JOINER_MODEL", "models/asr/joiner.onnx")
        tokens = os.environ.get("ASR_TOKENS", "models/asr/tokens.txt")

        if not all(os.path.exists(p) for p in [encoder, decoder, joiner, tokens]):
            logger.warning("TelegramModule: ASR model files not found, skipping transcription")
            return ""

        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=2,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
            )
            samples, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, samples)
            recognizer.decode_stream(stream)
            return stream.result.text.strip()
        except Exception as exc:
            logger.warning("TelegramModule: ASR decode error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # LingTu REST helpers
    # ------------------------------------------------------------------

    async def _fetch_lingtu_locations(self) -> list[str]:
        """GET LingTu POI list and return location names."""
        if not self._lingtu_url:
            return []
        result = await self._get_lingtu("/api/v1/locations")
        if result is None:
            return []
        # Accept: {"locations": ["厨房", "客厅"]} or {"pois": [...]}
        names: list[str] = result.get("locations", result.get("pois", []))
        if names and isinstance(names[0], dict):
            names = [n.get("name", "") for n in names if n.get("name")]
        return [str(n) for n in names if n]

    async def _get_lingtu(self, path: str) -> dict | None:
        import json
        import urllib.error
        import urllib.request

        url = f"{self._lingtu_url.rstrip('/')}{path}"
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(url, timeout=5).read(),
            )
            return json.loads(raw.decode())
        except Exception as exc:
            logger.warning("TelegramModule: GET %s failed: %s", path, exc)
            return None

    async def _post_lingtu(self, path: str, body: dict) -> dict | None:
        import json
        import urllib.error
        import urllib.request

        if not self._lingtu_url:
            return None
        url = f"{self._lingtu_url.rstrip('/')}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=5).read(),
            )
            return json.loads(raw.decode())
        except Exception as exc:
            logger.warning("TelegramModule: POST %s failed: %s", path, exc)
            return None
