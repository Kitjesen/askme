"""AlertDispatcher — multi-channel alert delivery for proactive robot notifications.

Channels:
    - voice (TTS)        — speak via AudioAgent
    - webhook            — POST JSON to arbitrary URL (dashboard, Slack, custom)
    - wecom  (企业微信)   — send to group chat via bot webhook
    - dingtalk (钉钉)     — send to group chat via bot webhook
    - feishu (飞书)       — send to group chat via bot webhook
    - log                — always on, writes to Python logger

Routing by severity:
    - info    → voice + log
    - warning → voice + webhook + log
    - error   → voice + webhook + wecom/dingtalk/feishu + log

Config (under ``proactive.alerts``):
    alerts:
      webhook_url: "https://your-dashboard.com/api/alerts"
      wecom_webhook: "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
      dingtalk_webhook: "https://oapi.dingtalk.com/robot/send?access_token=xxx"
      feishu_webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
      severity_routes:
        info: ["voice", "log"]
        warning: ["voice", "webhook", "log"]
        error: ["voice", "webhook", "wecom", "log"]
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Protocol
from urllib import error, request

logger = logging.getLogger(__name__)

_DEFAULT_ROUTES: dict[str, list[str]] = {
    "info": ["voice", "log"],
    "warning": ["voice", "webhook", "log"],
    "error": ["voice", "webhook", "wecom", "dingtalk", "feishu", "log"],
}


class VoiceSpeaker(Protocol):
    """Minimal interface for TTS output."""

    @property
    def is_busy(self) -> bool: ...

    def start_playback(self) -> None: ...

    def speak(self, text: str) -> None: ...

    def wait_speaking_done(self) -> None: ...

    def stop_playback(self) -> None: ...


class AlertDispatcher:
    """Routes alert messages to multiple notification channels."""

    def __init__(
        self,
        *,
        voice: VoiceSpeaker | None = None,
        config: dict[str, Any] | None = None,
        robot_id: str | None = None,
        robot_name: str = "Thunder",
    ) -> None:
        cfg = config or {}
        self._voice = voice
        self._robot_id = robot_id
        self._robot_name = robot_name

        # Channel URLs
        self._webhook_url: str | None = cfg.get("webhook_url")
        self._wecom_webhook: str | None = cfg.get("wecom_webhook")
        self._dingtalk_webhook: str | None = cfg.get("dingtalk_webhook")
        self._feishu_webhook: str | None = cfg.get("feishu_webhook")

        # Severity → channel routing
        self._routes: dict[str, list[str]] = cfg.get("severity_routes", _DEFAULT_ROUTES)

        # Rate limiting — seed with -inf so the FIRST dispatch always passes
        # the cooldown check.  ``time.monotonic()`` is not epoch-based; its
        # starting value depends on process/container uptime and can be
        # smaller than ``voice_cooldown``, which would otherwise cause the
        # very first voice alert to be suppressed.
        self._last_voice_time: float = float("-inf")
        self._voice_cooldown: float = float(cfg.get("voice_cooldown", 10))

    def dispatch(
        self,
        message: str,
        *,
        severity: str = "info",
        topic: str = "",
        payload: dict[str, Any] | None = None,
    ) -> list[str]:
        """Send alert to channels determined by severity. Returns list of channels sent to."""
        channels = self._routes.get(severity, self._routes.get("info", ["log"]))
        sent: list[str] = []
        image_path = (payload or {}).get("image_path")

        for channel in channels:
            try:
                if channel == "voice":
                    if self._send_voice(message):
                        sent.append("voice")
                elif channel == "webhook":
                    if self._send_webhook(message, severity, topic, payload):
                        sent.append("webhook")
                elif channel == "wecom":
                    if self._send_wecom(message, severity, image_path=image_path):
                        sent.append("wecom")
                elif channel == "dingtalk":
                    if self._send_dingtalk(message, severity):
                        sent.append("dingtalk")
                elif channel == "feishu":
                    if self._send_feishu(message, severity):
                        sent.append("feishu")
                elif channel == "log":
                    self._send_log(message, severity, topic)
                    sent.append("log")
            except Exception as exc:
                logger.warning("[Alert] Channel %s failed: %s", channel, exc)

        return sent

    # ── Voice ──

    def _send_voice(self, message: str) -> bool:
        if not self._voice:
            return False
        now = time.monotonic()
        if now - self._last_voice_time < self._voice_cooldown:
            logger.debug("[Alert] Voice suppressed by cooldown")
            return False
        if self._voice.is_busy:
            logger.debug("[Alert] Voice busy, skipping")
            return False
        self._voice.start_playback()
        self._voice.speak(message)
        self._voice.wait_speaking_done()
        self._voice.stop_playback()
        self._last_voice_time = time.monotonic()
        return True

    # ── Webhook (generic JSON POST) ──

    def _send_webhook(
        self,
        message: str,
        severity: str,
        topic: str,
        payload: dict[str, Any] | None,
    ) -> bool:
        if not self._webhook_url:
            return False
        body: dict[str, Any] = {
            "robot_id": self._robot_id,
            "robot_name": self._robot_name,
            "severity": severity,
            "topic": topic,
            "message": message,
            "payload": payload or {},
            "timestamp": time.time(),
        }
        # Attach image as base64 if available
        image_path = (payload or {}).get("image_path")
        if image_path:
            b64 = self._read_image_base64(image_path)
            if b64:
                body["image_base64"] = b64
        return self._post_json(self._webhook_url, body)

    # ── 企业微信 (WeCom) ──

    def _send_wecom(self, message: str, severity: str, image_path: str | None = None) -> bool:
        if not self._wecom_webhook:
            return False
        icon = {"info": "📋", "warning": "⚠️", "error": "🚨"}.get(severity, "📋")

        # Send text first
        text_body = {
            "msgtype": "markdown",
            "markdown": {
                "content": (
                    f"{icon} **{self._robot_name} 告警**\n"
                    f"> 级别: {severity}\n"
                    f"> {message}"
                ),
            },
        }
        ok = self._post_json(self._wecom_webhook, text_body)

        # Then send image if available (WeCom supports base64 image message)
        if image_path:
            b64 = self._read_image_base64(image_path)
            md5 = self._file_md5(image_path)
            if b64 and md5:
                img_body = {
                    "msgtype": "image",
                    "image": {"base64": b64, "md5": md5},
                }
                self._post_json(self._wecom_webhook, img_body)

        return ok

    # ── 钉钉 (DingTalk) ──

    def _send_dingtalk(self, message: str, severity: str) -> bool:
        if not self._dingtalk_webhook:
            return False
        icon = {"info": "📋", "warning": "⚠️", "error": "🚨"}.get(severity, "📋")
        body = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"{self._robot_name} 告警",
                "text": (
                    f"### {icon} {self._robot_name} 告警\n\n"
                    f"**级别**: {severity}\n\n"
                    f"{message}"
                ),
            },
        }
        return self._post_json(self._dingtalk_webhook, body)

    # ── 飞书 (Feishu / Lark) ──

    def _send_feishu(self, message: str, severity: str) -> bool:
        if not self._feishu_webhook:
            return False
        icon = {"info": "📋", "warning": "⚠️", "error": "🚨"}.get(severity, "📋")
        body = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": f"{icon} {self._robot_name} 告警"},
                    "template": {"warning": "orange", "error": "red"}.get(severity, "blue"),
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": f"**级别**: {severity}\n{message}",
                    },
                ],
            },
        }
        return self._post_json(self._feishu_webhook, body)

    # ── Log ──

    def _send_log(self, message: str, severity: str, topic: str) -> None:
        level = {"error": logging.ERROR, "warning": logging.WARNING}.get(severity, logging.INFO)
        logger.log(level, "[Alert] [%s] %s — %s", severity, topic, message)

    # ── Image helpers ──

    @staticmethod
    def _read_image_base64(path: str) -> str | None:
        """Read an image file and return base64-encoded string."""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except OSError:
            return None

    @staticmethod
    def _file_md5(path: str) -> str | None:
        """Compute MD5 hash of a file (required by WeCom image API)."""
        import hashlib
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except OSError:
            return None

    # ── HTTP helper ──

    @staticmethod
    def _post_json(url: str, body: dict[str, Any]) -> bool:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5) as resp:
                return resp.status < 400
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            logger.warning("[Alert] POST to %s failed: %s", url[:60], exc)
            return False
