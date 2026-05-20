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
import hashlib
import hmac
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request
from urllib.parse import quote_plus

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
        robot_name: str = "现场机器人",
    ) -> None:
        cfg = config or {}
        self._voice = voice
        self._robot_id = robot_id
        self._robot_name = robot_name

        # Channel URLs
        self._webhook_url: str | None = cfg.get("webhook_url")
        self._wecom_webhook: str | None = cfg.get("wecom_webhook")
        self._dingtalk_webhook: str | None = cfg.get("dingtalk_webhook")
        self._dingtalk_secret: str | None = cfg.get("dingtalk_secret")
        self._feishu_webhook: str | None = cfg.get("feishu_webhook")
        self._incident_archive_path: str | None = cfg.get("incident_archive_path")

        # Severity → channel routing
        self._routes: dict[str, list[str]] = cfg.get("severity_routes", _DEFAULT_ROUTES)
        self._last_delivery_report: list[dict[str, Any]] = []

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
        delivery_report: list[dict[str, Any]] = []
        image_path = (payload or {}).get("image_path")

        for channel in channels:
            try:
                delivery: bool | dict[str, Any] = False
                if channel == "voice":
                    delivery = self._send_voice(message)
                elif channel == "webhook":
                    delivery = self._send_webhook(message, severity, topic, payload)
                elif channel == "wecom":
                    delivery = self._send_wecom(message, severity, image_path=image_path)
                elif channel == "dingtalk":
                    delivery = self._send_dingtalk(
                        message,
                        severity,
                        dingtalk_text=(payload or {}).get("dingtalk_message"),
                    )
                elif channel == "feishu":
                    delivery = self._send_feishu(message, severity)
                elif channel == "log":
                    self._send_log(message, severity, topic)
                    delivery = True
                else:
                    delivery_report.append(
                        {"channel": channel, "status": "skipped", "reason": "unknown_channel"}
                    )
                    continue
                delivered = self._delivery_ok(delivery)
                report_item = self._delivery_report_item(channel, delivery, delivered)
                if delivered:
                    sent.append(channel)
                    delivery_report.append(report_item)
                else:
                    delivery_report.append(report_item)
            except Exception as exc:
                logger.warning("[Alert] Channel %s failed: %s", channel, exc)
                delivery_report.append(
                    {"channel": channel, "status": "failed", "reason": str(exc)}
                )

        self._last_delivery_report = delivery_report
        self._archive_incident(
            message,
            severity=severity,
            topic=topic,
            payload=payload,
            channels=sent,
            delivery_report=delivery_report,
        )
        return sent

    @property
    def last_delivery_report(self) -> list[dict[str, Any]]:
        """Detailed channel delivery status for the most recent dispatch."""

        return [dict(item) for item in self._last_delivery_report]

    @staticmethod
    def _delivery_ok(delivery: bool | dict[str, Any]) -> bool:
        if isinstance(delivery, dict):
            return bool(delivery.get("ok"))
        return bool(delivery)

    @staticmethod
    def _delivery_report_item(
        channel: str,
        delivery: bool | dict[str, Any],
        delivered: bool,
    ) -> dict[str, Any]:
        if not isinstance(delivery, dict):
            return {
                "channel": channel,
                "status": "sent" if delivered else "not_sent",
                "reason": "" if delivered else "not_configured_or_failed",
            }
        reason = str(delivery.get("reason") or ("" if delivered else "not_configured_or_failed"))
        item: dict[str, Any] = {
            "channel": channel,
            "status": "sent" if delivered else "not_sent",
            "reason": reason,
        }
        for key in ("http_status", "response_excerpt", "error_type"):
            value = delivery.get(key)
            if value not in (None, ""):
                item[key] = value
        return item

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
    ) -> bool | dict[str, Any]:
        if not self._webhook_url:
            return {"ok": False, "reason": "not_configured"}
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
        return self._post_json(self._webhook_url, body, return_result=True)

    # ── 企业微信 (WeCom) ──

    def _send_wecom(
        self, message: str, severity: str, image_path: str | None = None
    ) -> bool | dict[str, Any]:
        if not self._wecom_webhook:
            return {"ok": False, "reason": "not_configured"}
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
        result = self._post_json(self._wecom_webhook, text_body, return_result=True)

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

        return result

    # ── 钉钉 (DingTalk) ──

    def _send_dingtalk(
        self,
        message: str,
        severity: str,
        *,
        dingtalk_text: str | None = None,
    ) -> bool | dict[str, Any]:
        if not self._dingtalk_webhook:
            return {"ok": False, "reason": "not_configured"}
        icon = {"info": "📋", "warning": "⚠️", "error": "🚨"}.get(severity, "📋")
        text = dingtalk_text or message
        body = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"{self._robot_name} 告警",
                "text": (
                    f"### {icon} {self._robot_name} 告警\n\n"
                    f"**级别**: {severity}\n\n"
                    f"{text}"
                ),
            },
        }
        url = self._signed_dingtalk_url(self._dingtalk_webhook)
        return self._post_json(url, body, return_result=True)

    # ── 飞书 (Feishu / Lark) ──

    def _send_feishu(self, message: str, severity: str) -> bool | dict[str, Any]:
        if not self._feishu_webhook:
            return {"ok": False, "reason": "not_configured"}
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
        return self._post_json(self._feishu_webhook, body, return_result=True)

    # ── Log ──

    def _send_log(self, message: str, severity: str, topic: str) -> None:
        level = {"error": logging.ERROR, "warning": logging.WARNING}.get(severity, logging.INFO)
        logger.log(level, "[Alert] [%s] %s — %s", severity, topic, message)

    # ── Image helpers ──

    def _archive_incident(
        self,
        message: str,
        *,
        severity: str,
        topic: str,
        payload: dict[str, Any] | None,
        channels: list[str],
        delivery_report: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._incident_archive_path:
            return
        record = {
            "event_id": str((payload or {}).get("event_id") or uuid.uuid4()),
            "timestamp": time.time(),
            "robot_id": self._robot_id,
            "robot_name": self._robot_name,
            "severity": severity,
            "topic": topic,
            "message": message,
            "payload": payload or {},
            "channels": channels,
            "delivery_report": delivery_report or [],
        }
        try:
            path = Path(self._incident_archive_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.warning("[Alert] Incident archive write failed: %s", exc)

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

    def _signed_dingtalk_url(self, url: str) -> str:
        """Append DingTalk robot signature params when a secret is configured."""
        if not self._dingtalk_secret:
            return url
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self._dingtalk_secret}"
        digest = hmac.new(
            self._dingtalk_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = quote_plus(base64.b64encode(digest).decode("utf-8"))
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}timestamp={timestamp}&sign={sign}"

    # ── HTTP helper ──

    @staticmethod
    def _post_json(
        url: str,
        body: dict[str, Any],
        *,
        return_result: bool = False,
    ) -> bool | dict[str, Any]:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5) as resp:
                status = int(getattr(resp, "status", 0) or 0)
                excerpt = AlertDispatcher._read_response_excerpt(resp)
                ok = status < 400
                if return_result:
                    return {
                        "ok": ok,
                        "http_status": status,
                        "reason": "" if ok else f"http_{status}",
                        "response_excerpt": excerpt,
                    }
                return ok
        except error.HTTPError as exc:
            excerpt = AlertDispatcher._read_response_excerpt(exc)
            logger.warning("[Alert] POST to %s failed: %s", url[:60], exc)
            result = {
                "ok": False,
                "http_status": int(getattr(exc, "code", 0) or 0),
                "reason": f"http_{getattr(exc, 'code', 'error')}",
                "response_excerpt": excerpt,
                "error_type": type(exc).__name__,
            }
            return result if return_result else False
        except (error.URLError, TimeoutError) as exc:
            logger.warning("[Alert] POST to %s failed: %s", url[:60], exc)
            result = {
                "ok": False,
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
            return result if return_result else False

    @staticmethod
    def _read_response_excerpt(resp: Any, *, limit: int = 300) -> str:
        read = getattr(resp, "read", None)
        if not callable(read):
            return ""
        try:
            raw = read(limit)
        except Exception:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)[:limit]
