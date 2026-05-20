"""Voice profile catalog for customer-facing speech styles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

VOICE_PROFILE_ALIASES: dict[str, str] = {
    "patrol_notice": "patrol_default",
    "mission_control": "patrol_default",
    "visitor_service": "visitor_friendly",
    "service_notice": "visitor_friendly",
    "wayfinding_prompt": "visitor_friendly",
    "security_alert": "security_clear",
    "night_photo": "night_quiet",
    "emergency_alert": "emergency_short",
    "fire_alarm": "emergency_short",
    "night_security": "night_quiet",
    "cleaning_notice": "cleaning_soft",
    "trash_notice": "cleaning_soft",
    "operations_notice": "operations_calm",
    "crowd_notice": "crowd_clear",
    "escort_guide": "guide_leading",
    "robot_fault": "fault_urgent",
    "fault_alarm": "fault_urgent",
    "confirm_prompt": "confirm_clear",
}

_MOJIBAKE_HINTS = (
    "�",
    "鐎",
    "顓",
    "瀹",
    "濞",
    "缁",
    "婢",
    "é",
    "å",
    "æ",
    "ç",
    "î",
)


@dataclass(frozen=True)
class VoiceProfile:
    """One selectable voice style for TTS."""

    profile_id: str
    label: str
    use_case: str
    voice_id: str
    speed: float = 1.0
    volume: float = 1.0
    pitch: int = 0
    emotion: str = ""
    sample_text: str = "你好，我是 Thunder，正在待命。"
    category: str = "general"
    cue: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_voice_profiles(
    config: dict[str, Any] | None,
    *,
    default_voice_id: str,
) -> dict[str, VoiceProfile]:
    """Build the voice catalog from config plus safe built-in presets."""

    cfg = config or {}
    default_voice = str(default_voice_id or cfg.get("minimax_voice_id") or "male-qn-qingse")
    profiles = _builtin_profiles(default_voice, cfg)

    raw_profiles = cfg.get("voice_profiles") or cfg.get("minimax_voice_profiles") or {}
    if isinstance(raw_profiles, list):
        raw_profiles = {
            str(item.get("profile_id") or item.get("id") or ""): item
            for item in raw_profiles
            if isinstance(item, dict)
        }
    if isinstance(raw_profiles, dict):
        for profile_id, raw in raw_profiles.items():
            if not isinstance(raw, dict):
                continue
            pid = str(raw.get("profile_id") or raw.get("id") or profile_id).strip()
            if not pid:
                continue
            base = profiles.get(pid)
            profiles[pid] = VoiceProfile(
                profile_id=pid,
                label=_profile_text(raw.get("label"), base.label if base else pid),
                use_case=_profile_text(raw.get("use_case"), base.use_case if base else ""),
                voice_id=str(raw.get("voice_id") or (base.voice_id if base else default_voice)),
                speed=float(raw.get("speed", base.speed if base else 1.0)),
                volume=float(raw.get("volume", raw.get("vol", base.volume if base else 1.0))),
                pitch=int(raw.get("pitch", base.pitch if base else 0)),
                emotion=str(raw.get("emotion", base.emotion if base else "")),
                sample_text=_profile_text(
                    raw.get("sample_text"),
                    base.sample_text if base else "你好，我是 Thunder，正在待命。",
                ),
                category=str(raw.get("category") or (base.category if base else "general")),
                cue=str(raw.get("cue") or raw.get("sound_cue") or (base.cue if base else "none")),
            )
    return profiles


def resolve_voice_profile_id(profile_id: str) -> str:
    """Resolve product playbook voice labels to concrete TTS profile ids."""

    value = str(profile_id or "").strip()
    return VOICE_PROFILE_ALIASES.get(value, value)


def _builtin_profiles(default_voice: str, cfg: dict[str, Any]) -> dict[str, VoiceProfile]:
    speed = float(cfg.get("minimax_speed", cfg.get("speed", 1.0)))
    volume = float(cfg.get("minimax_vol", cfg.get("volume", 1.0)))
    pitch = int(cfg.get("minimax_pitch", 0))
    emotion = str(cfg.get("minimax_emotion", ""))
    return {
        "patrol_default": VoiceProfile(
            profile_id="patrol_default",
            label="巡检播报",
            use_case="日常巡检、状态播报、低风险任务确认。",
            voice_id=default_voice,
            speed=speed,
            volume=volume,
            pitch=pitch,
            emotion=emotion,
            sample_text="巡检模式已开启，我会记录沿途异常并保持安全距离。",
            category="operations",
            cue="soft_chime",
        ),
        "visitor_friendly": VoiceProfile(
            profile_id="visitor_friendly",
            label="访客服务",
            use_case="游客问路、帮助点主动问候、服务台式应答。",
            voice_id=default_voice,
            speed=0.96,
            volume=0.95,
            pitch=1,
            emotion="happy",
            sample_text="你好，请问需要指路吗？我可以告诉你园区内的路线。",
            category="visitor",
            cue="welcome_chime",
        ),
        "security_clear": VoiceProfile(
            profile_id="security_clear",
            label="安保提醒",
            use_case="夜间陌生人、违停、人群聚集等需要清晰提醒的场景。",
            voice_id=default_voice,
            speed=1.0,
            volume=1.0,
            pitch=-1,
            emotion="calm",
            sample_text="请注意，这里正在进行安全巡检，请保持通道畅通。",
            category="security",
            cue="notice_beep",
        ),
        "emergency_short": VoiceProfile(
            profile_id="emergency_short",
            label="紧急短句",
            use_case="火灾烟雾、摔倒无法恢复等高优先级异常，短句直接播报。",
            voice_id=default_voice,
            speed=1.08,
            volume=1.0,
            pitch=0,
            emotion="fearful",
            sample_text="发现紧急异常，请远离现场，安保人员正在赶来。",
            category="emergency",
            cue="emergency_tone",
        ),
        "night_quiet": VoiceProfile(
            profile_id="night_quiet",
            label="夜间低声",
            use_case="夜间巡检、窗边角落观察，降低扰民但保持可听清。",
            voice_id=default_voice,
            speed=0.9,
            volume=0.72,
            pitch=-2,
            emotion="calm",
            sample_text="夜间巡检中，请说明你的来意并离开窗边区域。",
            category="security",
            cue="quiet_ping",
        ),
        "cleaning_soft": VoiceProfile(
            profile_id="cleaning_soft",
            label="保洁通知",
            use_case="垃圾桶满溢、保洁派单等低压提醒。",
            voice_id=default_voice,
            speed=0.98,
            volume=0.88,
            pitch=0,
            emotion="calm",
            sample_text="发现垃圾桶已满，已通知保洁人员处理。",
            category="cleaning",
            cue="soft_chime",
        ),
        "operations_calm": VoiceProfile(
            profile_id="operations_calm",
            label="运维沉稳",
            use_case="设备巡检、任务交接、运维处理进度说明。",
            voice_id=default_voice,
            speed=1.0,
            volume=0.92,
            pitch=0,
            emotion="calm",
            sample_text="运维事件已记录，我会继续同步处理进度。",
            category="operations",
            cue="soft_chime",
        ),
        "crowd_clear": VoiceProfile(
            profile_id="crowd_clear",
            label="人群疏导",
            use_case="人群聚集、通道拥堵，需要礼貌但明确地提醒分散。",
            voice_id=default_voice,
            speed=0.98,
            volume=0.96,
            pitch=-1,
            emotion="calm",
            sample_text="请不要长时间聚集在主通道，感谢配合。",
            category="security",
            cue="notice_beep",
        ),
        "guide_leading": VoiceProfile(
            profile_id="guide_leading",
            label="带路引导",
            use_case="低速带路、路线提示、转弯和停靠提醒。",
            voice_id=default_voice,
            speed=0.94,
            volume=0.9,
            pitch=1,
            emotion="happy",
            sample_text="我会低速带你前往目的地，请跟我保持安全距离。",
            category="visitor",
            cue="welcome_chime",
        ),
        "fault_urgent": VoiceProfile(
            profile_id="fault_urgent",
            label="故障急报",
            use_case="关节电机故障、卡住无法运动、恶意挡路等机器人异常。",
            voice_id=default_voice,
            speed=1.05,
            volume=1.0,
            pitch=-1,
            emotion="fearful",
            sample_text="机器人发生故障，已停止移动并通知安保处理。",
            category="emergency",
            cue="fault_tone",
        ),
        "confirm_clear": VoiceProfile(
            profile_id="confirm_clear",
            label="确认提示",
            use_case="任务确认、二次确认、等待用户选择。",
            voice_id=default_voice,
            speed=0.96,
            volume=0.9,
            pitch=0,
            emotion="calm",
            sample_text="请确认是否执行这个任务。",
            category="interaction",
            cue="confirm_chime",
        ),
    }


def _profile_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or _looks_corrupted(text):
        return fallback
    return text


def _looks_corrupted(text: str) -> bool:
    if not text:
        return False
    hint_count = sum(1 for hint in _MOJIBAKE_HINTS if hint in text)
    if hint_count >= 1:
        return True
    if any("\ue000" <= char <= "\uf8ff" for char in text):
        return True
    suspicious_chars = sum(1 for char in text if char in "�éåæçî")
    return suspicious_chars >= 2
