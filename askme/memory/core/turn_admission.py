"""Policy-controlled admission for durable dialogue memory.

The dialogue runtime must not copy every user/assistant exchange into a
long-term store.  This module turns a *user utterance* into a small structured
candidate, or rejects it.  Assistant text is intentionally not an input: model
output is not a trustworthy source of facts about the user or the customer.

The classifier is deliberately local and deterministic.  It is not intended to
understand every possible fact; uncertain statements stay in short-term
conversation history instead of being promoted optimistically.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_CN_ID_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\w)")
_PAYMENT_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_SECRET_RE = re.compile(
    r"(?:密码|口令|验证码|支付码|银行卡|api[ _-]?key|access[ _-]?token|secret|token)"
    r"\s*(?:是|为|:|：|=)?\s*\S+",
    re.I,
)

_QUESTION_MARKERS = (
    "吗",
    "么",
    "哪里",
    "在哪",
    "怎么",
    "如何",
    "多少",
    "是否",
    "能不能",
    "可不可以",
    "?",
    "？",
)

_TRIVIAL_UTTERANCES = frozenset(
    {
        "你好",
        "您好",
        "谢谢",
        "再见",
        "好的",
        "好",
        "嗯",
        "哦",
        "收到",
        "没问题",
        "在吗",
    }
)

_PREFERENCE_PATTERNS = (
    re.compile(r"^(?:请)?记住[，,：:\s]*(?:我)?(?:更)?(?:喜欢|偏好|习惯|希望)(?P<value>.+)$"),
    re.compile(r"^(?:我)?(?:更)?(?:喜欢|偏好|习惯|希望)(?P<value>.+)$"),
    re.compile(r"^我不喜欢(?P<value>.+)$"),
    re.compile(r"^以后请(?P<value>.+)$"),
)

_EXPLICIT_FACT_RE = re.compile(r"^(?:请)?记住[，,：:\s]*(?P<value>.+)$")
_CORRECTION_RE = re.compile(
    r"^(?:更正一下[，,：:\s]*|纠正一下[，,：:\s]*|不是)(?P<body>.+)$"
)
_LOCATION_RE = re.compile(
    r"^(?P<subject>[\w\u4e00-\u9fff·\-]{1,40}?(?:卫生间|厕所|洗手间|服务台|前台|"
    r"会议室|停车场|充电桩|电梯|楼梯|设备|仓库|配电室|机房|入口|出口))"
    r"(?:在|位于|设在)(?P<value>[^，。！？?]{1,60})$"
)
_SOP_RE = re.compile(
    r"^(?P<subject>[\w\u4e00-\u9fff·\-]{1,40}?(?:SOP|sop|流程|规程|操作步骤|巡检步骤))"
    r"(?:是|为|包括|需要)(?P<value>[^。！？?]{2,120})$"
)
_DEVICE_RE = re.compile(
    r"^(?P<subject>[\w\u4e00-\u9fff·\-]{1,40}?(?:设备|传感器|机器人|机器狗|终端))"
    r"(?:型号是|编号是|状态是|负责人是|安装在|位于)(?P<value>[^，。！？?]{1,80})$"
)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """One governed durable-memory candidate."""

    record_id: str
    memory_type: str
    subject: str
    predicate: str
    value: str
    source: str
    confidence: float
    scope: str
    customer_id: str
    project_id: str
    user_id: str
    sensitivity: str
    created_at: str
    last_confirmed_at: str
    expires_at: str
    supersedes: str
    approval_status: str
    source_turn_id: str = ""
    source_event_id: str = ""
    source_sequence: int | None = None
    source_thread_id: str = ""
    idempotency_key: str = ""
    occurred_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Compact fact text stored by a backend, without assistant output."""

        if self.predicate == "location":
            return f"{self.subject}位于{self.value}"
        if self.predicate == "preference":
            return f"{self.subject}偏好{self.value}"
        if self.predicate == "interaction_preference":
            return f"{self.subject}希望交互时{self.value}"
        return f"{self.subject}：{self.value}"

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("metadata", None)
        return {**payload, **self.metadata}


@dataclass(frozen=True, slots=True)
class TurnAdmissionResult:
    """Outcome of one local turn-admission decision."""

    admitted: bool
    candidates: tuple[MemoryCandidate, ...] = ()
    rejected_reason: str = ""
    persisted_count: int = 0
    persistence_errors: tuple[str, ...] = ()


class TurnMemoryAdmission:
    """Conservative local classifier for durable dialogue memory."""

    def __init__(
        self,
        *,
        min_asr_confidence: float = 0.75,
        user_memory_ttl_days: int = 365,
        knowledge_review_ttl_days: int = 30,
    ) -> None:
        self._min_asr_confidence = min(max(float(min_asr_confidence), 0.0), 1.0)
        self._user_memory_ttl_days = max(1, int(user_memory_ttl_days))
        self._knowledge_review_ttl_days = max(1, int(knowledge_review_ttl_days))

    def classify(
        self,
        user_text: str,
        *,
        source_turn_id: str = "",
        source_event_id: str = "",
        source_sequence: int | None = None,
        source_thread_id: str = "",
        idempotency_key: str = "",
        observed_at: datetime | str | None = None,
        source: str = "dialogue",
        confidence: float = 1.0,
        customer_id: str = "",
        project_id: str = "",
        user_id: str = "",
    ) -> TurnAdmissionResult:
        text = _clean_text(user_text)
        if not text:
            return TurnAdmissionResult(False, rejected_reason="empty")
        if confidence < self._min_asr_confidence:
            return TurnAdmissionResult(False, rejected_reason="low_asr_confidence")
        if _contains_sensitive_data(text):
            return TurnAdmissionResult(False, rejected_reason="sensitive_or_pii")
        if text in _TRIVIAL_UTTERANCES:
            return TurnAdmissionResult(False, rejected_reason="trivial_dialogue")

        # Explicit "remember" wording does not change a customer/site fact into
        # a personal preference.  Strip the directive before domain routing.
        routed_text = re.sub(r"^(?:请)?记住[，,：:\s]*", "", text).strip() or text
        if _looks_like_question(routed_text):
            return TurnAdmissionResult(False, rejected_reason="question_not_fact")

        now = _observed_datetime(observed_at)
        provenance: dict[str, Any] = {
            "source_event_id": str(source_event_id or "").strip(),
            "source_sequence": int(source_sequence) if source_sequence is not None else None,
            "source_thread_id": str(source_thread_id or "").strip(),
            "idempotency_key": str(idempotency_key or "").strip(),
        }
        knowledge = self._knowledge_candidate(
            routed_text,
            now=now,
            source=source,
            source_turn_id=source_turn_id,
            customer_id=customer_id,
            project_id=project_id,
            user_id=user_id,
            **provenance,
        )
        if knowledge is not None:
            return TurnAdmissionResult(True, (knowledge,))

        preference = self._preference_candidate(
            text,
            now=now,
            source=source,
            source_turn_id=source_turn_id,
            customer_id=customer_id,
            project_id=project_id,
            user_id=user_id,
            **provenance,
        )
        if preference is not None:
            return TurnAdmissionResult(True, (preference,))

        correction = self._correction_candidate(
            text,
            now=now,
            source=source,
            source_turn_id=source_turn_id,
            customer_id=customer_id,
            project_id=project_id,
            user_id=user_id,
            **provenance,
        )
        if correction is not None:
            return TurnAdmissionResult(True, (correction,))

        explicit = _EXPLICIT_FACT_RE.match(text)
        if explicit:
            value = _clean_value(explicit.group("value"))
            if value and not _looks_like_question(value):
                return TurnAdmissionResult(
                    True,
                    (
                        self._candidate(
                            memory_type="user_fact",
                            subject=user_id or "user",
                            predicate="profile_fact",
                            value=value,
                            now=now,
                            ttl_days=self._user_memory_ttl_days,
                            source=source,
                            source_turn_id=source_turn_id,
                            **provenance,
                            customer_id=customer_id,
                            project_id=project_id,
                            user_id=user_id,
                            scope="user",
                            approval_status="active",
                            confidence=0.9,
                        ),
                    ),
                )

        return TurnAdmissionResult(False, rejected_reason="not_durable_memory")

    def _knowledge_candidate(
        self,
        text: str,
        **context: Any,
    ) -> MemoryCandidate | None:
        match = _LOCATION_RE.match(text)
        predicate = "location"
        category = "location"
        if match is None:
            match = _SOP_RE.match(text)
            predicate = "procedure"
            category = "sop"
        if match is None:
            match = _DEVICE_RE.match(text)
            predicate = "device_fact"
            category = "equipment"
        if match is None:
            return None
        subject = _clean_value(match.group("subject"))
        value = _clean_value(match.group("value"))
        if not subject or not value:
            return None
        return self._candidate(
            memory_type="knowledge_candidate",
            subject=subject,
            predicate=predicate,
            value=value,
            ttl_days=self._knowledge_review_ttl_days,
            scope="customer_project",
            approval_status="pending_review",
            confidence=0.85,
            metadata={"category": category, "type": "knowledge"},
            **context,
        )

    def _preference_candidate(
        self,
        text: str,
        **context: Any,
    ) -> MemoryCandidate | None:
        candidate_text = re.sub(r"^(?:更正一下|纠正一下)[，,：:\s]*", "", text).strip()
        for index, pattern in enumerate(_PREFERENCE_PATTERNS):
            match = pattern.match(candidate_text)
            if match is None:
                continue
            value = _clean_value(match.group("value"))
            if not value:
                continue
            if "不喜欢" in candidate_text:
                value = f"不喜欢{value}"
            predicate = "interaction_preference" if index == 3 else "preference"
            return self._candidate(
                memory_type="user_preference",
                subject=str(context.get("user_id") or "user"),
                predicate=predicate,
                value=value,
                ttl_days=self._user_memory_ttl_days,
                scope="user",
                approval_status="active",
                confidence=0.9,
                **context,
            )
        return None

    def _correction_candidate(
        self,
        text: str,
        **context: Any,
    ) -> MemoryCandidate | None:
        match = _CORRECTION_RE.match(text)
        if match is None:
            return None
        body = _clean_value(match.group("body"))
        if not body:
            return None
        value = body.rsplit("是", 1)[-1].strip("，,：: ") if "是" in body else body
        if not value:
            return None
        return self._candidate(
            memory_type="user_correction",
            subject=str(context.get("user_id") or "user"),
            predicate="profile_fact",
            value=value,
            ttl_days=self._user_memory_ttl_days,
            scope="user",
            approval_status="active",
            confidence=0.9,
            metadata={"correction": True},
            **context,
        )

    @staticmethod
    def _candidate(
        *,
        memory_type: str,
        subject: str,
        predicate: str,
        value: str,
        now: datetime,
        ttl_days: int,
        source: str,
        source_turn_id: str,
        source_event_id: str,
        source_sequence: int | None,
        source_thread_id: str,
        idempotency_key: str,
        customer_id: str,
        project_id: str,
        user_id: str,
        scope: str,
        approval_status: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidate:
        scope_key = "|".join((customer_id, project_id, user_id, memory_type, subject, predicate))
        record_id = "mem_" + hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:20]
        timestamp = now.isoformat()
        return MemoryCandidate(
            record_id=record_id,
            memory_type=memory_type,
            subject=subject,
            predicate=predicate,
            value=value,
            source=source,
            confidence=min(max(float(confidence), 0.0), 1.0),
            scope=scope,
            customer_id=customer_id,
            project_id=project_id,
            user_id=user_id,
            sensitivity="normal",
            created_at=timestamp,
            last_confirmed_at=timestamp,
            expires_at=(now + timedelta(days=ttl_days)).isoformat(),
            supersedes=record_id,
            approval_status=approval_status,
            source_turn_id=source_turn_id,
            source_event_id=source_event_id,
            source_sequence=source_sequence,
            source_thread_id=source_thread_id,
            idempotency_key=idempotency_key,
            occurred_at=timestamp,
            metadata=dict(metadata or {}),
        )


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split()).strip("。！! ")


def _observed_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, str):
        clean = value.strip()
        if clean.endswith("Z"):
            clean = f"{clean[:-1]}+00:00"
        value = datetime.fromisoformat(clean)
    if not isinstance(value, datetime):
        raise TypeError("observed_at must be a datetime, ISO timestamp, or None")
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clean_value(value: str) -> str:
    return str(value or "").strip("，,。！!：:;； \t\r\n")


def _looks_like_question(text: str) -> bool:
    clean = str(text or "").strip()
    return any(marker in clean for marker in _QUESTION_MARKERS)


def _contains_sensitive_data(text: str) -> bool:
    return bool(
        _PHONE_RE.search(text)
        or _CN_ID_RE.search(text)
        or _PAYMENT_RE.search(text)
        or _EMAIL_RE.search(text)
        or _SECRET_RE.search(text)
    )


__all__ = ["MemoryCandidate", "TurnAdmissionResult", "TurnMemoryAdmission"]
