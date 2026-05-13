"""Provider profiles for OpenAI-compatible LLM backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderProfile:
    """Non-secret metadata used to select and report an LLM provider."""

    name: str
    aliases: tuple[str, ...]
    model_prefixes: tuple[str, ...] = ()
    base_url_hints: tuple[str, ...] = ()
    openai_compatible: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    domestic: bool = False

    def matches_model(self, model: str) -> bool:
        model_lower = str(model or "").lower()
        return bool(model_lower) and any(model_lower.startswith(prefix) for prefix in self.model_prefixes)

    def matches_base_url(self, base_url: str) -> bool:
        base_lower = str(base_url or "").lower()
        return bool(base_lower) and any(hint in base_lower for hint in self.base_url_hints)


PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "openai_compatible": ProviderProfile(
        name="openai_compatible",
        aliases=("openai-compatible", "compatible", "custom"),
    ),
    "minimax": ProviderProfile(
        name="minimax",
        aliases=("minimax_chat", "minimax-openai"),
        model_prefixes=("minimax",),
        base_url_hints=("minimax",),
        domestic=True,
    ),
    "doubao": ProviderProfile(
        name="doubao",
        aliases=("volcengine", "ark", "bytedance"),
        model_prefixes=("doubao", "ep-"),
        base_url_hints=("volces", "ark.cn", "bytedance"),
        domestic=True,
    ),
    "dashscope": ProviderProfile(
        name="dashscope",
        aliases=("qwen", "aliyun", "tongyi"),
        model_prefixes=("qwen", "qwq", "qvq"),
        base_url_hints=("dashscope", "aliyuncs"),
        domestic=True,
    ),
    "deepseek": ProviderProfile(
        name="deepseek",
        aliases=("deepseek-chat",),
        model_prefixes=("deepseek",),
        base_url_hints=("deepseek",),
        domestic=True,
    ),
    "zhipu": ProviderProfile(
        name="zhipu",
        aliases=("glm", "bigmodel"),
        model_prefixes=("glm",),
        base_url_hints=("bigmodel", "zhipu"),
        domestic=True,
    ),
    "openai": ProviderProfile(
        name="openai",
        aliases=("gpt",),
        model_prefixes=("gpt-", "o1", "o3", "o4"),
        base_url_hints=("api.openai.com",),
    ),
}


_ALIASES: dict[str, str] = {}
for _name, _profile in PROVIDER_PROFILES.items():
    _ALIASES[_name] = _name
    for _alias in _profile.aliases:
        _ALIASES[_alias] = _name
        _ALIASES[_alias.replace("_", "-")] = _name


def normalize_provider_name(provider: str | None) -> str:
    value = str(provider or "").strip().lower().replace("_", "-")
    if not value:
        return ""
    return _ALIASES.get(value, value.replace("-", "_"))


def infer_provider_name(*, model: str, base_url: str) -> str:
    model_match = ""
    for name, profile in PROVIDER_PROFILES.items():
        if name == "openai_compatible":
            continue
        if profile.matches_model(model):
            model_match = name
            break
    url_match = ""
    for name, profile in PROVIDER_PROFILES.items():
        if name == "openai_compatible":
            continue
        if profile.matches_base_url(base_url):
            url_match = name
            break
    if model_match and url_match and model_match != url_match:
        if str(model or "") == "MiniMax-M2.7-highspeed" and url_match != "minimax":
            return url_match
        return model_match
    if model_match:
        return model_match
    if url_match:
        return url_match
    return "openai_compatible"


def provider_profile(name: str) -> ProviderProfile:
    normalized = normalize_provider_name(name) or "openai_compatible"
    return PROVIDER_PROFILES.get(normalized, PROVIDER_PROFILES["openai_compatible"])
