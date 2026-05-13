"""Operator identity and RBAC helpers for product-facing HTTP control paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_DEFAULT_PERMISSIONS: dict[str, set[str]] = {
    "operator": {
        "audit:read",
        "field:event:create",
        "field:event:acknowledge",
        "field:event:request_close",
        "knowledge:read",
        "knowledge:import",
        "knowledge:preview",
        "runtime:read",
        "runtime:pause",
        "runtime:resume",
        "skill:read",
        "voice:profile:read",
    },
    "supervisor": {
        "audit:read",
        "audit:export",
        "field:event:create",
        "field:event:acknowledge",
        "field:event:close",
        "field:event:request_close",
        "field:notification:test",
        "knowledge:read",
        "knowledge:import",
        "knowledge:approve",
        "knowledge:delete",
        "knowledge:rollback",
        "knowledge:rebuild",
        "runtime:read",
        "runtime:pause",
        "runtime:resume",
        "runtime:cancel",
        "runtime:advance",
        "skill:read",
        "skill:review",
        "voice:profile:read",
        "voice:profile:update",
    },
    "admin": {"*"},
}

_ROLE_METADATA: dict[str, dict[str, str]] = {
    "operator": {
        "label": "现场操作员",
        "description": "可创建和跟进现场事件、导入知识、暂停/恢复低风险运行。",
        "risk_level": "medium",
    },
    "supervisor": {
        "label": "现场主管",
        "description": "可审批知识、关闭高风险事件、测试通知、推进运行状态和审核技能。",
        "risk_level": "high",
    },
    "admin": {
        "label": "系统管理员",
        "description": "拥有全部配置和治理权限，仅应绑定企业身份系统后使用。",
        "risk_level": "critical",
    },
}
_ENTERPRISE_IDENTITY_PROVIDERS = {"oidc", "iam", "sso"}
_TRUTHY = {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class OperatorIdentity:
    """Resolved operator identity from demo config or enterprise identity claims."""

    operator_id: str
    roles: tuple[str, ...]
    source: str
    display_name: str = ""
    authenticated: bool = False
    known: bool = True


class OperatorDirectory:
    """Small RBAC boundary around the configured operator directory.

    The current product can run in demo-config mode, while production deployments
    can point this same contract at an IAM/OIDC adapter without changing routes.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        field_cfg = cfg.get("field_operations") if isinstance(cfg.get("field_operations"), dict) else {}
        directory_cfg = field_cfg.get("operator_directory") if isinstance(field_cfg.get("operator_directory"), dict) else {}
        self.mode = str(directory_cfg.get("mode") or "demo_config")
        self.identity_provider = str(directory_cfg.get("identity_provider") or "local_config")
        self.production_binding_required = bool(directory_cfg.get("production_binding_required", True))
        self.production_target = str(directory_cfg.get("production_target") or "enterprise_sso_or_iam")
        self.session_operator_header = str(directory_cfg.get("session_operator_header") or "x-askme-operator-id")
        self.oidc = directory_cfg.get("oidc") if isinstance(directory_cfg.get("oidc"), dict) else {}
        trusted_headers = (
            directory_cfg.get("trusted_identity_headers")
            if isinstance(directory_cfg.get("trusted_identity_headers"), dict)
            else {}
        )
        self.trusted_identity_headers_enabled = _as_bool(trusted_headers.get("enabled"), default=False)
        self.trusted_operator_id_header = str(
            trusted_headers.get("operator_id") or "x-askme-iam-operator-id"
        ).lower()
        self.trusted_roles_header = str(
            trusted_headers.get("roles") or "x-askme-iam-roles"
        ).lower()
        self.trusted_display_name_header = str(
            trusted_headers.get("display_name") or "x-askme-iam-display-name"
        ).lower()
        self.trusted_source_header = str(
            trusted_headers.get("source") or "x-askme-iam-source"
        ).lower()
        self.trusted_roles_required = _as_bool(trusted_headers.get("roles_required"), default=True)
        self.trusted_local_role_fallback = _as_bool(
            trusted_headers.get("fallback_to_local_roles"),
            default=False,
        )
        operators = field_cfg.get("operators") if isinstance(field_cfg.get("operators"), dict) else {}
        self._operators = operators
        self._permissions = self._permission_map(directory_cfg)

    def resolve(self, operator_id: str | None) -> OperatorIdentity:
        clean = str(operator_id or "").strip() or "dashboard.operator"
        raw = self._operators.get(clean) if isinstance(self._operators, dict) else None
        if raw is None:
            return OperatorIdentity(
                operator_id=clean,
                roles=(),
                display_name=clean,
                source=self.identity_provider,
                authenticated=False,
                known=False,
            )
        payload = raw if isinstance(raw, dict) else {}
        roles = payload.get("roles") if isinstance(payload.get("roles"), list) else ["operator"]
        normalized_roles = tuple(str(role or "").strip() for role in roles if str(role or "").strip())
        return OperatorIdentity(
            operator_id=clean,
            roles=normalized_roles or ("operator",),
            display_name=str(payload.get("display_name") or clean),
            source=self.identity_provider,
            authenticated=self.identity_provider not in {"", "local_config", "demo_config"},
            known=True,
        )

    def resolve_context(
        self,
        operator_id: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> OperatorIdentity:
        """Resolve identity from trusted IAM headers or the demo directory.

        In enterprise modes we only trust identity claims injected by a
        deployment-level IAM/OIDC gateway. Request bodies remain useful for
        demo mode, but they must not override verified upstream identity.
        """

        if self._uses_enterprise_identity():
            identity = self._identity_from_trusted_headers(headers)
            if identity is not None:
                return identity
            fallback_id = operator_id or _body_operator_id(body) or self._header_value(
                headers,
                self.session_operator_header,
                "x-operator-id",
            )
            return OperatorIdentity(
                operator_id=str(fallback_id or "unknown.operator").strip(),
                roles=(),
                display_name=str(fallback_id or "unknown.operator").strip(),
                source=self.identity_provider,
                authenticated=False,
                known=False,
            )
        return self.resolve(
            operator_id
            or _body_operator_id(body)
            or self._header_value(headers, self.session_operator_header, "x-operator-id")
        )

    def authorize(
        self,
        operator_id: str | None,
        permission: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = self.resolve_context(operator_id, headers=headers, body=body)
        allowed = self.has_permission(identity, permission)
        return {
            "allowed": allowed,
            "permission": permission,
            "operator": self.operator_payload(identity),
            "reason": "" if allowed else "operator_missing_permission",
            "audit": {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "identity_provider": self.identity_provider,
                "mode": self.mode,
                "production_binding_required": self.production_binding_required,
            },
        }

    def has_permission(self, identity: OperatorIdentity, permission: str) -> bool:
        if not identity.known:
            return False
        for role in identity.roles:
            permissions = self._permissions.get(role, set())
            if "*" in permissions or permission in permissions:
                return True
        return False

    def permissions_for(self, identity: OperatorIdentity) -> list[str]:
        if not identity.known:
            return []
        collected: set[str] = set()
        for role in identity.roles:
            permissions = self._permissions.get(role, set())
            if "*" in permissions:
                return ["*"]
            collected.update(permissions)
        return sorted(collected)

    def current_operator_payload(
        self,
        operator_id: str | None,
        *,
        headers: Mapping[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = self.resolve_context(operator_id, headers=headers, body=body)
        warnings = []
        if not identity.known:
            warnings.append("当前操作员不在服务端目录中，所有受控操作都会被拒绝。")
        if not identity.authenticated:
            warnings.append("当前身份来自本地 demo 配置，不等同于企业登录账号。")
        if self._uses_enterprise_identity() and not self.trusted_identity_headers_enabled:
            warnings.append("企业身份模式已配置，但未启用受信身份头，应用无法消费已验证登录态。")
        return {
            "operator": self.operator_payload(identity),
            "permissions": self.permissions_for(identity),
            "known": identity.known,
            "authenticated": identity.authenticated,
            "directory_mode": self.mode,
            "identity_provider": self.identity_provider,
            "warnings": warnings,
            "readiness": self.directory_readiness(),
        }

    def payload(self) -> dict[str, Any]:
        operators = [
            self.operator_payload(self.resolve(operator_id))
            for operator_id in sorted(self._operators.keys())
        ]
        return {
            "mode": self.mode,
            "identity_provider": self.identity_provider,
            "production_binding_required": self.production_binding_required,
            "production_target": self.production_target,
            "session_operator_header": self.session_operator_header,
            "operators": operators,
            "permissions": {
                role: sorted(values)
                for role, values in sorted(self._permissions.items())
            },
            "roles": self.roles_payload(),
            "authorization_matrix": self.authorization_matrix(),
            "readiness": self.directory_readiness(),
            "sso": {
                "configured": self.identity_provider.lower() in {"oidc", "iam", "sso"},
                "provider": self.identity_provider,
                "issuer": self.oidc.get("issuer", ""),
                "client_id_configured": bool(self.oidc.get("client_id")),
                "jwks_uri_configured": bool(self.oidc.get("jwks_uri")),
                "claim_operator_id": self.oidc.get("claim_operator_id", "sub"),
                "claim_roles": self.oidc.get("claim_roles", "roles"),
                "trusted_identity_headers_enabled": self.trusted_identity_headers_enabled,
                "trusted_operator_id_header": self.trusted_operator_id_header,
                "trusted_roles_header": self.trusted_roles_header,
                "trusted_display_name_header": self.trusted_display_name_header,
            },
            "limitations": self.limitations(),
        }

    def roles_payload(self) -> list[dict[str, Any]]:
        payload = []
        for role, permissions in sorted(self._permissions.items()):
            meta = _ROLE_METADATA.get(role, {})
            payload.append({
                "role": role,
                "label": meta.get("label", role),
                "description": meta.get("description", ""),
                "risk_level": meta.get("risk_level", "custom"),
                "permissions": sorted(permissions),
                "permission_count": len(permissions),
            })
        return payload

    def authorization_matrix(self) -> list[dict[str, Any]]:
        rows = []
        for role, permissions in sorted(self._permissions.items()):
            if "*" in permissions:
                rows.append({
                    "role": role,
                    "scope": "*",
                    "allowed": True,
                    "risk_level": _ROLE_METADATA.get(role, {}).get("risk_level", "critical"),
                })
                continue
            for permission in sorted(permissions):
                rows.append({
                    "role": role,
                    "scope": permission,
                    "allowed": True,
                    "risk_level": _ROLE_METADATA.get(role, {}).get("risk_level", "custom"),
                })
        return rows

    def directory_readiness(self) -> dict[str, Any]:
        findings = []
        provider = self.identity_provider.lower()
        if provider not in _ENTERPRISE_IDENTITY_PROVIDERS:
            findings.append({
                "severity": "warning",
                "code": "demo_identity_provider",
                "message": "当前使用本地 demo 操作员目录，适合演示和试点，不适合作为生产登录体系。",
            })
        if self.production_binding_required and provider not in _ENTERPRISE_IDENTITY_PROVIDERS:
            findings.append({
                "severity": "blocker",
                "code": "production_identity_binding_missing",
                "message": f"生产环境必须绑定 {self.production_target}。",
            })
        if provider in _ENTERPRISE_IDENTITY_PROVIDERS and not self.trusted_identity_headers_enabled:
            findings.append({
                "severity": "blocker",
                "code": "trusted_identity_headers_disabled",
                "message": "已选择企业身份模式，但未启用受信身份头；askme 无法消费网关验签后的登录态。",
            })
        if provider in _ENTERPRISE_IDENTITY_PROVIDERS and self.trusted_identity_headers_enabled:
            missing_headers = [
                name
                for name in (self.trusted_operator_id_header, self.trusted_roles_header)
                if not name
            ]
            if missing_headers:
                findings.append({
                    "severity": "error",
                    "code": "trusted_identity_header_missing",
                    "message": "受信身份头缺少 operator_id 或 roles 配置。",
                })
        configured_roles = set(self._permissions.keys())
        for operator_id, raw in sorted(self._operators.items()):
            operator = raw if isinstance(raw, dict) else {}
            roles = operator.get("roles") if isinstance(operator.get("roles"), list) else []
            unknown_roles = sorted(
                str(role)
                for role in roles
                if str(role) and str(role) not in configured_roles
            )
            if unknown_roles:
                findings.append({
                    "severity": "error",
                    "code": "operator_unknown_role",
                    "operator_id": str(operator_id),
                    "message": f"操作员 {operator_id} 引用了未配置角色：{', '.join(unknown_roles)}。",
                })
        if not self._operators and not (
            provider in _ENTERPRISE_IDENTITY_PROVIDERS
            and self.trusted_identity_headers_enabled
        ):
            findings.append({
                "severity": "error",
                "code": "operator_directory_empty",
                "message": "操作员目录为空，Dashboard 无法建立可审计身份。",
            })
        blocking = [item for item in findings if item.get("severity") in {"blocker", "error"}]
        return {
            "status": "production_ready" if not blocking and provider in _ENTERPRISE_IDENTITY_PROVIDERS else "demo_or_trial_only",
            "production_ready": not blocking and provider in _ENTERPRISE_IDENTITY_PROVIDERS,
            "finding_count": len(findings),
            "findings": findings,
        }

    def limitations(self) -> list[str]:
        if self.identity_provider.lower() in _ENTERPRISE_IDENTITY_PROVIDERS:
            return [
                "Enterprise identity mode expects deployment-level token validation before trusted identity headers reach askme.",
                "All high-risk control actions must remain RBAC-checked and audit logged.",
            ]
        return [
            "Current operator directory is demo/local config, not an enterprise account system.",
            "Production must bind operators to SSO/IAM and keep approvals, shutdowns, and runtime controls in a unified audit trail.",
        ]

    def _uses_enterprise_identity(self) -> bool:
        return self.identity_provider.lower() in _ENTERPRISE_IDENTITY_PROVIDERS

    def _identity_from_trusted_headers(
        self,
        headers: Mapping[str, str] | None,
    ) -> OperatorIdentity | None:
        if not self.trusted_identity_headers_enabled:
            return None
        operator_id = self._header_value(headers, self.trusted_operator_id_header)
        if not operator_id:
            return None
        roles = _parse_roles(self._header_value(headers, self.trusted_roles_header))
        if not roles and self.trusted_local_role_fallback:
            roles = self.resolve(operator_id).roles
        known = bool(operator_id and (roles or not self.trusted_roles_required))
        display_name = (
            self._header_value(headers, self.trusted_display_name_header)
            or self.resolve(operator_id).display_name
            or operator_id
        )
        source = self._header_value(headers, self.trusted_source_header) or self.identity_provider
        return OperatorIdentity(
            operator_id=operator_id,
            roles=roles,
            display_name=display_name,
            source=source,
            authenticated=known,
            known=known,
        )

    @staticmethod
    def _header_value(headers: Mapping[str, str] | None, *names: str) -> str:
        if not headers:
            return ""
        for name in names:
            if not name:
                continue
            direct = headers.get(name)
            if direct:
                return str(direct).strip()
            lower = headers.get(name.lower())
            if lower:
                return str(lower).strip()
        lower_headers = {str(key).lower(): str(value) for key, value in headers.items()}
        for name in names:
            if value := lower_headers.get(str(name).lower()):
                return value.strip()
        return ""

    @staticmethod
    def operator_payload(identity: OperatorIdentity) -> dict[str, Any]:
        return {
            "operator_id": identity.operator_id,
            "display_name": identity.display_name or identity.operator_id,
            "roles": list(identity.roles),
            "source": identity.source,
            "authenticated": identity.authenticated,
            "known": identity.known,
        }

    @staticmethod
    def _permission_map(directory_cfg: dict[str, Any]) -> dict[str, set[str]]:
        custom = directory_cfg.get("permissions") if isinstance(directory_cfg.get("permissions"), dict) else {}
        merged = {role: set(values) for role, values in _DEFAULT_PERMISSIONS.items()}
        for role, permissions in custom.items():
            if not isinstance(permissions, list):
                continue
            merged[str(role)] = {str(item) for item in permissions}
        return merged


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


def _body_operator_id(body: dict[str, Any] | None) -> str:
    if not isinstance(body, dict):
        return ""
    return str(body.get("operator_id") or "").strip()


def _parse_roles(raw: str) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    normalized = text.replace(";", ",").replace("|", ",")
    roles = [item.strip() for item in normalized.split(",") if item.strip()]
    return tuple(dict.fromkeys(roles))
