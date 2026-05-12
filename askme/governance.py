"""Operator identity and RBAC helpers for product-facing HTTP control paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


_DEFAULT_PERMISSIONS: dict[str, set[str]] = {
    "operator": {
        "field:event:create",
        "field:event:acknowledge",
        "field:event:request_close",
        "knowledge:read",
        "knowledge:import",
        "knowledge:preview",
        "runtime:read",
        "runtime:pause",
        "runtime:resume",
        "voice:profile:read",
    },
    "supervisor": {
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
        "voice:profile:read",
        "voice:profile:update",
    },
    "admin": {"*"},
}


@dataclass(frozen=True)
class OperatorIdentity:
    """Resolved operator identity from demo config or enterprise identity claims."""

    operator_id: str
    roles: tuple[str, ...]
    source: str
    display_name: str = ""
    authenticated: bool = False


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
        operators = field_cfg.get("operators") if isinstance(field_cfg.get("operators"), dict) else {}
        self._operators = operators
        self._permissions = self._permission_map(directory_cfg)

    def resolve(self, operator_id: str | None) -> OperatorIdentity:
        clean = str(operator_id or "").strip() or "dashboard.operator"
        raw = self._operators.get(clean) if isinstance(self._operators, dict) else None
        payload = raw if isinstance(raw, dict) else {}
        roles = payload.get("roles") if isinstance(payload.get("roles"), list) else ["operator"]
        normalized_roles = tuple(str(role or "").strip() for role in roles if str(role or "").strip())
        return OperatorIdentity(
            operator_id=clean,
            roles=normalized_roles or ("operator",),
            display_name=str(payload.get("display_name") or clean),
            source=self.identity_provider,
            authenticated=self.identity_provider not in {"", "local_config", "demo_config"},
        )

    def authorize(self, operator_id: str | None, permission: str) -> dict[str, Any]:
        identity = self.resolve(operator_id)
        allowed = self.has_permission(identity, permission)
        return {
            "allowed": allowed,
            "permission": permission,
            "operator": self.operator_payload(identity),
            "reason": "" if allowed else "operator_missing_permission",
            "audit": {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "identity_provider": self.identity_provider,
                "mode": self.mode,
                "production_binding_required": self.production_binding_required,
            },
        }

    def has_permission(self, identity: OperatorIdentity, permission: str) -> bool:
        for role in identity.roles:
            permissions = self._permissions.get(role, set())
            if "*" in permissions or permission in permissions:
                return True
        return False

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
            "sso": {
                "configured": self.identity_provider.lower() in {"oidc", "iam", "sso"},
                "provider": self.identity_provider,
                "issuer": self.oidc.get("issuer", ""),
                "client_id_configured": bool(self.oidc.get("client_id")),
                "jwks_uri_configured": bool(self.oidc.get("jwks_uri")),
                "claim_operator_id": self.oidc.get("claim_operator_id", "sub"),
                "claim_roles": self.oidc.get("claim_roles", "roles"),
            },
            "limitations": self.limitations(),
        }

    def limitations(self) -> list[str]:
        if self.identity_provider.lower() in {"oidc", "iam", "sso"}:
            return [
                "Enterprise identity mode is configured; routes still require deployment-level token validation before production exposure.",
                "All high-risk control actions must remain RBAC-checked and audit logged.",
            ]
        return [
            "Current operator directory is demo/local config, not an enterprise account system.",
            "Production must bind operators to SSO/IAM and keep approvals, shutdowns, and runtime controls in a unified audit trail.",
        ]

    @staticmethod
    def operator_payload(identity: OperatorIdentity) -> dict[str, Any]:
        return {
            "operator_id": identity.operator_id,
            "display_name": identity.display_name or identity.operator_id,
            "roles": list(identity.roles),
            "source": identity.source,
            "authenticated": identity.authenticated,
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
