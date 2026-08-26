"""Fail-closed LiteLLM secret policy for deployment startup.

The module validates format, placeholder resistance, basic strength, and role
separation without contacting the control plane or printing credential values.
"""

from __future__ import annotations

import argparse
import hmac
import os
import re
import sys
from collections.abc import Mapping, Sequence

_MIN_SECRET_LENGTH = 24
_MIN_UNIQUE_CHARACTERS = 8
_LITELLM_ACCESS_KEY_NAMES = frozenset(
    {
        "LITELLM_MASTER_KEY",
        "LITELLM_VIRTUAL_KEY",
        "ZEROCLAW_LITELLM_VIRTUAL_KEY",
    }
)
_PLACEHOLDER_MARKERS = (
    "changeme",
    "default",
    "dummy",
    "example",
    "fillme",
    "generated",
    "placeholder",
    "replace",
    "sample",
    "setme",
    "yourkey",
    "yourpassword",
    "yoursecret",
)
_PLACEHOLDER_REFERENCE = re.compile(r"\$\{[^}]+\}|\{\{[^}]+\}\}|<[^>]+>")


class KeyPolicyError(ValueError):
    """Raised when deployment secrets violate the startup policy."""


def _is_repeated_block(value: str) -> bool:
    for block_size in range(1, (len(value) // 3) + 1):
        if len(value) % block_size:
            continue
        repeats = len(value) // block_size
        if repeats >= 3 and value == value[:block_size] * repeats:
            return True
    return False


def _required_secret(environ: Mapping[str, str], name: str) -> str:
    raw_value = str(environ.get(name) or "")
    value = raw_value.strip()
    if len(value) < _MIN_SECRET_LENGTH:
        raise KeyPolicyError(f"{name} is required and must be a non-placeholder credential")
    if any(character.isspace() for character in raw_value):
        raise KeyPolicyError(f"{name} must not contain whitespace")
    if _PLACEHOLDER_REFERENCE.search(value):
        raise KeyPolicyError(f"{name} must not contain an unresolved secret reference")
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if any(marker in normalized for marker in _PLACEHOLDER_MARKERS):
        raise KeyPolicyError(f"{name} must not use a template placeholder credential")
    if name in _LITELLM_ACCESS_KEY_NAMES and not value.startswith("sk-"):
        raise KeyPolicyError(f"{name} must use the LiteLLM sk- credential format")
    if name == "LITELLM_DATABASE_PASSWORD" and re.fullmatch(r"[A-Za-z0-9._~-]+", value) is None:
        raise KeyPolicyError(
            "LITELLM_DATABASE_PASSWORD must contain only URL-safe unreserved characters"
        )
    strength_material = value[3:] if name in _LITELLM_ACCESS_KEY_NAMES else value
    if len(set(strength_material)) < _MIN_UNIQUE_CHARACTERS:
        raise KeyPolicyError(f"{name} must have sufficient character diversity")
    if _is_repeated_block(strength_material):
        raise KeyPolicyError(f"{name} must not be composed from a repeated character block")
    return value


def _require_distinct(first_name: str, first: str, second_name: str, second: str) -> None:
    if hmac.compare_digest(first, second):
        raise KeyPolicyError(
            "LiteLLM distinct credential roles are required: "
            f"{first_name} and {second_name} must not share a key"
        )


def validate_litellm_key_policy(
    environ: Mapping[str, str],
    *,
    require_application: bool = True,
    require_zeroclaw: bool = False,
) -> None:
    """Validate control-plane secrets and credential-role separation."""

    if require_zeroclaw and not require_application:
        raise KeyPolicyError("ZeroClaw validation requires the application credential policy")

    salt = _required_secret(environ, "LITELLM_SALT_KEY")
    database_password = _required_secret(environ, "LITELLM_DATABASE_PASSWORD")
    master = _required_secret(environ, "LITELLM_MASTER_KEY")
    credentials = [
        ("LITELLM_MASTER_KEY", master),
        ("LITELLM_SALT_KEY", salt),
        ("LITELLM_DATABASE_PASSWORD", database_password),
    ]

    if require_application:
        askme = _required_secret(environ, "LITELLM_VIRTUAL_KEY")
        credentials.insert(1, ("LITELLM_VIRTUAL_KEY", askme))

    zeroclaw_raw = str(environ.get("ZEROCLAW_LITELLM_VIRTUAL_KEY") or "").strip()
    if require_application and (require_zeroclaw or zeroclaw_raw):
        zeroclaw = _required_secret(environ, "ZEROCLAW_LITELLM_VIRTUAL_KEY")
        credentials.append(("ZEROCLAW_LITELLM_VIRTUAL_KEY", zeroclaw))

    for index, (first_name, first) in enumerate(credentials):
        for second_name, second in credentials[index + 1 :]:
            _require_distinct(first_name, first, second_name, second)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate LiteLLM deployment credential-role isolation."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--control-plane-only",
        action="store_true",
        help="Validate bootstrap master, salt, and database secrets before virtual keys exist.",
    )
    mode.add_argument(
        "--require-zeroclaw",
        action="store_true",
        help="Require and validate the dedicated ZeroClaw scoped key.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        validate_litellm_key_policy(
            os.environ,
            require_application=not bool(args.control_plane_only),
            require_zeroclaw=bool(args.require_zeroclaw),
        )
    except KeyPolicyError as exc:
        print(f"[litellm-key-policy] INVALID: {exc}", file=sys.stderr)
        return 1

    print("[litellm-key-policy] OK: deployment secret policy passed")
    return 0


__all__ = ["KeyPolicyError", "main", "validate_litellm_key_policy"]


if __name__ == "__main__":
    raise SystemExit(main())
