from __future__ import annotations

import pytest

from askme.llm.core.key_policy import (
    KeyPolicyError,
    main,
    validate_litellm_key_policy,
)


def _valid_environment() -> dict[str, str]:
    return {
        "LITELLM_MASTER_KEY": "sk-master-key-for-control-plane",
        "LITELLM_VIRTUAL_KEY": "sk-askme-scoped-virtual-key",
        "ZEROCLAW_LITELLM_VIRTUAL_KEY": "sk-zeroclaw-scoped-virtual-key",
        "LITELLM_SALT_KEY": "salt-test-A7m4Q9x2K8v6R3c5N1p0",
        "LITELLM_DATABASE_PASSWORD": "db-test-B8n5R2w9K4x7T1c6M3q0",
    }


def test_distinct_litellm_key_roles_pass_startup_policy() -> None:
    validate_litellm_key_policy(_valid_environment(), require_zeroclaw=True)


@pytest.mark.parametrize(
    "placeholder",
    [
        "sk-replace-with-a-long-random-key",
        "sk-this-is-a-placeholder-credential",
        "sk-example-credential-for-deployment",
        "sk-generated-askme-virtual-key",
        "sk-change-me-before-production-use",
    ],
)
def test_template_placeholders_never_pass_key_policy(placeholder: str) -> None:
    environment = _valid_environment()
    environment["LITELLM_MASTER_KEY"] = placeholder

    with pytest.raises(KeyPolicyError, match="LITELLM_MASTER_KEY") as error:
        validate_litellm_key_policy(environment)

    assert placeholder not in str(error.value)


@pytest.mark.parametrize(
    "placeholder",
    [
        "sk-dummy-credential-A7m4Q9x2K8v6R3c5",
        "sk-sample-credential-B8n5R2w9K4x7T1c6",
        "sk-default-credential-C9p6T3x0V5y8M2d7",
        "sk-your-key-goes-here-D0q7V4y1N6z9",
        "sk-fill-me-before-release-E1r8W5z2P7x0",
    ],
)
def test_common_placeholder_variants_fail_closed(placeholder: str) -> None:
    environment = _valid_environment()
    environment["LITELLM_VIRTUAL_KEY"] = placeholder

    with pytest.raises(KeyPolicyError, match="LITELLM_VIRTUAL_KEY") as error:
        validate_litellm_key_policy(environment)

    assert placeholder not in str(error.value)


@pytest.mark.parametrize(
    "malformed",
    [
        " sk-master-key-for-control-plane",
        "sk-master-key-for-control-plane ",
        "sk-master-key-for control-plane",
    ],
)
def test_access_keys_with_whitespace_fail_closed(malformed: str) -> None:
    environment = _valid_environment()
    environment["LITELLM_MASTER_KEY"] = malformed

    with pytest.raises(KeyPolicyError, match="LITELLM_MASTER_KEY") as error:
        validate_litellm_key_policy(environment)

    assert malformed not in str(error.value)


@pytest.mark.parametrize(
    ("name", "placeholder"),
    [
        ("LITELLM_MASTER_KEY", "sk-${LITELLM_MASTER_KEY_FROM_STORE}"),
        ("LITELLM_VIRTUAL_KEY", "sk-{{ASKME_VIRTUAL_KEY_FROM_STORE}}"),
        ("ZEROCLAW_LITELLM_VIRTUAL_KEY", "sk-<ZEROCLAW_KEY_FROM_STORE>"),
        ("LITELLM_SALT_KEY", "${LITELLM_SALT_KEY_FROM_STORE}"),
        ("LITELLM_DATABASE_PASSWORD", "<DATABASE_PASSWORD_FROM_STORE>"),
    ],
)
def test_unresolved_secret_references_fail_closed(name: str, placeholder: str) -> None:
    environment = _valid_environment()
    environment[name] = placeholder

    with pytest.raises(KeyPolicyError, match=name) as error:
        validate_litellm_key_policy(environment, require_zeroclaw=True)

    assert placeholder not in str(error.value)


@pytest.mark.parametrize(
    "name",
    ["LITELLM_SALT_KEY", "LITELLM_DATABASE_PASSWORD"],
)
def test_product_policy_requires_control_plane_secrets(name: str) -> None:
    environment = _valid_environment()
    environment[name] = ""

    with pytest.raises(KeyPolicyError, match=name):
        validate_litellm_key_policy(environment)


@pytest.mark.parametrize(
    "name",
    [
        "LITELLM_MASTER_KEY",
        "LITELLM_VIRTUAL_KEY",
        "ZEROCLAW_LITELLM_VIRTUAL_KEY",
        "LITELLM_SALT_KEY",
        "LITELLM_DATABASE_PASSWORD",
    ],
)
def test_every_protected_secret_meets_the_minimum_strength_floor(name: str) -> None:
    environment = _valid_environment()
    environment[name] = "sk-A7m4Q9x2K8v6R3c5"

    with pytest.raises(KeyPolicyError, match=name):
        validate_litellm_key_policy(environment, require_zeroclaw=True)


@pytest.mark.parametrize(
    "name",
    [
        "LITELLM_MASTER_KEY",
        "LITELLM_VIRTUAL_KEY",
        "ZEROCLAW_LITELLM_VIRTUAL_KEY",
        "LITELLM_SALT_KEY",
        "LITELLM_DATABASE_PASSWORD",
    ],
)
def test_repeated_low_diversity_secrets_fail_strength_checks(name: str) -> None:
    environment = _valid_environment()
    environment[name] = "sk-" + ("A" * 40)

    with pytest.raises(KeyPolicyError, match=name):
        validate_litellm_key_policy(environment, require_zeroclaw=True)


@pytest.mark.parametrize(
    "name",
    [
        "LITELLM_MASTER_KEY",
        "LITELLM_VIRTUAL_KEY",
        "ZEROCLAW_LITELLM_VIRTUAL_KEY",
    ],
)
def test_litellm_access_keys_require_the_sk_prefix(name: str) -> None:
    environment = _valid_environment()
    environment[name] = "not-sk-A7m4Q9x2K8v6R3c5N1p0T8w6"

    with pytest.raises(KeyPolicyError, match=name):
        validate_litellm_key_policy(environment, require_zeroclaw=True)


def test_database_password_must_be_safe_for_the_compose_database_url() -> None:
    environment = _valid_environment()
    environment["LITELLM_DATABASE_PASSWORD"] = "db-test-A7m4Q9x2@host:5432/path"

    with pytest.raises(KeyPolicyError, match="LITELLM_DATABASE_PASSWORD"):
        validate_litellm_key_policy(environment)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LITELLM_MASTER_KEY", "sk-" + ("A7m4Q9x2" * 4)),
        ("LITELLM_VIRTUAL_KEY", "sk-" + ("B8n5R2w9" * 4)),
        ("ZEROCLAW_LITELLM_VIRTUAL_KEY", "sk-" + ("C9p6T3x0" * 4)),
        ("LITELLM_SALT_KEY", "D0q7V4y1" * 4),
        ("LITELLM_DATABASE_PASSWORD", "E1r8W5z2" * 4),
    ],
)
def test_repeated_blocks_do_not_count_as_strong_secrets(name: str, value: str) -> None:
    environment = _valid_environment()
    environment[name] = value

    with pytest.raises(KeyPolicyError, match=name):
        validate_litellm_key_policy(environment, require_zeroclaw=True)


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("LITELLM_MASTER_KEY", "LITELLM_VIRTUAL_KEY"),
        ("LITELLM_MASTER_KEY", "ZEROCLAW_LITELLM_VIRTUAL_KEY"),
        ("LITELLM_VIRTUAL_KEY", "ZEROCLAW_LITELLM_VIRTUAL_KEY"),
        ("LITELLM_MASTER_KEY", "LITELLM_SALT_KEY"),
        ("LITELLM_MASTER_KEY", "LITELLM_DATABASE_PASSWORD"),
        ("LITELLM_VIRTUAL_KEY", "LITELLM_SALT_KEY"),
        ("LITELLM_VIRTUAL_KEY", "LITELLM_DATABASE_PASSWORD"),
        ("ZEROCLAW_LITELLM_VIRTUAL_KEY", "LITELLM_SALT_KEY"),
        ("ZEROCLAW_LITELLM_VIRTUAL_KEY", "LITELLM_DATABASE_PASSWORD"),
        ("LITELLM_SALT_KEY", "LITELLM_DATABASE_PASSWORD"),
    ],
)
def test_litellm_credential_roles_must_be_pairwise_distinct(
    first_name: str,
    second_name: str,
) -> None:
    environment = _valid_environment()
    environment[second_name] = environment[first_name]

    with pytest.raises(KeyPolicyError, match="distinct credential roles"):
        validate_litellm_key_policy(environment, require_zeroclaw=True)


def test_root_askme_policy_does_not_require_a_zeroclaw_key() -> None:
    environment = _valid_environment()
    environment.pop("ZEROCLAW_LITELLM_VIRTUAL_KEY")

    validate_litellm_key_policy(environment)


def test_control_plane_bootstrap_does_not_require_an_unissued_application_key() -> None:
    environment = _valid_environment()
    environment.pop("LITELLM_VIRTUAL_KEY")
    environment.pop("ZEROCLAW_LITELLM_VIRTUAL_KEY")

    validate_litellm_key_policy(environment, require_application=False)


def test_default_product_policy_still_requires_the_application_key() -> None:
    environment = _valid_environment()
    environment.pop("LITELLM_VIRTUAL_KEY")

    with pytest.raises(KeyPolicyError, match="LITELLM_VIRTUAL_KEY"):
        validate_litellm_key_policy(environment)


def test_full_product_policy_requires_a_zeroclaw_key() -> None:
    environment = _valid_environment()
    environment["ZEROCLAW_LITELLM_VIRTUAL_KEY"] = ""

    with pytest.raises(KeyPolicyError, match="ZEROCLAW_LITELLM_VIRTUAL_KEY"):
        validate_litellm_key_policy(environment, require_zeroclaw=True)


def test_cli_failure_never_echoes_credential_values(monkeypatch, capsys) -> None:
    environment = _valid_environment()
    exposed_secret = environment["LITELLM_MASTER_KEY"]
    environment["LITELLM_VIRTUAL_KEY"] = exposed_secret
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert main(["--require-zeroclaw"]) == 1

    output = capsys.readouterr()
    assert exposed_secret not in output.out
    assert exposed_secret not in output.err
    assert "INVALID" in output.err


def test_control_plane_cli_passes_before_virtual_keys_exist(monkeypatch, capsys) -> None:
    environment = _valid_environment()
    environment.pop("LITELLM_VIRTUAL_KEY")
    environment.pop("ZEROCLAW_LITELLM_VIRTUAL_KEY")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("LITELLM_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("ZEROCLAW_LITELLM_VIRTUAL_KEY", raising=False)

    assert main(["--control-plane-only"]) == 0

    output = capsys.readouterr()
    assert "OK" in output.out
    for value in environment.values():
        assert value not in output.out
        assert value not in output.err
