"""Compatibility imports for LiteLLM deployment key policy."""

from askme.llm.core.key_policy import KeyPolicyError, main, validate_litellm_key_policy

__all__ = ["KeyPolicyError", "main", "validate_litellm_key_policy"]


if __name__ == "__main__":
    raise SystemExit(main())
