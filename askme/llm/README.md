# AskMe LLM Package

The LLM package owns model-provider access and model-routing policy. It should
stay small at the root because product behavior depends on predictable provider
selection and auditable request handling.

## Owner Subpackages

- `core`: request contracts, config, provider factory, client, and gateway.
- `providers`: concrete provider adapters and provider capability profiles.
- `policy`: model selection and fallback policy.
- `streaming`: retry and streaming helpers shared by providers.
- `audit`: non-secret LLM request/response audit records and redaction helpers.

## Root Rule

Root modules such as `askme.llm.client`, `askme.llm.gateway`, and
`askme.llm.intent_router` are compatibility facades only. New implementation
must go into the owner subpackage first:

```python
from askme.llm.core.gateway import LLMGateway
from askme.llm.providers import MiniMaxProvider
from askme.llm.policy import ModelPolicy
```

Do not add product logic, provider-specific branching, or prompt policy into a
root compatibility file.

Intent routing is not LLM provider logic. New code should import intent routing
from `askme.robot_interaction`.

## Product Boundary

- Domestic low-latency providers live under `providers`.
- Fallback order lives under `policy`, not inside Dashboard or voice code.
- Prompt templates and customer-facing copy do not belong here; they are product
assets outside provider transport code.
- Provider health payloads must expose readiness and capability metadata without
leaking API keys or raw secrets.
