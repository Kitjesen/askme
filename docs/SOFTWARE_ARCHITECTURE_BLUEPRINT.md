# Software Architecture Blueprint

日期：2026-06-05

状态：高级软件架构蓝图。本文把 `docs/PRODUCT_REQUIREMENTS.md` 的 P0 和 R1-R7 产品需求翻译成 bounded contexts、事实源、包所有权、API 表面和 Architecture decision gates。它不是新的运行时实现，也不是重画目录结构；它是架构变更前的总约束。

## Reading Contract

- Product Requirements: `docs/PRODUCT_REQUIREMENTS.md`
- Demand-to-architecture trace: `docs/PRODUCT_ARCHITECTURE_TRACE.md`
- Module ownership: `docs/MODULE_OWNERSHIP.md`
- Runtime/security architecture: `docs/ARCHITECTURE_V2.md`
- Basic package dependency direction: `docs/ARCHITECTURE.md`

## System Context

AskMe sits between a solution-provider delivery team, customer site systems,
ZeroClaw or another MCP client, and Runtime / Safety / Hardware.

```text
Solution provider / field team
  -> Product/Admin surfaces
  -> AskMe bounded contexts
  -> Runtime / Safety / Hardware
  -> robot, sensors, maps, external systems

ZeroClaw or MCP client
  -> askme-mcp server
  -> controlled tools and resources
  -> TaskHandoff / SafetyPreflight / Runtime Arbiter
```

The product promise remains narrow: turn Demo-to-pilot work into reusable,
auditable, customer-signoff-ready delivery evidence. No raw hardware control is
exposed from Product, Admin, Platform, Dashboard, or MCP tools.

## Context map

| Bounded context | Owns | Does not own | Primary packages |
| --- | --- | --- | --- |
| Field Delivery Domain | customer project, managed object, site profile, field event, onsite evidence, acceptance dossier, customer signoff, readiness gaps, usage evidence | hardware execution, provider construction, Dashboard-only conclusions | `askme/pipeline/field`, `askme/api/routes/field_*`, `askme/api/services/field_*` |
| Interaction & Knowledge Domain | voice/text turn entry, InteractionGate, intent routing, knowledge/RAG evidence, customer-readable answers | direct robot dispatch, production readiness, field acceptance decisions | `askme/voice_gateway`, `askme/robot_interaction`, `askme/memory`, `askme/cognition` |
| Runtime Handoff Domain | TaskHandoff, TaskRun, runtime profile, pause/resume/cancel/advance, runtime evidence receipts | customer signoff, pricing, external-system fact ownership | `askme/runtime`, `askme/pipeline/core`, `askme/pipeline/channels` |
| Safety / Hardware Boundary | real execution, robot receipts, hardware status, takeover, rollback, safety stop | product promise, acceptance dossier, Dashboard status copy | `askme/robot`, `askme/providers`, `askme/ports` |
| Integration Contracts | VMS, CMMS, IAM, map, OEM fleet, notification, SIEM/WORM envelopes, failure_state, retry_policy, audit_export_id | replacing Field Delivery Domain facts | `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`, API route services |
| Observability / Audit | trace_id, health, metrics, SkillAuditLog, field audit, runtime audit, export hash | business truth without domain records | `askme/audit`, `askme/telemetry`, `askme/api/routes/health.py` |
| Capability Governance | skill packages, approvals, risk level, release channel, rollback, blocked uses | unchecked LLM draft execution or hardware bypass | `askme/skills`, `askme/pipeline/skills`, `askme/mcp/tools` |

## Container / Package Map

Use package ownership as the implementation boundary, not as a reason to add
new layers:

- `askme/blueprints` composes product presets and startup metadata.
- `askme/runtime` owns module graph and lifecycle.
- `askme/pipeline/field` owns Field Delivery Domain facts.
- `askme/api/routes/field_*` and `askme/api/services/field_*` expose field facts
  without becoming independent fact sources.
- `askme/static` renders customer and operator surfaces only from API payloads.
- `askme/memory`, `askme/cognition`, `askme/voice_gateway`, and
  `askme/robot_interaction` own interaction and knowledge context.
- `askme/skills` and `askme/pipeline/skills` own Capability Governance.
- `askme/ports` defines protocols; `askme/providers` selects concrete
  adapters; `askme/robot` owns hardware/service clients.

## API Surface Rules

Product/Admin/Platform/Internal must stay separate:

- Product explains customer-visible project, event, knowledge, scenario, and
  report status.
- Admin records governance, approval, audit, acceptance review, and launch
  readiness review.
- Platform reports health, readiness summary, metrics, and trace state without
  customer business authority.
- Internal handles runtime/device/vision callbacks only.

No Dashboard-only fact source is allowed. `askme/static` can display customer
signoff, blocked_uses, readiness gaps, and evidence links, but every claim must
come from Field Delivery Domain, Runtime receipts, or an explicit external
system contract.

## Architecture Invariants

- customer signoff != production readiness.
- Runtime / Safety / Hardware owns execution truth.
- No raw hardware control from Product, Admin, Platform, Dashboard, MCP, or
  sales-facing paths.
- No default-project fallback for tenant/customer/project/site/object scope
  mismatches.
- No Dashboard-only fact source for acceptance, readiness, ROI, pricing_signal,
  or launch state.
- No research claim becomes an architecture invariant without evidence_id and
  hypothesis_status from `docs/DEMAND_EVIDENCE_LEDGER.md`.
- No provider, robot, or hardware client imports product orchestration policy.
- No compatibility facade grows new business behavior.

## Architecture decision gates

Before adding or moving a product feature, record these answers in the PRD,
trace, issue, or review note:

1. Which R1-R7 requirement does it serve?
2. What is the evidence_id and hypothesis_status?
3. Which bounded context owns the fact?
4. Which API surface exposes it: Product, Admin, Platform, or Internal?
5. Which package owns implementation according to `docs/MODULE_OWNERSHIP.md`?
6. Which tests prove the contract, including boundary tests?
7. Does the change introduce fallback behavior? If yes, classify it as a
   grounded compatibility/fail-safe fallback or remove it.
8. Which release gate changes: Discovery, PRD-ready, Pilot-ready,
   Launch-readiness review, or Packaging-ready?

## Verification Contract

For architecture-only changes, run:

```powershell
python -m pytest tests/test_repository_layout.py -q
python -m pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q
python -m ruff check tests/test_repository_layout.py
```

For Field Delivery Domain changes, add the behavior checks listed in
`docs/PRODUCT_ARCHITECTURE_TRACE.md`, including
`tests/test_product_launch_readiness.py`. Full `pytest tests` remains the final
confidence gate before claiming the whole dirty worktree is clean.
