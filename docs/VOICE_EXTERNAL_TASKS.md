# Voice External Tasks

AskMe can submit a bounded voice task to an external runtime and keep the voice
session responsive while it runs. Full-duplex sessions can speak progress or
the final result as a new canonical conversation turn. Half-duplex sessions do
not promise proactive delivery; the acknowledgement asks the user to query
`任务状态` after submission.

## Product boundary

- The first supported local voice contracts are `status_report`,
  `inspection_patrol`, and `navigate_to`. Arbitrary agent requests are not
  relabelled as one of those tasks and do not fall back to the deprecated local
  `AgentShell`.
- Robot work is routed as the dedicated `runtime_task` intent. Generic
  `agent_task` requests such as coding, research, design or report authoring
  continue through the general Agent dispatcher even when they mention patrol
  or inspection as a subject.
- `status_report` is low-risk and can pass the acknowledgement barrier directly.
  Patrol and navigation are physical tasks: AskMe first persists a `waiting_user`
  `TaskRun`, speaks the exact target and risk, and submits only after a later
  explicit `确认执行` from the same conversation and trusted operator.
- A confirmation challenge is bound to the original prompt turn, operator,
  verified person, approval ID, payload digest, and expiry. A restart recovers
  `waiting_user` or `confirmed` state and continues the same `run_id`; it does
  not create a second task. Expired approval challenges are persisted as locally
  cancelled and never reach the executor.
- The first voice UX allows one non-terminal task per conversation thread. A
  second task request is rejected until the current task completes or is
  explicitly cancelled, so status, confirmation and cancellation cannot target
  different tasks implicitly.
- The external runtime is the execution authority. The runtime-owned
  `ExternalTaskSupervisor` is the single post-submit owner for polling,
  reconciliation and cancellation, so voice, HTTP and dashboards read the same
  `TaskRun` truth. The voice layer owns only acknowledgement, conversation
  correlation and delivery.
- `fake`, `sim` and `shadow` profiles remain development/test profiles. They are
  never presented as proof that external work or hardware actions completed.
- The first spoken turn says only that AskMe is preparing to submit. A task
  starts after that preparation acknowledgement is committed; only a successful
  submit produces the separate “任务已受理” turn. Interrupting the preparation
  acknowledgement leaves the reservation unsubmitted, and a crash before submit
  can never leave the user with a false acceptance claim.
- When VoiceTaskLifecycle is available, supported `runtime_task` turns bypass the
  remote voice bridge and enter the local persistent TaskRun path directly. The
  bridge remains available only when that task authority is absent; one turn is
  never submitted by both owners.
- A physical command without a target enters an owner/session-bound
  `collecting_parameters` context for a bounded TTL. A short next turn such as
  `北门` can fill the target only for the same operator, verified person and
  conversation thread. Another speaker, another thread, an expired context,
  a question or a generic acknowledgement cannot consume it. No TaskRun exists
  until the missing parameter has produced a complete mission plan.
- Before physical submission, the same owner may revise the pending target and
  inspection photo count. Revision cancels the old local prepared TaskRun and
  creates a new approval-bound run with a new digest and approval ID; an old
  confirmation can therefore never authorize modified parameters.
- Barge-in and `stop_speaking` stop playback only. `取消任务` sends a separate,
  idempotent cancellation request. AskMe says “cancelled” only after the external
  runtime confirms a terminal cancelled state.
- Cancelling `waiting_user` or `confirmed` ends the local TaskRun without calling
  the executor. Cancelling `submission_unknown` is persisted and automatically
  sent after reconciliation obtains the remote task ID.
- Delivery correlation is frozen from the claimed task event, not from the
  session's latest-task snapshot. A notification updates `repeat_last` only
  after the user actually hears it. Transient delivery failures are retried a
  bounded number of times instead of being silently suppressed.
- Final notification receipts are persisted with the `TaskRun`. After a
  restart, an offline terminal result is replayed once while a previously
  delivered, interrupted, suppressed or expired event is not announced again.
  Delivery is therefore at-least-once across a crash between speech and receipt
  persistence, not an impossible exactly-once guarantee.
- Executor updates may return structured `observation` / `observations` and
  `artifact` / `artifacts` payloads. They are normalized, deduplicated and
  persisted as `SkillResult` evidence, then surfaced by the same TaskRun report
  used by APIs and voice. `照片呢` / `任务证据` queries report artifact and image
  counts and attach bounded evidence metadata to the canonical conversation
  turn instead of inventing a natural-language completion claim.
- Speech text and model output cannot assert operator authentication. Every
  voice turn resolves a trusted operator/person principal from a verifier or an
  explicitly single-operator authenticated session. Absent, stale or incomplete
  turn identity fails closed before acknowledgement or submission.

## Configuration

External execution is disabled by default. A production endpoint must use
HTTPS and name a bearer-token environment variable. Credential-free transport
is accepted only for the explicit `lab` profile on a loopback endpoint.

```yaml
runtime_handoff:
  enabled: true
  profile: external
  enable_external_runtime: true
  store:
    enabled: true
    path: data/runtime/task-runs.json
    swallow_errors: false
  external_runtime:
    endpoint: https://runtime.example.com
    auth_token_env: ASKME_RUNTIME_EXECUTOR_TOKEN
    connect_timeout_seconds: 2
    read_timeout_seconds: 5
    total_timeout_seconds: 8
    max_response_bytes: 1048576
    max_retries: 1
    poll_initial_seconds: 0.25
    poll_max_seconds: 4
    poll_deadline_seconds: 300
  voice_task:
    enabled: true
    approval_ttl_seconds: 60
    clarification_ttl_seconds: 45
    delivery_ttl_seconds: 120
    delivery_retry_delay_seconds: 0.25
    max_delivery_attempts: 3
    operator:
      session_scope: per_turn
```

For a shared microphone, the speaker-verification adapter binds its signed or
otherwise trusted claims to the exact captured `voice_turn_id` through
`AudioAgent.bind_voice_turn_operator_context(...)`. `VoiceLoop` consumes that
binding once for the current turn and passes it to reservation, confirmation,
status, cancellation, Gateway context and notifications. A static
`person_id` with `source: speaker_verification` is not accepted as proof of the
current speaker.

An authenticated kiosk or gateway that is physically and operationally scoped
to one operator may opt into the explicit fallback:

```yaml
runtime_handoff:
  voice_task:
    operator:
      session_scope: single_operator
      operator_id: robot-device-7
      roles: [operator]
      authenticated: true
      source: authenticated_single_operator_gateway
      person_id: operator-person-7
      permissions: [runtime:read, runtime:submit, runtime:cancel]
```

The token is resolved from the named environment variable at request time. Its
value is never written to configuration snapshots, TaskRun persistence, runtime
events, traces or logs.

`operator.authenticated: true` and `person_id` are assertions made by the trust
boundary, not by recognized speech or model output. A device certificate alone
is not a speaker identity; without a current trusted `person_id`, patrol,
navigation, confirmation and cancellation fail closed. Task ownership compares
both `operator_id` and `person_id`, including restart recovery and notification
adoption.

Voice task startup fails closed unless the TaskRun store is enabled and
`swallow_errors` is false. This prevents the product from promising restart
recovery while silently discarding failed state writes.

`TaskRunService` is the single in-process writer. Its read-modify-persist
transactions are serialized, and stores sharing one path in the same process
use a path lock plus unique temporary files before atomic replacement. The JSON
store is not a distributed database: a deployment must not start multiple
robot-runtime processes against the same writable file. Multi-process/HA
deployments must provide one external state-store writer before enabling voice
tasks.

## Executor HTTP contract

- `POST /v1/tasks` submits a handoff and returns a stable remote task ID.
- `GET /v1/tasks/{task_id}?cursor=...` returns the current state and updates.
- `POST /v1/tasks/{task_id}/cancel` requests cancellation.

Submit and cancel requests carry stable idempotency keys. Remote states are
projected monotonically into the local `TaskRun`; duplicate or out-of-order
updates cannot regress a terminal task. On restart, both `submission_unknown`
and the crash-window shape `queued + idempotency_key + no remote_task_id` are
reconciled by replaying the same idempotent submission.

An update `payload` may contain one mapping in `observation` or `artifact`, or
lists of mappings in `observations` and `artifacts`. Invalid non-mapping entries
are ignored. Repeated canonical evidence is not appended twice, including after
restart. Stable `artifact_id`, `evidence_id` or `observation_id` values take
precedence over content hashes; refreshed signed URLs or timestamps update the
existing evidence record instead of inflating the artifact count. Artifact bodies remain external references (for example an object-store
URI); the TaskRun stores metadata and provenance, not large media blobs.

## Voice-to-action lifecycle

```text
voice intent
  -> MissionService draft
  -> [missing target: collect owner-bound parameter]
  -> RuntimeHandoff TaskRun
  -> [physical task: waiting_user -> optional revision -> confirmed]
  -> acknowledgement committed
  -> external executor submit
  -> ExternalTaskSupervisor reconcile / poll / cancel
  -> task status or completion spoken as a separate conversation turn
```

The media session is not the task owner. Reconnecting audio, interrupting TTS,
or restarting `VoiceLoop` does not stop an executing TaskRun. Only an explicit
task command changes TaskRun state.

## Verification status

The production transport contract is covered by fake-transport state tests and
loopback HTTP integration tests, including authentication headers, retry
idempotency, malformed/oversized responses and redirect rejection. A live
external runtime, real credentials, ZeroClaw/MCP implementation, robot and
hardware remain unverified until a controlled staging smoke test succeeds.
