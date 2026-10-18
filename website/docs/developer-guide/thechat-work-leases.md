# TheChat workspace work leases

This adapter implements the asynchronous Hermes contract in TheChat's
`docs/workspace-resource-lifecycle.md` (workspace migrations 0017–0019).
It does not provision or deploy TheChat, change its database limits, or isolate
customer runtimes. Enable it only as a coordinated, separately approved rollout.

## Configuration and compatibility

The default is **required** for a conversation with a non-null `workspaceId`,
including workspace DMs. Its event must contain the exact `workToken` issued by
TheChat. Invalid/missing tokens fail before attachment download or execution.
The adapter never invents, trims, normalizes, replaces or reacquires an epoch.

```yaml
platforms:
  thechat:
    workspace_work_lease_mode: required
```

This is a platform YAML setting, not a secret environment toggle. Omission or
an unrecognized value remains fail-closed. Personal/unscoped events without a
token keep the existing tokenless contract and do not require the heartbeat
endpoint. A token, when present, always activates fencing, even in legacy mode.

`workspace_work_lease_mode: legacy` is an explicit compatibility choice **only
for a verified old backend that has no workspace lease protocol**. Never use it
to make missing tokens or failed heartbeats work against a lease-enabled backend.
The currently inspected TheChat `/health` response does not advertise protocol
capability, so absence of a health field cannot safely auto-detect a legacy
server. An older workspace backend therefore requires explicit operator review;
personal backends do not.

Keep clocks synchronized between Hermes and TheChat. Lease deadlines are UTC
instants; the adapter converts the acknowledged deadline to monotonic time,
conservatively subtracting the request duration. There is no configurable local
TTL that can grant authority beyond the server's acknowledgement.

## Lifetime and callback behavior

1. Webhooks still authenticate their timestamped HMAC before durable admission.
   Polling uses the same validated event handler. Every claimed event acknowledges
   and renews its lease concurrently. Delivery reserves the canonical session's
   FIFO position before inbox/heartbeat awaits, so an attachment cannot be
   overtaken by later text in that session; other sessions remain concurrent.
2. Before any attachment/model/tool execution, Hermes consumes a durable local
   execution permit and obtains an initial successful
   `POST /hermes-platform/invocations/{invocationId}/heartbeat` response with
   `{ "ok": true, "leaseExpiresAt": "<UTC deadline>" }`.
3. A separate silent heartbeat task follows the invocation across attachment
   transfer, startup restoration, local queueing, model/tool execution and human
   waits. It does not depend on typing, generate progress, or send a chat message.
   Renewal is scheduled at one third of the remaining acknowledged interval,
   capped at 20 seconds; heartbeat I/O is bounded by the remaining interval and
   a 10-second request cap.
4. Progress (including title/approval/clarify/vault metadata), typing, text/media
   messages and completed/silent/failed/cancelled callbacks echo the same frozen
   token. Request payload fields cannot override it. Context snapshots retain the
   original lease; task-local context prevents a queued invocation from becoming
   the source of an older turn's unanchored progress.
5. Ordinary leased messages use the existing FIFO rather than merging prompts or
   steering another epoch. `/queue` acknowledgements are messages, **not terminal
   completion**. Each queued workspace turn owns its own adapter task and native
   run generation, not a recursive turn inside its predecessor's task. Native
   approval/clarify command bypasses remain available. The native busy handler
   resolves bare `yes`/`no` against a real blocking approval before FIFO admission;
   the answer completes under its own invocation, without closing the waiting
   turn. With no blocking approval, the same text remains an ordinary queued turn.
6. Native `/stop` and `/new` queue removal sends cancellation for the exact
   discarded pending invocation. `/new` also cancels FIFO overflow removed by the
   conversation reset. Retained internal wakes, `/stop` overflow and unrelated
   sessions are not closed by this discard cleanup.
7. Terminal callbacks are attempted while the epoch is live; the heartbeat then
   stops. If a terminal acknowledgement is lost, Hermes does not replay the model
   or tools. Server lease reconciliation is the capacity-release backstop.

An authority/protocol rejection (400/401/403/404/409/410/422 callback; non-transient
heartbeat rejection), malformed deadline, or elapsed last acknowledged deadline
fences further invocation-bound callbacks. Network/408/429/5xx heartbeat failures
may retry **only until that last acknowledged deadline**; other successful
callbacks do not invent a fresh local deadline. Admission failure does not start
execution or fall back to a tokenless request.

## Cancellation scope and limits

Loss cancels this invocation's intake task and exact processing task. It uses the
existing native runner hard interrupt, generation invalidation and turn-process
reaper only when the runner's current event and adapter owner task still match.
It does not call chat-wide `/stop` or clear unrelated sessions. Published native
approval requests are withdrawn by exact request ID, clarification waiters are
released without an answer, and the invocation's vault waiters are cancelled.
A queued event is rejected before entering its agent handler if its lease died.
Disconnect revokes local work rather than leaving execution alive without renewals.

These are **cooperative logical controls, not hard process isolation**:

- Cancelling an asyncio task cannot physically stop a running Python executor
  thread. Native hard interrupt signals the agent, its active children and tool
  workers, and requests scoped process cleanup; uncooperative code can remain.
- Already-issued HTTP requests, model calls, object-store signed URLs and external
  side effects cannot be recalled. An event loop blocked by synchronous code also
  delays its watchdog. The database still fences later writes independently.
- Detached/background work is not given a new epoch by this adapter. A restored
  workspace wake without live local invocation authority is refused. The backend
  needs a distinct new-invocation contract for autonomous continuations; never
  reuse a terminal inbound invocation to make their UI visible.
- Standalone proactive delivery is a separate, non-invocation API contract; it is
  not a per-tool sandbox or a tenant isolation boundary. Use separate bot
  credentials, Hermes profiles/runtimes, filesystem/tool stores and provider
  credentials for unrelated trust boundaries.

## Durable recovery and retry boundaries

The existing SQLite `inbound_event_receipts` is reused. The signed webhook inbox
still records receipt/admission separately from execution completion. An additional
namespace, `thechat:<base-url>:work:<bot-id>`, uses the canonical invocation ID as
its key and consumes an **at-most-once local execution permit** before effects.
Its `completed` state means "permit consumed", not "TheChat task succeeded".
The payload hash refuses altered duplicate epochs or bodies. A one-attempt claim
also fences a crash between claiming and marking the permit consumed.

A new process can recover a webhook that was accepted but never consumed. It may
not replay a consumed epoch, even if the remote heartbeat would still accept it.
This intentionally trades some availability for safety: a crash after consuming
the permit but before starting a tool can lose otherwise runnable work. Preserve
receipts, inspect external effects and ask for a **new user invocation** after
reconciliation. Do not delete receipts or turn an unknown outcome into an automatic
retry. Existing inbox retention/capacity limits still apply; after local retention
expires the remote terminal/expired epoch remains the authoritative fence.

After restart, invocation-anchored delivery retries without live context fail
closed instead of becoming proactive messages. Personal legacy completions retain
a bounded in-process set of known safe tokenless reply references. No workspace
token is stored in logs, spans or a reusable session credential.

## Known API integration boundaries

- Events carry `workToken` but not an acknowledged lease deadline. Initial
  heartbeat is mandatory, including for a recovered-but-not-consumed inbox item.
- `/health` currently has no lease capability/version field; no auto downgrade.
- Workspace `session.title` is fenced like other progress. A title generated after
  terminal completion is dropped, even though the legacy backend allowed it.
  Supporting late workspace titles requires a separately authorized metadata
  contract; do not revive the execution epoch.
- Approval/clarify/vault responses are signed interaction lanes tied to the
  original request, not new model invocations. The heartbeat stays alive during
  their waits; cancellation uses exact native waiter identities.
- Callback success does not prove an external tool stopped. A lost terminal
  acknowledgement can leave the backend to report lease expiry despite a local
  result having already been delivered. Reconcile, never replay uncertain effects.

## Verification and operations

Run the canonical hermetic runner in a disposable development environment:

```bash
scripts/run_tests.sh tests/gateway/test_thechat_work_leases.py \
  tests/gateway/test_thechat.py tests/gateway/test_thechat_attachments.py \
  tests/gateway/test_thechat_vault_unlock.py \
  tests/gateway/test_thechat_startup_vault_lifecycle.py \
  tests/e2e/test_thechat_queue_lifecycle.py
```

The workspace suite uses real loopback HTTP clients/servers, real signed webhook
admission, SQLite execution fencing, base adapter lifecycle and native runner
queue/cancellation methods. Agent/model execution and object-store upload are
synthetic; it does not contact production, a model provider or an external bot.
A paired real TheChat/Postgres/Redis acceptance run is a separate integration gate,
not something this suite claims to replace.

Monitor warning logs keyed by invocation ID for authority loss, rejected replay
and inability to acknowledge authority; correlate with TheChat lease/reconciliation
records. Existing message/progress tracing remains in place. Do not interpret a
heartbeat as user-visible progress, and never log credentials or work tokens.
