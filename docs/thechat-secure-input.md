# Native masked inputs in TheChat (v3)

TheChat is the prompt surface; Hermes owns the native consumers. A v3 prompt is
available only after bot-authenticated `/hermes-platform/health` advertises the
integer capability `capabilities.secureInput: 3` **and** the adapter has webhook
delivery configured. Missing, string, boolean, older or newer capability values
fail closed. The existing Bitwarden v2 lane remains independent, including its
shared-profile access semantics.

| Prompt kind | Exact encrypted fields | Authority | Consumer |
|---|---|---|---|
| `onepassword.unlock` | `password` | Bot owner | Native `op signin`; master is never persisted |
| `vault.save_login` | `identifier`, `password` | Bot owner, rechecked before persistence | Profile-local encrypted vault, bound to current browser origin |
| `vault.code` | `code` | Original admitted human requester | Origin-fenced browser fill; recovery codes are text, not numeric-only |
| `sudo.password` | `password` | Bot owner + profile opt-in | Local POSIX authentication-only pipe, then separate noninteractive command |
| `skill.secret` | `value` | Bot owner + profile opt-in | Existing profile-aware `.env` secret writer |

1Password remains owner-only even when its manager is already unlocked. Save-login
also checks owner authority after the prompt and immediately before the store call.
Identifiers retain their existing status as model-visible login metadata; passwords,
codes, master passwords and skill secrets are not model-visible.

## Opt in per execution profile

Settings belong in that profile's `config.yaml`, never `.env`:

```yaml
platforms:
  thechat:
    secure_input:
      allow_sudo: false
      allow_skill_secrets: false
```

Both flags default to false. Only a YAML boolean `true` enables them; strings,
integers, nulls and containers do not. The gateway's `PlatformConfig.from_dict`
promotes `secure_input` into adapter extras, so the documented keys above are the
loader-correct spelling. The turn reads the **execution profile's** gateway config,
not the receiving bot's possibly different transport profile. Set flags with:

```sh
hermes -p PROFILE config set platforms.thechat.secure_input.allow_sudo true
hermes -p PROFILE config set platforms.thechat.secure_input.allow_skill_secrets true
```

Policy is captured at turn entry; configure it before starting the turn. Current
owner, invocation lifetime and profile identity are rechecked while waiting and
before handing off the answer. Do not set `HERMES_INTERACTIVE` to enable these
prompts: it would also change unrelated approval policy.

Sudo capture does **not** approve the command or bypass existing command guards.
It is one-shot and not saved in `.env` or Hermes's interactive password cache.
The supported boundary is **foreground, local POSIX, one literal sudo command**:

- `sudo COMMAND ARG...`, optionally with separate `-u USER` / `--user USER` and
  `-g GROUP` / `--group GROUP` operands, and `--` before the command. Each selector
  may appear once. Names and quoted numeric `'#ID'` operands are supported.
- `sudo` resolves from the system default executable path, not the shell snapshot,
  aliases/functions or user PATH. The resolved binary and every parent directory
  must be root-owned and not group/world writable. Its canonical absolute path
  is also accepted. Custom sudo executables are not credential recipients.
- Arguments are literal argv (quoting/escaping via shlex); compound commands,
  pipelines, redirections, shell expansion/metacharacters (even quoted), wrappers,
  env assignments, and all other sudo flags/modes are refused **before capture**.
  Split compound work into separate approved calls. Background/PTY, delegated
  prompts and remote/container transports cannot use native capture.
- The exact single-line password is sent only to a direct `sudo -S -p '' ... -v`
  child. Its stdout/stderr are OS-level `/dev/null`, and its private stdin is closed
  before the actual command starts. Authentication is bounded to 15 seconds (or
  the smaller command timeout), with cancellation checks. CR/LF/NUL passwords
  cannot be represented by sudo's line protocol and are refused without execution.
- After authentication, a separate direct `sudo -n ... -- COMMAND ARG...` child
  receives **only caller stdin**, or `/dev/null`. Both children have Hermes as their
  parent for ppid-scoped timestamps; neither runs a shell/login rc/snapshot. The
  normal subprocess environment builder and foreground wait/cancellation machinery
  still apply. A cached passwordless validation cannot leak unread input to the
  command. Expired/mismatched timestamps or policies incompatible with `-v` fail
  noninteractively; no password fallback/retry is attempted. The OS sudo timestamp
  follows existing sudo policy, independently of Hermes's empty password cache.

An operator's pre-existing `SUDO_PASSWORD` configuration retains its **legacy**
behavior (including whole-command stdin on supported backends); this isolation
promise covers new native masked capture, not configured passwords or other
surfaces' inherited behavior.

Skill targets must be valid environment-variable names. Exact UTF-8, whitespace,
quotes and CR/LF are preserved through the existing secure writer's
`preserve_exact=True` path. It writes a standard double-quoted, single-line dotenv
value with the exact inline discriminator `# hermes:exact:v1`. Only these marked
assignments decode CR/LF in the scoped reader; unmarked historical double-quoted
`\\n`/`\\r` literals keep their prior scoped semantics. The real startup loader
also honors marked values without interpolation or ASCII normalization. Ordinary
unmarked startup values retain python-dotenv's existing interpretation and
credential sanitation. Marker-looking text **inside** the value is not a marker.
Secret setup returns only success/skip metadata. Existing CLI/Desktop capture
callbacks remain the fallback outside a native turn. A denied native turn installs
an explicit context-local denial, never borrowing a process-global callback.

## Protocol and lifecycle

Public progress types are `secure.input.request` and `secure.input.resolved`.
Requests contain only public context and an ephemeral RSA-3072 SPKI. The fixed
prompt kinds/field order above are the only supported forms. Each request expires
after 120 seconds. Resolution is exactly:

```json
{"version":3,"requestId":"UUID","sessionKey":"SESSION","outcome":"submitted"}
```

Other outcomes are `cancelled`, `expired`, and `failed`. These describe prompt
settlement, not whether the subsequent native consumer succeeded.

The encryption algorithm is `RSA-OAEP-3072-SHA256+A256GCM`. OAEP's label and GCM's
AAD are the same compact UTF-8 JSON array, in this order:

```text
[3,botId,requesterUserId,profileId,sessionKey,invocationId,conversationId,
 threadId-or-null,requestId,promptKind,fields,target,nonce,expiresAt]
```

Plaintext is a JSON object with exactly the declared fields. Duplicate keys,
extra/missing keys, non-string or empty values, invalid UTF-8, and invalid crypto
are rejected. Each value is at most 4096 UTF-8 bytes, identifiers at most 1024;
encoded plaintext is at most 16384 bytes. Wrapped key, IV and ciphertext are
canonical padded base64, decoding to 384 bytes, 12 bytes, and 17–16400 bytes.
The nonce is a canonical base64url encoding of 32 random bytes.

The signed direct relay type is `thechat.hermes_platform.secure_input`. Signature
and timestamp verification precede decoding/dispatch. The relay is handled before
the durable inbox, and all webhook responses use `Cache-Control: no-store`.
Ciphertext, private keys and plaintext handoffs are memory-only. Only public
metadata and bounded in-memory replay digests survive settlement. Matching retries
are acknowledged; conflicting retries cannot consume a waiter twice. Ambiguous
request-publication retries use the identical request rather than creating a new
key/nonce. Stop, replacement, lock (for 1Password), invocation finalization,
disconnect and expiry fence waiting consumers. A submitted-but-unconsumed answer
is cleared at teardown, including revocation during resolution publication.
Every v3 consumer also requires a successful authoritative resolution `SendResult`
before receiving the decrypted answer. HTTP denial, timeout, unavailable delivery
or publication exception clears the answer and releases its waiter, even when the
connection-time owner cache is unchanged. A resolution is not best-effort UI
cleanup for this handoff; no secret sink may ignore its failure.

TheChat is responsible for current human/conversation/bot authorization, current
owner validation, requester-only snapshot/realtime visibility, retention of pending
requests, and no-store client submission responses. No secret is sent through
chat, clarification or durable bot requests. A failed/unavailable prompt has no
plaintext fallback.

## Developer verification

From a fresh clone, no credentials or live password managers are needed:

```sh
uv sync --frozen --extra dev --extra messaging
scripts/run_tests.sh -j 2 --file-retries 0 \
  tests/gateway/test_thechat_secure_input.py \
  tests/gateway/test_thechat_secure_input_consumers.py \
  tests/gateway/test_thechat_secure_input_isolation.py \
  tests/gateway/test_thechat_secure_input_authorization.py \
  tests/tools/test_native_sudo_isolation.py \
  tests/hermes_cli/test_env_exact_values.py \
  tests/gateway/test_thechat_vault_unlock.py \
  tests/gateway/test_thechat_startup_vault_lifecycle.py \
  tests/tools/test_browser_vault.py \
  tests/tools/test_browser_vault_unlock_security.py \
  tests/tools/test_terminal_tool.py \
  tests/tools/test_subagent_sudo_prompt.py \
  tests/tools/test_skills_tool.py
```

These exercise the real turn runner, nested tool-worker propagation, registry
consumers, encrypted vault persistence, profile A→B→A under multiplexing, synthetic
`op`/`sudo` subprocesses, and a loopback HTTP health/signature/relay path. The sudo
fixture has no privileges; approval is a synthetic decision at the existing guard
boundary. Browser transport is a fixture; paired browser/WebCrypto/API acceptance
is a separate gate, not claimed by these tests.

Harness seams: `gateway.run_turn_runner_vault.vault_turn_scope(turn)` binds both
v2 and v3 on the executor thread; `TheChatAdapter._handle_webhook` receives the
signed relay; `SecureInputBroker.create/resolve` implements the wire boundary.
`tests/gateway/test_thechat_secure_input.py` exports independent `encrypt` and
`relay` helpers; `test_thechat_secure_input_consumers.py` exports `run_tool` and
`ReplyClient` for synthetic integration probes. Production never calls those test
helpers.
