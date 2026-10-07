# TheChat native Bitwarden unlock

Bitwarden access is shared by an agent's Hermes profile, not by bot ownership. A human who has already passed the existing gateway and conversation admission rules may initiate an unlock in their own invocation. The secure request and its resolution are private to that requester; no other human, including the bot owner, may submit or cancel it. This does not change admission, membership, or conversation permissions.

After a successful native unlock, admitted sessions in that same profile may list metadata, fill saved passwords, and use saved authenticator codes. Ending the initiating session does not relock the shared token. Pending unlock work remains session-owned: ending a session fences its own pending prompts and late token commits without invalidating another session's pending unlock or an already-shared token. Explicit Lock, 30-minute idle expiry, and process shutdown still discard tokens. Profiles remain isolated. CLI/Desktop token lifetime is unchanged. **1Password is not shared by this change** and retains TheChat's existing bot-owner access rule, including the fail-closed rule when owner metadata is unavailable.

## Version 2 wire contract

Hermes and TheChat must roll out this contract together. All Bitwarden request, resolution, and response envelopes use `version: 2`. Requests/responses retain `requesterUserId` and omit `ownerUserId`; ownership is not a shared credential principal. Legacy v1 responses, unknown fields, and mismatched context are rejected rather than translated. A mixed-version installation cannot complete the secure prompt: update both sides together, retire in-flight old prompts, and retry with a fresh invocation. There is no plaintext/chat fallback.

Request metadata contains `requestId`, `sessionKey`, `profileId`, `backend: "bitwarden"`, `requesterUserId`, `nonce`, `expiresAt`, `algorithm: "RSA-OAEP-3072-SHA256+A256GCM"`, and `publicKeySpkiB64`. The ordered authenticated context is compact UTF-8 JSON with no ASCII escaping:

```text
[2, botId, requesterUserId, profileId, sessionKey, invocationId, conversationId,
 threadId-or-null, requestId, "bitwarden", nonce, expiresAt]
```

Both RSA-OAEP's label and AES-256-GCM's additional authenticated data use those exact bytes. The master password is decoded as strict UTF-8, without trimming or normalization. Its UTF-8 byte length must be 1–4096.

The dedicated signed native relay carries the matching request fields (without the public key), plus `id`, `requestType: "vault.unlock.request"`, `invocationId`, `conversationId`, `threadId`, `actorUserId`, and `action: "submit" | "cancel"`. The actor must equal the initiating requester. Submit adds `wrappedKeyB64`, `ivB64`, and `ciphertextB64`; cancel contains no encrypted fields. Exact retries are idempotent; altered retries conflict. Request expiry, cancellation, interruption, disconnect, and native lock-generation fences remain enforced.

The relay bypasses the durable inbox. Ciphertext and master passwords are not chat messages or durable records; only public prompt metadata and bounded replay digests are retained. RSA private keys and unconsumed plaintext are cleared on settlement/teardown. A native resolution reports only prompt outcome, not whether the subsequent manager unlock succeeded.

## Native integration seams

`gateway.run_turn_runner_vault.vault_turn_scope(turn)` binds the verified source/requester context on the executor thread and restores prior callbacks/session context on exit. Tool workers must propagate that context. Do not expose authority or sharing options as model arguments.

For a synthetic transport harness, `VaultUnlockBroker.create(context=..., profile_id=..., session_key=...)` creates an in-memory request; `resolve({"type": "thechat.hermes_platform.vault_unlock", "interaction": ...})` accepts only the exact v2 relay. Exercise `TheChatAdapter._handle_webhook` with a correctly signed body when testing the transport boundary, rather than bypassing signature verification with a direct broker call. Native backend calls remain fenced by `unlock_attempt` and `store_session_token`; no harness needs real manager credentials.
