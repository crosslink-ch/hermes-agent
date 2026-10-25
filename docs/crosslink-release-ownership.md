# Crosslink distribution and release ownership

This is a downstream-only maintenance contract. It is not an upstream release pipeline.

## Pinned sync

- Upstream: `v0.22.5`, annotated tag object `56b6a31fededdf7364e1a0a79548d144e7fd81a3`, commit `bf0203a794a07aac3ff764972ab9087baa4e4f4e`.
- Fork parent: `37324f90833e68eb917c3e5b3dd4cecaafbb54bc`.
- PR destination: **crosslink-ch/hermes-agent**, never NousResearch/hermes-agent.

## Ownership boundary

`.github/workflows/`, `.github/actions/`, `scripts/release.py`, and `scripts/releases/`
are taken unchanged from the pinned upstream revision. Fork release targets, credentials,
version sequencing, promotion logic and publication fallbacks do not belong in those files.
Upstream's installer-test harness and sandbox release-tag selectors are likewise unchanged.
The change classifier includes the downstream distribution-identity module as an update-test
input; this is product test coverage, not a release controller.

The separate application in `crosslink-ch/hermes-release` owns downstream release orchestration:
immutable source specification, isolated tests/build, artifact receipt, approval and publication.
The release tag `crosslink-v1.0.0`, package version `1.0.0`, upstream version `v0.22.5`, source SHA
and image digest are different identifiers. Legacy `crosslink-v2026.10.12` date tags must not
be sorted into the new SemVer sequence. No source-install signed-channel migration is implied.

Restoring upstream workflow bytes does **not** make fork Actions safe to run. Before pushing a
candidate, opening its PR or creating a release, the operator must review the fork's repository
Actions settings and explicitly disable/otherwise contain unowned publisher, schedule,
privileged and runner-heavy workflows. Do not provide upstream workflows with fallback fork
publication credentials. The old fork Docker workflow responds to `release: published`; the
pinned upstream replacement removes that trigger but still has PR/main and reusable-workflow
entry points. Neither fact is permission to publish. Repository settings are an external gate,
not a reason to maintain fork release logic in upstream YAML.

## Downstream behavior retained

- Source installs, passive checks, archive fallback, reinstall hints and desktop bootstrap
  retain `crosslink-ch/hermes-agent` and `https://share.kihub.ch/hermes` authority.
- Unconfigured source installs follow `main`. Crosslink stable/canary/preview source channels
  fail closed until a separate source-channel migration is designed. They never consult the
  Nous signed feed or interpret the independent release service's tags as source subscriptions.
  Generic channel protocol tests explicitly select upstream authority; separate downstream
  tests exercise the real fork defaults.
- Known legacy Nous installer origins require migration consent; unrelated custom origins
  remain untouched. Windows clones set `core.autocrlf=false` before checkout.
- TheChat retains its adapter, signed requests, queue/session continuity, structured progress,
  attachment and direct-message routing, native clarification/approval and encrypted vault
  unlock. Platform registration bridges it into upstream's registry-owned environment,
  authorization, standalone sending and cron delivery paths.
- Compression retains todo snapshots, reply anchors and rotation-state behavior.
- SQLite journal selection and fail-open FTS repair remain downstream behavior.
- Honcho remains bundled for existing downstream configurations, including content-digest
  identity invalidation. Its `honcho-ai==2.2.0` SDK is a locked optional PM extra, not a call to
  retired lazy-dependency APIs. Existing identity/config data is not migrated to another provider.
- Vault errors remain redacted rather than returning password-manager stderr.

## Local validation

Use a disposable Linux worker and an isolated home. No controller publication credentials or
live user profile may reach source tests. Use the candidate's PM contract, not a pre-PM command
copied from the fork parent:

```sh
export HOME=/path/to/disposable-home
export HERMES_HOME="$HOME/.hermes"
export TMPDIR=/path/to/short-scratch
scripts/run-in-hermes-env python --version
HERMES_TEST_WORKERS=6 HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh --file-timeout 600
scripts/run_tests.sh --include-integration \
  tests/e2e/test_gateway_core_session_flows.py \
  tests/e2e/test_thechat_queue_lifecycle.py
scripts/run-in-hermes-env python scripts/check --only ruff
```

The runner uses its own disk-backed `/var/tmp/hermes-pytest-<uid>` scratch on Linux; merely
exporting `TMPDIR` does not override that runner policy. Default discovery excludes Docker,
E2E and integration roots. Never call a default pass proof of those suites, Windows/macOS,
real inference, a published image, or a managed-fleet rollout.

A release still requires exact-commit build and image/runtime evidence in the independent
release application's approval flow. Do not use the current stock KI Hub managed smoke script
with live secrets. Do not merge or publish automatically.
