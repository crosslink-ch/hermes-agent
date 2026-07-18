"""Resolve static named terminal execution targets.

The resolver is deliberately small and lazy: importing tool schemas never reads
user configuration, and configured names never enter a schema.  Legacy flat
terminal configuration stays on the existing environment-variable path.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Hashable, Iterable, Mapping, Optional


class ExecutionTargetError(ValueError):
    """Raised when terminal target configuration or selection is invalid."""


_NAMED_TARGET_BACKENDS = frozenset({
    "local", "docker", "ssh", "modal", "daytona", "singularity",
})


_effective_config_override: ContextVar[dict[str, Any] | None] = ContextVar(
    "hermes_execution_target_config", default=None,
)
_frozen_config_active: ContextVar[bool] = ContextVar(
    "hermes_execution_target_frozen", default=False,
)
_classic_config_lock = threading.RLock()
_classic_config_override: dict[str, Any] | None = None
_fingerprint_keys: dict[Path, bytes] = {}
_fingerprint_key_lock = threading.Lock()


def _target_fingerprint_key() -> bytes:
    """Return a stable, private machine/profile key for exposed target digests."""
    from hermes_constants import get_hermes_home

    path = Path(get_hermes_home()) / ".execution-target-fingerprint-key"
    with _fingerprint_key_lock:
        cached = _fingerprint_keys.get(path)
        if cached is not None:
            return cached

        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = os.urandom(32)
        temp = path.with_name(
            f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        )
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(candidate)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                pass
            except OSError:
                # Filesystems without hard-link support still get O_EXCL
                # publication; readers below wait for the winning writer.
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(candidate)
                        handle.flush()
                        os.fsync(handle.fileno())
        finally:
            try:
                temp.unlink()
            except OSError:
                pass

        key = b""
        for _ in range(20):
            try:
                key = path.read_bytes()
            except OSError:
                key = b""
            if len(key) >= 32:
                break
            time.sleep(0.01)
        if len(key) < 32:
            raise ExecutionTargetError(
                f"Could not initialize private execution-target key at {path}."
            )
        try:
            path.chmod(0o600)
        except OSError:
            pass
        key = key[:32]
        _fingerprint_keys[path] = key
        return key


def set_execution_target_config_source(config: Mapping[str, Any] | None) -> None:
    """Use an entry point's already-effective config for target resolution.

    The classic CLI has project-fallback and ``--ignore-user-config`` rules
    that differ from the shared loader. Registering its merged result here
    keeps tool routing on the same authority boundary.
    """
    global _classic_config_override
    value = deepcopy(dict(config)) if isinstance(config, Mapping) else None
    _effective_config_override.set(value)
    with _classic_config_lock:
        _classic_config_override = deepcopy(value)


@contextmanager
def execution_target_config_scope(config: Mapping[str, Any]):
    """Temporarily pin target resolution in the current context only."""
    token = _effective_config_override.set(deepcopy(dict(config)))
    frozen_token = _frozen_config_active.set(True)
    try:
        yield
    finally:
        _frozen_config_active.reset(frozen_token)
        _effective_config_override.reset(token)


def _active_profile_scope() -> str:
    """Return a stable multiplex-profile scope, empty in legacy mode."""
    try:
        from agent.secret_scope import is_multiplex_active
        if not is_multiplex_active():
            return ""
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home().resolve())
        return hashlib.sha256(home.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


@dataclass(frozen=True)
class ExecutionTargetResolution:
    """A resolved execution target and its inherited terminal configuration."""

    target: str
    backend: str
    config: Mapping[str, Any]
    named: bool
    is_default: bool = True
    profile_scope: str = ""

    @property
    def spec_fingerprint(self) -> str:
        """Hash the fully inherited target spec used to create resources.

        Target names are mutable configuration aliases, not security identities.
        Including the complete resolved spec prevents a hot config edit from
        reusing an environment created for a different host/backend/image.
        """
        canonical = json.dumps(
            {
                "config": dict(self.config),
                "is_default": self.is_default,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=repr,
        )
        return hmac.new(
            _target_fingerprint_key(),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:20]

    @property
    def security_scope(self) -> str:
        """Return the profile/name/spec identity for approvals and credentials."""
        value = f"{self.profile_scope}:{self.target}:{self.spec_fingerprint}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def scope_task_key(self, task_key: Hashable) -> Hashable:
        if not self.profile_scope:
            return task_key
        return f"profile-{self.profile_scope}:{task_key}"

    def environment_key(self, task_key: Hashable) -> Hashable:
        """Scope an already-collapsed environment key to this target."""
        scoped = self.scope_task_key(task_key)
        return (scoped, self.target) if self.named else scoped

    def session_key(self, raw_session_key: Optional[str]) -> Hashable:
        """Scope per-session state without changing legacy string keys."""
        key = str(self.scope_task_key(str(raw_session_key or "default")))
        return (key, self.target) if self.named else key

    def backend_task_id(self, task_key: Hashable) -> str:
        """Return a bounded backend-safe isolation id.

        Docker labels are limited to 63 characters. Keep fixed hash suffixes of
        both the complete task identity and target spec so long task IDs cannot
        truncate away the only target-distinguishing bytes.
        """
        base = str(self.scope_task_key(task_key))
        if not self.named:
            return base
        base_digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
        return f"task-{base_digest}-target-{self.security_scope}"

    def storage_task_id(self, base_task_id: str) -> str:
        """Stable, bounded identity for persistent storage owned by a target.

        Runtime identity includes the full effective config so a repointed
        target cannot reuse a stale container. Persistent filesystem ownership
        intentionally depends only on profile + target name, keeping data stable
        across timeout, lifetime, image, and other policy edits.
        """
        if not self.named:
            return base_task_id
        base_digest = hashlib.sha256(
            base_task_id.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:16]
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home().resolve())
        owner = f"{self.profile_scope}\0{home}\0{self.target}"
        owner_digest = hashlib.sha256(owner.encode("utf-8")).hexdigest()[:20]
        return f"task-{base_digest}-storage-{owner_digest}"

    def legacy_backend_task_id(self, base_task_id: str) -> str:
        """Return the pre-fingerprint named-target backend ID for migration."""
        base = str(self.scope_task_key(base_task_id))
        if not self.named:
            return base
        target_digest = hashlib.sha256(
            self.target.encode("utf-8")
        ).hexdigest()[:12]
        return f"{base}-target-{target_digest}"

    def metadata(self, *, cwd: Optional[str] = None) -> dict[str, Any]:
        data: dict[str, Any] = {"target": self.target, "backend": self.backend}
        if self.named:
            data["runtime_scope"] = self.security_scope
        if cwd:
            data["cwd"] = cwd
        return data


def _load_unscoped_config() -> dict[str, Any]:
    """Load the live source while ignoring a nested frozen RPC snapshot."""
    try:
        from agent.secret_scope import is_multiplex_active

        multiplex = is_multiplex_active()
    except Exception:
        multiplex = False
    if not multiplex:
        with _classic_config_lock:
            override = deepcopy(_classic_config_override)
        if override is not None:
            return override
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    return config if isinstance(config, dict) else {}


def _load_merged_config() -> dict[str, Any]:
    """Load merged config lazily so importing fixed tool schemas stays cheap."""
    override = _effective_config_override.get()
    if override is not None:
        return deepcopy(override)
    return _load_unscoped_config()


def _available(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(none)"


def resolve_execution_target(
    target: Optional[str] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> ExecutionTargetResolution:
    """Resolve *target* against merged ``terminal`` configuration.

    With no non-empty ``terminal.targets`` mapping, only omitted target and
    explicit ``"default"`` are valid and the returned configuration marks the
    legacy env-driven path.  With named targets, top-level terminal settings
    are inherited and the selected target mapping overrides them.
    """
    if target is not None and (not isinstance(target, str) or not target):
        raise ExecutionTargetError("Execution target must be a non-empty string.")

    root = config if config is not None else _load_merged_config()
    profile_scope = _active_profile_scope()
    if not isinstance(root, Mapping):
        raise ExecutionTargetError("Terminal configuration must be a mapping.")
    terminal = root.get("terminal", {})
    if terminal is None:
        terminal = {}
    if not isinstance(terminal, Mapping):
        raise ExecutionTargetError("terminal must be a mapping.")

    raw_targets = terminal.get("targets")
    if raw_targets is None or raw_targets == {}:
        if target not in (None, "default"):
            raise ExecutionTargetError(
                f"Unknown execution target {target!r}. Named targets are not configured. "
                f"Available targets: {_available(['default'])}."
            )
        flat = dict(terminal)
        flat.pop("targets", None)
        flat.pop("default_target", None)
        # Legacy flat mode preserves the historical TERMINAL_ENV precedence.
        # Metadata must describe the backend that will actually execute, not a
        # stale config.yaml value hidden by an explicit launcher/.env override.
        backend = str(
            os.getenv("TERMINAL_ENV")
            or flat.get("backend")
            or flat.get("env_type")
            or "local"
        ).strip().lower() or "local"
        return ExecutionTargetResolution(
            target="default", backend=backend, config=flat, named=False,
            is_default=True, profile_scope=profile_scope,
        )

    if not isinstance(raw_targets, Mapping):
        raise ExecutionTargetError("terminal.targets must be a mapping.")

    invalid_names = [
        name for name in raw_targets
        if not isinstance(name, str) or not name
    ]
    if invalid_names:
        raise ExecutionTargetError("terminal.targets names must be non-empty strings.")

    names = sorted(raw_targets)
    for name in names:
        if not isinstance(raw_targets[name], Mapping):
            raise ExecutionTargetError(
                f"terminal.targets[{name!r}] must be a mapping. "
                f"Available targets: {_available(names)}."
            )

    selected = target if target is not None else terminal.get("default_target")
    if not isinstance(selected, str) or not selected or selected not in raw_targets:
        if target is not None:
            prefix = f"Unknown execution target {target!r}."
        else:
            prefix = (
                "terminal.default_target must be a non-empty name present in "
                "terminal.targets."
            )
        raise ExecutionTargetError(
            f"{prefix} Available targets: {_available(names)}."
        )

    merged = {
        key: value
        for key, value in terminal.items()
        if key not in {"targets", "default_target"}
    }
    merged.update(dict(raw_targets[selected]))
    backend = str(
        merged.get("backend") or merged.get("env_type") or "local"
    ).strip().lower() or "local"
    if backend not in _NAMED_TARGET_BACKENDS:
        available_backends = ", ".join(sorted(_NAMED_TARGET_BACKENDS))
        raise ExecutionTargetError(
            f"Execution target {selected!r} has unknown backend {backend!r}. "
            f"Available backends: {available_backends}."
        )
    return ExecutionTargetResolution(
        target=selected, backend=backend, config=merged, named=True,
        is_default=selected == terminal.get("default_target"),
        profile_scope=profile_scope,
    )


def resolve_live_execution_target(
    target: Optional[str] = None,
) -> ExecutionTargetResolution:
    """Resolve against the live source when nested RPC routing is frozen."""
    if not _frozen_config_active.get():
        return resolve_execution_target(target)
    return resolve_execution_target(target, config=_load_unscoped_config())


def list_execution_targets(
    *, config: Optional[Mapping[str, Any]] = None,
) -> tuple[ExecutionTargetResolution, ...]:
    """Return configured targets in deterministic order for runtime guidance.

    Tool schemas remain static for prompt-cache stability; callers such as the
    system-prompt environment hint can use this lightweight runtime inventory
    to tell the model which fixed-string target values are available.
    """
    root = config if config is not None else _load_merged_config()
    terminal = root.get("terminal", {}) if isinstance(root, Mapping) else {}
    targets = terminal.get("targets") if isinstance(terminal, Mapping) else None
    if not isinstance(targets, Mapping) or not targets:
        return (resolve_execution_target(config=root),)
    # Resolve the default first so name/default validation uses the same clear
    # ExecutionTargetError shape as an ordinary omitted-selector tool call.
    resolve_execution_target(config=root)
    return tuple(
        resolve_execution_target(name, config=root)
        for name in sorted(targets)
    )
