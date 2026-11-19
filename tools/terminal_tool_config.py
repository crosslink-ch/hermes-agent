"""Terminal backend configuration: scope-aware TERMINAL_* reads, env-var parsing,
container-cwd sanity checks, plugin-backend classification, and the ``_quiet``
best-effort block shared by the terminal_tool_* modules.

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

import functools
import logging
import json
import os
import posixpath
import re
from contextlib import contextmanager
from typing import Any, Optional, overload

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


@contextmanager
def _quiet(label: str, *args, exc=Exception, level: int = logging.DEBUG):
    """Best-effort block: swallow *exc*, log *label* (``%``-formatted with *args*)
    at *level* with the traceback. For janitor/advisory work that must never
    take the terminal tool down."""
    try:
        yield
    except exc:
        logger.log(level, label, *args, exc_info=True)


def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an env var with *converter*, raising a clear ValueError on bad
    values (e.g. TERMINAL_TIMEOUT=5m) instead of an opaque crash. TERMINAL_*
    names are read scope-aware via :func:`_tenv`."""
    raw = _tenv(name, default) if name.startswith("TERMINAL_") else os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


# (current home, Hermes root) -> profile name; only homes that exist are stored (see home_profile_name).
_HOME_PROFILE_NAMES: dict[tuple[str, str], str] = {}


def home_profile_name() -> str:
    """Profile name of the current Hermes home (the process home, or a routed turn's bound home);
    ``"default"`` when it cannot be resolved. One lookup for every reader that must agree: the
    persistent-Docker container key, the container's profile label, the name the TUI stamps on its
    sessions (``get_active_profile_name``, so ``custom`` for a home inside the root that is not a
    ``profiles/<name>`` home), and the file panel's sandbox names.

    Memoized per (home, root) path pair: the lookup resolves three paths and runs on every
    container-key resolution, several times per tool call."""
    try:
        from hermes_cli.profiles import get_active_profile_name
        from hermes_constants import get_default_hermes_root, get_hermes_home

        home = get_hermes_home()
        key = (str(home), str(get_default_hermes_root()))
        if (cached := _HOME_PROFILE_NAMES.get(key)) is not None:
            return cached
        name = get_active_profile_name() or "default"
        if home.exists():  # a home created later may resolve differently
            _HOME_PROFILE_NAMES[key] = name
        return name
    except (ImportError, OSError, RuntimeError, ValueError):  # an unresolvable home keeps the shared default key
        return "default"


def profile_container_key(profile: str) -> str:
    """Persistent-Docker container key of a profile: the default profile stays literally ``"default"``."""
    return "default" if profile == "default" else f"profile:{profile}"


def _safe_getcwd() -> str:
    """``os.getcwd()`` tolerant of a deleted cwd (FileNotFoundError) or a macOS
    TCC-protected one without Full Disk Access (PermissionError); falls back
    to TERMINAL_CWD, then the home directory."""
    try:
        return os.getcwd()
    except (FileNotFoundError, PermissionError):
        return _tenv("TERMINAL_CWD") or os.path.expanduser("~")


# Host-cwd shapes that cannot exist inside a container sandbox: POSIX user dirs and ANY Windows
# drive path (``D:\\...``, ``e:/...``) as they leak toward a Linux ``-w`` flag.
_HOST_CWD_PREFIXES = ("/Users/", "/home/")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_host_cwd(path: str) -> bool:
    return path.startswith(_HOST_CWD_PREFIXES) or bool(_WINDOWS_DRIVE_RE.match(path))


def _is_windows_drive_path(path: str) -> bool:
    """True for any ``D:\\...`` / ``e:/...`` path. Not a username check."""
    return bool(path) and bool(_WINDOWS_DRIVE_RE.match(path))


def _host_path_key(path: str) -> str:
    """Compare host paths without caring about slash style or drive-letter case.

    A trailing slash is not significant, except we never collapse a drive root
    (``C:/``) into a bare ``C:`` that would prefix-match every path on that drive.
    """
    text = (path or "").replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = text[0].lower() + text[1:]
    if len(text) > 3:
        text = text.rstrip("/")
    return text


def translate_mounted_host_path(path: str, host_root: str, container_root: str) -> str | None:
    """Map *path* onto *container_root* when it is *host_root* or a child of it.

    Returns None when *path* is not under that host directory. Slash style and
    drive-letter case do not matter; a sibling directory (``proj`` vs ``proj-other``)
    is not a child.
    """
    if not isinstance(path, str) or not isinstance(host_root, str) or not isinstance(container_root, str):
        return None
    if not path or not host_root or not container_root:
        return None
    key = _host_path_key(path)
    root = _host_path_key(host_root)
    if not key or not root:
        return None
    mount = container_root.rstrip("/") or "/"
    if key == root:
        return mount
    prefix = root if root.endswith("/") else root + "/"
    if not key.startswith(prefix):
        return None
    return f"{mount}/{key[len(prefix):]}"


def cwd_follows_host_mount(cwd: str, mount: str) -> bool:
    """True when *cwd* was the host workspace (or the assumed ``/workspace`` view of it).

    An explicit in-container path other than that assumption is left alone.
    """
    if not mount or not cwd or cwd == mount:
        return False
    if _is_unusable_container_cwd(cwd):
        return True
    return cwd == "/workspace" and mount != "/workspace"

_CONTAINER_BACKENDS = frozenset({"docker", "modal", "daytona", "vercel_sandbox"})
_BUILTIN_BACKENDS = _CONTAINER_BACKENDS | {"local", "ssh", "managed_modal"}


def _plugin_registry_lookup(env_type: str, fn_name: str, default, *args):
    """Call ``agent.terminal_env_registry.<fn_name>(env_type, *args)`` for a
    plugin backend. Fail-soft: *default* for built-in/empty backends, when the
    registry is unavailable, or when the provider raises — a misbehaving plugin
    must never take the terminal tool down."""
    if not env_type or env_type in _BUILTIN_BACKENDS:
        return default
    try:
        import agent.terminal_env_registry as reg

        return getattr(reg, fn_name)(env_type, *args)
    except Exception:
        return default


def _plugin_env_flag(env_type: str, attr: str, default=False):
    """Classification flag of a plugin-registered backend (fail-soft, see above)."""
    return _plugin_registry_lookup(env_type, "provider_flag", default, attr, default)


def _is_container_backend(env_type: str) -> bool:
    """True for built-in container backends and plugins declaring ``is_container``."""
    return env_type in _CONTAINER_BACKENDS or _plugin_env_flag(env_type, "is_container")


def _get_plugin_env_provider(env_type: str):
    """Return the registered plugin provider for *env_type*, or None."""
    return _plugin_registry_lookup(env_type, "get_provider", None)


def plugin_sandbox_home(env: Any) -> str | None:
    """The home a plugin environment started in (``_build_plugin_env`` stamps it per environment, so two
    sandboxes of one backend never share it), or None."""
    home = getattr(env, "_hermes_sandbox_home", None)
    return home if isinstance(home, str) and posixpath.isabs(home) and home != "/" else None


def _in_sandbox_home(cwd: Any, env: Any) -> bool:
    """True for *cwd* at or under *env*'s sandbox home: a ``cd`` inside that sandbox, not a host workspace,
    even when it looks like one (``/home/...``)."""
    home = plugin_sandbox_home(env)
    if not home or not isinstance(cwd, str) or not posixpath.isabs(cwd):
        return False
    home, cwd = posixpath.normpath(home), posixpath.normpath(cwd)
    return cwd == home or cwd.startswith(home + "/")


@functools.lru_cache(maxsize=32)
def _launch_dirs(home: str, native_home: str) -> frozenset[str]:
    """The lexical launch set for one Hermes home. The root depends on that home and the platform default home
    (``native_home``, which keys the cache too); the install root is fixed per process. Cached: the shared root
    memo holds a single entry and thrashed when called once per command."""
    from hermes_cli._startup_fast import project_root_str
    from hermes_constants import get_default_hermes_root

    root = posixpath.normpath(str(get_default_hermes_root(home=home)))
    return frozenset({home, root, posixpath.join(home, "home"), posixpath.join(root, "home"), project_root_str()})


def _is_container_launch_dir(path: str) -> bool:
    """True when *path* (absolute) IS a directory Hermes starts in inside a container: the Hermes home, the
    Hermes root (``/opt/data`` in the official image, which is also the image user's HOME), ``home/`` under
    each, or the install root (``/opt/hermes``, as its real path). Those exist only in the container, never on
    an SSH target. The candidate is compared lexically: a symlink on this machine says nothing about the target.
    On a host they are left alone, since an SSH target may share the layout (``~/.hermes`` on an NFS home)."""
    from hermes_constants import _get_platform_default_hermes_home, get_hermes_home, is_container

    if not is_container():
        return False
    home = posixpath.normpath(str(get_hermes_home()))
    return posixpath.normpath(path) in _launch_dirs(home, str(_get_platform_default_hermes_home()))


@overload
def coerce_ssh_remote_cwd(cwd: str, env_type: str | None) -> str: ...
@overload
def coerce_ssh_remote_cwd(cwd: None, env_type: str | None) -> None: ...
def coerce_ssh_remote_cwd(cwd: str | None, env_type: str | None) -> str | None:
    """Cwd to send to an SSH backend.

    ``~``-prefixed paths stay literal so the remote shell expands them to the
    SSH user's home. The Hermes process's subprocess home (``/opt/data/home``
    in the official Docker image) is a directory on the machine running
    Hermes: it and anything under it are rewritten onto the remote ``~``, since
    ``cd`` into the host path exits 126 on the target. In a container, the
    directories Hermes starts in (the Hermes home, its root ``/opt/data``,
    which is also the image user's HOME, ``home/`` under each, the install
    root) map to ``~`` too; folders below them are kept, except under the
    subprocess home (above), since a remote may use the same path. A
    subprocess home that is the OS user's real home is left alone: a remote
    path may legitimately match it. Other backends are unchanged.
    """
    if not isinstance(cwd, str) or (env_type or "").strip().lower() != "ssh":
        return cwd
    text = cwd.strip()
    if not text or text.startswith("~"):
        return text or "~"
    if not posixpath.isabs(text):
        return text
    if _is_container_launch_dir(text):
        return "~"
    from hermes_constants import get_real_home, get_subprocess_home

    home = get_subprocess_home()
    if not home or posixpath.normpath(home) == posixpath.normpath(get_real_home()):
        return text
    rel = posixpath.relpath(posixpath.normpath(text), posixpath.normpath(home))
    if rel == ".":
        return "~"
    return text if rel == ".." or rel.startswith("../") else f"~/{rel}"


def _is_mounted_host_cwd(cwd: str, mounted_host: str | None) -> bool:
    """True when *cwd* is the host directory mounted at ``/workspace``.

    Checked before the ``/Users`` / ``/home`` / drive-letter heuristic: a WSL
    checkout under ``/mnt/...`` or an absolute path under ``/srv/...`` is
    ``os.path.isabs`` and would otherwise look like a valid container workdir.
    """
    if not cwd or not isinstance(mounted_host, str) or not mounted_host:
        return False
    try:
        left = os.path.normpath(os.path.abspath(os.path.expanduser(cwd)))
        right = os.path.normpath(os.path.abspath(os.path.expanduser(mounted_host)))
    except (OSError, ValueError):
        return False
    return left == right


def _is_unusable_container_cwd(cwd: str, *, mounted_host: str | None = None) -> bool:
    """True if *cwd* is a host or relative path that can't be a container
    workdir: ``docker run -w`` needs an absolute in-sandbox path, otherwise the
    container fails to start (exit 125). Windows drive paths aren't ``isabs``
    on POSIX, so they're caught by the prefix check.

    The directory mounted at ``/workspace`` is unusable inside the container
    even when it matches none of those prefixes. That equality check runs
    first so ``/mnt/...`` and ``/srv/...`` are not wrapped as ``cd`` targets.
    """
    if not cwd:
        return False
    if _is_mounted_host_cwd(cwd, mounted_host):
        return True
    return _is_host_cwd(cwd) or not os.path.isabs(cwd)


def _tenv(name: str, default: str = "") -> str:
    """Scope-aware read of a ``TERMINAL_*`` variable. Every terminal setting
    must go through this: under gateway multiplexing the active profile's
    config arrives via a per-turn scope, and a raw ``os.getenv`` would read
    whatever a previous turn pinned into the process env (cross-profile leak)."""
    from tools.terminal_scope import terminal_env

    return terminal_env(name, default)


def _tenv_bool(name: str, default: str) -> bool:
    """Scope-aware boolean ``TERMINAL_*`` read: true/1/yes (case-insensitive)."""
    return _tenv(name, default).lower() in {"true", "1", "yes"}


def _apply_task_cwd_override(
    config: dict[str, Any], cwd: str, cwd_override: str | None,
) -> str:
    """Apply a task workspace cwd without leaking host paths into containers.

    Docker's explicit mount-cwd mode is the exception: a registered host
    workspace should become the bind source and commands should run in
    ``/workspace``. Other container backends fall back to the target's already
    sanitized configured cwd.
    """
    env_type = config.get("env_type")
    if (
        env_type == "docker"
        and config.get("docker_mount_cwd_to_workspace")
        and isinstance(cwd_override, str)
        and cwd_override.strip()
    ):
        candidate = os.path.abspath(os.path.expanduser(cwd_override))
        is_host_path = (
            any(candidate.startswith(prefix) for prefix in _HOST_CWD_PREFIXES)
            or (
                os.path.isabs(candidate)
                and os.path.isdir(candidate)
            )
        )
        if is_host_path:
            config["host_cwd"] = candidate
            return "/workspace"
    if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
        return config["cwd"]
    return cwd


def _get_env_config(terminal_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return canonical terminal config for legacy env vars or a target mapping.

    ``terminal_config is None`` is the historical flat/env-driven path.  A
    selected named target passes its inherited mapping here directly; values
    are parsed without mutating ``os.environ`` so concurrent target calls
    cannot affect each other.
    """
    from tools import terminal_tool as _tt  # late-bound: tests and callers patch these names there
    from hermes_cli.config_defaults import DEFAULT_SANDBOX_IMAGE as default_image
    if terminal_config is None:
        _tt._ensure_terminal_env_bridged()

    def _get(key: str, env_name: str, default: Any) -> Any:
        if terminal_config is None:
            return _tt._tenv(env_name, str(default) if not isinstance(default, (list, dict)) else json.dumps(default))
        return terminal_config.get(key, default)

    def _coerce(value: Any, converter: Any, label: str, key: str) -> Any:
        if converter is json.loads and isinstance(value, (list, dict)):
            return value
        try:
            return converter(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            source = (
                f"terminal target setting {key}"
                if terminal_config is not None
                else f"TERMINAL_{key.upper()}"
            )
            raise ValueError(f"Invalid value for {source}: {value!r} (expected {label}).")

    def _bool(key: str, env_name: str, default: bool) -> bool:
        value = _get(key, env_name, default)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if terminal_config is not None:
            raise ValueError(
                f"Invalid value for terminal target setting {key}: {value!r} "
                "(expected boolean)."
            )
        # Preserve the legacy env-var behavior for unknown strings.
        return False

    def _json_shape(
        key: str,
        env_name: str,
        default: Any,
        expected_type: type,
        label: str,
    ) -> Any:
        value = _coerce(_get(key, env_name, default), json.loads, "valid JSON", key)
        if not isinstance(value, expected_type):
            source = (
                f"terminal target setting {key}"
                if terminal_config is not None
                else env_name
            )
            raise ValueError(
                f"Invalid value for {source}: {value!r} (expected {label})."
            )
        return value

    env_type = str(
        _get(
            "backend", "TERMINAL_ENV",
            terminal_config.get("env_type", "local") if terminal_config else "local",
        )
    ).strip().lower() or "local"

    mount_docker_cwd = _bool(
        "docker_mount_cwd_to_workspace", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", False,
    )
    container_backend = env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}
    docker_backend = env_type == "docker"

    # Docker/container-only env vars may be bridged from config.yaml even when
    # the active backend is local/ssh.  Do not parse their JSON/numeric payloads
    # until a backend that can consume them is selected; a stale or invalid
    # Docker value should not make local terminal/execute_code unusable.
    if container_backend:
        container_cpu = _coerce(_get("container_cpu", "TERMINAL_CONTAINER_CPU", 1), float, "number", "container_cpu")
        container_memory = _coerce(_get("container_memory", "TERMINAL_CONTAINER_MEMORY", 5120), int, "integer", "container_memory")
        container_disk = _coerce(_get("container_disk", "TERMINAL_CONTAINER_DISK", 51200), int, "integer", "container_disk")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _json_shape(
            "docker_forward_env", "TERMINAL_DOCKER_FORWARD_ENV", [], list, "list",
        )
        docker_volumes = _json_shape(
            "docker_volumes", "TERMINAL_DOCKER_VOLUMES", [], list, "list",
        )
        docker_env = _json_shape(
            "docker_env", "TERMINAL_DOCKER_ENV", {}, dict, "mapping",
        )
        docker_extra_args = _json_shape(
            "docker_extra_args", "TERMINAL_DOCKER_EXTRA_ARGS", [], list, "list",
        )
        docker_shm_size = str(_get("docker_shm_size", "TERMINAL_DOCKER_SHM_SIZE", "1g") or "")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []
        docker_shm_size = "1g"

    # Default cwd: local uses the host's current directory, ssh uses the
    # remote home, Vercel uses its documented workspace root, and everything
    # else starts in the backend's default root-like cwd.
    default_cwd = _tt._backend_default_cwd(env_type)

    # Read TERMINAL_CWD but sanity-check it for container backends.
    # If Docker cwd passthrough is explicitly enabled, remap the host path to
    # /workspace and track the original host path separately. Otherwise keep the
    # normal sandbox behavior and discard host paths.
    cwd = str(_get("cwd", "TERMINAL_CWD", default_cwd) or default_cwd)
    if env_type == "local" and cwd in {".", "./", "auto", "cwd"}:
        cwd = _tt._safe_getcwd()
    from hermes_cli.config import _is_ssh_remote_tilde_cwd
    if cwd and cwd != default_cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = (
            (_tt._tenv("TERMINAL_CWD") or _tt._safe_getcwd())
            if terminal_config is None
            else (cwd or _tt._safe_getcwd())
        )
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            any(candidate.startswith(p) for p in _tt._HOST_CWD_PREFIXES)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif _tt._is_container_backend(env_type) and cwd:
        # Host paths and relative paths that won't work inside containers
        if _tt._is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    if terminal_config is None:
        cwd, host_cwd = _tt._resolve_config_cwd(env_type, mount_docker_cwd)

    return {
        "env_type": env_type,
        "modal_mode": _tt.coerce_modal_mode(_get("modal_mode", "TERMINAL_MODAL_MODE", "auto")),
        "docker_image": str(_get("docker_image", "TERMINAL_DOCKER_IMAGE", default_image)),
        "docker_forward_env": docker_forward_env,
        "modal_image": str(_get("modal_image", "TERMINAL_MODAL_IMAGE", default_image)),
        "daytona_image": str(_get("daytona_image", "TERMINAL_DAYTONA_IMAGE", default_image)),
        "vercel_runtime": str(_get("vercel_runtime", "TERMINAL_VERCEL_RUNTIME", "")).strip(),
        "docker_image_pinned": _bool("docker_image_pinned", "TERMINAL_DOCKER_IMAGE_PINNED", False),
        "vercel_image": str(_get("vercel_image", "TERMINAL_VERCEL_IMAGE", "")).strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _coerce(_get("timeout", "TERMINAL_TIMEOUT", 180), int, "integer", "timeout"),
        "lifetime_seconds": _coerce(_get("lifetime_seconds", "TERMINAL_LIFETIME_SECONDS", 300), int, "integer", "lifetime_seconds"),
        # SSH-specific config
        "ssh_host": str(_get("ssh_host", "TERMINAL_SSH_HOST", "")),
        "ssh_user": str(_get("ssh_user", "TERMINAL_SSH_USER", "")),
        "ssh_port": _coerce(_get("ssh_port", "TERMINAL_SSH_PORT", 22), int, "integer", "ssh_port"),
        "ssh_key": str(_get("ssh_key", "TERMINAL_SSH_KEY", "")),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local defaults on.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": _bool(
            "ssh_persistent", "TERMINAL_SSH_PERSISTENT",
            _bool("persistent_shell", "TERMINAL_PERSISTENT_SHELL", True),
        ),
        "local_persistent": _bool("local_persistent", "TERMINAL_LOCAL_PERSISTENT", True),
        # Container resource config (applies to docker, modal,
        # daytona, and vercel_sandbox -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": _bool("container_persistent", "TERMINAL_CONTAINER_PERSISTENT", True),
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": _bool("docker_run_as_host_user", "TERMINAL_DOCKER_RUN_AS_HOST_USER", False),
        "docker_snap_compat": _bool("docker_snap_compat", "TERMINAL_DOCKER_SNAP_COMPAT", False),
        "docker_network": _bool("docker_network", "TERMINAL_DOCKER_NETWORK", True),
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process container reuse (issue #20561).  The docs claim
        # "ONE long-lived container shared across sessions" — this toggle
        # makes that real by probing for a labeled container at startup and
        # attaching to it instead of always starting a fresh one.  Set to
        # ``false`` for hard per-process isolation (no reuse, container is
        # removed on exit).
        "docker_persist_across_processes": _bool(
            "docker_persist_across_processes", "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", True,
        ),
        "docker_shared_container_key": str(_get("docker_shared_container_key", "TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "")).strip(),
        # Startup orphan reaper for hermes-tagged containers left behind by
        # crashed / SIGKILL'd previous processes that bypassed atexit.
        # Conservative: only sweeps Exited containers older than 2× the
        # idle-reap window AND scoped to the current profile. Issue #20561.
        "docker_orphan_reaper": _bool(
            "docker_orphan_reaper", "TERMINAL_DOCKER_ORPHAN_REAPER", True,
        ),
    }
