"""Terminal defaults, including named-target inheritance and sandbox policy."""


def terminal_defaults(sandbox_image: str, vercel_image: str) -> dict:
    """Return a fresh terminal section with the canonical sandbox images."""
    return {
        "backend": "local",
        # Optional named execution targets. Empty preserves the legacy flat
        # terminal config and environment-variable behavior.
        "default_target": "",
        "targets": {},
        "modal_mode": "auto",
        # Remote-backend connection-class failures (SSH host unreachable, Docker daemon down):
        # "warn" = structured degraded tool result with reason + retry hint; "fail" = raise error +
        # traceback.
        "degraded_mode": "warn",
        "cwd": ".",  # Use current directory
        # Root for terminal session temp files (background logs/pid/exit files, code-exec
        # sandboxes). Empty = TMPDIR/TMP/TEMP if set, else HERMES_HOME/cache/terminal (auto-pruned
        # after 72h) — NOT tmpfs /tmp, which is RAM-capped and fills under load. Must be an existing
        # absolute POSIX path; user-set paths are never auto-pruned.
        "temp_dir": "",
        # CSS font-family for the desktop app's xterm.js terminal (e.g. "'CaskaydiaCoveNerdFont',
        # monospace"). Empty = built-in default ("'JetBrains Mono', 'Cascadia Code', 'SF Mono',
        # Menlo, Consolas, monospace"). Lets users use a Nerd Font without patching the app.
        "font_family": "",
        "timeout": 180,
        # Seconds between SIGTERM and escalated SIGKILL for host process trees (browser daemons). 0
        # = SIGTERM only.
        "daemon_term_grace_seconds": 2.0,
        # Max seconds a one-shot CLI run (-q/-Q/-z) lingers for tracked notify_on_complete
        # background processes to finish. The dying parent owns their stdout pipes, so exiting
        # immediately kills the delivery (e.g. Bot Mode handoff replies via message_agent /
        # bot_relay). Plain background processes without notify_on_complete are never waited on. 0
        # disables.
        "oneshot_completion_wait_seconds": 600.0,
        "process_wait_auto_notify": True,  # 2 unchanged wait/poll -> notify_on_complete + stop; False = old polling
        # Env vars passed into sandboxed terminal/execute_code (plus skill-declared required_environment_variables).
        "env_passthrough": [],
        "client_env_passthrough": [],  # Extra non-secret toolchain names a CLI/TUI client sends for its tools
        # Remote-backend sync-back refuses to extract a downloaded state archive larger than this
        # (bytes); raise it for a ~/.hermes tree that legitimately exceeds 2 GiB.
        "sync_back_max_bytes": 2 * 1024 * 1024 * 1024,
        # HOME for host tool subprocesses: "auto" = host keeps the real OS-user HOME, containers use
        # HERMES_HOME/home; "real" = force real HOME; "profile" = force HERMES_HOME/home when it
        # exists (strict per-profile isolation).
        "home_mode": "auto",
        # Extra files sourced in the login shell when building the per-session env snapshot — for
        # nvm/pyenv/asdf/PATH entries registered by files a bash login shell skips (~/.bashrc,
        # ~/.zshrc, ~/.zprofile). Supports ~ and ${VAR}; missing files skipped. When empty and the
        # shell is bash, ~/.profile, ~/.bash_profile, ~/.bashrc are auto-sourced in that order (see
        # auto_source_bashrc).
        "shell_init_files": [],
        # Source ~/.profile, ~/.bash_profile, ~/.bashrc in the snapshot login shell to capture PATH
        # additions, functions, and aliases that `bash -l -c` misses (bash skips bashrc when
        # non-interactive; Debian/Ubuntu ~/.bashrc short-circuits). ~/.profile and ~/.bash_profile
        # go first because n/nvm/asdf write PATH exports there without an interactivity guard. Turn
        # off if an rc file misbehaves when sourced non-interactively (exits on TTY check).
        "auto_source_bashrc": True,
        # The default sandbox for every container backend: the nikolaik/python-nodejs base
        # (Python 3.13 / Node 26) plus a display stack, so Bot Screen, computer_use and the
        # bot's browser run INSIDE the sandbox and the pane can watch them (see bot_desktop).
        "docker_image": sandbox_image,
        "docker_forward_env": [],
        # Exact key-value env pairs set inside Docker containers (unlike docker_forward_env, which
        # reads host values) — useful under systemd without the user's shell env. Example:
        # {"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock"}
        "docker_env": {},
        "modal_image": sandbox_image,
        "daytona_image": sandbox_image,
        "vercel_image": vercel_image,  # vercel_sandbox backend only: a Vercel-managed or VCR image
        "vercel_runtime": "",  # deprecated by Vercel; a legacy runtime pin (node24 | node22 | python3.13) overrides vercel_image
        # Container limits (docker, modal, daytona, vercel_sandbox, container plugin backends; not local/ssh).
        "container_cpu": 1,
        "container_memory": 5120,       # MB (default 5GB)
        "container_disk": 51200,        # MB (default 50GB)
        "container_persistent": True,   # Persist filesystem across sessions
        # Docker volume mounts, "host_path:container_path" (docker -v syntax), e.g.
        # ["/home/user/.hermes/cache/documents:/output"]. For gateway MEDIA delivery, write to
        # /output/... inside Docker and emit the host-visible path in MEDIA:, not the container one.
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,  # mount host cwd at /workspace (weakens isolation)
        "docker_network": True,  # false = --network=none, no network access from commands
        "docker_extra_args": [],        # Extra flags passed verbatim to docker run
        # /dev/shm size for the Docker sandbox. Docker's 64 MB default silently breaks
        # Chromium/Playwright and PyTorch DataLoader workers; tmpfs is lazily allocated so the
        # higher ceiling is free until used. "" or "0" = omit the flag (Docker default).
        "docker_shm_size": "1g",
        # Run the container as the host uid:gid (`--user`) so files written to bind mounts
        # (docker_volumes, persistent workspace, mounted cwd) are owned by you, not root. Off by
        # default for images whose entrypoints must start as root (e.g. the bundled Hermes image,
        # which drops to `hermes` via s6-setuidgid). When on, SETUID/SETGID caps are omitted.
        "docker_run_as_host_user": False,
        # Snap-packaged Docker under AppArmor (Ubuntu cloud images; LP#1908448) refuses to exec
        # anything under `--init` or `--security-opt no-new-privileges` ("operation not
        # permitted"). True drops those two flags; every other hardening stays. See #9730.
        "docker_snap_compat": False,
        # Trusted profiles sharing one Docker container identity; empty = per-profile boundary.
        "docker_shared_container_key": "",
        # SSH: carry exported env vars between commands (cwd always carries). false = every command
        # starts from the login env, profile functions included. Local: TERMINAL_LOCAL_PERSISTENT.
        "persistent_shell": True,
    }
