"""Left-core migration at gateway start: install the catalog plugins a home still needs."""
from pathlib import Path


def recover_left_core_in(home: Path, *, hydrate_secrets: bool = True) -> list[str]:
    """Install the catalog plugin of every feature that left core (Home Assistant) *home* uses, in
    *home*'s runtime scope (its config, secrets and allow_lazy_installs); once per process per home.
    Returns the installed plugin names. Blocking: call it off the event loop."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli.left_core_migration import recover_at_startup
    with _profile_runtime_scope(Path(home), hydrate_secrets=hydrate_secrets):
        return recover_at_startup(gateway_start=True)


def recover_left_core_at_gateway_start() -> bool:
    """Left-core migration for the homes whose platform config the runner loads first: the launch
    home and the default root (the multiplex primary; a named launcher is then served as a
    secondary). Every secondary runs :func:`recover_left_core_in` before its plugins are discovered,
    so a platform plugin it gets starts its adapter in the same gateway start. True when a plugin
    that ships a gateway platform was installed: platform config loaded before it lacks the platform."""
    from hermes_cli.left_core_migration import LEFT_CORE
    from hermes_constants import get_default_hermes_root, get_hermes_home
    # The launch home too runs in its own scope: a multiplexing config load already armed fail-closed
    # secret reads, so an unscoped install read of its own .env raised. Its sources were hydrated at boot.
    launch = Path(get_hermes_home())
    installed = recover_left_core_in(launch, hydrate_secrets=False)
    root = Path(get_default_hermes_root())
    if root.resolve() != launch.resolve():
        installed += recover_left_core_in(root)
    return any(feature.platform and feature.plugin in installed for feature in LEFT_CORE)
