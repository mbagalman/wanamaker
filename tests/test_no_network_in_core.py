"""CI gate: core code paths must not import HTTP / network libraries.

Per AGENTS.md Hard Rule 1 and FR-Privacy.1, core CLI commands must not make
outbound network calls.

This test enforces the rule architecturally by walking the import graph
of every core module and rejecting any transitive import of a known HTTP
or telemetry library.

The acceptance test in FR-Privacy.1 (running each command in a network-
isolated container and comparing output) lives separately as an
integration test; this is the cheap fast guardrail.

If you're adding a legitimate dependency that happens to use the network
in a non-core code path (docs build, example downloads), exclude it from
the core modules walked here and document the reason — don't widen
``BANNED_MODULES``.
"""

from __future__ import annotations

import importlib
import pkgutil

# Modules that must not appear anywhere in the core import graph.
BANNED_MODULES: frozenset[str] = frozenset(
    {
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "urllib.request",
        "http.client",
        "openai",
        "anthropic",
        "sentry_sdk",
        "posthog",
        "mixpanel",
        "segment",
    }
)

# Subpackages whose code paths back the core local workflow.
CORE_SUBPACKAGES: tuple[str, ...] = (
    "wanamaker.cli",
    "wanamaker.config",
    "wanamaker.data",
    "wanamaker.diagnose",
    "wanamaker.engine",
    "wanamaker.transforms",
    "wanamaker.model",
    "wanamaker.refresh",
    "wanamaker.forecast",
    "wanamaker.trust_card",
    "wanamaker.advisor",
    "wanamaker.reports",
    "wanamaker.artifacts",
    "wanamaker.seeding",
)


def _walk_submodules(root: str) -> list[str]:
    mod = importlib.import_module(root)
    if not hasattr(mod, "__path__"):
        return [root]
    out = [root]
    for info in pkgutil.walk_packages(mod.__path__, prefix=f"{root}."):
        out.append(info.name)
    return out


def test_core_modules_do_not_import_network_libraries() -> None:
    """Walk every core module and verify the import graph is network-free."""
    import sys

    before = set(sys.modules)
    for root in CORE_SUBPACKAGES:
        for name in _walk_submodules(root):
            importlib.import_module(name)
    after = set(sys.modules)

    loaded = after - before | {n for n in after if n.startswith("wanamaker")}
    offenders = sorted(loaded & BANNED_MODULES)
    assert not offenders, (
        "Core code paths transitively imported banned network/telemetry modules: "
        f"{offenders}. See AGENTS.md Hard Rule 1 and FR-Privacy.1."
    )
