# SPDX-License-Identifier: LGPL-3.0-or-later
"""Process-wide network policy for Unsloth: offline unless the user turns access on.

One environment variable, ``UNSLOTH_NETWORK_POLICY``, carries the policy as JSON so every
process started from an Unsloth one (Studio workers are ``spawn`` children) inherits it:

    {"enabled": true, "services": ["hf_hub", "wandb"], "allow_lan": false}

Absent, empty or unparsable means everything off. ``enabled`` is the master switch; a
service is allowed only when the master switch is on and the service is listed.

Enforcement has two layers:

* Call sites that reach a service ask first: ``require("hf_hub")`` raises
  ``NetworkDisabledError`` with a message saying which setting to turn on.
* ``activate()`` installs a socket guard that refuses DNS lookups and connections to
  anything but loopback (plus private ranges with ``allow_lan``) while the master switch
  is off. It is the backstop for third-party libraries that make their own requests. The
  guard reads the policy on every call, so turning access on later takes effect at once.
  It cannot see subprocesses (git, curl, llama-server); those are gated at their call sites.

While Hugging Face access is off, ``activate()`` also sets ``HF_HUB_OFFLINE``,
``TRANSFORMERS_OFFLINE`` and ``HF_DATASETS_OFFLINE`` so those libraries use the local cache
instead of trying the network and failing on the guard.

Standard library only, and a top-level module rather than part of ``unsloth_zoo``, so a
process can import it without running ``unsloth_zoo/__init__.py``.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Union

__all__ = [
    "ENV_VAR",
    "SERVICES",
    "NetworkDisabledError",
    "Policy",
    "activate",
    "allowed",
    "current",
    "encode",
    "install_socket_guard",
    "require",
    "set_policy",
    "sync_library_env",
]

ENV_VAR = "UNSLOTH_NETWORK_POLICY"

# id -> (label, what it covers). The ids are the stable contract; labels are for UIs.
SERVICES: dict[str, tuple[str, str]] = {
    "hf_hub": (
        "Hugging Face Hub",
        "Model and dataset downloads, model search, and pushing to the Hub.",
    ),
    "modelscope": ("ModelScope", "Model downloads from ModelScope."),
    "github": ("GitHub", "Source and release downloads from GitHub."),
    "pypi": ("PyPI", "Package metadata and installs from PyPI."),
    "llm_providers": (
        "External AI providers",
        "Chat and data generation through OpenAI, Anthropic, Gemini and other hosted APIs.",
    ),
    "wandb": ("Weights & Biases", "Sending training metrics to wandb.ai."),
    "tunnel": ("Public tunnel", "Exposing Studio on a public Cloudflare URL."),
    "web_research": (
        "Web tools",
        "Web search, page fetching, YouTube transcripts and arXiv lookups.",
    ),
    "remote_media": (
        "Remote dataset media",
        "Images, video and audio that datasets reference by http(s) URL.",
    ),
    "mcp_remote": ("Remote MCP servers", "Tool servers reached over the network."),
}

# Set by sync_library_env so it only ever undoes offline variables it set itself.
_OWNED_OFFLINE_VAR = "UNSLOTH_NETWORK_POLICY_OWNS_OFFLINE"
_HF_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
# Opt-outs of library usage reporting; harmless whatever the policy says.
_ALWAYS_SET = {"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}

_LOCAL_NAMES = frozenset(("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"))


class NetworkDisabledError(PermissionError):
    """Raised when a network operation is refused by the policy.

    A PermissionError, hence an OSError: libraries that treat connection failures as
    "offline" (requests, urllib, huggingface_hub) handle it the same way.
    """


@dataclass(frozen = True)
class Policy:
    enabled: bool = False
    services: frozenset = field(default_factory = frozenset)
    allow_lan: bool = False

    def allows(self, service: str) -> bool:
        _check_service(service)
        return self.enabled and service in self.services

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "services": sorted(self.services),
            "allow_lan": self.allow_lan,
        }


def _check_service(service: str) -> None:
    if service not in SERVICES:
        raise ValueError(f"Unknown network service {service!r}; expected one of {sorted(SERVICES)}")


def _parse(raw: Optional[str]) -> Policy:
    if not raw or not raw.strip():
        return Policy()
    try:
        data = json.loads(raw)
    except ValueError:
        return Policy()
    if not isinstance(data, dict):
        return Policy()
    services = data.get("services", ())
    if isinstance(services, Mapping):
        services = [name for name, on in services.items() if on is True]
    if not isinstance(services, (list, tuple)):
        services = ()
    # Unknown ids are dropped, not fatal: a policy written by a newer version still loads.
    known = frozenset(s for s in services if isinstance(s, str) and s in SERVICES)
    return Policy(
        enabled = data.get("enabled") is True,
        services = known,
        allow_lan = data.get("allow_lan") is True,
    )


_cache_lock = threading.Lock()
_cache: tuple[Optional[str], Policy] = (None, Policy())


def current() -> Policy:
    """The policy in effect, read from the environment (cached per distinct value)."""
    global _cache
    raw = os.environ.get(ENV_VAR)
    cached_raw, cached = _cache
    if raw == cached_raw:
        return cached
    policy = _parse(raw)
    with _cache_lock:
        _cache = (raw, policy)
    return policy


def allowed(service: str) -> bool:
    return current().allows(service)


def require(service: str, what: Optional[str] = None) -> None:
    """Raise NetworkDisabledError unless ``service`` is allowed. ``what`` names the action."""
    policy = current()
    if policy.allows(service):
        return
    label = SERVICES[service][0]
    action = what or SERVICES[service][1].rstrip(".")
    if not policy.enabled:
        how = f"turn on Network access and {label} in Studio Settings"
    else:
        how = f"turn on {label} under Network access in Studio Settings"
    raise NetworkDisabledError(
        f"Unsloth: network access is off, so this cannot run: {action}. To allow it, {how}, "
        f"or set {ENV_VAR} (for example {encode(Policy(True, frozenset([service])))!r})."
    )


def encode(policy: Union[Policy, Mapping]) -> str:
    if not isinstance(policy, Policy):
        policy = _parse(json.dumps(dict(policy)))
    return json.dumps(policy.to_dict(), separators = (",", ":"))


def set_policy(policy: Union[Policy, Mapping, None]) -> Policy:
    """Make ``policy`` current for this process and the processes it starts."""
    if policy is None:
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = encode(policy)
    sync_library_env()
    return current()


# ── library environment ──────────────────────────────────────────────────────


def sync_library_env() -> None:
    """Point Hugging Face libraries at the local cache while Hub access is off.

    Variables the user set themselves are left alone; ones this module set are removed
    again once Hub access is allowed. Libraries already imported read their offline flag at
    import time, so their constants are updated too.
    """
    for key, value in _ALWAYS_SET.items():
        os.environ.setdefault(key, value)

    offline = not allowed("hf_hub")
    # Comma list of the offline variables this module set (and so may unset).
    owned = {v for v in os.environ.get(_OWNED_OFFLINE_VAR, "").split(",") if v in _HF_OFFLINE_VARS}
    if offline:
        for var in _HF_OFFLINE_VARS:
            if os.environ.get(var) != "1":
                os.environ[var] = "1"
                owned.add(var)
        if owned:
            os.environ[_OWNED_OFFLINE_VAR] = ",".join(sorted(owned))
    elif owned:
        for var in owned:
            os.environ.pop(var, None)
        os.environ.pop(_OWNED_OFFLINE_VAR, None)

    _sync_imported_constants(offline)


def _sync_imported_constants(offline: bool) -> None:
    import sys

    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None and hasattr(hub_constants, "HF_HUB_OFFLINE"):
        hub_constants.HF_HUB_OFFLINE = offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    datasets_config = sys.modules.get("datasets.config")
    if datasets_config is not None and hasattr(datasets_config, "HF_HUB_OFFLINE"):
        datasets_config.HF_HUB_OFFLINE = offline or os.environ.get("HF_DATASETS_OFFLINE") == "1"


# ── socket guard ─────────────────────────────────────────────────────────────

_guard_lock = threading.Lock()
_guard_installed = False
_originals: dict[str, object] = {}


def _host_is_local(host: object, allow_lan: bool) -> bool:
    if host is None:
        return True  # getaddrinfo(None, port): the wildcard or loopback address, no lookup
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    name = host.strip().rstrip(".").lower()
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]
    if not name or name in _LOCAL_NAMES or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return False  # a name: resolving it is already a network request
    if address.is_loopback or address.is_unspecified:
        return True
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
        if address.is_loopback:
            return True
    return allow_lan and (address.is_private or address.is_link_local)


def _refuse(host: object) -> NetworkDisabledError:
    return NetworkDisabledError(
        f"Unsloth: network access is off, so the connection to {host!r} was blocked. "
        f"Turn on Network access in Studio Settings, or set {ENV_VAR}."
    )


def _check_address(family: int, address: object) -> None:
    if getattr(socket, "AF_UNIX", None) is not None and family == socket.AF_UNIX:
        return
    policy = current()
    if policy.enabled:
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if not _host_is_local(host, policy.allow_lan):
        raise _refuse(host)


def _check_host(host: object) -> None:
    policy = current()
    if policy.enabled:
        return
    if not _host_is_local(host, policy.allow_lan):
        raise _refuse(host)


def install_socket_guard() -> None:
    """Patch the socket module so the master switch is enforced in this process. Idempotent."""
    global _guard_installed
    with _guard_lock:
        if _guard_installed:
            return
        sock_cls = socket.socket
        _originals.update(
            connect = sock_cls.connect,
            connect_ex = sock_cls.connect_ex,
            sendto = sock_cls.sendto,
            getaddrinfo = socket.getaddrinfo,
            gethostbyname = socket.gethostbyname,
            gethostbyname_ex = socket.gethostbyname_ex,
        )

        def connect(self, address):
            _check_address(self.family, address)
            return _originals["connect"](self, address)

        def connect_ex(self, address):
            _check_address(self.family, address)
            return _originals["connect_ex"](self, address)

        def sendto(self, data, *args):
            # sendto(data, address) or sendto(data, flags, address)
            if args:
                _check_address(self.family, args[-1])
            return _originals["sendto"](self, data, *args)

        def getaddrinfo(host, *args, **kwargs):
            _check_host(host)
            return _originals["getaddrinfo"](host, *args, **kwargs)

        def gethostbyname(host):
            _check_host(host)
            return _originals["gethostbyname"](host)

        def gethostbyname_ex(host):
            _check_host(host)
            return _originals["gethostbyname_ex"](host)

        sock_cls.connect = connect
        sock_cls.connect_ex = connect_ex
        sock_cls.sendto = sendto
        socket.getaddrinfo = getaddrinfo
        socket.gethostbyname = gethostbyname
        socket.gethostbyname_ex = gethostbyname_ex
        _guard_installed = True


def _uninstall_socket_guard() -> None:
    """Tests only."""
    global _guard_installed
    with _guard_lock:
        if not _guard_installed:
            return
        socket.socket.connect = _originals["connect"]
        socket.socket.connect_ex = _originals["connect_ex"]
        socket.socket.sendto = _originals["sendto"]
        socket.getaddrinfo = _originals["getaddrinfo"]
        socket.gethostbyname = _originals["gethostbyname"]
        socket.gethostbyname_ex = _originals["gethostbyname_ex"]
        _guard_installed = False


def activate() -> Policy:
    """Apply the current policy to this process: library environment, then the socket guard."""
    sync_library_env()
    install_socket_guard()
    return current()


def services_from(names: Iterable[str]) -> frozenset:
    """Validate service ids from user input."""
    names = list(names)
    for name in names:
        _check_service(name)
    return frozenset(names)
