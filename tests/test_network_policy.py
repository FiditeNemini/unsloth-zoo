# SPDX-License-Identifier: LGPL-3.0-or-later
"""unsloth_network_policy: parsing, require(), library env, and the socket guard.

No test touches the real network: allowed lookups and connections are either loopback or
go to a stubbed original.
"""

import asyncio
import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

import unsloth_network_policy as np_

ALL_VARS = (np_.ENV_VAR, "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", np_._OWNED_OFFLINE_VAR)


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for var in ALL_VARS:
        monkeypatch.delenv(var, raising = False)
    yield
    np_._uninstall_socket_guard()


@pytest.fixture
def guard():
    np_.install_socket_guard()
    yield
    np_._uninstall_socket_guard()


def set_env(monkeypatch, **policy):
    monkeypatch.setenv(np_.ENV_VAR, json.dumps(policy))


# ── parsing ─────────────────────────────────────────────────────────────────


def test_absent_policy_is_everything_off():
    policy = np_.current()
    assert policy == np_.Policy()
    assert not any(np_.allowed(service) for service in np_.SERVICES)


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "42", '{"enabled": "yes"}'])
def test_malformed_policy_is_everything_off(monkeypatch, raw):
    monkeypatch.setenv(np_.ENV_VAR, raw)
    assert not np_.current().enabled
    assert not np_.allowed("hf_hub")


def test_service_needs_master_switch(monkeypatch):
    set_env(monkeypatch, enabled = False, services = ["hf_hub"])
    assert not np_.allowed("hf_hub")
    set_env(monkeypatch, enabled = True, services = ["hf_hub"])
    assert np_.allowed("hf_hub")
    assert not np_.allowed("wandb")


def test_services_as_mapping_and_unknown_ids_dropped(monkeypatch):
    set_env(monkeypatch, enabled = True, services = {"hf_hub": True, "wandb": False, "from_the_future": True})
    policy = np_.current()
    assert policy.services == frozenset({"hf_hub"})


def test_unknown_service_query_is_an_error():
    with pytest.raises(ValueError):
        np_.allowed("hf")


def test_encode_round_trip():
    policy = np_.Policy(True, frozenset({"wandb", "hf_hub"}), allow_lan = True)
    assert np_._parse(np_.encode(policy)) == policy
    assert json.loads(np_.encode(policy))["services"] == ["hf_hub", "wandb"]


def test_policy_is_reread_when_env_changes(monkeypatch):
    assert not np_.allowed("hf_hub")
    set_env(monkeypatch, enabled = True, services = ["hf_hub"])
    assert np_.allowed("hf_hub")
    monkeypatch.delenv(np_.ENV_VAR)
    assert not np_.allowed("hf_hub")


# ── require ─────────────────────────────────────────────────────────────────


def test_require_names_the_setting_when_master_off():
    with pytest.raises(np_.NetworkDisabledError) as info:
        np_.require("hf_hub", "download unsloth/foo")
    message = str(info.value)
    assert "download unsloth/foo" in message
    assert "Network access and Hugging Face Hub" in message
    assert np_.ENV_VAR in message


def test_require_names_only_the_service_when_master_on(monkeypatch):
    set_env(monkeypatch, enabled = True, services = [])
    with pytest.raises(np_.NetworkDisabledError) as info:
        np_.require("wandb")
    assert "turn on Weights & Biases under Network access" in str(info.value)


def test_require_passes_when_allowed(monkeypatch):
    set_env(monkeypatch, enabled = True, services = ["wandb"])
    np_.require("wandb")


def test_error_is_an_oserror():
    assert issubclass(np_.NetworkDisabledError, OSError)


# ── library env ─────────────────────────────────────────────────────────────


def test_offline_vars_set_while_hub_off_and_removed_when_on(monkeypatch):
    np_.sync_library_env()
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert np_.os.environ[var] == "1"
    np_.set_policy({"enabled": True, "services": ["hf_hub"]})
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert var not in np_.os.environ


def test_user_offline_var_survives_enabling_hub(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    np_.sync_library_env()
    np_.set_policy({"enabled": True, "services": ["hf_hub"]})
    assert np_.os.environ["HF_HUB_OFFLINE"] == "1"
    assert "TRANSFORMERS_OFFLINE" not in np_.os.environ


def test_telemetry_opt_outs_always_set(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising = False)
    np_.set_policy({"enabled": True, "services": list(np_.SERVICES)})
    assert np_.os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_imported_hub_constant_follows_policy(monkeypatch):
    import sys, types
    fake = types.ModuleType("huggingface_hub.constants")
    fake.HF_HUB_OFFLINE = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)
    np_.sync_library_env()
    assert fake.HF_HUB_OFFLINE is True
    np_.set_policy({"enabled": True, "services": ["hf_hub"]})
    assert fake.HF_HUB_OFFLINE is False


# ── socket guard ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "host, lan, local",
    [
        (None, False, True),
        ("localhost", False, True),
        ("LOCALHOST.", False, True),
        ("studio.localhost", False, True),
        ("127.0.0.1", False, True),
        ("127.8.9.10", False, True),
        ("::1", False, True),
        ("[::1]", False, True),
        ("::ffff:127.0.0.1", False, True),
        ("0.0.0.0", False, True),
        ("example.com", False, False),
        ("8.8.8.8", False, False),
        ("192.168.1.20", False, False),
        ("192.168.1.20", True, True),
        ("10.0.0.5", True, True),
        ("fe80::1%en0", True, True),
        ("example.com", True, False),
        ("8.8.8.8", True, False),
    ],
)
def test_host_is_local(host, lan, local):
    assert np_._host_is_local(host, lan) is local


def test_guard_blocks_dns_lookup(guard):
    with pytest.raises(np_.NetworkDisabledError):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(np_.NetworkDisabledError):
        socket.gethostbyname("example.com")


def test_guard_blocks_connect_to_public_ip(guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(np_.NetworkDisabledError):
            sock.connect(("8.8.8.8", 53))
        with pytest.raises(np_.NetworkDisabledError):
            sock.connect_ex(("8.8.8.8", 53))


def test_guard_blocks_udp_sendto(guard):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(np_.NetworkDisabledError):
            sock.sendto(b"x", ("8.8.8.8", 53))


def _loopback_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    def accept_once():
        try:
            server.accept()[0].close()
        except OSError:
            pass  # the test closed the server first

    threading.Thread(target = accept_once, daemon = True).start()
    return server


def test_guard_allows_loopback(guard):
    server = _loopback_server()
    try:
        port = server.getsockname()[1]
        with socket.create_connection(("localhost", port), timeout = 5):
            pass
    finally:
        server.close()


def test_guard_blocks_urllib(guard):
    with pytest.raises(urllib.error.URLError) as info:
        urllib.request.urlopen("http://example.com/", timeout = 5)
    assert isinstance(info.value.reason, np_.NetworkDisabledError)


def test_guard_blocks_asyncio(guard):
    async def go():
        await asyncio.open_connection("example.com", 80)

    with pytest.raises(np_.NetworkDisabledError):
        asyncio.run(go())


def test_guard_lets_everything_through_when_master_on(guard, monkeypatch):
    calls = []
    monkeypatch.setitem(np_._originals, "getaddrinfo", lambda host, *a, **k: calls.append(host) or [])
    set_env(monkeypatch, enabled = True, services = [])
    assert socket.getaddrinfo("example.com", 443) == []
    assert calls == ["example.com"]


def test_guard_reads_policy_live(guard, monkeypatch):
    monkeypatch.setitem(np_._originals, "getaddrinfo", lambda *a, **k: [])
    with pytest.raises(np_.NetworkDisabledError):
        socket.getaddrinfo("example.com", 443)
    set_env(monkeypatch, enabled = True)
    assert socket.getaddrinfo("example.com", 443) == []


def test_guard_allows_lan_when_configured(guard, monkeypatch):
    set_env(monkeypatch, enabled = False, allow_lan = True)
    np_._check_address(socket.AF_INET, ("192.168.1.20", 80))
    with pytest.raises(np_.NetworkDisabledError):
        np_._check_address(socket.AF_INET, ("8.8.8.8", 80))


def test_guard_allows_unix_sockets(guard):
    np_._check_address(socket.AF_UNIX, "/tmp/some.sock")


def test_install_is_idempotent():
    np_.install_socket_guard()
    patched = socket.getaddrinfo
    np_.install_socket_guard()
    assert socket.getaddrinfo is patched
    np_._uninstall_socket_guard()
    assert socket.getaddrinfo is np_._originals["getaddrinfo"]


def test_activate_applies_env_and_guard():
    policy = np_.activate()
    assert policy == np_.Policy()
    assert np_.os.environ["HF_HUB_OFFLINE"] == "1"
    with pytest.raises(np_.NetworkDisabledError):
        socket.getaddrinfo("example.com", 443)
