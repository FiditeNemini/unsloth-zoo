# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Decide, per machine, whether to START a download on Xet or go straight to HTTP.

``hf_xet_fallback`` recovers from a bad Xet attempt, but recovery costs the whole stalled attempt
EVERY time. Where Xet reliably fails (blocked CAS endpoint, a proxy that mangles range requests, too
little RAM) that toll repeats forever, so this module remembers the outcomes of real downloads and
forgets them after a while in case the cause was temporary. It never probes the network itself.

Design rules:
  * Never block a download: every failure path answers "use Xet", since a wrong "healthy" costs one
    fallback while a wrong "unhealthy" downgrades a working machine.
  * Cheapest checks first: override, hf_xet presence, RAM, then the remembered verdict.
  * A verdict is scoped to what could invalidate it (hf_xet version, endpoint), and expires.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .hf_xet_tuning import MIN_XET_RAM_BYTES, system_profile

logger = logging.getLogger(__name__)

__all__ = [
    "XetHealth",
    "xet_health",
    "record_xet_outcome",
    "clear_xet_health",
    "health_state_path",
]

STATE_FILENAME = "unsloth_xet_health.json"

# A healthy verdict is re-probed weekly; a demotion is retried the next day, as its causes (flaky
# link, transient CAS outage, loaded box) are usually short-lived.
HEALTHY_TTL_SECONDS = 7 * 24 * 3600
DEMOTED_TTL_SECONDS = 24 * 3600

# One bad download is noise (a dropped wifi packet fails HTTP too); two in a row is a pattern.
DEMOTION_THRESHOLD = 2


_TRUTHY = {"1", "true", "yes", "on"}
_LOCK = threading.Lock()
# Serialises the read-modify-write on the state FILE, which _LOCK (in-memory memo only) does not
# cover. A success RESETS the streak while a failure INCREMENTS it, so interleaving the two records
# two consecutive failures that never happened and demotes a healthy machine to HTTP for a day.
# Concurrent writers are ordinary here: same-repo GGUF variants run at once, and every rank of a
# multi-rank launch calls from_pretrained.
_STATE_MUTEX = threading.Lock()
# How long a writer waits for a peer PROCESS before writing unserialised. The critical section is one
# small read plus one os.replace, and degrading is safe: never blocking a download outranks the race.
_STATE_LOCK_TIMEOUT = 5.0
# (timestamp, verdict, was_probed). The probe flag is part of the key because the download path calls
# with probe = False every time: without it that cheap lookup memoizes the optimistic "defaulting to
# Xet" answer and disarms an explicit preflight for a minute, on exactly the CAS-blocked machine the
# probe exists to catch.
_CACHED: "Optional[tuple[float, XetHealth, bool]]" = None
_MEMO_SECONDS = 60.0  # a snapshot download asks repeatedly; don't re-probe per file
# Bumped whenever a recorded outcome invalidates the memo. _evaluate() runs OUTSIDE the lock, so
# without this a verdict read before a concurrent demotion could be published after it and keep
# sending downloads to Xet for another memo window.
_GENERATION = 0


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


@dataclass(frozen = True)
class XetHealth:
    use_xet: bool
    reason: str
    source: str

    def __bool__(self) -> bool:  # `if xet_health():` reads as "is Xet usable"
        return self.use_xet


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT") or "https://huggingface.co"


def _hf_xet_version() -> Optional[str]:
    try:
        import hf_xet  # type: ignore
    except Exception:
        return None
    version = getattr(hf_xet, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as _version

        return str(_version("hf_xet"))
    except Exception:
        return "unknown"


def _machine_id() -> str:
    """Identity of the box a verdict was measured on.

    HF_HOME is routinely shared across a cluster, so without this one node with blocked CAS persists
    an HTTP verdict every node honours -- and no node then starts on Xet, so nothing can record the
    success that would clear it before the TTL expires.
    """
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path) as f:
                value = f.read().strip()
            if value:
                return value
        except OSError:
            pass
    try:
        import platform

        return platform.node() or "unknown"
    except Exception:
        return "unknown"


def _state_filename() -> str:
    """One state file PER MACHINE.

    Scoping only the READ was half a fix: record_xet_outcome still incremented whatever streak the
    last writer left, so on a shared HF_HOME one node's failure demoted a healthy peer, and a healthy
    peer's success kept zeroing a broken node's streak so it could never demote.
    """
    safe = "".join(c for c in _machine_id() if c.isalnum())[:32] or "unknown"
    return f"unsloth_xet_health.{safe}.json"


def health_state_path() -> Optional[Path]:
    """Where the verdict lives: beside the HF cache, so it follows a relocated ``HF_HOME``."""
    try:
        from .hf_cache import _active_caches

        from .hf_cache import _is_writable

        hf_home, hub_cache, _ = _active_caches()
        # HF_HOME resolves to a default even when read-only, so preferring it unconditionally sent
        # the verdict to an unwritable path whenever only HF_HUB_CACHE was relocated. Writes then
        # failed silently and, the streak being rebuilt from this file alone, the machine could never
        # be demoted no matter how often Xet stalled.
        candidates = [hf_home, hub_cache.parent if hub_cache is not None else None]
        for base in candidates:
            if base is not None and _is_writable(base):
                return base / _state_filename()
        base = next((c for c in candidates if c is not None), None)
        return base / _state_filename() if base is not None else None
    except Exception:
        return None


@contextlib.contextmanager
def _state_file_guard():
    """Hold the cross-process lock on the state file, or fall through unserialised.

    The lock is a SIDECAR file: _write_state swaps the state file's inode via os.replace, so writers
    locking the state file itself would hold locks on different objects.
    """
    lock = None
    try:
        from filelock import FileLock   # a huggingface_hub dependency, always installed

        path = health_state_path()
        if path is None:
            # No state file to serialise, and a lock path built from "None" would litter the cwd.
            yield
            return
        lock = FileLock(str(path) + ".lock", timeout = _STATE_LOCK_TIMEOUT)
        lock.acquire()
    except Exception as e:  # Timeout, read-only cache dir, filelock somehow absent
        logger.debug("Xet health state lock unavailable (%s); writing unserialised", e)
        lock = None
    try:
        yield
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def _read_state() -> dict:
    path = health_state_path()
    if path is None:
        return {}
    try:
        with open(path, "r") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(state: dict) -> None:
    """Atomic replace so a concurrent reader never sees a half-written verdict."""
    path = health_state_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        fd, tmp = tempfile.mkstemp(dir = str(path.parent), prefix = ".xet_health_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.debug("Could not persist Xet health state: %s", e)


def _state_is_current(state: dict) -> bool:
    """A verdict only speaks for the version and endpoint it was measured against."""
    if not state or "verdict" not in state:
        return False
    if state.get("hf_xet_version") != _hf_xet_version():
        return False
    if state.get("endpoint") != _endpoint():
        return False
    if state.get("machine") != _machine_id():
        # A foreign node's verdict on a shared cache. Losing it falls open to Xet, the safe side.
        return False
    ttl = HEALTHY_TTL_SECONDS if state.get("verdict") == "xet" else DEMOTED_TTL_SECONDS
    try:
        return (time.time() - float(state.get("ts", 0))) < ttl
    except (TypeError, ValueError):
        return False


def xet_health(*, force: bool = False, probe: bool = True) -> XetHealth:
    """Whether a download should START on Xet. Cheap and safe to call per download."""
    global _CACHED, _GENERATION

    if _is_true(os.environ.get("UNSLOTH_DISABLE_XET")) or _is_true(
        os.environ.get("UNSLOTH_STABLE_DOWNLOADS")
    ) or _is_true(os.environ.get("HF_HUB_DISABLE_XET")):
        return XetHealth(False, "Xet disabled by environment", "forced")
    if _is_true(os.environ.get("UNSLOTH_FORCE_XET")):
        return XetHealth(True, "Xet forced by environment", "forced")

    with _LOCK:
        generation = _GENERATION
        if (
            not force
            and _CACHED is not None
            and (time.monotonic() - _CACHED[0]) < _MEMO_SECONDS
            # An unprobed optimistic default only satisfies a caller that did not ask to probe; any
            # real verdict short-circuits every caller.
            and (_CACHED[2] or not probe or _CACHED[1].source != "default")
        ):
            return _CACHED[1]

    result = _evaluate(force = force, probe = probe)
    with _LOCK:
        if generation == _GENERATION:
            _CACHED = (time.monotonic(), result, probe)
    return result


def _evaluate(*, force: bool, probe: bool) -> XetHealth:
    if _hf_xet_version() is None:
        return XetHealth(False, "hf_xet is not installed", "unavailable")

    profile = system_profile()
    if profile.total_ram_bytes and profile.total_ram_bytes < MIN_XET_RAM_BYTES:
        gb = profile.total_ram_bytes / 1e9
        return XetHealth(
            False,
            f"only {gb:.1f}GB RAM available to this process; HTTP uses far less",
            "low-ram",
        )

    state = _read_state()
    if not force and _state_is_current(state):
        verdict = state.get("verdict") == "xet"
        return XetHealth(verdict, str(state.get("reason") or "remembered verdict"), "cached")

    # No fresh verdict: Xet is the better default, and the fallback ladder covers a bad guess.
    # There is no network probe; verdicts come only from real downloads (record_xet_outcome).
    return XetHealth(True, "no cached verdict; defaulting to Xet", "default")


def record_xet_outcome(ok: bool, reason: str = "") -> None:
    """Feed a finished Xet attempt back in. Two consecutive failures demote the machine to HTTP.

    Called from the download ladder, so it must never raise and never slow the caller down.
    """
    global _CACHED, _GENERATION
    try:
        # The whole read-modify-write: the streak comes from disk, so a peer landing between the
        # read and the write would be dropped.
        with _STATE_MUTEX, _state_file_guard():
            _record_outcome_locked(ok, reason)
        with _LOCK:
            _CACHED = None  # the next call must see the new verdict
            _GENERATION += 1
    except Exception as e:
        logger.debug("Could not record Xet outcome: %s", e)


def _record_outcome_locked(ok: bool, reason: str) -> None:
    """Body of record_xet_outcome. Caller holds _STATE_MUTEX and the state file guard."""
    state = _read_state()
    # A streak recorded against a different hf_xet build says nothing about this one.
    if state.get("hf_xet_version") != _hf_xet_version() or state.get("endpoint") != _endpoint():
        state = {}
    failures = 0 if ok else int(state.get("consecutive_failures") or 0) + 1

    if ok:
        verdict, note = "xet", reason or "Xet download succeeded"
    elif failures >= DEMOTION_THRESHOLD:
        verdict = "http"
        note = reason or f"Xet failed {failures} times in a row on this machine"
        logger.warning(
            "Xet has now failed %d times in a row; new downloads will use HTTP for the next "
            "%dh (set UNSLOTH_FORCE_XET=1 to override).",
            failures, DEMOTED_TTL_SECONDS // 3600,
        )
    else:
        # Below the threshold, keep whatever verdict stands rather than promoting on a failure.
        verdict = str(state.get("verdict") or "xet")
        note = str(state.get("reason") or "")

    _write_state({
        "verdict": verdict,
        "reason": note,
        "ts": time.time(),
        "hf_xet_version": _hf_xet_version(),
        "endpoint": _endpoint(),
        "machine": _machine_id(),
        "consecutive_failures": failures,
    })


def clear_xet_health() -> None:
    """Forget the verdict (used by tests and by an explicit user reset)."""
    global _CACHED, _GENERATION
    with _LOCK:
        _CACHED = None
        # Bump the generation too, or an evaluation started before this clear passes its generation
        # guard and republishes the verdict we just forgot.
        _GENERATION += 1
    path = health_state_path()
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass
