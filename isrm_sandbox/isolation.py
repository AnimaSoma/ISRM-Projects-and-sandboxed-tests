"""Isolation primitives for the ISRM validation sandbox.

Each guarantee below is enforced at runtime and verified by tests/test_protocol.py.

1. No network. Creating a socket raises SandboxViolation, so nothing can fetch
   data, models or results during a run.
2. No imports from the original ISRM project trees. A meta-path finder resolves
   every import and refuses any module whose file lives under a forbidden root
   (by default the whole drive the sandbox lives on) unless it is under an
   allowed root (by default this sandbox and its virtual environment). sys.path
   is scrubbed of forbidden entries first so the author's packages cannot be
   picked up by accident.
3. Derived seeds. A seed is a SHA-256 of (spec_hash, arm, replicate). Seeds are
   therefore reproducible on any machine and cannot be hand-picked.
4. Input provenance. Every external input is referenced by SHA-256 in the frozen
   spec and verified before a run.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SANDBOX_ROOT = Path(__file__).resolve().parent.parent


class SandboxViolation(RuntimeError):
    """Raised when code inside the sandbox tries to cross an isolation boundary."""


# --------------------------------------------------------------------------- hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: str | os.PathLike, suffixes: Sequence[str] = (".py", ".json", ".toml")) -> str:
    """Content hash of a source tree: sorted relative paths and file bytes."""
    root = Path(root)
    h = hashlib.sha256()
    skip = {".venv", ".uv-cache", "__pycache__", ".pytest_cache", ".pytest_tmp", "reports"}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in suffixes:
            continue
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        h.update(str(p.relative_to(root)).replace("\\", "/").encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def derive_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary parts (spec hash, arm, replicate)."""
    joined = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big") >> 1


# --------------------------------------------------------------------------- network
def _blocked(*_args, **_kwargs):
    raise SandboxViolation("network access is disabled inside the ISRM sandbox")


class _BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        _blocked()


_NETWORK_STATE: dict = {"active": False, "original": None}


def enable_no_network() -> None:
    """Replace socket construction with a raising stub. Idempotent."""
    if _NETWORK_STATE["active"]:
        return
    _NETWORK_STATE["original"] = (
        socket.socket,
        socket.create_connection,
        socket.getaddrinfo,
        socket.socketpair,
    )
    socket.socket = _BlockedSocket  # type: ignore[misc,assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.socketpair = _blocked  # type: ignore[assignment]
    _NETWORK_STATE["active"] = True


def disable_no_network() -> None:
    """Restore sockets (used only by the protocol tests)."""
    if not _NETWORK_STATE["active"]:
        return
    socket.socket, socket.create_connection, socket.getaddrinfo, socket.socketpair = _NETWORK_STATE["original"]  # type: ignore[misc]
    _NETWORK_STATE["active"] = False


def network_blocked() -> bool:
    return bool(_NETWORK_STATE["active"])


# --------------------------------------------------------------------------- imports
def _norm(p: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(str(p))).rstrip("\\/")


def _is_under(path: str, root: str) -> bool:
    path, root = _norm(path), _norm(root)
    return path == root or path.startswith(root + os.sep)


@dataclass
class ImportGuard(importlib.abc.MetaPathFinder):
    """Refuses imports whose source file resolves under a forbidden root.

    Allowed roots win over forbidden roots, so the sandbox itself (and its
    virtual environment) may live inside a forbidden drive.
    """

    forbidden_roots: tuple[str, ...]
    allowed_roots: tuple[str, ...]
    violations: list[str] = field(default_factory=list)

    def _permitted(self, origin: str | None) -> bool:
        if not origin or origin in {"built-in", "frozen"}:
            return True
        if any(_is_under(origin, a) for a in self.allowed_roots):
            return True
        return not any(_is_under(origin, f) for f in self.forbidden_roots)

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        origins = [spec.origin] if spec.origin else []
        if spec.submodule_search_locations:
            origins.extend(spec.submodule_search_locations)
        for origin in origins:
            if not self._permitted(origin):
                self.violations.append(f"{fullname} <- {origin}")
                raise SandboxViolation(
                    f"import of {fullname!r} from {origin!r} is forbidden: it lies inside an "
                    "original ISRM project tree. Re-implement it inside the sandbox instead."
                )
        return None  # let the normal finders import it


_GUARD: ImportGuard | None = None


def default_forbidden_roots() -> tuple[str, ...]:
    env = os.environ.get("ISRM_SANDBOX_FORBIDDEN")
    if env:
        return tuple(p for p in env.split(os.pathsep) if p)
    return (str(SANDBOX_ROOT.anchor),)  # the whole drive the sandbox lives on


def default_allowed_roots() -> tuple[str, ...]:
    roots = [str(SANDBOX_ROOT)]
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(venv)
    roots.append(sys.prefix)
    roots.append(sys.base_prefix)
    return tuple(dict.fromkeys(roots))


def install_import_guard(
    forbidden_roots: Iterable[str] | None = None,
    allowed_roots: Iterable[str] | None = None,
) -> ImportGuard:
    """Scrub sys.path and install the guard at the front of sys.meta_path. Idempotent."""
    global _GUARD
    forbidden = tuple(forbidden_roots) if forbidden_roots is not None else default_forbidden_roots()
    allowed = tuple(allowed_roots) if allowed_roots is not None else default_allowed_roots()
    if _GUARD is not None and _GUARD in sys.meta_path:
        _GUARD.forbidden_roots, _GUARD.allowed_roots = forbidden, allowed
        return _GUARD
    cwd = os.getcwd()
    cleaned: list[str] = []
    for entry in sys.path:
        resolved = cwd if entry in ("", ".") else entry
        if any(_is_under(resolved, a) for a in allowed):
            cleaned.append(entry)
        elif any(_is_under(resolved, f) for f in forbidden):
            continue
        else:
            cleaned.append(entry)
    sys.path[:] = cleaned
    _GUARD = ImportGuard(forbidden, allowed)
    sys.meta_path.insert(0, _GUARD)
    return _GUARD


def import_guard_active() -> bool:
    return _GUARD is not None and _GUARD in sys.meta_path


def uninstall_import_guard() -> None:
    global _GUARD
    if _GUARD is not None and _GUARD in sys.meta_path:
        sys.meta_path.remove(_GUARD)
    _GUARD = None


# --------------------------------------------------------------------------- inputs
def verify_inputs(inputs: Iterable[dict]) -> list[dict]:
    """Check that each referenced input exists and matches its recorded SHA-256."""
    report = []
    for ref in inputs:
        path = ref["path"]
        expected = ref["sha256"].lower()
        if not os.path.exists(path):
            raise SandboxViolation(f"input missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SandboxViolation(f"input hash mismatch for {path}: expected {expected}, got {actual}")
        report.append({"path": path, "sha256": actual, "bytes": os.path.getsize(path)})
    return report


# --------------------------------------------------------------------------- fingerprint
def env_fingerprint() -> dict:
    fp = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "network_blocked": network_blocked(),
        "import_guard": import_guard_active(),
    }
    for mod in ("numpy", "scipy", "pandas"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "?")
        except Exception:  # pragma: no cover - optional dependency
            fp[mod] = None
    return fp


def activate(strict: bool = True) -> dict:
    """Turn on every isolation guarantee and return the fingerprint."""
    install_import_guard()
    enable_no_network()
    fp = env_fingerprint()
    if strict and not (fp["network_blocked"] and fp["import_guard"]):
        raise SandboxViolation("isolation could not be activated")
    return fp
