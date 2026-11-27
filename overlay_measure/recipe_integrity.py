from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sidecar_path(path: str | Path) -> Path:
    recipe = Path(path)
    return recipe.with_suffix(recipe.suffix + ".sha256")


def seal_recipe(path: str | Path) -> str:
    digest = file_sha256(path)
    seal = sidecar_path(path)
    tmp = seal.with_suffix(seal.suffix + ".tmp")
    tmp.write_text(digest + "\n", encoding="ascii")
    tmp.replace(seal)
    return digest


def _read_expected_digest(seal: Path) -> str | None:
    try:
        raw = seal.read_text(encoding="utf-8-sig").strip().upper()
    except (UnicodeDecodeError, OSError):
        return None
    if len(raw) != 64 or any(ch not in "0123456789ABCDEF" for ch in raw):
        return None
    return raw


def verify_and_read_recipe_bytes(path: str | Path) -> tuple[str, str, bytes]:
    """Read recipe bytes once and verify SHA-256 against sidecar without TOCTOU."""
    raw_bytes = Path(path).read_bytes()
    digest = bytes_sha256(raw_bytes)
    seal = sidecar_path(path)
    if not seal.exists():
        return "Unsealed", digest, raw_bytes
    expected = _read_expected_digest(seal)
    if expected is None or expected != digest:
        return "Mismatch", digest, raw_bytes
    return "Verified", digest, raw_bytes


def verify_recipe(path: str | Path) -> tuple[str, str]:
    """Return (status, digest): Verified, Unsealed, or Mismatch."""
    status, digest, _ = verify_and_read_recipe_bytes(path)
    return status, digest
