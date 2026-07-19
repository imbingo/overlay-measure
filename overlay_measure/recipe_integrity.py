from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sidecar_path(path: str | Path) -> Path:
    recipe = Path(path)
    return recipe.with_suffix(recipe.suffix + ".sha256")


def seal_recipe(path: str | Path) -> str:
    digest = file_sha256(path)
    sidecar_path(path).write_text(digest + "\n", encoding="ascii")
    return digest


def verify_recipe(path: str | Path) -> tuple[str, str]:
    """Return (status, digest): Verified, Unsealed, or Mismatch."""
    digest = file_sha256(path)
    seal = sidecar_path(path)
    if not seal.exists():
        return "Unsealed", digest
    expected = seal.read_text(encoding="ascii").strip().upper()
    return ("Verified" if expected == digest else "Mismatch"), digest
