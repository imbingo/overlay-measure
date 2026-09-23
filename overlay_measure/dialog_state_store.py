from __future__ import annotations

import json
import os
from pathlib import Path


class DialogStateStore:
    """Persist only file-dialog locations in the per-user application data area."""

    def __init__(self, path: Path | None = None):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "OverlayMeasure"
        self.path = Path(path) if path is not None else base / "dialog_state.json"

    def _load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def entry(self, key: str) -> dict:
        value = self._load().get(key, {})
        return value if isinstance(value, dict) else {}

    def directory(self, key: str) -> str:
        directory = str(self.entry(key).get("directory", ""))
        return directory if Path(directory).is_dir() else ""

    def last_path(self, key: str) -> str:
        return str(self.entry(key).get("last_path", ""))

    def paths(self, key: str) -> list[str]:
        values = self.entry(key).get("paths", [])
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if str(value)]

    def remember_file(self, key: str, file_path: str) -> None:
        path = Path(file_path).expanduser().resolve(strict=False)
        payload = self._load()
        payload[key] = {"directory": str(path.parent), "last_path": str(path)}
        self._save(payload)

    def remember_directory(self, key: str, directory: str, paths: list[str] | None = None) -> None:
        root = Path(directory).expanduser().resolve(strict=False)
        payload = self._load()
        payload[key] = {"directory": str(root), "paths": [str(Path(item).expanduser().resolve(strict=False)) for item in (paths or [])]}
        self._save(payload)
