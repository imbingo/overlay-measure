from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class RecentImageStore:
    """Small user-local history for quick image reopening, outside recipes."""

    def __init__(self, path: Path | None = None, max_entries: int = 20):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "OverlayMeasure"
        self.path = Path(path) if path is not None else base / "recent_images.json"
        self.max_entries = max(1, int(max_entries))

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError, TypeError):
            return []

    def _save(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records[: self.max_entries], ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, image_path: str, layer: str) -> None:
        path = str(Path(image_path).expanduser().resolve(strict=False))
        record = {"path": path, "layer": layer, "used_at": datetime.now(timezone.utc).isoformat()}
        records = [
            item for item in self._load()
            if not (str(item.get("path", "")) == path and str(item.get("layer", "upper")) == layer)
        ]
        self._save([record, *records])

    def entries(self, existing_only: bool = True) -> list[dict]:
        records = self._load()
        if existing_only:
            records = [item for item in records if Path(str(item.get("path", ""))).is_file()]
        return records[: self.max_entries]

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
