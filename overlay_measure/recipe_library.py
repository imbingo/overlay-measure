from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .recipe_manager import load_recipe


@dataclass(frozen=True)
class RecipeLibraryEntry:
    path: Path
    name: str
    material_code: str
    version: str
    status: str
    source: str
    favorite: bool = False
    last_used: str = ""
    modified_at: float = 0.0


class RecipeLibrary:
    """Managed local recipe storage with optional read-only shared discovery."""

    def __init__(self, root: Optional[Path] = None):
        configured_root = os.environ.get("OVERLAY_MEASURE_RECIPE_LIBRARY", "").strip()
        if root is not None:
            self.root = Path(root)
        elif configured_root:
            self.root = Path(configured_root)
        else:
            local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
            self.root = base / "OverlayMeasure" / "recipes"
        self.root = self.root.expanduser().resolve()
        self.state_path = self.root / ".library_state.json"
        self._ensure_directories()

    @property
    def validated_dir(self) -> Path:
        return self.root / "validated"

    @property
    def draft_dir(self) -> Path:
        return self.root / "draft"

    @property
    def archived_dir(self) -> Path:
        return self.root / "archived"

    def _ensure_directories(self) -> None:
        for directory in (self.root, self.validated_dir, self.draft_dir, self.archived_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> dict:
        default = {"favorites": [], "recent": {}, "shared_library": ""}
        if not self.state_path.exists():
            return default
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return default
        if not isinstance(data, dict):
            return default
        return {
            "favorites": list(data.get("favorites", [])),
            "recent": dict(data.get("recent", {})),
            "shared_library": str(data.get("shared_library", "")),
        }

    def _save_state(self, state: dict) -> None:
        self._ensure_directories()
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    @staticmethod
    def _key(path: Path | str) -> str:
        return os.path.normcase(str(Path(path).expanduser().resolve()))

    @property
    def shared_library(self) -> Optional[Path]:
        value = self._load_state().get("shared_library", "").strip()
        return Path(value) if value else None

    def set_shared_library(self, path: Optional[Path | str]) -> None:
        state = self._load_state()
        state["shared_library"] = "" if path is None else str(Path(path).expanduser().resolve())
        self._save_state(state)

    @staticmethod
    def _status_directory(status: str) -> str:
        normalized = status.strip().lower()
        if any(token in normalized for token in ("validated", "approved", "released", "已验证", "已批准", "已发布")):
            return "validated"
        if any(token in normalized for token in ("archived", "obsolete", "retired", "已归档", "已停用", "作废")):
            return "archived"
        return "draft"

    @staticmethod
    def _safe_stem(value: str) -> str:
        clean = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value.strip())
        clean = re.sub(r"\s+", "_", clean).strip("._")
        return clean[:100] or "overlay_recipe"

    @staticmethod
    def _same_file_content(left: Path, right: Path) -> bool:
        if not left.exists() or not right.exists() or left.stat().st_size != right.stat().st_size:
            return False
        digest = lambda path: hashlib.sha256(path.read_bytes()).digest()
        return digest(left) == digest(right)

    def import_recipe(self, source_path: Path | str) -> Path:
        source = Path(source_path).expanduser().resolve()
        config, _, _ = load_recipe(str(source))
        status = str(getattr(config, "recipe_validation_status", ""))
        destination_dir = self.root / self._status_directory(status)
        name = str(getattr(config, "recipe_name", "")).strip() or source.stem
        version = str(getattr(config, "recipe_version", "")).strip()
        stem = self._safe_stem(f"{name}_{version}" if version else name)
        destination = destination_dir / f"{stem}.json"
        if destination == source:
            return destination
        if destination.exists():
            if self._same_file_content(source, destination):
                return destination
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = destination_dir / f"{stem}_{timestamp}.json"
        shutil.copy2(source, destination)
        return destination

    def mark_used(self, path: Path | str) -> None:
        state = self._load_state()
        state["recent"][self._key(path)] = datetime.now(timezone.utc).isoformat()
        if len(state["recent"]) > 100:
            keep = sorted(state["recent"].items(), key=lambda item: item[1], reverse=True)[:100]
            state["recent"] = dict(keep)
        self._save_state(state)

    def toggle_favorite(self, path: Path | str) -> bool:
        state = self._load_state()
        key = self._key(path)
        favorites = set(state["favorites"])
        if key in favorites:
            favorites.remove(key)
            enabled = False
        else:
            favorites.add(key)
            enabled = True
        state["favorites"] = sorted(favorites)
        self._save_state(state)
        return enabled

    def _iter_recipe_files(self) -> Iterable[tuple[Path, str]]:
        seen: set[str] = set()
        for directory in (self.validated_dir, self.draft_dir, self.archived_dir):
            for path in directory.glob("*.json"):
                key = self._key(path)
                if key not in seen:
                    seen.add(key)
                    yield path, "本机"
        shared = self.shared_library
        if shared and shared.exists():
            for path in shared.rglob("*.json"):
                key = self._key(path)
                if key not in seen:
                    seen.add(key)
                    yield path, "共享"

    def scan(self) -> list[RecipeLibraryEntry]:
        state = self._load_state()
        favorites = set(state["favorites"])
        recent = state["recent"]
        entries: list[RecipeLibraryEntry] = []
        for path, source in self._iter_recipe_files():
            try:
                config, _, _ = load_recipe(str(path))
                stat = path.stat()
            except (OSError, ValueError, TypeError, KeyError):
                continue
            key = self._key(path)
            entries.append(
                RecipeLibraryEntry(
                    path=path,
                    name=str(getattr(config, "recipe_name", "")).strip() or path.stem,
                    material_code=str(getattr(config, "material_code", "")).strip() or "-",
                    version=str(getattr(config, "recipe_version", "")).strip() or "-",
                    status=str(getattr(config, "recipe_validation_status", "")).strip() or "未验证",
                    source=source,
                    favorite=key in favorites,
                    last_used=str(recent.get(key, "")),
                    modified_at=stat.st_mtime,
                )
            )
        entries.sort(key=lambda entry: entry.name.lower())
        entries.sort(key=lambda entry: entry.modified_at, reverse=True)
        entries.sort(key=lambda entry: entry.last_used, reverse=True)
        entries.sort(key=lambda entry: entry.favorite, reverse=True)
        return entries
