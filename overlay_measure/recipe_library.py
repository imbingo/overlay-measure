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
from .recipe_integrity import seal_recipe


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


@dataclass(frozen=True)
class RecipeLibraryMigration:
    old_root: Path
    new_root: Path
    migrated: bool
    copied: int = 0
    reused: int = 0
    renamed: int = 0


class RecipeLibrary:
    """Managed local recipe storage with optional read-only shared discovery."""

    def __init__(self, root: Optional[Path] = None, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path is not None else self.default_config_path()
        configured_root = os.environ.get("OVERLAY_MEASURE_RECIPE_LIBRARY", "").strip()
        if root is not None:
            self.root = Path(root)
        elif configured_root:
            self.root = Path(configured_root)
        else:
            configured = self._load_library_config().get("local_library", "").strip()
            self.root = Path(configured) if configured else self.default_root()
        self.root = self.root.expanduser().resolve()
        self.state_path = self.root / ".library_state.json"
        self._ensure_directories()

    @staticmethod
    def application_data_root() -> Path:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
        return base / "OverlayMeasure"

    @classmethod
    def default_root(cls) -> Path:
        return (cls.application_data_root() / "recipes").expanduser().resolve()

    @classmethod
    def default_config_path(cls) -> Path:
        return cls.application_data_root() / "recipe_library_config.json"

    @property
    def environment_override(self) -> str:
        return os.environ.get("OVERLAY_MEASURE_RECIPE_LIBRARY", "").strip()

    def _load_library_config(self) -> dict:
        if not self.config_path.exists():
            return {"local_library": ""}
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"local_library": ""}
        return {"local_library": str(data.get("local_library", ""))} if isinstance(data, dict) else {"local_library": ""}

    def _save_library_config(self, root: Path) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"local_library": str(root)}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.config_path)

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
    def _load_state_at(path: Path) -> dict:
        default = {"favorites": [], "recent": {}, "shared_library": ""}
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return default
        if not isinstance(data, dict):
            return default
        return {
            "favorites": list(data.get("favorites", [])),
            "recent": dict(data.get("recent", {})),
            "shared_library": str(data.get("shared_library", "")),
        }

    @staticmethod
    def _write_state_at(path: Path, state: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _unique_migration_destination(destination: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = destination.with_name(f"{destination.stem}_migrated_{timestamp}{destination.suffix}")
        index = 2
        while candidate.exists():
            candidate = destination.with_name(
                f"{destination.stem}_migrated_{timestamp}_{index}{destination.suffix}"
            )
            index += 1
        return candidate

    @classmethod
    def _remap_state_key(
        cls,
        key: str,
        old_root: Path,
        new_root: Path,
        mapping: dict[str, Path],
    ) -> str:
        normalized = cls._key(key)
        if normalized in mapping:
            return cls._key(mapping[normalized])
        try:
            relative = Path(key).expanduser().resolve().relative_to(old_root)
        except (OSError, ValueError):
            return normalized
        return cls._key(mapping.get(normalized, new_root / relative))

    def change_local_library(self, new_root: Path | str, migrate: bool = True) -> RecipeLibraryMigration:
        target = Path(new_root).expanduser().resolve()
        old_root = self.root
        if target == old_root:
            return RecipeLibraryMigration(old_root, target, migrate)
        if target.exists() and not target.is_dir():
            raise ValueError("所选本机配方库路径不是文件夹")
        if target.is_relative_to(old_root):
            raise ValueError("新配方库不能位于当前配方库内部")
        if old_root.is_relative_to(target):
            raise ValueError("新配方库不能包含当前配方库目录")

        for directory in (target, target / "validated", target / "draft", target / "archived"):
            directory.mkdir(parents=True, exist_ok=True)

        copied = reused = renamed = 0
        mapping: dict[str, Path] = {}
        created_files: list[Path] = []
        target_state_path = target / ".library_state.json"
        previous_state = target_state_path.read_bytes() if target_state_path.exists() else None
        try:
            if migrate:
                for category in ("validated", "draft", "archived"):
                    source_directory = old_root / category
                    if not source_directory.exists():
                        continue
                    for source in source_directory.rglob("*.json"):
                        relative = source.relative_to(source_directory)
                        destination = target / category / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            if self._same_file_content(source, destination):
                                reused += 1
                            else:
                                destination = self._unique_migration_destination(destination)
                                renamed += 1
                        if not destination.exists():
                            shutil.copy2(source, destination)
                            created_files.append(destination)
                            copied += 1
                        mapping[self._key(source)] = destination

                        source_sidecar = source.with_suffix(source.suffix + ".sha256")
                        destination_sidecar = destination.with_suffix(destination.suffix + ".sha256")
                        if source_sidecar.exists() and not destination_sidecar.exists():
                            shutil.copy2(source_sidecar, destination_sidecar)
                            created_files.append(destination_sidecar)

                source_state = self._load_state()
                target_state = self._load_state_at(target_state_path)
                favorites = set(target_state["favorites"])
                favorites.update(
                    self._remap_state_key(key, old_root, target, mapping) for key in source_state["favorites"]
                )
                recent = dict(target_state["recent"])
                for key, used_at in source_state["recent"].items():
                    remapped = self._remap_state_key(key, old_root, target, mapping)
                    if str(used_at) > str(recent.get(remapped, "")):
                        recent[remapped] = used_at
                target_state = {
                    "favorites": sorted(favorites),
                    "recent": recent,
                    "shared_library": source_state["shared_library"] or target_state["shared_library"],
                }
                self._write_state_at(target_state_path, target_state)
            else:
                source_state = self._load_state()
                target_state = self._load_state_at(target_state_path)
                if source_state["shared_library"] and not target_state["shared_library"]:
                    target_state["shared_library"] = source_state["shared_library"]
                    self._write_state_at(target_state_path, target_state)

            self._save_library_config(target)
        except Exception:
            for path in reversed(created_files):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                if previous_state is None:
                    target_state_path.unlink(missing_ok=True)
                else:
                    target_state_path.write_bytes(previous_state)
            except OSError:
                pass
            raise

        self.root = target
        self.state_path = target_state_path
        self._ensure_directories()
        return RecipeLibraryMigration(old_root, target, migrate, copied, reused, renamed)

    def restore_default_library(self, migrate: bool = True) -> RecipeLibraryMigration:
        return self.change_local_library(self.default_root(), migrate=migrate)

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
            seal_recipe(destination)
            return destination
        if destination.exists():
            if self._same_file_content(source, destination):
                seal_recipe(destination)
                return destination
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = destination_dir / f"{stem}_{timestamp}.json"
        shutil.copy2(source, destination)
        seal_recipe(destination)
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
