from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path, PureWindowsPath


def sanitize_filename_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(value or ""))
    return cleaned.strip(" ._") or "Overlay"


def build_export_filename(source_path: str = "", now: datetime | None = None, extension: str = ".xlsx") -> str:
    if source_path:
        source_text = str(source_path)
        source_stem = PureWindowsPath(source_text).stem if "\\" in source_text else Path(source_text).stem
    else:
        source_stem = "Overlay"
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{sanitize_filename_stem(source_stem)}_Misalignment_Result_{timestamp}{ext}"
