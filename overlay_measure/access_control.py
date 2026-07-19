from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path


DEFAULT_ENGINEERING_PASSWORD = "admin123"


class AccessController:
    """Local production/engineering access gate.

    The password is stored as a salted PBKDF2 digest. The first run initializes
    the requested default password without writing the plain text value.
    """

    def __init__(self, settings_dir: Path | None = None):
        if settings_dir is None:
            local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
            settings_dir = base / "OverlayMeasure"
        self.settings_dir = Path(settings_dir)
        self.settings_path = self.settings_dir / "security.json"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings_path.exists():
            self.set_password(DEFAULT_ENGINEERING_PASSWORD)

    @staticmethod
    def _digest(password: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()

    def set_password(self, password: str) -> None:
        if len(password) < 6:
            raise ValueError("工程模式密码至少需要 6 个字符")
        salt = os.urandom(16)
        data = {"salt": salt.hex(), "digest": self._digest(password, salt), "iterations": 200_000}
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.settings_path)

    def verify(self, password: str) -> bool:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            salt = bytes.fromhex(data["salt"])
            expected = str(data["digest"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return hmac.compare_digest(self._digest(password, salt), expected)
