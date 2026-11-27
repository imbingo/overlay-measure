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
    def _digest(password: str, salt: bytes, iterations: int = 200_000) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations)).hex()

    def set_password(self, password: str) -> None:
        if len(password) < 6:
            raise ValueError("工程模式密码至少需要 6 个字符")
        import tempfile
        salt = os.urandom(16)
        iterations = 200_000
        data = {
            "salt": salt.hex(),
            "digest": self._digest(password, salt, iterations),
            "iterations": iterations,
            "default_password": password == DEFAULT_ENGINEERING_PASSWORD,
            "warning_acknowledged": False,
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".security_", suffix=".tmp", dir=str(self.settings_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(data, indent=2))
            os.replace(tmp_name, self.settings_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def verify(self, password: str) -> bool:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            salt = bytes.fromhex(data["salt"])
            expected = str(data["digest"])
            iterations = int(data.get("iterations", 200_000))
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return hmac.compare_digest(self._digest(password, salt, iterations), expected)

    def should_warn_default_password(self) -> bool:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return bool(data.get("default_password", False) and not data.get("warning_acknowledged", False))

    def acknowledge_default_password_warning(self) -> None:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        data["warning_acknowledged"] = True
        temporary = self.settings_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.settings_path)
