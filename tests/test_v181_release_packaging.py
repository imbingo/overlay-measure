from __future__ import annotations

import json
from pathlib import Path

from overlay_measure import __version__
from scripts.generate_release_metadata import version_tuple, write_manifest, write_version_info


ROOT = Path(__file__).resolve().parents[1]


def test_installer_uses_onedir_and_stable_upgrade_identity():
    spec = (ROOT / "installer" / "overlay_measure.spec").read_text(encoding="utf-8")
    iss = (ROOT / "installer" / "OverlayMeasure.iss").read_text(encoding="utf-8")
    assert "COLLECT(" in spec
    assert "exclude_binaries=True" in spec
    assert "console=False" in spec
    assert '"PyQt6"' in spec
    assert '"matplotlib"' in spec
    assert "AppIdValue" in iss
    assert "{c27d6ac0-0b46-4782-9a97-04f96bcdfcd8}" in iss
    assert "UsePreviousAppDir=yes" in iss
    assert "CloseApplications=force" in iss
    assert 'icon=str(application_icon)' in spec
    assert '(str(assets), "assets")' in spec
    assert "SetupIconFile=..\\assets\\overlay_measure.ico" in iss
    assert (ROOT / "assets" / "overlay_measure_icon.png").is_file()
    assert (ROOT / "assets" / "overlay_measure.ico").is_file()


def test_release_metadata_contains_hash_and_full_installer_policy(tmp_path):
    artifact = tmp_path / "OverlayMeasure_Setup.exe"
    artifact.write_bytes(b"setup payload")
    manifest = tmp_path / "update_manifest.json"
    version_info = tmp_path / "version_info.txt"

    assert version_tuple(__version__) == (2, 4, 4, 0)
    write_version_info(version_info, __version__)
    write_manifest(manifest, artifact, __version__)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["version"] == __version__
    assert payload["artifact_type"] == "full_installer"
    assert len(payload["sha256"]) == 64
    assert payload["size_bytes"] == len(b"setup payload")
    assert artifact.with_suffix(".exe.sha256").exists()
    assert "ProductVersion" in version_info.read_text(encoding="utf-8")
