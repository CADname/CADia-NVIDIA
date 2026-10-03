from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_metadata_is_consistent():
    init = (ROOT / "src" / "standalonecad" / "__init__.py").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
    project_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    assert version == project_version == "10.6.0"


def test_release_dependency_versions_match_verified_stack():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "cadquery==2.8.0" in req
    assert "vtk==9.6.2" in req
    assert "PySide6-Essentials==6.11.2" in req
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'cadquery==2.8.0' in pyproject
    assert 'vtk==9.6.2' in pyproject
    assert 'PySide6-Essentials==6.11.2' in pyproject
    assert 'requires-python = ">=3.11,<3.13"' in pyproject


def test_normal_entry_point_is_single_click_and_no_manual_codex_config():
    start = (ROOT / "START_STANDALONE_CAD.bat").read_text(encoding="utf-8")
    readme = (ROOT / "README_KO.md").read_text(encoding="utf-8")
    assert "FIRST_RUN.bat --no-pause" in start
    assert "pythonw.exe" in start
    assert "START_STANDALONE_CAD.bat" in readme
    assert "config.toml" in readme
