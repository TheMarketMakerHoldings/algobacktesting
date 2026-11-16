"""Packaging smoke tests: the public API is importable and the version is consistent."""

import tomllib
from pathlib import Path

import algobacktesting

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert algobacktesting.__version__ == pyproject["project"]["version"]


def test_public_api_is_exported() -> None:
    for name in algobacktesting.__all__:
        assert hasattr(algobacktesting, name), name


def test_typed_marker_ships() -> None:
    assert (ROOT / "src" / "algobacktesting" / "py.typed").exists()


def test_citation_file_has_the_same_version() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"\nversion: {algobacktesting.__version__}\n" in citation
