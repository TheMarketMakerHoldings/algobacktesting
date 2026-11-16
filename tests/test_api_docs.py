"""The committed API reference must match the code."""

import importlib.util
from pathlib import Path

import algobacktesting

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _render() -> str:
    spec = importlib.util.spec_from_file_location("build_api", DOCS / "build_api.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.render()
    assert isinstance(text, str)
    return text


def test_api_reference_is_up_to_date() -> None:
    committed = (DOCS / "api.md").read_text(encoding="utf-8")
    assert committed == _render(), "run `uv run python docs/build_api.py` and commit docs/api.md"


def test_every_public_name_is_documented() -> None:
    text = (DOCS / "api.md").read_text(encoding="utf-8")
    for name in algobacktesting.__all__:
        if not name.startswith("__"):
            assert f"`{name}`" in text, name
        assert getattr(algobacktesting, name).__doc__ or name.isupper() or name.startswith("__"), (
            f"{name} has no docstring"
        )
