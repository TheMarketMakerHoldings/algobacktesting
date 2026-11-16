"""Generate ``docs/api.md`` from the package's docstrings.

Run from the repository root::

    uv run python docs/build_api.py

``tests/test_api_docs.py`` fails if the committed file no longer matches the code.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import algobacktesting

TARGET = Path(__file__).resolve().parent / "api.md"

GROUPS = {
    "algobacktesting.sharpe": "Is this Sharpe ratio real?",
    "algobacktesting.pbo": "Did the search just pick the luckiest strategy?",
    "algobacktesting.cv": "Cross-validation without leakage",
}


def _summary(obj: object) -> str:
    """Return the first paragraph of an object's docstring as one line."""
    doc = inspect.getdoc(obj) or ""
    return " ".join(doc.split("\n\n")[0].split())


def _signature(name: str, obj: object) -> str:
    assert callable(obj)
    text = str(inspect.signature(obj)).replace("'", "")
    # Python 3.14 prints unresolved annotations as ForwardRef(X); show plain X everywhere.
    marker = "ForwardRef("
    while marker in text:
        start = text.index(marker)
        end = text.index(")", start)
        inner = text[start + len(marker) : end].split(",")[0]
        text = text[:start] + inner + text[end + 1 :]
    return name + text


def render() -> str:
    """Return the full text of the API reference."""
    lines = [
        "# API reference",
        "",
        "Generated from the docstrings by `docs/build_api.py`. Do not edit by hand.",
        "Each function's full docstring cites the equation and page it implements.",
        "",
        f"Version {algobacktesting.__version__}. Everything below is importable from",
        "`algobacktesting` directly.",
    ]
    names = [n for n in algobacktesting.__all__ if not n.startswith("__")]
    by_module: dict[str, list[str]] = {}
    constants: list[str] = []
    for name in names:
        obj = getattr(algobacktesting, name)
        if callable(obj):
            by_module.setdefault(obj.__module__, []).append(name)
        else:
            constants.append(name)
    for module in sorted(by_module, key=lambda m: list(GROUPS).index(m) if m in GROUPS else 99):
        lines += ["", f"## {GROUPS.get(module, module)}", "", f"Module `{module}`."]
        for name in sorted(by_module[module]):
            obj = getattr(algobacktesting, name)
            lines += ["", f"### `{name}`", "", "```python", _signature(name, obj), "```", ""]
            lines.append(_summary(obj))
    if constants:
        lines += ["", "## Constants", ""]
        lines += [
            f"- `{name}` = `{getattr(algobacktesting, name)!r}`" for name in sorted(constants)
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    TARGET.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.name}")
