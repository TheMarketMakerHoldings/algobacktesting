"""The documentation must stay true: its code runs, its numbers match, its links resolve."""

from __future__ import annotations

import importlib.util
import re
import struct
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPO_URL = "https://github.com/TheMarketMakerHoldings/algobacktesting"
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _pages() -> list[Path]:
    """Markdown files that belong to the repository: tracked, or new and not ignored."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(ROOT / line for line in listed.stdout.splitlines() if (ROOT / line).exists())


PAGES = _pages()  # empty outside a git checkout, which skips the link test


def _load_example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table_rows(heading_cell: str) -> list[list[str]]:
    """Return the body rows of the README table whose first header cell is ``heading_cell``."""
    lines = README.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {heading_cell} |"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_readme_quickstart_runs_and_matches_its_table() -> None:
    blocks = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)
    assert len(blocks) == 1
    namespace: dict[str, float] = {}
    exec(blocks[0], namespace)

    # The quickstart is the "no edge" world, so it must give that row's numbers.
    no_edge = _table_rows("World")[0]
    assert f"{namespace['dsr']:.3f}" == no_edge[2]
    assert f"{namespace['pbo']:.3f}" == no_edge[3]


def test_readme_overfit_table_matches_the_demo() -> None:
    results = _load_example("overfit_demo").main()
    rows = _table_rows("World")
    assert len(rows) == len(results) == 3
    for row, r in zip(rows, results.values(), strict=True):
        assert row[1:] == [
            f"{r['best_annual_sharpe']:.2f}",
            f"{r['deflated_sharpe']:.3f}",
            f"{r['pbo']:.3f}",
        ]


def test_readme_has_no_relative_links() -> None:
    # PyPI shows the README too, and relative links are dead there.
    for target in LINK.findall(README):
        assert target.startswith(("https://", "#")), target


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_links_into_the_repository_resolve(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        target = target.split("#")[0]
        if not target:
            continue  # a link to a heading on the same page
        if target.startswith(REPO_URL):
            match = re.match(r"/(?:blob|tree)/main/(.+)", target.removeprefix(REPO_URL))
            if match:
                assert (ROOT / match.group(1)).exists(), f"{page.name}: {target}"
        elif not target.startswith(("https://", "http://", "mailto:")):
            assert (page.parent / target).exists(), f"{page.name}: {target}"


def test_site_links_into_the_repository_resolve() -> None:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    targets = re.findall(rf'href="{re.escape(REPO_URL)}/(?:blob|tree)/main/([^"#]+)', html)
    assert targets, "the site should link to the source"
    for target in targets:
        assert (ROOT / target).exists(), target


def test_link_preview_image_matches_its_tags() -> None:
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'property="og:image" content="[^"]*/([^"/]+)"', html)
    assert match is not None
    data = (ROOT / "docs" / match.group(1)).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1200, 630)  # the size link previews expect
    assert f'property="og:image:width" content="{width}"' in html
    assert f'property="og:image:height" content="{height}"' in html


def test_secret_scanner_versions_agree() -> None:
    # The same scanner runs before each commit and again in CI; keep the two in step.
    hooks = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    hook = re.search(r"gitleaks/gitleaks\n\s+rev: v([\d.]+)", hooks)
    pinned = re.search(r"GITLEAKS_VERSION: ([\d.]+)", ci)
    assert hook is not None
    assert pinned is not None
    assert hook.group(1) == pinned.group(1)
