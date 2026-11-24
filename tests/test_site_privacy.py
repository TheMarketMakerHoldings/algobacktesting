"""The website must not be able to send a visitor's data anywhere, and must be findable.

``docs/index.html`` reads trade and bar files in the browser. These checks keep that promise
enforceable: the page carries a Content-Security-Policy that only allows network requests to
the site itself, and it loads no scripts, styles, fonts or images from other sites.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
SITE = "https://algobacktesting.com/"


def _csp() -> dict[str, str]:
    match = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', HTML)
    assert match, "the page must carry a Content-Security-Policy"
    rules = {}
    for part in match.group(1).split(";"):
        name, _, value = part.strip().partition(" ")
        rules[name] = value
    return rules


def test_page_can_only_talk_to_itself() -> None:
    csp = _csp()
    assert csp["default-src"] == "'none'"
    assert csp["connect-src"] == "'self'"  # fetch and XHR: this site only
    assert csp["img-src"] == "'self' data:"  # no tracking pixels or image beacons
    assert csp["form-action"] == "'none'"  # no form can post anywhere
    assert csp["base-uri"] == "'none'"
    assert "http" not in " ".join(csp.values())  # no outside host allowed anywhere


def test_nothing_is_loaded_from_other_sites() -> None:
    loaded = re.findall(r"<(?:script|link|img|iframe|source|video|audio)\b[^>]*>", HTML)
    for tag in loaded:
        if re.search(r'rel="canonical"', tag):
            continue  # names the page's own address; the browser fetches nothing
        assert not re.search(r'(?:src|href)="(?:https?:)?//', tag), tag
    assert "fonts.googleapis" not in HTML
    assert "<iframe" not in HTML


def test_findable_by_search_engines() -> None:
    robots = (ROOT / "docs" / "robots.txt").read_text(encoding="utf-8")
    assert "Disallow: /\n" not in robots
    assert f"Sitemap: {SITE}sitemap.xml" in robots
    sitemap = (ROOT / "docs" / "sitemap.xml").read_text(encoding="utf-8")
    assert f"<loc>{SITE}</loc>" in sitemap
    assert f'<link rel="canonical" href="{SITE}">' in HTML
    description = re.search(r'<meta name="description" content="([^"]+)"', HTML)
    assert description
    assert 70 <= len(description.group(1)) <= 300
    data = re.search(r'<script type="application/ld\+json">(.*?)</script>', HTML, re.DOTALL)
    assert data
    info = json.loads(data.group(1))
    assert info["url"] == SITE
    assert info["author"]["name"] == "Daniel Heidary"
    assert (ROOT / "docs" / "CNAME").read_text(encoding="utf-8").strip() == "algobacktesting.com"
