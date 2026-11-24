"""The website's strategy comparison must compute the difference test correctly.

``docs/index.html`` compares two groups of trades with Welch's two-sample test (normal
approximation). This test runs the page's code (between ``/* compare:start */`` and
``/* compare:end */``) in Node.js and checks it two ways:

1. Against the same formula computed independently in Python.
2. By calibration: when both groups come from one distribution, the page should call the
   difference "real" (95% or more) about 5% of the time.

Skipped without Node.js.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_compare_harness.js"
NODE = shutil.which("node")
NULL_PAIRS = 2000

RNG = np.random.default_rng(2460551)
EXACT = [
    [RNG.normal(10, 80, 40).round(2).tolist(), RNG.normal(-5, 60, 55).round(2).tolist()],
    [RNG.normal(0, 30, 25).round(2).tolist(), RNG.normal(0, 90, 200).round(2).tolist()],
]
# Same distribution for A and B: a skewed, trade-like mix of small losses and larger wins.
NULL = [
    [
        np.where(RNG.random(n) < 0.35, RNG.normal(250, 60, n), RNG.normal(-130, 20, n))
        .round(2)
        .tolist()
        for n in (int(RNG.integers(60, 150)), int(RNG.integers(60, 150)))
    ]
    for _ in range(NULL_PAIRS)
]


@pytest.fixture(scope="module")
def js() -> list[dict[str, Any]]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "compare")
    )
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "compare.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps({"pairs": EXACT + NULL}), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)], capture_output=True, text=True, check=True, timeout=120
        )
    out: list[dict[str, Any]] = json.loads(done.stdout)
    return out


def test_matches_welch_formula(js: list[dict[str, Any]]) -> None:
    for (a, b), got in zip(EXACT, js[: len(EXACT)], strict=True):
        x, y = np.array(a), np.array(b)
        se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        diff = float(x.mean() - y.mean())
        assert got["diff"] == pytest.approx(diff, rel=1e-12)
        assert got["se"] == pytest.approx(se, rel=1e-12)
        assert got["probABetter"] == pytest.approx(NormalDist().cdf(diff / se), abs=1e-12)


def test_no_difference_is_rarely_called_real(js: list[dict[str, Any]]) -> None:
    # Both groups share one distribution, so A is "really better" (95% or more) about 5% of
    # the time; allow four binomial standard errors (about 2 points) either way.
    probs = np.array([r["probABetter"] for r in js[len(EXACT) :]])
    called = float(np.mean(probs >= 0.95))
    tolerance = 4 * math.sqrt(0.05 * 0.95 / NULL_PAIRS)
    assert abs(called - 0.05) < tolerance, called
