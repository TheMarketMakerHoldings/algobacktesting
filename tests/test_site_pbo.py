"""The website's overfitting check must give the library's answer.

``docs/index.html`` carries a port of ``algobacktesting.pbo`` (between ``/* pbo:start */``
and ``/* pbo:end */``) for the strategy builder: the probability that the best of the
versions a trader tested is overfit (Bailey, Borwein, Lopez de Prado and Zhu, 2017,
Algorithm 2.3). This test runs the page's code in Node.js and checks it against
``probability_of_backtest_overfitting`` on random matrices of several shapes, plus the
helpers that turn the builder's trades into daily profit and loss.

Skipped without Node.js.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from algobacktesting import probability_of_backtest_overfitting

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_pbo_harness.js"
NODE = shutil.which("node")

RNG = np.random.default_rng(2460556)


def _matrix(rows: int, cols: int, skill: float) -> list[list[float]]:
    """Daily P&L of `cols` versions; the last version has a small real edge."""
    data = RNG.normal(0, 100, (rows, cols))
    data[:, -1] += skill
    return [[float(v) for v in row] for row in data.round(2)]


# (rows, versions, periods, edge of the last version): even and odd numbers of versions, the
# paper's 16 periods, and one case where one version is clearly better.
SHAPES = [(40, 2, 4, 0.0), (48, 3, 8, 0.0), (96, 5, 8, 0.0), (160, 8, 16, 0.0), (144, 6, 16, 60.0)]
CASES: list[dict[str, Any]] = [{"matrix": _matrix(t, n, edge), "s": s} for t, n, s, edge in SHAPES]
BAD = [
    {"matrix": _matrix(40, 1, 0.0), "s": 4},  # one version
    {"matrix": _matrix(40, 3, 0.0), "s": 3},  # odd number of periods
    {"matrix": _matrix(42, 3, 0.0), "s": 4},  # rows do not split evenly
    {"matrix": [[0.0, float(i)] for i in range(40)], "s": 4},  # a version that never changes
]
DAILY = [
    {
        # Two versions on five days, two periods: the oldest day is left out (5 is odd).
        "versions": [
            [{"day": 1, "pnl": 10.0}, {"day": 3, "pnl": -5.0}, {"day": 3, "pnl": 2.0}],
            [{"day": 2, "pnl": 7.0}, {"day": 9, "pnl": 1.0}],  # day 9 is not a bar day
        ],
        "days": [1, 2, 3, 4, 5],
        "s": 2,
    }
]


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = text[text.index("/* pbo:start */") : text.index("/* pbo:end */")]
    payload = {
        "cases": CASES,
        "bad": BAD,
        "ranks": [[0.5, 1.1, 0.7], [2.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0]],
        "splits": [2, 4, 8, 16],
        "periods": [10, 19, 20, 39, 40, 153, 400],
        "daily": DAILY,
    }
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "pbo.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)],
            capture_output=True,
            encoding="utf-8",
            check=True,
            timeout=120,
        )
    out: dict[str, Any] = json.loads(done.stdout)
    return out


def test_matches_the_library(js: dict[str, Any]) -> None:
    for case, got in zip(CASES, js["cases"], strict=True):
        want = probability_of_backtest_overfitting(np.array(case["matrix"]), case["s"])
        assert got["pbo"] == want.pbo  # a share of splits: exact
        assert got["probOosLoss"] == want.prob_oos_loss
        assert got["nCombinations"] == want.n_combinations
        assert got["nStrategies"] == want.n_strategies
        # Sharpe ratios summed in a different order agree to about 1e-15; 1e-9 is ample.
        assert got["logits"] == pytest.approx(want.logits.tolist(), abs=1e-9)
        assert got["relativeRanks"] == pytest.approx(want.relative_ranks.tolist(), abs=1e-12)
        assert got["isPerformance"] == pytest.approx(want.is_performance.tolist(), rel=1e-9)
        assert got["oosPerformance"] == pytest.approx(want.oos_performance.tolist(), rel=1e-9)
        assert got["degradationSlope"] == pytest.approx(want.degradation_slope, rel=1e-6)
        assert got["degradationIntercept"] == pytest.approx(want.degradation_intercept, rel=1e-6)


def test_a_real_edge_is_rarely_overfit(js: dict[str, Any]) -> None:
    # The last case gives one version a real edge (0.6 of a standard deviation a day): it wins
    # in-sample and stays near the top out of sample, so PBO is low.
    edge = js["cases"][-1]
    assert edge["pbo"] < 0.1
    assert math.isclose(sum(edge["relativeRanks"]) / edge["nCombinations"], 1, abs_tol=0.2)


def test_bad_input_is_refused(js: dict[str, Any]) -> None:
    errors = [r.get("error", "") for r in js["bad"]]
    assert errors == [
        "Need at least 2 versions to compare.",
        "The number of periods must be even.",
        "The rows must split into equal periods.",
        "A version never changes within some periods, so it cannot be ranked there.",
    ]


def test_helpers(js: dict[str, Any]) -> None:
    # [BBLZ] p. 9 ranking example, then ties sharing the average rank.
    assert js["ranks"] == [[1, 3, 2], [2.5, 1, 2.5, 4], [2, 2, 2]]
    assert js["splits"] == [math.comb(s, s // 2) for s in (2, 4, 8, 16)]
    # Up to 16 periods of at least 5 days each, even, and none below 20 days.
    assert js["periods"] == [0, 0, 4, 6, 8, 16, 16]
    daily = js["daily"][0]
    assert daily["dropped"] == 1  # 5 days into 2 periods: the oldest day is left out
    assert daily["matrix"] == [[0, 7], [-3, 0], [0, 0], [0, 0]]
