"""The website's calculator must agree with the library.

``docs/index.html`` answers "is your edge real?" from a win rate and reward-to-risk ratio,
or from pasted trade results. Its JavaScript re-implements the library's statistics. This
test extracts that code, runs it with Node.js, and compares every output with the Python
library on random inputs across the calculator's whole range. Skipped without Node.js.

Model used by the page (documented on the page itself):

* Quick check: each trade wins ``reward_risk`` or loses 1 (units of risk), so trade results
  follow a two-point distribution. Its moments are checked below against ``return_stats``.
* Luck adjustment: the Deflated Sharpe Ratio with the trial variance set to its value when
  no strategy has an edge, ``1 / (n_trades - 1)``.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest

from algobacktesting import deflated_sharpe_ratio, expected_max_sharpe, return_stats

PAGE = Path(__file__).resolve().parents[1] / "docs" / "index.html"
NODE = shutil.which("node")

HARNESS = """
const cases = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
function quick(c) {
  const s = statsFromWinRate(c[0], c[1]);
  const need = tradesNeeded(s, c[3], 0.95);
  return [s.mean, s.sd, s.sharpe, s.skew, s.kurtosis, edgeProbability(s, c[2], c[3]),
          luckPerTrade(s, c[2], c[3]), isFinite(need) ? need : null];
}
function pasted(c) {
  const s = statsFromTrades(c[0]);
  return [s.sharpe, s.skew, s.kurtosis, s.n, s.winRate, edgeProbability(s, s.n, c[1])];
}
console.log(JSON.stringify({
  cdf: cases.x.map(normCdf),
  inv: cases.p.map(normInv),
  quick: cases.quick.map(quick),
  pasted: cases.pasted.map(pasted),
}));
"""


def js_fixed(value: float, digits: int) -> str:
    """Format like JavaScript's ``toFixed``: exact ties round up, not to even."""
    step = Decimal(1).scaleb(-digits)
    return f"{Decimal(value).quantize(step, rounding=ROUND_HALF_UP):+}"


def two_point(win_rate: float, reward_risk: float) -> tuple[float, float, float, float]:
    """Mean, s.d., skew and raw kurtosis of a trade that wins ``reward_risk`` or loses 1."""
    spread = math.sqrt(win_rate * (1 - win_rate))
    skew = (1 - 2 * win_rate) / spread
    return win_rate * reward_risk - (1 - win_rate), spread * (reward_risk + 1), skew, 1 + skew**2


def edge_probability(
    sharpe: float, skew: float, kurtosis: float, n_trades: float, setups: int
) -> float:
    """The page's headline number, computed with the library."""
    return deflated_sharpe_ratio(
        sharpe,
        n_trades,
        n_trials=setups,
        sharpe_variance=1 / (n_trades - 1),
        skew=skew,
        kurtosis=kurtosis,
    )


def _cases() -> dict[str, list[Any]]:
    rng = np.random.default_rng(1821643)
    quick: list[Any] = [[0.45, 1.5, 120, 5]]  # the quick check's default values
    for _ in range(400):
        quick.append(
            [
                int(rng.integers(5, 96)) / 100,
                int(rng.integers(2, 51)) / 10,
                int(rng.integers(20, 5001)),
                int(rng.integers(1, 201)),
            ]
        )
    pasted: list[Any] = []
    for i in range(120):
        n = int(rng.integers(10, 301))
        if i % 2:
            values = rng.normal(rng.uniform(-0.2, 0.4), 1.0, n)
        else:  # lopsided results: many small losses, a few large wins
            values = rng.exponential(1.0, n) - rng.uniform(0.6, 1.2)
        pasted.append([values.tolist(), int(rng.integers(1, 51))])
    return {
        "x": np.linspace(-8.0, 8.0, 321).tolist(),
        "p": [1e-9, 1e-6, 1e-4, 0.01, 0.02425, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97575, 0.999, 1 - 1e-9],
        "quick": quick,
        "pasted": pasted,
    }


@pytest.fixture(scope="module")
def js() -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    import tempfile

    text = PAGE.read_text(encoding="utf-8")
    block = text[text.index("/* stats:start */") : text.index("/* stats:end */")]
    cases = _cases()
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "calc.js", Path(tmp) / "cases.json"
        script.write_text(block + HARNESS, encoding="utf-8")
        data.write_text(json.dumps(cases), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)], capture_output=True, text=True, check=True, timeout=60
        )
    return cases, json.loads(done.stdout)


def test_normal_functions(js: tuple[dict[str, list[Any]], dict[str, list[Any]]]) -> None:
    cases, got = js
    normal = NormalDist()
    for x, value in zip(cases["x"], got["cdf"], strict=True):
        assert value == pytest.approx(normal.cdf(x), abs=1e-12)
    for p, value in zip(cases["p"], got["inv"], strict=True):
        assert value == pytest.approx(normal.inv_cdf(p), abs=1e-8)


def test_quick_check_matches_library(js: tuple[dict[str, list[Any]], dict[str, list[Any]]]) -> None:
    cases, got = js
    for (win_rate, reward_risk, n_trades, setups), out in zip(
        cases["quick"], got["quick"], strict=True
    ):
        mean, sd, skew, kurtosis = two_point(win_rate, reward_risk)
        sharpe = mean / sd
        assert out[:5] == pytest.approx([mean, sd, sharpe, skew, kurtosis], rel=1e-10, abs=1e-12)
        assert out[5] == pytest.approx(
            edge_probability(sharpe, skew, kurtosis, n_trades, setups), abs=1e-9
        )
        assert out[6] == pytest.approx(
            expected_max_sharpe(setups, 1 / (n_trades - 1)) * sd, rel=1e-9, abs=1e-12
        )
        needed = out[7]
        if sharpe <= 0:
            assert needed is None  # a losing record can never be proven profitable
        elif needed >= 2:
            # At exactly this many trades the chance is 95%: the definition of "needed".
            at_needed = edge_probability(sharpe, skew, kurtosis, needed, setups)
            assert at_needed == pytest.approx(0.95, abs=1e-9)


def test_pasted_trades_match_library(js: tuple[dict[str, list[Any]], dict[str, list[Any]]]) -> None:
    cases, got = js
    for (values, setups), out in zip(cases["pasted"], got["pasted"], strict=True):
        stats = return_stats(values)
        assert out[:3] == pytest.approx([stats.sharpe, stats.skew, stats.kurtosis], rel=1e-9)
        assert out[3] == stats.n_obs
        assert out[4] == pytest.approx(float(np.mean(np.array(values) > 0)))
        expected = edge_probability(stats.sharpe, stats.skew, stats.kurtosis, stats.n_obs, setups)
        assert out[5] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("wins", "losses", "reward_risk"), [(45, 55, 1.5), (30, 70, 3.0), (60, 40, 0.8)]
)
def test_two_point_model_matches_real_trades(wins: int, losses: int, reward_risk: float) -> None:
    # The quick check's formulas must describe an actual list of such trades.
    n = wins + losses
    mean, sd, skew, kurtosis = two_point(wins / n, reward_risk)
    stats = return_stats([reward_risk] * wins + [-1.0] * losses)
    assert stats.skew == pytest.approx(skew)
    assert stats.kurtosis == pytest.approx(kurtosis)
    # return_stats uses the sample standard deviation (n - 1); the model uses the population one.
    assert stats.sharpe == pytest.approx(mean / sd * math.sqrt((n - 1) / n))
