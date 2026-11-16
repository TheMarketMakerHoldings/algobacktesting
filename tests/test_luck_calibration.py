"""How often does the site's check call a trader with NO edge "proven"?

The site reports the chance an edge is real as a Deflated Sharpe Ratio whose trial variance
is ``1 / (n_trades - 1)`` (its value when no setup has an edge), and calls the edge proven
at 95%. This simulates traders whose setups all have zero expectancy, keeps the best setup,
and counts how often that verdict appears.

Measured with 20,000 simulated traders per cell (seed 7), across 40, 120 and 500 trades:

* one setup: 3.8% to 5.9%, i.e. close to the intended 5%;
* five setups: 0.2% to 1.2%; twenty: 0.1% to 0.4%; a hundred: 0.0% to 0.2%.

So the luck adjustment is conservative: beating the *expected* best of N lucky setups by a
95% margin is harder than beating 95% of lucky outcomes. The test uses fewer simulations
and loose bounds; each tolerance is at least 4 binomial standard errors wide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from algobacktesting import deflated_sharpe_ratio

EXPERIMENTS = 6000
N_TRADES = 120


def false_alarm_rate(setups: int, win_rate: float, reward_risk: float, seed: int) -> float:
    """Share of no-edge traders whose best of ``setups`` setups is called proven (>= 95%)."""
    rng = np.random.default_rng(seed)
    wins = rng.binomial(N_TRADES, win_rate, size=(EXPERIMENTS, setups))
    p = wins / N_TRADES
    usable = (p > 0) & (p < 1)
    spread = np.sqrt(np.where(usable, p * (1 - p), 1.0))
    mean = p * reward_risk - (1 - p)
    sd = spread * (reward_risk + 1) * math.sqrt(N_TRADES / (N_TRADES - 1))
    sharpe = np.where(usable, mean / sd, -np.inf)
    best = sharpe.argmax(axis=1)
    rows = np.arange(EXPERIMENTS)
    proven = 0
    for sr, p_best in zip(sharpe[rows, best], p[rows, best], strict=True):
        skew = (1 - 2 * p_best) / math.sqrt(p_best * (1 - p_best))
        chance = deflated_sharpe_ratio(
            float(sr),
            N_TRADES,
            n_trials=setups,
            sharpe_variance=1 / (N_TRADES - 1),
            skew=float(skew),
            kurtosis=float(1 + skew**2),
        )
        proven += chance >= 0.95
    return proven / EXPERIMENTS


# Zero-expectancy traders: win rate equals 1 / (1 + reward-to-risk).
NO_EDGE = [(0.40, 1.5), (0.50, 1.0), (0.25, 3.0)]


@pytest.mark.parametrize(("win_rate", "reward_risk"), NO_EDGE)
def test_one_setup_false_alarms_are_near_five_percent(win_rate: float, reward_risk: float) -> None:
    assert win_rate * reward_risk - (1 - win_rate) == pytest.approx(0.0)  # truly no edge
    rate = false_alarm_rate(1, win_rate, reward_risk, seed=7)
    # 4 binomial s.e. at 5% is 0.011; the lopsided 25% / 3R case runs slightly high (5.9%).
    assert 0.03 < rate < 0.075


@pytest.mark.parametrize(("win_rate", "reward_risk"), NO_EDGE)
@pytest.mark.parametrize("setups", [5, 20])
def test_luck_adjustment_is_conservative(win_rate: float, reward_risk: float, setups: int) -> None:
    # Observed at most 1.2%; 2% leaves more than 4 s.e. (0.0014 each) of room.
    assert false_alarm_rate(setups, win_rate, reward_risk, seed=7) < 0.02
