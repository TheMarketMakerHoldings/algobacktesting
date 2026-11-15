"""Catch an overfit backtest: a researcher tries 200 strategies that have no edge.

Run it::

    uv run python examples/overfit_demo.py

Every strategy below is pure noise, so any "edge" is luck. The best backtest still
looks impressive. The Deflated Sharpe Ratio and the Probability of Backtest Overfitting
expose it. A second run plants a real edge in one strategy, which both tests recognise.
"""

from __future__ import annotations

import numpy as np

from algobacktesting import (
    annualize_sharpe,
    column_sharpe_ratios,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_stats,
)

DAYS = 1008  # 4 years of trading days; divisible by 16 for PBO's equal blocks
TRIALS = 200


def analyse(returns: np.ndarray) -> dict[str, float]:
    """Pick the best backtest the way a researcher would, then test it honestly."""
    sharpes = column_sharpe_ratios(returns)
    best = int(np.argmax(sharpes))
    stats = return_stats(returns[:, best])
    dsr = deflated_sharpe_ratio(
        stats.sharpe,
        stats.n_obs,
        n_trials=TRIALS,
        sharpe_variance=float(np.var(sharpes, ddof=1)),
        skew=stats.skew,
        kurtosis=stats.kurtosis,
    )
    pbo = probability_of_backtest_overfitting(returns, n_partitions=16)
    return {
        "best_annual_sharpe": annualize_sharpe(stats.sharpe, 252),
        "deflated_sharpe": dsr,
        "pbo": pbo.pbo,
        "prob_oos_loss": pbo.prob_oos_loss,
    }


def simulate(seed: int, *, daily_edge: float) -> np.ndarray:
    """200 strategies with 1% daily volatility; strategy 7 gets ``daily_edge`` of real drift."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(DAYS, TRIALS))
    returns[:, 7] += daily_edge
    return returns


def main() -> dict[str, dict[str, float]]:
    """Run the three scenarios, print a short report, and return the numbers."""
    results = {
        "200 strategies, no edge at all": analyse(simulate(42, daily_edge=0.0)),
        # True annual Sharpe of about 1.3: real, but the luckiest noise strategy still wins.
        "plus one modest real edge (~1.3)": analyse(simulate(42, daily_edge=0.0008)),
        # True annual Sharpe of about 2.4: strong enough to win and to survive both tests.
        "plus one strong real edge (~2.4)": analyse(simulate(42, daily_edge=0.0015)),
    }
    for label, r in results.items():
        print(f"\n{label}")
        print(f"  best backtest, annual Sharpe : {r['best_annual_sharpe']:.2f}")
        print(f"  Deflated Sharpe Ratio        : {r['deflated_sharpe']:.3f}   (want > 0.95)")
        print(f"  Prob. of Backtest Overfitting: {r['pbo']:.3f}   (want near 0)")
        print(f"  Prob. winner loses OOS       : {r['prob_oos_loss']:.3f}")
    return results


if __name__ == "__main__":
    main()
