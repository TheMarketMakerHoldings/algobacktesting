"""Reproduce every number quoted in docs/paper/verification-report.md.

Run from the repository root::

    uv run python docs/paper/reproduce_numbers.py

All randomness is seeded. Expect a runtime of under a minute.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from algobacktesting import (
    annualize_sharpe,
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_stats,
)


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n== {title}")


def mixture_moments() -> tuple[float, float, float, float]:
    """Moments of the BLdP (2012) two-Normal mixture (p=.15: N(-.1,.12^2); N(.06,.03^2))."""
    p, m, s = 0.15, (-0.1, 0.06), (0.12, 0.03)

    def raw(k: int) -> float:
        out = 0.0
        for w, mu, sd in ((p, m[0], s[0]), (1 - p, m[1], s[1])):
            out += (
                w
                * [
                    1,
                    mu,
                    mu**2 + sd**2,
                    mu**3 + 3 * mu * sd**2,
                    mu**4 + 6 * mu**2 * sd**2 + 3 * sd**4,
                ][k]
            )
        return out

    r1, r2, r3, r4 = raw(1), raw(2), raw(3), raw(4)
    var = r2 - r1**2
    skew = (r3 - 3 * r1 * r2 + 2 * r1**3) / var**1.5
    kurt = (r4 - 4 * r1 * r3 + 6 * r1**2 * r2 - 3 * r1**4) / var**2
    return r1, math.sqrt(var), skew, kurt


def exact_expected_max(n: int) -> float:
    """E[max of n iid N(0,1)] by numerical integration."""
    x = np.linspace(-12.0, 12.0, 240_001)
    phi = np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    big_phi = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    f = x * n * phi * big_phi ** (n - 1)
    # Trapezoid rule, written out so it runs on every supported NumPy version.
    return float(np.sum((f[1:] + f[:-1]) * np.diff(x)) / 2)


def false_alarm_rate(n_trades: int, setups: int, win_rate: float, reward_risk: float) -> float:
    """Share of 20,000 no-edge traders whose best setup the site would call proven."""
    experiments = 20_000
    rng = np.random.default_rng(7)
    wins = rng.binomial(n_trades, win_rate, size=(experiments, setups))
    p = wins / n_trades
    usable = (p > 0) & (p < 1)
    spread = np.sqrt(np.where(usable, p * (1 - p), 1.0))
    sd = spread * (reward_risk + 1) * math.sqrt(n_trades / (n_trades - 1))
    sharpe = np.where(usable, (p * reward_risk - (1 - p)) / sd, -np.inf)
    best = sharpe.argmax(axis=1)
    rows = np.arange(experiments)
    proven = 0
    for sr, p_best in zip(sharpe[rows, best], p[rows, best], strict=True):
        skew = (1 - 2 * p_best) / math.sqrt(p_best * (1 - p_best))
        chance = deflated_sharpe_ratio(
            float(sr),
            n_trades,
            n_trials=setups,
            sharpe_variance=1 / (n_trades - 1),
            skew=float(skew),
            kurtosis=float(1 + skew**2),
        )
        proven += chance >= 0.95
    return proven / experiments


def main() -> None:
    """Print every reported number."""
    section("DSR worked example, BLdP (2014) p.10")
    sr, v = 2.5 / math.sqrt(250), 1 / (2 * 250)
    sr0 = expected_max_sharpe(100, v)
    print(f"SR0 = {sr0:.6f}")
    for n, sk, ku in ((100, -3, 10), (46, -3, 10), (88, 0, 3)):
        dsr = deflated_sharpe_ratio(sr, 1250, n_trials=n, sharpe_variance=v, skew=sk, kurtosis=ku)
        print(f"DSR N={n:<3} skew={sk:<2} kurt={ku:<2}: {dsr:.6f}")
    print(
        f"PSR vs rounded SR0 0.1132: "
        f"{probabilistic_sharpe_ratio(sr, 1250, benchmark=0.1132, skew=-3, kurtosis=10):.6f}"
    )

    section("PSR worked example, BLdP (2012) pp.9-10 and Figure 6")
    mean, sd, skew, kurt = mixture_moments()
    msr = mean / sd
    print(
        f"mixture: mean={mean:.6f} sd={sd:.6f} skew={skew:.6f} kurt={kurt:.6f} "
        f"SR={msr:.6f} annual={annualize_sharpe(msr, 12):.6f}"
    )
    for n, sk, ku in ((24, 0.0, 3.0), (24, skew, kurt), (36, skew, kurt)):
        print(
            f"PSR n={n} {'normal' if sk == 0 else 'mixture'}: "
            f"{probabilistic_sharpe_ratio(msr, n, skew=sk, kurtosis=ku):.6f}"
        )
    print(
        f"(with the printed SR 0.458, n=36: "
        f"{probabilistic_sharpe_ratio(0.458, 36, skew=-2.448, kurtosis=10.164):.6f})"
    )

    section("MinTRL, BLdP (2012) p.11 (annual SR 2 vs 1, 95%)")
    for ppy, sk, ku in ((252, 0, 3), (52, 0, 3), (12, 0, 3), (12, -0.72, 5.78)):
        r = math.sqrt(ppy)
        n = min_track_record_length(2 / r, benchmark=1 / r, skew=sk, kurtosis=ku)
        print(f"ppy={ppy:<3} skew={sk:<5} kurt={ku:<4}: {n:.3f} obs = {n / ppy:.4f} years")

    section("SR0 closed form vs exact expected maximum (V = 1)")
    for n in (2, 10, 46, 100, 500, 1000):
        exact, approx = exact_expected_max(n), expected_max_sharpe(n, 1.0)
        print(f"N={n:<5} exact={exact:.5f} closed={approx:.5f} gap={approx - exact:+.5f}")

    section("SR0 closed form near N = 1 (unguarded formula)")
    g = 0.5772156649015329
    nd = __import__("statistics").NormalDist()

    def raw_sr0(n: float) -> float:
        return (1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e))

    for n in (1.01, 1.1, 1.5, 2.0):
        print(f"N={n}: {raw_sr0(n):+.4f}")
    lo, hi = 1.01, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if raw_sr0(mid) < 0 else (lo, mid)
    print(f"root N0 = {hi:.4f}")

    section("PSR calibration under the null (4000 seeded runs each)")
    p_, m_, s_ = 0.15, (-0.1, 0.06), (0.12, 0.03)

    def draw_mix(rng: np.random.Generator, n: int) -> np.ndarray:
        pick = rng.random(n) < p_
        return np.where(pick, rng.normal(m_[0], s_[0], n), rng.normal(m_[1], s_[1], n))

    def draw_normal(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.1, 1.0, n)

    def rate(
        draw: Callable[[np.random.Generator, int], np.ndarray],
        true_sr: float,
        n: int,
        rng: np.random.Generator,
        use_moments: bool,
    ) -> float:
        hits = 0
        for _ in range(4000):
            st = return_stats(draw(rng, n))
            sk, ku = (st.skew, st.kurtosis) if use_moments else (0.0, 3.0)
            hits += (
                probabilistic_sharpe_ratio(st.sharpe, n, benchmark=true_sr, skew=sk, kurtosis=ku)
                > 0.95
            )
        return hits / 4000

    normal_rate = rate(draw_normal, 0.1, 250, np.random.default_rng(1821643), True)
    print(f"normal, n=250: {normal_rate:.4f}")
    rng = np.random.default_rng(1821644)
    print(f"mixture, n=1000, corrected: {rate(draw_mix, msr, 1000, rng, True):.4f}")
    print(f"mixture, n=1000, naive:     {rate(draw_mix, msr, 1000, rng, False):.4f}")

    section("PBO under pure noise")
    vals = [
        probability_of_backtest_overfitting(np.random.default_rng(s).normal(size=(800, 50)), 8).pbo
        for s in range(40)
    ]
    print(
        f"N=50: mean={np.mean(vals):.4f} sd={np.std(vals, ddof=1):.4f} "
        f"min={min(vals):.3f} max={max(vals):.3f}"
    )
    vals3 = [
        probability_of_backtest_overfitting(
            np.random.default_rng(1000 + s).normal(size=(400, 3)), 8
        ).pbo
        for s in range(60)
    ]
    print(f"N=3:  mean={np.mean(vals3):.4f} (theory 2/3) sd={np.std(vals3, ddof=1):.4f}")

    section("PBO, rebuilt seasonal experiment (BBLZ Section 6)")
    for effect in (False, True):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            ret = rng.normal(0.0, 1.0, 1008)
            day = np.arange(1008) % 21 + 1
            if effect:
                ret = ret + np.where(day <= 5, 0.25, 0.0)
            cols = []
            for entry in range(1, 22):
                for hold in range(1, 21):
                    held = (day >= entry) & (day < entry + hold)
                    cols.extend([held * ret, -(held * ret)])
            r = probability_of_backtest_overfitting(np.column_stack(cols), 16)
            print(
                f"effect={effect!s:<5} seed={seed}: PBO={r.pbo:.3f} "
                f"slope={r.degradation_slope:+.2f} combos={r.n_combinations}"
            )

    section("CPCV combinatorics")
    for n_groups, k in ((6, 2), (10, 4)):
        res = combinatorial_purged_cv(np.arange(60.0), np.arange(60.0), n_groups, k)
        print(f"N={n_groups} k={k}: splits={len(res.splits)} paths={res.n_paths}")
    section("Luck adjustment calibration: no-edge traders called proven (120 trades)")
    for win_rate, reward_risk in ((0.40, 1.5), (0.50, 1.0), (0.25, 3.0)):
        rates = [false_alarm_rate(120, n, win_rate, reward_risk) for n in (1, 5, 20, 100)]
        cells = "  ".join(f"N={n}: {r:.3f}" for n, r in zip((1, 5, 20, 100), rates, strict=True))
        print(f"win rate {win_rate:.0%}, reward-to-risk {reward_risk}: {cells}")


if __name__ == "__main__":
    main()
