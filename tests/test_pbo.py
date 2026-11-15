"""Tests for algobacktesting.pbo (CSCV, [BBLZ] = SSRN 2326253, revised Feb 2015).

The paper's CSCV results come from its own random draws, so they cannot be reproduced
digit for digit. What we test instead:

1. Exact items from the paper: the ranking example (p. 9) and the number of
   combinations (Eq. 2.3, Figure 1).
2. The algorithm's mechanics: train/test blocks are equal, complementary, in time order,
   and every split is evaluated.
3. Statistical behaviour: pure noise gives PBO = 0.5 on average (theory, not a tuned
   number), a genuinely skilled strategy gives PBO near 0, and a rebuilt version of the
   paper's seasonal experiment (Section 6) shows the paper's qualitative result.
4. Fast path vs generic path agreement, regression and loss statistics, validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from algobacktesting.pbo import (
    average_ranks,
    column_sharpe_ratios,
    probability_of_backtest_overfitting,
)

# ---------------------------------------------------------------------------------------
# 1. Exact items from the paper
# ---------------------------------------------------------------------------------------


def test_ranking_example_from_paper() -> None:
    # p. 9: R = (0.5, 1.1, 0.7) -> r = (1, 3, 2); R_bar = (0.6, 0.7, 1.3) -> r_bar = (1, 2, 3)
    assert average_ranks(np.array([0.5, 1.1, 0.7])).tolist() == [1.0, 3.0, 2.0]
    assert average_ranks(np.array([0.6, 0.7, 1.3])).tolist() == [1.0, 2.0, 3.0]


def test_ties_share_the_average_rank() -> None:
    assert average_ranks(np.array([2.0, 1.0, 2.0, 3.0])).tolist() == [2.5, 1.0, 2.5, 4.0]


@pytest.mark.parametrize(("s", "expected"), [(2, 2), (4, 6), (8, 70), (16, 12_870)])
def test_number_of_combinations(s: int, expected: int) -> None:
    # Eq. (2.3): C(S, S/2) = prod_{i=0}^{S/2-1} (S - i) / (S/2 - i). Figure 1: S = 4 gives 6.
    # The paper's text says S = 16 gives "12,780"; C(16, 8) is 12,870 (a typo in the paper).
    product = math.prod((s - i) / (s // 2 - i) for i in range(s // 2))
    assert round(product) == math.comb(s, s // 2) == expected
    x = np.random.default_rng(s).normal(size=(s * 4, 3))
    assert probability_of_backtest_overfitting(x, s).n_combinations == expected


# ---------------------------------------------------------------------------------------
# 2. Mechanics: record exactly which rows each evaluation sees
# ---------------------------------------------------------------------------------------


def test_splits_are_equal_complementary_and_time_ordered() -> None:
    t, s = 48, 8
    x = np.random.default_rng(0).normal(size=(t, 3))
    x[:, 0] = np.arange(t)  # column 0 = time index, so we can see which rows were used
    seen: list[tuple[int, ...]] = []

    def recording_metric(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        idx = rows[:, 0].astype(int)
        assert np.all(np.diff(idx) > 0), "rows must arrive in time order"
        seen.append(tuple(idx))
        return np.asarray([*rows[:, 1:].mean(axis=0).tolist(), 0.0], dtype=np.float64)

    result = probability_of_backtest_overfitting(x, s, metric=recording_metric)
    assert len(seen) == 2 * result.n_combinations  # one IS and one OOS evaluation each
    every_row = set(range(t))
    for i in range(0, len(seen), 2):
        train, test = set(seen[i]), set(seen[i + 1])
        assert len(train) == len(test) == t // 2
        assert train | test == every_row
        assert not train & test
    train_sets = {frozenset(seen[i]) for i in range(0, len(seen), 2)}
    assert len(train_sets) == math.comb(s, s // 2)
    assert all(frozenset(every_row - ts) in train_sets for ts in train_sets)  # symmetric


def test_logits_are_finite_even_at_extreme_ranks() -> None:
    # Winner OOS-best gives rank N, OOS-worst gives rank 1. Dividing by N + 1 keeps logits finite.
    x = np.random.default_rng(1).normal(size=(64, 4))
    x[:, 0] += 5.0  # always best in and out of sample
    best = probability_of_backtest_overfitting(x, 8)
    assert np.all(np.isfinite(best.logits))
    assert np.allclose(best.relative_ranks, 4 / 5)
    assert best.pbo == 0.0


def test_in_sample_tie_selects_first_column() -> None:
    # Metric: if the rows include time 0, columns 1 and 2 tie for best (1.0 each);
    # otherwise column 1 scores 0.5 and column 2 scores 0.9. So whenever block 0 is in the
    # training set there is an in-sample tie, and the OOS value reveals who was picked:
    # 0.5 means column 1 (first), 0.9 would mean column 2.
    t, s = 32, 4
    x = np.zeros((t, 3))
    x[:, 0] = np.arange(t)

    def tie_metric(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        if 0 in rows[:, 0]:
            return np.array([0.0, 1.0, 1.0])
        return np.array([0.0, 0.5, 0.9])

    result = probability_of_backtest_overfitting(x, s, metric=tie_metric)
    tied_combos = math.comb(s - 1, s // 2 - 1)  # training sets that contain block 0
    assert int(np.sum(result.oos_performance == 0.5)) == tied_combos
    assert 0.9 not in result.oos_performance


def test_deterministic() -> None:
    x = np.random.default_rng(3).normal(size=(80, 6))
    a, b = probability_of_backtest_overfitting(x, 10), probability_of_backtest_overfitting(x, 10)
    assert a.pbo == b.pbo
    assert np.array_equal(a.logits, b.logits)


# ---------------------------------------------------------------------------------------
# 3. Statistical behaviour
# ---------------------------------------------------------------------------------------


def test_pure_noise_averages_to_one_half() -> None:
    # With N even and no skill, the IS winner's OOS rank is uniform on 1..N, so
    # P(rank <= N/2) = 0.5 exactly. One dataset's PBO is noisy (combinations are
    # correlated), so average 40 seeded datasets. Tolerance: 4 standard errors of that mean,
    # estimated from the 40 values themselves.
    values = np.array(
        [
            probability_of_backtest_overfitting(
                np.random.default_rng(seed).normal(size=(800, 50)), 8
            ).pbo
            for seed in range(40)
        ]
    )
    standard_error = values.std(ddof=1) / math.sqrt(values.size)
    assert abs(values.mean() - 0.5) < 4 * standard_error


def test_odd_strategy_count_bias_matches_theory() -> None:
    # Documented consequence of the paper's "logit <= 0" estimator: with N = 3, ranks 1 and 2
    # (of 3) both count as overfit, so pure noise averages 2/3, not 1/2. Tolerance: 4 SE of
    # the mean over 60 seeded datasets.
    values = np.array(
        [
            probability_of_backtest_overfitting(
                np.random.default_rng(1000 + seed).normal(size=(400, 3)), 8
            ).pbo
            for seed in range(60)
        ]
    )
    standard_error = values.std(ddof=1) / math.sqrt(values.size)
    assert abs(values.mean() - 2 / 3) < 4 * standard_error


def test_identical_strategies_give_pbo_of_one() -> None:
    # Every OOS rank ties at the median (logit 0), which the <= 0 estimator counts as overfit.
    x = np.tile(np.random.default_rng(1).normal(size=(64, 1)), (1, 5))
    result = probability_of_backtest_overfitting(x, 8)
    assert result.pbo == 1.0
    assert np.all(result.logits == 0.0)


def test_rejects_too_many_combinations() -> None:
    # S = 24 would mean 2.7 million splits; S = 40 would be 137 billion (previously: a hang).
    with pytest.raises(ValueError, match="use S <= 22"):
        probability_of_backtest_overfitting(np.ones((240, 3)), 24)


def test_genuine_skill_gives_low_pbo() -> None:
    x = np.random.default_rng(7).normal(size=(800, 50))
    x[:, 3] += 0.3  # one strategy with a real per-period Sharpe of about 0.3
    result = probability_of_backtest_overfitting(x, 8)
    assert result.pbo == 0.0
    assert result.prob_oos_loss == 0.0


def _seasonal_strategies(seed: int, *, effect: bool) -> NDArray[np.float64]:
    """Rebuild [BBLZ] Section 6 (pp. 25-27) on a smaller grid.

    The paper uses a 1,000-day random walk and searches entry day 1-22, holding period
    1-20, stop loss 0-10 and side +/-1 (8,800 rules). Here: 48 months of 21 days (1,008
    days, so S = 16 blocks are equal), entry 1-21, holding 1-20, side +/-1, no stop loss
    (840 rules). With ``effect`` the first 5 days of each month are shifted up by a
    quarter standard deviation, as in the paper's second case.
    """
    days, month = 1008, 21
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 1.0, days)
    day_of_month = np.arange(days) % month + 1
    if effect:
        ret = ret + np.where(day_of_month <= 5, 0.25, 0.0)
    columns = []
    for entry in range(1, 22):
        for hold in range(1, 21):
            held = (day_of_month >= entry) & (day_of_month < entry + hold)
            columns.extend([held * ret, -(held * ret)])
    return np.column_stack(columns)


def test_paper_seasonal_experiment_qualitative() -> None:
    # Paper: random walk, PBO ~ 55% (overfit); with a planted monthly effect, PBO ~ 13%.
    # Our draws differ, so we test the paper's claim, not its digits: on every seed the
    # planted effect lowers PBO, and on average the gap is large. Observed with these
    # seeds: random 0.258 / 0.640 / 0.430; effect 0.085 / 0.165 / 0.003.
    random_walk = [
        probability_of_backtest_overfitting(_seasonal_strategies(s, effect=False), 16).pbo
        for s in (1, 2, 3)
    ]
    seasonal = [
        probability_of_backtest_overfitting(_seasonal_strategies(s, effect=True), 16).pbo
        for s in (1, 2, 3)
    ]
    assert all(e < r for e, r in zip(seasonal, random_walk, strict=True))
    assert np.mean(random_walk) - np.mean(seasonal) > 0.2


# ---------------------------------------------------------------------------------------
# 4. Fast path, summary statistics, validation
# ---------------------------------------------------------------------------------------


def test_fast_sharpe_path_matches_generic_metric() -> None:
    x = np.random.default_rng(9).normal(0.01, 1.0, size=(320, 12)) + 50.0  # large offset
    fast = probability_of_backtest_overfitting(x, 8)
    generic = probability_of_backtest_overfitting(x, 8, metric=column_sharpe_ratios)
    assert np.allclose(fast.is_performance, generic.is_performance, rtol=1e-10, atol=1e-12)
    assert np.allclose(fast.oos_performance, generic.oos_performance, rtol=1e-10, atol=1e-12)
    assert np.array_equal(fast.logits, generic.logits)


def test_degradation_regression_and_loss_probability() -> None:
    result = probability_of_backtest_overfitting(np.random.default_rng(4).normal(size=(160, 8)), 8)
    slope, intercept = np.polyfit(result.is_performance, result.oos_performance, 1)
    assert result.degradation_slope == pytest.approx(slope)
    assert result.degradation_intercept == pytest.approx(intercept)
    assert result.prob_oos_loss == pytest.approx(np.mean(result.oos_performance < 0))
    assert result.pbo == pytest.approx(np.mean(result.logits <= 0))


def test_degradation_is_nan_when_in_sample_never_varies() -> None:
    def constant_winner(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray([1.0, float(rows[:, 1].mean())], dtype=np.float64)

    result = probability_of_backtest_overfitting(
        np.random.default_rng(5).normal(size=(16, 2)), 4, metric=constant_winner
    )
    assert math.isnan(result.degradation_slope)
    assert math.isnan(result.degradation_intercept)


@pytest.mark.parametrize(
    ("shape", "s", "message"),
    [
        ((64,), 8, "2-D"),
        ((64, 1), 8, "at least 2 strategy"),
        ((64, 3), 7, "even number"),
        ((64, 3), 0, "even number"),
        ((60, 3), 8, "trim 4 row"),
        ((2, 3), 2, "at least 2 rows"),
    ],
)
def test_rejects_bad_shapes(shape: tuple[int, ...], s: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        probability_of_backtest_overfitting(np.ones(shape), s)


def test_rejects_non_finite_input() -> None:
    x = np.random.default_rng(6).normal(size=(32, 3))
    x[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        probability_of_backtest_overfitting(x, 4)


def test_rejects_zero_variance_strategy() -> None:
    x = np.random.default_rng(6).normal(size=(32, 3))
    x[:, 2] = 0.0  # a strategy that never trades: Sharpe undefined
    with pytest.raises(ValueError, match="finite"):
        probability_of_backtest_overfitting(x, 4)


def test_rejects_metric_with_wrong_shape() -> None:
    def bad_metric(rows: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(1)

    with pytest.raises(ValueError, match="3 finite numbers"):
        probability_of_backtest_overfitting(np.ones((32, 3)), 4, metric=bad_metric)
