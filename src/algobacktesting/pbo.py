"""Probability of Backtest Overfitting (PBO) via combinatorially symmetric cross-validation.

You tried N strategy configurations and kept the one with the best backtest. PBO estimates
how often that selection procedure picks a configuration that then performs *below the
median* of the N configurations out of sample. A PBO near 0 means the selection procedure
generalizes; near 0.5 or above means it is mostly picking noise.

Reference
---------
[BBLZ] Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017). "The
Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4).
SSRN 2326253 (revised February 2015). Algorithm 2.3 (CSCV), pp. 11-12; Section 3,
pp. 13-14.

Implementation choices the paper leaves open (documented, deliberate):

* PBO is the share of combinations whose logit is <= 0, the paper's estimator
  phi = integral of f(lambda) from -inf to 0 (Section 3.1). Definition 2.2 states a strict
  "rank < N/2"; the two differ only when the selected strategy lands exactly on the median.
  Consequence worth knowing: with an **odd** number of strategies N, the exact median rank
  counts as overfit, so pure noise gives PBO = ((N + 1) / 2) / N on average instead of
  0.5. That is 2/3 for N = 3 and 0.6 for N = 5, but negligible for large N. For even N it
  is exactly 0.5. Identical strategies all tie at the median, giving PBO = 1.
* Ranks are ascending (rank N = best), as in the paper's example on p. 9. Ties in a
  ranking get the average of the tied ranks. If several strategies tie for best in
  sample, the first column is selected.
* The rows must split into S blocks of exactly equal length, as the paper requires;
  otherwise a ``ValueError`` asks you to trim the series.

"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PBOResult",
    "average_ranks",
    "column_sharpe_ratios",
    "probability_of_backtest_overfitting",
]

Metric: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]

MAX_COMBINATIONS = 1_000_000  # C(22, 11) = 705,432 fits; C(24, 12) = 2,704,156 does not


@dataclass(frozen=True)
class PBOResult:
    """Output of :func:`probability_of_backtest_overfitting`.

    All arrays have one entry per train/test combination, in the same order.

    Attributes:
        pbo: Probability of backtest overfitting: share of combinations whose logit <= 0.
        logits: lambda_c = ln(w_c / (1 - w_c)) for each combination.
        relative_ranks: w_c = OOS rank of the in-sample winner / (N + 1), in (0, 1).
        is_performance: In-sample performance of the in-sample winner.
        oos_performance: Out-of-sample performance of that same strategy.
        degradation_slope: OLS slope of ``oos_performance`` on ``is_performance``
            ([BBLZ] Section 3.2). Negative means better IS results foretell worse OOS
            results. NaN if in-sample performance never varies.
        degradation_intercept: OLS intercept of that regression (NaN likewise).
        prob_oos_loss: Share of combinations where the in-sample winner has negative
            out-of-sample performance ([BBLZ] Section 3.2).
        n_partitions: S, the number of row blocks.
        n_strategies: N, the number of strategy columns.
        n_combinations: C(S, S/2), the number of train/test splits evaluated.

    """

    pbo: float
    logits: NDArray[np.float64]
    relative_ranks: NDArray[np.float64]
    is_performance: NDArray[np.float64]
    oos_performance: NDArray[np.float64]
    degradation_slope: float
    degradation_intercept: float
    prob_oos_loss: float
    n_partitions: int
    n_strategies: int
    n_combinations: int


def column_sharpe_ratios(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the per-period Sharpe ratio of each column (mean / std with ddof=1).

    This is the default performance metric for
    :func:`probability_of_backtest_overfitting`.
    """
    return np.asarray(returns.mean(axis=0) / returns.std(axis=0, ddof=1), dtype=np.float64)


def average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank values ascending from 1 to N; tied values share the average of their ranks.

    >>> average_ranks(np.array([0.5, 1.1, 0.7]))  # [BBLZ] p. 9 example
    array([1., 3., 2.])

    """
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    below = np.cumsum(counts) - counts  # how many values are strictly smaller
    return np.asarray((below + (counts + 1) / 2.0)[inverse], dtype=np.float64)


def _validate(matrix: NDArray[np.float64], n_partitions: int) -> None:
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2-D array: rows = time, columns = strategies")
    t, n = matrix.shape
    if n < 2:
        raise ValueError("need at least 2 strategy columns to rank")
    if n_partitions < 2 or n_partitions % 2:
        raise ValueError("n_partitions (S) must be an even number >= 2")
    n_combinations = math.comb(n_partitions, n_partitions // 2)
    if n_combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"n_partitions={n_partitions} means {n_combinations:,} train/test splits "
            f"(limit {MAX_COMBINATIONS:,}); use S <= 22. The paper's example uses S = 16"
        )
    if t % n_partitions:
        raise ValueError(
            f"{t} rows do not split into {n_partitions} equal blocks; "
            f"trim {t % n_partitions} row(s) from the start or end of the series"
        )
    if t // 2 < 2:
        raise ValueError("each half of the sample needs at least 2 rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns contain NaN or infinite values")


def probability_of_backtest_overfitting(
    returns: ArrayLike,
    n_partitions: int = 16,
    *,
    metric: Metric | None = None,
) -> PBOResult:
    """Estimate the probability of backtest overfitting with CSCV ([BBLZ] Algorithm 2.3).

    The rows of ``returns`` are split into ``n_partitions`` (S) equal, consecutive blocks.
    For every way of choosing S/2 blocks as the training set (and the other S/2 as the
    test set), the strategy with the best in-sample performance is found, and its
    out-of-sample rank among all N strategies is recorded. Blocks are joined in their
    original time order on both sides.

    Args:
        returns: Matrix of shape (T, N): T synchronized observations (rows, oldest first)
            of N strategy configurations (columns), e.g. daily P&L or returns.
        n_partitions: S, an even number of equal row blocks. The number of combinations
            evaluated is C(S, S/2), e.g. 12,870 for S = 16.
        metric: Function mapping a (rows, N) array to N performance numbers, higher is
            better. Defaults to :func:`column_sharpe_ratios`. Order-dependent metrics
            (e.g. drawdown-based) receive the rows in time order.

    Returns:
        A :class:`PBOResult`.

    Raises:
        ValueError: Bad shape, odd or too small S, rows not divisible by S, non-finite
            input, or a performance value that is NaN or infinite.

    """
    matrix = np.asarray(returns, dtype=np.float64)
    _validate(matrix, n_partitions)
    t, n = matrix.shape
    s = n_partitions
    blocks = matrix.reshape(s, t // s, n)  # consecutive blocks, time order preserved

    if metric is None:
        evaluate = _block_sharpe_evaluator(blocks)
    else:
        user_metric = metric

        def evaluate(chosen: tuple[int, ...]) -> NDArray[np.float64]:
            rows = blocks[list(chosen)].reshape(-1, n)  # chosen is sorted: time order
            return np.asarray(user_metric(rows), dtype=np.float64)

    combos = list(itertools.combinations(range(s), s // 2))
    is_perf = np.empty(len(combos))
    oos_perf = np.empty(len(combos))
    omega = np.empty(len(combos))
    all_blocks = frozenset(range(s))

    for i, train in enumerate(combos):
        test = tuple(sorted(all_blocks.difference(train)))
        r_is, r_oos = evaluate(train), evaluate(test)
        for label, values in (("in-sample", r_is), ("out-of-sample", r_oos)):
            if values.shape != (n,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{label} performance must be {n} finite numbers; got {values!r}. "
                    "A strategy may have zero variance within a block selection."
                )
        best = int(np.argmax(r_is))  # first column wins an in-sample tie
        is_perf[i] = r_is[best]
        oos_perf[i] = r_oos[best]
        omega[i] = average_ranks(r_oos)[best] / (n + 1)

    logits = np.log(omega / (1.0 - omega))
    slope, intercept = _ols(is_perf, oos_perf)
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        relative_ranks=omega,
        is_performance=is_perf,
        oos_performance=oos_perf,
        degradation_slope=slope,
        degradation_intercept=intercept,
        prob_oos_loss=float(np.mean(oos_perf < 0.0)),
        n_partitions=s,
        n_strategies=n,
        n_combinations=len(combos),
    )


def _block_sharpe_evaluator(
    blocks: NDArray[np.float64],
) -> Callable[[tuple[int, ...]], NDArray[np.float64]]:
    """Fast exact column Sharpe ratios for any union of blocks, from per-block sums.

    Data are centred on each column's overall mean first, so the sums of squares stay well
    conditioned. The result equals ``column_sharpe_ratios`` on the joined rows.
    """
    shift = blocks.mean(axis=(0, 1))
    centred = blocks - shift
    sums = centred.sum(axis=1)  # (S, N)
    squares = (centred**2).sum(axis=1)  # (S, N)
    rows_per_block = blocks.shape[1]

    def evaluate(chosen: tuple[int, ...]) -> NDArray[np.float64]:
        idx = list(chosen)
        count = rows_per_block * len(idx)
        mean_c = sums[idx].sum(axis=0) / count
        var = (squares[idx].sum(axis=0) - count * mean_c**2) / (count - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray((mean_c + shift) / np.sqrt(var), dtype=np.float64)

    return evaluate


def _ols(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Return (slope, intercept) of y on x, or NaNs if x has no variation."""
    x_var = float(np.var(x))
    if x_var == 0.0 or math.isclose(x_var, 0.0, abs_tol=1e-300):
        return math.nan, math.nan
    slope = float(np.mean((x - x.mean()) * (y - y.mean())) / x_var)
    return slope, float(y.mean() - slope * x.mean())
