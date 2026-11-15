"""Is a Sharpe ratio statistically real? Probabilistic and Deflated Sharpe Ratios.

Every Sharpe ratio in this module is **per period** (not annualized): if you have daily
returns, pass the daily Sharpe ratio and the number of daily observations. Use
:func:`deannualize_sharpe` to convert an annual figure. The variance of trial Sharpe
ratios passed to :func:`expected_max_sharpe` and :func:`deflated_sharpe_ratio` must be in
the same per-period units.

Kurtosis is the raw (non-excess) kurtosis, so a Normal distribution has kurtosis 3.

References
----------
[BLdP2012] Bailey, D. H. and López de Prado, M. (2012). "The Sharpe Ratio Efficient
Frontier." *Journal of Risk* 15(2). SSRN 1821643.

[BLdP2014] Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of
Portfolio Management* 40(5). SSRN 2460551.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "EULER_GAMMA",
    "ReturnStats",
    "annualize_sharpe",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "return_stats",
    "sharpe_ratio",
]

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class ReturnStats:
    """Moments of a return series needed to judge its Sharpe ratio.

    Attributes:
        sharpe: Per-period Sharpe ratio, mean / sample standard deviation (ddof=1).
        n_obs: Number of return observations.
        skew: Sample skewness, m3 / m2**1.5 with population central moments
            (0 for a symmetric distribution).
        kurtosis: Sample raw kurtosis, m4 / m2**2 with population central moments
            (3 for a Normal distribution).

    """

    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float


def return_stats(returns: ArrayLike) -> ReturnStats:
    """Compute the Sharpe ratio, length, skewness and kurtosis of a return series.

    Args:
        returns: Per-period returns (e.g. daily), in excess of the risk-free rate if
            relevant.

    Raises:
        ValueError: Fewer than 2 observations, non-finite values, or zero variance.

    >>> s = return_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> round(s.sharpe, 4), s.n_obs, round(s.skew, 4), round(s.kurtosis, 4)
    (1.8974, 5, 0.0, 1.7)

    """
    x = np.asarray(returns, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("need at least 2 returns")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain NaN or infinite values")
    n = x.size
    dev = x - x.mean()
    m2 = float(np.mean(dev**2))
    if m2 == 0.0:
        raise ValueError("returns have zero variance; the Sharpe ratio is undefined")
    std = math.sqrt(m2 * n / (n - 1))
    return ReturnStats(
        sharpe=float(x.mean()) / std,
        n_obs=n,
        skew=float(np.mean(dev**3)) / m2**1.5,
        kurtosis=float(np.mean(dev**4)) / m2**2,
    )


def sharpe_ratio(returns: ArrayLike) -> float:
    """Return the per-period Sharpe ratio: mean / sample standard deviation (ddof=1)."""
    return return_stats(returns).sharpe


def annualize_sharpe(sharpe: float, periods_per_year: float) -> float:
    """Scale a per-period Sharpe ratio to annual, assuming i.i.d. returns.

    >>> round(annualize_sharpe(0.4576, 12), 3)  # [BLdP2012] Figure 6: monthly -> 1.585
    1.585

    """
    return sharpe * math.sqrt(periods_per_year)


def deannualize_sharpe(annual_sharpe: float, periods_per_year: float) -> float:
    """Convert an annual Sharpe ratio to per-period, assuming i.i.d. returns.

    >>> round(deannualize_sharpe(2.5, 250), 4)
    0.1581

    """
    return annual_sharpe / math.sqrt(periods_per_year)


def _sharpe_dispersion(sharpe: float, skew: float, kurtosis: float) -> float:
    """Return sqrt(1 - skew*SR + (kurtosis-1)/4 * SR^2), the PSR denominator's shape term."""
    value = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if value <= 0.0:
        raise ValueError(
            "skew/kurtosis/Sharpe combination gives a non-positive variance; "
            "check that kurtosis is raw (Normal = 3), not excess"
        )
    return math.sqrt(value)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: float,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Return the probability that the true Sharpe ratio exceeds ``benchmark`` (PSR).

    Implements [BLdP2012] Eq. (11)::

        PSR = Z[ (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew*SR + (kurtosis - 1)/4 * SR^2) ]

    where Z is the standard Normal CDF. Longer track records raise PSR; negative skew and
    fat tails lower it, because they make a given Sharpe ratio less reliable.

    Args:
        sharpe: Observed per-period Sharpe ratio.
        n_obs: Number of return observations behind it (the paper's n; at least 2).
        benchmark: Per-period Sharpe ratio to beat (0 asks "is there any skill at all?").
        skew: Skewness of the returns.
        kurtosis: Raw kurtosis of the returns (Normal = 3).

    Returns:
        A probability in [0, 1].

    [BLdP2012] p. 9: a monthly SR of 0.458 over two years, assuming Normal returns.

    >>> round(probabilistic_sharpe_ratio(0.458, 24), 3)
    0.982

    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / _sharpe_dispersion(sharpe, skew, kurtosis)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: float, sharpe_variance: float) -> float:
    """Return the expected best Sharpe ratio among ``n_trials`` skill-less strategies (SR0).

    Even if every strategy tried has a true Sharpe ratio of zero, the best of them looks
    good by luck. [BLdP2014] Eq. (2), from the extreme-value result in Appendix A.1,
    Eq. (6) with mean zero::

        SR0 = sqrt(V) * ( (1 - g) * Z^-1[1 - 1/N] + g * Z^-1[1 - 1/(N e)] )

    where g is the Euler-Mascheroni constant. The approximation slightly overestimates the
    exact expected maximum for small N and converges as N grows ([BLdP2014] A.2).

    Args:
        n_trials: Number of independent strategy configurations tried (N >= 1).
        sharpe_variance: Variance of the Sharpe ratios across those trials (V), in the
            same per-period units as the Sharpe ratio it will be compared against.

    >>> round(expected_max_sharpe(100, 1 / (2 * 250)), 4)  # [BLdP2014] p. 10
    0.1132

    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance cannot be negative")
    if n_trials == 1:
        return 0.0  # a single trial involves no selection, so no selection bias
    g = EULER_GAMMA
    return math.sqrt(sharpe_variance) * (
        (1 - g) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + g * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: float,
    *,
    n_trials: float,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Return the probability that the best of ``n_trials`` strategies is real (DSR).

    [BLdP2014] Eq. (2): the PSR of the selected strategy, measured against
    :func:`expected_max_sharpe` instead of zero, so the strategy is penalized for every
    alternative tried before it was picked.

    Args:
        sharpe: Per-period Sharpe ratio of the selected strategy.
        n_obs: Number of return observations behind it (the paper's T).
        n_trials: Number of independent configurations tried during research (N).
        sharpe_variance: Variance of the per-period Sharpe ratios across those trials.
        skew: Skewness of the selected strategy's returns.
        kurtosis: Raw kurtosis of the selected strategy's returns (Normal = 3).

    [BLdP2014] p. 10 example: N = 100, annualized SR variance 1/2, annualized SR 2.5
    over T = 1250 daily observations (250 per year), skew -3, kurtosis 10.

    >>> dsr = deflated_sharpe_ratio(
    ...     deannualize_sharpe(2.5, 250), 1250,
    ...     n_trials=100, sharpe_variance=1 / (2 * 250), skew=-3, kurtosis=10,
    ... )
    >>> round(dsr, 4)
    0.9004

    """
    return probabilistic_sharpe_ratio(
        sharpe,
        n_obs,
        benchmark=expected_max_sharpe(n_trials, sharpe_variance),
        skew=skew,
        kurtosis=kurtosis,
    )


def min_track_record_length(
    sharpe: float,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Return how many observations are needed before ``sharpe`` beats ``benchmark`` (MinTRL).

    Implements [BLdP2012] Eq. (13)::

        MinTRL = 1 + [1 - skew*SR + (kurtosis - 1)/4 * SR^2] * (Z_alpha / (SR - SR*))^2

    with Z_alpha the standard Normal quantile at ``confidence``. The result is a number of
    observations at the frequency of the per-period Sharpe ratio, not years.

    ``confidence`` must be at least 0.5: Eq. (13) squares Z_alpha, so below 0.5 it would
    silently return the answer for ``1 - confidence``.

    [BLdP2012] p. 11: annualized SR 2 against 1 at 95%, daily Normal returns, is 2.73 years.

    >>> round(min_track_record_length(2 / 252**0.5, benchmark=1 / 252**0.5) / 252, 2)
    2.73

    """
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be at least 0.5 and below 1")
    if sharpe <= benchmark:
        raise ValueError("sharpe must exceed benchmark; otherwise no track record suffices")
    z = _NORMAL.inv_cdf(confidence)
    return 1.0 + _sharpe_dispersion(sharpe, skew, kurtosis) ** 2 * (z / (sharpe - benchmark)) ** 2
