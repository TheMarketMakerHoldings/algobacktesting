"""Is a Sharpe ratio statistically real? Probabilistic and Deflated Sharpe Ratios.

Every Sharpe ratio in this module is **per period** (not annualized): if you have daily
returns, pass the daily Sharpe ratio and the number of daily observations. Use
:func:`deannualize_sharpe` to convert an annual figure.

Kurtosis is the ordinary (non-excess) kurtosis, so a Normal distribution has kurtosis 3.

References
----------
Bailey, D. H. and López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
*Journal of Risk* 15(2).

Bailey, D. H. and López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*
40(5). SSRN 2460551.

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
        skew: Sample skewness (0 for a symmetric distribution).
        kurtosis: Sample kurtosis, non-excess (3 for a Normal distribution).

    """

    sharpe: float
    n_obs: int
    skew: float
    kurtosis: float


def return_stats(returns: ArrayLike) -> ReturnStats:
    """Compute the Sharpe ratio, length, skewness and kurtosis of a return series.

    Args:
        returns: Per-period returns (e.g. daily), excess of the risk-free rate if relevant.

    Raises:
        ValueError: Fewer than 2 observations, non-finite values, or zero variance.

    >>> s = return_stats([0.01, -0.02, 0.03, 0.00, 0.01])
    >>> round(s.sharpe, 4), s.n_obs
    (0.3586, 5)

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
    """Per-period Sharpe ratio: mean / sample standard deviation (ddof=1)."""
    return return_stats(returns).sharpe


def annualize_sharpe(sharpe: float, periods_per_year: float) -> float:
    """Scale a per-period Sharpe ratio to annual (assumes i.i.d. returns).

    >>> round(annualize_sharpe(0.1, 252), 4)
    1.5875

    """
    return sharpe * math.sqrt(periods_per_year)


def deannualize_sharpe(annual_sharpe: float, periods_per_year: float) -> float:
    """Convert an annual Sharpe ratio to per-period (assumes i.i.d. returns).

    >>> round(deannualize_sharpe(2.5, 250), 4)
    0.1581

    """
    return annual_sharpe / math.sqrt(periods_per_year)


def _sharpe_dispersion(sharpe: float, skew: float, kurtosis: float) -> float:
    """sqrt(1 - skew*SR + (kurtosis-1)/4 * SR^2): scales the Sharpe ratio's standard error."""
    value = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if value <= 0.0:
        raise ValueError(
            "skew/kurtosis/Sharpe combination gives a non-positive variance; "
            "check that kurtosis is non-excess (Normal = 3)"
        )
    return math.sqrt(value)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark`` (PSR).

    Accounts for track-record length and for non-Normal returns: negative skew and fat
    tails make a given Sharpe ratio *less* convincing.

    Args:
        sharpe: Observed per-period Sharpe ratio.
        n_obs: Number of return observations behind it.
        benchmark: Per-period Sharpe ratio to beat (0 = "is it positive at all?").
        skew: Skewness of the returns.
        kurtosis: Non-excess kurtosis of the returns (Normal = 3).

    Returns:
        A probability in [0, 1].

    >>> round(probabilistic_sharpe_ratio(0.1, 252), 4)
    0.9437

    """
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / _sharpe_dispersion(sharpe, skew, kurtosis)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: float, sharpe_variance: float) -> float:
    """Return the expected best Sharpe ratio among ``n_trials`` skill-less strategies (SR0).

    Even if every strategy you tried has a true Sharpe ratio of zero, the *best* of them
    will look good by luck. This is that luck threshold, from extreme value theory.

    Args:
        n_trials: Number of independent strategy configurations tried (>= 1).
        sharpe_variance: Variance of the Sharpe ratios across those trials, in the same
            per-period units as the Sharpe ratio you will compare against it.

    >>> round(expected_max_sharpe(100, 0.5 / 250), 4)  # Bailey & López de Prado (2014)
    0.1132

    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance cannot be negative")
    if n_trials == 1:
        return 0.0  # no selection among trials, so no selection bias
    g = EULER_GAMMA
    return math.sqrt(sharpe_variance) * (
        (1 - g) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + g * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    n_trials: float,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the best of ``n_trials`` strategies has a real (positive) Sharpe (DSR).

    The PSR of the selected strategy, measured against :func:`expected_max_sharpe` instead
    of zero, so it is penalised for every strategy you tried before picking this one.

    Args:
        sharpe: Per-period Sharpe ratio of the strategy you selected.
        n_obs: Number of return observations behind it.
        n_trials: Number of independent configurations tried during research.
        sharpe_variance: Variance of the per-period Sharpe ratios across those trials.
        skew: Skewness of the selected strategy's returns.
        kurtosis: Non-excess kurtosis of the selected strategy's returns.

    The numerical example of Bailey & López de Prado (2014): 100 trials, annualized SR
    variance 0.5, a 2.5 annualized SR over 5 years of daily data, skew -3, kurtosis 10.

    >>> dsr = deflated_sharpe_ratio(
    ...     deannualize_sharpe(2.5, 250), 1250,
    ...     n_trials=100, sharpe_variance=0.5 / 250, skew=-3, kurtosis=10,
    ... )
    >>> round(dsr, 3)
    0.9

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
    """Observations needed before ``sharpe`` beats ``benchmark`` at ``confidence`` (MinTRL).

    Answers "how long must this track record be before I can trust it?". The result is a
    number of return observations, in the same frequency as the per-period Sharpe ratio.

    >>> round(min_track_record_length(0.1))  # daily SR 0.1 (~1.6 annual), Normal returns
    272

    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if sharpe <= benchmark:
        raise ValueError("sharpe must exceed benchmark; otherwise no track record suffices")
    z = _NORMAL.inv_cdf(confidence)
    return 1.0 + _sharpe_dispersion(sharpe, skew, kurtosis) ** 2 * (z / (sharpe - benchmark)) ** 2
