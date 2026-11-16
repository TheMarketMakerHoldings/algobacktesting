"""Tests for algobacktesting.sharpe.

Three kinds of evidence:

1. Paper targets. Every number printed in the source papers is reproduced to the
   precision the paper prints it (rounded the same way, no extra tolerance).
2. Independent checks. Exact numerical integration, seeded Monte Carlo and closed-form
   small cases verify the formulas by a different route. Each tolerance states its reason.
3. Properties and input validation.

Sources:
  [BLdP2012] Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier",
             SSRN 1821643.
  [BLdP2014] Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", SSRN 2460551.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from algobacktesting.sharpe import (
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_stats,
    sharpe_ratio,
)

# ---------------------------------------------------------------------------------------
# 1a. [BLdP2014] "A numerical example", p. 9-10
#     N = 100, V[annualized SR] = 1/2, 250 obs/year, T = 1250, skew -3, kurtosis 10,
#     selected annualized SR = 2.5.
# ---------------------------------------------------------------------------------------

SR_DSR = 2.5 / math.sqrt(250)
VAR_DSR = 1 / (2 * 250)  # per-period variance, as written in the paper's SR0 expression


def test_dsr_paper_sr0() -> None:
    # p. 10: SR0 ~= 0.1132 (non-annualized)
    assert round(expected_max_sharpe(100, VAR_DSR), 4) == 0.1132


def test_dsr_paper_main_example() -> None:
    # p. 10: DSR = 0.9004 < 0.95. The paper prints SR0 rounded to 0.1132, but its 0.9004
    # comes from the unrounded SR0 (plugging in 0.1132 itself gives 0.9003).
    dsr = deflated_sharpe_ratio(
        SR_DSR, 1250, n_trials=100, sharpe_variance=VAR_DSR, skew=-3, kurtosis=10
    )
    assert round(dsr, 4) == 0.9004


def test_dsr_paper_fewer_trials() -> None:
    # p. 10: "after running only N=46 independent trials ... DSR would have been 0.9505"
    dsr = deflated_sharpe_ratio(
        SR_DSR, 1250, n_trials=46, sharpe_variance=VAR_DSR, skew=-3, kurtosis=10
    )
    assert round(dsr, 4) == 0.9505


def test_dsr_paper_normal_returns() -> None:
    # p. 10: "If the strategy had exhibited Normal returns ... DSR = 0.9505 after N=88"
    dsr = deflated_sharpe_ratio(SR_DSR, 1250, n_trials=88, sharpe_variance=VAR_DSR)
    assert round(dsr, 4) == 0.9505


# ---------------------------------------------------------------------------------------
# 1b. [BLdP2012] Section 3 PSR example, p. 9-10, Figures 6-7 (p. 28)
#     The track record is a two-Normal mixture whose parameters the paper states on p. 10:
#     mu1=-0.1, mu2=0.06, sigma1=0.12, sigma2=0.03, p=0.15. Figure 6 prints its moments
#     rounded (SR 0.458, skew -2.448, kurt 10.164, Ann. SR 1.585). The PSR values in the
#     text use the unrounded moments, so we rebuild them exactly from the mixture.
# ---------------------------------------------------------------------------------------


def _normal_mixture_moments(
    p: float, mu1: float, sd1: float, mu2: float, sd2: float
) -> tuple[float, float, float, float]:
    """Return (mean, sd, skew, raw kurtosis) of p*N(mu1, sd1^2) + (1-p)*N(mu2, sd2^2)."""

    def raw(k: int, m: float, s: float) -> float:  # raw moments of a Normal
        return [1.0, m, m * m + s * s, m**3 + 3 * m * s * s, m**4 + 6 * m * m * s * s + 3 * s**4][k]

    r1, r2, r3, r4 = (p * raw(k, mu1, sd1) + (1 - p) * raw(k, mu2, sd2) for k in (1, 2, 3, 4))
    var = r2 - r1**2
    c3 = r3 - 3 * r1 * r2 + 2 * r1**3
    c4 = r4 - 4 * r1 * r3 + 6 * r1**2 * r2 - 3 * r1**4
    return r1, math.sqrt(var), c3 / var**1.5, c4 / var**2


MIX = (0.15, -0.1, 0.12, 0.06, 0.03)
MIX_MEAN, MIX_SD, MIX_SKEW, MIX_KURT = _normal_mixture_moments(*MIX)
MIX_SR = MIX_MEAN / MIX_SD


def test_psr_paper_inputs_match_figure_6() -> None:
    # Figure 6 (p. 28) printed values; confirms kurtosis is raw (excess would be 7.164).
    assert round(MIX_MEAN, 3) == 0.036  # "0.036%"
    assert round(MIX_SD, 3) == 0.079  # "0.079%"
    assert round(MIX_SKEW, 3) == -2.448
    assert round(MIX_KURT, 3) == 10.164
    assert round(MIX_SR, 3) == 0.458
    assert round(annualize_sharpe(MIX_SR, 12), 3) == 1.585


def test_psr_paper_two_years_normal() -> None:
    # p. 9: assuming Normality, PSR(0) = 0.982 for the 24-month record
    assert round(probabilistic_sharpe_ratio(MIX_SR, 24), 3) == 0.982


def test_psr_paper_two_years_non_normal() -> None:
    # p. 9: with skew and kurtosis, PSR(0) = 0.913
    psr = probabilistic_sharpe_ratio(MIX_SR, 24, skew=MIX_SKEW, kurtosis=MIX_KURT)
    assert round(psr, 3) == 0.913


def test_psr_paper_three_years_non_normal() -> None:
    # p. 10: "should we have 3 years instead of 2, PSR(0) = 0.953"
    psr = probabilistic_sharpe_ratio(MIX_SR, 36, skew=MIX_SKEW, kurtosis=MIX_KURT)
    assert round(psr, 3) == 0.953


# ---------------------------------------------------------------------------------------
# 1c. [BLdP2012] Section 5 MinTRL examples, p. 11
#     Annualized SR = 2 vs benchmark 1 at 95%. Periods per year 252 / 52 / 12, matching
#     the paper's own conversion on p. 11 (30 obs = 2.5 years monthly, 0.5769 weekly,
#     0.119 daily).
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("periods_per_year", "skew", "kurtosis", "years"),
    [
        (252, 0.0, 3.0, 2.73),  # Figure 8: daily IID Normal
        (52, 0.0, 3.0, 2.83),  # Figure 9: weekly
        (12, 0.0, 3.0, 3.24),  # Figure 10: monthly
        (12, -0.72, 5.78, 4.99),  # Figure 11: monthly, HFR index moments
    ],
)
def test_min_trl_paper_examples(
    periods_per_year: int, skew: float, kurtosis: float, years: float
) -> None:
    root = math.sqrt(periods_per_year)
    n = min_track_record_length(2 / root, benchmark=1 / root, skew=skew, kurtosis=kurtosis)
    assert round(n / periods_per_year, 2) == years


# ---------------------------------------------------------------------------------------
# 2a. SR0: exact expected maximum by numerical integration
#     E[max of N iid N(0,1)] = integral of x * N * phi(x) * Phi(x)^(N-1) dx. [BLdP2014]
#     A.2 / Exhibit 3.1 report that the closed form overestimates this by less than 0.05
#     (variance 1), shrinking as N grows. We check exactly that claim.
# ---------------------------------------------------------------------------------------


def _exact_expected_max(n: int) -> float:
    x = np.linspace(-12.0, 12.0, 240_001)
    phi = np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    big_phi = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    f = x * n * phi * big_phi ** (n - 1)
    # Trapezoid rule, written out so it runs on every supported NumPy version.
    return float(np.sum((f[1:] + f[:-1]) * np.diff(x)) / 2)


@pytest.mark.parametrize("n", [2, 10, 46, 100, 500, 1000])
def test_sr0_overestimates_exact_by_less_than_paper_bound(n: int) -> None:
    gap = expected_max_sharpe(n, 1.0) - _exact_expected_max(n)
    assert 0.0 < gap < 0.05 or n == 2  # paper's claim covers N >= 10
    if n == 2:  # outside the paper's range; E[max of 2] = 1/sqrt(pi) exactly
        assert _exact_expected_max(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)


def test_sr0_gap_shrinks_with_more_trials() -> None:
    gaps = [expected_max_sharpe(n, 1.0) - _exact_expected_max(n) for n in (10, 100, 1000)]
    assert gaps[0] > gaps[1] > gaps[2] > 0


def test_exact_integral_agrees_with_seeded_monte_carlo() -> None:
    # Validates the reference integral itself. Tolerance = 5 Monte Carlo standard errors.
    rng = np.random.default_rng(2460551)
    maxima = rng.standard_normal((20_000, 100)).max(axis=1)
    se = maxima.std(ddof=1) / math.sqrt(maxima.size)
    assert abs(maxima.mean() - _exact_expected_max(100)) < 5 * se


def test_sr0_scales_with_standard_deviation() -> None:
    # Appendix A.1, Eq. (3)-(4): max{mu + sigma x} = mu + sigma max{x}
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))


# ---------------------------------------------------------------------------------------
# 2b. PSR calibration by seeded simulation
#     If the true SR equals the benchmark, PSR > 0.95 should happen ~5% of the time.
#     Tolerance: 4 binomial standard errors (4 * sqrt(.05*.95/4000) = 0.0138), which covers
#     Monte Carlo noise plus PSR's small finite-sample bias (it is an asymptotic result,
#     [BLdP2012] p. 11).
# ---------------------------------------------------------------------------------------

REPS = 4000
TOL = 4 * math.sqrt(0.05 * 0.95 / REPS)


def _false_positive_rate(
    draw: object, true_sr: float, n: int, rng: np.random.Generator, *, use_moments: bool
) -> float:
    assert callable(draw)
    hits = 0
    for _ in range(REPS):
        s = return_stats(draw(rng, n))
        skew, kurt = (s.skew, s.kurtosis) if use_moments else (0.0, 3.0)
        psr = probabilistic_sharpe_ratio(s.sharpe, n, benchmark=true_sr, skew=skew, kurtosis=kurt)
        hits += psr > 0.95
    return hits / REPS


def _draw_mixture(rng: np.random.Generator, n: int) -> np.ndarray:
    p, mu1, sd1, mu2, sd2 = MIX
    pick = rng.random(n) < p
    return np.where(pick, rng.normal(mu1, sd1, n), rng.normal(mu2, sd2, n))


def test_psr_is_calibrated_for_normal_returns() -> None:
    rng = np.random.default_rng(1821643)
    rate = _false_positive_rate(lambda g, n: g.normal(0.1, 1.0, n), 0.1, 250, rng, use_moments=True)
    assert abs(rate - 0.05) < TOL


def test_psr_moment_correction_matters_for_fat_tails() -> None:
    # Negative skew + fat tails: corrected PSR stays near 5%; ignoring the moments roughly
    # triples false positives (observed ~15%). This is the point of PSR.
    rng = np.random.default_rng(1821644)
    corrected = _false_positive_rate(_draw_mixture, MIX_SR, 1000, rng, use_moments=True)
    naive = _false_positive_rate(_draw_mixture, MIX_SR, 1000, rng, use_moments=False)
    assert abs(corrected - 0.05) < TOL
    assert naive > 0.10


# ---------------------------------------------------------------------------------------
# 2c. return_stats: closed-form small cases and a large Normal sample
# ---------------------------------------------------------------------------------------


def test_return_stats_hand_computed() -> None:
    s = return_stats([1.0, 2.0, 3.0, 4.0, 5.0])  # mean 3, m2 2, m4 6.8, sample var 2.5
    assert s.sharpe == pytest.approx(3 / math.sqrt(2.5))
    assert s.skew == pytest.approx(0.0, abs=1e-12)
    assert s.kurtosis == pytest.approx(6.8 / 4)


def test_return_stats_skewed_hand_computed() -> None:
    s = return_stats([0.0, 0.0, 0.0, 1.0])  # Bernoulli(1/4): skew 2/sqrt(3), kurt 7/3
    assert s.skew == pytest.approx(2 / math.sqrt(3))
    assert s.kurtosis == pytest.approx(7 / 3)
    assert sharpe_ratio([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.25 / math.sqrt(0.25))


def test_return_stats_large_normal_sample() -> None:
    # Tolerance: 5 asymptotic standard errors, sqrt(6/n) for skew and sqrt(24/n) for kurtosis.
    n = 1_000_000
    s = return_stats(np.random.default_rng(3).normal(0.0, 1.0, n))
    assert abs(s.skew) < 5 * math.sqrt(6 / n)
    assert abs(s.kurtosis - 3.0) < 5 * math.sqrt(24 / n)


# ---------------------------------------------------------------------------------------
# 3. Properties
# ---------------------------------------------------------------------------------------

sharpes = st.floats(0.01, 1.0)
skews = st.floats(-3.0, 3.0)
kurts = st.floats(1.0, 20.0)


@given(sr=sharpes, gap=st.floats(0.005, 1.0), skew=skews, kurt=kurts, conf=st.floats(0.5, 0.999))
def test_min_trl_inverts_psr(sr: float, gap: float, skew: float, kurt: float, conf: float) -> None:
    # Eq. (13) is Eq. (11) solved for n: PSR at n = MinTRL equals the confidence level.
    assume(kurt >= 1 + skew**2 and 1 - skew * sr + (kurt - 1) / 4 * sr**2 > 1e-6)
    n = min_track_record_length(sr, benchmark=sr - gap, skew=skew, kurtosis=kurt, confidence=conf)
    assert n >= 1.0
    assume(n >= 2)  # PSR needs at least 2 observations to estimate a Sharpe ratio
    psr = probabilistic_sharpe_ratio(sr, n, benchmark=sr - gap, skew=skew, kurtosis=kurt)
    assert psr == pytest.approx(conf, abs=1e-9)


def test_min_trl_is_one_observation_at_fifty_percent() -> None:
    # Z at 0.5 is 0, so Eq. (13) gives exactly 1: a coin-flip level of confidence needs no data.
    assert min_track_record_length(0.1, confidence=0.5) == 1.0


@given(sr=st.floats(-1.0, 1.0), n=st.integers(2, 10_000), skew=skews, kurt=kurts)
def test_psr_is_one_half_at_the_benchmark(sr: float, n: int, skew: float, kurt: float) -> None:
    assume(kurt >= 1 + skew**2 and 1 - skew * sr + (kurt - 1) / 4 * sr**2 > 1e-6)
    psr = probabilistic_sharpe_ratio(sr, n, benchmark=sr, skew=skew, kurtosis=kurt)
    assert psr == pytest.approx(0.5)


def test_psr_rises_with_track_record_length() -> None:
    values = [probabilistic_sharpe_ratio(0.1, n) for n in (10, 100, 1000)]
    assert values == sorted(values)


def test_psr_penalizes_negative_skew_and_fat_tails() -> None:
    normal = probabilistic_sharpe_ratio(0.2, 100)
    assert probabilistic_sharpe_ratio(0.2, 100, skew=-1.0) < normal
    assert probabilistic_sharpe_ratio(0.2, 100, kurtosis=10.0) < normal


def test_sr0_is_zero_for_one_trial_and_grows() -> None:
    assert expected_max_sharpe(1, 0.5) == 0.0
    values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000, 10_000)]
    assert values == sorted(values)


def test_dsr_falls_as_trials_rise() -> None:
    values = [
        deflated_sharpe_ratio(SR_DSR, 1250, n_trials=n, sharpe_variance=VAR_DSR)
        for n in (1, 10, 100, 1000)
    ]
    assert values == sorted(values, reverse=True)


def test_dsr_with_one_trial_equals_psr_against_zero() -> None:
    assert deflated_sharpe_ratio(0.1, 500, n_trials=1, sharpe_variance=0.01) == pytest.approx(
        probabilistic_sharpe_ratio(0.1, 500)
    )


@given(sr=st.floats(-5.0, 5.0), ppy=st.sampled_from([12, 52, 250, 252, 365]))
def test_annualize_round_trip(sr: float, ppy: int) -> None:
    assert deannualize_sharpe(annualize_sharpe(sr, ppy), ppy) == pytest.approx(sr)


# ---------------------------------------------------------------------------------------
# 3b. Input validation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "returns", [[], [0.01], [0.02, 0.02, 0.02], [0.01, float("nan")], [0.01, float("inf")]]
)
def test_return_stats_rejects_bad_input(returns: list[float]) -> None:
    with pytest.raises(ValueError, match=r"at least 2|NaN|zero variance"):
        return_stats(returns)


def test_psr_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        probabilistic_sharpe_ratio(0.1, 1)


@pytest.mark.parametrize(
    ("skew", "kurtosis"),
    [
        (0.0, 0.0),  # excess kurtosis of a Normal passed by mistake
        (0.0, 0.5),
        (2.0, 3.0),  # kurtosis 3 is impossible with skew 2 (needs >= 5)
        (10.0, 3.0),
    ],
)
def test_impossible_moments_rejected(skew: float, kurtosis: float) -> None:
    # Pearson's inequality: every distribution has kurtosis >= 1 + skew^2.
    with pytest.raises(ValueError, match="impossible moments"):
        probabilistic_sharpe_ratio(0.1, 100, skew=skew, kurtosis=kurtosis)
    with pytest.raises(ValueError, match="impossible moments"):
        min_track_record_length(0.1, skew=skew, kurtosis=kurtosis)


def test_moments_on_the_pearson_boundary_are_allowed() -> None:
    # The DSR paper's own example sits exactly on the boundary: skew -3, kurtosis 10.
    assert 0.0 < probabilistic_sharpe_ratio(0.1, 100, skew=-3.0, kurtosis=10.0) < 1.0


def test_degenerate_zero_standard_error_rejected() -> None:
    # skew 2, kurtosis 5 (boundary) and SR 1: 1 - 2*1 + (5-1)/4*1 = 0
    with pytest.raises(ValueError, match="zero standard error"):
        probabilistic_sharpe_ratio(1.0, 100, skew=2.0, kurtosis=5.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: probabilistic_sharpe_ratio(float("nan"), 100),
        lambda: probabilistic_sharpe_ratio(0.1, float("nan")),
        lambda: probabilistic_sharpe_ratio(0.1, 100, benchmark=float("inf")),
        lambda: probabilistic_sharpe_ratio(0.1, 100, skew=float("nan")),
        lambda: expected_max_sharpe(float("nan"), 1.0),
        lambda: expected_max_sharpe(10, float("nan")),
        lambda: min_track_record_length(float("nan")),
        lambda: annualize_sharpe(float("inf"), 12),
    ],
)
def test_non_finite_inputs_rejected(call: object) -> None:
    assert callable(call)
    with pytest.raises(ValueError, match="finite"):
        call()


@pytest.mark.parametrize("n", [1.0001, 1.3, 1.5, 1.999])
def test_sr0_rejects_between_one_and_two_trials(n: float) -> None:
    # The closed form turns negative near N = 1 (-0.78 at N = 1.01 before this guard),
    # which would turn the selection penalty into a bonus.
    with pytest.raises(ValueError, match="between 1 and 2"):
        expected_max_sharpe(n, 1.0)


def test_sr0_accurate_for_huge_trial_counts() -> None:
    # Evaluated as -Z^-1[1/N]: stays finite and increasing far beyond where 1 - 1/N == 1.0
    values = [expected_max_sharpe(n, 1.0) for n in (1e6, 1e12, 1e17, 1e100, 1e300)]
    assert all(math.isfinite(v) for v in values)
    assert values == sorted(values)


def test_return_stats_rejects_a_matrix() -> None:
    # Previously flattened a (T, N) matrix into one series without complaint.
    with pytest.raises(ValueError, match="1-D series"):
        return_stats(np.ones((3, 2)))


@pytest.mark.parametrize("ppy", [0, -12, float("nan")])
def test_bad_periods_per_year_rejected(ppy: float) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        annualize_sharpe(0.1, ppy)
    with pytest.raises(ValueError, match="periods_per_year"):
        deannualize_sharpe(1.0, ppy)


def test_sr0_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError, match="negative"):
        expected_max_sharpe(10, -1.0)


@pytest.mark.parametrize("sr", [0.1, 0.05])
def test_min_trl_rejects_sharpe_at_or_below_benchmark(sr: float) -> None:
    with pytest.raises(ValueError, match="must exceed benchmark"):
        min_track_record_length(sr, benchmark=0.1)


@pytest.mark.parametrize("conf", [0.0, 0.3, 0.4999, 1.0, 1.5, -0.1])
def test_min_trl_rejects_bad_confidence(conf: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        min_track_record_length(0.2, confidence=conf)
