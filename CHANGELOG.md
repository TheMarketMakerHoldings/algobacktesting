# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `algobacktesting.sharpe`, verified against Bailey & López de Prado (2012, 2014):
  - `probabilistic_sharpe_ratio` (PSR)
  - `deflated_sharpe_ratio` (DSR)
  - `expected_max_sharpe` (SR0)
  - `min_track_record_length` (MinTRL)
  - `return_stats`, `sharpe_ratio`, `annualize_sharpe`, `deannualize_sharpe`
- `algobacktesting.pbo`, verified against Bailey, Borwein, López de Prado & Zhu (2017):
  - `probability_of_backtest_overfitting` (CSCV), with logits, performance degradation
    and the probability of out-of-sample loss
- `algobacktesting.cv`, built from the standard definitions in López de Prado (2018).
  **Pending review against the book.**
  - `purged_kfold` (purging and embargo)
  - `combinatorial_purged_cv` (CPCV, with backtest paths)
- Plain-English explainers in `docs/explainers/` and a runnable demo in
  `examples/overfit_demo.py`.
- `examples/leakage_demo.py`: a no-skill model scores +0.82 under shuffled k-fold,
  +0.05 under unpurged contiguous k-fold and 0.00 under purged k-fold.

- `docs/index.html`: a one-page site that asks "are you actually profitable, or just
  lucky?". Enter a win rate, reward-to-risk ratio and number of trades, or paste trade
  results, and it reports the chance the edge is real and the trades needed to prove it.
  `tests/test_site.py` runs the page's JavaScript with Node.js and checks it against the
  library on more than 500 cases.

### Safety

These inputs raise clear errors instead of returning plausible wrong numbers:
- NaN or infinite inputs
- impossible moments (kurtosis < 1 + skew², which usually means excess kurtosis was
  passed)
- an effective number of trials between 1 and 2
- a matrix passed where a single return series is expected
- more than one million CSCV splits
