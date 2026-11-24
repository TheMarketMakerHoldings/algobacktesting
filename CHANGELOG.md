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

- `tests/test_luck_calibration.py` and report section 5.6: with no edge, the site's check gives
  a false "proven" about 5% of the time for one setup and about 1% or less for five or more.
- `docs/explainers/04-profitable-or-lucky.md`: how the website's number is computed.

- API reference (`docs/api.md`), generated from the docstrings and checked by a test.
- `CONTRIBUTING.md` with the verification standard, `CITATION.cff`, `SECURITY.md`, and
  issue and pull-request templates.
- CI also runs on Windows and macOS, and on NumPy 1.26, the oldest version supported.
  Actions are pinned to exact commits, and the secret scan runs a pinned,
  checksum-verified scanner.
- `tests/test_docs.py`: the README's code is run, its tables are compared with the
  demos' output, and links into the repository must resolve.
- Website: screen-reader support (slider values, announced result, described chart),
  text contrast that meets WCAG AA in both themes, a link-preview image and a
  not-found page.
- Website redesigned as a trading terminal, with a trade importer: drop a CSV exported
  from TopstepX, Quantower, NinjaTrader or thinkorswim (or any file with a P&L column),
  pick one strategy, and get the verdict, an equity curve against what luck alone would
  have made, and a full breakdown: win rate with and without breakevens, what breakevens
  cost in fees, payoff, profit factor, largest win and loss, longest losing streak and
  maximum drawdown. Fees are subtracted unless the P&L column is already net.
  `tests/test_site_import.py` checks the importer on sample exports from all four
  platforms and on the built-in example.
- `RELEASING.md` (release checklist), `CODE_OF_CONDUCT.md`, and an issue template for
  suggesting a new check. The release workflow validates the package description and
  opens a GitHub release.

### Safety

These inputs raise clear errors instead of returning plausible wrong numbers:
- NaN or infinite inputs
- impossible moments (kurtosis < 1 + skew², which usually means excess kurtosis was
  passed)
- an effective number of trials between 1 and 2
- a matrix passed where a single return series is expected
- more than one million CSCV splits
