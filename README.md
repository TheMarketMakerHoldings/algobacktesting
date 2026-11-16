# algobacktesting

**Is your backtest real? Overfitting checks for trading research.**

> **Status: pre-release.** Not yet on PyPI. The Sharpe and PBO modules are verified
> against their source papers. The cross-validation module is built from the standard
> definitions and is awaiting review against the book.

Try enough strategy variations and one of them will look brilliant by luck alone.
`algobacktesting` implements statistical tests from Bailey & López de Prado's research that
tell a real edge apart from an overfit one.

| Question | Tool |
| --- | --- |
| Is this Sharpe ratio still impressive after counting every strategy I tried? | `deflated_sharpe_ratio` (DSR), `probabilistic_sharpe_ratio` (PSR), `min_track_record_length` |
| How likely is it that my "best" strategy is just the luckiest one? | `probability_of_backtest_overfitting` (PBO, via combinatorially symmetric cross-validation) |
| How do I cross-validate on time series without leaking the future? | `purged_kfold` (with embargo), `combinatorial_purged_cv` (CPCV) |

## Why this matters

<!-- DANIEL: write this section yourself, in your own words. Placeholder until then. -->

_To be written by the author._

## Quickstart

```python
import numpy as np
from algobacktesting import (
    column_sharpe_ratios,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_stats,
)

returns = np.random.default_rng(42).normal(0, 0.01, size=(1008, 200))  # 200 strategies, no edge

sharpes = column_sharpe_ratios(returns)  # per-day Sharpe of each strategy
best = return_stats(returns[:, sharpes.argmax()])  # the backtest you'd be tempted to trade

dsr = deflated_sharpe_ratio(
    best.sharpe,
    best.n_obs,
    n_trials=200,
    sharpe_variance=sharpes.var(ddof=1),
    skew=best.skew,
    kurtosis=best.kurtosis,
)
pbo = probability_of_backtest_overfitting(returns, n_partitions=16).pbo
```

`examples/overfit_demo.py` runs this for three worlds: no edge, a modest real edge, and a
strong real edge. Its output:

| World | Best backtest (annual Sharpe) | Deflated Sharpe | PBO |
| --- | --- | --- | --- |
| 200 strategies, no edge | 1.69 | 0.654 | 0.275 |
| plus one modest real edge (true Sharpe ~1.3) | 1.69 | 0.642 | 0.274 |
| plus one strong real edge (true Sharpe ~2.4) | 2.22 | 0.906 | 0.082 |

- **No edge:** pure luck produced a 1.69 Sharpe. DSR refuses to call it real.
- **Modest edge:** it exists, but the luckiest noise strategy still wins the search.
  Searching 200 variants buries a 1.3 Sharpe edge.
- **Strong edge:** it wins, and PBO drops to 0.08, so the selection process works. DSR is
  still below 0.95: after 200 tries, four years of data is not enough to *prove* it.

### Why purging matters

`examples/leakage_demo.py` cross-validates a model that has **no real skill**: a
nearest-neighbour model predicting 20-day returns from pure noise. Averaged over 200
datasets:

| Cross-validation scheme | Out-of-sample correlation (honest value: 0) |
| --- | --- |
| Shuffled k-fold | +0.816 (s.e. 0.002) |
| Contiguous k-fold, no purging | +0.050 (s.e. 0.005) |
| Purged k-fold | +0.000 (s.e. 0.005) |
| Purged k-fold with a 5% embargo | -0.001 (s.e. 0.005) |
| Leak removed by purging (paired difference) | +0.050 (s.e. 0.002) |

Shuffling makes a worthless model look excellent. Without purging, even unshuffled
folds leak at their edges. One dataset's score varies by about 0.07, so that leak is
invisible in a single backtest while it inflates every one of them. Purged k-fold
gives the honest answer.

Each function's docstring cites the equation and page it implements. `docs/explainers/`
has a plain-English guide to each module.

## Installation

Not yet published. After v0.1.0:

```bash
pip install algobacktesting
```

Requires Python 3.11+ and NumPy.

## Verification

- **Paper numbers:** every number printed in the source papers that these functions
  compute is reproduced in the tests at the paper's printed precision. For example: DSR
  0.9004, PSR 0.982 / 0.913 / 0.953, and minimum track record 2.73 / 2.83 / 3.24 / 4.99
  years.
- **Independent checks:** exact numerical integration, seeded Monte Carlo calibration,
  and a brute-force reference implementation for cross-validation, run on hundreds of
  random datasets.
- **Coverage and platforms:** 100% line and branch coverage, tested on Python 3.11–3.14.

## References

- Bailey, D. H. and López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
  *Journal of Risk* 15(2).
- Bailey, D. H. and López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*
  40(5).
- Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017). The Probability of
  Backtest Overfitting. *Journal of Computational Finance* 20(4).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## Disclaimer

Statistical tools for research. Not investment advice.

## License

MIT © 2026 Daniel Heidary. Published by The Market Maker Holdings & Co.
