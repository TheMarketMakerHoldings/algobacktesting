# algobacktesting

**Is your backtest real? Overfitting checks for trading research.**

> **Status: pre-release, under construction.** Not yet on PyPI. The functions below are
> being built and verified against their source papers; do not rely on them until v0.1.0.

Try enough strategy variations and one of them will look brilliant by luck alone.
`algobacktesting` implements statistical tests from Bailey & López de Prado's research that
tell a real edge apart from an overfit one.

## Planned for v0.1

| Question | Tool |
| --- | --- |
| Is this Sharpe ratio still impressive after counting every strategy I tried? | Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR), minimum track record length |
| How likely is it that my "best" strategy is just the luckiest one? | Probability of Backtest Overfitting (PBO) via combinatorially symmetric cross-validation |
| How do I cross-validate on time series without leaking the future? | Purged k-fold with embargo, combinatorial purged cross-validation |

## Why this matters

<!-- DANIEL: write this section yourself, in your own words. Placeholder until then. -->

_To be written by the author._

## Installation

Not yet published. After v0.1.0:

```bash
pip install algobacktesting
```

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
