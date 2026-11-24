# Backlog (after v0.1)

Ideas that are deliberately **not** in v0.1. Nothing here gets built until v0.1.0 ships.

- From chart data to a verdict: load historical price bars (CSV) and simple entry and
  exit rules, produce per-trade results, and run them through the checks. A bridge into
  the existing tests, not a full backtesting engine. Private market data stays local.
- Multiple-testing haircuts for Sharpe ratios: Bonferroni, Holm, Benjamini-Hochberg and
  Harvey-Liu style.
- Estimating the effective number of independent trials by clustering correlated
  strategies.
- A False Strategy Theorem simulator.
- pandas and scikit-learn adapters (optional extras, never required).
- An example notebook on real futures data (local only; licensed data is never published).
- Stochastic dominance analysis for PBO (first- and second-order), from the PBO
  paper Section 3.3.
