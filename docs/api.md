# API reference

Generated from the docstrings by `docs/build_api.py`. Do not edit by hand.
Each function's full docstring cites the equation and page it implements.

Version 0.1.0. Everything below is importable from
`algobacktesting` directly.

## Is this Sharpe ratio real?

Module `algobacktesting.sharpe`.

### `ReturnStats`

```python
ReturnStats(sharpe: float, n_obs: int, skew: float, kurtosis: float) -> None
```

Moments of a return series needed to judge its Sharpe ratio.

### `annualize_sharpe`

```python
annualize_sharpe(sharpe: float, periods_per_year: float) -> float
```

Scale a per-period Sharpe ratio to annual, assuming i.i.d. returns.

### `deannualize_sharpe`

```python
deannualize_sharpe(annual_sharpe: float, periods_per_year: float) -> float
```

Convert an annual Sharpe ratio to per-period, assuming i.i.d. returns.

### `deflated_sharpe_ratio`

```python
deflated_sharpe_ratio(sharpe: float, n_obs: float, *, n_trials: float, sharpe_variance: float, skew: float = 0.0, kurtosis: float = 3.0) -> float
```

Return the probability that the best of ``n_trials`` strategies is real (DSR).

### `expected_max_sharpe`

```python
expected_max_sharpe(n_trials: float, sharpe_variance: float) -> float
```

Return the expected best Sharpe ratio among ``n_trials`` skill-less strategies (SR0).

### `min_track_record_length`

```python
min_track_record_length(sharpe: float, *, benchmark: float = 0.0, skew: float = 0.0, kurtosis: float = 3.0, confidence: float = 0.95) -> float
```

Return how many observations are needed before ``sharpe`` beats ``benchmark`` (MinTRL).

### `probabilistic_sharpe_ratio`

```python
probabilistic_sharpe_ratio(sharpe: float, n_obs: float, *, benchmark: float = 0.0, skew: float = 0.0, kurtosis: float = 3.0) -> float
```

Return the probability that the true Sharpe ratio exceeds ``benchmark`` (PSR).

### `return_stats`

```python
return_stats(returns: ArrayLike) -> ReturnStats
```

Compute the Sharpe ratio, length, skewness and kurtosis of a return series.

### `sharpe_ratio`

```python
sharpe_ratio(returns: ArrayLike) -> float
```

Return the per-period Sharpe ratio: mean / sample standard deviation (ddof=1).

## Did the search just pick the luckiest strategy?

Module `algobacktesting.pbo`.

### `PBOResult`

```python
PBOResult(pbo: float, logits: NDArray[np.float64], relative_ranks: NDArray[np.float64], is_performance: NDArray[np.float64], oos_performance: NDArray[np.float64], degradation_slope: float, degradation_intercept: float, prob_oos_loss: float, n_partitions: int, n_strategies: int, n_combinations: int) -> None
```

Output of :func:`probability_of_backtest_overfitting`.

### `column_sharpe_ratios`

```python
column_sharpe_ratios(returns: NDArray[np.float64]) -> NDArray[np.float64]
```

Return the per-period Sharpe ratio of each column (mean / std with ddof=1).

### `probability_of_backtest_overfitting`

```python
probability_of_backtest_overfitting(returns: ArrayLike, n_partitions: int = 16, *, metric: Metric | None = None) -> PBOResult
```

Estimate the probability of backtest overfitting with CSCV ([BBLZ] Algorithm 2.3).

## Cross-validation without leakage

Module `algobacktesting.cv`.

### `CPCVResult`

```python
CPCVResult(splits: list[Split], test_groups: list[tuple[int, ...]], groups: list[IntArray], paths: IntArray) -> None
```

Output of :func:`combinatorial_purged_cv`.

### `Split`

```python
Split(train: IntArray, test: IntArray)
```

Row indices for one train/test split (both sorted ascending).

### `combinatorial_purged_cv`

```python
combinatorial_purged_cv(label_starts: ArrayLike, label_ends: ArrayLike, n_groups: int = 6, n_test_groups: int = 2, *, embargo: float = 0.0) -> CPCVResult
```

Return combinatorial purged CV splits and backtest paths (AFML Ch. 12, from definitions).

### `purged_kfold`

```python
purged_kfold(label_starts: ArrayLike, label_ends: ArrayLike, n_splits: int = 5, *, embargo: float = 0.0) -> list[Split]
```

Return k-fold splits with purging and embargo (AFML Ch. 7, from the definitions).

## Constants

- `EULER_GAMMA` = `0.5772156649015329`
