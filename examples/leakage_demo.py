"""Why purging matters: cross-validating a model that has no real skill.

Run it::

    uv run python examples/leakage_demo.py

Daily returns here are pure noise, so nothing can predict them. The label for each day is
the return over the *next* ``HORIZON`` days, so neighbouring days share most of their
label. The features are trailing moving averages, which change slowly from day to day.
A nearest-neighbour model therefore finds yesterday and tomorrow as "similar" days and
copies their almost identical labels.

Three ways of cross-validating the same model:

* shuffled k-fold (a common default): neighbours of every test day sit in the training set;
* contiguous k-fold without purging: only days next to a fold edge leak;
* purged k-fold: training days whose label overlaps the test fold are removed.

The honest score is zero. Anything above it is leakage.

One dataset's score is noisy (it varies by about 0.07 from dataset to dataset), so a
leak of 0.05 cannot be seen in a single backtest even though it inflates every one of
them. The demo therefore averages many datasets and also reports the paired difference
between the unpurged and purged scores on the same data.

The full run takes about two minutes. Reference result (200 datasets): shuffled +0.816,
unpurged +0.050, purged +0.000, purged with a 5% embargo -0.001, leak removed by purging
+0.050 (s.e. 0.002), unpurged above purged in 199 of 200 datasets.

The returns here are independent from day to day, so purging alone is enough. An embargo
is for data whose returns or features stay correlated beyond the label window.
"""

from __future__ import annotations

import numpy as np

from algobacktesting import purged_kfold

DAYS = 3000
HORIZON = 20  # each label is the next 20 days' return
WINDOWS = (20, 40, 80, 160, 320)  # trailing moving-average features
FOLDS = 30
NEIGHBOURS = 5
DATASETS = 200


def make_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, labels) built from i.i.d. noise returns."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 1.0, DAYS + max(WINDOWS) + HORIZON)
    csum = np.concatenate([[0.0], np.cumsum(returns)])
    t = np.arange(max(WINDOWS), max(WINDOWS) + DAYS)  # index of "today" for each row
    features = np.column_stack([(csum[t + 1] - csum[t + 1 - w]) / w for w in WINDOWS])
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    labels = csum[t + 1 + HORIZON] - csum[t + 1]  # sum of the next HORIZON returns
    return features, labels


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Predict each test row as the mean label of its nearest training rows."""
    d2 = (
        (x_test**2).sum(axis=1)[:, None]
        + (x_train**2).sum(axis=1)[None, :]
        - 2.0 * x_test @ x_train.T
    )  # squared distances, test rows by training rows
    nearest = np.argpartition(d2, NEIGHBOURS, axis=1)[:, :NEIGHBOURS]
    return np.asarray(y_train[nearest].mean(axis=1))


def oos_correlation(
    features: np.ndarray, labels: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]
) -> float:
    """Correlation between out-of-sample predictions and labels, pooled over folds."""
    predictions = np.full(labels.size, np.nan)
    for train, test in splits:
        predictions[test] = knn_predict(features[train], labels[train], features[test])
    return float(np.corrcoef(predictions, labels)[0, 1])


def splits_for(scheme: str, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the train/test splits for one cross-validation scheme."""
    rows = np.arange(DAYS)
    if scheme == "shuffled":
        folds = np.array_split(np.random.default_rng(seed).permutation(DAYS), FOLDS)
        return [(np.setdiff1d(rows, fold), fold) for fold in folds]
    if scheme == "contiguous":
        folds = np.array_split(rows, FOLDS)
        return [(np.setdiff1d(rows, fold), fold) for fold in folds]
    # Row t's label covers days t+1 .. t+HORIZON, so its interval is [t, t + HORIZON].
    embargo = 0.05 if scheme == "purged+embargo" else 0.0
    purged = purged_kfold(rows, rows + HORIZON, FOLDS, embargo=embargo)
    return [(s.train, s.test) for s in purged]


def main(n_datasets: int = DATASETS) -> dict[str, tuple[float, float]]:
    """Score every scheme on ``n_datasets`` seeded datasets; return mean and standard error."""
    schemes = {
        "shuffled k-fold": "shuffled",
        "contiguous k-fold, no purging": "contiguous",
        "purged k-fold": "purged",
        "purged k-fold, 5% embargo": "purged+embargo",
    }
    scores: dict[str, list[float]] = {label: [] for label in schemes}
    for seed in range(n_datasets):
        features, labels = make_dataset(seed)
        for label, scheme in schemes.items():
            scores[label].append(oos_correlation(features, labels, splits_for(scheme, seed)))

    def mean_and_se(values: np.ndarray) -> tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))

    results = {label: mean_and_se(np.array(v)) for label, v in scores.items()}
    leak = np.array(scores["contiguous k-fold, no purging"]) - np.array(scores["purged k-fold"])
    results["leak removed by purging"] = mean_and_se(leak)

    print(f"A model with zero real skill, averaged over {n_datasets} datasets. Honest score: 0")
    print()
    for label, (mean, se) in results.items():
        print(f"  {label:<32} {mean:+.3f}  (s.e. {se:.3f})")
    above = int(np.sum(leak > 0))
    print(f"\n  unpurged score above purged score in {above} of {n_datasets} datasets")
    return results


if __name__ == "__main__":
    main()
