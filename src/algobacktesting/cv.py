"""Leak-free cross-validation for financial labels: purged k-fold and CPCV.

In finance a label usually spans time. A trade opened on Monday may be labelled by what
happens up to Friday. If a training observation's label overlaps the test period, the
model has seen test-period information, and the cross-validated score is too good. Two
fixes:

* **Purging:** drop every training observation whose label interval overlaps a test
  block. Labels are compared by their *start and end* times, not just their start.
* **Embargo:** after each test block, also drop a few training observations that come
  right after it. Serially correlated features can still carry test-period information
  forward. The embargo applies only *after* a test block, never before.

**Combinatorial purged cross-validation (CPCV)** splits the data into N groups and uses
every choice of k groups as the test set. The test groups' out-of-sample predictions can
then be stitched into C(N-1, k-1) complete backtest paths instead of one.

.. warning::
   **Not verified against a primary source.** These follow the standard definitions in
   López de Prado (2018), *Advances in Financial Machine Learning*, Chapters 7 (purged
   k-fold, embargo) and 12 (CPCV). The book was not available when this was written, so
   the implementation is built from those definitions and checked against a brute-force
   reference, not against the book's code. It needs review against AFML before being
   relied on.

   Partial check (2026-11): a secondary reproduction of AFML Snippet 7.3 (``PurgedKFold``)
   was compared rule by rule and in a 200-case randomized test. On tie-free timestamps the
   training sets are identical. The only difference is at exactly touching labels, which
   the snippet keeps and this module purges. With zero-length labels the snippet's rules
   can place test rows in the training set; closed intervals prevent that.

Conventions (deliberate, documented):

* Label intervals are **closed**: [start, end]. Two labels overlap if they share any
  instant, including an end that equals another's start. This is the conservative
  choice: it purges slightly more.
* Purging around one contiguous test block uses that block's **span**: from the first
  test label's start to the latest test label's end. Separate (non-adjacent) test blocks
  in CPCV are purged and embargoed **separately**, never as one span from the first block
  to the last.
* The embargo is a number of observations: ``floor(embargo * n_observations)``. It is
  counted from the first observation after the test block that survives purging.
* Groups and folds are contiguous runs of observations, sized with ``numpy.array_split``
  (sizes differ by at most one).

"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CPCVResult",
    "Split",
    "combinatorial_purged_cv",
    "purged_kfold",
]

IntArray = NDArray[np.int64]


class Split(NamedTuple):
    """Row indices for one train/test split (both sorted ascending)."""

    train: IntArray
    test: IntArray


@dataclass(frozen=True)
class CPCVResult:
    """Output of :func:`combinatorial_purged_cv`.

    Attributes:
        splits: One :class:`Split` per combination of test groups, in the order of
            ``test_groups``.
        test_groups: The k group numbers used as the test set in each split.
        groups: Row indices belonging to each of the N groups.
        paths: Array of shape (n_paths, N). ``paths[p, g]`` is the index (into ``splits``)
            of the split whose out-of-sample predictions fill group ``g`` in backtest
            path ``p``. Every group appears exactly once per path, and every
            (split, test group) pair is used by exactly one path.

    """

    splits: list[Split]
    test_groups: list[tuple[int, ...]]
    groups: list[IntArray]
    paths: IntArray

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths, C(N - 1, k - 1)."""
        return int(self.paths.shape[0])


def _intervals(label_starts: ArrayLike, label_ends: ArrayLike) -> tuple[NDArray[Any], NDArray[Any]]:
    starts = np.asarray(label_starts)
    ends = np.asarray(label_ends)
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError("label_starts and label_ends must be 1-D")
    if starts.shape != ends.shape:
        raise ValueError(
            f"label_starts and label_ends differ in length: {starts.size} vs {ends.size}"
        )
    if starts.size < 2:
        raise ValueError("need at least 2 observations")
    if starts.dtype.kind in "fc" and not (
        np.all(np.isfinite(starts)) and np.all(np.isfinite(ends))
    ):
        raise ValueError("label times contain NaN or infinite values")
    if starts.dtype.kind == "M" and (np.any(np.isnat(starts)) or np.any(np.isnat(ends))):
        raise ValueError("label times contain NaT")
    if np.any(starts[1:] < starts[:-1]):
        raise ValueError("observations must be sorted by label start time")
    if np.any(ends < starts):
        raise ValueError("every label must end at or after it starts")
    return starts, ends


def _embargo_size(embargo: float, n: int) -> int:
    if not (math.isfinite(embargo) and 0.0 <= embargo < 1.0):
        raise ValueError("embargo must be a fraction in [0, 1)")
    return int(embargo * n)


def _train_mask(
    starts: NDArray[Any],
    ends: NDArray[Any],
    test_blocks: Sequence[IntArray],
    embargo_obs: int,
) -> NDArray[np.bool_]:
    """Rows usable for training, given contiguous test blocks (each purged separately)."""
    n = starts.size
    keep = np.ones(n, dtype=bool)
    for block in test_blocks:
        keep[block] = False
        block_start = starts[block[0]]
        block_end = np.max(ends[block])
        # Purge: closed-interval overlap with [block_start, block_end].
        keep &= ~((starts <= block_end) & (ends >= block_start))
        # Embargo: the next `embargo_obs` rows after the block that survived purging.
        after = np.flatnonzero((np.arange(n) > block[-1]) & (starts > block_end))
        keep[after[:embargo_obs]] = False
    return keep


def _contiguous_blocks(indices: IntArray) -> list[IntArray]:
    """Split sorted row indices into runs of consecutive integers."""
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return [b for b in np.split(indices, breaks) if b.size]


def purged_kfold(
    label_starts: ArrayLike,
    label_ends: ArrayLike,
    n_splits: int = 5,
    *,
    embargo: float = 0.0,
) -> list[Split]:
    """Return k-fold splits with purging and embargo (AFML Ch. 7, from the definitions).

    Folds are contiguous and are not shuffled. For each fold used as the test set, the
    training set is every other observation, minus those whose label overlaps the test
    fold (purging), minus the ``embargo`` share of observations right after it.

    Args:
        label_starts: Start time of each observation's label (e.g. the event time), sorted
            ascending. Numbers or ``numpy.datetime64``.
        label_ends: End time of each label (e.g. when a barrier was touched). Same type.
        n_splits: Number of folds k (at least 2).
        embargo: Share of all observations to drop after each test fold, in [0, 1).
            AFML suggests about 0.01.

    Returns:
        A list of :class:`Split` (train and test row indices), one per fold, in time order.

    >>> s = purged_kfold([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], n_splits=3)
    >>> s[1].test.tolist(), s[1].train.tolist()  # test rows 2-3; rows 1 and 4 overlap
    ([2, 3], [0, 5])

    """
    starts, ends = _intervals(label_starts, label_ends)
    n = starts.size
    if not 2 <= n_splits <= n:
        raise ValueError(f"n_splits must be between 2 and the number of observations ({n})")
    h = _embargo_size(embargo, n)
    folds = np.array_split(np.arange(n, dtype=np.int64), n_splits)
    splits = []
    for fold in folds:
        keep = _train_mask(starts, ends, [fold], h)
        splits.append(Split(train=np.flatnonzero(keep).astype(np.int64), test=fold))
    return splits


def combinatorial_purged_cv(
    label_starts: ArrayLike,
    label_ends: ArrayLike,
    n_groups: int = 6,
    n_test_groups: int = 2,
    *,
    embargo: float = 0.0,
) -> CPCVResult:
    """Return combinatorial purged CV splits and backtest paths (AFML Ch. 12, from definitions).

    The observations are cut into ``n_groups`` (N) contiguous groups. Every choice of
    ``n_test_groups`` (k) groups is a test set, giving C(N, k) splits. Adjacent test
    groups merge into one block; each separate test block is purged and embargoed on its
    own. The splits' test predictions combine into C(N - 1, k - 1) backtest paths.

    Args:
        label_starts: Start time of each label, sorted ascending.
        label_ends: End time of each label.
        n_groups: N, the number of groups (at least 2).
        n_test_groups: k, groups per test set (1 <= k < N).
        embargo: Share of all observations to drop after each test block, in [0, 1).

    >>> r = combinatorial_purged_cv(list(range(12)), list(range(12)), n_groups=6, n_test_groups=2)
    >>> len(r.splits), r.n_paths  # C(6, 2) = 15 splits, C(5, 1) = 5 paths
    (15, 5)

    """
    starts, ends = _intervals(label_starts, label_ends)
    n = starts.size
    if not 2 <= n_groups <= n:
        raise ValueError(f"n_groups must be between 2 and the number of observations ({n})")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be at least 1 and less than n_groups")
    h = _embargo_size(embargo, n)
    groups = [g.astype(np.int64) for g in np.array_split(np.arange(n), n_groups)]

    test_groups = list(itertools.combinations(range(n_groups), n_test_groups))
    splits = []
    for chosen in test_groups:
        test = np.concatenate([groups[g] for g in chosen])
        keep = _train_mask(starts, ends, _contiguous_blocks(test), h)
        splits.append(Split(train=np.flatnonzero(keep).astype(np.int64), test=test))

    # Path p takes, for each group g, the p-th split (in order) that tests g.
    n_paths = math.comb(n_groups - 1, n_test_groups - 1)
    paths = np.empty((n_paths, n_groups), dtype=np.int64)
    for g in range(n_groups):
        testing_g = [i for i, chosen in enumerate(test_groups) if g in chosen]
        paths[:, g] = testing_g
    return CPCVResult(splits=splits, test_groups=test_groups, groups=groups, paths=paths)
