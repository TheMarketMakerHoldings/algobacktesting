"""Tests for algobacktesting.cv (purged k-fold, embargo, CPCV).

There is no source PDF for these methods in this project (they come from López de Prado's
book, AFML Chapters 7 and 12), so correctness is established three ways:

1. Invariants checked by a slow, separately written brute-force reference on thousands of
   random, irregular datasets (hypothesis): no leakage, embargo respected, nothing removed
   without a reason, train and test disjoint.
2. Hand-worked examples for each rule: purging by label END time, embargo only AFTER a test
   block, separate purging of non-adjacent CPCV test blocks.
3. CPCV combinatorics: C(N, k) splits, C(N-1, k-1) paths, every group once per path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algobacktesting.cv import combinatorial_purged_cv, purged_kfold

# ---------------------------------------------------------------------------------------
# Brute-force reference: plain Python loops, written independently of the module
# ---------------------------------------------------------------------------------------


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Closed intervals [a_start, a_end] and [b_start, b_end] share at least one instant."""
    return not (a_end < b_start or b_end < a_start)


def _blocks(test: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for i in sorted(test):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _reference_train(starts: list[float], ends: list[float], test: list[int], h: int) -> set[int]:
    n = len(starts)
    excluded = set(test)
    for block in _blocks(test):
        span_start, span_end = starts[block[0]], max(ends[j] for j in block)
        purged = {i for i in range(n) if _overlaps(starts[i], ends[i], span_start, span_end)}
        excluded |= purged
        later_survivors = [i for i in range(block[-1] + 1, n) if starts[i] > span_end]
        excluded |= set(later_survivors[:h])
    return set(range(n)) - excluded


@st.composite
def labelled_data(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    n = draw(st.integers(4, 60))
    gaps = draw(st.lists(st.integers(0, 3), min_size=n, max_size=n))  # repeats allowed
    starts = np.cumsum(gaps).astype(float).tolist()
    lengths = draw(st.lists(st.integers(0, 12), min_size=n, max_size=n))
    ends = [s + d for s, d in zip(starts, lengths, strict=True)]
    return starts, ends


# ---------------------------------------------------------------------------------------
# 1. Invariants against the reference
# ---------------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(data=labelled_data(), k=st.integers(2, 6), embargo=st.sampled_from([0.0, 0.05, 0.2]))
def test_purged_kfold_matches_reference(
    data: tuple[list[float], list[float]], k: int, embargo: float
) -> None:
    starts, ends = data
    n = len(starts)
    k = min(k, n)
    splits = purged_kfold(starts, ends, k, embargo=embargo)
    assert np.array_equal(np.concatenate([s.test for s in splits]), np.arange(n))
    for split in splits:
        test = split.test.tolist()
        train = set(split.train.tolist())
        assert train == _reference_train(starts, ends, test, int(embargo * n))
        assert not train & set(test)
        for i in train:  # the guarantee that matters: no training label touches a test label
            for j in test:
                assert not _overlaps(starts[i], ends[i], starts[j], ends[j])


@settings(max_examples=200, deadline=None)
@given(
    data=labelled_data(),
    n_groups=st.integers(3, 7),
    k=st.integers(1, 3),
    embargo=st.sampled_from([0.0, 0.1]),
)
def test_cpcv_matches_reference(
    data: tuple[list[float], list[float]], n_groups: int, k: int, embargo: float
) -> None:
    starts, ends = data
    n = len(starts)
    n_groups = min(n_groups, n)
    k = min(k, n_groups - 1)
    result = combinatorial_purged_cv(starts, ends, n_groups, k, embargo=embargo)
    for split, chosen in zip(result.splits, result.test_groups, strict=True):
        expected_test = sorted(i for g in chosen for i in result.groups[g].tolist())
        assert split.test.tolist() == expected_test
        train = set(split.train.tolist())
        assert train == _reference_train(starts, ends, expected_test, int(embargo * n))
        for i in train:
            for j in expected_test:
                assert not _overlaps(starts[i], ends[i], starts[j], ends[j])


# ---------------------------------------------------------------------------------------
# 2. Hand-worked rules
# ---------------------------------------------------------------------------------------


def test_no_overlap_and_no_embargo_is_plain_kfold() -> None:
    starts = np.arange(20.0)
    splits = purged_kfold(starts, starts, 4)  # zero-length labels at distinct times
    for split in splits:
        assert sorted(split.train.tolist() + split.test.tolist()) == list(range(20))


def test_purging_uses_label_end_not_just_start() -> None:
    # Row 3 starts before the test fold (rows 5-9) but its label runs into it: purged.
    # Row 2 starts earlier and ends before the fold: kept. Only the end time tells them apart.
    starts = np.arange(10.0)
    ends = starts.copy()
    ends[3] = 6.0  # long label: [3, 6] overlaps test rows starting at 5
    ends[2] = 4.0  # [2, 4] ends before 5
    train = purged_kfold(starts, ends, 2)[1].train.tolist()
    assert 3 not in train
    assert 2 in train


def test_closed_intervals_purge_touching_labels() -> None:
    # Row 4's label ends exactly when the test fold starts (time 5): shared instant, purged.
    starts = np.arange(10.0)
    ends = starts.copy()
    ends[4] = 5.0
    assert 4 not in purged_kfold(starts, ends, 2)[1].train.tolist()


def test_embargo_only_after_the_test_block() -> None:
    starts = np.arange(100.0)  # zero-length labels: purging removes nothing
    splits = purged_kfold(starts, starts, 5, embargo=0.03)  # h = 3 rows
    middle = splits[2]  # test rows 40-59
    train = set(middle.train.tolist())
    assert {37, 38, 39} <= train  # right before the test block: kept
    assert not {60, 61, 62} & train  # right after: embargoed
    assert 63 in train


def test_embargo_counts_from_first_unpurged_row() -> None:
    starts = np.arange(30.0)
    ends = starts.copy()
    ends[19] = 22.0  # middle fold = rows 10-19; its latest label ends at 22
    split = purged_kfold(starts, ends, 3, embargo=0.1)[1]  # h = floor(0.1 * 30) = 3
    assert split.test.tolist() == list(range(10, 20))
    train = set(split.train.tolist())
    assert not {20, 21, 22} & train  # purged: they start at or before 22
    assert not {23, 24, 25} & train  # embargo: the next 3 rows that survived purging
    assert 26 in train


def test_cpcv_purges_each_test_block_separately() -> None:
    # Test groups 1 and 3 are not adjacent. Purging the span from the first block to the
    # last would wipe out group 2 entirely; purging each block separately keeps its middle.
    starts = np.arange(50.0)
    ends = starts + 1.0  # each label touches the next row
    result = combinatorial_purged_cv(starts, ends, n_groups=5, n_test_groups=2)
    split = result.splits[result.test_groups.index((1, 3))]
    train = set(split.train.tolist())
    assert {21, 22, 23, 24, 25, 26, 27, 28} <= train  # group 2's interior survives
    # Only the rows whose labels touch a test block are purged: 9 and 20 around block
    # 10-19, 29 and 40 around block 30-39.
    assert not {9, 20, 29, 40} & train
    assert {8, 41} <= train


def test_cpcv_adjacent_test_groups_form_one_block() -> None:
    starts = np.arange(60.0)
    result = combinatorial_purged_cv(starts, starts, n_groups=6, n_test_groups=2, embargo=0.05)
    split = result.splits[result.test_groups.index((2, 3))]  # rows 20-39, one block
    train = set(split.train.tolist())
    assert {40, 41, 42} & train == set()  # h = 3 after the merged block only
    assert 30 not in train  # inside the merged test block, not an embargo gap
    assert 19 in train  # embargo never applies before a block


def test_datetime64_labels() -> None:
    starts = np.arange("2026-01-01", "2026-01-21", dtype="datetime64[D]")
    ends = starts + np.timedelta64(2, "D")
    splits = purged_kfold(starts, ends, 4)
    assert all(
        np.array_equal(a.test, b)
        for a, b in zip(splits, np.array_split(np.arange(20), 4), strict=True)
    )
    assert 4 not in splits[1].train.tolist()  # label [Jan 5, Jan 7] overlaps test starting Jan 6


# ---------------------------------------------------------------------------------------
# 3. CPCV combinatorics
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n_groups", "k"), [(2, 1), (5, 1), (6, 2), (6, 3), (8, 2), (10, 4)])
def test_cpcv_counts_and_paths(n_groups: int, k: int) -> None:
    starts = np.arange(float(n_groups * 5))
    result = combinatorial_purged_cv(starts, starts, n_groups, k)
    assert len(result.splits) == math.comb(n_groups, k)
    assert (
        result.n_paths == math.comb(n_groups - 1, k - 1) == math.comb(n_groups, k) * k // n_groups
    )
    used: set[tuple[int, int]] = set()
    for path in result.paths:
        assert len(path) == n_groups
        for group, split_index in enumerate(path.tolist()):
            assert group in result.test_groups[split_index]  # that split really tested it
            used.add((split_index, group))
    expected_pairs = {(i, g) for i, chosen in enumerate(result.test_groups) for g in chosen}
    assert used == expected_pairs  # each (split, test group) used by exactly one path
    assert len(used) == result.n_paths * n_groups


def test_groups_partition_rows_in_order() -> None:
    result = combinatorial_purged_cv(np.arange(23.0), np.arange(23.0), n_groups=5, n_test_groups=2)
    assert np.array_equal(np.concatenate(result.groups), np.arange(23))
    assert sorted(len(g) for g in result.groups) == [4, 4, 5, 5, 5]


# ---------------------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("starts", "ends", "message"),
    [
        ([0.0, 2.0, 1.0], [0.0, 2.0, 1.0], "sorted"),
        ([0.0, 1.0, 2.0], [0.0, 0.5, 2.0], "end at or after"),
        ([0.0, 1.0], [0.0, 1.0, 2.0], "differ in length"),
        ([0.0, float("nan"), 2.0], [0.0, 1.0, 2.0], "NaN"),
        ([0.0], [0.0], "at least 2"),
        ([[0.0, 1.0]], [[0.0, 1.0]], "1-D"),
    ],
)
def test_rejects_bad_labels(starts: list[float], ends: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        purged_kfold(starts, ends, 2)


def test_rejects_nat() -> None:
    starts = np.array(["2026-01-01", "NaT"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="NaT"):
        purged_kfold(starts, starts, 2)


@pytest.mark.parametrize("embargo", [-0.1, 1.0, float("nan")])
def test_rejects_bad_embargo(embargo: float) -> None:
    with pytest.raises(ValueError, match="embargo"):
        purged_kfold(np.arange(10.0), np.arange(10.0), 2, embargo=embargo)


@pytest.mark.parametrize("k", [1, 11])
def test_rejects_bad_fold_count(k: int) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        purged_kfold(np.arange(10.0), np.arange(10.0), k)


@pytest.mark.parametrize(("n_groups", "k"), [(1, 1), (11, 1), (5, 0), (5, 5)])
def test_rejects_bad_cpcv_settings(n_groups: int, k: int) -> None:
    with pytest.raises(ValueError, match=r"n_groups|n_test_groups"):
        combinatorial_purged_cv(np.arange(10.0), np.arange(10.0), n_groups, k)


# ---------------------------------------------------------------------------------------
# 5. Agreement with AFML Snippet 7.3's rules (from a secondary reproduction of the snippet)
#    Book: keep pre-test rows if label end <= test start; keep post-test rows from the first
#    row whose start >= the latest test label end, after skipping int(n * pct) rows.
#    Ours is identical except that exactly-touching labels are purged (closed intervals).
#    With continuous, tie-free timestamps no label touches exactly, so both must agree.
# ---------------------------------------------------------------------------------------


def _afml_snippet_7_3(
    starts: np.ndarray, ends: np.ndarray, n_splits: int, pct_embargo: float
) -> list[list[int]]:
    n = starts.size
    h = int(n * pct_embargo)
    trains = []
    for fold in np.array_split(np.arange(n), n_splits):
        t0 = starts[fold[0]]
        before = [i for i in range(n) if ends[i] <= t0]
        first_after = int(np.searchsorted(starts, ends[fold].max(), side="left"))
        after = list(range(first_after + h, n)) if first_after < n else []
        trains.append(sorted(set(before) | set(after)))
    return trains


@settings(max_examples=200, deadline=None)
@given(
    seed=st.integers(0, 100_000),
    n=st.integers(6, 80),
    k=st.integers(2, 6),
    embargo=st.sampled_from([0.0, 0.03, 0.1]),
)
def test_matches_afml_snippet_7_3_without_exact_ties(
    seed: int, n: int, k: int, embargo: float
) -> None:
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, 100, n))
    ends = starts + rng.exponential(3.0, n)  # continuous: no label touches another exactly
    k = min(k, n)
    ours = [s.train.tolist() for s in purged_kfold(starts, ends, k, embargo=embargo)]
    assert ours == _afml_snippet_7_3(starts, ends, k, embargo)


def test_afml_rules_leak_test_rows_with_zero_length_labels() -> None:
    # Zero-length labels (end == start, e.g. a barrier hit on the entry bar) plus row 4's
    # label ending exactly at the test start (time 5). Under the AFML rules:
    #   * "keep if label end <= test start" admits row 4 (touching) AND test row 5 itself;
    #   * "first row with start >= latest test end (9)" lands on test row 9 itself.
    # So AFML's training set contains test rows 5 and 9: leakage. Ours never overlaps.
    starts = np.arange(10.0)
    ends = starts.copy()
    ends[4] = 5.0
    test = purged_kfold(starts, ends, 2)[1].test.tolist()
    afml = _afml_snippet_7_3(starts, ends, 2, 0.0)[1]
    ours = purged_kfold(starts, ends, 2)[1].train.tolist()
    assert test == [5, 6, 7, 8, 9]
    assert afml == [0, 1, 2, 3, 4, 5, 9]
    assert set(afml) & set(test) == {5, 9}  # test rows inside AFML's training set
    assert ours == [0, 1, 2, 3]
    assert not set(ours) & set(test)
    assert sorted(set(afml) - {4, 5, 9}) == ours  # otherwise identical
