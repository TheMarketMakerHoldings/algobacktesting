# Explainer 3: Cross-validation that doesn't peek at the future

Module: `algobacktesting.cv`
Functions: `purged_kfold(label_starts, label_ends, n_splits=5, *, embargo=0.0)`,
`combinatorial_purged_cv(label_starts, label_ends, n_groups=6, n_test_groups=2, *,
embargo=0.0)`

> ⚠️ **Review flag.** These come from López de Prado's book *Advances in Financial
> Machine Learning* (AFML), Chapters 7 and 12. The book wasn't available while building
> them, so they're built from the standard definitions and checked against a brute-force
> reference, **not against the book**. Before v0.1.0 ships, compare them with AFML.

## The question they answer

"If I train a model on part of my history and test it on another part, is the test score
honest?"

Ordinary k-fold cross-validation assumes rows are independent. In trading they aren't:

1. **Labels span time.** "Did this trade hit its profit target within 5 days?" is a label
   that runs from day *t* to day *t + 5*. If a training row's 5-day window overlaps the
   test period, the model trained on information *from* the test period. That's leakage,
   and the score comes out too good.
2. **Features are correlated over time.** Rows right after a test period can still "remember"
   it through moving averages and similar features.

## The idea, in words

**Purging.** For each test block, remove every training row whose label window
[start, end] overlaps the test block's window. This compares the **end** time too, not
just the start. A row that starts before the test block but whose label runs into it is
the classic leak.

**Embargo.** After each test block, also drop the next few training rows (a set share of
all rows, e.g. 1%). This happens only *after* a test block. Information flows forward in
time, so rows *before* the block are safe once purged.

**Combinatorial purged CV (CPCV).** Normal k-fold gives you one backtest path. CPCV:

- Cut the data into N groups and use *every* choice of k groups as the test set:
  C(N, k) splits.
- Each group gets tested in several splits. Stitching those test results together gives
  C(N−1, k−1) complete, separate backtest paths.
- That gives a distribution of backtest performance instead of one lucky (or unlucky)
  history.
- Example: N = 6, k = 2 gives 15 splits and 5 paths.

**CPCV subtlety we handle:** if the test groups are, say, 2 and 4 (not adjacent), each is
purged *separately*. Purging the whole span from group 2 to group 4 would throw away
group 3 for no reason. Adjacent test groups (2 and 3) merge into one block.

## Worked examples (from the tests)

- **Purging uses the end time.** Rows at times 0–9, test fold = rows 5–9.
  - Row 3's label runs [3, 6]: it reaches into the test window, so it's purged.
  - Row 2's label runs [2, 4]: it ends before the test starts, so it's kept.
  - Looking only at start times, you couldn't tell these apart.
- **Touching counts as overlap.** A label ending exactly at the test's first timestamp is
  purged. This is conservative: we treat label windows as closed intervals.
- **Embargo is only after.** With 100 rows, 5 folds and a 3% embargo (3 rows), testing
  rows 40–59 keeps rows 37–39 but drops rows 60–62.
- **CPCV keeps the middle.** With 50 rows in 5 groups, testing groups 1 and 3 purges only
  the edge rows 9, 20, 29 and 40. Group 2's interior (rows 21–28) stays in training.

## How we verified it (no source paper here)

- **A slow, separately written brute-force reference** (plain Python loops) recomputes
  every training set. It matched exactly on 500 random datasets with irregular
  timestamps, repeated timestamps, varied label lengths, fold counts and embargoes.
- **The guarantee that matters, checked directly:** no training label overlaps any test
  label, in every split of every random dataset.
- **CPCV bookkeeping:** C(N, k) splits and C(N−1, k−1) paths, every group exactly once
  per path, every test result used by exactly one path.

## Choices we made (review these against AFML)

| Question | Our choice |
| --- | --- |
| Do labels that merely touch (one ends when the other starts) overlap? | Yes. Closed intervals, the conservative choice |
| What window is purged around one contiguous test block? | From the first test label's start to the latest test label's end |
| How big is the embargo? | `floor(embargo × number of rows)` rows, counted from the first row after the block that survived purging |
| How are folds and groups sized? | Contiguous, sizes differ by at most one row, never shuffled |
| How are paths assembled? | Path p takes, for each group, the p-th split (in order) that tested that group |

## Review checklist against AFML

The table compares ours with AFML's purged k-fold code (Snippet 7.3) as reproduced in a
**secondary source** (a published reproduction of the snippet, not the book itself).
A randomized test (`test_matches_afml_snippet_7_3_without_exact_ties`) confirms the
training sets are **identical on 200 random datasets** whenever no label touches another
exactly. Still to do: confirm against the book itself.

| Point | AFML 7.3 (secondary source) | Ours | Effect of the difference |
| --- | --- | --- | --- |
| Training rows before the test fold | keep if label end ≤ test start | keep if label end **<** test start | We also drop a label that ends *exactly* at the test start (at most a row or two) |
| Training rows after the test fold | keep if start ≥ the latest test label end, after the embargo | keep if start **>** that end, after the embargo | Same, at the equality boundary only |
| Embargo size | `int(n × pct_embargo)` rows | `floor(n × embargo)` rows | Same |
| Embargo position | right after the test fold's latest label end | same, after purging | Same |
| Folds | `np.array_split`, no shuffling | same | Same |
| CPCV paths | C(N−1, k−1) = k·C(N, k)/N | same | Same. As far as recalled, the book gives no code for CPCV, so check the path assembly against its Chapter 12 description |

**A leak in the book's rules, and why we keep closed intervals.** With **zero-length
labels** (a label that ends on its own start, e.g. a barrier hit on the entry bar of
daily data), the snippet's rules put *test rows into the training set*:
- the first test row passes "label end ≤ test start";
- "first row starting at or after the latest test end" lands on the last test row.

In our test case, AFML trains on test rows 5 and 9. Our closed intervals make that
impossible: train and test are proven disjoint. So the stricter choice is kept on
purpose.

## When it misleads

- **Wrong label end times = wrong purging.** The function trusts the ends you give it.
  If labels actually look 10 days ahead but you pass 5, leakage remains.
- **Embargo size is a judgment call.** Too small leaves leakage through slow features;
  too large wastes data. About 1% is a common starting point.
- **CPCV's paths aren't independent.** They share training data, so treat their spread as
  a range of outcomes, not as independent samples.
- **Not yet reviewed against the book.** See the flag at the top.

## How to say it in an interview

> "In finance, labels span time, so standard k-fold leaks: a training label can overlap
> the test period. Purging removes training rows whose label interval overlaps the test
> block, comparing start *and* end times. An embargo drops rows just after the test block
> to stop serial correlation leaking forward. CPCV tests every combination of k out of N
> groups, which turns one backtest into C(N−1, k−1) paths, so you see a distribution of
> outcomes. I verified my implementation against a brute-force reference on hundreds of
> random irregular datasets."

---

## Self-check questions

1. Why is ordinary k-fold cross-validation dishonest for trading labels like "hit the
   target within 5 days"?
2. Why must purging look at a label's *end* time, not just its start?
3. Why does the embargo apply only *after* a test block?
4. With N = 6 groups and k = 2 test groups, how many splits and how many backtest paths?
5. Test groups are 1 and 3. Why would purging the span from group 1 to group 3 be wrong?

### Answers

1. A training row's 5-day label window can overlap the test period, so the model learns
   from outcomes inside the test period, and the test score is inflated.
2. A row can *start* safely before the test block but *end* inside it. Its label uses
   test-period prices, so it leaks. Only the end time reveals that.
3. Information flows forward in time. Features computed after the test block can carry
   test-period information forward. Rows before the block can't see the future, and
   purging already removed any whose labels reach into it.
4. C(6, 2) = **15 splits** and C(5, 1) = **5 paths** (equivalently 15 × 2 / 6).
5. Group 2 sits between them and doesn't overlap either test block except at its edges.
   Purging the whole span would discard all of group 2's data for no reason. Each block
   should be purged on its own.
