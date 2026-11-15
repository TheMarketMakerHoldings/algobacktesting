# Explainer 2: Did my backtest just pick the luckiest strategy? (PBO)

Module: `algobacktesting.pbo`
Function: `probability_of_backtest_overfitting(returns, n_partitions=16)` returns a
`PBOResult`.

## The question it answers

You tried N versions of a strategy (different lookbacks, thresholds, stop losses...) and
kept the one with the best backtest. **The Probability of Backtest Overfitting (PBO)
answers: how often does that "pick the best" process choose a version that then does
*worse than the typical version* on data it hasn't seen?**

- PBO near 0: your selection process finds something real.
- PBO near 0.5: picking the winner is no better than picking at random.
- PBO above 0.5: picking the winner actively hurts. You are selecting noise.

Note that PBO judges the **research process** (your whole set of N trials), not one
strategy on its own. That's why it needs the return series of *every* version you tried,
not just the winner's.

## The idea, in words

1. **Line up the trials.** Put the N strategies' returns side by side: a table with T rows
   (days) and N columns (versions).
2. **Cut time into S equal chunks.** Use an even number, e.g. 16.
3. **Try every way of splitting the chunks in half.** One half is "training"
   (in-sample), the other half "testing" (out-of-sample). With 16 chunks there are
   C(16, 8) = **12,870** ways. Each chunk spends exactly as much time in training as in
   testing, hence "combinatorially symmetric". The chunks keep their time order inside
   each half.
4. **In each split, do what a researcher does.** Find the best version on the training
   half. Then look at where that same version ranks among all N on the testing half.
5. **Score the rank.** Relative rank w = rank / (N + 1). Dividing by N + 1 instead of N
   keeps w strictly between 0 and 1, so the next step never hits infinity. Then the
   logit is λ = ln(w / (1 − w)). Positive λ means the training winner landed in the top
   half on testing; negative means the bottom half.
6. **PBO = the share of splits where λ ≤ 0.**

The result also includes two other diagnostics from the paper:

- **Performance degradation:** a regression of the winner's testing performance on its
  training performance. A negative slope means "the better it looked in training, the
  worse it did in testing", which is the classic overfitting signature.
- **Probability of loss:** the share of splits where the training winner *lost money* in
  testing. Low PBO with a high loss probability means the selection isn't overfit, but
  the whole idea just doesn't make money.

## Worked examples (our runs, reproducible with the test suite's seeds)

**Pure noise:** 50 random strategies with no edge, 800 days, S = 8.
- Averaged over 40 datasets, PBO = **0.498**. Theory says exactly 0.5, because with no
  skill the winner's testing rank is a coin flip.
- But single datasets ranged from 0.20 to 0.89. **One PBO number from one dataset is
  noisy.** Treat it as evidence, not proof.

**One real edge among noise:** the same setup, but one strategy has a real per-day Sharpe
of about 0.3. PBO = **0.0**: the training winner was that strategy in every split, and
it never lost out of sample.

**The paper's seasonal experiment (Bailey, Borwein, López de Prado & Zhu, Section 6),
rebuilt:**
- A researcher searches 840 monthly trading rules (entry day × holding period × long or
  short) on about 4 years of daily data.
- **Random walk** (there is nothing to find): the best rule still shows an annualized
  Sharpe of 1.1–1.4. But PBO = 0.26, 0.64 and 0.43 on three seeds. The paper got 55%.
  Overfit.
- **A real monthly effect planted** (the first 5 days of each month drift up): PBO =
  0.085, 0.165 and 0.003. The paper got 13%. The process finds it.
- In every case the degradation slope was negative (−0.5 to −0.95).

## Choices the paper leaves open (and what we did)

| Gap in the paper | Our choice | Why |
| --- | --- | --- |
| Definition 2.2 says rank strictly below N/2; the estimator in §3.1 counts λ ≤ 0 | Use the estimator (λ ≤ 0) | It's what the paper actually computes; the two only differ when the winner lands exactly on the median |
| Ties in a ranking | Average rank | Neutral: doesn't favour either tied strategy |
| Two versions tie for best in training | Pick the first column | Deterministic and documented |
| Rows that don't split evenly into S chunks | Raise an error and ask you to trim | The paper requires equal chunks; silently dropping data is worse |

Two errors in the paper itself:
- It says S = 16 gives "12,780" combinations. The formula gives **12,870**.
- Step 4(c) labels the in-sample set as "the testing set". It should be the training set.

## When it misleads

- **It needs all the trials.** If you only kept the 10 versions that looked good, PBO
  can't see the 790 you threw away.
- **Choosing S is a trade-off.** Too few chunks means few splits and a noisy estimate. Too
  many means tiny chunks that can break up real time-patterns, such as a monthly
  seasonal effect.
- **One dataset gives a noisy answer.** The standard deviation was about 0.17 for pure
  noise. A PBO of 0.3 vs 0.4 is not a meaningful difference.
- **Small, odd numbers of strategies are biased upward.** The paper's estimator counts
  landing exactly on the median rank as overfit. With N odd, pure noise therefore
  averages ((N + 1)/2)/N instead of 0.5: 2/3 for N = 3 and 0.6 for N = 5. Our tests pin
  this exact value. It's negligible with dozens of strategies. Identical strategies all
  tie at the median, so they give PBO = 1.
- **Too many chunks.** S = 24 would mean 2.7 million splits and S = 40 about 137 billion,
  which used to freeze the computer. The function now stops above 1 million splits
  (S ≤ 22).
- **Mirror-image strategies.** If every long rule has a short twin, the median testing
  performance is exactly 0, so PBO equals the probability of loss. You can see this in
  the seasonal experiment.
- **Costs and execution aren't modeled.** PBO only knows the returns you feed it.
- **Strategies that never trade** have no defined Sharpe ratio in some chunks. The
  function raises an error rather than guess.

## How to say it in an interview

> "PBO measures whether a strategy-selection process generalizes. I split the history
> into S chunks and try every way of using half for selection and half for testing. In
> each split I take the in-sample winner and see where it ranks out of sample among all N
> candidates. PBO is the fraction of splits where it lands below the median. Pure noise
> gives about 0.5. In my implementation I rebuilt the paper's seasonal experiment: a
> random walk scored a high PBO even though its best rule had a Sharpe above 1, and
> planting a real effect drove PBO toward zero."

---

## Self-check questions

1. Why does PBO need the returns of all N versions you tried, not just the winner's?
2. With pure noise, what PBO do you expect, and why?
3. Why divide the rank by N + 1 instead of N?
4. Your PBO is 0.05, but the probability of loss is 0.60. What does that tell you?
5. Why must each training and testing half keep its chunks in time order, even though
   the order doesn't change a Sharpe ratio?

### Answers

1. PBO measures where the training winner ranks *among all candidates* on testing data.
   Without the losers there is nothing to rank against, and you can't reconstruct which
   version would have won in each training half.
2. **0.5.** With no skill, the training winner's testing rank is equally likely to be
   anything from 1 to N, so it lands in the bottom half half the time.
3. A rank of N divided by N would give w = 1, and ln(w/(1 − w)) would be infinite.
   Dividing by N + 1 keeps w strictly between 0 and 1, so every logit is a finite number.
4. The selection process is **not** overfitting: it reliably picks a version that beats
   the typical one. But that version still loses money 60% of the time out of sample. The
   whole strategy family is weak. Low PBO is not the same as profitable.
5. Some performance measures depend on order, like maximum drawdown or anything
   path-dependent. Keeping time order means any metric you plug in sees realistic return
   sequences. It also matches the paper's algorithm exactly.
