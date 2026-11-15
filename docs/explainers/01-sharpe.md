# Explainer 1: Is this Sharpe ratio real?

Module: `algobacktesting.sharpe`
Functions: `probabilistic_sharpe_ratio` (PSR), `min_track_record_length` (MinTRL),
`expected_max_sharpe` (SR0), `deflated_sharpe_ratio` (DSR), `return_stats`

## The questions these answer

| Function | Question |
| --- | --- |
| PSR | "Given this track record, how likely is it that the *true* Sharpe ratio beats X?" |
| MinTRL | "How long does the track record have to be before I can trust it?" |
| SR0 | "If none of my strategies had any skill, how good would the best one look by luck?" |
| DSR | "I tried many strategies and kept the best. Is the winner real?" |

## The idea, in words

**A Sharpe ratio from data is an estimate, and estimates have error.** With 24 monthly
returns you have seen very little. The same strategy over the next 24 months could easily
show a very different number. PSR measures how far the observed Sharpe is above the bar,
in units of that estimation error, and turns the distance into a probability using the
Normal curve.

**Skew and fat tails make the error bigger.** A strategy that earns small, steady gains and
occasionally crashes (negative skew, fat tails) can show a smooth, high Sharpe for a long
time. A short record may simply not contain the crash yet. So for the same Sharpe ratio,
PSR is lower when returns are negatively skewed or fat-tailed. The formula's denominator
grows with `−skew × SR` and `(kurtosis − 1)/4 × SR²`. Note that *raw* kurtosis is used,
where a Normal distribution has 3.

**MinTRL just runs PSR backwards.** It asks for which track-record length PSR reaches 95%,
and solves for that length.

**DSR fixes selection bias.** Test 100 strategies that are all pure noise, and the best one
still shows a positive Sharpe, because you picked the luckiest. Extreme value theory gives
the expected best of N lucky draws: SR0. DSR is simply PSR with SR0 as the bar instead of
zero. The more strategies you tried, the higher SR0, and the harder it is to pass.

## Worked examples (numbers from the papers, reproduced by our tests)

**DSR, Bailey & López de Prado (2014):** a researcher tries 100 versions of a Treasury
auction strategy and finds one with an annualized Sharpe of 2.5 over 5 years of daily data
(1,250 days). Its returns have skew −3 and kurtosis 10. The Sharpe ratios of the 100 trials
have an annualized variance of 0.5.

- Convert to daily units: Sharpe = 2.5/√250 = 0.158. Trial variance = 0.5/250 = 0.002.
- The luck bar: SR0 = 0.1132 per day, which is about **1.79 annualized**. That's what the
  best of 100 skill-less strategies would show on average.
- DSR = **0.9004**: only a 90% chance the strategy is real. That's below 95%, so reject.
- If only 46 versions had been tried, DSR = 0.9505 and it passes. With Normal returns
  (no skew, no fat tails), it would pass with up to 88 tries.

**PSR, Bailey & López de Prado (2012):** a hedge fund shows 24 monthly returns with an
annualized Sharpe of 1.585 (0.458 per month).

- Assume Normal returns: PSR = **98.2%**. Looks great.
- Use its real skew (−2.45) and kurtosis (10.16): PSR = **91.3%**. Not proven at 95%.
- With 36 months instead of 24: PSR = **95.3%**. Time can earn back the trust.

**MinTRL, same paper:** to show an annualized Sharpe of 2 beats 1 at 95%, you need
**2.73 years** of daily data, 3.24 years of monthly data, or **4.99 years** of monthly
data with typical hedge-fund skew and kurtosis.

## What we verified ourselves

- **Every number above matches the papers exactly**, to the decimals they print.
  - One detail: the paper's DSR of 0.9004 uses the unrounded SR0. Plugging in the printed
    0.1132 gives 0.9003.
  - Likewise, the hedge fund's monthly Sharpe is 0.4576 before the paper rounds it to
    0.458.
- **Simulation check on false positives.** When a strategy has no edge, PSR should pass the
  95% bar about 5% of the time.
  - It does. In the test suite's simulations (4,000 runs each, fixed seeds) it passes 4.95%
    of the time with Normal returns, and 4.7% with crash-prone returns modeled on the
    paper's hedge fund.
  - If you *ignore* skew and kurtosis on those crash-prone returns, the false-positive rate
    is 15.05%, three times too high. That's the whole reason PSR exists.
- **The luck formula (SR0) is a slight overestimate.** Exact integration shows it is too
  high by 0.036 standard deviations at 10 trials and 0.014 at 1,000.
  - That's within the paper's stated bound of 0.05, and it errs on the safe side.
  - (The paper also claims the error falls to 0.006 by 1,000 trials. Exact math says 0.014.
    This doesn't change the formula, just one sentence of the paper's commentary.)

## When it misleads

- **N must count *independent* trials.** Twenty near-identical parameter tweaks are closer
  to one trial than twenty. Under-counting N makes DSR look better than it is, which is the
  easiest way to fool yourself (or an investor) with this tool.
- **Units must match.** Every Sharpe ratio and the trial variance must be per period
  (daily, weekly, etc.). Mixing annual and daily numbers is the most common mistake.
- **Returns are assumed independent over time.** Smoothed or illiquid returns (prices that
  update slowly) inflate the Sharpe ratio, and PSR does not correct for that.
- **It's an approximation for large samples.** Below about 30 observations, and when skew
  and kurtosis are estimated from short data, treat the result as rough.
- **"Real" only means "above zero" (or above your benchmark).** DSR does not say the edge
  survives trading costs or is big enough to be worth trading.

## How to say it in an interview

> "A backtest Sharpe is an estimate with error, and that error is bigger when returns are
> negatively skewed or fat-tailed. PSR turns the Sharpe into a probability of beating a
> benchmark, given the track record's length and shape. DSR goes further: if you tried N
> strategies, the best one looks good by luck, so DSR raises the bar to the Sharpe you'd
> expect from the best of N skill-less strategies. I implemented both and reproduced every
> number in Bailey and López de Prado's papers. I also simulated that ignoring skew and
> kurtosis triples the false-positive rate."

---

## Self-check questions

1. Two strategies both have a monthly Sharpe of 0.4 over 3 years. One has Normal returns;
   the other has skew −2. Which has the higher PSR, and why?
2. You tested 200 parameter combinations, but they fall into about 10 clusters of
   near-duplicates. What N should you use, and what goes wrong if you claim N = 10 when
   the honest number is 50?
3. A daily strategy has an annualized Sharpe of 2.0, and the trial Sharpe ratios have an
   annualized variance of 0.5. What exact numbers do you pass to `deflated_sharpe_ratio`
   for `sharpe` and `sharpe_variance` (assume 252 days a year)?
4. Why does DSR with N = 1 give exactly the same answer as PSR against zero?
5. MinTRL says 688 daily observations. Roughly how many years is that? Why would a monthly
   strategy with the same annualized Sharpe need *more* years?

### Answers

1. **The Normal one.** Negative skew enlarges the Sharpe ratio's estimation error (the
   `−skew × SR` term in the denominator grows), so the same Sharpe is less convincing.
   Crash-prone strategies can look smooth until the crash.
2. **Use the number of independent clusters, about 10.** If the honest number is 50 and
   you say 10, SR0 is set too low. DSR then overstates the chance the strategy is real.
   Counting too many trials is the safe direction; counting too few is how people fool
   themselves.
3. `sharpe = 2.0 / sqrt(252)` (about 0.126) and `sharpe_variance = 0.5 / 252` (about
   0.00198). Both must be daily. A Sharpe ratio scales with √(periods), its variance with
   periods.
4. With a single trial there is no "picking the best", so the luck bar SR0 is 0, and DSR is
   PSR measured against 0.
5. **About 2.73 years** (688 / 252). Converted to years, MinTRL is roughly
   `(Z / gap)² × (1 + SR_per_period² / 2) + 1 / periods_per_year`, using annualized Sharpe
   numbers for the gap. Monthly data has a bigger per-period Sharpe, so the `SR²` term is
   larger, and the formula's extra "+1 observation" is a whole month instead of a day.
   Both push the requirement up: 3.24 years in the paper's example.
