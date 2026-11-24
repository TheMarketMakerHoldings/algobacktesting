# Explainer 4: The website's "profitable or lucky" number

Page: `docs/index.html`
Tests: `tests/test_site.py` (the page's math against the library),
`tests/test_site_import.py` (reading trade exports) and
`tests/test_luck_calibration.py` (how often it gives a false alarm)

## The question it answers

"My trading results are positive. How likely is it that I have a real edge, and not a lucky
streak?"

It answers from the trader's own trade history (an exported CSV file or pasted results),
or, in the quick check, from four numbers they already know: win rate, average win ÷
average loss, number of trades, and how many setups they tried before this one.

## The idea, in words

**Step 1: turn the results into a signal-to-noise number.**
In the quick check, every trade either wins R (the reward-to-risk ratio) or loses 1.
- Average result per trade = win rate × R − (1 − win rate).
- Swing per trade (standard deviation) = √(win rate × (1 − win rate)) × (R + 1).
- Average ÷ swing is the per-trade Sharpe ratio. The page never shows that word, but this is
  the number everything else is built on.

**Step 2: ask how surprising that is for someone with no edge.**
With more trades, luck averages out. The Probabilistic Sharpe Ratio turns "signal-to-noise
× √(number of trades)" into a probability. It also corrects for lopsided results: a low win
rate with big wins behaves differently from a high win rate with small wins.

**Step 3: raise the bar for every setup tried.**
If you tried 5 setups and kept the best, the best one looks good even if none works. The
page computes what the best of N no-edge setups would show on average, then measures your
results against *that* instead of zero. This is the Deflated Sharpe Ratio.

**Step 4: say how many trades would settle it.**
The same formula solved for the number of trades gives "trades needed to prove it": the
count at which the answer reaches 95%.

## Worked example (the quick check's default values)

45% win rate, 1.5 reward-to-risk, 120 trades, 5 setups tried.

- Average per trade: 0.45 × 1.5 − 0.55 = **+0.125 R**. It looks profitable.
- Swing per trade: √(0.45 × 0.55) × 2.5 = 1.244 R. Signal-to-noise: 0.1005.
- If this were the only setup ever tried: **87%** chance the edge is real.
- Best of 5 no-edge setups would average **+0.136 R** per trade. That's more than this
  trader's +0.125 R.
- After that adjustment: **46%**. The verdict is "could easily be luck".
- Trades needed to reach 95%: **789**. The trader has 120.

## The two assumptions (both stated on the page)

1. **Quick check: every trade risks the same amount and ends at its target or its stop.**
   That is exact for fixed-risk traders. If wins and losses vary in size, the real swing is
   larger than the model's, so the quick check is too optimistic. That's why the page offers
   "paste my trades", which uses the real average, swing, skew and kurtosis.
2. **Luck adjustment: the setups you tried were independent, tested on a similar number of
   trades, and the trial spread is what pure luck would produce.** The Deflated Sharpe
   Ratio needs the variance of the trial Sharpe ratios. A trader doesn't know it, so the
   page uses 1 ÷ (trades − 1), the variance of a Sharpe ratio measured on that many trades
   when the true value is zero.

## Reading a trade file

The import reads the file in the browser; nothing is uploaded. It:

- **finds the profit/loss column by its name**, preferring one marked net (after fees),
  and never a running total, a percentage, a daily or open P&L, or a price;
- **subtracts fees** when a fee or commission column exists and the P&L column is not
  marked net, because a breakeven trade still pays commission. The user can switch it off;
- **combines fills into trades** when the file has a position ID (Quantower), and pairs
  buys and sells first in, first out when it only lists fills (thinkorswim);
- **separates breakevens**: a trade that was flat *before* fees is a breakeven, reported
  apart from wins and losses, together with what the breakevens cost in fees. The win rate
  is shown both with and without them;
- **filters to one strategy**, so a file with several setups is judged one setup at a time.

Only Quantower's column names come from a published list. The TopstepX, NinjaTrader and
thinkorswim samples in `tests/exports/` are modelled on secondhand descriptions, so every
choice the importer makes is shown on the page and can be changed.

## How we know it behaves

- **The page's math equals the library's.** A test runs the page's own code and compares it
  with the Python library on 401 quick-check cases and 120 pasted-trade cases.
- **"Trades needed" is exact.** At that number of trades the library returns exactly 95%.
- **False alarms were measured.** We simulated traders with no edge at all (20,000 per
  case, 120 trades each) and counted how often the page would call the edge proven:

| Win rate, reward-to-risk | 1 setup | 5 setups | 20 setups | 100 setups |
| --- | --- | --- | --- | --- |
| 40%, 1.5 | 4.2% | 1.1% | 0.3% | 0.1% |
| 50%, 1.0 | 4.3% | 1.0% | 0.3% | 0.2% |
| 25%, 3.0 | 5.9% | 0.8% | 0.2% | 0.0% |

  With one setup it's close to the intended 5%. With several setups it's far below 5%, so
  the check is **stricter than advertised**. The reason: it asks you to beat the *average*
  best-of-N lucky result by a 95% margin. The best-of-N result varies less than a single
  result does, so that margin is hard to clear by luck.

## When it misleads

- **It judges the past, not the future.** A real edge can stop working when the market
  changes.
- **Under-counting setups flatters you.** The number of setups tried is the input people
  are most tempted to shade.
- **Trades must be independent.** Ten entries on the same move are closer to one trade
  than ten.
- **Fewer than about 30 trades is rough.** The method is an approximation that improves
  with sample size.
- **Costs must already be in the results.** If the pasted trades ignore commissions and
  slippage, the answer does too.
- **A low win rate with large wins runs slightly hot** with one setup: 5.9% false alarms
  instead of 5% in the table above.

## How to say it in an interview

> "The site turns a trader's win rate and reward-to-risk into a per-trade Sharpe ratio,
> then reports a Deflated Sharpe Ratio: the probability the edge is real after allowing for
> every setup they tried. Retail traders can't supply the variance of their trial results,
> so I set it to its value under the null, one over n minus one. I then checked that choice
> by simulating traders with no edge: the false-positive rate is about 5% with one setup and
> about 1% or less with five or more, so it errs on the cautious side."

---

## Self-check questions

1. A trader has a 60% win rate at 1.0 reward-to-risk. What is their average result per
   trade, in R?
2. Why does the same record score 87% with one setup tried but 46% with five?
3. Why does the quick check assume each trade wins R or loses 1, and when is that wrong?
4. The page sets the trial variance to 1 ÷ (trades − 1). Where does that come from?
5. With five setups, a no-edge trader is called "proven" about 1% of the time, not 5%. Is
   that a flaw?

### Answers

1. 0.6 × 1.0 − 0.4 = **+0.20 R** per trade.
2. With five setups, the best of five lucky ones would already show about +0.136 R per
   trade. The page measures the trader's +0.125 R against that bar instead of against
   zero, and +0.125 R doesn't clear it.
3. It lets a trader answer with two numbers they know. It is exact for traders who risk a
   fixed amount and exit at a fixed target or stop. If results vary in size, the true swing
   is larger, so the quick check overstates the chance. Pasting real trades fixes that.
4. It's the variance of a Sharpe ratio estimated from that many trades when the true Sharpe
   ratio is zero. In other words: how much the setups' results would differ if none of them
   had an edge.
5. No. It means the check is cautious. It's harder to be wrongly called proven, at the
   price of needing somewhat more trades to prove a real edge. For a tool that tells people
   whether to trust their trading, that's the safer side to err on.
