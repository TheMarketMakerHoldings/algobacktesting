# Explainer 4: The website's "profitable or lucky" number

Page: `docs/index.html`
Tests: `tests/test_site.py` (the page's math against the library),
`tests/test_site_import.py` (reading trade exports), `tests/test_site_lab.py` (the stop lab)
and
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
  In a TradingView export the setup is the entry signal (such as `RTH_L`). The exit row's
  signal (`RTH_L_x`) only says how the trade closed, so the filter reads each trade's entry
  row, matched to its exit row by trade number.

Only Quantower's column names come from a published list. The TopstepX, NinjaTrader and
thinkorswim samples in `tests/exports/` are modelled on secondhand descriptions, so every
choice the importer makes is shown on the page and can be changed.

## The stop lab

"Would a different stop have done better?" The lab keeps each trade's entry (time, side,
price) from the trade file and replays it on the trader's own price bars with a fixed stop,
a fixed target and, optionally, a stop moved to the entry price once the trade is some
ticks in profit. It does that for every stop size in a range, then reports each size's win
rate, average per trade, net result, drawdown and the chance its edge is real.

The rules, all stated on the page:

- **When a bar touches both the stop and the target, the stop counts.** One-minute bars do
  not say which came first, so the lab assumes the worse one.
- **The entry bar can only stop a trade out**, unless the entry was at that bar's open: its
  earlier prices happened before the trade existed.
- **Stops and targets fill at their exact price.** Real fills can be worse.
- **Trades and bars are lined up by time.** If the two files use different clocks, the lab
  tries whole-hour shifts and keeps the one where most entry prices fall inside their bar.

**The search counts.** Comparing 19 stop sizes and keeping the best is itself a search,
just like trying 19 setups. So the lab scores each size as one of (setups tried × stop
sizes) trials. On the page's example, built on a random walk where no edge exists, the best
stop looks clearly better than the real one, and the lab still says it is luck. That is the
point: tuning a stop to past data is the easiest way to fool yourself.

The lab also reports how far winning trades went against the entry before they closed
("9 in 10 of your winners never went more than 6 ticks against you"), which shows how tight a
stop can be before it starts cutting off winners. Bar data stays on the trader's computer;
licensed exchange data must not be published.

## The prop-firm simulator

"Would this strategy pass an evaluation?" The simulator replays the trader's own results
5,000 times. Each simulated day draws a number of trades (from the file's real trading days,
or a fixed number) and that many results at random from the trade list, with replacement.
The run passes as soon as the balance reaches the profit target and fails as soon as it
falls to the loss limit. The limit stays fixed, trails the highest end-of-day balance, or
trails the highest balance after any trade, and never rises above the starting balance. A
daily loss limit either ends the day or fails the evaluation.

It also runs every evaluation again with the average result subtracted from each trade:
the same swings with no edge. The gap between the two pass rates is what the edge is worth,
if the edge is real; if the check says it could be luck, the zero-edge figure is the honest
one.

Checks: with fair +-1 trades, a fixed limit of 20 and a target of 30, the pass rate must be
20 / (20 + 30) = 40%, the gambler's ruin answer, and a biased coin must match its formula
too. Because each run sees the same trades in the same order under every rule set, a fixed
limit can never pass less often than an end-of-day trailing one, which can never pass less
often than one that trails every trade; the tests check that ordering.

## Timing: when does it work?

The timing section splits the trades by the hour of day or the weekday in the file's time
column. Finding a "best hour" means looking at every hour, which is a search like trying
setups, so each slice with at least 5 trades is scored with the Deflated Sharpe Ratio using
(slices with enough trades) x (setups tried) as the number of tries. The same check on the
negated results scores whether a losing slice's losses are a real pattern. On the page's
random-walk example the best hour reaches only "promising, not proven": exactly what should
happen where no edge exists.

## Comparing strategies, and saving them

"Which of my setups is really better?" The compare section takes two groups of trades and
reports the difference in average result per trade, and the chance the better one is really
better. That chance is the difference divided by its standard error, read off the normal
curve (Welch's two-sample comparison). The standard error is the square root of each group's
variance divided by its number of trades, added together. Each group needs 20 trades.

A group can be:

- one value of a column in the loaded file: a strategy, long or short trades, a contract;
- the trades now in the check;
- a strategy saved earlier. A TradingView export holds one strategy, so saving is how two
  exports get compared: load one, type a name, save, then load the next and compare.

Saved strategies are kept in the browser's own storage on that device (localStorage), under
one key. Nothing is uploaded, and clearing the site's data in the browser deletes them. Each
keeps its per-trade results, its unit and the number of setups tried when it was saved, so
its chance of being real is scored with its own count of setups. Results are rounded to six
decimals, which moves the average by at most 0.0000005.

**The comparison assumes the two groups are separate trades.** Welch's test treats them as
independent samples. ORB trades and long trades can share trades, and so can the trades in
the check and a group from the same file, so the page refuses those pairs. It cannot see
overlap between two saved strategies, so the page warns: comparing a strategy with part of
itself is not a fair test.

Checks: the formula matches a separate Python computation to 12 decimal places. With 2,000
pairs drawn from one distribution, about 5% are called different (within four binomial
standard errors). A saved strategy read back gives the same statistics and the same
comparison as its original trades to within 0.000001. The stored list is checked entry by
entry, so a damaged or hand-edited list is dropped, never trusted.

## Sharing a result

"Share" under the verdict makes a link to the result. For the quick check the link holds its
four inputs. For a list of trades it holds the numbers the verdict is computed from, not the
trades: the number of trades and setups, the average and standard deviation of the results,
their skew and kurtosis, plus the win, loss and breakeven counts and the max drawdown for
display. That is enough because the chance the edge is real depends on the trades only
through those statistics.

The numbers sit after the "#" in the link, which browsers never send to a server. The page
that opens the link works the verdict out again from them, and the visitor can change the
setups to see how much the verdict depends on that count. A link can be edited by hand, so
the page checks every number. It refuses impossible ones, such as a kurtosis below
1 + skew², and says a shared result is a claim it cannot check against real trades.

Checks: on nine trade lists of very different shapes, from 10 to 5,000 trades, a link read
back gives the same chance (to 0.000000001), the same trades needed and the same luck line
as the trades themselves. The link is the same length for 10 trades or 5,000. Damaged or
edited links are refused with a message.

## How we know it behaves

- **The page's math equals the library's.** A test runs the page's own code and compares it
  with the Python library on 401 quick-check cases and 120 pasted-trade cases.
- **"Trades needed" is exact.** At that number of trades the library returns exactly 95%.
- **The stop lab replays exactly.** A separate Python version of its rules gives the same
  result on every example trade at every stop size, and replaying the example strategy with
  its own settings reproduces each of its 151 trades to the cent.
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
