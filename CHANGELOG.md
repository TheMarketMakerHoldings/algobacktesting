# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Overfitting check in the strategy builder: once two or more versions have been tested on
  the same bars, the page reports the probability that the best version is overfit, using a
  port of the library's `probability_of_backtest_overfitting` (CSCV, Bailey et al. 2017) on
  each version's daily profit and loss, next to what versions with no real difference would
  give. `tests/test_site_pbo.py` checks the port against the library on five shapes of data.

### Changed

- The website's developer section installs from PyPI: `pip install algobacktesting`.

## [0.1.0] - 2026-11-24

### Added

- `algobacktesting.sharpe`, verified against Bailey & López de Prado (2012, 2014):
  - `probabilistic_sharpe_ratio` (PSR)
  - `deflated_sharpe_ratio` (DSR)
  - `expected_max_sharpe` (SR0)
  - `min_track_record_length` (MinTRL)
  - `return_stats`, `sharpe_ratio`, `annualize_sharpe`, `deannualize_sharpe`
- `algobacktesting.pbo`, verified against Bailey, Borwein, López de Prado & Zhu (2017):
  - `probability_of_backtest_overfitting` (CSCV), with logits, performance degradation
    and the probability of out-of-sample loss
- `algobacktesting.cv`, built from the standard definitions in López de Prado (2018).
  **Pending review against the book.**
  - `purged_kfold` (purging and embargo)
  - `combinatorial_purged_cv` (CPCV, with backtest paths)
- Plain-English explainers in `docs/explainers/` and a runnable demo in
  `examples/overfit_demo.py`.
- `examples/leakage_demo.py`: a no-skill model scores +0.82 under shuffled k-fold,
  +0.05 under unpurged contiguous k-fold and 0.00 under purged k-fold.

- `docs/index.html`: a one-page site that asks "are you actually profitable, or just
  lucky?". Enter a win rate, reward-to-risk ratio and number of trades, or paste trade
  results, and it reports the chance the edge is real and the trades needed to prove it.
  `tests/test_site.py` runs the page's JavaScript with Node.js and checks it against the
  library on more than 500 cases.

- `tests/test_luck_calibration.py` and report section 5.6: with no edge, the site's check gives
  a false "proven" about 5% of the time for one setup and about 1% or less for five or more.
- `docs/explainers/04-profitable-or-lucky.md`: how the website's number is computed.

- API reference (`docs/api.md`), generated from the docstrings and checked by a test.
- `CONTRIBUTING.md` with the verification standard, `CITATION.cff`, `SECURITY.md`, and
  issue and pull-request templates.
- CI also runs on Windows and macOS, and on NumPy 1.26, the oldest version supported.
  Actions are pinned to exact commits, and the secret scan runs a pinned,
  checksum-verified scanner.
- `tests/test_docs.py`: the README's code is run, its tables are compared with the
  demos' output, and links into the repository must resolve.
- Website: screen-reader support (slider values, announced result, described chart),
  text contrast that meets WCAG AA in both themes, a link-preview image and a
  not-found page.
- Website redesigned as a trading terminal, with a trade importer: drop a CSV exported
  from TopstepX, Quantower, NinjaTrader or thinkorswim (or any file with a P&L column),
  pick one strategy, and get the verdict, an equity curve against what luck alone would
  have made, and a full breakdown: win rate with and without breakevens, what breakevens
  cost in fees, payoff, profit factor, largest win and loss, longest losing streak and
  maximum drawdown. Fees are subtracted unless the P&L column is already net.
  `tests/test_site_import.py` checks the importer on sample exports from all four
  platforms and on the built-in example.
- Stop lab on the website: replays each imported trade's entry on the trader's own price bars
  (CSV, such as a Quantower history export) with a range of stop sizes, a target and an
  optional move to breakeven. Each stop size counts as a trial, so a stop that only looks
  better by luck is reported as luck. `docs/build_example.py` generates the example: a
  synthetic random-walk market (`docs/example-bars.csv`) and two strategies traded on it.
  `tests/test_site_lab.py` checks the replay against a separate Python implementation.
- The website moved to https://algobacktesting.com/.
- Compare two strategies on the website: any two groups in a trade file (strategies, long
  against short, contracts) side by side, with the chance the better one is really better
  (Welch's two-sample comparison). `tests/test_site_compare.py` checks the formula and that
  groups with no real difference are called different about 5% of the time.
- TradingView Strategy Tester exports ("List of trades") are read correctly: each trade's
  Entry and Exit rows count once, the "Date and time" column is used, and the stop lab
  pairs entries with exits. Before this, every trade was counted twice.
- Timing breakdown on the website: results by hour of day or by weekday, with each slice's
  chance of being real scored as one of all the slices looked at (times the setups tried),
  for gains and for losses. `tests/test_site_when.py` checks it against the library.
- Fix: with an empty paste box or an error in the check, the stop lab, simulator and timing
  sections kept showing the previous trades' results; they now clear.
- Prop-firm challenge simulator on the website: replays the trader's own results in random
  order 5,000 times under an evaluation's rules (profit target; a loss limit that trails the
  end-of-day balance, trails every trade or stays fixed; an optional daily limit; a day
  limit) and reports the chance to pass, the chance to fail and the median days to pass,
  plus the pass rate with the edge set to zero. `tests/test_site_prop.py` checks it against
  the gambler's ruin formulas and hand-worked cases.
- "Save as PDF" on the website: a one-page report of the verdict and breakdown, with the
  equity chart, the data used and the date.
- Stop lab replays thinkorswim trades: pairing buys and sells first in, first out now keeps
  each round trip's opening fill (time, price, side, size) and closing fill. Checked by
  writing the example's 151 trades as a thinkorswim statement: all 151 line up, and at every
  stop size the result differs from the original file's by exactly the $4.12 fee that the
  statement leaves out.
- Strategy builder reads a strategy written in plain words ("go long when price breaks the
  first 15 minutes after 9:30, stop 12 ticks, target 24, out by 11:30"), shows how it read
  them with every default marked as assumed, and says plainly what it could not understand.
  The result card shows the trades, net after fees, win rate, the chance the edge is real, a
  running-profit line and the last trades, and a list of every version tested, each judged
  against all of them. The detailed fields stay available under "Edit the details". The
  reader runs in the page: no AI service, nothing sent anywhere. `tests/test_site_build.py`
  checks twelve phrasings, including stop and target written either way round.
- Stop lab lines up TradingView trades from continuous contracts (such as `NQ1!`): their
  prices are adjusted at each roll, so they sit a fixed gap away from real contract prices,
  and each trade can be stamped a chart bar before it filled. The lab searches minute shifts
  and finds the gap for each stretch between rolls, then replays the trades on real prices;
  every move, and so every result, is unchanged. It says what it corrected, and says plainly
  when the trades and the bars cover different dates. `tests/test_site_lab.py` disguises the
  example's 151 trades this way and checks that the shift and both gaps are recovered and
  the replay is identical, and that unrelated trades are never forced to line up.
- Website made easier to read, after a review by its first user:
  - Stop lab: the contract is read from the file names (TradingView's `MNQ1!`, a bar file
    called `ES.csv`); the stops tried, the target and the hold time start from the trader's
    own trades; the settings sit under "Adjust the test"; and the result starts with one
    plain answer (keep the stop, or which one to use).
  - Timing groups trades by when they were entered (TradingView entry rows, TopstepX
    EnteredAt, a position's first fill), says the hours are in the file's own time zone, and
    explains its "Chance it's real" column.
  - The prop-firm simulator takes trades per day from the file whenever it has dates.
  - The setups question, the profit-versus-luck chart and the trades-to-prove-it chart say
    in plain words what they mean; the proof chart marks where 95% is reached.
  - Filter menus for TradingView files read "Entry signal" and "Long trades / Short trades".
- Strategy builder on the website: write a rule as a sentence with blanks (opening-range
  breakout, a set time with or against the morning's move, a moving-average cross, or a break
  of the previous day's high or low; stop, target, move to breakeven, exit time, trades a
  day) and test it on your own bars. Every version tested counts as a setup. The trades open
  in the check with all its tools. `tests/test_site_build.py` checks every rule against a
  separate Python version, on hand-worked cases, and by reproducing the example file's two
  strategies trade for trade.
- Shareable result links on the website: "Share" under the verdict gives a link to copy or
  post on X. It holds summary numbers only (the quick check's inputs, or for a list of
  trades the statistics the verdict uses and a few counts), after the "#" so no server sees
  them. The page that opens it works the verdict out again and checks every number.
  `tests/test_site_share.py` checks that a link reproduces the sender's verdict and that
  damaged links are refused.
- Save and name strategies on the website: type a name and save the trades in the check;
  they are kept in the browser's own storage on that device, never uploaded. Compare lists
  saved strategies, so one export can be compared with another, and each can be deleted.
  `tests/test_site_saved.py` checks names, limits, reading back damaged storage, and that a
  saved strategy gives the same statistics and comparison as its original trades.
- Fix: for TradingView exports, "Strategy / filter" listed both entry and exit signals, and
  the exit ones matched no trade. It now lists each trade's entry signal (its setup), in the
  check, the stop lab and Compare.
- Fix: Compare no longer offers pairs that share trades, such as a strategy against long
  trades from the same file.
- Fix: "Save as PDF" no longer prints the simulator, timing and compare sections.
- Website redesigned as a light, corporate product page. It now loads nothing from other
  sites (system fonts, no third-party scripts) and carries a Content-Security-Policy that
  only allows network requests to the site itself, so a visitor's files cannot be sent
  anywhere. Added robots.txt, a sitemap and structured data for search engines.
  `tests/test_site_privacy.py` guards both.
- `RELEASING.md` (release checklist), `CODE_OF_CONDUCT.md`, and an issue template for
  suggesting a new check. The release workflow validates the package description and
  opens a GitHub release.

### Safety

These inputs raise clear errors instead of returning plausible wrong numbers:
- NaN or infinite inputs
- impossible moments (kurtosis < 1 + skew², which usually means excess kurtosis was
  passed)
- an effective number of trials between 1 and 2
- a matrix passed where a single return series is expected
- more than one million CSCV splits
