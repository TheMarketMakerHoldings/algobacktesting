"""The website's trade importer must read each platform's export correctly.

``docs/index.html`` reads trade exports in the browser. This test extracts the page's importer
(between ``/* csv:start */`` and ``/* csv:end */``) and its statistics, runs them with Node.js
on the sample files in ``tests/exports/``, and compares the per-trade results with values
worked out by hand from those files. The page's built-in example is also checked against the
Python library. Skipped without Node.js.

The sample files imitate each platform's export; they are not real exports. See
``tests/exports/README.md`` for where each set of column names comes from.
"""

from __future__ import annotations

import calendar
import csv
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from algobacktesting import deflated_sharpe_ratio, return_stats

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
EXPORTS = ROOT / "tests" / "exports"
NODE = shutil.which("node")

HARNESS = """
const input = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
function run(text, filterSpec) {
  const r = readExport(text), t = r.table;
  if (!t) return null;
  let filter = null;
  if (filterSpec) filter = { col: t.header.indexOf(filterSpec[0]), value: filterSpec[1] };
  const res = t.kind === "fills" ? fifoTrades(t, t.cols, filter) : tradesFromTable(t, {
    pnl: t.cols.pnl, fees: t.cols.fees, subtractFees: r.subtractFees,
    pnlIsNet: t.cols.pnlRank === 3, filter: filter, group: t.cols.group, time: t.cols.time
  });
  const n = res.values.length, stats = n > 1 ? statsFromTrades(res.values) : null;
  return {
    platform: r.platform, kind: t.kind, title: t.title,
    pnl: t.cols.pnl >= 0 ? t.header[t.cols.pnl] : null,
    fees: t.cols.fees.map(function (c) { return t.header[c]; }),
    subtractFees: r.subtractFees, values: res.values, before: res.before, open: res.open,
    skipped: res.skipped, unknown: res.unknownMultiplier,
    filters: t.cols.filters.map(function (f) { return [f.name, f.values]; }),
    labels: t.cols.filters.map(function (f) { return [f.name, f.labels]; }),
    times: res.times, entryTimes: res.entryTimes, roundTrips: res.roundTrips || null,
    summary: tradeSummary(res.values, res.before),
    dsr5: stats ? edgeProbability(stats, n, 5) : null
  };
}
console.log(JSON.stringify({
  runs: input.runs.map(function (r) { return run(r[0], r[1]); }),
  numbers: input.numbers.map(function (c) { return parseNumber(c[0], c[1]); }),
  times: input.times.map(parseTime),
  mult: input.mult.map(function (c) { return contractMultiplier(c[0], c[1]); })
}));
"""

INLINE = {
    "decimal comma": "Symbol;Net P/L\nES;1.234,50\nES;-12,25\nES;0\n",
    "total row": "Trade,P&L\n1,10\n2,-5\nTotal,5\n",
    "no p&l": "Name,Colour\nA,red\nB,blue\n",
    "quoted": 'Strategy,P&L\n"Break, retest",10\n"He said ""go""",-5\n',
}


def _export(name: str) -> str:
    return (EXPORTS / f"{name}.csv").read_text(encoding="utf-8")


RUNS: list[tuple[str, str, list[str] | None]] = [
    ("topstepx", _export("topstepx"), None),
    ("quantower", _export("quantower"), None),
    ("quantower orb", _export("quantower"), ["Comment", "ORB"]),
    ("ninjatrader", _export("ninjatrader"), None),
    ("ninjatrader orb", _export("ninjatrader"), ["Strategy", "ORB"]),
    ("thinkorswim", _export("thinkorswim"), None),
    ("tradingview", _export("tradingview"), None),
    ("tradingview short", _export("tradingview"), ["Type", "Exit short"]),
    ("tradingview signal L", _export("tradingview"), ["Signal", "L"]),
    ("tradingview exit signal", _export("tradingview"), ["Signal", "L_x"]),
    *((name, text, None) for name, text in INLINE.items()),
]

NUMBERS: list[tuple[str, str, float | None]] = [
    ("$1,234.50", ",", 1234.5),
    ("(25.00)", ",", -25.0),
    ("($150.00)", ",", -150.0),
    ("-$12.50", ",", -12.5),
    ("$-12.50", ",", -12.5),
    ("12.50-", ",", -12.5),
    ("+3", ",", 3.0),
    ("\u22127.5", ",", -7.5),  # Unicode minus sign
    ("1.234,56", ";", 1234.56),
    ("12,5", ";", 12.5),
    ("\u20ac1.000,00", ";", 1000.0),
    ("USD 42", ",", 42.0),
    ("1,5", ",", None),  # ambiguous with a comma delimiter: refused, not guessed
    ("12%", ",", None),
    ("abc", ",", None),
    ("", ",", None),
]

TIMES: list[tuple[str, list[int] | None]] = [
    ("6/2/26 09:58:30", [2026, 6, 2, 9, 58, 30]),
    ("06/01/2026 9:31 PM", [2026, 6, 1, 21, 31, 0]),
    ("6/1/2026 12:05:00 AM", [2026, 6, 1, 0, 5, 0]),
    ("6/1/2026 12:05:00 PM", [2026, 6, 1, 12, 5, 0]),
    ("2026-06-02 13:20:30", [2026, 6, 2, 13, 20, 30]),
    ("06/05/2026 10:09:40 -04:00", [2026, 6, 5, 10, 9, 40]),
    ("", None),
]

MULTIPLIERS: list[tuple[str, str, float, bool]] = [
    ("/ESU26", "FUTURE", 50, True),
    ("/MESZ25:XCME", "", 5, True),
    ("/NQH26", "", 20, True),
    ("/MNQM6", "", 2, True),
    ("/ZNH26", "", 1000, True),
    ("/6EZ25", "", 125000, True),
    ("/XYZU26", "FUTURE", 1, False),  # unknown contract: reported, results stay in points
    ("AAPL", "STOCK", 1, True),
    ("SPY", "CALL", 100, True),
]


def _example_csv() -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index('<script type="text/plain" id="example-csv">')
    start = text.index(">", start) + 1
    return text[start : text.index("</script>", start)].strip()


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = text[text.index("/* stats:start */") : text.index("/* stats:end */")]
    code += text[text.index("/* csv:start */") : text.index("/* csv:end */")]
    runs = [*RUNS, ("example", _example_csv(), ["Strategy", "ORB"])]
    payload = {
        "runs": [[r[1], r[2]] for r in runs],
        "numbers": [[raw, delim] for raw, delim, _ in NUMBERS],
        "times": [raw for raw, _ in TIMES],
        "mult": [[sym, typ] for sym, typ, _, _ in MULTIPLIERS],
    }
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "import.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS, encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)], capture_output=True, text=True, check=True, timeout=60
        )
    out: dict[str, Any] = json.loads(done.stdout)
    out["by_name"] = {name: result for (name, _, _), result in zip(runs, out["runs"], strict=True)}
    return out


def test_topstepx_export(js: dict[str, Any]) -> None:
    r = js["by_name"]["topstepx"]
    assert (r["platform"], r["kind"], r["pnl"], r["fees"]) == ("TopstepX", "pnl", "PnL", ["Fees"])
    assert r["subtractFees"] is True  # PnL is not labelled net, and a fee column exists
    # Rows are newest first in the file; trades come out in the order they closed.
    assert r["before"] == pytest.approx([-100.0, 310.0, 0.0, -200.0, 400.0, -200.0])
    assert r["values"] == pytest.approx([-102.84, 307.16, -2.84, -205.68, 397.16, -202.84])
    s = r["summary"]
    assert (s["wins"], s["losses"], s["breakevens"]) == (2, 3, 1)
    assert s["breakevenCost"] == pytest.approx(-2.84)  # flat before fees, a small loss after
    assert s["winRateDecided"] == pytest.approx(2 / 5)


def test_quantower_fills_are_combined_by_position(js: dict[str, Any]) -> None:
    r = js["by_name"]["quantower"]
    assert (r["platform"], r["pnl"], r["fees"]) == ("Quantower", "Net P/L", ["Fee"])
    assert r["subtractFees"] is False  # Net P/L already has the fee taken off
    # Position 7002 has three fills: -1.24 + 24.38 - 10.62.
    assert r["values"] == pytest.approx([18.76, 12.52, -26.24, -1.24])
    assert r["before"] == pytest.approx([20.0, 15.0, -25.0, 0.0])
    assert r["summary"]["breakevens"] == 1
    assert js["by_name"]["quantower orb"]["values"] == pytest.approx([18.76, -26.24])
    assert ["Comment", ["ORB", "Fade"]] in r["filters"]


def test_ninjatrader_export(js: dict[str, Any]) -> None:
    r = js["by_name"]["ninjatrader"]
    assert (r["platform"], r["pnl"], r["fees"]) == ("NinjaTrader", "Profit", ["Commission"])
    assert r["subtractFees"] is True
    # "Cum. net profit" is a running total and must never be read as a trade's result.
    assert r["values"] == pytest.approx([245.88, -154.12, 91.76, -4.12, 1220.88])
    assert js["by_name"]["ninjatrader orb"]["values"] == pytest.approx([245.88, -154.12, -4.12])
    assert ["Strategy", ["ORB", "VWAP"]] in r["filters"]


def test_thinkorswim_fills_are_paired_first_in_first_out(js: dict[str, Any]) -> None:
    r = js["by_name"]["thinkorswim"]
    assert (r["platform"], r["kind"]) == ("thinkorswim", "fills")
    assert r["title"] == "Account Trade History"
    # /MES +4.25 pt x $5; SPY 60 x +0.90 and 40 x -0.60; /MNQ short 2 x 9.5 pt x $2;
    # one SPY call +0.60 x 100. The /ES buy is still open and not counted.
    assert r["values"] == pytest.approx([21.25, 30.0, 38.0, 60.0])
    assert r["open"] == 1
    assert r["unknown"] is False
    # Each round trip keeps its opening and closing fills, so the stop lab can replay it.
    keys = ("entryRaw", "entryPrice", "side", "qty", "symbol", "exitRaw", "exitPrice")
    assert [tuple(t[k] for k in keys) for t in r["roundTrips"]] == [
        ("6/2/26 09:41:12", 5900.25, 1, 1, "/MESU26", "6/2/26 09:58:30", 5904.5),
        ("6/3/26 10:01:00", 590.1, 1, 100, "SPY", "6/3/26 11:00:00", 589.5),  # closed in two sells
        ("6/4/26 09:45:00", 21000.0, -1, 2, "/MNQU26", "6/4/26 10:15:00", 20990.5),
        ("6/5/26 10:00:00", 2.5, 1, 1, "SPY", "6/5/26 10:45:00", 3.1),
    ]


@pytest.mark.parametrize(("raw", "delimiter", "expected"), NUMBERS)
def test_money_cells(js: dict[str, Any], raw: str, delimiter: str, expected: float | None) -> None:
    got = js["numbers"][[n[0] for n in NUMBERS].index(raw)]
    assert got == (None if expected is None else pytest.approx(expected))


def test_timestamps(js: dict[str, Any]) -> None:
    for (raw, parts), got in zip(TIMES, js["times"], strict=True):
        if parts is None:
            assert got is None, raw
        else:
            assert got == calendar.timegm((*parts, 0, 0, 0)) * 1000, raw


def test_contract_multipliers(js: dict[str, Any]) -> None:
    for (symbol, _kind, mult, known), got in zip(MULTIPLIERS, js["mult"], strict=True):
        assert (got["mult"], got["known"]) == (mult, known), symbol


def test_awkward_files(js: dict[str, Any]) -> None:
    by = js["by_name"]
    assert by["decimal comma"]["values"] == pytest.approx([1234.5, -12.25, 0.0])
    assert by["total row"]["values"] == pytest.approx([10.0, -5.0])
    assert by["no p&l"] is None
    assert by["quoted"]["values"] == pytest.approx([10.0, -5.0])
    assert ["Strategy", ["Break, retest", 'He said "go"']] in by["quoted"]["filters"]


def test_example_matches_the_library(js: dict[str, Any]) -> None:
    # The page opens on this file, filtered to the ORB strategy, with 5 setups tried.
    rows = list(csv.DictReader(io.StringIO(_example_csv())))
    net = [float(r["P&L"]) - float(r["Commission"]) for r in rows if r["Strategy"] == "ORB"]
    r = js["by_name"]["example"]
    assert r["subtractFees"] is True
    assert r["values"] == pytest.approx(net)  # already in time order in the file
    stats = return_stats(net)
    expected = deflated_sharpe_ratio(
        stats.sharpe,
        stats.n_obs,
        n_trials=5,
        sharpe_variance=1 / (stats.n_obs - 1),
        skew=stats.skew,
        kurtosis=stats.kurtosis,
    )
    assert r["dsr5"] == pytest.approx(expected, abs=1e-9)
    flat = sum(1 for row in rows if row["Strategy"] == "ORB" and float(row["P&L"]) == 0)
    assert r["summary"]["breakevens"] == flat > 0


def test_tradingview_counts_each_trade_once(js: dict[str, Any]) -> None:
    r = js["by_name"]["tradingview"]
    assert (r["platform"], r["pnl"]) == ("TradingView", "Net PnL USD")
    # Each trade has an Entry row and an Exit row with the same result: only exits count.
    assert r["values"] == pytest.approx([195.5, -104.5, -4.5])
    assert r["subtractFees"] is False  # net already
    # Only exit values are offered, shown as the long and the short trades.
    assert ["Direction", ["Exit long", "Exit short"]] in r["filters"]
    assert ["Direction", ["Long trades", "Short trades"]] in r["labels"]
    assert js["by_name"]["tradingview short"]["values"] == pytest.approx([-104.5])


def test_tradingview_signal_names_the_setup(js: dict[str, Any]) -> None:
    # The Signal column names the setup on entry rows (L, S) and how the trade closed on exit
    # rows (L_x, S_x). Only entry signals are offered, and choosing one picks every trade that
    # entry opened (trades 1 and 3), although results are read from the exit rows.
    assert ["Entry signal", ["L", "S"]] in js["by_name"]["tradingview"]["filters"]
    assert js["by_name"]["tradingview signal L"]["values"] == pytest.approx([195.5, -4.5])
    assert js["by_name"]["tradingview exit signal"]["values"] == []  # never offered


def _ms(*parts: int) -> int:
    return calendar.timegm((*parts, 0, 0, 0)) * 1000 if len(parts) == 6 else 0


def test_entry_times(js: dict[str, Any]) -> None:
    # The timing section groups trades by when they were entered.
    by = js["by_name"]
    # TradingView: each trade's Entry row.
    assert by["tradingview"]["entryTimes"] == [
        _ms(2026, 3, 2, 9, 40, 0),
        _ms(2026, 3, 3, 10, 20, 0),
        _ms(2026, 3, 4, 9, 31, 0),
    ]
    # Quantower: a position's earliest fill (7001 bought at 9:31:05, sold at 9:40:10).
    assert by["quantower"]["entryTimes"][0] == _ms(2026, 6, 1, 9, 31, 5)
    # TopstepX and NinjaTrader: their entry time columns, never after the exit.
    for name in ("topstepx", "ninjatrader"):
        entries, exits = by[name]["entryTimes"], by[name]["times"]
        assert all(e < x for e, x in zip(entries, exits, strict=True)), name
    assert _ms(2026, 6, 5, 10, 2, 11) in by["topstepx"]["entryTimes"]
    assert by["ninjatrader"]["entryTimes"][0] == _ms(2026, 6, 1, 9, 31, 0)
