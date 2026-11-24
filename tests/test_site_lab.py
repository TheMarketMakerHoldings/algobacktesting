"""The website's stop lab must replay trades exactly as its rules say.

``docs/index.html`` replays each imported trade's entry on price bars with a chosen stop,
target and breakeven rule. This test runs the page's code (between ``/* lab:start */`` and
``/* lab:end */``, with the statistics and importer it uses) in Node.js and checks it three
ways:

1. Against a separate Python implementation of the same rules, on every trade of both example
   strategies, at 19 stop sizes, with and without a breakeven rule.
2. Against the example trade file itself: ``docs/build_example.py`` generated those trades on
   the example bars, so replaying them with the strategy's own bracket must give the same
   result for every trade.
3. On small hand-made cases for each rule (stop before target, the entry bar, session gaps,
   breakeven, short trades) and on time-zone alignment.

Skipped without Node.js.
"""

from __future__ import annotations

import calendar
import csv
import io
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
BARS = ROOT / "docs" / "example-bars.csv"
NODE = shutil.which("node")
TICK, POINT, FEE = 0.25, 50.0, 4.12
STOPS = list(range(4, 41, 2))

HARNESS = ROOT / "tests" / "site_lab_harness.js"


def _example_trades() -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index('<script type="text/plain" id="example-csv">')
    start = text.index(">", start) + 1
    return text[start : text.index("</script>", start)].strip()


# ---- an independent Python version of the page's replay rules ------------------------------


def _bars() -> tuple[list[datetime], list[tuple[float, float, float, float]]]:
    times, prices = [], []
    for row in csv.DictReader(io.StringIO(BARS.read_text(encoding="utf-8"))):
        times.append(datetime.strptime(row["DateTime"], "%m/%d/%Y %I:%M:%S %p"))
        prices.append(
            (float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]))
        )
    return times, prices


def _replay(
    times: list[datetime],
    prices: list[tuple[float, float, float, float]],
    start: int,
    side: int,
    stop: float,
    target: float,
    breakeven: float,
) -> float:
    """Points won on one trade entered at bar ``start``'s open (all example entries are)."""
    entry = prices[start][0]
    stop_price, moved, last = entry - side * stop, False, start
    for j in range(start, len(prices)):
        if j > start and (
            times[j] - times[start] > timedelta(minutes=240)
            or times[j] - times[j - 1] > timedelta(minutes=30)
        ):
            break
        _, high, low, _ = prices[j]
        if (side > 0 and low <= stop_price) or (side < 0 and high >= stop_price):
            return side * (stop_price - entry)
        if target > 0 and (
            (side > 0 and high >= entry + target) or (side < 0 and low <= entry - target)
        ):
            return target
        gain = high - entry if side > 0 else entry - low
        if breakeven > 0 and not moved and gain >= breakeven:
            stop_price, moved = entry, True
        last = j
    return side * (prices[last][3] - entry)


def _python_scan(strategy: str, target: int, breakeven: int) -> dict[int, list[float]]:
    times, prices = _bars()
    index = {t: i for i, t in enumerate(times)}
    rows = [r for r in csv.DictReader(io.StringIO(_example_trades())) if r["Strategy"] == strategy]
    result: dict[int, list[float]] = {}
    for stop in STOPS:
        values = []
        for r in rows:
            start = index[datetime.strptime(r["Entry time"], "%Y-%m-%d %H:%M:%S")]
            side = 1 if r["Side"] == "Long" else -1
            points = _replay(
                times, prices, start, side, stop * TICK, target * TICK, breakeven * TICK
            )
            values.append(round(points * POINT - FEE, 8))
        result[stop] = values
    return result


# ---- hand-made cases -----------------------------------------------------------------------

MINUTE = 60_000
# Each case: bars as [time, open, high, low, close], the entry bar, the replay options, expected.
CASES: list[tuple[str, list[list[float]], int, dict[str, Any], dict[str, Any]]] = [
    (
        "stop and target in one bar: the stop counts",
        [[0, 100, 100.5, 99.5, 100], [MINUTE, 100, 103, 97, 100]],
        0,
        {"side": 1, "entry": 100, "stop": 2, "target": 2, "breakeven": 0, "atOpen": True},
        {"points": -2, "exit": "stop", "index": 1},
    ),
    (
        "entry mid-bar: that bar's high came before the entry, so no target there",
        [[0, 100, 103, 99.5, 100], [MINUTE, 100, 100.5, 99.5, 100.25]],
        0,
        {"side": 1, "entry": 100, "stop": 2, "target": 2, "breakeven": 0, "atOpen": False},
        {"points": 0.25, "exit": "time", "index": 1},
    ),
    (
        "entry at the open: the same bar can reach the target",
        [[0, 100, 103, 99.5, 100], [MINUTE, 100, 100.5, 99.5, 100.25]],
        0,
        {"side": 1, "entry": 100, "stop": 2, "target": 2, "breakeven": 0, "atOpen": True},
        {"points": 2, "exit": "target", "index": 0},
    ),
    (
        "a session gap ends the trade at the last bar before it",
        [
            [0, 100, 100.5, 99.5, 100.5],
            [MINUTE, 100.5, 101, 100, 101],
            [90 * MINUTE, 110, 112, 109, 111],
        ],
        0,
        {"side": 1, "entry": 100, "stop": 5, "target": 5, "breakeven": 0, "atOpen": True},
        {"points": 1, "exit": "time", "index": 1},
    ),
    (
        "breakeven: after 1 point of profit the stop moves to the entry",
        [[0, 100, 101, 99.75, 100.5], [MINUTE, 100.5, 100.75, 99.5, 99.75]],
        0,
        {"side": 1, "entry": 100, "stop": 2, "target": 3, "breakeven": 1, "atOpen": True},
        {"points": 0, "exit": "stop", "index": 1},
    ),
    (
        "short trade reaching its target",
        [[0, 100, 100.5, 99.75, 99.75], [MINUTE, 99.75, 100, 97.5, 98]],
        0,
        {"side": -1, "entry": 100, "stop": 1, "target": 2, "breakeven": 0, "atOpen": True},
        {"points": 2, "exit": "target", "index": 1},
    ),
]
for _case in CASES:
    _case[3].update({"endTime": 10**12, "gapMs": 30 * MINUTE})

QUANTOWER_BARS = (
    "\ufeffDateTime,UTC,Open,High,Low,Close,Volume,Delta\n"
    "11/2/2026 9:30:00 AM,-05:00,6000,6001,5999.75,6000.5,100,12\n"
    "11/2/2026 9:31:00 AM,-05:00,6000.5,6002,6000.25,6001.75,90,-4\n"
    "11/2/2026 9:32:00 AM,-05:00,6001.75,6003,6001.5,6002.5,80,3\n"
)
ALIGN = [
    # Trade times with no zone, written 5 hours ahead of the bars' clock: found by the shift.
    {
        "bars": QUANTOWER_BARS,
        "trades": "Entry time,Side,Entry price,Symbol,P&L\n"
        "2026-11-02 14:30:00,Long,6000,ESZ6,25\n2026-11-02 14:31:20,Short,6001,ESZ6,-12.5\n",
        "expect": {"useUtc": False, "shift": -5, "matched": 2, "entries": 2},
    },
    # Both files give UTC: TopstepX-style offsets on the trades, a UTC column on the bars.
    {
        "bars": QUANTOWER_BARS,
        "trades": "EnteredAt,Type,EntryPrice,ContractName,PnL\n"
        "11/02/2026 09:30:00 -05:00,Long,6000,ESZ6,25\n"
        "11/02/2026 15:32:30 +01:00,Long,6002,ESZ6,-12.5\n",
        "expect": {"useUtc": True, "shift": 0, "matched": 2, "entries": 2},
    },
]
TRADINGVIEW = (ROOT / "tests" / "exports" / "tradingview.csv").read_text(encoding="utf-8")
TV_BARS = (
    "DateTime,Open,High,Low,Close\n"
    "3/2/2026 9:40:00 AM,20100,20104,20096,20101\n"
    "3/3/2026 10:20:00 AM,20145,20149,20140,20146\n"
    "3/4/2026 9:31:00 AM,20200,20203,20190,20199\n"
)
ALIGN.append(
    # TradingView: entries come from each trade's Entry row, paired with its Exit row.
    {
        "bars": TV_BARS,
        "trades": TRADINGVIEW,
        "expect": {"useUtc": False, "shift": 0, "matched": 3, "entries": 3},
    }
)
ROOTS = [
    ("ESU6", "ES"),
    ("MESZ25", "MES"),
    ("ES 09-26", "ES"),
    ("/MNQH26:XCME", "MNQ"),
    ("NQ.FUT", "NQ"),
    ("ESZ6-ESH7", "ES"),
    ("AAPL", ""),
    ("MES", "MES"),
]

# Wide exports: Quantower writes about 40 columns with quoted, comma-holding names; some tools
# quote prices that contain thousands separators. Only the needed columns may be read.
WIDE = [
    'DateTime,UTC,Open,High,Low,Close,"Buy (Ask) volume, %","Sell (Bid) volume, %",Delta\n'
    "11/2/2026 9:30:00 AM,-05:00,6000,6001,5999.75,6000.5,55.5,44.5,12\n"
    '"11/2/2026 9:31:00 AM","-05:00","6,000.5","6,002","6,000.25","6,001.75","49,1",50.9,-4\n',
    "no bars here\njust,text\n",
]


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "lab")
    )
    scans = [
        {"strategy": "ORB", "stops": STOPS, "target": 24, "breakeven": 0},
        {"strategy": "ORB", "stops": STOPS, "target": 24, "breakeven": 12},
        {"strategy": "Fade", "stops": STOPS, "target": 10, "breakeven": 6},
        {"strategy": "ORB", "stops": [12], "target": 24, "breakeven": 12},
    ]
    payload = {
        "page": {"trades": _example_trades()},
        "bars": BARS.read_text(encoding="utf-8"),
        "scans": scans,
        "cases": [{"bars": c[1], "start": c[2], "opt": c[3]} for c in CASES],
        "align": [{"bars": a["bars"], "trades": a["trades"]} for a in ALIGN],
        "roots": [r[0] for r in ROOTS],
        "wide": WIDE,
    }
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "lab.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)], capture_output=True, text=True, check=True, timeout=120
        )
    out: dict[str, Any] = json.loads(done.stdout)
    out["scan_specs"] = scans
    return out


def test_example_bars_are_read(js: dict[str, Any]) -> None:
    bars = js["bars"]
    assert bars["n"] == 153 * 120
    assert bars["barMs"] == 60_000
    # 6/1/2026 9:30 AM, -04:00: the clock as written and UTC four hours later.
    clock = calendar.timegm((2026, 6, 1, 9, 30, 0)) * 1000
    assert bars["first"]["clock"] == clock
    assert bars["first"]["utc"] == clock + 4 * 3_600_000


@pytest.mark.parametrize("which", [0, 1, 2])
def test_replay_matches_an_independent_python_version(js: dict[str, Any], which: int) -> None:
    spec, got = js["scan_specs"][which], js["scans"][which]
    assert got["matched"] == got["entries"] > 140  # every entry lines up with its bar
    assert got["shift"] == 0
    assert got["atOpen"] == got["matched"]  # example entries are all at a bar's open
    expected = _python_scan(spec["strategy"], spec["target"], spec["breakeven"])
    for row in got["rows"]:
        assert row["values"] == pytest.approx(expected[row["ticks"]], abs=1e-6), row["ticks"]


def test_replay_reproduces_the_example_trades(js: dict[str, Any]) -> None:
    # The generator traded ORB with a 12-tick stop, 24-tick target and breakeven at 12 ticks.
    rows = [r for r in csv.DictReader(io.StringIO(_example_trades())) if r["Strategy"] == "ORB"]
    actual = [float(r["P&L"]) - float(r["Commission"]) for r in rows]
    assert js["scans"][3]["rows"][0]["values"] == pytest.approx(actual, abs=1e-6)


def test_choosing_a_stop_counts_as_a_search(js: dict[str, Any]) -> None:
    # 5 setups tried, times 19 stop sizes compared.
    assert js["scans"][0]["trials"] == 5 * len(STOPS)
    assert js["scans"][3]["trials"] == 5


@pytest.mark.parametrize("case", range(len(CASES)), ids=[c[0] for c in CASES])
def test_replay_rules(js: dict[str, Any], case: int) -> None:
    assert js["cases"][case] == CASES[case][4]


def test_trades_are_lined_up_with_bars(js: dict[str, Any]) -> None:
    for spec, got in zip(ALIGN, js["align"], strict=True):
        assert got == spec["expect"]


def test_instrument_roots(js: dict[str, Any]) -> None:
    assert js["roots"] == [r[1] for r in ROOTS]


def test_wide_and_quoted_bar_files(js: dict[str, Any]) -> None:
    nine_thirty = calendar.timegm((2026, 11, 2, 9, 30, 0)) * 1000
    five_hours = 5 * 3_600_000
    assert js["wide"][0] == [
        [nine_thirty, nine_thirty + five_hours, 6000, 6001, 5999.75, 6000.5],
        [nine_thirty + 60_000, nine_thirty + 60_000 + five_hours, 6000.5, 6002, 6000.25, 6001.75],
    ]
    assert js["wide"][1] is None
