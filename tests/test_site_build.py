"""The website's strategy builder must trade each rule exactly as written.

``docs/index.html`` turns a rule written as a sentence with blanks into trades on the
trader's own bars (between ``/* build:start */`` and ``/* build:end */``). This test runs the
page's code in Node.js and checks it three ways:

1. Against a separate Python version of every rule (below), trade by trade, on a random
   market with missing bars, short days and late starts.
2. On small hand-worked cases.
3. On the page's example bars, where ``docs/build_example.py``, written separately, traded an
   opening-range breakout and a 10:30 fade: the builder must take the same trades at the same
   prices for the same results.

Skipped without Node.js.
"""

from __future__ import annotations

import calendar
import csv
import io
import itertools
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_build_harness.js"
NODE = shutil.which("node")
DAY_MS = 86_400_000

Bar = dict[str, float]
Rule = dict[str, Any]
Trade = tuple[int, int, float, float, str, float, float, int]

BASE: Rule = {
    "entry": "orb",
    "side": "both",
    "rangeStart": 570,
    "rangeMinutes": 15,
    "at": 630,
    "from": 570,
    "until": 660,
    "maBars": 20,
    "exitAt": 689,
    "stopTicks": 12,
    "targetTicks": 24,
    "breakevenTicks": 0,
    "maxPerDay": 1,
    "tick": 0.25,
    "point": 50,
    "fee": 4.12,
}


def _rule(**changes: Any) -> Rule:
    return {**BASE, **changes}


# ---------- the separate Python version ----------


def _js_round(x: float) -> float:
    """Math.round(x * 1e8) / 1e8, as the page rounds money (halves go up, unlike round())."""
    return math.floor(x * 1e8 + 0.5) / 1e8


def _minute(bar: Bar) -> int:
    return int(bar["clock"] // 60000) % 1440


def _bracket(day: list[Bar], e: int, side: int, rule: Rule) -> tuple[float, int, str]:
    tick, entry = rule["tick"], day[e]["o"]
    stop_px = entry - side * (rule["stopTicks"] * tick)
    target, breakeven = rule["targetTicks"] * tick, rule["breakevenTicks"] * tick
    moved, last = False, e
    for j in range(e, len(day)):
        if j > e and _minute(day[j]) > rule["exitAt"]:
            break
        bar = day[j]
        if (side > 0 and bar["l"] <= stop_px) or (side < 0 and bar["h"] >= stop_px):
            return side * (stop_px - entry), j, "stop"
        reached = bar["h"] >= entry + target if side > 0 else bar["l"] <= entry - target
        if target > 0 and reached:
            return target, j, "target"
        gain = bar["h"] - entry if side > 0 else entry - bar["l"]
        if breakeven > 0 and not moved and gain >= breakeven:
            stop_px, moved = entry, True
        last = j
    return side * (day[last]["c"] - entry), last, "time"


def _time_signal(day: list[Bar], rule: Rule, after: int) -> tuple[int, int] | None:
    for k in range(after + 1, len(day)):
        if _minute(day[k]) != rule["at"]:
            continue
        if rule["side"] in ("long", "short"):
            return (1 if rule["side"] == "long" else -1), k
        base = next((j for j in range(k) if _minute(day[j]) == rule["from"]), None)
        if base is None or day[k]["o"] == day[base]["o"]:
            return None
        move = 1 if day[k]["o"] > day[base]["o"] else -1
        return (move if rule["side"] == "with" else -move), k
    return None


def _signal(
    day: list[Bar], prev: list[Bar] | None, rule: Rule, after: int
) -> tuple[int, int] | None:
    if rule["entry"] == "time":
        return _time_signal(day, rule, after)
    first, hi, lo = rule["from"], -math.inf, math.inf
    if rule["entry"] == "orb":
        first = rule["rangeStart"] + rule["rangeMinutes"]
        rng = [b for b in day if rule["rangeStart"] <= _minute(b) < first]
        if not rng:
            return None
        hi, lo = max(b["h"] for b in rng), min(b["l"] for b in rng)
    elif rule["entry"] == "prior":
        if not prev:
            return None
        hi, lo = max(b["h"] for b in prev), min(b["l"] for b in prev)
    n = rule["maBars"]
    for k in range(after + 1, len(day)):
        if _minute(day[k]) < first:
            continue
        if _minute(day[k]) > rule["until"]:
            return None
        c = day[k]["c"]
        if rule["entry"] == "ma":
            if k < n:
                continue
            now = sum(b["c"] for b in day[k - n + 1 : k + 1]) / n
            before = sum(b["c"] for b in day[k - n : k]) / n
            up = day[k - 1]["c"] <= before and c > now
            down = day[k - 1]["c"] >= before and c < now
            move = 1 if up else -1 if down else 0
        else:
            move = 1 if c > hi else -1 if c < lo else 0
        wanted = {"both": move, "long": max(move, 0), "short": min(move, 0)}[rule["side"]]
        if wanted:
            if k + 1 < len(day) and _minute(day[k + 1]) <= rule["exitAt"]:
                return wanted, k + 1
            return None
    return None


def reference_trades(bars: list[Bar], rule: Rule) -> list[Trade]:
    days: list[list[Bar]] = []
    for bar in bars:
        if days and days[-1][0]["clock"] // DAY_MS == bar["clock"] // DAY_MS:
            days[-1].append(bar)
        else:
            days.append([bar])
    trades: list[Trade] = []
    for d, day in enumerate(days):
        after, count = -1, 0
        while count < rule["maxPerDay"]:
            sig = _signal(day, days[d - 1] if d else None, rule, after)
            if sig is None:
                break
            side, e = sig
            points, j, how = _bracket(day, e, side, rule)
            gross = _js_round(points * rule["point"])
            entry = day[e]["o"]
            trades.append(
                (
                    int(day[e]["clock"]),
                    side,
                    entry,
                    entry + side * points,
                    how,
                    gross,
                    _js_round(gross - rule["fee"]),
                    int(day[j]["clock"]),
                )
            )
            count, after = count + 1, j
    return trades


# ---------- inputs ----------


def _clock(text: str) -> int:
    return calendar.timegm(datetime.strptime(text, "%Y-%m-%d %H:%M").timetuple()) * 1000


def _random_market() -> list[Bar]:
    """Forty days of one-minute bars on a 0.25 grid, with gaps, short days and late starts."""
    rng = np.random.default_rng(2460554)
    bars: list[Bar] = []
    price = 24000
    for d in range(40):
        first, count = (600, 60) if d == 7 else (570, 30) if d == 11 else (570, 120)
        price += int(rng.normal(0, 30))
        for k in range(count):
            steps = np.rint(rng.normal(0.15 if d % 5 == 0 else 0, 1.8, 6)).astype(int)
            path = price + np.concatenate(([0], np.cumsum(steps)))
            price = int(path[-1])
            if d % 4 == 1 and rng.random() < 0.1:
                continue  # a missing minute
            day = f"2026-03-{2 + d:02d}" if d < 30 else f"2026-04-{d - 28:02d}"
            clock = _clock(day + " 00:00")
            bars.append(
                {
                    "clock": clock + (first + k) * 60000,
                    "o": int(path[0]) * 0.25,
                    "h": int(path.max()) * 0.25,
                    "l": int(path.min()) * 0.25,
                    "c": int(path[-1]) * 0.25,
                }
            )
    return bars


BARS = _random_market()
RULES: list[Rule] = [
    _rule(),
    _rule(side="long", rangeMinutes=5, targetTicks=0, breakevenTicks=8),
    _rule(side="short", rangeMinutes=30, until=630, stopTicks=6),
    _rule(maxPerDay=3, stopTicks=4, targetTicks=8),
    _rule(entry="time", side="long", at=600),
    _rule(entry="time", side="short", at=630, targetTicks=0),
    _rule(entry="time", side="with", at=630, breakevenTicks=6),
    _rule(entry="time", side="against", at=615, **{"from": 585}),
    _rule(entry="ma", maBars=5, **{"from": 580}),
    _rule(entry="ma", maBars=20, side="long", maxPerDay=4, stopTicks=8, targetTicks=16),
    _rule(entry="ma", maBars=3, side="short", maxPerDay=2, until=640),
    _rule(entry="prior", **{"from": 575}),
    _rule(entry="prior", side="long", maxPerDay=2, stopTicks=8, targetTicks=12),
    _rule(entry="prior", side="short", exitAt=660, until=650),
]


def _bars(*rows: tuple[str, float, float, float, float]) -> list[Bar]:
    return [{"clock": _clock(t), "o": o, "h": h, "l": lo, "c": c} for t, o, h, lo, c in rows]


SMALL = {"tick": 1, "point": 1, "fee": 0.5, "breakevenTicks": 0, "maxPerDay": 1}
MA2 = _rule(**SMALL, entry="ma", maBars=2, until=600, stopTicks=4, targetTicks=1, **{"from": 570})
CASES: list[tuple[str, list[Bar], Rule, list[Trade]]] = [
    (
        # Range 09:30-09:31 is 98 to 102; 09:32 closes at 103, so long at 09:33's open, 103.
        # That bar reaches 105, the 2-tick target, and the entry bar may hit it: entry at open.
        "breakout long hits its target on the entry bar",
        _bars(
            ("2026-05-04 09:30", 100, 101, 99, 100),
            ("2026-05-04 09:31", 100, 102, 98, 101),
            ("2026-05-04 09:32", 101, 103, 101, 103),
            ("2026-05-04 09:33", 103, 105, 102, 104),
        ),
        _rule(**SMALL, rangeStart=570, rangeMinutes=2, until=600, stopTicks=3, targetTicks=2),
        [(_clock("2026-05-04 09:33"), 1, 103, 105, "target", 2.0, 1.5, _clock("2026-05-04 09:33"))],
    ),
    (
        # The 09:33 bar reaches both 100 (stop) and 106 (target): the stop counts.
        "stop first when a bar reaches both",
        _bars(
            ("2026-05-04 09:30", 100, 101, 99, 100),
            ("2026-05-04 09:31", 100, 102, 98, 101),
            ("2026-05-04 09:32", 101, 103, 101, 103),
            ("2026-05-04 09:33", 103, 106, 100, 104),
        ),
        _rule(**SMALL, rangeStart=570, rangeMinutes=2, until=600, stopTicks=3, targetTicks=3),
        [(_clock("2026-05-04 09:33"), 1, 103, 100, "stop", -3.0, -3.5, _clock("2026-05-04 09:33"))],
    ),
    (
        # The previous day ranged 90 to 110. 09:31 closes at 89, below it: short at 09:32's
        # open, 89. Nothing is hit, so the trade closes at the 09:33 close, 87, at exit time.
        "prior-day break short, closed at the exit time",
        _bars(
            ("2026-05-04 09:30", 100, 110, 95, 100),
            ("2026-05-04 09:31", 100, 104, 90, 92),
            ("2026-05-05 09:30", 95, 96, 91, 92),
            ("2026-05-05 09:31", 92, 92, 88, 89),
            ("2026-05-05 09:32", 89, 90, 87, 88),
            ("2026-05-05 09:33", 88, 89, 86, 87),
            ("2026-05-05 09:34", 87, 88, 70, 71),
        ),
        _rule(**SMALL, entry="prior", until=600, exitAt=573, stopTicks=5, targetTicks=0),
        [(_clock("2026-05-05 09:32"), -1, 89, 87, "time", 2.0, 1.5, _clock("2026-05-05 09:33"))],
    ),
    (
        # 2-bar average: closes 10, 10, 10, 12. At 09:32 the close equals its average (10), so
        # no cross. At 09:33 the close 12 is above its average 11, and the 09:32 close was at
        # its average 10: a cross up. Long at 09:34's open, 12; 09:34 reaches 13, the target.
        "moving-average cross up",
        _bars(
            ("2026-05-04 09:30", 10, 10, 10, 10),
            ("2026-05-04 09:31", 10, 10, 10, 10),
            ("2026-05-04 09:32", 10, 10, 10, 10),
            ("2026-05-04 09:33", 10, 12, 10, 12),
            ("2026-05-04 09:34", 12, 13, 11, 13),
        ),
        MA2,
        [(_clock("2026-05-04 09:34"), 1, 12, 13, "target", 1.0, 0.5, _clock("2026-05-04 09:34"))],
    ),
    (
        # Closes 10, 10, 9. At 09:32 the average is 9.5 and the close 9 is below it, while the
        # 09:31 close sat on its average 10: a cross down. Short at 09:33's open, 9. The stop
        # is 13, never reached; the 1-tick target, 8, is reached on that bar.
        "moving-average cross down",
        _bars(
            ("2026-05-04 09:30", 10, 10, 10, 10),
            ("2026-05-04 09:31", 10, 10, 10, 10),
            ("2026-05-04 09:32", 10, 10, 9, 9),
            ("2026-05-04 09:33", 9, 9, 7, 8),
        ),
        MA2,
        [(_clock("2026-05-04 09:33"), -1, 9, 8, "target", 1.0, 0.5, _clock("2026-05-04 09:33"))],
    ),
    (
        # No move from 09:30 to 10:30, so "against the move" has no direction: no trade.
        "no move, no fade",
        _bars(("2026-05-04 09:30", 50, 51, 49, 50), ("2026-05-04 10:30", 50, 52, 48, 51)),
        _rule(**SMALL, entry="time", side="against", at=630, stopTicks=5, targetTicks=5),
        [],
    ),
]


def _example_trades() -> list[dict[str, str]]:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index(">", text.index('<script type="text/plain" id="example-csv">')) + 1
    return list(csv.DictReader(io.StringIO(text[start : text.index("</script>", start)].strip())))


# The two strategies docs/build_example.py traded on the example bars.
EXAMPLE_RULES = {
    "ORB": _rule(breakevenTicks=12),
    "Fade": _rule(
        entry="time", side="against", at=630, stopTicks=10, targetTicks=10, breakevenTicks=6
    ),
}
PROBLEMS: list[tuple[Rule, str]] = [
    (_rule(), ""),
    (_rule(entry="spin"), "Pick how the trade starts."),
    (_rule(until=None), "Write each time as hours and minutes, such as 09:30."),
    (_rule(side="with"), "Pick a direction that fits this kind of entry."),
    (_rule(entry="time", side="both"), "Pick a direction that fits this kind of entry."),
    (_rule(rangeMinutes=0), "The opening range must be 1 to 600 minutes."),
    (_rule(until=580), "The last entry time must be after the opening range ends."),
    (_rule(entry="ma", maBars=1), "The moving average must use 2 to 1,000 bars."),
    (_rule(entry="prior", until=560), "The last entry time must not be before the first."),
    (
        _rule(entry="time", side="with", **{"from": 640}),
        "Measure the move from a time before the entry time.",
    ),
    (_rule(exitAt=660), "The exit time must be after the last entry time."),
    (_rule(stopTicks=0), "The stop must be at least 1 tick."),
    (_rule(targetTicks=-1), "Target and breakeven must be 0 or more ticks."),
    (_rule(maxPerDay=0), "Trades a day must be 1 to 50."),
    (_rule(fee=-1), "Pick the instrument and a fee of 0 or more."),
]


# Pairs of rules, and whether they are the same version for counting setups tried.
_TIME_LONG = _rule(entry="time", side="long")
_FADE = _rule(entry="time", side="against")
KEY_PAIRS: list[tuple[Rule, Rule, bool]] = [
    (_rule(), _rule(maBars=50, at=700, fee=9.5, tick=1, point=5), True),  # hidden blanks
    (_rule(), _rule(stopTicks=13), False),
    (_rule(), _rule(rangeMinutes=30), False),
    (_rule(), _rule(side="long"), False),
    (_TIME_LONG, {**_TIME_LONG, "from": 600}, True),  # a fixed direction ignores the move
    (_FADE, {**_FADE, "from": 600}, False),
    (_rule(entry="ma"), _rule(entry="ma", rangeMinutes=5), True),
    (_rule(entry="ma"), _rule(entry="prior"), False),
]


# Strategies written in plain words, the contract's tick size (0 = no contract picked), and
# what the reader must make of them: rule fields, or the start of its first problem.
PHRASES: list[tuple[str, float, dict[str, Any] | str]] = [
    (
        "Go long when price breaks the high of the first 15 minutes after 9:30, "
        "stop 12 ticks, target 24, out by 11:30",
        0.25,
        {
            "entry": "orb",
            "side": "long",
            "rangeMinutes": 15,
            "rangeStart": 570,
            "until": 660,
            "stopTicks": 12,
            "targetTicks": 24,
            "breakevenTicks": 0,
            "exitAt": 690,
            "maxPerDay": 1,
        },
    ),
    (
        "ORB 30 min, 12 tick stop 24 tick target, breakeven after 12 ticks, flat by 11:29am",
        0.25,
        {
            "entry": "orb",
            "side": "both",
            "rangeMinutes": 30,
            "stopTicks": 12,
            "targetTicks": 24,
            "breakevenTicks": 12,
            "exitAt": 689,
        },
    ),
    (
        "Fade the move since 9:30 at 10:30, stop 10 target 10, move stop to entry after 6 ticks",
        0.25,
        {
            "entry": "time",
            "side": "against",
            "at": 630,
            "from": 570,
            "stopTicks": 10,
            "targetTicks": 10,
            "breakevenTicks": 6,
            "exitAt": 1439,
        },
    ),
    (
        "Buy when the close crosses above the 20-bar moving average between 9:45 and 11:00, "
        "stop 8 points target 16 points, up to 3 trades a day",
        0.25,
        {
            "entry": "ma",
            "side": "long",
            "maBars": 20,
            "from": 585,
            "until": 660,
            "stopTicks": 32,
            "targetTicks": 64,
            "maxPerDay": 3,
        },
    ),
    (
        "Short when price breaks yesterday's low, stop 20 ticks, no target, out by 15:00",
        0.25,
        {
            "entry": "prior",
            "side": "short",
            "stopTicks": 20,
            "targetTicks": 0,
            "exitAt": 900,
            "from": 0,
            "until": 899,
        },
    ),
    (
        "Go long at 9:45, 12 tick stop, target 24 ticks, out by 10:45, breakeven 8",
        0.25,
        {
            "entry": "time",
            "side": "long",
            "at": 585,
            "stopTicks": 12,
            "targetTicks": 24,
            "exitAt": 645,
            "breakevenTicks": 8,
        },
    ),
    (
        "target 24 ticks stop 12 ticks, ORB of 5 min long or short, 2 trades a day",
        0.25,
        {
            "entry": "orb",
            "side": "both",
            "rangeMinutes": 5,
            "stopTicks": 12,
            "targetTicks": 24,
            "maxPerDay": 2,
        },
    ),
    ("Sell at 2pm", 0.25, "Add a stop"),
    ("Trade the news, stop 10", 0.25, "I could not tell how a trade starts"),
    ("Fade the opening range breakout, stop 10", 0.25, "Fading a signal is not supported"),
    ("At 10:30, stop 10 ticks", 0.25, "Say what to do at that time"),
    ("Buy the first 15 minute breakout, stop 3 points", 0, "Pick the contract"),
]


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "lab", "build")
    )
    payload = {
        "bars": BARS,
        "rules": RULES,
        "cases": [{"bars": c[1], "rule": c[2]} for c in CASES],
        "exampleBars": (ROOT / "docs" / "example-bars.csv").read_text(encoding="utf-8"),
        "exampleRules": list(EXAMPLE_RULES.values()),
        "problems": [p[0] for p in PROBLEMS],
        "clocks": ["09:30", "9:05", "23:59", "24:00", "09:60", "930", ""],
        "sentences": [EXAMPLE_RULES["ORB"], EXAMPLE_RULES["Fade"]],
        "keyRules": [rule for pair in KEY_PAIRS for rule in pair[:2]],
        "phrases": [[p[0], p[1]] for p in PHRASES],
    }
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "build.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)],
            capture_output=True,
            encoding="utf-8",
            check=True,
            timeout=120,
        )
    out: dict[str, Any] = json.loads(done.stdout)
    return out


def _as_tuples(rows: list[list[Any]]) -> list[Trade]:
    return [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows]


def test_matches_the_python_version(js: dict[str, Any]) -> None:
    total = 0
    for rule, got in zip(RULES, js["random"], strict=True):
        want = reference_trades(BARS, rule)
        assert _as_tuples(got) == want, rule
        total += len(want)
    # Every kind of rule took trades, so the comparison covered each one.
    for kind in ("orb", "time", "ma", "prior"):
        trades = [g for r, g in zip(RULES, js["random"], strict=True) if r["entry"] == kind]
        assert sum(len(t) for t in trades) >= 20, kind
    assert total > 400
    multi = js["random"][3]  # up to 3 trades a day
    days = [t[0] // DAY_MS for t in multi]
    assert max(days.count(d) for d in set(days)) == 3
    # A later trade on the same day starts only after the earlier one closed.
    for a, b in itertools.pairwise(multi):
        if a[0] // DAY_MS == b[0] // DAY_MS:
            assert b[0] > a[7]


def test_hand_worked_cases(js: dict[str, Any]) -> None:
    for (name, _, _, want), got in zip(CASES, js["cases"], strict=True):
        assert _as_tuples(got) == want, name


@pytest.mark.parametrize("strategy", ["ORB", "Fade"])
def test_reproduces_the_example_strategies(js: dict[str, Any], strategy: str) -> None:
    got = js["example"][list(EXAMPLE_RULES).index(strategy)]
    rows = [r for r in _example_trades() if r["Strategy"] == strategy]
    assert len(got) == len(rows) > 100
    for t, row in zip(got, rows, strict=True):
        side = 1 if row["Side"] == "Long" else -1
        entry_time = _clock(row["Entry time"][:16])
        assert (t[0], t[1], t[2]) == (entry_time, side, float(row["Entry price"]))
        assert t[3] == float(row["Exit price"])
        assert t[5] == pytest.approx(float(row["P&L"]), abs=1e-9)
        assert t[6] == pytest.approx(float(row["P&L"]) - float(row["Commission"]), abs=1e-9)
        assert t[7] == _clock(row["Exit time"][:16])  # the exit bar


def test_rule_problems(js: dict[str, Any]) -> None:
    assert js["problems"] == [p[1] for p in PROBLEMS]
    assert js["clocks"][:3] == [570, 545, 1439]
    assert all(v is None for v in js["clocks"][3:])  # NaN arrives as null


def test_versions_count_only_the_blanks_in_use(js: dict[str, Any]) -> None:
    keys = js["keys"]
    for i, (_, _, same) in enumerate(KEY_PAIRS):
        assert (keys[2 * i] == keys[2 * i + 1]) is same, KEY_PAIRS[i]


def test_sentences(js: dict[str, Any]) -> None:
    assert js["sentences"] == [
        "Long or short when a bar closes beyond the range of the first 15 minutes after 09:30, "
        "until 11:00; stop 12, target 24, breakeven at 12 (ticks); out by 11:29.",
        "Against the move at 10:30 (move since 09:30); stop 10, target 10, breakeven at 6 "
        "(ticks); out by 11:29.",
    ]


def test_reads_strategies_written_in_plain_words(js: dict[str, Any]) -> None:
    for (text, _, want), got in zip(PHRASES, js["parsed"], strict=True):
        if isinstance(want, str):
            assert got["problems"], text
            assert got["problems"][0].startswith(want), (text, got["problems"])
            continue
        assert got["problems"] == [], (text, got["problems"])
        assert got["problem"] == "", (text, got["problem"])  # the engine accepts the rule
        for key, value in want.items():
            assert got["rule"][key] == value, (text, key, got["rule"])


def test_says_what_it_assumed(js: dict[str, Any]) -> None:
    first = js["parsed"][0]
    # Said: direction, range, start, stop, target, exit. Not said: entries until, one a day.
    assert "long only" in first["read"]
    assert "entries until 11:00" in first["assumed"]
    assert "one trade a day" in first["assumed"]
    fade = js["parsed"][2]
    assert "out at the end of each day in your bars" in fade["assumed"]
    assert "move measured from 09:30" in fade["read"]
