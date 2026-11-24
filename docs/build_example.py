"""Generate the website's example: a synthetic market and two strategies traded on it.

Run from the repository root::

    uv run python docs/build_example.py

It writes ``docs/example-bars.csv`` (one-minute bars, 09:30 to 11:29 New York time, laid out
like a Quantower export) and replaces the example trade file embedded in ``docs/index.html``.
Nothing here is real market data.

The market is a random walk, so no strategy traded on it has a real edge. That makes it a fair
test of the page: whatever the example strategies show, the honest answer is "luck".

- **ORB**: opening-range breakout. The range is 09:30 to 09:44. The first one-minute close
  outside it, at or before 11:00, enters at the next bar's open. Bracket: 12-tick stop,
  24-tick target; once a bar reaches 12 ticks in profit the stop moves to the entry price.
  Exit at the 11:29 close.
- **Fade**: at the 10:30 open, trade against the move since 09:30. Bracket: 10-tick stop,
  10-tick target; the stop moves to the entry price once a bar reaches 6 ticks in profit.
  Same time exit. Moving the stop to entry is what produces breakeven trades.

Both trade one ES contract ($50 a point, 0.25 a tick) and pay $4.12 per round trip. When a bar
touches both the stop and the target, the stop counts first. ``tests/test_site_lab.py`` checks
that the page's stop lab replays every ORB trade to the same result.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np

DOCS = Path(__file__).resolve().parent
SEED = 4
TICK, POINT, COMMISSION = 0.25, 50.0, 4.12
FIRST_DAY, TRADING_DAYS = date(2026, 6, 1), 153
OPEN_MINUTE, BARS_PER_DAY = 9 * 60 + 30, 120  # 09:30 to 11:29
TRADE_HEADER = (
    "Trade #,Strategy,Instrument,Side,Qty,Entry price,Exit price,"
    "Entry time,Exit time,P&L,Commission"
)


class Bar(NamedTuple):
    """One minute of prices, in ticks."""

    time: datetime
    offset: str
    open: int
    high: int
    low: int
    close: int
    volume: int


class Rule(NamedTuple):
    """An entry and its bracket."""

    name: str
    start: int
    side: int
    stop: int
    target: int
    breakeven_at: int


def _days() -> list[date]:
    days, d = [], FIRST_DAY
    while len(days) < TRADING_DAYS:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _utc_offset(d: date) -> str:
    """Return New York's offset from UTC (daylight saving ends the first Sunday of November)."""
    nov1 = date(d.year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return "-04:00" if d < dst_end else "-05:00"


def _market(rng: np.random.Generator) -> list[Bar]:
    """Each minute is six random sub-steps on the tick grid; each day opens with a gap."""
    bars: list[Bar] = []
    price = int(5900 / TICK)
    for d in _days():
        price += round(rng.normal(0, 40))
        for k in range(BARS_PER_DAY):
            steps = np.rint(rng.normal(0, 1.6, 6)).astype(int)
            path = price + np.concatenate(([0], np.cumsum(steps)))
            start = datetime(d.year, d.month, d.day) + timedelta(minutes=OPEN_MINUTE + k)
            bars.append(
                Bar(
                    time=start,
                    offset=_utc_offset(d),
                    open=int(path[0]),
                    high=int(path.max()),
                    low=int(path.min()),
                    close=int(path[-1]),
                    volume=int(rng.integers(300, 4000)),
                )
            )
            price = int(path[-1])
    return bars


def _bracket(day: list[Bar], rule: Rule) -> tuple[int, int, int]:
    """Ticks won (negative if lost), the exit bar, and the seconds into it."""
    side, entry = rule.side, day[rule.start].open
    stop_px, moved = entry - side * rule.stop, False
    for j in range(rule.start, len(day)):
        bar = day[j]
        if (side > 0 and bar.low <= stop_px) or (side < 0 and bar.high >= stop_px):
            return side * (stop_px - entry), j, 30
        if (side > 0 and bar.high >= entry + rule.target) or (
            side < 0 and bar.low <= entry - rule.target
        ):
            return rule.target, j, 30
        best = bar.high - entry if side > 0 else entry - bar.low
        if rule.breakeven_at and not moved and best >= rule.breakeven_at:
            stop_px, moved = entry, True
    return side * (day[-1].close - entry), len(day) - 1, 60


def _rules(day: list[Bar]) -> list[Rule]:
    rules = []
    high = max(b.high for b in day[:15])
    low = min(b.low for b in day[:15])
    for k in range(15, 91):  # signal bars 09:45 to 11:00; entry at the next bar's open
        if day[k].close > high or day[k].close < low:
            rules.append(Rule("ORB", k + 1, 1 if day[k].close > high else -1, 12, 24, 12))
            break
    drift = day[60].open - day[0].open
    if drift:
        rules.append(Rule("Fade", 60, -1 if drift > 0 else 1, 10, 10, 6))
    return sorted(rules, key=lambda r: r.start)


def _trades(bars: list[Bar]) -> list[str]:
    lines = []
    for first in range(0, len(bars), BARS_PER_DAY):
        day = bars[first : first + BARS_PER_DAY]
        for rule in _rules(day):
            ticks, j, seconds = _bracket(day, rule)
            entry = day[rule.start].open
            t_in, t_out = day[rule.start].time, day[j].time + timedelta(seconds=seconds)
            contract = "ESU6" if t_in < datetime(2026, 9, 18) else "ESZ6"
            cells = [
                str(len(lines) + 1),
                rule.name,
                contract,
                "Long" if rule.side > 0 else "Short",
                "1",
                f"{entry * TICK:.2f}",
                f"{(entry + rule.side * ticks) * TICK:.2f}",
                f"{t_in:%Y-%m-%d %H:%M:%S}",
                f"{t_out:%Y-%m-%d %H:%M:%S}",
                f"{ticks * TICK * POINT:.2f}",
                f"{COMMISSION:.2f}",
            ]
            lines.append(",".join(cells))
    return lines


def _stamp(t: datetime) -> str:
    """Quantower's style: 6/1/2026 9:30:00 AM."""
    hour = (t.hour - 1) % 12 + 1
    return f"{t.month}/{t.day}/{t.year} {hour}:{t.minute:02d}:00 {'AM' if t.hour < 12 else 'PM'}"


def build() -> tuple[str, str]:
    """Return the bar file and the trade file as text."""
    bars = _market(np.random.default_rng(SEED))
    bar_lines = ["DateTime,UTC,Open,High,Low,Close,Volume"]
    for b in bars:
        prices = ",".join(f"{v * TICK:g}" for v in (b.open, b.high, b.low, b.close))
        bar_lines.append(f"{_stamp(b.time)},{b.offset},{prices},{b.volume}")
    trade_lines = [TRADE_HEADER, *_trades(bars)]
    return "\n".join(bar_lines) + "\n", "\n".join(trade_lines) + "\n"


def main() -> None:
    """Write the bar file and put the trade file into the page."""
    bar_text, trade_text = build()
    (DOCS / "example-bars.csv").write_text(bar_text, encoding="utf-8", newline="\n")
    page = DOCS / "index.html"
    html = page.read_text(encoding="utf-8")
    block = re.compile(r'(<script type="text/plain" id="example-csv">\n).*?(</script>)', re.DOTALL)
    assert block.search(html), "example block not found in docs/index.html"
    html = block.sub(lambda m: m.group(1) + trade_text + m.group(2), html)
    page.write_text(html, encoding="utf-8", newline="\n")
    print(f"{bar_text.count(chr(10)) - 1} bars, {trade_text.count(chr(10)) - 1} trades")


if __name__ == "__main__":
    main()
