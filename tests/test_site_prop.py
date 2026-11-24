"""The website's prop-firm simulator must follow its rules exactly.

``docs/index.html`` replays a trader's results in random order under an evaluation's rules
(profit target, loss limit, daily limit). This test runs the page's simulator (between
``/* prop:start */`` and ``/* prop:end */``) in Node.js and checks it against:

1. Exact answers: the gambler's ruin probabilities for fair and biased +-1 trades with a
   fixed loss limit, and hand-worked cases.
2. Orderings that must hold run by run, because each run sees the same trades in the same
   order under every rule set: a fixed limit is never harder to pass than one that trails the
   end-of-day balance, which is never harder than one that trails every trade.

Skipped without Node.js.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_prop_harness.js"
NODE = shutil.which("node")
RUNS = 20_000

STATIC = {"maxLoss": 20, "target": 30, "drawdown": "static", "dailyLoss": 0, "maxDays": 0}
MIXED = [120.0, -80.0, 45.0, -60.0, 200.0, -75.0, -40.0, 90.0, -110.0, 30.0, 15.0, -55.0]
RULES = {"target": 600, "maxLoss": 400, "dailyLoss": 0, "dailyLossFails": False, "maxDays": 0}

SIMS: dict[str, dict[str, Any]] = {
    "fair coin": {
        "values": [1, -1],
        "rules": STATIC,
        "opts": {"runs": RUNS, "seed": 11, "tradesPerDay": 1},
    },
    "biased coin": {
        "values": [1, 1, -1],
        "rules": {**STATIC, "maxLoss": 3, "target": 5},
        "opts": {"runs": RUNS, "seed": 12, "tradesPerDay": 1},
    },
    "all wins": {
        "values": [100] * 5,
        "rules": {**STATIC, "maxLoss": 2000, "target": 3000},
        "opts": {"runs": 200, "seed": 1, "tradesPerDay": 3},
    },
    "all losses": {
        "values": [-100] * 5,
        "rules": {**STATIC, "maxLoss": 2000, "target": 3000},
        "opts": {"runs": 200, "seed": 1, "tradesPerDay": 3},
    },
    "zero edge": {
        "values": [100] * 5,
        "rules": {**STATIC, "maxLoss": 2000, "target": 3000, "maxDays": 30},
        "opts": {"runs": 200, "seed": 1, "tradesPerDay": 3, "zeroEdge": True},
    },
    "daily limit ends the day": {
        "values": [-100],
        "rules": {**STATIC, "maxLoss": 1000, "target": 3000, "dailyLoss": 150, "maxDays": 3},
        "opts": {"runs": 50, "seed": 1, "tradesPerDay": 2},
    },
    "daily limit fails": {
        "values": [-100],
        "rules": {
            **STATIC,
            "maxLoss": 1000,
            "target": 3000,
            "dailyLoss": 150,
            "dailyLossFails": True,
            "maxDays": 3,
        },
        "opts": {"runs": 50, "seed": 1, "tradesPerDay": 2},
    },
}
for kind in ("static", "eod", "intraday"):
    SIMS[f"order {kind}"] = {
        "values": MIXED,
        "rules": {**RULES, "drawdown": kind},
        "opts": {"runs": 4000, "seed": 5, "perDay": [1, 2, 3, 4]},
    }
for fails in (False, True):
    SIMS[f"order daily {fails}"] = {
        "values": MIXED,
        "rules": {**RULES, "drawdown": "eod", "dailyLoss": 150, "dailyLossFails": fails},
        "opts": {"runs": 4000, "seed": 5, "perDay": [1, 2, 3, 4]},
    }
DAY = 86_400_000
PER_DAY = [[0, 1000, DAY + 5, DAY + 6, DAY + 7, 3 * DAY, None], []]


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "prop")
    )
    payload = {"sims": list(SIMS.values()), "perDay": PER_DAY}
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "prop.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)], capture_output=True, text=True, check=True, timeout=120
        )
    out: dict[str, Any] = json.loads(done.stdout)
    out["by_name"] = dict(zip(SIMS, out["sims"], strict=True))
    return out


def _within(got: float, exact: float, runs: int) -> None:
    # Four standard errors of a proportion: a false failure about 1 time in 16,000.
    tolerance = 4 * math.sqrt(exact * (1 - exact) / runs)
    assert abs(got - exact) < tolerance, (got, exact, tolerance)


def test_gamblers_ruin_fair_coin(js: dict[str, Any]) -> None:
    # Fair +-1 steps from 0, absorbed at -20 or +30: P(reach +30 first) = 20 / (20 + 30).
    r = js["by_name"]["fair coin"]
    _within(r["pass"], 20 / 50, RUNS)
    assert r["open"] == 0
    assert r["pass"] + r["fail"] == pytest.approx(1.0)


def test_gamblers_ruin_biased_coin(js: dict[str, Any]) -> None:
    # Up with p = 2/3: P = (1 - r**3) / (1 - r**8) with r = q / p = 1/2.
    r = js["by_name"]["biased coin"]
    _within(r["pass"], (1 - 0.5**3) / (1 - 0.5**8), RUNS)


def test_hand_worked_cases(js: dict[str, Any]) -> None:
    by = js["by_name"]
    # +100 per trade, 3 trades a day: the target of 3,000 is reached on trade 30, day 10.
    assert (by["all wins"]["pass"], by["all wins"]["medianDays"]) == (1.0, 10)
    assert by["all losses"]["fail"] == 1.0
    # With the edge removed every result is 0: nothing can pass or fail in 30 days.
    assert by["zero edge"]["open"] == 1.0
    # -100 twice a day: the 150 daily limit is hit on each day's second trade. If that only
    # ends the day, 3 days cost 600 and the 1,000 limit still holds; if it fails, day 1 fails.
    assert by["daily limit ends the day"]["open"] == 1.0
    assert by["daily limit fails"]["fail"] == 1.0


def test_stricter_rules_never_pass_more_often(js: dict[str, Any]) -> None:
    by = js["by_name"]
    static, eod, intraday = (by[f"order {k}"] for k in ("static", "eod", "intraday"))
    assert static["pass"] >= eod["pass"] >= intraday["pass"]
    assert static["fail"] <= eod["fail"] <= intraday["fail"]
    assert intraday["pass"] < static["pass"]  # the rules really differ on this data
    assert by["order daily False"]["pass"] >= by["order daily True"]["pass"]


def test_trades_per_day_from_times(js: dict[str, Any]) -> None:
    assert sorted(js["perDay"][0]) == [1, 2, 3]  # unknown times are skipped
    assert js["perDay"][1] == []
