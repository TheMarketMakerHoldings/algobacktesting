"""The website's timing breakdown must group trades correctly and score slices honestly.

``docs/index.html`` splits a strategy's trades by hour or weekday. This test runs the page's
code (between ``/* when:start */`` and ``/* when:end */``) in Node.js and checks the grouping
on hand-made times and each slice's chance against the Python library's Deflated Sharpe Ratio,
with every slice that has enough trades counted as a try. Skipped without Node.js.
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from algobacktesting import deflated_sharpe_ratio, return_stats

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_when_harness.js"
NODE = shutil.which("node")


def _ms(*parts: int) -> int:
    return calendar.timegm((*parts, 0, 0, 0)) * 1000


# Monday 1 June 2026 09:30 and 09:59, Monday 10:15, Sunday 7 June 16:00, and no time.
TIMES = [
    _ms(2026, 6, 1, 9, 30, 0),
    _ms(2026, 6, 1, 9, 59, 0),
    _ms(2026, 6, 1, 10, 15, 0),
    _ms(2026, 6, 7, 16, 0, 0),
    None,
]
VALUES = [10.0, -5.0, 7.0, 3.0, 99.0]

RNG = np.random.default_rng(42)
DAYS = [_ms(2026, 6, 1 + (i % 25), 9 + (i % 4), (7 * i) % 60, 0) for i in range(240)]
RESULTS = [round(float(x), 2) for x in RNG.normal(5, 60, 240)]
SLICES: list[dict[str, Any]] = [
    {"values": RESULTS, "times": DAYS, "by": "hour", "setups": 3, "minTrades": 5},
    {"values": RESULTS, "times": DAYS, "by": "weekday", "setups": 1, "minTrades": 5},
]


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "when")
    )
    payload = {
        "groups": [
            {"values": VALUES, "times": TIMES, "by": "hour"},
            {"values": VALUES, "times": TIMES, "by": "weekday"},
        ],
        "slices": SLICES,
    }
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "when.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)], capture_output=True, text=True, check=True, timeout=60
        )
    out: dict[str, Any] = json.loads(done.stdout)
    return out


def test_grouping_by_hour_and_weekday(js: dict[str, Any]) -> None:
    by_hour, by_day = js["groups"]
    assert by_hour == {"9": [10.0, -5.0], "10": [7.0], "16": [3.0]}  # no time: left out
    assert by_day == {"0": [10.0, -5.0, 7.0], "6": [3.0]}  # Monday is 0, Sunday 6


def _dsr(values: list[float], trials: int) -> float:
    s = return_stats(values)
    return deflated_sharpe_ratio(
        s.sharpe,
        s.n_obs,
        n_trials=trials,
        sharpe_variance=1 / (s.n_obs - 1),
        skew=s.skew,
        kurtosis=s.kurtosis,
    )


@pytest.mark.parametrize("which", [0, 1])
def test_slices_match_the_library(js: dict[str, Any], which: int) -> None:
    spec, rep = SLICES[which], js["slices"][which]
    groups: dict[int, list[float]] = {}
    for v, t in zip(spec["values"], spec["times"], strict=True):
        moment = dt.datetime.fromtimestamp(t / 1000, tz=dt.UTC)
        key = moment.hour if spec["by"] == "hour" else moment.weekday()
        groups.setdefault(key, []).append(v)
    eligible = sum(1 for g in groups.values() if len(g) >= spec["minTrades"])
    assert rep["slices"] == eligible
    assert rep["trials"] == eligible * spec["setups"]
    assert [r["key"] for r in rep["rows"]] == sorted(groups)
    for row in rep["rows"]:
        vals = groups[row["key"]]
        assert row["n"] == len(vals)
        assert row["total"] == pytest.approx(sum(vals))
        assert row["chance"] == pytest.approx(_dsr(vals, rep["trials"]), abs=1e-9)
        assert row["lossChance"] == pytest.approx(_dsr([-v for v in vals], rep["trials"]), abs=1e-9)
