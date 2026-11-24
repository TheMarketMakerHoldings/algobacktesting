"""The website must save, read back and compare named strategies without changing them.

``docs/index.html`` keeps saved strategies in the browser's own storage (localStorage) as
JSON text. This test runs the page's code (between ``/* saved:start */`` and
``/* saved:end */``, with the statistics it uses) in Node.js and checks:

1. Names are cleaned and capped, and bad input is refused with a clear message.
2. Adding, replacing and deleting keep the list consistent.
3. Text read back from storage is never trusted: malformed entries are dropped.
4. A strategy saved and read back gives the same statistics and the same comparison as the
   trades it came from, to within the 6-decimal rounding the page applies.

Skipped without Node.js.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_saved_harness.js"
NODE = shutil.which("node")
NOW = 1_795_600_000_000  # a fixed clock in milliseconds, so ids and dates are deterministic

RNG = np.random.default_rng(2460552)
TEN = [10.0, -5.0, 2.5, -1.25, 7.0, -3.0, 4.0, -2.0, 1.0, -0.5]
# Awkward cents, as fees produce: 12.299999999999999 and similar.
TRADES_A = (RNG.normal(12, 90, 500) - 4.6).tolist()
TRADES_B = (RNG.normal(-3, 70, 320) - 4.6).tolist()
EMOJI = chr(0x1F600)


def _item(i: int, name: str | None = None) -> dict[str, Any]:
    return {"id": f"s{i}", "name": name or f"Strategy {i}", "values": TEN, "unit": "$", "setups": 1}


def _stored(**changes: Any) -> dict[str, Any]:
    entry = {
        "id": "abc",
        "name": "ORB",
        "values": TEN,
        "unit": "$",
        "setups": 3,
        "savedAt": "2026-11-24T23:00:00.000Z",
    }
    entry.update(changes)
    return entry


def _text(entries: list[Any], version: int = 1) -> str:
    return json.dumps({"version": version, "strategies": entries})


FULL = [_item(i) for i in range(50)]
CALLS: dict[str, tuple[str, list[Any]]] = {
    "name_spaces": ("cleanName", ["  NQ   opening\trange  v2 "]),
    "name_control": ("cleanName", [f"a{chr(0)}b{chr(31)}c{chr(127)}d{chr(150)}e"]),
    "name_long": ("cleanName", ["x" * 75]),
    "name_emoji_edge": ("cleanName", ["x" * 59 + EMOJI + EMOJI]),
    "name_markup": ("cleanName", ["<b>ORB</b> & fade"]),
    "name_none": ("cleanName", [None]),
    "make_ok": ("makeSaved", ["  ORB  ", [1.23456789, -0.0000001, *TEN[2:]], "$", 4.6, NOW]),
    "make_no_name": ("makeSaved", ["   ", TEN, "$", 1, NOW]),
    "make_nine": ("makeSaved", ["ORB", TEN[:9], "$", 1, NOW]),
    "make_nan": ("makeSaved", ["ORB", [*TEN[:9], "x"], "$", 1, NOW]),
    "make_unit_pts": ("makeSaved", ["ORB", TEN, "pts", 0, NOW]),
    "make_unit_r": ("makeSaved", ["ORB", TEN, "R", 99999, NOW]),
    "add_new": ("addSaved", [[_item(1)], _item(2)]),
    "add_same_name": (
        "addSaved",
        [[_item(1, "ORB"), _item(2, "Fade")], {**_item(9, "orb"), "setups": 7}],
    ),
    "add_full": ("addSaved", [FULL, _item(99)]),
    "add_full_replace": ("addSaved", [FULL, {**_item(99, "strategy 3"), "setups": 2}]),
    "add_same_id": (
        "addSaved",
        [[_item(1), {**_item(1), "id": "s1-1", "name": "B"}], {**_item(1), "name": "C"}],
    ),
    "remove": ("removeSaved", [[_item(1), _item(2), _item(3)], "s2"]),
    "remove_missing": ("removeSaved", [[_item(1)], "nope"]),
    "by_name": ("savedByName", [[_item(1, "NQ ORB")], "  nq   orb "]),
    "parse_garbage": ("parseSaved", ["not json at all"]),
    "parse_null": ("parseSaved", [None]),
    "parse_empty_object": ("parseSaved", ["{}"]),
    "parse_wrong_version": ("parseSaved", [_text([_stored()], version=2)]),
    "parse_good": ("parseSaved", [_text([_stored()])]),
    "parse_bad_entries": (
        "parseSaved",
        [
            _text(
                [
                    None,
                    5,
                    "text",
                    [],
                    _stored(id="../etc"),
                    _stored(id="ABC"),
                    _stored(id="x" * 41),
                    _stored(id=7),
                    _stored(id="ok1", name="   "),
                    _stored(id="ok2", values=TEN[:9]),
                    _stored(id="ok3", values=[*TEN[:9], "5"]),
                    _stored(id="ok4", values=[*TEN[:9], None]),
                    _stored(id="ok5", values="1,2,3"),
                    _stored(id="keep1", name="First"),
                    _stored(id="keep1", name="Same id"),
                    _stored(id="keep2", name="FIRST"),
                    _stored(
                        id="keep3",
                        name="Second",
                        unit="<script>",
                        setups="lots",
                        savedAt="yesterday",
                    ),
                ]
            )
        ],
    ),
    "parse_infinity": (
        "parseSaved",
        ['{"version":1,"strategies":[{"id":"a","name":"A","values":[1e400,1,2,3,4,5,6,7,8,9]}]}'],
    ),
    "parse_too_many": (
        "parseSaved",
        [_text([_stored(id=f"i{i}", name=f"S{i}") for i in range(60)])],
    ),
    "same_yes": ("sameTrades", [TEN, list(TEN)]),
    "same_rounding": ("sameTrades", [TEN, [TEN[0] + 5e-7, *TEN[1:]]]),
    "same_no": ("sameTrades", [TEN, [TEN[0] + 2e-6, *TEN[1:]]]),
    "same_length": ("sameTrades", [TEN, TEN[:9]]),
}
# The second is saved on the same clock tick as the first, so their ids must not clash; the
# third has the first one's name in other capitals, so it replaces it.
ROUNDTRIP = [
    {"name": "NQ opening range", "values": TRADES_A, "unit": "$", "setups": 5, "now": NOW},
    {"name": "NQ fade", "values": TRADES_B, "unit": "$", "setups": 2, "now": NOW},
    {"name": "nq OPENING range", "values": TRADES_A, "unit": "$", "setups": 6, "now": NOW + 1},
]


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "compare", "saved")
    )
    calls = [{"fn": fn, "args": args} for fn, args in CALLS.values()]
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "saved.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps({"calls": calls, "roundtrip": ROUNDTRIP}), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)],
            capture_output=True,
            encoding="utf-8",
            check=True,
            timeout=60,  # Node writes UTF-8
        )
    out = json.loads(done.stdout)
    result: dict[str, Any] = dict(zip(CALLS, out["calls"], strict=True))
    result["roundtrip"] = out["roundtrip"]
    return result


def _base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out


def test_names_are_cleaned_and_capped(js: dict[str, Any]) -> None:
    assert js["name_spaces"] == "NQ opening range v2"
    assert js["name_control"] == "a b c d e"
    assert js["name_long"] == "x" * 60
    # 60 whole characters: the first emoji is kept, never half of one.
    assert js["name_emoji_edge"] == "x" * 59 + EMOJI
    # Markup stays as plain text; the page shows names with textContent, never as HTML.
    assert js["name_markup"] == "<b>ORB</b> & fade"
    assert js["name_none"] == ""


def test_making_a_saved_strategy(js: dict[str, Any]) -> None:
    item = js["make_ok"]["item"]
    assert item["name"] == "ORB"
    assert item["id"] == "s" + _base36(NOW)
    stamp = datetime.fromtimestamp(NOW / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    assert item["savedAt"] == stamp
    assert item["unit"] == "$"
    assert item["setups"] == 5  # 4.6 rounds to a whole number of setups
    assert item["values"][:2] == [1.234568, 0]  # 6 decimals; -0.0000001 rounds to plain 0
    assert item["values"][2:] == TEN[2:]
    assert js["make_no_name"] == {"error": "Type a name for these trades first."}
    assert js["make_nine"] == {"error": "At least 10 trades are needed to save a strategy."}
    assert js["make_nan"] == {"error": "Every trade needs a number to be saved."}
    assert js["make_unit_pts"]["item"]["unit"] == ""  # only $ and R are kept as units
    assert js["make_unit_pts"]["item"]["setups"] == 1
    assert js["make_unit_r"]["item"]["unit"] == "R"
    assert js["make_unit_r"]["item"]["setups"] == 10000


def test_adding_replacing_and_removing(js: dict[str, Any]) -> None:
    assert [s["id"] for s in js["add_new"]["list"]] == ["s1", "s2"]
    assert js["add_new"]["replaced"] is False
    same = js["add_same_name"]
    assert same["replaced"] is True
    assert [(s["id"], s["name"]) for s in same["list"]] == [("s1", "orb"), ("s2", "Fade")]
    assert same["list"][0]["setups"] == 7  # the new trades and settings, under the old id
    assert js["add_full"] == {"error": "This page keeps up to 50 strategies. Delete one first."}
    full_replace = js["add_full_replace"]
    assert full_replace["replaced"] is True
    assert len(full_replace["list"]) == 50
    assert full_replace["list"][3]["setups"] == 2
    assert [s["id"] for s in js["add_same_id"]["list"]] == ["s1", "s1-1", "s1-2"]
    assert [s["id"] for s in js["remove"]] == ["s1", "s3"]
    assert [s["id"] for s in js["remove_missing"]] == ["s1"]
    assert js["by_name"]["id"] == "s1"


def test_stored_text_is_never_trusted(js: dict[str, Any]) -> None:
    for key in ("parse_garbage", "parse_null", "parse_empty_object", "parse_wrong_version"):
        assert js[key] == [], key
    assert js["parse_good"] == [_stored()]
    kept = js["parse_bad_entries"]
    # Only the first of a repeated id or name survives, and odd fields fall back to safe values.
    assert [(s["id"], s["name"]) for s in kept] == [("keep1", "First"), ("keep3", "Second")]
    assert kept[1]["unit"] == ""
    assert kept[1]["setups"] == 1
    assert kept[1]["savedAt"] == ""
    assert js["parse_infinity"] == []  # 1e400 reads as Infinity, which is not a trade result
    assert len(js["parse_too_many"]) == 50


def test_same_trades(js: dict[str, Any]) -> None:
    assert js["same_yes"] is True
    assert js["same_rounding"] is True
    assert js["same_no"] is False
    assert js["same_length"] is False


def test_saved_strategies_give_the_same_answers(js: dict[str, Any]) -> None:
    trip = js["roundtrip"]
    assert trip["back"] == trip["list"]  # read back exactly as written
    assert [s["name"] for s in trip["list"]] == ["nq OPENING range", "NQ fade"]
    assert [s["setups"] for s in trip["list"]] == [6, 2]
    first, second = trip["list"][0]["id"], trip["list"][1]["id"]
    assert first == "s" + _base36(NOW)
    assert second == first + "-1"
    a, b = np.array(TRADES_A), np.array(TRADES_B)
    # Each stored trade is within 0.0000005 of the original, so the average, the standard
    # deviation and the difference in averages move by at most that much. With a standard
    # error near 6, the chance A is better moves by under 0.4 * 1e-6 / 6: 1e-6 is ample.
    for got, x in zip(trip["stats"], (a, b), strict=True):
        assert got["mean"] == pytest.approx(x.mean(), abs=1e-6)
        assert got["sd"] == pytest.approx(x.std(ddof=1), abs=1e-6)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = float(a.mean() - b.mean())
    assert trip["compare"]["diff"] == pytest.approx(diff, abs=1e-6)
    assert trip["compare"]["probABetter"] == pytest.approx(NormalDist().cdf(diff / se), abs=1e-6)
    # 820 trades fit in about 9 KB, so the browser's usual 5 MB allowance holds hundreds of
    # strategies; the page caps the list at 50.
    assert trip["bytes"] < 12_000


def test_names_are_never_written_as_html() -> None:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("/* ---------- saved strategies ---------- */")
    block = text[start : text.index("/* ---------- timing breakdown ---------- */", start)]
    assert "innerHTML" not in block
    compare = text[text.index("/* ---------- strategy comparison ---------- */") : start]
    assert "innerHTML" not in compare
