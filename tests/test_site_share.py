"""A shared result link must reproduce the sender's verdict and carry nothing else.

``docs/index.html`` turns a result into the text after "#" in a link (between
``/* share:start */`` and ``/* share:end */``). For a list of trades the link holds the
statistics the verdict is computed from (trades, setups, average, standard deviation, skew,
kurtosis) and a few counts to display, never a trade. This test runs the page's code in
Node.js and checks:

1. A link read back gives the same chance, trades needed and luck line as the trades it came
   from, on lists of very different shapes and sizes.
2. Its length does not grow with the number of trades, and it holds only the known fields.
3. Damaged or edited links are refused with a message, never shown as a result.

Skipped without Node.js.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
HARNESS = ROOT / "tests" / "site_share_harness.js"
NODE = shutil.which("node")

RNG = np.random.default_rng(2460553)


def _trade_like(n: int, win_rate: float, win: float, loss: float) -> list[float]:
    wins = RNG.random(n) < win_rate
    results = np.where(wins, RNG.normal(win, win / 4, n), RNG.normal(-loss, loss / 6, n))
    return [float(x) for x in results.round(2)]


# Breakevens: flat before fees, a small loss after them (as the page's import produces).
_BEFORE = np.where(RNG.random(150) < 0.25, 0.0, RNG.normal(20, 150, 150)).round(2)
_AFTER = (_BEFORE - 4.12).round(2)
LISTS: list[dict[str, Any]] = [
    {"values": RNG.normal(12, 90, 59).round(2).tolist(), "unit": "$", "setups": 5},
    {"values": _trade_like(151, 0.35, 250, 130), "unit": "$", "setups": 5},
    {"values": _trade_like(2604, 0.5, 40, 38), "unit": "$", "setups": 1},
    {"values": _trade_like(5000, 0.2, 900, 120), "unit": "$", "setups": 20},
    {"values": RNG.normal(-8, 60, 300).round(2).tolist(), "unit": "$", "setups": 3},  # losing
    {"values": RNG.normal(0.1, 1.1, 40).round(4).tolist(), "unit": "", "setups": 2},  # pasted R
    {"values": (RNG.normal(3, 40, 80) * 1e5).round(0).tolist(), "unit": "$", "setups": 7},  # large
    {"values": RNG.normal(0.002, 0.01, 10).tolist(), "unit": "", "setups": 1},  # 10 tiny trades
    {"values": _AFTER.tolist(), "before": _BEFORE.tolist(), "unit": "$", "setups": 4},
]
QUICK: list[dict[str, Any]] = [
    {"kind": "quick", "winRatePct": 45, "rr": 1.5, "trades": 120, "setups": 5},
    {"kind": "quick", "winRatePct": 33.3, "rr": 2.25, "trades": 10, "setups": 10000},
]
GOOD_T = "share=1&k=t&n=59&s=5&u=%24&m=144.2&sd=520.5&sk=0.4&ku=3.1&w=20&l=25&b=14&dd=2345.5"
DECODE: dict[str, str | None] = {
    "empty": None,
    "section anchor": "#compare",
    "good": "#" + GOOD_T,
    "good, no hash": GOOD_T,
    "good, extra field ignored": "#" + GOOD_T + "&x=1",
    "first of a repeated field wins": "#" + GOOD_T.replace("&s=5", "&s=5&s=9"),
    "no unit": "#" + GOOD_T.replace("&u=%24", ""),
    "other version": "#" + GOOD_T.replace("share=1", "share=2"),
    "no kind": "#share=1",
    "unknown kind": "#share=1&k=x&n=59&s=5",
    "quick ok": "#share=1&k=q&w=45&rr=1.5&n=120&s=5",
    "quick win rate 0": "#share=1&k=q&w=0&rr=1.5&n=120&s=5",
    "quick win rate 100": "#share=1&k=q&w=100&rr=1.5&n=120&s=5",
    "quick rr 0": "#share=1&k=q&w=45&rr=0&n=120&s=5",
    "quick 9 trades": "#share=1&k=q&w=45&rr=1.5&n=9&s=5",
    "quick fractional trades": "#share=1&k=q&w=45&rr=1.5&n=10.5&s=5",
    "quick 0 setups": "#share=1&k=q&w=45&rr=1.5&n=120&s=0",
    "quick text": "#share=1&k=q&w=abc&rr=1.5&n=120&s=5",
    "quick infinity": "#share=1&k=q&w=1e400&rr=1.5&n=120&s=5",
    "quick missing rr": "#share=1&k=q&w=45&n=120&s=5",
    "sd 0": "#" + GOOD_T.replace("sd=520.5", "sd=0"),
    "sd negative": "#" + GOOD_T.replace("sd=520.5", "sd=-1"),
    "kurtosis impossible": "#" + GOOD_T.replace("ku=3.1", "ku=1.1"),  # below 1 + 0.4 squared
    "counts above trades": "#" + GOOD_T.replace("w=20", "w=21"),
    "drawdown negative": "#" + GOOD_T.replace("dd=2345.5", "dd=-5"),
    "unit euro": "#" + GOOD_T.replace("u=%24", "u=EUR"),
    "unit bad encoding": "#" + GOOD_T.replace("u=%24", "u=%E0"),
    "mean missing": "#" + GOOD_T.replace("&m=144.2", ""),
    "nine trades": "#" + GOOD_T.replace("n=59", "n=9"),
    "hex number": "#" + GOOD_T.replace("m=144.2", "m=0x90"),
}
DAMAGED = {"error": "This shared link is damaged or incomplete, so it cannot be shown."}
FIELDS = {"share", "k", "n", "s", "u", "m", "sd", "sk", "ku", "w", "l", "b", "dd"}


@pytest.fixture(scope="module")
def js() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    text = PAGE.read_text(encoding="utf-8")
    code = "".join(
        text[text.index(f"/* {name}:start */") : text.index(f"/* {name}:end */")]
        for name in ("stats", "csv", "share")
    )
    payload = {"lists": LISTS, "quick": QUICK, "decode": list(DECODE.values())}
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "share.js", Path(tmp) / "input.json"
        script.write_text(code + HARNESS.read_text(encoding="utf-8"), encoding="utf-8")
        data.write_text(json.dumps(payload), encoding="utf-8")
        done = subprocess.run(
            [NODE, str(script), str(data)],
            capture_output=True,
            encoding="utf-8",
            check=True,
            timeout=60,
        )
    out: dict[str, Any] = json.loads(done.stdout)
    out["by_name"] = dict(zip(DECODE, out["decode"], strict=True))
    return out


def test_a_link_gives_the_senders_verdict(js: dict[str, Any]) -> None:
    for case, r in zip(LISTS, js["lists"], strict=True):
        sent, got = r["sent"], r["got"]
        # Ten significant digits on each statistic move the chance by far less than 1e-9;
        # the page shows it to the whole percent.
        assert got["prob"] == pytest.approx(sent["prob"], abs=1e-9)
        assert got["need"] == sent["need"]
        assert got["luck"] == pytest.approx(sent["luck"], rel=1e-9)
        assert got["net"] == pytest.approx(sent["net"], rel=1e-8, abs=1e-6)
        assert (got["wins"], got["losses"], got["breakevens"]) == (
            sent["wins"],
            sent["losses"],
            sent["breakevens"],
        )
        assert got["maxDrawdown"] == pytest.approx(sent["maxDrawdown"], rel=1e-9)
        assert (got["kind"], got["trades"], got["setups"], got["unit"]) == (
            "trades",
            len(case["values"]),
            case["setups"],
            case["unit"],
        )
    flat = js["lists"][-1]["got"]["breakevens"]
    assert flat == int((np.array(_BEFORE) == 0).sum()) > 0  # breakevens use results before fees


def test_a_link_holds_summary_numbers_only(js: dict[str, Any]) -> None:
    for r in js["lists"]:
        text = r["text"]
        assert len(text) < 260  # the same size for 10 trades or 5,000: no trade is in it
        keys = [k for k, _ in parse_qsl(text, keep_blank_values=True)]
        assert set(keys) <= FIELDS
        assert len(keys) == len(set(keys))
    assert "u=%24" in js["lists"][0]["text"]  # "$" is encoded
    assert "&u=" not in js["lists"][5]["text"]  # pasted results carry no unit


def test_quick_check_links(js: dict[str, Any]) -> None:
    for sent, r in zip(QUICK, js["quick"], strict=True):
        assert r["got"] == sent
    assert js["quick"][0]["text"] == "share=1&k=q&w=45&rr=1.5&n=120&s=5"


def test_damaged_links_are_refused(js: dict[str, Any]) -> None:
    got = js["by_name"]
    assert got["empty"] is None
    assert got["section anchor"] is None  # an ordinary "#compare" is not a shared result
    good = got["good"]
    assert good["kind"] == "trades"
    assert good["unit"] == "$"
    assert good["stats"]["sharpe"] == pytest.approx(144.2 / 520.5)
    assert got["good, no hash"] == good
    assert got["good, extra field ignored"] == good
    assert got["first of a repeated field wins"] == good
    assert got["no unit"]["unit"] == ""
    assert got["quick ok"] == QUICK[0]
    assert "different version" in got["other version"]["error"]
    for name in (
        "no kind",
        "unknown kind",
        "quick win rate 0",
        "quick win rate 100",
        "quick rr 0",
        "quick 9 trades",
        "quick fractional trades",
        "quick 0 setups",
        "quick text",
        "quick infinity",
        "quick missing rr",
        "sd 0",
        "sd negative",
        "kurtosis impossible",
        "counts above trades",
        "drawdown negative",
        "unit euro",
        "unit bad encoding",
        "mean missing",
        "nine trades",
        "hex number",
    ):
        assert got[name] == DAMAGED, name


def test_links_use_the_part_after_the_hash() -> None:
    # The part after "#" never reaches a server, GitHub's included.
    text = PAGE.read_text(encoding="utf-8")
    assert '"#" + encodeShare(' in text
    assert "?share=" not in text
