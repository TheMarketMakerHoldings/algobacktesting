"""The README demo must keep running and keep telling the same story."""

import importlib.util
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "examples" / "overfit_demo.py"


def test_overfit_demo_story() -> None:
    spec = importlib.util.spec_from_file_location("overfit_demo", DEMO)
    assert spec is not None
    assert spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    results = demo.main()
    noise, modest, strong = results.values()

    # Pure noise: an impressive-looking winner that the tests refuse to confirm.
    assert noise["best_annual_sharpe"] > 1.0
    assert noise["deflated_sharpe"] < 0.95
    # A modest real edge is out-searched by luck: the winning backtest is still noise.
    assert modest["best_annual_sharpe"] == noise["best_annual_sharpe"]
    # A strong edge wins the search, and PBO drops well below the noise case.
    assert strong["best_annual_sharpe"] > noise["best_annual_sharpe"]
    assert strong["pbo"] < noise["pbo"] / 2
    assert strong["deflated_sharpe"] > noise["deflated_sharpe"]
