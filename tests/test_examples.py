"""The demos must keep running and keep telling the same story."""

import importlib.util
from pathlib import Path
from types import ModuleType

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overfit_demo_story() -> None:
    results = _load("overfit_demo").main()
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


def test_leakage_demo_story() -> None:
    # 12 datasets keep the test fast. The full 200-dataset run (see the demo's docstring)
    # gave: shuffled +0.816, unpurged +0.050, purged +0.000, leak +0.050 (s.e. 0.002).
    # One dataset's score varies by about 0.07, so with only 12 datasets we assert the two
    # robust facts and deliberately do NOT assert that the purged score is near zero.
    results = _load("leakage_demo").main(n_datasets=12)
    shuffled, _ = results["shuffled k-fold"]
    leak, leak_se = results["leak removed by purging"]

    assert shuffled > 0.5  # shuffling overlapping labels leaks massively
    assert leak > 0.02  # purging removes a real, positive leak...
    assert leak / leak_se > 4  # ...that is clear even on 12 datasets (paired comparison)
