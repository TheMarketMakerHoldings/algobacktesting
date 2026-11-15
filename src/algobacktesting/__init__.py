"""algobacktesting: tools for telling real trading edges from backtest overfitting.

Published by The Market Maker Holdings & Co. Not investment advice.
"""

from algobacktesting.cv import (
    CPCVResult,
    Split,
    combinatorial_purged_cv,
    purged_kfold,
)
from algobacktesting.pbo import (
    PBOResult,
    column_sharpe_ratios,
    probability_of_backtest_overfitting,
)
from algobacktesting.sharpe import (
    EULER_GAMMA,
    ReturnStats,
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    return_stats,
    sharpe_ratio,
)

__version__ = "0.1.0"

__all__ = [
    "EULER_GAMMA",
    "CPCVResult",
    "PBOResult",
    "ReturnStats",
    "Split",
    "__version__",
    "annualize_sharpe",
    "column_sharpe_ratios",
    "combinatorial_purged_cv",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold",
    "return_stats",
    "sharpe_ratio",
]
