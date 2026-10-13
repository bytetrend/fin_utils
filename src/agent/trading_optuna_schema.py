import subprocess
import json
import os
from typing import Optional, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

#  1. The LangChain Tool Definition
#  This tool maps your schema attributes directly to the Optuna CLI tool parameters while
#  handling structural cleanup.
class TradingOptunaSchema(BaseModel):
    trades_csv: str = Field(
        ..., description="Path to your 500-entry trades log spreadsheet (CSV format)."
    )
    n_trials: int = Field(
        default=1000, description="Optuna mathematical trials to run per direction."
    )
    min_n: int = Field(
        default=30, description="Overfitting guard. Minimum trade count allowed in any sub-bucket filter."
    )
    test_fraction: float = Field(
        default=0.25, description="Chronological holdout test set slice."
    )
    cv_folds: int = Field(
        default=1, description="Cross-validation splits. Set >1 to enable cross-validation."
    )
    params: Optional[str] = Field(
        default="ind_PatternEntryScore,ind_CVDEntryScore,ind_CVDSpeedPct,ind_CVDDeltaPct",
        description="Comma-separated target numeric ind_* features for Optuna to optimize thresholds over."
    )
    direction: Optional[Literal["long", "short"]] = Field(
        default=None, description="Restrict optimization to 'long' or 'short' entries."
    )
    warm_start: bool = Field(
        default=False, description="Seed the Optuna study with historical trade rows as warm-start trials."
    )
    min_pattern_score: Optional[float] = Field(
        default=None, description="Minimum threshold constraint for entry stratification."
    )
    min_cvd_score: Optional[float] = Field(
        default=None, description="Companion threshold to min_pattern_score."
    )


@tool(args_schema=TradingOptunaSchema)
def run_trading_optimization(
        trades_csv: str,
        n_trials: int = 1000,
        min_n: int = 30,
        test_fraction: float = 0.25,
        cv_folds: int = 1,
        params: Optional[str] = None,
        direction: Optional[Literal["long", "short"]] = None,
        warm_start: bool = False,
        min_pattern_score: Optional[float] = None,
        min_cvd_score: Optional[float] = None,
) -> str:
    """
    Runs the Optuna optimizer over existing historical trades data.
    Finds optimal math barriers for ind_* columns to isolate higher profit/loss metrics.
    """
    output_report = "optuna_trade_report.json"

    cmd = [
        "python", "ats_optuna_optimizer.py", trades_csv,
        "--n-trials", str(n_trials),
        "--min-n", str(min_n),
        "--test-fraction", str(test_fraction),
        "--cv-folds", str(cv_folds),
        "--output", output_report,
        "--pattern-score-col", "ind_PatternEntryScore",
        "--cvd-score-col", "ind_CVDEntryScore"
    ]

    if warm_start:
        cmd.append("--warm-start")
    if params:
        cmd.extend(["--params", params])
    if direction:
        cmd.extend(["--direction", direction])
    if min_pattern_score is not None:
        cmd.extend(["--min-pattern-score", str(min_pattern_score)])
    if min_cvd_score is not None:
        cmd.extend(["--min-cvd-score", str(min_cvd_score)])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(output_report):
            with open(output_report, "r") as f:
                data = json.load(f)
            os.remove(output_report)
            return json.dumps(data, indent=2)
        return "Optimization completed but output file was missing."
    except subprocess.CalledProcessError as e:
        return f"Optuna Optimization Execution Failed: {e.stderr}"
