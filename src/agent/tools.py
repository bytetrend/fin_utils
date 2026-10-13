import subprocess
import json
import os
from typing import Optional, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class OptunaOptimizerArgs(BaseModel):
    trades_csv: str = Field(
        ...,
        description="Path to the source trades CSV file (e.g., 'trades.csv')."
    )
    min_n: int = Field(
        default=30,
        description="Minimum trades required in any filtered training subset (overfitting guard)."
    )
    n_trials: int = Field(
        default=1000,
        description="Optuna trials to run per direction."
    )
    test_fraction: float = Field(
        default=0.25,
        description="Fraction of trades (chronologically last) held out as the test set."
    )
    cv_folds: int = Field(
        default=1,
        description="If > 1, cross-validate the objective across this many folds of the training data."
    )
    params: Optional[str] = Field(
        default=None,
        description="Comma-separated list of parameter columns to search over (e.g., 'a,b,c'). Acceptable LHS comparisons like 'ind_X<=5'."
    )
    direction: Optional[Literal["long", "short"]] = Field(
        default=None,
        description="Restrict optimization to 'long' or 'short' trades. If omitted, optimizes both."
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility."
    )
    warm_start: bool = Field(
        default=False,
        description="Seed the Optuna study with historical trade rows as warm-start trials (agentic/LLM use case)."
    )
    min_pattern_score: Optional[float] = Field(
        default=None,
        description="Enables entry-path stratification. Min PatternEntryScore threshold. Requires min_cvd_score."
    )
    min_cvd_score: Optional[float] = Field(
        default=None,
        description="Companion to min_pattern_score. Min CVDEntryScore threshold."
    )
    pattern_score_col: Optional[str] = Field(
        default=None,
        description="Explicit PatternEntryScore column name. Default auto-detects 'ind_PatternEntryScore'."
    )
    cvd_score_col: Optional[str] = Field(
        default=None,
        description="Explicit CVDEntryScore column name. Default auto-detects 'ind_CVDEntryScore'."
    )


@tool(args_schema=OptunaOptimizerArgs)
def run_ats_optuna_optimizer(
        trades_csv: str,
        min_n: int = 30,
        n_trials: int = 1000,
        test_fraction: float = 0.25,
        cv_folds: int = 1,
        params: Optional[str] = None,
        direction: Optional[Literal["long", "short"]] = None,
        seed: int = 42,
        warm_start: bool = False,
        min_pattern_score: Optional[float] = None,
        min_cvd_score: Optional[float] = None,
        pattern_score_col: Optional[str] = None,
        cvd_score_col: Optional[str] = None,
) -> str:
    """
    Executes the algorithmic trading simulation optimizer study using Optuna.
    Runs simulation trials locally and returns a structured performance report string.
    """
    output_json_path = f"temp_optuna_report_{seed}.json"

    # Construct shell flags dynamically matching script specifications
    cmd = ["python", "optimizer/ats_optuna_optimizer.py", trades_csv]
    cmd.extend(["--min-n", str(min_n)])
    cmd.extend(["--n-trials", str(n_trials)])
    cmd.extend(["--test-fraction", str(test_fraction)])
    cmd.extend(["--cv-folds", str(cv_folds)])
    cmd.extend(["--seed", str(seed)])
    if warm_start:
        cmd.extend(["--warm-start"])
    cmd.extend(["--output", output_json_path])

    if params:
        cmd.extend(["--params", params])
    if direction:
        cmd.extend(["--direction", direction])
    if min_pattern_score is not None:
        cmd.extend(["--min-pattern-score", str(min_pattern_score)])
    if min_cvd_score is not None:
        cmd.extend(["--min-cvd-score", str(min_cvd_score)])
    if pattern_score_col:
        cmd.extend(["--pattern-score-col", pattern_score_col])
    if cvd_score_col:
        cmd.extend(["--cvd-score-col", cvd_score_col])

    try:
        # Run local CLI optimization script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Read structured JSON report to pass metrics back to the LLM agent
        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                report_data = json.load(f)
            os.remove(output_json_path)  # Clean up temporary step files
            return json.dumps(report_data, indent=2)

        return f"Optimization finished. Standard Output: {result.stdout}"

    except subprocess.CalledProcessError as e:
        return f"Optimizer script failed error: {e.stderr or e.stdout}"


# Backward-compatible alias for older imports and scripts that expect
# run_trading_optimization to exist in this module.
run_trading_optimization = run_ats_optuna_optimizer

__all__ = [
    "OptunaOptimizerArgs",
    "run_ats_optuna_optimizer",
    "run_trading_optimization",
]
