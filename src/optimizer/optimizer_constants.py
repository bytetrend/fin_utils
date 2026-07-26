"""
optimizer_constants.py

Shared constants, enums, and utility classes used across the entire ATS
parameter-optimization toolchain:

    ats_param_optimizer.py           -- GridSearchOptimizer
    ats_optuna_optimizer.py          -- BayesianThresholdOptimizer
    ats_feature_importance.py        -- FeatureImportanceAnalyzer
    ats_entryscore_weight_optimizer.py -- ParameterOptimizer
    ats_performance_report.py        -- PerformanceReportGenerator

Every script imports `Column` and whichever shared helper classes it needs
from here rather than redefining its own copy, so a fix or CSV-schema change
only has to be made once.
"""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Column enum -- every column name from the merged trade CSV, defined once.
# (str, Enum) so members behave as plain strings everywhere a string is
# expected (df[Column.PROFIT_LOSS], f"{Column.PROFIT_LOSS.value}", dict keys,
# JSON, etc.) -- always use `.value` when interpolating into an f-string or
# writing to JSON/dict keys, to get a clean string rather than an Enum repr.
# ============================================================================
class Column(str, Enum):
    SYMBOL = "Symbol"
    ENTRY_DATE = "EntryDate"
    ENTRY_TIME = "EntryTime"
    ENTRY_NAME = "EntryName"
    ENTRY_PRICE = "EntryPrice"
    EXIT_DATE = "ExitDate"
    EXIT_TIME = "ExitTime"
    EXIT_NAME = "ExitName"
    EXIT_PRICE = "ExitPrice"
    SHARES = "Shares"
    PROFIT_LOSS = "Profit/Loss"
    BAR_NUMBER = "BarNumber"
    SIGNAL_BAR = "SignalBar"
    R_T = "R/T"
    IND_BAR_DATE = "ind_BarDate"
    IND_BAR_TIME = "ind_BarTime"
    IND_BAR_NUMBER = "ind_BarNumber"
    IND_TICK = "ind_Tick"
    IND_INTERVAL = "ind_Interval"
    IND_SIGNAL_SENT = "ind_SignalSent"
    IND_CVD_SPEED_PCT = "ind_CVDSpeedPct"
    IND_CVD_DELTA_PCT = "ind_CVDDeltaPct"
    IND_CVD_ACEL_PCT = "ind_CVDAcelPct"
    IND_REV_ATRS_PER_SEC = "ind_RevATRsPerSec"
    IND_DELTA_ATRS = "ind_DeltaATRs"
    IND_AVG_ATR = "ind_AvgATR"
    IND_BAR_ATR = "ind_BarATR"
    IND_TREND_BAR_COUNT = "ind_TrendBarCount"
    IND_ANGLE = "ind_Angle"
    IND_PIP_SPEED = "ind_PipSpeed"
    IND_PIP_SPEED_NORM = "ind_PipSpeedNorm"
    IND_PIP_SPEED_ACEL = "ind_PipSpeedAcel"
    IND_PIP_SPEED_ACEL_NORM = "ind_PipSpeedAcelNorm"
    IND_ATRS_FROM_HMA = "ind_ATRsFromHma"
    IND_DELTA_PIPS = "ind_DeltaPips"
    IND_PATTERN_ENTRY_SCORE = "ind_PatternEntryScore"
    IND_CVD_ENTRY_SCORE = "ind_CVDEntryScore"
    IND_PIP_SPEED_TREND_PCT = "ind_PipSpeedTrendPct"
    IND_HMA_GAP_STD_DEV = "ind_HMAGapStdDev"
    IND_HMA_GAP_CV = "ind_HMAGapCV"
    IND_C1 = "ind_C1"
    IND_C2 = "ind_C2"
    IND_C3 = "ind_C3"
    IND_C4 = "ind_C4"
    IND_C5 = "ind_C5"
    IND_C6 = "ind_C6"
    IND_C7 = "ind_C7"
    IND_C8 = "ind_C8"
    IND_C9 = "ind_C9"
    IND_C10 = "ind_C10"
    IND_C11 = "ind_C11"
    IND_C12 = "ind_C12"
    IND_C13 = "ind_C13"
    IND_C14 = "ind_C14"
    IND_C15 = "ind_C15"
    IND_C16 = "ind_C16"
    IND_CLOSE = "ind_Close"
    IND_R_T = "ind_R/T"
    IND_COMPUTER_TIME = "ind_computertime"

    @classmethod
    def values(cls) -> list:
        return [c.value for c in cls]


# ============================================================================
# Shared defaults
# ============================================================================
MIN_N_DEFAULT = 30
MIN_TRADES_FOR_MODEL_DEFAULT = 60

# Derived, script-internal column (not present in the source CSV) -- added by
# TradeDataLoader.load_trades to every loaded DataFrame.
PROFIT_HIT_COL = "ProfitHit"

# Prefix used by the strategy's boolean condition flags (C1, C2, ... C16).
BOOLEAN_FLAG_PREFIX = "ind_C"

# Columns that are identifiers / timestamps / raw metadata, not tunable
# continuous entry-filter parameters. Used by scripts that auto-detect
# candidate parameters from the CSV.
NON_PARAM_COLUMNS = {
    Column.SYMBOL.value, Column.ENTRY_DATE.value, Column.ENTRY_TIME.value,
    Column.ENTRY_NAME.value, Column.ENTRY_PRICE.value, Column.EXIT_DATE.value,
    Column.EXIT_TIME.value, Column.EXIT_NAME.value, Column.EXIT_PRICE.value,
    Column.SHARES.value, Column.PROFIT_LOSS.value, Column.BAR_NUMBER.value,
    Column.SIGNAL_BAR.value, Column.R_T.value, Column.IND_BAR_DATE.value,
    Column.IND_BAR_TIME.value, Column.IND_BAR_NUMBER.value, Column.IND_TICK.value,
    Column.IND_INTERVAL.value, Column.IND_SIGNAL_SENT.value, Column.IND_CLOSE.value,
    Column.IND_R_T.value, Column.IND_COMPUTER_TIME.value, PROFIT_HIT_COL,
}

# Entry-path stratification candidates (see EntryPathAnalyzer below).
PATTERN_SCORE_COL_CANDIDATES = [Column.IND_PATTERN_ENTRY_SCORE.value]
CVD_SCORE_COL_CANDIDATES = [Column.IND_CVD_ENTRY_SCORE.value]


# ============================================================================
# TradeDataLoader -- shared CSV loading + direction splitting.
# ============================================================================
class TradeDataLoader:

    @staticmethod
    def load_trades(csv_path: str, sort_by_entry_date: bool = False) -> pd.DataFrame:
        """Loads a merged trade CSV, strips column whitespace, validates the
        two required columns, and adds the derived ProfitHit column.

        If `sort_by_entry_date` is True (needed by anything doing a
        chronological train/test split), EntryDate is parsed and the
        DataFrame sorted by it, with a warning if EntryDate isn't present
        (falls back to row order as the chronological proxy).
        """
        import warnings

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if Column.PROFIT_LOSS.value not in df.columns:
            raise ValueError(f"Expected a '{Column.PROFIT_LOSS.value}' column in the CSV.")
        if Column.IND_SIGNAL_SENT.value not in df.columns:
            raise ValueError(f"Expected an '{Column.IND_SIGNAL_SENT.value}' column (1=long, -1=short).")
        df[PROFIT_HIT_COL] = df[Column.PROFIT_LOSS.value] > 0

        if sort_by_entry_date:
            if Column.ENTRY_DATE.value in df.columns:
                df[Column.ENTRY_DATE.value] = pd.to_datetime(df[Column.ENTRY_DATE.value])
                df = df.sort_values(Column.ENTRY_DATE.value).reset_index(drop=True)
            else:
                warnings.warn(
                    "No EntryDate column found -- using row order as the chronological "
                    "proxy for the train/test split. If your CSV isn't already in "
                    "chronological order, the train/test split is not meaningful."
                )
        return df

    @staticmethod
    def split_by_direction(df: pd.DataFrame) -> tuple:
        """Returns (long_df, short_df), each reset-indexed."""
        long_df = df[df[Column.IND_SIGNAL_SENT.value] == 1].reset_index(drop=True)
        short_df = df[df[Column.IND_SIGNAL_SENT.value] == -1].reset_index(drop=True)
        return long_df, short_df


# ============================================================================
# ColumnSelector -- shared "which columns are tunable continuous parameters"
# logic. Parametrized because scripts differ slightly in what they want:
# ats_param_optimizer.py / ats_optuna_optimizer.py exclude boolean C-flags
# (handled separately) and require >=3 unique values; ats_feature_importance.py
# wants the C-flags included (it ranks them alongside everything else) and
# only requires >=2 unique values.
# ============================================================================
class ColumnSelector:

    @staticmethod
    def get_param_columns(df: pd.DataFrame, explicit: Optional[list] = None,
                           non_param_columns: set = NON_PARAM_COLUMNS,
                           exclude_boolean_flags: bool = True,
                           min_unique: int = 3) -> list:
        if explicit:
            missing = [c for c in explicit if c not in df.columns]
            if missing:
                raise ValueError(f"Requested parameters not found in CSV: {missing}")
            return explicit

        params = []
        for c in df.columns:
            if c in non_param_columns:
                continue
            if exclude_boolean_flags and c.startswith(BOOLEAN_FLAG_PREFIX) and c[len(BOOLEAN_FLAG_PREFIX):].isdigit():
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if df[c].nunique(dropna=True) < min_unique:
                continue
            params.append(c)
        return params

    @staticmethod
    def get_boolean_flag_columns(df: pd.DataFrame) -> list:
        return [c for c in df.columns
                if c.startswith(BOOLEAN_FLAG_PREFIX) and c[len(BOOLEAN_FLAG_PREFIX):].isdigit()]


# ============================================================================
# EntryPathAnalyzer -- shared entry-path stratification.
#
# If a strategy enters via (PatternEntryScore >= Min Or CVDEntryScore >= Min),
# pooling all trades together for analysis mixes two potentially distinct
# trade populations. These helpers classify each trade as pattern_only /
# cvd_only / both / neither, based on which side of that OR-gate actually
# fired, so each can be analyzed independently.
# ============================================================================
class EntryPathAnalyzer:

    @staticmethod
    def autodetect_column(df: pd.DataFrame, candidates: list, override: Optional[str] = None) -> Optional[str]:
        if override:
            return override if override in df.columns else None
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def compute_entry_path(df: pd.DataFrame, pattern_col: str, cvd_col: str,
                            min_pattern: float, min_cvd: float) -> pd.Series:
        """Returns a Series of 'pattern_only' / 'cvd_only' / 'both' / 'neither'
        per trade. 'neither' should be rare/absent in a clean log (a trade
        shouldn't have been taken if neither cleared its minimum) but is kept
        as a visible bucket rather than silently dropped, in case of logging
        quirks or a formula that's evolved since the trade was taken."""
        pattern_fired = df[pattern_col] >= min_pattern
        cvd_fired = df[cvd_col] >= min_cvd
        path = np.select(
            [pattern_fired & cvd_fired, pattern_fired & ~cvd_fired, (~pattern_fired) & cvd_fired],
            ["both", "pattern_only", "cvd_only"],
            default="neither",
        )
        return pd.Series(path, index=df.index)

    @staticmethod
    def resolve_entry_path_config(df: pd.DataFrame, pattern_score_col: Optional[str],
                                   cvd_score_col: Optional[str], min_pattern_score: Optional[float],
                                   min_cvd_score: Optional[float]) -> Optional[dict]:
        """Returns a dict with resolved column names if entry-path splitting is
        requested and possible, else None (caller should behave exactly as if
        entry-path stratification were never requested)."""
        if min_pattern_score is None or min_cvd_score is None:
            return None
        pattern_col = EntryPathAnalyzer.autodetect_column(df, PATTERN_SCORE_COL_CANDIDATES, pattern_score_col)
        cvd_col = EntryPathAnalyzer.autodetect_column(df, CVD_SCORE_COL_CANDIDATES, cvd_score_col)
        if pattern_col is None or cvd_col is None:
            missing = []
            if pattern_col is None:
                missing.append(f"pattern score column (tried {pattern_score_col or PATTERN_SCORE_COL_CANDIDATES})")
            if cvd_col is None:
                missing.append(f"cvd score column (tried {cvd_score_col or CVD_SCORE_COL_CANDIDATES})")
            print(f"WARNING: --min-pattern-score/--min-cvd-score given but couldn't find: {'; '.join(missing)}. "
                  f"Skipping entry-path stratification. Use --pattern-score-col/--cvd-score-col to specify exact "
                  f"column names if your CSV uses different ones.")
            return None
        return {"pattern_col": pattern_col, "cvd_col": cvd_col,
                "min_pattern": min_pattern_score, "min_cvd": min_cvd_score}

    @staticmethod
    def add_cli_args(ap):
        """Registers the --min-pattern-score / --min-cvd-score / --pattern-score-col
        / --cvd-score-col arguments shared by every script that supports
        entry-path stratification."""
        ap.add_argument("--min-pattern-score", type=float, default=None,
                        help="Enables entry-path stratification (requires --min-cvd-score too). "
                             "The MinPatternEntryScore threshold used live, so trades can be classified "
                             "as pattern_only / cvd_only / both based on which side of the OR-gate fired.")
        ap.add_argument("--min-cvd-score", type=float, default=None,
                        help="Companion to --min-pattern-score; the MinCVDEntryScore threshold used live.")
        ap.add_argument("--pattern-score-col", default=None,
                        help=f"Column holding the PatternEntryScore value. Auto-detected from "
                             f"{PATTERN_SCORE_COL_CANDIDATES} if not given.")
        ap.add_argument("--cvd-score-col", default=None,
                        help=f"Column holding the CVDEntryScore value. Auto-detected from "
                             f"{CVD_SCORE_COL_CANDIDATES} if not given.")

    @staticmethod
    def strip_column(components: list, col: str, component_columns_fn) -> list:
        """Drops any component that reads from `col` -- used to exclude a
        bucket's own classification column (tautological within that
        bucket). `component_columns_fn` extracts the column(s) a component
        reads from (a script-specific detail, e.g. ats_entryscore_weight_
        optimizer.py's components can read from 2 columns via an 'or' type)."""
        return [c for c in components if col not in component_columns_fn(c)]
