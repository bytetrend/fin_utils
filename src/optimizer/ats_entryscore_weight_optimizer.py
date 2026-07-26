#!/usr/bin/env python3
"""
ats_entryscore_weight_optimizer.py

Optimizes the per-component WEIGHTS (and the score cutoff) of the
AtsFastReversal / AtsSlowReversal EntryScore formula. The strategy currently
splits entry evidence into two independently-weighted group scores:

    PatternEntryScore = IFF(C5, 1, 0)            // Speed flip
                       + IFF(C6 Or C10, 1, 0)     // There must be at least 1 CVD confirmation
                       + IFF(C14, 3, 0)           // bullish/bearish bar formation
                       + IFF(C15, 3, 0)           // bullish/bearish bar formation

    CVDEntryScore     = IFF(C6, 2, 0)             // CVDSpeedPct
                       + IFF(C10, 4, 0)           // CVD accel last 2 bars
                       + IFF(C13, 1, 0)           // CVDDeltaPct confirms
                       + IFF(C11, 2, 0)           // ATRsFromHma bar expansion

    C3  = PatternEntryScore >= MinPatternEntryScore
    C12 = CVDEntryScore     >= MinCVDEntryScore
    Entry: If (C3 Or C12) And C7 And C8 Then ...

This script searches for better weights AND a matching score threshold using
Optuna (TPE), using the SAME overfitting controls as ats_optuna_optimizer.py:

  1. CHRONOLOGICAL TRAIN/TEST SPLIT. Weights and threshold are only ever
     fit on the training window (earliest trades). They are then evaluated
     ONCE on the held-out test window. Only the test-window numbers should
     inform a live parameter change.
  2. MIN-N GUARD. Any weight/threshold combination that drops the training
     subset below --min-n trades is rejected during the search.
  3. Optional k-fold cross-validation within the training window (--cv-folds).
  4. MIN-THRESHOLD-FRAC GUARD. Prevents the search from "winning" with a
     near-zero threshold that just means "any single component fired" --
     a degenerate filter indistinguishable from no filter at all.

This is a DIFFERENT search problem than ats_optuna_optimizer.py, which finds
independent >=/<= thresholds on continuous columns. This script instead
searches a shared integer weight per binary component plus one score cutoff
-- the actual decision variables in your EntryScore formula.

All CSV column names are centralized in the `Column` enum below rather than
scattered as string literals, and all the core logic is consolidated into
the `ParameterOptimizer` class.

Component columns default to the current PatternEntryScore/CVDEntryScore
formula above -- override with --components / --components-long /
--components-short if your column names or formula differ. Note the new
"or" comparison type, needed for "C6 Or C10": format a component as
'Label:col1|col2:or' to OR two flag columns together.

Usage:
    python ats_entryscore_weight_optimizer.py trades.csv
    python ats_entryscore_weight_optimizer.py trades.csv --n-trials 3000 --min-n 20
    python ats_entryscore_weight_optimizer.py trades.csv --max-weight 3 --cv-folds 3
    python ats_entryscore_weight_optimizer.py trades.csv --output entryscore_report.json
    python ats_entryscore_weight_optimizer.py trades.csv --min-pattern-score 3 --min-cvd-score 4

Requires: pandas, numpy, optuna
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("This script requires optuna: pip install optuna --break-system-packages", file=sys.stderr)
    raise


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


# Required columns for this script to function at all.
REQUIRED_COLUMNS = [Column.PROFIT_LOSS, Column.IND_SIGNAL_SENT]

# Derived, script-internal column (not present in the source CSV).
PROFIT_HIT_COL = "ProfitHit"

MIN_N_DEFAULT = 30

# Default EntryScore components: (label, column(s), comparison)
# comparison is one of:
#   "flag"  -- column is already 0/1, tested as != 0
#   "gt0"   -- column tested as > 0
#   "lt0"   -- column tested as < 0
#   "or"    -- column is a 2-tuple of columns, tested as (col1 != 0) OR (col2 != 0)
#
# This reflects the CURRENT PatternEntryScore/CVDEntryScore formula (see
# module docstring). Because both group scores share several underlying
# flags (C6, C10, C11, C13), some flags appear here more than once -- once
# standalone (as CVDEntryScore's own term) and once inside the "C6 Or C10"
# derived term (as PatternEntryScore's CVD-confirmation term) -- letting the
# weight search evaluate both roles independently.
DEFAULT_COMPONENTS = [
    ("Speed flip (C5)",                    Column.IND_C5.value,                              "flag"),
    ("At least 1 CVD confirm (C6 Or C10)",  (Column.IND_C6.value, Column.IND_C10.value),        "or"),
    ("Bar formation (C14)",                 Column.IND_C14.value,                             "flag"),
    ("Bar formation (C15)",                 Column.IND_C15.value,                             "flag"),
    ("CVDSpeedPct (C6)",                    Column.IND_C6.value,                              "flag"),
    ("CVD accel last 2 bars (C10)",         Column.IND_C10.value,                             "flag"),
    ("CVDDeltaPct confirms (C13)",          Column.IND_C13.value,                             "flag"),
    ("ATRsFromHma bar expansion (C11)",     Column.IND_C11.value,                              "flag"),
]

COMPARISON_SYMBOLS = {"flag": "!=0", "gt0": ">0", "lt0": "<0", "or": "OR"}

# Entry-path stratification candidates (see ParameterOptimizer.resolve_entry_path_config).
PATTERN_SCORE_COL_CANDIDATES = [Column.IND_PATTERN_ENTRY_SCORE.value]
CVD_SCORE_COL_CANDIDATES = [Column.IND_CVD_ENTRY_SCORE.value]


@dataclass
class WeightResult:
    direction_label: str
    total_train_n: int
    total_test_n: int
    weights: dict           # {label: weight}
    threshold: int
    max_possible_score: int
    train_n: int
    train_expectancy: float
    train_hit_rate: float
    test_n: int
    test_expectancy: float
    test_hit_rate: float
    test_total_pl: float
    baseline_train_expectancy: float   # equal-weight (all 1), best threshold, train
    baseline_test_expectancy: float    # same filter, evaluated on test
    equal_weight_best_threshold: int
    n_trials_run: int
    warning: Optional[str] = None


# ============================================================================
# ParameterOptimizer -- consolidates all the logic used to compute the
# EntryScore weight/threshold optimization: loading data, parsing/building
# components, the chronological train/test split, the Optuna objective and
# search, entry-path stratification, and result formatting.
# ============================================================================
class ParameterOptimizer:

    def __init__(self, min_n: int = MIN_N_DEFAULT, n_trials: int = 2000,
                 test_fraction: float = 0.25, cv_folds: int = 1,
                 max_weight: int = 5, min_threshold_frac: float = 0.3,
                 seed: int = 42):
        self.min_n = min_n
        self.n_trials = n_trials
        self.test_fraction = test_fraction
        self.cv_folds = cv_folds
        self.max_weight = max_weight
        self.min_threshold_frac = min_threshold_frac
        self.seed = seed

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_trades(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        missing = [c.value for c in REQUIRED_COLUMNS if c.value not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required column(s): {missing}")
        df[PROFIT_HIT_COL] = df[Column.PROFIT_LOSS.value] > 0
        if Column.ENTRY_DATE.value in df.columns:
            df[Column.ENTRY_DATE.value] = pd.to_datetime(df[Column.ENTRY_DATE.value])
            df = df.sort_values(Column.ENTRY_DATE.value).reset_index(drop=True)
        else:
            warnings.warn("No EntryDate column -- using row order as the chronological proxy "
                           "for the train/test split.")
        return df

    @staticmethod
    def split_by_direction(df: pd.DataFrame) -> tuple:
        long_df = df[df[Column.IND_SIGNAL_SENT.value] == 1].reset_index(drop=True)
        short_df = df[df[Column.IND_SIGNAL_SENT.value] == -1].reset_index(drop=True)
        return long_df, short_df

    # ------------------------------------------------------------------
    # Component parsing / matrix construction
    # ------------------------------------------------------------------
    @staticmethod
    def parse_components_arg(spec: str) -> list:
        """Parses a '--components'-style override string:
        'Label1:col1:flag,Label2:col2:gt0,Label3:col3:lt0,Label4:colA|colB:or'
        """
        comps = []
        for part in spec.split(","):
            label, col_spec, cmp = part.split(":")
            label = label.strip()
            if cmp == "or":
                cols = tuple(c.strip() for c in col_spec.split("|"))
                if len(cols) != 2:
                    raise ValueError(f"'or' comparison requires exactly 2 columns separated by "
                                      f"'|' for component '{label}', got: {col_spec}")
                comps.append((label, cols, cmp))
            elif cmp in ("flag", "gt0", "lt0"):
                comps.append((label, col_spec.strip(), cmp))
            else:
                raise ValueError(f"Unknown comparison '{cmp}' for component '{label}' "
                                  f"(use 'flag', 'gt0', 'lt0', or 'or')")
        return comps

    @staticmethod
    def component_columns(component: tuple) -> tuple:
        """Returns the column(s) a component reads from, always as a tuple."""
        _, col, cmp = component
        return col if cmp == "or" else (col,)

    @staticmethod
    def build_component_matrix(df: pd.DataFrame, components: list) -> np.ndarray:
        cols = []
        for label, col, cmp in components:
            if cmp == "or":
                col1, col2 = col
                for c in (col1, col2):
                    if c not in df.columns:
                        raise ValueError(f"Component column '{c}' (for '{label}') not found in CSV.")
                vals = ((df[col1].fillna(0) != 0) | (df[col2].fillna(0) != 0)).astype(int).to_numpy()
            else:
                if col not in df.columns:
                    raise ValueError(f"Component column '{col}' (for '{label}') not found in CSV.")
                if cmp == "flag":
                    vals = (df[col].fillna(0) != 0).astype(int).to_numpy()
                elif cmp == "gt0":
                    vals = (df[col].fillna(0) > 0).astype(int).to_numpy()
                elif cmp == "lt0":
                    vals = (df[col].fillna(0) < 0).astype(int).to_numpy()
                else:
                    raise ValueError(f"Unknown comparison '{cmp}' for component '{label}'")
            cols.append(vals)
        return np.column_stack(cols)  # shape (n_trades, n_components)

    @staticmethod
    def print_components(components: list, indent: str = "    "):
        for label, col, cmp in components:
            col_str = " | ".join(col) if cmp == "or" else col
            print(f"{indent}{label:<38} <- {col_str} ({COMPARISON_SYMBOLS[cmp]})")

    # ------------------------------------------------------------------
    # Train/test split & cross-validation folds
    # ------------------------------------------------------------------
    def chronological_split(self, d: pd.DataFrame) -> tuple:
        n = len(d)
        if self.test_fraction <= 0:
            return d.reset_index(drop=True), d.iloc[0:0].reset_index(drop=True)
        n_test = max(1, int(round(n * self.test_fraction)))
        n_train = n - n_test
        return d.iloc[:n_train].reset_index(drop=True), d.iloc[n_train:].reset_index(drop=True)

    @staticmethod
    def kfold_indices(n: int, k: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            yield train_idx, val_idx

    def best_equal_weight_threshold(self, comp_matrix: np.ndarray, pl: np.ndarray) -> tuple:
        """Baseline: current formula (all weights = 1). Finds the best score
        cutoff for THIS baseline, so the optimized weights have a fair
        comparison point (not just compared against taking every trade)."""
        scores = comp_matrix.sum(axis=1)
        max_score = comp_matrix.shape[1]
        best = None
        for t in range(1, max_score + 1):
            mask = scores >= t
            n = int(mask.sum())
            if n < self.min_n:
                continue
            exp = float(pl[mask].mean())
            if best is None or exp > best[0]:
                best = (exp, t, n)
        if best is None:
            return float(pl.mean()), 0, len(pl)
        return best[0], best[1], best[2]

    # ------------------------------------------------------------------
    # Optuna objective & search
    # ------------------------------------------------------------------
    def make_objective(self, comp_matrix: np.ndarray, pl: np.ndarray, n_components: int):
        n = len(pl)
        fold_splits = list(self.kfold_indices(n, self.cv_folds, self.seed)) if self.cv_folds > 1 else None

        def objective(trial):
            weights = np.array([trial.suggest_int(f"w_{i}", 0, self.max_weight) for i in range(n_components)])
            max_score = int(weights.sum())
            if max_score == 0:
                return -1_000_000.0
            # Constrain the threshold to at least min_threshold_frac of the max
            # achievable score. Without this, the search can "win" by picking a
            # near-zero threshold (e.g. 1 out of a possible 17) which just means
            # "any single component fired" -- a degenerate filter that matches
            # almost every trade and is indistinguishable from no filter at all,
            # regardless of what the weights are. That is not a real reweighting,
            # it's the search finding a trivial way to reproduce the baseline.
            low = max(1, int(np.ceil(self.min_threshold_frac * max_score)))
            if low > max_score:
                return -1_000_000.0
            threshold = trial.suggest_int("threshold", low, max_score)
            scores = comp_matrix @ weights

            if fold_splits is None:
                mask = scores >= threshold
                cnt = int(mask.sum())
                if cnt < self.min_n:
                    return -1_000_000.0 + cnt
                return float(pl[mask].mean())
            else:
                fold_expectancies = []
                for _, val_idx in fold_splits:
                    val_mask = scores[val_idx] >= threshold
                    cnt = int(val_mask.sum())
                    if cnt < max(5, self.min_n // self.cv_folds):
                        return -1_000_000.0 + cnt
                    fold_expectancies.append(float(pl[val_idx][val_mask].mean()))
                return float(np.mean(fold_expectancies))

        return objective

    def optimize_direction(self, df: pd.DataFrame, label: str, components: list,
                            seed_offset: int = 0) -> WeightResult:
        labels = [c[0] for c in components]
        train_df, test_df = self.chronological_split(df)

        baseline_train_exp_all = float(train_df[Column.PROFIT_LOSS.value].mean()) if len(train_df) else float("nan")
        baseline_test_exp_all = float(test_df[Column.PROFIT_LOSS.value].mean()) if len(test_df) else float("nan")

        if len(train_df) < self.min_n:
            return WeightResult(
                direction_label=label, total_train_n=len(train_df), total_test_n=len(test_df),
                weights={}, threshold=0, max_possible_score=0,
                train_n=len(train_df), train_expectancy=baseline_train_exp_all,
                train_hit_rate=float(train_df[PROFIT_HIT_COL].mean()) if len(train_df) else float("nan"),
                test_n=len(test_df), test_expectancy=baseline_test_exp_all,
                test_hit_rate=float(test_df[PROFIT_HIT_COL].mean()) if len(test_df) else float("nan"),
                test_total_pl=float(test_df[Column.PROFIT_LOSS.value].sum()) if len(test_df) else 0.0,
                baseline_train_expectancy=baseline_train_exp_all, baseline_test_expectancy=baseline_test_exp_all,
                equal_weight_best_threshold=0, n_trials_run=0,
                warning=f"Training set ({len(train_df)}) is smaller than --min-n ({self.min_n}); skipped.",
            )

        train_matrix = self.build_component_matrix(train_df, components)
        test_matrix = self.build_component_matrix(test_df, components) if len(test_df) else np.zeros((0, len(components)))
        train_pl = train_df[Column.PROFIT_LOSS.value].to_numpy(dtype=float)
        test_pl = test_df[Column.PROFIT_LOSS.value].to_numpy(dtype=float) if len(test_df) else np.array([])

        # Fair baseline: current equal-weight formula, but with its OWN best
        # threshold (found only on training data), evaluated the same way as
        # the optimized weights.
        baseline_exp, baseline_thresh, _ = self.best_equal_weight_threshold(train_matrix, train_pl)
        if len(test_df) and baseline_thresh > 0:
            baseline_scores_test = test_matrix.sum(axis=1)
            baseline_mask_test = baseline_scores_test >= baseline_thresh
            baseline_test_exp = float(test_pl[baseline_mask_test].mean()) if baseline_mask_test.sum() else float("nan")
        else:
            baseline_test_exp = baseline_test_exp_all

        objective = self.make_objective(train_matrix, train_pl, len(labels))
        sampler = optuna.samplers.TPESampler(seed=self.seed + seed_offset)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_trial.params
        weights = np.array([best_params[f"w_{i}"] for i in range(len(labels))])
        threshold = int(best_params["threshold"])
        weight_dict = dict(zip(labels, [int(w) for w in weights]))

        train_scores = train_matrix @ weights
        train_mask = train_scores >= threshold
        train_n = int(train_mask.sum())
        train_exp = float(train_pl[train_mask].mean()) if train_n else float("nan")
        train_hit = float(train_df[PROFIT_HIT_COL].to_numpy()[train_mask].mean()) if train_n else float("nan")

        warning = None
        if len(test_df) == 0:
            test_n, test_exp, test_hit, test_pl_sum = 0, float("nan"), float("nan"), 0.0
            warning = "No held-out test data available."
        else:
            test_scores = test_matrix @ weights
            test_mask = test_scores >= threshold
            test_n = int(test_mask.sum())
            if test_n == 0:
                test_exp, test_hit, test_pl_sum = float("nan"), float("nan"), 0.0
                warning = "Weighted filter matched ZERO trades in the held-out test window -- cannot confirm."
            else:
                test_exp = float(test_pl[test_mask].mean())
                test_hit = float(test_df[PROFIT_HIT_COL].to_numpy()[test_mask].mean())
                test_pl_sum = float(test_pl[test_mask].sum())
                if test_n < self.min_n:
                    warning = f"Held-out test subset only has {test_n} trades (< --min-n {self.min_n}) -- low confidence."

        return WeightResult(
            direction_label=label, total_train_n=len(train_df), total_test_n=len(test_df),
            weights=weight_dict, threshold=threshold, max_possible_score=int(weights.sum()),
            train_n=train_n, train_expectancy=train_exp, train_hit_rate=train_hit,
            test_n=test_n, test_expectancy=test_exp, test_hit_rate=test_hit, test_total_pl=test_pl_sum,
            baseline_train_expectancy=baseline_exp, baseline_test_expectancy=baseline_test_exp,
            equal_weight_best_threshold=baseline_thresh, n_trials_run=self.n_trials, warning=warning,
        )

    # ------------------------------------------------------------------
    # Entry-path stratification
    # ------------------------------------------------------------------
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
        pattern_fired = df[pattern_col] >= min_pattern
        cvd_fired = df[cvd_col] >= min_cvd
        path = np.select(
            [pattern_fired & cvd_fired, pattern_fired & ~cvd_fired, (~pattern_fired) & cvd_fired],
            ["both", "pattern_only", "cvd_only"],
            default="neither",
        )
        return pd.Series(path, index=df.index)

    def resolve_entry_path_config(self, df: pd.DataFrame, pattern_score_col: Optional[str],
                                   cvd_score_col: Optional[str], min_pattern_score: Optional[float],
                                   min_cvd_score: Optional[float]) -> Optional[dict]:
        if min_pattern_score is None or min_cvd_score is None:
            return None
        pattern_col = self.autodetect_column(df, PATTERN_SCORE_COL_CANDIDATES, pattern_score_col)
        cvd_col = self.autodetect_column(df, CVD_SCORE_COL_CANDIDATES, cvd_score_col)
        if pattern_col is None or cvd_col is None:
            missing = []
            if pattern_col is None:
                missing.append(f"pattern score column (tried {pattern_score_col or PATTERN_SCORE_COL_CANDIDATES})")
            if cvd_col is None:
                missing.append(f"cvd score column (tried {cvd_score_col or CVD_SCORE_COL_CANDIDATES})")
            print(f"WARNING: --min-pattern-score/--min-cvd-score given but couldn't find: {'; '.join(missing)}. "
                  f"Skipping entry-path stratification.")
            return None
        return {"pattern_col": pattern_col, "cvd_col": cvd_col,
                "min_pattern": min_pattern_score, "min_cvd": min_cvd_score}

    @staticmethod
    def strip_column(components: list, col: str) -> list:
        """Drops any component that reads from `col` -- used to exclude a
        bucket's own classification column (tautological within that
        bucket), including when `col` appears inside an 'or' component."""
        return [c for c in components if col not in ParameterOptimizer.component_columns(c)]

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    @staticmethod
    def print_result(r: WeightResult, min_n: int):
        print(f"\n{'='*72}")
        print(f" {r.direction_label.upper()}")
        print(f"{'='*72}")
        print(f"Training window: {r.total_train_n} trades   Test window (held-out): {r.total_test_n} trades")

        if r.n_trials_run == 0:
            print(f"\n  SKIPPED: {r.warning}")
            return

        print(f"\nCurrent formula (all weights=1), best threshold found on training data: "
              f"score >= {r.equal_weight_best_threshold}")
        print(f"  TRAIN expectancy: ${r.baseline_train_expectancy:.2f}/trade")
        print(f"  TEST expectancy (same filter, held-out):  ${r.baseline_test_expectancy:.2f}/trade")

        print(f"\nOptuna-optimized weights ({r.n_trials_run} trials searched):")
        for label, w in r.weights.items():
            bar = "#" * w if w > 0 else "-"
            print(f"    {label:<38} weight = {w:<3} {bar}")
        print(f"  Score threshold: EntryScore >= {r.threshold}  (max possible score = {r.max_possible_score})")

        print(f"\n  TRAIN performance (subset matching the optimized weights, within the "
              f"{r.total_train_n}-trade training window):")
        print(f"    n={r.train_n}  hit_rate={r.train_hit_rate:.1%}  expectancy=${r.train_expectancy:.2f}/trade")

        print(f"\n  >>> TEST performance (held-out, the only trustworthy number) <<<")
        if r.test_n == 0:
            print(f"    n=0 -- filter never fired in the test window. Cannot confirm.")
        else:
            confidence = "OK" if r.test_n >= min_n else f"LOW CONFIDENCE (n={r.test_n} < min-n={min_n})"
            print(f"    n={r.test_n}  hit_rate={r.test_hit_rate:.1%}  expectancy=${r.test_expectancy:.2f}/trade  "
                  f"total_pl=${r.test_total_pl:.2f}  [{confidence}]")
            vs_baseline = r.test_expectancy - r.baseline_test_expectancy
            sign = "+" if vs_baseline >= 0 else ""
            print(f"    vs. equal-weight formula on same test window: {sign}${vs_baseline:.2f}/trade")
            if r.train_expectancy > 0 and (r.train_expectancy - r.test_expectancy) > 0.5 * abs(r.train_expectancy):
                print(f"    NOTE: expectancy dropped substantially from train to test "
                      f"(${r.train_expectancy:.2f} -> ${r.test_expectancy:.2f}) -- sign of overfitting. "
                      f"Treat these weights with skepticism.")

        if r.warning:
            print(f"\n  WARNING: {r.warning}")


# ============================================================================
# CLI
# ============================================================================
def add_entry_path_args(ap):
    ap.add_argument("--min-pattern-score", type=float, default=None,
                    help="Enables entry-path stratification (requires --min-cvd-score too).")
    ap.add_argument("--min-cvd-score", type=float, default=None,
                    help="Companion to --min-pattern-score.")
    ap.add_argument("--pattern-score-col", default=None,
                    help=f"Column holding PatternEntryScore. Auto-detected from "
                         f"{PATTERN_SCORE_COL_CANDIDATES} if not given.")
    ap.add_argument("--cvd-score-col", default=None,
                    help=f"Column holding CVDEntryScore. Auto-detected from "
                         f"{CVD_SCORE_COL_CANDIDATES} if not given.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Path to the merged trade CSV")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT,
                    help=f"Minimum trades required in any filtered training subset (default {MIN_N_DEFAULT})")
    ap.add_argument("--n-trials", type=int, default=2000,
                    help="Number of Optuna trials per direction (default 2000 -- this search space "
                         "is larger than the threshold sweep, so it benefits from more trials)")
    ap.add_argument("--test-fraction", type=float, default=0.25,
                    help="Fraction of trades (chronologically last) held out as the test set (default 0.25)")
    ap.add_argument("--cv-folds", type=int, default=1,
                    help="If > 1, cross-validate the objective across this many folds of training data (default 1 = off)")
    ap.add_argument("--max-weight", type=int, default=5,
                    help="Maximum integer weight searched per component (default 5)")
    ap.add_argument("--min-threshold-frac", type=float, default=0.3,
                    help="Minimum score threshold searched, as a fraction of that trial's max "
                         "possible score (default 0.3). Prevents the search from 'winning' with "
                         "a near-zero threshold that just means 'any single component fired' -- "
                         "a degenerate filter indistinguishable from no filter, regardless of the "
                         "weights found. Set to 0 to disable and restore the unconstrained search.")
    ap.add_argument("--components", default=None,
                    help="Override the default component list, applied to BOTH directions. Format: "
                         "'Label1:col1:flag,Label2:col2:gt0,Label3:col3:lt0,Label4:colA|colB:or,...' "
                         "where comparison is 'flag' (column already 0/1), 'gt0' (column > 0), "
                         "'lt0' (column < 0), or 'or' (either of two flag columns is nonzero).")
    ap.add_argument("--components-long", default=None,
                    help="Override the component list for LONG only (same format as --components). "
                         "Falls back to --components or the default list if not given.")
    ap.add_argument("--components-short", default=None,
                    help="Override the component list for SHORT only (same format as --components-long).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output", default=None, help="Optional path to write the full JSON report")
    add_entry_path_args(ap)
    args = ap.parse_args()

    opt = ParameterOptimizer(min_n=args.min_n, n_trials=args.n_trials, test_fraction=args.test_fraction,
                              cv_folds=args.cv_folds, max_weight=args.max_weight,
                              min_threshold_frac=args.min_threshold_frac, seed=args.seed)

    df = opt.load_trades(args.csv_path)
    base_components = opt.parse_components_arg(args.components) if args.components else DEFAULT_COMPONENTS
    components_long = opt.parse_components_arg(args.components_long) if args.components_long else base_components
    components_short = opt.parse_components_arg(args.components_short) if args.components_short else base_components

    long_df, short_df = opt.split_by_direction(df)

    print(f"Loaded {len(df)} trades from {args.csv_path}")
    print(f"  Long: {len(long_df)}   Short: {len(short_df)}")
    print(f"  LONG EntryScore components ({len(components_long)}):")
    opt.print_components(components_long)
    print(f"  SHORT EntryScore components ({len(components_short)}):")
    opt.print_components(components_short)
    print(f"  max_weight={args.max_weight}, n_trials={args.n_trials}, "
          f"test_fraction={args.test_fraction}, cv_folds={args.cv_folds}")

    path_cfg = opt.resolve_entry_path_config(df, args.pattern_score_col, args.cvd_score_col,
                                              args.min_pattern_score, args.min_cvd_score)
    if path_cfg:
        print(f"  Entry-path stratification ON: pattern_col={path_cfg['pattern_col']} "
              f"(>= {path_cfg['min_pattern']}), cvd_col={path_cfg['cvd_col']} (>= {path_cfg['min_cvd']})")

    def run_one(sub_df: pd.DataFrame, label: str, seed_offset: int, use_components: list) -> WeightResult:
        r = opt.optimize_direction(sub_df, label, use_components, seed_offset)
        opt.print_result(r, args.min_n)
        return r

    long_result = run_one(long_df, "long", 0, components_long)
    short_result = run_one(short_df, "short", 1, components_short)

    by_entry_path = {}
    if path_cfg:
        components_by_direction_and_path = {
            "long": {
                "pattern_only": opt.strip_column(components_long, path_cfg["pattern_col"]),
                "cvd_only": opt.strip_column(components_long, path_cfg["cvd_col"]),
                "both": components_long,
                "neither": components_long,
            },
            "short": {
                "pattern_only": opt.strip_column(components_short, path_cfg["pattern_col"]),
                "cvd_only": opt.strip_column(components_short, path_cfg["cvd_col"]),
                "both": components_short,
                "neither": components_short,
            },
        }

        print(f"\n{'#'*72}")
        print(" ENTRY-PATH BREAKDOWN")
        print(f"{'#'*72}")
        seed_offset = 2
        for direction_label, direction_df in [("long", long_df), ("short", short_df)]:
            path_series = opt.compute_entry_path(direction_df, path_cfg["pattern_col"], path_cfg["cvd_col"],
                                                  path_cfg["min_pattern"], path_cfg["min_cvd"])
            by_entry_path[direction_label] = {}
            for path_name in ["pattern_only", "cvd_only", "both", "neither"]:
                sub = direction_df[path_series == path_name].reset_index(drop=True)
                if path_name == "neither" and len(sub) == 0:
                    continue
                sub_label = f"{direction_label} ({path_name})"
                use_components = components_by_direction_and_path[direction_label][path_name]
                if len(sub) == 0 or not use_components:
                    print(f"\n{'='*72}\n {sub_label.upper()}\n{'='*72}")
                    print(f"  n={len(sub)}, components={len(use_components)} -- skipping "
                          f"(no trades and/or no components left to search).")
                    continue
                r = run_one(sub, sub_label, seed_offset, use_components)
                by_entry_path[direction_label][path_name] = r
                seed_offset += 1

    print(f"\n{'='*72}")
    print(" SUMMARY")
    print(f"{'='*72}")
    print("Only the TEST-window expectancy and the 'vs. equal-weight formula' line above")
    print("should inform any change to the live EntryScore weights. A positive number there")
    print("means the reweighted formula beat the current equal-weight formula on data neither")
    print("saw during the search. A filter with n=0 or LOW CONFIDENCE in the test window is")
    print("not yet confirmed -- gather more trades before changing the live weights.")
    if path_cfg:
        print("If the best weights differ between pattern_only and cvd_only for the same")
        print("direction, that's a real reason to weight those two entry paths' components")
        print("differently rather than share one formula across both.")

    if args.output:
        report = {
            "csv_path": args.csv_path, "min_n": args.min_n, "n_trials": args.n_trials,
            "test_fraction": args.test_fraction, "cv_folds": args.cv_folds, "max_weight": args.max_weight,
            "min_threshold_frac": args.min_threshold_frac,
            "long": asdict(long_result), "short": asdict(short_result),
        }
        if path_cfg:
            report["entry_path_config"] = path_cfg
            report["by_entry_path"] = {
                direction: {path_name: asdict(r) for path_name, r in paths.items()}
                for direction, paths in by_entry_path.items()
            }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()