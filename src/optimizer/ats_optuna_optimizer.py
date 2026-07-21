#!/usr/bin/env python3
"""
ats_optuna_optimizer.py

Bayesian (Optuna/TPE) threshold optimization for AtsFastReversal /
AtsSlowReversal entry-filter parameters.

Why this instead of the grid sweep (ats_param_optimizer.py)?
  - The grid sweep tests parameters one or two at a time. Optuna can search
    ALL candidate parameters jointly in one search -- letting the optimizer
    discover a 3-, 4-, or 5-way combination gate, and deciding for itself
    which parameters to include (via a "none" choice per parameter) rather
    than requiring you to hand-pick which pairs to test.

How overfitting is controlled (read this before trusting the output):
  1. CHRONOLOGICAL TRAIN/TEST SPLIT. Trades are sorted by EntryDate. Optuna
     only ever sees the training window while searching. The recommended
     filter is then evaluated -- once -- on the held-out test window. The
     test-window numbers are the only numbers that mean anything; the
     training-window numbers are what Optuna was allowed to fit to, and
     will always look good by construction.
  2. MIN-N GUARD. Any candidate filter that drops the training subset below
     `--min-n` trades is rejected (heavily penalized) during the search.
  3. K-FOLD CROSS-VALIDATION WITHIN TRAINING DATA (optional, --cv-folds > 1).
     Instead of optimizing raw training expectancy (which a large enough
     search will always eventually overfit), the objective can average
     expectancy across K folds of the training data, which penalizes filters
     that only work on a lucky subset of the training window.

NEW: Entry-path stratification. If your strategy enters via
`(PatternEntryScore >= Min Or CVDEntryScore >= Min)`, pass
--min-pattern-score and --min-cvd-score to additionally search each
direction's pattern_only / cvd_only / both subsets independently -- a
parameter that matters for Pattern-triggered trades may not matter (or may
even point the other way) for CVD-triggered ones.

Usage:
    python ats_optuna_optimizer.py trades.csv
    python ats_optuna_optimizer.py trades.csv --n-trials 2000 --min-n 30
    python ats_optuna_optimizer.py trades.csv --test-fraction 0.3 --cv-folds 5
    python ats_optuna_optimizer.py trades.csv --params ind_FullDeltaATRs,ind_FullAngle,ind_ATRsFromHma
    python ats_optuna_optimizer.py trades.csv --output optuna_report.json
    python ats_optuna_optimizer.py trades.csv --min-pattern-score 3 --min-cvd-score 4

Requires: pandas, numpy, optuna
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("This script requires optuna: pip install optuna --break-system-packages", file=sys.stderr)
    raise

NON_PARAM_COLUMNS = {
    "Symbol", "EntryDate", "EntryTime", "EntryName", "EntryPrice", "ExitDate",
    "ExitTime", "ExitName", "ExitPrice", "Shares", "Profit/Loss", "BarNumber",
    "SignalBar", "R/T", "ind_BarDate", "ind_BarTime", "ind_BarNumber",
    "ind_Tick", "ind_SignalSent", "ind_Close", "ind_R/T", "ind_computertime",
    "ProfitHit", "ind_Interval","ind_Tick","ind_BarDate","ind_BarTime","ind_BarNumber"

}
BOOLEAN_FLAG_PREFIX = "ind_C"
MIN_N_DEFAULT = 30


@dataclass
class FilterClause:
    param: str
    direction: str   # "ge" or "le"
    threshold: float

    def __str__(self):
        sym = ">=" if self.direction == "ge" else "<="
        return f"{self.param}{sym}{self.threshold:g}"


@dataclass
class OptimizationResult:
    direction_label: str          # "long" / "short"
    clauses: list                  # list[FilterClause]
    total_train_n: int             # full training window size (before any filter)
    total_test_n: int              # full held-out test window size (before any filter)
    train_n: int                   # trades matching the found filter, within training window
    train_expectancy: float
    train_hit_rate: float
    test_n: int                    # trades matching the found filter, within test window
    test_expectancy: float
    test_hit_rate: float
    test_total_pl: float
    baseline_train_expectancy: float
    baseline_test_expectancy: float
    n_trials_run: int
    warning: Optional[str] = None


def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Profit/Loss" not in df.columns or "ind_SignalSent" not in df.columns:
        raise ValueError("CSV must have 'Profit/Loss' and 'ind_SignalSent' columns.")
    df["ProfitHit"] = df["Profit/Loss"] > 0
    if "EntryDate" in df.columns:
        df["EntryDate"] = pd.to_datetime(df["EntryDate"])
        df = df.sort_values("EntryDate").reset_index(drop=True)
    else:
        warnings.warn("No EntryDate column found -- using row order as the chronological "
                       "proxy for the train/test split. If your CSV isn't already in "
                       "chronological order, the train/test split below is not meaningful.")
    return df


# --------------------------------------------------------------------------
# Entry-path stratification (see ats_feature_importance.py for the same
# pattern/rationale): if your strategy fires on
# (PatternEntryScore >= Min Or CVDEntryScore >= Min), split each direction
# into pattern_only / cvd_only / both subsets and run the same optimizer on
# each independently.
# --------------------------------------------------------------------------
PATTERN_SCORE_COL_CANDIDATES = ["ind_PatternEntryScore", "ind_SpeedEntryScore"]
CVD_SCORE_COL_CANDIDATES = ["ind_CVDEntryScore"]


def autodetect_column(df: pd.DataFrame, candidates: list, override: Optional[str] = None) -> Optional[str]:
    if override:
        return override if override in df.columns else None
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def resolve_entry_path_config(df: pd.DataFrame, args) -> Optional[dict]:
    if args.min_pattern_score is None or args.min_cvd_score is None:
        return None
    pattern_col = autodetect_column(df, PATTERN_SCORE_COL_CANDIDATES, args.pattern_score_col)
    cvd_col = autodetect_column(df, CVD_SCORE_COL_CANDIDATES, args.cvd_score_col)
    if pattern_col is None or cvd_col is None:
        missing = []
        if pattern_col is None:
            missing.append(f"pattern score column (tried {args.pattern_score_col or PATTERN_SCORE_COL_CANDIDATES})")
        if cvd_col is None:
            missing.append(f"cvd score column (tried {args.cvd_score_col or CVD_SCORE_COL_CANDIDATES})")
        print(f"WARNING: --min-pattern-score/--min-cvd-score given but couldn't find: {'; '.join(missing)}. "
              f"Skipping entry-path stratification.")
        return None
    return {"pattern_col": pattern_col, "cvd_col": cvd_col,
            "min_pattern": args.min_pattern_score, "min_cvd": args.min_cvd_score}


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


def get_param_columns(df: pd.DataFrame, explicit: Optional[list] = None) -> list:
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Requested parameters not found in CSV: {missing}")
        return explicit
    params = []
    for c in df.columns:
        if c in NON_PARAM_COLUMNS:
            continue
        if c.startswith(BOOLEAN_FLAG_PREFIX) and c[len(BOOLEAN_FLAG_PREFIX):].isdigit():
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) < 3:
            continue
        params.append(c)
    return params


def chronological_split(d: pd.DataFrame, test_fraction: float) -> tuple:
    n = len(d)
    n_test = max(1, int(round(n * test_fraction)))
    n_train = n - n_test
    return d.iloc[:n_train].reset_index(drop=True), d.iloc[n_train:].reset_index(drop=True)


def apply_clauses(d: pd.DataFrame, clauses: list) -> pd.Series:
    mask = pd.Series(True, index=d.index)
    for c in clauses:
        col = d[c.param]
        mask &= (col >= c.threshold) if c.direction == "ge" else (col <= c.threshold)
    return mask


def kfold_indices(n: int, k: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def make_objective(train_df: pd.DataFrame, params: list, min_n: int, cv_folds: int):
    pl = train_df["Profit/Loss"].to_numpy(dtype=float)
    param_arrays = {p: train_df[p].to_numpy(dtype=float) for p in params}
    param_ranges = {p: (float(np.nanmin(param_arrays[p])), float(np.nanmax(param_arrays[p]))) for p in params}
    n = len(train_df)

    fold_splits = list(kfold_indices(n, cv_folds)) if cv_folds > 1 else None

    def build_mask(trial, idx=None) -> np.ndarray:
        if idx is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
        for p in params:
            choice = trial.suggest_categorical(f"{p}__dir", ["none", "ge", "le"])
            if choice == "none":
                continue
            lo, hi = param_ranges[p]
            if lo == hi:
                continue
            thresh = trial.suggest_float(f"{p}__thresh", lo, hi)
            vals = param_arrays[p]
            if choice == "ge":
                mask &= (vals >= thresh)
            else:
                mask &= (vals <= thresh)
        return mask

    def objective(trial):
        if fold_splits is None:
            mask = build_mask(trial)
            cnt = int(mask.sum())
            if cnt < min_n:
                return -1_000_000.0 + cnt  # steer search toward feasible region
            return float(pl[mask].mean())
        else:
            # Build the mask ONCE against the full training set using the trial's
            # suggested thresholds, then evaluate expectancy fold-by-fold so the
            # same filter has to work across multiple slices of the training data,
            # not just the training set as a whole.
            full_mask = build_mask(trial)
            fold_expectancies = []
            for train_idx, val_idx in fold_splits:
                val_mask = full_mask[val_idx]
                cnt = int(val_mask.sum())
                if cnt < max(5, min_n // cv_folds):
                    return -1_000_000.0 + cnt
                fold_expectancies.append(float(pl[val_idx][val_mask].mean()))
            return float(np.mean(fold_expectancies))

    return objective


def clauses_from_trial(trial_params: dict, params: list) -> list:
    clauses = []
    for p in params:
        choice = trial_params.get(f"{p}__dir")
        if choice in (None, "none"):
            continue
        thresh = trial_params.get(f"{p}__thresh")
        clauses.append(FilterClause(param=p, direction=choice, threshold=float(thresh)))
    return clauses


def optimize_direction(df: pd.DataFrame, label: str, params: list, min_n: int,
                        n_trials: int, test_fraction: float, cv_folds: int,
                        seed: int) -> OptimizationResult:
    train_df, test_df = chronological_split(df, test_fraction)
    baseline_train_exp = float(train_df["Profit/Loss"].mean()) if len(train_df) else float("nan")
    baseline_test_exp = float(test_df["Profit/Loss"].mean()) if len(test_df) else float("nan")

    if len(train_df) < min_n:
        return OptimizationResult(
            direction_label=label, clauses=[], total_train_n=len(train_df), total_test_n=len(test_df),
            train_n=len(train_df),
            train_expectancy=baseline_train_exp, train_hit_rate=float(train_df["ProfitHit"].mean()) if len(train_df) else float("nan"),
            test_n=len(test_df), test_expectancy=baseline_test_exp,
            test_hit_rate=float(test_df["ProfitHit"].mean()) if len(test_df) else float("nan"),
            test_total_pl=float(test_df["Profit/Loss"].sum()) if len(test_df) else 0.0,
            baseline_train_expectancy=baseline_train_exp, baseline_test_expectancy=baseline_test_exp,
            n_trials_run=0, warning=f"Training set ({len(train_df)}) is smaller than --min-n ({min_n}); skipped optimization.",
        )

    objective = make_objective(train_df, params, min_n, cv_folds)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    clauses = clauses_from_trial(study.best_trial.params, params)

    train_mask = apply_clauses(train_df, clauses)
    train_n = int(train_mask.sum())
    train_exp = float(train_df.loc[train_mask, "Profit/Loss"].mean()) if train_n else float("nan")
    train_hit = float(train_df.loc[train_mask, "ProfitHit"].mean()) if train_n else float("nan")

    warning = None
    if len(test_df) == 0:
        test_n, test_exp, test_hit, test_pl = 0, float("nan"), float("nan"), 0.0
        warning = "No held-out test data available (test_fraction too small or dataset too small)."
    else:
        test_mask = apply_clauses(test_df, clauses)
        test_n = int(test_mask.sum())
        if test_n == 0:
            test_exp, test_hit, test_pl = float("nan"), float("nan"), 0.0
            warning = "Filter matched ZERO trades in the held-out test window -- cannot confirm this filter."
        else:
            test_exp = float(test_df.loc[test_mask, "Profit/Loss"].mean())
            test_hit = float(test_df.loc[test_mask, "ProfitHit"].mean())
            test_pl = float(test_df.loc[test_mask, "Profit/Loss"].sum())
            if test_n < min_n:
                warning = f"Held-out test subset only has {test_n} trades (< --min-n {min_n}) -- low confidence."

    return OptimizationResult(
        direction_label=label, clauses=clauses, total_train_n=len(train_df), total_test_n=len(test_df),
        train_n=train_n, train_expectancy=train_exp,
        train_hit_rate=train_hit, test_n=test_n, test_expectancy=test_exp, test_hit_rate=test_hit,
        test_total_pl=test_pl, baseline_train_expectancy=baseline_train_exp,
        baseline_test_expectancy=baseline_test_exp, n_trials_run=n_trials, warning=warning,
    )


def print_result(r: OptimizationResult, min_n: int):
    print(f"\n{'='*72}")
    print(f" {r.direction_label.upper()}")
    print(f"{'='*72}")
    print(f"Training window: {r.total_train_n} trades total, baseline_expectancy=${r.baseline_train_expectancy:.2f}")
    print(f"Test window (held-out, never seen by optimizer): {r.total_test_n} trades total, "
          f"baseline_expectancy=${r.baseline_test_expectancy:.2f}")

    if r.n_trials_run == 0:
        print(f"\n  SKIPPED: {r.warning}")
        return

    if not r.clauses:
        print("\n  Optuna found no filter that beat baseline expectancy -- "
              "recommendation is to leave parameters unfiltered for this direction.")
        return

    print(f"\nOptuna-found filter ({r.n_trials_run} trials searched):")
    for c in r.clauses:
        print(f"    {c}")

    print(f"\n  TRAIN performance (subset matching the filter, within the {r.total_train_n}-trade training window):")
    print(f"    n={r.train_n}  hit_rate={r.train_hit_rate:.1%}  expectancy=${r.train_expectancy:.2f}/trade")

    print(f"\n  >>> TEST performance (subset matching the filter, within the {r.total_test_n}-trade "
          f"held-out window -- the only trustworthy number) <<<")
    if r.test_n == 0:
        print(f"    n=0 -- filter never fired in the test window. Cannot confirm.")
    else:
        confidence = "OK" if r.test_n >= min_n else f"LOW CONFIDENCE (n={r.test_n} < min-n={min_n})"
        print(f"    n={r.test_n}  hit_rate={r.test_hit_rate:.1%}  expectancy=${r.test_expectancy:.2f}/trade  "
              f"total_pl=${r.test_total_pl:.2f}  [{confidence}]")
        train_test_gap = r.train_expectancy - r.test_expectancy
        if train_test_gap > 0.5 * abs(r.train_expectancy) and r.train_expectancy > 0:
            print(f"    NOTE: expectancy dropped substantially from train to test "
                  f"(${r.train_expectancy:.2f} -> ${r.test_expectancy:.2f}). This is a sign "
                  f"of overfitting to the training window -- treat this filter with skepticism.")

    if r.warning:
        print(f"\n  WARNING: {r.warning}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Path to the merged trade CSV")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT,
                    help=f"Minimum trades required in any filtered subset (default {MIN_N_DEFAULT})")
    ap.add_argument("--n-trials", type=int, default=1000,
                    help="Number of Optuna trials per direction (default 1000)")
    ap.add_argument("--test-fraction", type=float, default=0.25,
                    help="Fraction of trades (chronologically last) held out as the test set (default 0.25)")
    ap.add_argument("--cv-folds", type=int, default=1,
                    help="If > 1, cross-validate the objective across this many folds of the "
                         "training data instead of using raw training expectancy (default 1 = off)")
    ap.add_argument("--params", default=None,
                    help="Comma-separated list of parameter columns to search over "
                         "(default: auto-detect all numeric ind_* columns)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output", default=None, help="Optional path to write the full JSON report")
    add_entry_path_args(ap)
    args = ap.parse_args()

    df = load_trades(args.csv_path)
    explicit_params = [p.strip() for p in args.params.split(",")] if args.params else None
    params = get_param_columns(df, explicit_params)

    long_df = df[df["ind_SignalSent"] == 1].reset_index(drop=True)
    short_df = df[df["ind_SignalSent"] == -1].reset_index(drop=True)

    print(f"Loaded {len(df)} trades from {args.csv_path}")
    print(f"  Long: {len(long_df)}   Short: {len(short_df)}")
    print(f"  Searching over {len(params)} parameters: {params}")
    print(f"  Train/test split: {1-args.test_fraction:.0%}/{args.test_fraction:.0%} chronological, "
          f"{args.n_trials} trials, cv_folds={args.cv_folds}")

    path_cfg = resolve_entry_path_config(df, args)
    if path_cfg:
        print(f"  Entry-path stratification ON: pattern_col={path_cfg['pattern_col']} "
              f"(>= {path_cfg['min_pattern']}), cvd_col={path_cfg['cvd_col']} (>= {path_cfg['min_cvd']})")

    def run_one(sub_df: pd.DataFrame, label: str, seed_offset: int, use_params: list):
        r = optimize_direction(sub_df, label, use_params, args.min_n, args.n_trials,
                                args.test_fraction, args.cv_folds, args.seed + seed_offset)
        print_result(r, args.min_n)
        return r

    long_result = run_one(long_df, "long", 0, params)
    short_result = run_one(short_df, "short", 1, params)

    by_entry_path = {}
    if path_cfg:
        params_excl_pattern = [p for p in params if p != path_cfg["pattern_col"]]
        params_excl_cvd = [p for p in params if p != path_cfg["cvd_col"]]
        params_by_path = {
            "pattern_only": params_excl_pattern,
            "cvd_only": params_excl_cvd,
            "both": params,
            "neither": params,
        }

        print(f"\n{'#'*72}")
        print(" ENTRY-PATH BREAKDOWN")
        print(f"{'#'*72}")
        seed_offset = 2
        for direction_label, direction_df in [("long", long_df), ("short", short_df)]:
            path_series = compute_entry_path(direction_df, path_cfg["pattern_col"], path_cfg["cvd_col"],
                                               path_cfg["min_pattern"], path_cfg["min_cvd"])
            by_entry_path[direction_label] = {}
            for path_name in ["pattern_only", "cvd_only", "both", "neither"]:
                sub = direction_df[path_series == path_name].reset_index(drop=True)
                if path_name == "neither" and len(sub) == 0:
                    continue
                sub_label = f"{direction_label} ({path_name})"
                if len(sub) == 0:
                    print(f"\n{'='*72}\n {sub_label.upper()}\n{'='*72}\n  n=0 -- no trades in this bucket.")
                    continue
                r = run_one(sub, sub_label, seed_offset, params_by_path[path_name])
                by_entry_path[direction_label][path_name] = r
                seed_offset += 1

    print(f"\n{'='*72}")
    print(" SUMMARY")
    print(f"{'='*72}")
    print("Only the TEST-window numbers above should inform any parameter change.")
    print("If a filter's test expectancy is close to or below its train expectancy drop-off,")
    print("or the test warning flags low confidence / zero matches, treat it as unconfirmed")
    print("and gather more data before changing the live strategy's parameters.")
    if path_cfg:
        print("If the best filter/threshold differs between pattern_only and cvd_only for the")
        print("same direction, that's a real reason to tune those two entry paths separately")
        print("rather than share one parameter set across both.")

    if args.output:
        report = {
            "csv_path": args.csv_path, "min_n": args.min_n, "n_trials": args.n_trials,
            "test_fraction": args.test_fraction, "cv_folds": args.cv_folds,
            "long": {**asdict(long_result), "clauses": [str(c) for c in long_result.clauses]},
            "short": {**asdict(short_result), "clauses": [str(c) for c in short_result.clauses]},
        }
        if path_cfg:
            report["entry_path_config"] = path_cfg
            report["by_entry_path"] = {
                direction: {
                    path_name: {**asdict(r), "clauses": [str(c) for c in r.clauses]}
                    for path_name, r in paths.items()
                }
                for direction, paths in by_entry_path.items()
            }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()
