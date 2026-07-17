#!/usr/bin/env python3
"""
ats_entryscore_weight_optimizer.py

Optimizes the per-component WEIGHTS (and the score cutoff) of the
AtsFastReversal / AtsSlowReversal EntryScore formula:

    EntryScore = W1 * IFF(C5, 1, 0)                                  // Speed flip
               + W2 * IFF(RevATRsPerSec > RevATRsPerSecLim, 1, 0)    // Fast reversal   (== C13)
               + W3 * IFF(CVDAvg >= CVDAvgLim, 1, 0)                 // Volume directional (== C6)
               + W4 * IFF(CountIf(CVDAcel>=CVDAcelLim,2)>0, 1, 0)    // Accel last 2 bars  (== C10)
               + W5 * IFF(CVDDeltaPct>0, 1, 0)                          // Delta confirms
               + W6 * IFF(C7, 1, 0)                                  // PipSpeedPct >= HMinPipSpeedPct
               + W7 * IFF(C9, 1, 0)                                  // HMAGapCV <= HMinHMAGapCV

    Take the trade if EntryScore >= Threshold

Currently W1..W7 = 1 for all components. This script searches for better
weights AND a matching score threshold using Optuna (TPE), using the SAME
overfitting controls as ats_optuna_optimizer.py:

  1. CHRONOLOGICAL TRAIN/TEST SPLIT. Weights and threshold are only ever
     fit on the training window (earliest trades). They are then evaluated
     ONCE on the held-out test window. Only the test-window numbers should
     inform a live parameter change.
  2. MIN-N GUARD. Any weight/threshold combination that drops the training
     subset below --min-n trades is rejected during the search.
  3. Optional k-fold cross-validation within the training window (--cv-folds).

This is a DIFFERENT search problem than ats_optuna_optimizer.py, which finds
independent >=/<= thresholds on continuous columns. This script instead
searches a shared integer weight per binary component plus one score cutoff
-- the actual decision variables in your EntryScore formula.

Component columns are auto-detected from your CSV's own C-flags (C5, C13, C6,
C10, C7, C9) plus a direct CVDDeltaPct > 0 check, matching your formula exactly.
Override with --components if your column names differ.

Usage:
    python ats_entryscore_weight_optimizer.py trades.csv
    python ats_entryscore_weight_optimizer.py trades.csv --n-trials 3000 --min-n 20
    python ats_entryscore_weight_optimizer.py trades.csv --max-weight 3 --cv-folds 3
    python ats_entryscore_weight_optimizer.py trades.csv --output entryscore_report.json

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

MIN_N_DEFAULT = 30

# Default EntryScore components: (label, column, comparison)
# comparison is one of: "flag" (column is already 0/1), "gt0" (column > 0)
DEFAULT_COMPONENTS = [
    ("Speed flip (C5)",              "ind_C5",       "flag"),
    ("Fast reversal (C13)",          "ind_C13",      "flag"),
    ("Volume directional (C6)",      "ind_C6",       "flag"),
    ("CVD accel last 2 bars (C10)",  "ind_C10",      "flag"),
    ("CVDDeltaPct confirms",           "ind_CVDDeltaPct", "gt0"),
    ("PipSpeedPct sustained (C7)",   "ind_C7",       "flag"),
    ("HMAGapCV consistent (C9)",     "ind_C9",       "flag"),
]


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
        warnings.warn("No EntryDate column -- using row order as the chronological proxy "
                       "for the train/test split.")
    return df


def parse_components_arg(spec: str) -> list:
    """Parses '--components' override string like:
    'Label1:col1:flag,Label2:col2:gt0'
    """
    comps = []
    for part in spec.split(","):
        label, col, cmp = part.split(":")
        if cmp not in ("flag", "gt0"):
            raise ValueError(f"Unknown comparison '{cmp}' for component '{label}' (use 'flag' or 'gt0')")
        comps.append((label.strip(), col.strip(), cmp))
    return comps


def build_component_matrix(df: pd.DataFrame, components: list) -> np.ndarray:
    cols = []
    for label, col, cmp in components:
        if col not in df.columns:
            raise ValueError(f"Component column '{col}' (for '{label}') not found in CSV.")
        if cmp == "flag":
            cols.append((df[col].fillna(0) != 0).astype(int).to_numpy())
        elif cmp == "gt0":
            cols.append((df[col].fillna(0) > 0).astype(int).to_numpy())
    return np.column_stack(cols)  # shape (n_trades, n_components)


def chronological_split(d: pd.DataFrame, test_fraction: float) -> tuple:
    n = len(d)
    if test_fraction <= 0:
        return d.reset_index(drop=True), d.iloc[0:0].reset_index(drop=True)
    n_test = max(1, int(round(n * test_fraction)))
    n_train = n - n_test
    return d.iloc[:n_train].reset_index(drop=True), d.iloc[n_train:].reset_index(drop=True)


def kfold_indices(n: int, k: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def best_equal_weight_threshold(comp_matrix: np.ndarray, pl: np.ndarray, min_n: int) -> tuple:
    """Baseline: current formula (all weights = 1). Finds the best score cutoff
    for THIS baseline, so the optimized weights have a fair comparison point
    (not just compared against taking every trade)."""
    scores = comp_matrix.sum(axis=1)
    max_score = comp_matrix.shape[1]
    best = None
    for t in range(1, max_score + 1):
        mask = scores >= t
        n = int(mask.sum())
        if n < min_n:
            continue
        exp = float(pl[mask].mean())
        if best is None or exp > best[0]:
            best = (exp, t, n)
    if best is None:
        # fall back to no filter if nothing clears min_n
        return float(pl.mean()), 0, len(pl)
    return best[0], best[1], best[2]


def make_objective(train_df: pd.DataFrame, comp_matrix: np.ndarray, labels: list,
                    min_n: int, cv_folds: int, max_weight: int, min_threshold_frac: float):
    pl = train_df["Profit/Loss"].to_numpy(dtype=float)
    n = len(train_df)
    fold_splits = list(kfold_indices(n, cv_folds)) if cv_folds > 1 else None

    def objective(trial):
        weights = np.array([trial.suggest_int(f"w_{i}", 0, max_weight) for i in range(len(labels))])
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
        low = max(1, int(np.ceil(min_threshold_frac * max_score)))
        if low > max_score:
            return -1_000_000.0
        threshold = trial.suggest_int("threshold", low, max_score)
        scores = comp_matrix @ weights

        if fold_splits is None:
            mask = scores >= threshold
            cnt = int(mask.sum())
            if cnt < min_n:
                return -1_000_000.0 + cnt
            return float(pl[mask].mean())
        else:
            fold_expectancies = []
            for train_idx, val_idx in fold_splits:
                val_mask = scores[val_idx] >= threshold
                cnt = int(val_mask.sum())
                if cnt < max(5, min_n // cv_folds):
                    return -1_000_000.0 + cnt
                fold_expectancies.append(float(pl[val_idx][val_mask].mean()))
            return float(np.mean(fold_expectancies))

    return objective


def optimize_direction(df: pd.DataFrame, label: str, components: list, min_n: int,
                        n_trials: int, test_fraction: float, cv_folds: int,
                        max_weight: int, seed: int, min_threshold_frac: float) -> WeightResult:
    labels = [c[0] for c in components]
    train_df, test_df = chronological_split(df, test_fraction)

    baseline_train_exp_all = float(train_df["Profit/Loss"].mean()) if len(train_df) else float("nan")
    baseline_test_exp_all = float(test_df["Profit/Loss"].mean()) if len(test_df) else float("nan")

    if len(train_df) < min_n:
        return WeightResult(
            direction_label=label, total_train_n=len(train_df), total_test_n=len(test_df),
            weights={}, threshold=0, max_possible_score=0,
            train_n=len(train_df), train_expectancy=baseline_train_exp_all,
            train_hit_rate=float(train_df["ProfitHit"].mean()) if len(train_df) else float("nan"),
            test_n=len(test_df), test_expectancy=baseline_test_exp_all,
            test_hit_rate=float(test_df["ProfitHit"].mean()) if len(test_df) else float("nan"),
            test_total_pl=float(test_df["Profit/Loss"].sum()) if len(test_df) else 0.0,
            baseline_train_expectancy=baseline_train_exp_all, baseline_test_expectancy=baseline_test_exp_all,
            equal_weight_best_threshold=0, n_trials_run=0,
            warning=f"Training set ({len(train_df)}) is smaller than --min-n ({min_n}); skipped.",
        )

    train_matrix = build_component_matrix(train_df, components)
    test_matrix = build_component_matrix(test_df, components) if len(test_df) else np.zeros((0, len(components)))
    train_pl = train_df["Profit/Loss"].to_numpy(dtype=float)
    test_pl = test_df["Profit/Loss"].to_numpy(dtype=float) if len(test_df) else np.array([])

    # Fair baseline: current equal-weight formula, but with its OWN best threshold
    # (found only on training data), evaluated the same way as the optimized weights.
    baseline_exp, baseline_thresh, baseline_n_train = best_equal_weight_threshold(train_matrix, train_pl, min_n)
    if len(test_df) and baseline_thresh > 0:
        baseline_scores_test = test_matrix.sum(axis=1)
        baseline_mask_test = baseline_scores_test >= baseline_thresh
        baseline_test_exp = float(test_pl[baseline_mask_test].mean()) if baseline_mask_test.sum() else float("nan")
    else:
        baseline_test_exp = baseline_test_exp_all

    objective = make_objective(train_df, train_matrix, labels, min_n, cv_folds, max_weight, min_threshold_frac)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    weights = np.array([best_params[f"w_{i}"] for i in range(len(labels))])
    threshold = int(best_params["threshold"])
    weight_dict = dict(zip(labels, [int(w) for w in weights]))

    train_scores = train_matrix @ weights
    train_mask = train_scores >= threshold
    train_n = int(train_mask.sum())
    train_exp = float(train_pl[train_mask].mean()) if train_n else float("nan")
    train_hit = float(train_df["ProfitHit"].to_numpy()[train_mask].mean()) if train_n else float("nan")

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
            test_hit = float(test_df["ProfitHit"].to_numpy()[test_mask].mean())
            test_pl_sum = float(test_pl[test_mask].sum())
            if test_n < min_n:
                warning = f"Held-out test subset only has {test_n} trades (< --min-n {min_n}) -- low confidence."

    return WeightResult(
        direction_label=label, total_train_n=len(train_df), total_test_n=len(test_df),
        weights=weight_dict, threshold=threshold, max_possible_score=int(weights.sum()),
        train_n=train_n, train_expectancy=train_exp, train_hit_rate=train_hit,
        test_n=test_n, test_expectancy=test_exp, test_hit_rate=test_hit, test_total_pl=test_pl_sum,
        baseline_train_expectancy=baseline_exp, baseline_test_expectancy=baseline_test_exp,
        equal_weight_best_threshold=baseline_thresh, n_trials_run=n_trials, warning=warning,
    )


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
        print(f"    {label:<32} weight = {w:<3} {bar}")
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
                    help="Override the default component list. Format: "
                         "'Label1:col1:flag,Label2:col2:gt0,...' where comparison is "
                         "'flag' (column already 0/1) or 'gt0' (column > 0).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output", default=None, help="Optional path to write the full JSON report")
    args = ap.parse_args()

    df = load_trades(args.csv_path)
    components = parse_components_arg(args.components) if args.components else DEFAULT_COMPONENTS

    long_df = df[df["ind_SignalSent"] == 1].reset_index(drop=True)
    short_df = df[df["ind_SignalSent"] == -1].reset_index(drop=True)

    print(f"Loaded {len(df)} trades from {args.csv_path}")
    print(f"  Long: {len(long_df)}   Short: {len(short_df)}")
    print(f"  EntryScore components ({len(components)}):")
    for label, col, cmp in components:
        print(f"    {label:<32} <- {col} ({'!=0' if cmp=='flag' else '>0'})")
    print(f"  max_weight={args.max_weight}, n_trials={args.n_trials}, "
          f"test_fraction={args.test_fraction}, cv_folds={args.cv_folds}")

    long_result = optimize_direction(long_df, "long", components, args.min_n, args.n_trials,
                                      args.test_fraction, args.cv_folds, args.max_weight, args.seed,
                                      args.min_threshold_frac)
    short_result = optimize_direction(short_df, "short", components, args.min_n, args.n_trials,
                                       args.test_fraction, args.cv_folds, args.max_weight, args.seed + 1,
                                       args.min_threshold_frac)

    print_result(long_result, args.min_n)
    print_result(short_result, args.min_n)

    print(f"\n{'='*72}")
    print(" SUMMARY")
    print(f"{'='*72}")
    print("Only the TEST-window expectancy and the 'vs. equal-weight formula' line above")
    print("should inform any change to the live EntryScore weights. A positive number there")
    print("means the reweighted formula beat the current equal-weight formula on data neither")
    print("saw during the search. A filter with n=0 or LOW CONFIDENCE in the test window is")
    print("not yet confirmed -- gather more trades before changing the live weights.")

    if args.output:
        report = {
            "csv_path": args.csv_path, "min_n": args.min_n, "n_trials": args.n_trials,
            "test_fraction": args.test_fraction, "cv_folds": args.cv_folds, "max_weight": args.max_weight,
            "min_threshold_frac": args.min_threshold_frac,
            "long": asdict(long_result), "short": asdict(short_result),
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()
