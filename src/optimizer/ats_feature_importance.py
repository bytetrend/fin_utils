#!/usr/bin/env python3
"""
ats_feature_importance.py

Trains a Random Forest (or Gradient Boosting) to predict win/loss from your ind_* columns, cross-validated
Reports AUC first — if it's near 0.5 (or even below it), it says explicitly that the ranking below is noise, not signal
Uses permutation importance (unbiased, unlike raw impurity importance) plus an independent SHAP cross-check
On your data: AUC came in at 0.369 (long) and 0.358 (short) — both below chance, which the script now explicitly flags as "sampling noise, not an inverse signal," not something to act on

Ranks which entry-filter parameters matter most for AtsFastReversal /
AtsSlowReversal trade outcomes, using cross-validated classifier feature
importance (Random Forest + permutation importance) and, if the `shap`
package is installed, SHAP values for interaction / directional effects.

Why this is a DIFFERENT tool than the threshold sweeps:
  - ats_param_optimizer.py and ats_optuna_optimizer.py search for the specific
    threshold/combo that maximizes expectancy.
  - This script instead asks a more modest question first: "does this
    parameter carry any real information about win/loss at all, and how
    much, relative to the others?" That ranking is a sanity check you should
    run BEFORE trusting a threshold search -- if a parameter has near-zero
    importance here, a threshold found for it by the other scripts is very
    likely noise.

Honesty guardrails baked in:
  - Reports cross-validated ROC-AUC first. If AUC is close to 0.5, the
    model has no real predictive edge over guessing, and the importance
    ranking below it is not meaningful -- the script says so explicitly.
  - Uses permutation importance (measured on held-out folds), not raw
    impurity-based importance, since impurity importance is biased toward
    high-cardinality continuous columns regardless of whether they're
    predictive.
  - Long and short are modeled and reported separately.
  - Small-sample warning if a direction has too few trades for a stable
    model (default threshold: 60 trades minimum to attempt modeling).

Usage:
    python ats_feature_importance.py trades.csv
    python ats_feature_importance.py trades.csv --model gradient_boosting --cv-folds 5
    python ats_feature_importance.py trades.csv --params ind_FullDeltaATRs,ind_FullAngle,ind_ATRsFromHma
    python ats_feature_importance.py trades.csv --output importance_report.json
    python ats_feature_importance.py trades.csv --no-shap   # skip SHAP even if installed

Requires: pandas, numpy, scikit-learn. Optional: shap (for interaction/direction plots data).
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

NON_PARAM_COLUMNS = {
    "Symbol", "EntryDate", "EntryTime", "EntryName", "EntryPrice", "ExitDate",
    "ExitTime", "ExitName", "ExitPrice", "Shares", "Profit/Loss", "BarNumber",
    "SignalBar", "R/T", "ind_BarDate", "ind_BarTime", "ind_BarNumber",
    "ind_Tick", "ind_SignalSent", "ind_Close", "ind_R/T", "ind_computertime",
    "ProfitHit",
}
MIN_TRADES_FOR_MODEL = 60


@dataclass
class ParamImportance:
    param: str
    permutation_importance_mean: float
    permutation_importance_std: float
    direction_of_effect: str      # "higher = more likely to win", "lower = ...", "unclear"
    hit_median: float
    loss_median: float


@dataclass
class DirectionModelResult:
    direction_label: str
    n: int
    baseline_hit_rate: float
    cv_auc: float
    cv_auc_std: float
    has_signal: bool
    model_used: str
    importances: list             # list[ParamImportance], sorted descending
    warning: Optional[str] = None


def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Profit/Loss" not in df.columns or "ind_SignalSent" not in df.columns:
        raise ValueError("CSV must have 'Profit/Loss' and 'ind_SignalSent' columns.")
    df["ProfitHit"] = (df["Profit/Loss"] > 0).astype(int)
    return df


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
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) < 2:
            continue
        params.append(c)
    return params


def build_model(kind: str, seed: int):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, max_depth=4, min_samples_leaf=8,
            random_state=seed, n_jobs=-1, class_weight="balanced",
        )
    elif kind == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def analyze_direction(df: pd.DataFrame, label: str, params: list, model_kind: str,
                       cv_folds: int, seed: int, run_shap: bool) -> DirectionModelResult:
    n = len(df)
    baseline_hit = float(df["ProfitHit"].mean()) if n else float("nan")

    if n < MIN_TRADES_FOR_MODEL:
        return DirectionModelResult(
            direction_label=label, n=n, baseline_hit_rate=baseline_hit,
            cv_auc=float("nan"), cv_auc_std=float("nan"), has_signal=False,
            model_used=model_kind, importances=[],
            warning=f"Only {n} trades (< {MIN_TRADES_FOR_MODEL}) -- too few to fit a stable "
                    f"model. Skipping. Gather more trades before modeling this direction.",
        )

    X_raw = df[params].to_numpy(dtype=float)
    y = df["ProfitHit"].to_numpy(dtype=int)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    n_folds = min(cv_folds, min(np.bincount(y)))
    if n_folds < 2:
        return DirectionModelResult(
            direction_label=label, n=n, baseline_hit_rate=baseline_hit,
            cv_auc=float("nan"), cv_auc_std=float("nan"), has_signal=False,
            model_used=model_kind, importances=[],
            warning="Not enough examples of one outcome class to cross-validate. Skipping.",
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Cross-validated AUC: fit on each train fold, score probability on held-out fold.
    fold_aucs = []
    oof_proba = np.zeros(n)
    for train_idx, test_idx in skf.split(X, y):
        model = build_model(model_kind, seed)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = proba
        if len(np.unique(y[test_idx])) > 1:
            fold_aucs.append(roc_auc_score(y[test_idx], proba))

    cv_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    cv_auc_std = float(np.std(fold_aucs)) if fold_aucs else float("nan")
    # Only count it as real signal if the model does BETTER than chance.
    # AUC noticeably BELOW 0.5 is not an "inverse" signal to trust -- with
    # small per-fold sample sizes it's just what noise looks like, and
    # treating it as informative is the same mistake as chasing a lucky
    # threshold. Both get reported, but only above-chance AUC is flagged
    # as usable.
    has_signal = (not np.isnan(cv_auc)) and (cv_auc - 0.5) >= 0.05

    # Fit a final model on all data to compute permutation importance
    # (measured via cross-validated permutation importance on held-out folds
    # for an honest estimate rather than a single in-sample fit).
    perm_importances = {p: [] for p in params}
    for train_idx, test_idx in skf.split(X, y):
        model = build_model(model_kind, seed)
        model.fit(X[train_idx], y[train_idx])
        if len(np.unique(y[test_idx])) < 2:
            continue
        result = permutation_importance(
            model, X[test_idx], y[test_idx], n_repeats=10, random_state=seed,
            scoring="roc_auc", n_jobs=-1,
        )
        for i, p in enumerate(params):
            perm_importances[p].append(result.importances_mean[i])

    importances = []
    hit_df = df[df["ProfitHit"] == 1]
    loss_df = df[df["ProfitHit"] == 0]
    for p in params:
        vals = perm_importances[p]
        mean_imp = float(np.mean(vals)) if vals else 0.0
        std_imp = float(np.std(vals)) if vals else 0.0
        hit_med = float(hit_df[p].median()) if len(hit_df) else float("nan")
        loss_med = float(loss_df[p].median()) if len(loss_df) else float("nan")
        if abs(mean_imp) < 1e-4:
            direction = "unclear (importance ~0)"
        elif hit_med > loss_med:
            direction = "higher = more likely to win"
        elif hit_med < loss_med:
            direction = "lower = more likely to win"
        else:
            direction = "unclear"
        importances.append(ParamImportance(
            param=p, permutation_importance_mean=mean_imp, permutation_importance_std=std_imp,
            direction_of_effect=direction, hit_median=hit_med, loss_median=loss_med,
        ))
    importances.sort(key=lambda r: r.permutation_importance_mean, reverse=True)

    warning = None
    if not has_signal:
        if not np.isnan(cv_auc) and cv_auc < 0.45:
            warning = (f"Cross-validated AUC ({cv_auc:.3f}) is BELOW 0.5. This is not an "
                       f"inverse signal to trust -- with only {n} trades split across "
                       f"{cv_folds} folds, an AUC below 0.5 is what sampling noise looks "
                       f"like, not evidence the parameters predict the opposite outcome. "
                       f"Treat the importance ranking below as noise, not a guide for "
                       f"parameter changes.")
        else:
            warning = (f"Cross-validated AUC ({cv_auc:.3f}) is close to 0.5 -- the model found "
                       f"little to no real predictive signal in these parameters for this "
                       f"direction. Treat the importance ranking below as noise, not a guide "
                       f"for parameter changes.")

    return DirectionModelResult(
        direction_label=label, n=n, baseline_hit_rate=baseline_hit, cv_auc=cv_auc,
        cv_auc_std=cv_auc_std, has_signal=has_signal, model_used=model_kind,
        importances=importances, warning=warning,
    )


def shap_summary(df: pd.DataFrame, params: list, model_kind: str, seed: int) -> Optional[dict]:
    """Fits one model on all data (for interpretability only, not evaluation)
    and returns mean absolute SHAP value per parameter -- a second, independent
    view of importance that also captures interaction effects the permutation
    importance above can miss."""
    if not SHAP_AVAILABLE:
        return None
    X_raw = df[params].to_numpy(dtype=float)
    y = df["ProfitHit"].to_numpy(dtype=int)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    model = build_model(model_kind, seed)
    model.fit(X, y)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (win) for binary classifiers that return a list
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return dict(zip(params, [float(v) for v in mean_abs_shap]))
    except Exception as e:
        warnings.warn(f"SHAP computation failed ({e}); continuing without it.")
        return None


def print_result(r: DirectionModelResult, shap_vals: Optional[dict]):
    print(f"\n{'='*72}")
    print(f" {r.direction_label.upper()}  (n={r.n}, baseline hit rate={r.baseline_hit_rate:.1%})")
    print(f"{'='*72}")

    if r.warning and not r.importances:
        print(f"  SKIPPED: {r.warning}")
        return

    print(f"Model: {r.model_used}   Cross-validated AUC: {r.cv_auc:.3f} (+/- {r.cv_auc_std:.3f})")
    print(f"  (0.50 = no better than guessing, 1.00 = perfect separation)")

    if not r.has_signal:
        print(f"\n  *** {r.warning} ***")

    print(f"\nParameter importance ranking (cross-validated permutation importance,\n"
          f"drop in ROC-AUC when the column is shuffled -- higher = more important):")
    print(f"{'Parameter':<26}{'Importance':>12}{'  Direction of effect'}")
    for imp in r.importances:
        print(f"{imp.param:<26}{imp.permutation_importance_mean:>12.4f}  {imp.direction_of_effect}")

    if shap_vals:
        print(f"\nSHAP mean |impact| (independent cross-check, also captures interactions):")
        ranked = sorted(shap_vals.items(), key=lambda kv: kv[1], reverse=True)
        for p, v in ranked:
            print(f"  {p:<26}{v:>10.4f}")
    elif not SHAP_AVAILABLE:
        print(f"\n(shap not installed -- run `pip install shap --break-system-packages` "
              f"for an independent interaction-aware cross-check.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Path to the merged trade CSV")
    ap.add_argument("--model", choices=["random_forest", "gradient_boosting"], default="random_forest",
                    help="Classifier used for importance ranking (default random_forest)")
    ap.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds (default 5)")
    ap.add_argument("--params", default=None,
                    help="Comma-separated list of parameter columns to rank "
                         "(default: auto-detect all numeric ind_* columns)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--no-shap", action="store_true", help="Skip SHAP computation even if installed")
    ap.add_argument("--output", default=None, help="Optional path to write the full JSON report")
    args = ap.parse_args()

    df = load_trades(args.csv_path)
    explicit_params = [p.strip() for p in args.params.split(",")] if args.params else None
    params = get_param_columns(df, explicit_params)

    long_df = df[df["ind_SignalSent"] == 1].reset_index(drop=True)
    short_df = df[df["ind_SignalSent"] == -1].reset_index(drop=True)

    print(f"Loaded {len(df)} trades from {args.csv_path}")
    print(f"  Long: {len(long_df)}   Short: {len(short_df)}")
    print(f"  Ranking {len(params)} parameters: {params}")
    print(f"  Model: {args.model}, cv_folds={args.cv_folds}")
    if not SHAP_AVAILABLE and not args.no_shap:
        print("  (shap not installed -- skipping SHAP cross-check. "
              "pip install shap --break-system-packages to enable it.)")

    long_result = analyze_direction(long_df, "long", params, args.model, args.cv_folds, args.seed, not args.no_shap)
    short_result = analyze_direction(short_df, "short", params, args.model, args.cv_folds, args.seed + 1, not args.no_shap)

    long_shap = None
    short_shap = None
    if not args.no_shap and SHAP_AVAILABLE:
        if long_result.importances:
            long_shap = shap_summary(long_df, params, args.model, args.seed)
        if short_result.importances:
            short_shap = shap_summary(short_df, params, args.model, args.seed + 1)

    print_result(long_result, long_shap)
    print_result(short_result, short_shap)

    print(f"\n{'='*72}")
    print(" HOW TO USE THIS")
    print(f"{'='*72}")
    print("- If cv_auc is near 0.5 for a direction, none of these parameters reliably")
    print("  predict outcome on their own -- don't chase the top-ranked one.")
    print("- Parameters ranked highest here are the best candidates to feed into")
    print("  ats_optuna_optimizer.py's --params list, instead of searching all of them.")
    print("- 'Direction of effect' tells you which way to set a threshold (>= vs <=),")
    print("  but NOT the right threshold value -- use the optimizer scripts for that.")

    if args.output:
        report = {
            "csv_path": args.csv_path, "model": args.model, "cv_folds": args.cv_folds,
            "long": {**asdict(long_result), "shap": long_shap},
            "short": {**asdict(short_result), "shap": short_shap},
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()
