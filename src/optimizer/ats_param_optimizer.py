#!/usr/bin/env python3
"""
ats_param_optimizer.py

Optimizes entry-filter parameters for AtsFastReversal / AtsSlowReversal
trade logs and recommends parameter changes.

Methodology (matches the strategy's documented analysis rules):
  - Outcome definition: ProfitHit = Profit/Loss > 0, PureLoss = Profit/Loss < 0
    (ALL exit types included -- StpLoss with P/L > 0 is a partial win, not a loss)
  - Always split by direction (ind_SignalSent: 1 = long, -1 = short) before
    drawing any conclusion -- longs and shorts respond to different parameters.
  - Primary significance test: Mann-Whitney U, ProfitHit group vs PureLoss group.
  - Optimize for Expectancy = mean(Profit/Loss) over the filtered trade set,
    not win rate alone (a filter can raise win rate and still lose money, or
    vice versa).
  - Overfitting guard: no filtered subset is reported unless it has at least
    `--min-n` trades (default 30).
  - Single-parameter sweep AND pairwise combination sweep are both run, since
    the strategy's own findings (e.g. FullDeltaATRs + FullAngle for longs)
    are combinations, not single gates.
  - Every recommendation is labeled with its sample size and significance so
    it can be told apart from a small-sample fluke. This script does NOT
    silently promote an unproven combo to a "recommended" parameter --
    that judgment call is left to the report, which flags confidence level.

All CSV column names and shared helpers (Column enum, TradeDataLoader,
ColumnSelector) live in optimizer_constants.py. All the core logic here is
consolidated into the `GridSearchOptimizer` class.

Usage:
    python ats_param_optimizer.py trades.csv
    python ats_param_optimizer.py trades.csv --min-n 30 --top-n 8
    python ats_param_optimizer.py trades.csv --output report.json
    python ats_param_optimizer.py trades.csv --compare-filter "ind_FullDeltaATRs>=9,ind_FullAngle>=26" --direction long

Requires: pandas, numpy, scipy
"""

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from optimizer_constants import (
    Column, MIN_N_DEFAULT, PROFIT_HIT_COL, NON_PARAM_COLUMNS,
    TradeDataLoader, ColumnSelector,
)

MAX_CANDIDATES = 25  # cap distinct thresholds tested per parameter (quantile-spaced)


@dataclass
class ThresholdResult:
    param: str
    direction: str          # "ge" (>=) or "le" (<=)
    threshold: float
    n: int
    hit_rate: float
    expectancy: float
    total_pl: float
    baseline_expectancy: float
    baseline_n: int
    improvement: float       # expectancy - baseline_expectancy


@dataclass
class ComboResult:
    filters: str             # human readable, e.g. "FullDeltaATRs>=9 AND FullAngle>=26"
    n: int
    hit_rate: float
    expectancy: float
    total_pl: float
    baseline_expectancy: float
    improvement: float


@dataclass
class SignificanceResult:
    param: str
    hit_median: float
    loss_median: float
    p_value: float
    n_hit: int
    n_loss: int
    verdict: str              # "hard gate" (p<0.01), "secondary" (p<0.10), "no signal"


# ============================================================================
# GridSearchOptimizer -- consolidates all the logic used to compute the
# significance tests, single/combo threshold sweeps, boolean flag tests, and
# forward-test (--compare-filter) evaluation.
# ============================================================================
class GridSearchOptimizer:

    def __init__(self, min_n: int = MIN_N_DEFAULT, top_n: int = 8,
                 max_candidates: int = MAX_CANDIDATES):
        self.min_n = min_n
        self.top_n = top_n
        self.max_candidates = max_candidates

    # ------------------------------------------------------------------
    # Data loading / column selection
    # ------------------------------------------------------------------
    @staticmethod
    def load_trades(csv_path: str) -> pd.DataFrame:
        return TradeDataLoader.load_trades(csv_path, sort_by_entry_date=False)

    @staticmethod
    def split_by_direction(df: pd.DataFrame) -> tuple:
        return TradeDataLoader.split_by_direction(df)

    @staticmethod
    def get_param_columns(df: pd.DataFrame) -> list:
        return ColumnSelector.get_param_columns(df, non_param_columns=NON_PARAM_COLUMNS,
                                                 exclude_boolean_flags=True, min_unique=3)

    @staticmethod
    def get_boolean_flag_columns(df: pd.DataFrame) -> list:
        return ColumnSelector.get_boolean_flag_columns(df)

    # ------------------------------------------------------------------
    # Core stats
    # ------------------------------------------------------------------
    @staticmethod
    def expectancy_stats(d: pd.DataFrame) -> tuple:
        """Returns (n, hit_rate, expectancy, total_pl) for a trade subset."""
        n = len(d)
        if n == 0:
            return 0, float("nan"), float("nan"), 0.0
        return (n, float(d[PROFIT_HIT_COL].mean()), float(d[Column.PROFIT_LOSS.value].mean()),
                float(d[Column.PROFIT_LOSS.value].sum()))

    def candidate_thresholds(self, values: np.ndarray, max_candidates: Optional[int] = None) -> np.ndarray:
        """Distinct, quantile-spaced candidate thresholds -- keeps sweeps fast on
        columns with many unique values without materially changing the result
        (the expectancy-vs-threshold curve is smooth enough that sampling ~25
        points along it finds the same optimum as testing every unique value)."""
        max_candidates = max_candidates or self.max_candidates
        uniq = np.unique(values)
        if len(uniq) <= max_candidates:
            return uniq
        qs = np.linspace(0, 1, max_candidates)
        return np.unique(np.quantile(uniq, qs))

    # ------------------------------------------------------------------
    # Single-parameter sweep, significance test, boolean flag test
    # ------------------------------------------------------------------
    def single_param_sweep(self, d: pd.DataFrame, param: str) -> Optional[ThresholdResult]:
        """Finds the threshold/direction for `param` that maximizes expectancy,
        subject to the minimum sample size guard. Vectorized with numpy for speed."""
        baseline_n, _, baseline_exp, _ = self.expectancy_stats(d)
        if baseline_n == 0 or np.isnan(baseline_exp):
            return None

        vals = d[param].to_numpy(dtype=float)
        pl = d[Column.PROFIT_LOSS.value].to_numpy(dtype=float)
        hit = d[PROFIT_HIT_COL].to_numpy(dtype=bool)
        mask_valid = ~np.isnan(vals)
        vals, pl, hit = vals[mask_valid], pl[mask_valid], hit[mask_valid]
        if len(vals) < 3:
            return None

        cands = self.candidate_thresholds(vals)
        if len(cands) < 1:
            return None

        best = None
        for direction in ("ge", "le"):
            for thresh in cands:
                m = vals >= thresh if direction == "ge" else vals <= thresh
                n = int(m.sum())
                if n < self.min_n:
                    continue
                exp = float(pl[m].mean())
                if best is None or exp > best.expectancy:
                    hit_rate = float(hit[m].mean())
                    total_pl = float(pl[m].sum())
                    best = ThresholdResult(
                        param=param, direction=direction, threshold=float(thresh),
                        n=n, hit_rate=hit_rate, expectancy=exp, total_pl=total_pl,
                        baseline_expectancy=baseline_exp, baseline_n=baseline_n,
                        improvement=exp - baseline_exp,
                    )
        return best

    @staticmethod
    def significance_test(d: pd.DataFrame, param: str) -> Optional[SignificanceResult]:
        hit = d.loc[d[PROFIT_HIT_COL], param].dropna()
        loss = d.loc[~d[PROFIT_HIT_COL], param].dropna()
        if len(hit) < 5 or len(loss) < 5:
            return None
        try:
            _, p = mannwhitneyu(hit, loss, alternative="two-sided")
        except ValueError:
            return None
        if p < 0.01:
            verdict = "hard gate candidate (p<0.01)"
        elif p < 0.05:
            verdict = "secondary signal (p<0.05)"
        elif p < 0.10:
            verdict = "weak signal (p<0.10)"
        else:
            verdict = "no signal"
        return SignificanceResult(
            param=param, hit_median=float(hit.median()), loss_median=float(loss.median()),
            p_value=float(p), n_hit=len(hit), n_loss=len(loss), verdict=verdict,
        )

    def boolean_flag_test(self, d: pd.DataFrame, flag: str) -> Optional[ThresholdResult]:
        baseline_n, _, baseline_exp, _ = self.expectancy_stats(d)
        if baseline_n == 0:
            return None
        sub = d[d[flag] == 1]
        n = len(sub)
        if n < self.min_n:
            return None
        _, hit_rate, exp, total_pl = self.expectancy_stats(sub)
        return ThresholdResult(
            param=flag, direction="==1", threshold=1.0, n=n, hit_rate=hit_rate,
            expectancy=exp, total_pl=total_pl, baseline_expectancy=baseline_exp,
            baseline_n=baseline_n, improvement=exp - baseline_exp,
        )

    # ------------------------------------------------------------------
    # 2-parameter combo sweep
    # ------------------------------------------------------------------
    def combo_sweep(self, d: pd.DataFrame, param_a: str, param_b: str,
                     directions_a=("ge", "le"), directions_b=("ge", "le")) -> Optional[ComboResult]:
        """Two-parameter joint threshold sweep, maximizing expectancy.
        Vectorized with numpy and capped candidate thresholds so the O(V_a * V_b)
        grid stays fast even when called across many parameter pairs."""
        baseline_n, _, baseline_exp, _ = self.expectancy_stats(d)
        if baseline_n == 0:
            return None

        a_full = d[param_a].to_numpy(dtype=float)
        b_full = d[param_b].to_numpy(dtype=float)
        pl_full = d[Column.PROFIT_LOSS.value].to_numpy(dtype=float)
        hit_full = d[PROFIT_HIT_COL].to_numpy(dtype=bool)
        valid = ~np.isnan(a_full) & ~np.isnan(b_full)
        a, b, pl, hit = a_full[valid], b_full[valid], pl_full[valid], hit_full[valid]
        if len(a) < self.min_n:
            return None

        cands_a = self.candidate_thresholds(a, max_candidates=12)
        cands_b = self.candidate_thresholds(b, max_candidates=12)
        if len(cands_a) < 1 or len(cands_b) < 1:
            return None

        best = None
        for da in directions_a:
            mask_a_all = (a[:, None] >= cands_a[None, :]) if da == "ge" else (a[:, None] <= cands_a[None, :])
            for ia, ta in enumerate(cands_a):
                mask_a = mask_a_all[:, ia]
                n_a = int(mask_a.sum())
                if n_a < self.min_n:
                    continue
                b_sub = b[mask_a]
                pl_sub = pl[mask_a]
                hit_sub = hit[mask_a]
                for db in directions_b:
                    for tb in cands_b:
                        m = (b_sub >= tb) if db == "ge" else (b_sub <= tb)
                        n = int(m.sum())
                        if n < self.min_n:
                            continue
                        exp = float(pl_sub[m].mean())
                        if best is None or exp > best[0]:
                            sym_a = ">=" if da == "ge" else "<="
                            sym_b = ">=" if db == "ge" else "<="
                            label = f"{param_a}{sym_a}{ta:g} AND {param_b}{sym_b}{tb:g}"
                            best = (exp, ComboResult(
                                filters=label, n=n, hit_rate=float(hit_sub[m].mean()),
                                expectancy=exp, total_pl=float(pl_sub[m].sum()),
                                baseline_expectancy=baseline_exp, improvement=exp - baseline_exp,
                            ))
        return best[1] if best else None

    # ------------------------------------------------------------------
    # Per-direction analysis
    # ------------------------------------------------------------------
    def analyze_direction(self, d: pd.DataFrame, label: str, params: list, flags: list) -> dict:
        baseline_n, baseline_hit, baseline_exp, baseline_pl = self.expectancy_stats(d)
        result = {
            "direction": label,
            "baseline": {
                "n": baseline_n, "hit_rate": baseline_hit,
                "expectancy": baseline_exp, "total_pl": baseline_pl,
            },
            "significance": [],
            "single_param_best": [],
            "boolean_flags": [],
            "top_combos": [],
        }

        # Significance ranking (matches strategy's own Mann-Whitney methodology)
        sig_results = []
        for p in params:
            r = self.significance_test(d, p)
            if r:
                sig_results.append(r)
        sig_results.sort(key=lambda r: r.p_value)
        result["significance"] = [asdict(r) for r in sig_results]

        # Single-parameter expectancy-maximizing thresholds
        single_results = []
        for p in params:
            r = self.single_param_sweep(d, p)
            if r and r.improvement > 0:
                single_results.append(r)
        single_results.sort(key=lambda r: r.improvement, reverse=True)
        result["single_param_best"] = [asdict(r) for r in single_results]

        # Boolean flag tests (C1-C16 style conditions)
        flag_results = []
        for f in flags:
            r = self.boolean_flag_test(d, f)
            if r:
                flag_results.append(r)
        flag_results.sort(key=lambda r: r.improvement, reverse=True)
        result["boolean_flags"] = [asdict(r) for r in flag_results]

        # Pairwise combos among the top single params (avoids O(P^2) blowup on
        # every column -- combining the strongest individual signals is where
        # real edges tend to show up, per the strategy's own findings).
        top_params = [r.param for r in single_results[:self.top_n]] or [r.param for r in sig_results[:self.top_n]]
        combos = []
        for a, b in itertools.combinations(top_params, 2):
            c = self.combo_sweep(d, a, b)
            if c and c.improvement > 0:
                combos.append(c)
        combos.sort(key=lambda c: c.improvement, reverse=True)
        result["top_combos"] = [asdict(c) for c in combos[:self.top_n]]

        return result

    # ------------------------------------------------------------------
    # --compare-filter forward-test evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def parse_filter_string(filter_str: str) -> list:
        """Parses 'ind_FullDeltaATRs>=9,ind_FullAngle>=26' into [(col, op, val), ...]."""
        clauses = []
        for part in filter_str.split(","):
            part = part.strip()
            for op in (">=", "<=", "==", ">", "<"):
                if op in part:
                    col, val = part.split(op)
                    clauses.append((col.strip(), op, float(val.strip())))
                    break
            else:
                raise ValueError(f"Could not parse filter clause: {part!r}")
        return clauses

    @staticmethod
    def apply_filter(d: pd.DataFrame, clauses: list) -> pd.DataFrame:
        sub = d
        ops = {
            ">=": lambda s, c, v: s[s[c] >= v],
            "<=": lambda s, c, v: s[s[c] <= v],
            "==": lambda s, c, v: s[s[c] == v],
            ">": lambda s, c, v: s[s[c] > v],
            "<": lambda s, c, v: s[s[c] < v],
        }
        for col, op, val in clauses:
            sub = ops[op](sub, col, val)
        return sub

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    def print_direction_report(self, rep: dict) -> None:
        b = rep["baseline"]
        print(f"\n{'='*70}")
        print(f" {rep['direction'].upper()}  (n={b['n']}, hit_rate={b['hit_rate']:.1%}, "
              f"expectancy=${b['expectancy']:.2f}/trade, total P/L=${b['total_pl']:.2f})")
        print(f"{'='*70}")

        print(f"\n-- Statistical significance (Mann-Whitney U, ProfitHit vs PureLoss) --")
        if not rep["significance"]:
            print("  (not enough data to test)")
        for r in rep["significance"][:10]:
            print(f"  {r['param']:<24} hit_med={r['hit_median']:<8.2f} "
                  f"loss_med={r['loss_median']:<8.2f} p={r['p_value']:.3f}  [{r['verdict']}]")

        print(f"\n-- Best single-parameter threshold (min n={self.min_n}, ranked by expectancy gain) --")
        if not rep["single_param_best"]:
            print("  (no threshold beat baseline with required sample size)")
        for r in rep["single_param_best"][:10]:
            sym = ">=" if r["direction"] == "ge" else ("<=" if r["direction"] == "le" else r["direction"])
            print(f"  {r['param']:<24} {sym} {r['threshold']:<8.3g} "
                  f"n={r['n']:<4} hit={r['hit_rate']:.1%}  "
                  f"expectancy=${r['expectancy']:.2f} (+${r['improvement']:.2f} vs baseline)")

        print(f"\n-- Boolean condition flags (require flag == 1) --")
        if not rep["boolean_flags"]:
            print("  (none tested or none improved on baseline)")
        for r in rep["boolean_flags"][:10]:
            print(f"  {r['param']:<24} n={r['n']:<4} hit={r['hit_rate']:.1%}  "
                  f"expectancy=${r['expectancy']:.2f} (+${r['improvement']:.2f} vs baseline)")

        print(f"\n-- Best 2-parameter combinations (min n={self.min_n}) --")
        if not rep["top_combos"]:
            print("  (no combination beat baseline with required sample size)")
        for c in rep["top_combos"]:
            print(f"  {c['filters']}")
            print(f"      n={c['n']:<4} hit={c['hit_rate']:.1%}  "
                  f"expectancy=${c['expectancy']:.2f} (+${c['improvement']:.2f} vs baseline, "
                  f"total P/L=${c['total_pl']:.2f})")

    def print_recommendations(self, long_rep: dict, short_rep: dict) -> None:
        print(f"\n{'='*70}")
        print(" RECOMMENDATIONS")
        print(f"{'='*70}")
        print(f"(Overfitting guard: every result above required n>={self.min_n}. Results with")
        print(f" p>=0.10 significance are directional only -- confirm with more trades")
        print(f" or an out-of-sample forward test before wiring into the live strategy.)\n")

        for label, rep in (("LONG", long_rep), ("SHORT", short_rep)):
            best_combo = rep["top_combos"][0] if rep["top_combos"] else None
            best_single = rep["single_param_best"][0] if rep["single_param_best"] else None
            sig_params = {r["param"] for r in rep["significance"] if r["p_value"] < 0.10}

            print(f"[{label}] baseline expectancy: ${rep['baseline']['expectancy']:.2f}/trade "
                  f"over {rep['baseline']['n']} trades")

            if best_combo and best_combo["improvement"] > 0:
                confidence = "HIGH" if any(p in best_combo["filters"] for p in sig_params) else "LOW (small-sample, unconfirmed)"
                print(f"  -> Best combo: {best_combo['filters']}")
                print(f"     n={best_combo['n']}, expectancy=${best_combo['expectancy']:.2f} "
                      f"(+${best_combo['improvement']:.2f}/trade)  confidence: {confidence}")
            elif best_single and best_single["improvement"] > 0:
                param = best_single["param"]
                confidence = "HIGH" if param in sig_params else "LOW (small-sample, unconfirmed)"
                sym = ">=" if best_single["direction"] == "ge" else "<="
                print(f"  -> Best single filter: {param} {sym} {best_single['threshold']:g}")
                print(f"     n={best_single['n']}, expectancy=${best_single['expectancy']:.2f} "
                      f"(+${best_single['improvement']:.2f}/trade)  confidence: {confidence}")
            else:
                print("  -> No filter improved on baseline with sufficient sample size. "
                      "Leave current parameters as-is.")
            print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Path to the merged trade CSV")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT,
                    help=f"Minimum trades required in a filtered subset (default {MIN_N_DEFAULT})")
    ap.add_argument("--top-n", type=int, default=8,
                    help="How many top single-parameters to carry into combo search / report (default 8)")
    ap.add_argument("--output", default=None,
                    help="Optional path to write the full JSON report")
    ap.add_argument("--compare-filter", default=None,
                    help="Evaluate a specific filter string instead of/alongside the sweep, "
                         "e.g. \"ind_FullDeltaATRs>=9,ind_FullAngle>=26\"")
    ap.add_argument("--direction", choices=["long", "short", "both"], default="both",
                    help="Restrict --compare-filter evaluation to one direction (default both)")
    args = ap.parse_args()

    opt = GridSearchOptimizer(min_n=args.min_n, top_n=args.top_n)

    df = opt.load_trades(args.csv_path)
    params = opt.get_param_columns(df)
    flags = opt.get_boolean_flag_columns(df)

    long_df, short_df = opt.split_by_direction(df)

    print(f"Loaded {len(df)} trades from {args.csv_path}")
    print(f"  Long: {len(long_df)}   Short: {len(short_df)}")
    print(f"  Candidate continuous parameters: {len(params)}")
    print(f"  Candidate boolean flags: {len(flags)}")

    long_rep = opt.analyze_direction(long_df, "long", params, flags)
    short_rep = opt.analyze_direction(short_df, "short", params, flags)

    opt.print_direction_report(long_rep)
    opt.print_direction_report(short_rep)
    opt.print_recommendations(long_rep, short_rep)

    if args.compare_filter:
        clauses = opt.parse_filter_string(args.compare_filter)
        print(f"\n{'='*70}")
        print(f" CUSTOM FILTER CHECK: {args.compare_filter}")
        print(f"{'='*70}")
        targets = []
        if args.direction in ("long", "both"):
            targets.append(("LONG", long_df))
        if args.direction in ("short", "both"):
            targets.append(("SHORT", short_df))
        for label, d in targets:
            baseline_n, baseline_hit, baseline_exp, baseline_pl = opt.expectancy_stats(d)
            sub = opt.apply_filter(d, clauses)
            n, hit, exp, pl = opt.expectancy_stats(sub)
            flag = "OK" if n >= args.min_n else f"WARNING: n < {args.min_n}, low confidence"
            print(f"[{label}] baseline: n={baseline_n} hit={baseline_hit:.1%} exp=${baseline_exp:.2f}")
            print(f"[{label}] filtered: n={n} hit={hit:.1%} exp=${exp:.2f} "
                  f"total_pl=${pl:.2f}  ({flag})")

    if args.output:
        full_report = {
            "csv_path": args.csv_path,
            "min_n": args.min_n,
            "long": long_rep,
            "short": short_rep,
        }
        with open(args.output, "w") as f:
            json.dump(full_report, f, indent=2, default=float)
        print(f"\nFull JSON report written to {args.output}")


if __name__ == "__main__":
    main()
