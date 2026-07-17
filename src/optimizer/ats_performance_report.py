"""
ATS Performance Report
======================
Generates a TradeStation-style performance report from the AtsFastReversal
(or AtsSlowReversal) merged CSV trade file.

Usage:
    python ats_performance_report.py <csv_file> [--strategy NAME] [--excel FILE] [--output FILE]

Output:
    Console report + optional Excel file with three sheets:
      - Trade Summary: Key metrics and statistics
      - Daily Performance: Daily P/L and win rates
      - Performance By Symbol: Per-symbol statistics
    OR optional CSV summary (legacy)
"""

import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ATS Performance Report")
    p.add_argument("csv_file", help="Path to merged trade CSV")
    p.add_argument("--strategy", default=None, help="Strategy name override")
    p.add_argument("--output", default=None, help="Save summary to CSV file (deprecated, use --excel)")
    p.add_argument("--excel", default=None, help="Save report to Excel file with multiple sheets")
    p.add_argument("--direction", choices=["long","short","both"], default="both",
                   help="Filter by trade direction")
    p.add_argument("--symbol", default=None, help="Filter by symbol")
    p.add_argument("--start", default=None, help="Start date mm/dd/yyyy")
    p.add_argument("--end",   default=None, help="End date   mm/dd/yyyy")
    return p.parse_args()


def load_trades(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Parse dates / times
    df["EntryDT"] = pd.to_datetime(
        df["EntryDate"].astype(str) + " " +
        df["EntryTime"].astype(str).str.zfill(4).str[:2] + ":" +
        df["EntryTime"].astype(str).str.zfill(4).str[2:],
        format="%m/%d/%Y %H:%M", errors="coerce"
    )
    df["ExitDT"] = pd.to_datetime(
        df["ExitDate"].astype(str) + " " +
        df["ExitTime"].astype(str).str.zfill(4).str[:2] + ":" +
        df["ExitTime"].astype(str).str.zfill(4).str[2:],
        format="%m/%d/%Y %H:%M", errors="coerce"
    )

    df["PL"]       = pd.to_numeric(df["Profit/Loss"], errors="coerce").fillna(0)
    df["Shares"]   = pd.to_numeric(df["Shares"],      errors="coerce").fillna(0)
    df["IsLong"]   = df["ind_SignalSent"].eq(1) if "ind_SignalSent" in df.columns \
                     else df["EntryName"].str.startswith("LE")
    df["Win"]      = df["PL"] > 0
    df["ProfitHit"]= df["PL"] > 0   # at least one target hit (see context doc)

    return df


def filter_trades(df, args):
    if args.direction == "long":
        df = df[df["IsLong"]]
    elif args.direction == "short":
        df = df[~df["IsLong"]]
    if args.symbol:
        df = df[df["Symbol"].str.upper() == args.symbol.upper()]
    if args.start:
        df = df[df["EntryDT"] >= pd.to_datetime(args.start, format="%m/%d/%Y")]
    if args.end:
        df = df[df["EntryDT"] <= pd.to_datetime(args.end, format="%m/%d/%Y")]
    return df.copy()


# ── core metrics ─────────────────────────────────────────────────────────────

def calc_equity_curve(pl_series):
    """Cumulative P/L series."""
    return pl_series.cumsum()


def calc_drawdown(equity):
    """Returns drawdown series, max drawdown value, and max drawdown duration (bars)."""
    roll_max = equity.cummax()
    dd       = equity - roll_max                 # always <= 0
    max_dd   = dd.min()

    # Duration: longest consecutive period underwater
    underwater = (dd < 0).astype(int)
    max_dur, cur_dur = 0, 0
    for v in underwater:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return dd, max_dd, max_dur


def calc_sharpe(pl_series, periods_per_year=252):
    """Daily Sharpe ratio (assumes each row = 1 trading day unit; adjust as needed)."""
    if pl_series.std() == 0:
        return np.nan
    return (pl_series.mean() / pl_series.std()) * np.sqrt(periods_per_year)


def calc_sortino(pl_series, periods_per_year=252):
    """Sortino ratio using downside deviation."""
    neg = pl_series[pl_series < 0]
    if len(neg) == 0 or neg.std() == 0:
        return np.nan
    return (pl_series.mean() / neg.std()) * np.sqrt(periods_per_year)


def calc_profit_factor(pl_series):
    gross_win  = pl_series[pl_series > 0].sum()
    gross_loss = abs(pl_series[pl_series < 0].sum())
    return gross_win / gross_loss if gross_loss > 0 else np.inf


def calc_expectancy(pl_series):
    wins   = pl_series[pl_series > 0]
    losses = pl_series[pl_series < 0]
    wr     = len(wins) / len(pl_series) if len(pl_series) > 0 else 0
    avg_w  = wins.mean()   if len(wins)   > 0 else 0
    avg_l  = losses.mean() if len(losses) > 0 else 0
    return wr * avg_w + (1 - wr) * avg_l


def calc_max_capital(df):
    """
    Approximate max capital required:
    max(EntryPrice * Shares) across all trades — the largest single position cost.
    For a more accurate margin calc you'd need the broker margin rate.
    """
    if "EntryPrice" in df.columns:
        df2 = df.copy()
        df2["PositionValue"] = pd.to_numeric(df2["EntryPrice"], errors="coerce") * df2["Shares"]
        return df2["PositionValue"].max()
    return np.nan


def calc_consecutive(win_series):
    """Max consecutive wins and losses."""
    max_w = max_l = cur_w = cur_l = 0
    for w in win_series:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def exit_breakdown(df):
    return df.groupby("ExitName").agg(
        Count   =("PL", "size"),
        TotalPL =("PL", "sum"),
        AvgPL   =("PL", "mean"),
        WinRate =("Win", "mean"),
    ).sort_values("Count", ascending=False)


def daily_stats(df):
    df2 = df.copy()
    df2["Date"] = df2["EntryDT"].dt.date
    return df2.groupby("Date").agg(
        Trades  =("PL", "size"),
        TotalPL =("PL", "sum"),
        WinRate =("Win", "mean"),
    )


def symbol_stats(df):
    return df.groupby("Symbol").agg(
        Trades  =("PL", "size"),
        TotalPL =("PL", "sum"),
        AvgPL   =("PL", "mean"),
        WinRate =("Win", "mean"),
    ).sort_values("TotalPL", ascending=False)


# ── report printer ────────────────────────────────────────────────────────────

def hr(char="─", width=64):
    print(char * width)


def fmt(val, prefix="$", decimals=2):
    if pd.isna(val):
        return "N/A"
    if val == np.inf:
        return "∞"
    return f"{prefix}{val:,.{decimals}f}" if prefix else f"{val:.{decimals}f}"


def pct(val):
    return "N/A" if pd.isna(val) else f"{val*100:.1f}%"


def print_report(df, strategy_name):
    pl = df["PL"]
    wins   = df[df["Win"]]
    losses = df[~df["Win"]]
    longs  = df[df["IsLong"]]
    shorts = df[~df["IsLong"]]

    equity       = calc_equity_curve(pl.reset_index(drop=True))
    dd, max_dd, max_dd_dur = calc_drawdown(equity)
    max_cap      = calc_max_capital(df)
    max_w, max_l = calc_consecutive(df["Win"].tolist())

    # Daily P/L for Sharpe (group by date)
    df2 = df.copy()
    df2["Date"] = df2["EntryDT"].dt.date
    daily_pl = df2.groupby("Date")["PL"].sum()

    print()
    hr("═")
    print(f"  PERFORMANCE REPORT — {strategy_name}")
    hr("═")

    # ── Summary ──
    print("\n  TRADE SUMMARY")
    hr()
    print(f"  {'Total Net Profit':<35} {fmt(pl.sum())}")
    print(f"  {'Gross Profit':<35} {fmt(pl[pl>0].sum())}")
    print(f"  {'Gross Loss':<35} {fmt(pl[pl<0].sum())}")
    print(f"  {'Profit Factor':<35} {fmt(calc_profit_factor(pl), prefix='', decimals=3)}")
    print(f"  {'Expectancy (per trade)':<35} {fmt(calc_expectancy(pl))}")
    print()
    print(f"  {'Total Trades':<35} {len(df)}")
    print(f"  {'  Winning Trades':<35} {len(wins)}  ({pct(len(wins)/len(df) if df.size else 0)})")
    print(f"  {'  Losing Trades':<35} {len(losses)}  ({pct(len(losses)/len(df) if df.size else 0)})")
    print(f"  {'  Scratch (P/L = 0)':<35} {len(df[pl==0])}")
    print()
    print(f"  {'Long Trades':<35} {len(longs)}  ({pct(len(longs)/len(df) if df.size else 0)})")
    print(f"  {'  Long Win Rate':<35} {pct(longs['Win'].mean())}")
    print(f"  {'  Long Net P/L':<35} {fmt(longs['PL'].sum())}")
    print(f"  {'Short Trades':<35} {len(shorts)}  ({pct(len(shorts)/len(df) if df.size else 0)})")
    print(f"  {'  Short Win Rate':<35} {pct(shorts['Win'].mean())}")
    print(f"  {'  Short Net P/L':<35} {fmt(shorts['PL'].sum())}")

    # ── Win/Loss Detail ──
    print("\n  WIN / LOSS DETAIL")
    hr()
    print(f"  {'Avg Winning Trade':<35} {fmt(wins['PL'].mean())}")
    print(f"  {'Avg Losing Trade':<35} {fmt(losses['PL'].mean())}")
    print(f"  {'Largest Win':<35} {fmt(wins['PL'].max())}")
    print(f"  {'Largest Loss':<35} {fmt(losses['PL'].min())}")
    print(f"  {'Avg Win / Avg Loss Ratio':<35} {fmt(abs(wins['PL'].mean()/losses['PL'].mean()) if losses['PL'].mean()!=0 else np.nan, prefix='', decimals=3)}")
    print(f"  {'Max Consecutive Winners':<35} {max_w}")
    print(f"  {'Max Consecutive Losers':<35} {max_l}")
    print(f"  {'Avg Shares per Trade':<35} {fmt(df['Shares'].mean(), prefix='', decimals=1)}")

    # ── Drawdown & Capital ──
    print("\n  DRAWDOWN & CAPITAL")
    hr()
    print(f"  {'Max Drawdown (P/L units)':<35} {fmt(max_dd)}")
    print(f"  {'Max Drawdown Duration (trades)':<35} {max_dd_dur}")
    print(f"  {'Max Capital Required (est.)':<35} {fmt(max_cap)}")
    print(f"  {'Return on Max Capital':<35} {pct(pl.sum()/max_cap if max_cap and max_cap>0 else np.nan)}")

    # ── Risk Ratios ──
    print("\n  RISK-ADJUSTED METRICS  (annualised, 252 trading days)")
    hr()
    print(f"  {'Sharpe Ratio':<35} {fmt(calc_sharpe(daily_pl), prefix='', decimals=3)}")
    print(f"  {'Sortino Ratio':<35} {fmt(calc_sortino(daily_pl), prefix='', decimals=3)}")
    print(f"  {'Calmar Ratio (Net/MaxDD)':<35} {fmt(pl.sum()/abs(max_dd) if max_dd!=0 else np.nan, prefix='', decimals=3)}")

    # ── Exit Breakdown ──
    print("\n  EXIT TYPE BREAKDOWN")
    hr()
    eb = exit_breakdown(df)
    print(f"  {'Exit':<20} {'Count':>6} {'TotalPL':>10} {'AvgPL':>8} {'WinRate':>8}")
    hr()
    for name, row in eb.iterrows():
        print(f"  {name:<20} {int(row.Count):>6} {fmt(row.TotalPL):>10} {fmt(row.AvgPL):>8} {pct(row.WinRate):>8}")

    # ── Daily ──
    print("\n  DAILY PERFORMANCE")
    hr()
    ds = daily_stats(df)
    print(f"  {'Date':<14} {'Trades':>6} {'P/L':>10} {'WinRate':>8}")
    hr()
    for date, row in ds.iterrows():
        print(f"  {str(date):<14} {int(row.Trades):>6} {fmt(row.TotalPL):>10} {pct(row.WinRate):>8}")
    print(f"\n  {'Best Day':<35} {fmt(ds['TotalPL'].max())}")
    print(f"  {'Worst Day':<35} {fmt(ds['TotalPL'].min())}")
    print(f"  {'Avg Daily P/L':<35} {fmt(ds['TotalPL'].mean())}")
    print(f"  {'Profitable Days':<35} {(ds['TotalPL']>0).sum()} / {len(ds)}")

    # ── By Symbol ──
    if df["Symbol"].nunique() > 1:
        print("\n  PERFORMANCE BY SYMBOL")
        hr()
        ss = symbol_stats(df)
        print(f"  {'Symbol':<10} {'Trades':>6} {'TotalPL':>10} {'AvgPL':>8} {'WinRate':>8}")
        hr()
        for sym, row in ss.iterrows():
            print(f"  {sym:<10} {int(row.Trades):>6} {fmt(row.TotalPL):>10} {fmt(row.AvgPL):>8} {pct(row.WinRate):>8}")

    hr("═")
    print()

    return {
        "TotalTrades": len(df),
        "WinRate": df["Win"].mean(),
        "NetPL": pl.sum(),
        "GrossProfit": pl[pl>0].sum(),
        "GrossLoss": pl[pl<0].sum(),
        "ProfitFactor": calc_profit_factor(pl),
        "Expectancy": calc_expectancy(pl),
        "AvgWin": wins["PL"].mean(),
        "AvgLoss": losses["PL"].mean(),
        "LargestWin": wins["PL"].max(),
        "LargestLoss": losses["PL"].min(),
        "MaxDrawdown": max_dd,
        "MaxDrawdownDuration": max_dd_dur,
        "MaxCapitalRequired": max_cap,
        "Sharpe": calc_sharpe(daily_pl),
        "Sortino": calc_sortino(daily_pl),
        "MaxConsecWins": max_w,
        "MaxConsecLosses": max_l,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def create_excel_report(df, strategy_name, output_path):
    """Create an Excel workbook with three sheets: Trade Summary, Daily Performance, Performance By Symbol."""
    
    pl = df["PL"]
    wins   = df[df["Win"]]
    losses = df[~df["Win"]]
    longs  = df[df["IsLong"]]
    shorts = df[~df["IsLong"]]

    equity       = calc_equity_curve(pl.reset_index(drop=True))
    dd, max_dd, max_dd_dur = calc_drawdown(equity)
    max_cap      = calc_max_capital(df)
    max_w, max_l = calc_consecutive(df["Win"].tolist())

    df2 = df.copy()
    df2["Date"] = df2["EntryDT"].dt.date
    daily_pl = df2.groupby("Date")["PL"].sum()

    # Create workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Trade Summary"

    # ── Sheet 1: Trade Summary ──
    row = 1
    
    # Helper to style section headers
    def add_section(title, start_row):
        ws[f"A{start_row}"] = title
        ws[f"A{start_row}"].font = Font(bold=True, size=12)
        ws[f"A{start_row}"].fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
        return start_row + 1

    # Summary section
    row = add_section("TRADE SUMMARY", row)
    summary_data = [
        ("Total Net Profit", fmt(pl.sum())),
        ("Gross Profit", fmt(pl[pl>0].sum())),
        ("Gross Loss", fmt(pl[pl<0].sum())),
        ("Profit Factor", fmt(calc_profit_factor(pl), prefix='', decimals=3)),
        ("Expectancy (per trade)", fmt(calc_expectancy(pl))),
    ]
    for label, value in summary_data:
        ws[f"A{row}"] = label
        ws[f"B{row}"] = value
        row += 1
    row += 1

    # Trade count section
    row = add_section("TRADE COUNT", row)
    trade_data = [
        ("Total Trades", len(df)),
        ("Winning Trades", f"{len(wins)} ({pct(len(wins)/len(df) if df.size else 0)})"),
        ("Losing Trades", f"{len(losses)} ({pct(len(losses)/len(df) if df.size else 0)})"),
        ("Scratch (P/L = 0)", len(df[pl==0])),
        ("Long Trades", f"{len(longs)} ({pct(len(longs)/len(df) if df.size else 0)})"),
        ("Long Win Rate", pct(longs['Win'].mean())),
        ("Long Net P/L", fmt(longs['PL'].sum())),
        ("Short Trades", f"{len(shorts)} ({pct(len(shorts)/len(df) if df.size else 0)})"),
        ("Short Win Rate", pct(shorts['Win'].mean())),
        ("Short Net P/L", fmt(shorts['PL'].sum())),
    ]
    for label, value in trade_data:
        ws[f"A{row}"] = label
        ws[f"B{row}"] = value
        row += 1
    row += 1

    # Win/Loss Detail section
    row = add_section("WIN / LOSS DETAIL", row)
    winloss_data = [
        ("Avg Winning Trade", fmt(wins['PL'].mean())),
        ("Avg Losing Trade", fmt(losses['PL'].mean())),
        ("Largest Win", fmt(wins['PL'].max())),
        ("Largest Loss", fmt(losses['PL'].min())),
        ("Avg Win / Avg Loss Ratio", fmt(abs(wins['PL'].mean()/losses['PL'].mean()) if losses['PL'].mean()!=0 else np.nan, prefix='', decimals=3)),
        ("Max Consecutive Winners", max_w),
        ("Max Consecutive Losers", max_l),
        ("Avg Shares per Trade", fmt(df['Shares'].mean(), prefix='', decimals=1)),
    ]
    for label, value in winloss_data:
        ws[f"A{row}"] = label
        ws[f"B{row}"] = value
        row += 1
    row += 1

    # Drawdown & Capital section
    row = add_section("DRAWDOWN & CAPITAL", row)
    dd_data = [
        ("Max Drawdown (P/L units)", fmt(max_dd)),
        ("Max Drawdown Duration (trades)", max_dd_dur),
        ("Max Capital Required (est.)", fmt(max_cap)),
        ("Return on Max Capital", pct(pl.sum()/max_cap if max_cap and max_cap>0 else np.nan)),
    ]
    for label, value in dd_data:
        ws[f"A{row}"] = label
        ws[f"B{row}"] = value
        row += 1
    row += 1

    # Risk-Adjusted Metrics section
    row = add_section("RISK-ADJUSTED METRICS (annualised, 252 trading days)", row)
    risk_data = [
        ("Sharpe Ratio", fmt(calc_sharpe(daily_pl), prefix='', decimals=3)),
        ("Sortino Ratio", fmt(calc_sortino(daily_pl), prefix='', decimals=3)),
        ("Calmar Ratio (Net/MaxDD)", fmt(pl.sum()/abs(max_dd) if max_dd!=0 else np.nan, prefix='', decimals=3)),
    ]
    for label, value in risk_data:
        ws[f"A{row}"] = label
        ws[f"B{row}"] = value
        row += 1

    # Set column widths
    ws.column_dimensions['A'].width = 40
    ws.column_dimensions['B'].width = 20

    # ── Sheet 2: Daily Performance ──
    ws_daily = wb.create_sheet("Daily Performance")
    ds = daily_stats(df)
    ds_reset = ds.reset_index()
    
    # Write headers
    headers = ["Date", "Trades", "Total P/L", "Win Rate"]
    for col_num, header in enumerate(headers, 1):
        cell = ws_daily.cell(row=1, column=col_num)
        cell.value = header
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")

    # Write data
    for row_num, row_data in enumerate(dataframe_to_rows(ds_reset, index=False, header=False), 2):
        for col_num, value in enumerate(row_data, 1):
            cell = ws_daily.cell(row=row_num, column=col_num)
            cell.value = value

    # Set column widths
    ws_daily.column_dimensions['A'].width = 15
    ws_daily.column_dimensions['B'].width = 12
    ws_daily.column_dimensions['C'].width = 15
    ws_daily.column_dimensions['D'].width = 12

    # ── Sheet 3: Performance By Symbol ──
    ws_symbol = wb.create_sheet("Performance By Symbol")
    if df["Symbol"].nunique() > 1:
        ss = symbol_stats(df)
        ss_reset = ss.reset_index()
        
        # Write headers
        headers = ["Symbol", "Trades", "Total P/L", "Avg P/L", "Win Rate"]
        for col_num, header in enumerate(headers, 1):
            cell = ws_symbol.cell(row=1, column=col_num)
            cell.value = header
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")

        # Write data
        for row_num, row_data in enumerate(dataframe_to_rows(ss_reset, index=False, header=False), 2):
            for col_num, value in enumerate(row_data, 1):
                cell = ws_symbol.cell(row=row_num, column=col_num)
                cell.value = value

        # Set column widths
        ws_symbol.column_dimensions['A'].width = 12
        ws_symbol.column_dimensions['B'].width = 12
        ws_symbol.column_dimensions['C'].width = 15
        ws_symbol.column_dimensions['D'].width = 15
        ws_symbol.column_dimensions['E'].width = 12
    else:
        ws_symbol[f"A1"] = "Only one symbol in dataset"

    # Save workbook
    wb.save(output_path)


# ── main ─────────────────────────────────────────────────────────────────────
#--excel argument: Added command-line option to specify Excel output file path
# python ats_performance_report.py trades.csv --excel report.xlsx --strategy MyStrategy
#Key Changes:
# 1.Added Excel dependencies: Imported openpyxl for Excel file creation and styling
# 2.New --excel argument: Added command-line option to
#   specify Excel output file path3.Three-sheet workbook: create_excel_report() function generates:
#  ◦Trade Summary: Key metrics (Net Profit, Profit Factor, Trade counts, Win/Loss details,
#   Drawdown, Risk ratios)
#  ◦Daily Performance: Date-by-date breakdown (Trades, Total P/L, Win Rate)
#  ◦Performance By Symbol: Symbol-by-symbol statistics (Trades, Total P/L, Avg P/L, Win Rate)4.Professional formatting:◦Section headers with gray background◦Column headers with blue background and white text◦Appropriately sized columns for readability5.Backward compatibility: The original --output CSV option still works
#
#  Usage:
#  python ats_performance_report.py trades.csv --output my_trades.csv
#

def main():
    args = parse_args()
    df = load_trades(args.csv_file)
    df = filter_trades(df, args)

    if df.empty:
        print("No trades match the specified filters.")
        sys.exit(1)

    strategy = args.strategy or args.csv_file.split("/")[-1].replace(".csv","")
    summary  = print_report(df, strategy)

    # Handle Excel output (preferred)
    if args.excel:
        create_excel_report(df, strategy, args.excel)
        print(f"Excel report saved to {args.excel}")
    
    # Handle legacy CSV output
    if args.output:
        pd.DataFrame([summary]).to_csv(args.output, index=False)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
