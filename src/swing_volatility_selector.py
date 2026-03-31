import os

import pandas as pd

from constants import OUTPUT_DIR

#This process adds a column reversal score to the output file of swing_volatility_calculator
# then it generates a new file sorted descending on reverse score column.
#Revers score applies weights to multiple columns to generate reversal score with the best reversal candidates.

df = pd.read_csv(os.path.join(OUTPUT_DIR,"volatility_beta_report.csv"))

# Sort by reversal ratio descending (most reversals first)
df_sorted = df.sort_values(by="Reversal Ratio (%)", ascending=False)

# Save or display
df_sorted.to_csv("reversal_ranked_report.csv", index=False)
# Example hybrid score: weight reversal ratio most heavily
df['ReversalScore'] = (
    0.6 * df["Reversal Ratio (%)"] +
    0.2 * df["Volatility StdDev (%)"] +
    0.1 * abs(df["Beta vs SPY (30d)"]) +
    0.1 * (df["Alpha (annualized)"] < 0).astype(int) * 5
)

# Then sort:
df = df.sort_values("ReversalScore", ascending=False)
if not df.empty:
    csv_path = os.path.join(OUTPUT_DIR, "volatility_beta_selection.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV Report saved: {csv_path}")