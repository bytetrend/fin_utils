import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text

# Part 1: The Machine Learning Python TemplateSave this script as multicharts_ml_optimizer.py inside your RunPod Jupyter or code execution environment. Place your exported MultiCharts CSV in the same folder.This script loads the data via pandas, trains a Random Forest Regressor to find non-linear feature importances, and applies a Decision Tree to map out the exact parameter boundaries (clusters) that yield maximum profit with minimum drawdown.
# =====================================================================
# 1. CONFIGURATION & CONFIG ZONE
# =====================================================================
# Update these string names to exactly match your MultiCharts CSV column headers
CSV_FILE_PATH = "MultiCharts_Results.csv"

# Define your Strategy Input Parameters (The variables you optimized in MultiCharts)
INPUT_PARAMETERS = [
    "Length1",
    "Length2",
    "StopLoss_Ticks",
    "ProfitTarget_Ticks",
    "RegimeFilter_Period"
]

# Define your Performance Metrics (The target outputs you want to optimize)
NET_PROFIT_COL = "Total Net Profit"
DRAWDOWN_COL = "Max Intraday Drawdown"
PROFIT_FACTOR_COL = "Profit Factor"

# =====================================================================
# 2. DATA LOADING & PREPROCESSING
# =====================================================================
print(f"[*] Loading MultiCharts optimization data from: {CSV_FILE_PATH}")
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"Could not find {CSV_FILE_PATH}. Please upload it to your workspace.")

df = pd.read_csv(CSV_FILE_PATH)
print(f"[+] Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# Clean numeric columns (Remove $, %, or commas if MultiCharts formatted them as strings)
for col in [NET_PROFIT_COL, DRAWDOWN_COL, PROFIT_FACTOR_COL] + INPUT_PARAMETERS:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(r'[$\s,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=[NET_PROFIT_COL, DRAWDOWN_COL, PROFIT_FACTOR_COL] + INPUT_PARAMETERS)

# Create a Custom Quant Metric: Profit-to-Drawdown Ratio (To avoid overfitted absolute net profit)
# We add a small epsilon to avoid division by zero
df['Quant_Score'] = df[NET_PROFIT_COL] / (df[DRAWDOWN_COL].abs() + 0.001)
# Filter for baseline viability (e.g., only look at combinations where Profit Factor > 1.1)
viable_df = df[df[PROFIT_FACTOR_COL] > 1.1].copy()

if viable_df.empty:
    print("[!] Warning: No parameter combinations met the Profit Factor > 1.1 baseline. Using full dataset.")
    viable_df = df.copy()

X = viable_df[INPUT_PARAMETERS]
y = viable_df['Quant_Score']

# =====================================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# =====================================================================
print("\n[*] Training Random Forest Regressor to determine parameter sensitivity...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n=== FEATURE IMPORTANCE RANKING (Which parameters impact performance most) ===")
for rank in range(X.shape[1]):
    print(f"{rank + 1}. Parameter '{INPUT_PARAMETERS[indices[rank]]}' : {importances[indices[rank]]*100:.2f}% importance")

# =====================================================================
# 4. DECISION TREE CLUSTERING (Isolating Robust Parameter Zones)
# =====================================================================
print("\n[*] Training Decision Tree to isolate the highest-performing parameter clusters...")

# Create a binary label: 1 for Top 15% performing settings, 0 for the rest
threshold = viable_df['Quant_Score'].quantile(0.85)
y_binary = (viable_df['Quant_Score'] >= threshold).astype(int)

# Use a shallow tree to create human-readable rules (prevents curve-fitting)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y_binary)

tree_rules = export_text(dt, feature_names=INPUT_PARAMETERS)
print("\n=== ROBUST PARAMETER BOUNDARIES (Target the paths leading to class 1) ===")
print(tree_rules)

# =====================================================================
# 5. EXTRACTING TOP COMBINATIONS FOR DEEPSEEK-R1 REVIEW
# =====================================================================
print("\n=== TOP 5 ROBUST PARAMETER COMBINATIONS ===")
top_combinations = viable_df.sort_values(by='Quant_Score', ascending=False).head(5)
summary_output = top_combinations[INPUT_PARAMETERS + [NET_PROFIT_COL, DRAWDOWN_COL, PROFIT_FACTOR_COL, 'Quant_Score']]
print(summary_output.to_string(index=False))

# Export to a tiny payload file that you can easily pipe to your LLM prompt
summary_output.to_csv("llm_payload_summary.csv", index=False)
print("\n[+] Optimization complete. Created 'llm_payload_summary.csv' for LLM analysis.")
