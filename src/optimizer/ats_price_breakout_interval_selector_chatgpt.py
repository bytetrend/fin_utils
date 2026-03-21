import pandas as pd
import os

#CHATGPT
# Input CSV file
input_path = "C:\Invest\logs\screener"
input_file = "AtsPriceBrkout.csv"

# Output directory
output_dir = "C:\Invest\logs\screener"
output_file = "AtsPriceBrkout_IntervalSelector_chatgpt.csv"
os.makedirs(output_dir, exist_ok=True)

# Column names since CSV has no header
cols = [
    "Symbol",
    "Interval",
    "BarNumber",
    "BarDate",
    "BarTime",
    "SignalCount",
    "ComputerDateTime"
]

# Read CSV
df = pd.read_csv( os.path.join(input_path,input_file), header=None, names=cols)

# Convert types
df["BarNumber"] = pd.to_numeric(df["BarNumber"], errors="coerce")
df["SignalCount"] = pd.to_numeric(df["SignalCount"], errors="coerce")

# Create sortable datetime column from BarDate + BarTime
df["BarDateTime"] = pd.to_datetime(
    df["BarDate"].astype(str) + " " + df["BarTime"].astype(str),
    errors="coerce"
)

# -------------------------------------------------------------------
# STEP 1: Keep only the latest row per (Symbol, Interval)
# -------------------------------------------------------------------

df_latest = (
    df.sort_values(["Symbol", "Interval", "BarDateTime", "BarNumber"])
      .groupby(["Symbol", "Interval"], as_index=False)
      .tail(1)
)

# ---------------------------------------------------
# Step 2: Select highest SignalCount per Symbol
# ---------------------------------------------------

df_result = (
    df_latest.sort_values(["Symbol", "SignalCount"], ascending=[True, False])
             .groupby("Symbol", as_index=False)
             .head(1)
)

# Keep required columns
df_result = df_result[["Symbol", "Interval", "SignalCount", "ComputerDateTime"]]

# ---------------------------------------------------
# Step 3: Write single output file
# ---------------------------------------------------

outfile = os.path.join(output_dir, output_file)
df_result.to_csv(outfile, index=False)

print("Processing completed.")