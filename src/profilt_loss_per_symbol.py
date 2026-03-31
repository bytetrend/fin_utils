import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'C:/Invest/logs/signal/sum.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Convert 'Profit/Loss' column to numeric, handling any potential errors
df['Profit/Loss'] = pd.to_numeric(df['Profit/Loss'], errors='coerce')

# Group the data by 'Symbol' and sum the 'Profit/Loss'
profit_summary_by_symbol = df.groupby('Symbol')['Profit/Loss'].sum()

# Print the profit summary for each symbol
print(profit_summary_by_symbol.sort_values(ascending=False))
# Calculate the total sum of all profits across all symbols
total_profit = profit_summary_by_symbol.sum()

# Print the profit summary for each symbol
print(total_profit)

# Print the total sum of all profits