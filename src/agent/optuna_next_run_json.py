# Call your LangChain tool directly with refined target arguments
from agent.trading_optuna_schema import run_trading_optimization

optuna_next_run_json = run_trading_optimization.invoke({
    "trades_csv": "C:\\Invest\\logs\\merged\\AtsFastReversal-merged.csv",
    "n_trials": 2000,           # Double the trials to exhaustively search the tighter space
    "min_n": 35,                # Slightly increase overfitting protection
    "test_fraction": 0.25,
    "direction": "long",        # Restrict focus solely to refining this long cluster
    # Lock the search space down specifically to the high-alpha structural indicators
    "params": "ind_HMAGapMean,ind_TrendBarCount,ind_HMAGapCV,ind_ATRsFromHma"
})

print(optuna_next_run_json)
