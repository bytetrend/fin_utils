from llama_index.core.tools import FunctionTool

from agent.tools import run_ats_optuna_optimizer
from agent.trading_optuna_schema import run_trading_optimization

#  2. The LlamaIndex Tool Integration
#  To use this inside a LlamaIndex application wrapper, translate your function logic
#  directly into a FunctionTool.

llama_trading_optuna_tool = FunctionTool.from_defaults(
    fn=run_trading_optimization,
    name="ats_optuna_optimizer",
    description="Optimizes trading parameter columns like ind_PatternEntryScore and ind_CVDEntryScore against a trades CSV pool."
)

# Wrap standard python execution logic seamlessly for LlamaIndex architectures
llama_optuna_tool = FunctionTool.from_defaults(
    fn=run_ats_optuna_optimizer,
    name="ats_optuna_optimizer",
    description=(
        "Executes localized hyperparameter trading simulation optimizations "
        "across specific datasets using Optuna, enforcing strict overfitting filters."
    )
)
