import json
import time

# 3. The Continuous Loop Implementation
# Since your physical simulations take days, this agentic pipeline uses a stateful loop.
# It parses your historical run data, generates optimal mathematical cutoffs with Optuna,
# leverages your RunPod model to reason about what strategy to run next, and persists
# the analysis history.
def run_analysis_loop(csv_path: str, llm_handler: callable, max_iterations: int = 5):
    """
    Core entry point loop.
    Accepts an llm_handler function/lambda to remain agnostic of the backend provider.
    """
    experiment_history = []

    # Active state parameters that optimize dynamically
    current_min_n = 30
    current_test_fraction = 0.25
    target_params = ("ind_CVDSpeedPct,ind_CVDDeltaPct,ind_CVDAcelPct,ind_CVDAcelPct_1,ind_RevATRsPerSec,"
                     "ind_DeltaATRs,ind_TrendBarCount,ind_Angle,ind_PipSpeed,ind_PipSpeedNorm,ind_PipSpeedAcel,"
                     "ind_PipSpeedAcelNorm,ind_ATRsFromHma,ind_DeltaPips,ind_PipSpeedTrendPct,ind_HMAGapStdDev,"
                     "ind_HMAGapMean,ind_HMAGapCV")

    for iteration in range(max_iterations):
        print(f"\n🚀 Starting Optimization Automation Loop Run #{iteration + 1}...")

        # 1. Execute your LangChain/LlamaIndex Optuna Tool logic locally
        # (Assuming run_trading_optimization tool is imported or accessible)
        try:
            try:
                from tools import run_trading_optimization
            except ModuleNotFoundError:
                from agent.tools import run_trading_optimization

            optuna_raw_json = run_trading_optimization.invoke({
                "trades_csv": csv_path,
                "min_n": current_min_n,
                "test_fraction": current_test_fraction,
                "params": target_params
            })
            optuna_results = json.loads(optuna_raw_json)
        except Exception as e:
            print(f"❌ Error invoking local Optuna tool script: {e}")
            break

        # 2. Construct the abstract analytical reasoning prompt
        prompt = f"""
            You are a quantitative risk management agent. Your task is to analyze optimization metrics and suggest adjustments.

            LATEST OPTUNA RUN PERFORMANCE DETAILS:
            {json.dumps(optuna_results, indent=2)}

            PREVIOUS EXPERIMENT RUN HISTORY:
            {json.dumps(experiment_history, indent=2)}

            INSTRUCTIONS:
            Output your strategic thoughts followed by a raw JSON update block matching this structure:
            ```json
            {{
              "reasoning": "Your logic here",
              "next_min_n": 40,
              "next_params": "ind_PatternEntryScore,ind_CVDEntryScore"
            }}
            ```
            """

        # 3. Call the injected lambda/callback function
        print("🧠 Routing prompt to the selected LLM provider...")
        llm_output = llm_handler(prompt)

        print(f"\n🤖 LLM Agent Feedback:\n{llm_output}")

        # 4. Save historical loop metrics
        experiment_history.append({
            "iteration": iteration,
            "best_value": optuna_results.get("single_param_best", [{}])[0].get("expectancy", "N/A")
        })

        # 5. Extract JSON and adjust variables for the next loop run
        try:
            if "```json" in llm_output:
                json_start = llm_output.find("```json") + 7
                json_end = llm_output.find("```", json_start)
                updates = json.loads(llm_output[json_start:json_end].strip())
            else:
                updates = json.loads(llm_output.strip())

            current_min_n = updates.get("next_min_n", current_min_n)
            target_params = updates.get("next_params", target_params)
        except Exception:
            print("⚠️ Warning: Could not cleanly isolate JSON adjustment configurations. Retaining current bounds.")

        time.sleep(1)

