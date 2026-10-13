from agent.run_analysis_loop import run_analysis_loop

TARGET_CSV = "C:\\Invest\\logs\\merged\\AtsFastReversal-merged.csv"

# --- STRATEGY A: Run purely Local and Free using Ollama ---
from provider_ollama import get_ollama_handler
local_handler = get_ollama_handler()
run_analysis_loop(TARGET_CSV, llm_handler=local_handler, max_iterations=3)

# --- STRATEGY B: Run on Metered Cloud credits using RunPod ---
# from provider_runpod import get_runpod_handler
# cloud_handler = get_runpod_handler()
# run_analysis_loop(TARGET_CSV, llm_handler=cloud_handler, max_iterations=3)
