import requests


class OllamaClient:
    def __init__(self, host_url: str = "http://localhost:11434/api/chat", model_name: str = "deepseek-r1:14b"):
        self.url = host_url
        self.model = model_name

    def query(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 65536  # <-- Best way to force your 64K context limit safely via code!
            }
        }
        try:
            # Note: Deep reasoning blocks take longer to process. A 180s timeout may be necessary
            response = requests.post(self.url, json=payload, timeout=180)
            response.raise_for_status()

            # The full JSON payload from the API
            data = response.json()

            # 1. Clean response for your agent (unaffected by metrics)
            content = data['message']['content']

            # 2. Extract metrics safely from the root object
            prompt_tokens = data.get('prompt_eval_count', 0)
            generation_tokens = data.get('eval_count', 0)
            total_context_used = prompt_tokens + generation_tokens

            # Print to your console terminal for debugging (hidden from the agent)
            print(
                f"--- [Ollama Metrics] Prompt Tokens: {prompt_tokens} | Gen Tokens: {generation_tokens} | Total Context Used: {total_context_used}/65536 ---")

            return content
        except Exception as e:
            return f"Ollama local pipeline execution error occurred: {str(e)}"


# Instantiation template helper for your master main launch sequence
def get_ollama_handler():
    client = OllamaClient()
    return lambda prompt: client.query(prompt)
