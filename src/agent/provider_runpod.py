import os
import requests


class RunPodClient:
    def __init__(self, endpoint_url: str = None, api_key: str = None):
        # Fallback automatically to environment variables if parameters are omitted
        self.url = endpoint_url or os.environ.get("RUNPOD_LLM_URL", "https://runpod.ai")
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")

    def query(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"RunPod execution connection error occurred: {str(e)}"


# Instantiation template helper for your master main launch sequence
def get_runpod_handler():
    client = RunPodClient()
    return lambda prompt: client.query(prompt)
