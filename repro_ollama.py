import os
import sys
import litellm
from src.config import MODEL_CONFIGS, ACTIVE_MODEL_TIER, OLLAMA_BASE_URL, OLLAMA_API_KEY, validate_ollama_config

# Mock logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ollama():
    try:
        model = MODEL_CONFIGS.get(ACTIVE_MODEL_TIER, MODEL_CONFIGS["default"])
        print(f"Testing model: {model}")
        print(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
        # Dont print API key for security, just check if it exists
        print(f"OLLAMA_API_KEY exists: {bool(OLLAMA_API_KEY)}")

        kwargs = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello, are you working?"}
            ],
            "max_tokens": 100,
            "temperature": 0.2,
            "timeout": 30
        }

        if "ollama" in model:
            validate_ollama_config()
            kwargs["api_base"] = OLLAMA_BASE_URL
            kwargs["api_key"] = OLLAMA_API_KEY
            print("Injected Ollama config into kwargs")

        print("Calling litellm.completion...")
        response = litellm.completion(**kwargs)
        print("Response received:")
        print(response)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama()
