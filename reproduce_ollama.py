import litellm
import time
import os

# Configuration from the codebase
model_string = "ollama_chat/deepseek-v3.1:671b-cloud"
api_base = "http://localhost:11434"

print(f"Testing connection to {api_base} with model {model_string}")

try:
    start = time.time()
    response = litellm.completion(
        model=model_string,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        temperature=0.0,
        api_base=api_base,
        timeout=30 # Short timeout for test
    )
    print("Success!")
    print(response.choices[0].message.content)
    print(f"Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")

# Try with 127.0.0.1 to see if it fixes localhost resolution issues
api_base_ip = "http://127.0.0.1:11434"
print(f"\nTesting connection to {api_base_ip} with model {model_string}")
try:
    start = time.time()
    response = litellm.completion(
        model=model_string,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        temperature=0.0,
        api_base=api_base_ip,
        timeout=30
    )
    print("Success with IP!")
    print(response.choices[0].message.content)
    print(f"Time taken: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error with IP: {e}")
