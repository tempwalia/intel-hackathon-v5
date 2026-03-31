import os
import requests

# Set the QWEN_API_URL from the environment variable
QWEN_API_URL = os.getenv("QWEN_API_URL")
if not QWEN_API_URL:
    print("❌ QWEN_API_URL is not set.")
    exit(1)

# Configuration variables
TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("QWN_MAX_TOKENS", "2000"))
DEBUG_OUTPUT = os.getenv("BRD_DEBUG_OUTPUT", "false").lower() in {"1", "true", "yes", "on"}

def send_request_to_qwen(prompt):
    """
    Sends a request to the Qwen API with the given prompt.
    """
    headers = {
        "Content-Type": "application/json",
        # Add Authorization header if required
        # "Authorization": "Bearer YOUR_API_KEY"
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_new_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    if DEBUG_OUTPUT:
        print("Payload:", payload)
        print("Headers:", headers)

    try:
        print("🔄 Sending request to Qwen API...")
        response = requests.post(QWEN_API_URL, json=payload, headers=headers)
        if DEBUG_OUTPUT:
            print("Response Status Code:", response.status_code)
            print("Response Text:", response.text)
        response.raise_for_status()
        result = response.json()
        print("✅ API Response:", result)
        return result
    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
