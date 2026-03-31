import os
import json
import requests
from transformers import pipeline

# ==============================
# QWEN API CONFIGURATION (PRIMARY)
# ==============================
QWEN_API_URL = os.getenv("QWEN_API_URL", "").strip()
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "2000"))
DEBUG_OUTPUT = os.getenv("BRD_DEBUG_OUTPUT", "false").lower() in {"1", "true", "yes", "on"}

# ==============================
# LOCAL MODEL CONFIGURATION (FALLBACK)
# ==============================
QWEN_MODEL = os.getenv("QWEN_MODEL", "meta-llama/Llama-2-7b-chat-hf")
FALLBACK_QWEN_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/phi-2",
]

_qwen_pipe = None
_loaded_model = None


class QwenAPIWrapper:
    """Wraps Qwen API (messages-based) to match HuggingFace pipeline interface."""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/chat/completions"):
            self.api_url = self.api_url + "/chat/completions"

    def __call__(self, prompt, **kwargs):
        """Call Qwen API with messages format."""
        max_tokens = kwargs.get("max_new_tokens", QWEN_MAX_TOKENS)
        temperature = kwargs.get("temperature", QWEN_TEMPERATURE)

        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

        if DEBUG_OUTPUT:
            print("[QwenAPIWrapper] Payload:", json.dumps(payload, indent=2)[:200])
            print("[QwenAPIWrapper] Headers:", headers)

        try:
            print(f"[QwenAPIWrapper] 🔄 Sending request to Qwen API: {self.api_url}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            
            if DEBUG_OUTPUT:
                print("[QwenAPIWrapper] Response Status Code:", response.status_code)
                print("[QwenAPIWrapper] Response Text:", response.text[:500])
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response (handle various API response formats)
            text = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice:
                    text = choice["message"].get("content", "")
                elif "text" in choice:
                    text = choice["text"]
            
            if DEBUG_OUTPUT:
                print("[QwenAPIWrapper] ✅ API Response received")
            
            # Return in HF pipeline format
            return [{"generated_text": prompt + text}]
        
        except requests.RequestException as e:
            print(f"[QwenAPIWrapper] ❌ API request failed: {e}")
            raise RuntimeError(f"Qwen API error: {e}")


def _try_qwen_api(api_url: str):
    """Attempt to connect to Qwen API.
    Returns QwenAPIWrapper on success, None on failure."""
    if not api_url:
        return None
    try:
        wrapper = QwenAPIWrapper(api_url)
        # Test with a simple request
        test_response = requests.post(
            api_url if api_url.endswith("/chat/completions") else api_url + "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "max_new_tokens": 10,
                "temperature": 0.1,
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        test_response.raise_for_status()
        print(f"[qwen_setup] ✅ Connected to Qwen API: {api_url}")
        return wrapper
    except Exception as e:
        print(f"[qwen_setup] ⚠️ Qwen API '{api_url}' not reachable: {e}")
        return None


class LocalModelWrapper:
    """Wraps local HuggingFace model to match HuggingFace pipeline interface."""

    def __init__(self, pipeline_obj):
        self.pipeline = pipeline_obj

    def __call__(self, prompt, **kwargs):
        return self.pipeline(prompt, **kwargs)


def torch_available():
    """Check if torch with GPU support is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_qwen_pipe():
    global _qwen_pipe, _loaded_model

    if _qwen_pipe is None:
        # --- Priority 1: Try Qwen API ---
        if QWEN_API_URL:
            print("[qwen_setup] Trying Qwen API (Priority 1)...")
            api_wrapper = _try_qwen_api(QWEN_API_URL)
            if api_wrapper is not None:
                _qwen_pipe = api_wrapper
                _loaded_model = f"Qwen API: {QWEN_API_URL}"
                return _qwen_pipe
            print("[qwen_setup] Qwen API unavailable, falling back to local models...")
        else:
            print("[qwen_setup] QWEN_API_URL not set, using local models...")

        # --- Priority 2+: Local models ---
        candidate_models = [QWEN_MODEL] + [m for m in FALLBACK_QWEN_MODELS if m != QWEN_MODEL]
        last_error = None

        for model_name in candidate_models:
            try:
                print(f"[qwen_setup] Loading local model: {model_name}...")
                device = 0 if torch_available() else -1
                local_pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device=device,
                    model_kwargs={"load_in_8bit": True} if device >= 0 else {}
                )
                _qwen_pipe = LocalModelWrapper(local_pipe)
                _loaded_model = model_name
                if model_name != QWEN_MODEL:
                    print(f"[qwen_setup] ⚠️ Requested model '{QWEN_MODEL}' failed, using fallback '{model_name}'")
                else:
                    print(f"[qwen_setup] ✅ Loaded local model: {model_name}")
                break
            except ValueError as e:
                last_error = e
                if "does not recognize this architecture" in str(e) or "model type" in str(e):
                    continue
                raise
            except Exception as e:
                last_error = e
                print(f"[qwen_setup] ⚠️ Failed to load {model_name}: {e}")
                continue

        if _qwen_pipe is None:
            raise ValueError(
                "Could not load any model or API. "
                "1. Set QWEN_API_URL to a Qwen API endpoint (recommended), or "
                "2. Set QWEN_MODEL to a compatible local model. "
                f"Last error: {last_error}"
            )

    return _qwen_pipe


def get_loaded_model_name() -> str | None:
    return _loaded_model
