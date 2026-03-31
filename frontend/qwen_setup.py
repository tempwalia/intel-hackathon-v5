import os
import json
import requests
from transformers import pipeline

QWEN_MODEL = os.getenv("QWEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
QWEN_API_URL = os.getenv("QWEN_API_URL", "").strip()
FALLBACK_QWEN_MODELS = [
    "microsoft/phi-2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

_qwen_pipe = None
_loaded_model = None


class QwenAPIWrapper:
    """Wraps a remote Qwen-compatible API endpoint to match
    the HuggingFace pipeline __call__ interface so that
    brd_gen.call_qwen() works without changes."""

    def __init__(self, base_url: str):
        # Normalise: strip trailing slash, ensure /v1/completions path
        self.base_url = base_url.rstrip("/")
        if "/v1/" not in self.base_url:
            self.completions_url = self.base_url + "/v1/completions"
        else:
            self.completions_url = self.base_url

    def __call__(self, prompt, **kwargs):
        max_tokens = kwargs.get("max_new_tokens", 2000)
        temperature = kwargs.get("temperature", 0.2)
        do_sample = kwargs.get("do_sample", temperature > 0)

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature if do_sample else 0,
        }

        resp = requests.post(
            self.completions_url,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible format: {"choices": [{"text": "..."}]}
        if "choices" in data and data["choices"]:
            text = data["choices"][0].get("text", "")
            return [{"generated_text": prompt + text}]

        # Fallback: raw "text" or "generated_text" key
        text = data.get("generated_text") or data.get("text") or json.dumps(data)
        return [{"generated_text": prompt + text}]


def _try_api_endpoint(url: str):
    """Attempt a lightweight connection check to the API URL.
    Returns a QwenAPIWrapper on success, None on failure."""
    try:
        wrapper = QwenAPIWrapper(url)
        # Quick health probe — GET the base URL (most servers respond to this)
        requests.get(url.rstrip("/"), timeout=5)
        return wrapper
    except Exception as e:
        print(f"⚠️ QWEN_API_URL '{url}' not reachable: {e}")
        return None


def get_qwen_pipe():
    global _qwen_pipe, _loaded_model

    if _qwen_pipe is None:
        # --- Priority 1: Try remote API endpoint ---
        if QWEN_API_URL:
            print(f"[qwen_setup] Trying API endpoint: {QWEN_API_URL}")
            api_pipe = _try_api_endpoint(QWEN_API_URL)
            if api_pipe is not None:
                _qwen_pipe = api_pipe
                _loaded_model = f"api:{QWEN_API_URL}"
                print(f"[qwen_setup] Connected to API endpoint: {QWEN_API_URL}")
                return _qwen_pipe
            print("[qwen_setup] API endpoint unavailable, falling back to local models...")

        # --- Priority 2 & 3: Local models ---
        candidate_models = [QWEN_MODEL] + [m for m in FALLBACK_QWEN_MODELS if m != QWEN_MODEL]
        last_error = None

        for model_name in candidate_models:
            try:
                _qwen_pipe = pipeline("text-generation", model=model_name)
                _loaded_model = model_name
                if model_name != QWEN_MODEL:
                    print(f"⚠️ Requested model '{QWEN_MODEL}' failed, using fallback '{model_name}'")
                break
            except ValueError as e:
                last_error = e
                if "does not recognize this architecture" in str(e) or "model type" in str(e):
                    continue
                raise
            except Exception as e:
                last_error = e
                continue

        if _qwen_pipe is None:
            raise ValueError(
                "Could not load any Qwen text-generation model. "
                "Set QWEN_API_URL to a remote endpoint, or "
                "set QWEN_MODEL to a compatible model (e.g., Qwen/Qwen2.5-0.5B-Instruct), "
                "or upgrade transformers. "
                f"Last error: {last_error}"
            )

    return _qwen_pipe


def get_loaded_model_name() -> str | None:
    return _loaded_model
