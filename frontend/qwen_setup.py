import os
from transformers import pipeline

QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-0.8B")
FALLBACK_QWEN_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

_qwen_pipe = None
_loaded_model = None


def get_qwen_pipe():
    global _qwen_pipe, _loaded_model

    if _qwen_pipe is None:
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
                "Set QWEN_MODEL to a compatible model (e.g., Qwen/Qwen2.5-0.5B-Instruct) "
                "or upgrade transformers. "
                f"Last error: {last_error}"
            )

    return _qwen_pipe


def get_loaded_model_name() -> str | None:
    return _loaded_model