"""RunPod Serverless handler for LLM inference.

This module wraps the inference pipeline for deployment as a
RunPod Serverless endpoint. The model loads once on cold start,
then handles requests until the worker scales down.

Environment variables:
    BASE_MODEL: HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
    ADAPTER_REPO: HuggingFace repo with LoRA adapter (e.g., rohit07cf/medical-qa-qlora-adapter)
    QUANTIZATION: Quantization bits, 0/4/8 (default: 4)
"""

import os
import torch

try:
    import runpod
except ImportError:
    raise ImportError("Install runpod: pip install runpod")

# ── Global model state (loaded once per cold start) ────────────────────

MODEL = None
TOKENIZER = None


def load_model():
    """Download adapter from HuggingFace and load the model."""
    global MODEL, TOKENIZER

    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    adapter_repo = os.environ.get("ADAPTER_REPO", "")
    quantization = int(os.environ.get("QUANTIZATION", "4"))

    adapter_path = None
    if adapter_repo:
        from huggingface_hub import snapshot_download
        adapter_path = snapshot_download(repo_id=adapter_repo)
        print(f"Downloaded adapter from {adapter_repo} -> {adapter_path}")

    from src.inference import load_model as _load
    MODEL, TOKENIZER = _load(
        base_model=base_model,
        adapter_path=adapter_path,
        quantization=quantization,
    )
    print(f"Model loaded: {base_model} (quantization={quantization})")


def handler(event):
    """Handle a single inference request.

    Expected input:
        {
            "input": {
                "prompt": "Explain heart attack symptoms",
                "max_new_tokens": 512,
                "temperature": 0.7
            }
        }

    Returns:
        {
            "response": "...",
            "tokens_generated": 42
        }
    """
    global MODEL, TOKENIZER

    if MODEL is None:
        load_model()

    job_input = event.get("input", {})
    prompt = job_input.get("prompt", "")

    if not prompt:
        return {"error": "No prompt provided"}

    max_new_tokens = job_input.get("max_new_tokens", 512)
    temperature = job_input.get("temperature", 0.7)
    top_p = job_input.get("top_p", 0.9)

    from src.inference import generate_response
    response_text = generate_response(
        model=MODEL,
        tokenizer=TOKENIZER,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    tokens_generated = len(TOKENIZER.encode(response_text))

    return {
        "response": response_text,
        "tokens_generated": tokens_generated,
    }


# ── RunPod entry point ─────────────────────────────────────────────────

runpod.serverless.start({"handler": handler})
