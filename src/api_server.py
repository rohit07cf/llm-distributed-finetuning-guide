#!/usr/bin/env python3
"""FastAPI server for serving the fine-tuned LLM.

Exposes a REST API for text generation and health checking.

Usage:
    uvicorn src.api_server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Inference API",
    description="REST API for fine-tuned LLM text generation",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model references (loaded on startup)
_model = None
_tokenizer = None


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(..., min_length=1, description="The input prompt")
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    """Response body for the /generate endpoint."""

    response: str
    tokens_generated: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Response body for the /healthz endpoint."""

    status: str
    model_loaded: bool
    model_name: str


def load_model_on_startup():
    """Load the model and tokenizer when the server starts."""
    global _model, _tokenizer

    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    adapter_path = os.environ.get("ADAPTER_PATH", "outputs/lora_sft")
    quantization = int(os.environ.get("QUANTIZATION", "0"))

    from src.inference import load_model

    _model, _tokenizer = load_model(
        base_model=base_model,
        adapter_path=adapter_path,
        quantization=quantization,
    )
    logger.info(f"Model loaded: {base_model} + adapter from {adapter_path}")


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    try:
        load_model_on_startup()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("Server starting without model — /generate will return errors")


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for liveness and readiness probes."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_name=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a text response for the given prompt."""
    if _model is None or _tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    from src.inference import generate_response

    start = time.time()
    try:
        response_text = generate_response(
            model=_model,
            tokenizer=_tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.time() - start) * 1000, 2)
    tokens_generated = len(_tokenizer.encode(response_text))

    return GenerateResponse(
        response=response_text,
        tokens_generated=tokens_generated,
        latency_ms=latency_ms,
    )


@app.get("/")
async def read_index():
    """Serve the recruiter-friendly demo page."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
    )
