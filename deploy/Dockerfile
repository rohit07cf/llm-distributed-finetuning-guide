FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

LABEL maintainer="Rohit Goswami"
LABEL description="LLM fine-tuning inference API server"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY static/ static/

# Environment variables (override at runtime)
ENV BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
ENV ADAPTER_PATH=outputs/lora_sft
ENV QUANTIZATION=0
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["python3", "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
