"""Streamlit UI for the Medical QA Assistant.

Deploy for free on Streamlit Community Cloud.

Usage (local):
    RUNPOD_ENDPOINT_ID=your-endpoint-id \
    RUNPOD_API_KEY=your-api-key \
    streamlit run ui/app.py
"""

import os
import time

import requests
import streamlit as st

# ── Configuration ──────────────────────────────────────────────────────

RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


# ── RunPod API helpers ─────────────────────────────────────────────────

def call_runpod_sync(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> dict:
    """Send a synchronous request to RunPod serverless endpoint."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
    }

    # Use /runsync for synchronous (waits for result, up to 30s)
    resp = requests.post(f"{RUNPOD_BASE_URL}/runsync", json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "COMPLETED":
        return data.get("output", {})

    # If it takes longer, poll /status
    job_id = data.get("id")
    return poll_runpod_job(job_id, headers)


def poll_runpod_job(job_id: str, headers: dict, max_wait: int = 300) -> dict:
    """Poll a RunPod async job until completion."""
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(f"{RUNPOD_BASE_URL}/status/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        if status == "COMPLETED":
            return data.get("output", {})
        elif status == "FAILED":
            return {"error": data.get("error", "Job failed")}
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            time.sleep(2)
        else:
            return {"error": f"Unknown status: {status}"}

    return {"error": "Request timed out"}


# ── Streamlit UI ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medical QA Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("Medical QA Assistant")
st.caption("Powered by fine-tuned LLaMA 3 with QLoRA")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max tokens", 64, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What are the symptoms of a heart attack?",
        "Explain the difference between type 1 and type 2 diabetes",
        "What is the treatment for community-acquired pneumonia?",
        "Describe the management of acute kidney injury",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
        st.warning("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in Streamlit secrets")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prefill = st.session_state.pop("prefill", "")
prompt = st.chat_input("Ask a medical question...") or prefill

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
            st.error("RunPod credentials not configured. Set them in Streamlit Cloud secrets.")
        else:
            with st.spinner("Thinking..."):
                try:
                    result = call_runpod_sync(prompt, max_tokens, temperature)
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        response = f"Error: {result['error']}"
                    else:
                        response = result.get("response", "No response received")
                        st.markdown(response)
                        tokens = result.get("tokens_generated", "?")
                        st.caption(f"Tokens generated: {tokens}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
                    response = f"Connection error: {e}"

            st.session_state.messages.append({"role": "assistant", "content": response})
