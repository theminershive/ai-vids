#!/usr/bin/env python3
import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.176:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")


def main() -> None:
    tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    tags.raise_for_status()
    data = tags.json()
    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
    print("Available models:", models)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    content = (resp.json().get("message") or {}).get("content", "")
    print("Response:", content[:200])


if __name__ == "__main__":
    main()
