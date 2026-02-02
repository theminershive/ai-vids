#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import requests
from openai import OpenAI

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.176:11434").rstrip("/")
OLLAMA_NUM_CTX = os.getenv("OLLAMA_NUM_CTX")
OLLAMA_TEMPERATURE = os.getenv("OLLAMA_TEMPERATURE")

_OLLAMA_TAGS_CACHE: Optional[List[str]] = None


def build_openai_client() -> OpenAI:
    return OpenAI(http_client=httpx.Client())


def _ollama_tags() -> List[str]:
    global _OLLAMA_TAGS_CACHE
    if _OLLAMA_TAGS_CACHE is not None:
        return _OLLAMA_TAGS_CACHE
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _OLLAMA_TAGS_CACHE = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        _OLLAMA_TAGS_CACHE = []
    return _OLLAMA_TAGS_CACHE


def _should_use_ollama(model: str) -> bool:
    if not model:
        return False
    m = model.strip().lower()
    if ":" in m or m.startswith("qwen"):
        return True
    return model in _ollama_tags()


def _ollama_chat(
    messages: List[Dict[str, str]],
    model: str,
    *,
    temperature: Optional[float] = None,
    timeout: int = 180,
) -> str:
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    options: Dict[str, Any] = {}
    if OLLAMA_NUM_CTX:
        try:
            options["num_ctx"] = int(OLLAMA_NUM_CTX)
        except ValueError:
            pass
    if temperature is not None:
        options["temperature"] = temperature
    elif OLLAMA_TEMPERATURE:
        try:
            options["temperature"] = float(OLLAMA_TEMPERATURE)
        except ValueError:
            pass
    if options:
        payload["options"] = options

    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama chat failed HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    message = data.get("message") or {}
    return (message.get("content") or "").strip()


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
) -> str:
    provider = LLM_PROVIDER
    if provider == "ollama" or (provider == "auto" and _should_use_ollama(model)):
        return _ollama_chat(messages, model, temperature=temperature, timeout=timeout or 180)

    client = build_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()
