#!/usr/bin/env python3
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import requests


def _read_text_or_path(value: str) -> str:
    if not value:
        return ""
    p = Path(value)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return value.strip()


def _read_ref_audio(value: str) -> str:
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        try:
            r = requests.get(value, timeout=30)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        except Exception as exc:
            logging.warning("Failed to fetch ref audio URL: %s", exc)
            return ""
    p = Path(value)
    if p.exists():
        try:
            return base64.b64encode(p.read_bytes()).decode("utf-8")
        except Exception:
            return ""
    # assume base64
    try:
        base64.b64decode(value, validate=True)
        return value.strip()
    except Exception:
        return ""


def resolve_voice(
    requested_voice: str | None,
    default_voice: str,
    fallback_voice: str,
    clone_path: str | None = None,
) -> Tuple[str, Dict[str, str], str]:
    """
    Resolve requested voice into a safe final voice + clone extras.
    Returns: (final_voice, extras, reason)
    """
    req = (requested_voice or "").strip()
    if not req or req.lower() in ("default", "auto"):
        req = default_voice

    extras: Dict[str, str] = {}
    reason = "ok"

    if req.lower() == "valentino":
        prompt = _read_text_or_path(os.getenv("TTS_VALENTINO_VOICE_CLONE_PROMPT", "").strip())
        ref_audio = _read_ref_audio(os.getenv("TTS_VALENTINO_REF_AUDIO", "").strip())
        x_vector_only = os.getenv("TTS_X_VECTOR_ONLY_MODE", "0").strip() == "1"
        ref_text = os.getenv("TTS_VALENTINO_REF_TEXT", "").strip()

        if prompt:
            extras["voice_clone_prompt"] = prompt
        else:
            if not ref_audio and clone_path:
                ref_audio = _read_ref_audio(clone_path)
            if ref_audio:
                extras["ref_audio"] = ref_audio
                extras["reference_audio"] = ref_audio
                if ref_text:
                    extras["ref_text"] = ref_text
            else:
                reason = "missing_clone_prompt_or_ref_audio"
                return req, {}, reason

    return req, extras, reason
