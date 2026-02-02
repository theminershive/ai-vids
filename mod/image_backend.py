#!/usr/bin/env python3
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from config import AppConfig
from openai_client import build_openai_client
from visuals import generate_visual


@dataclass(frozen=True)
class ImageResult:
    url: Optional[str] = None
    b64: Optional[str] = None
    extension: str = ".png"

IMAGE_DOWNLOAD_TIMEOUT_S = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_S", "900"))


def generate_openai_image(prompt: str, config: AppConfig) -> ImageResult:
    client = build_openai_client()
    response = client.images.generate(
        model=config.openai.image_model,
        prompt=prompt,
        size="1024x1792",
        quality="standard",
    )
    if response.data and response.data[0].url:
        return ImageResult(url=response.data[0].url, extension=".png")
    if response.data and response.data[0].b64_json:
        return ImageResult(b64=response.data[0].b64_json, extension=".png")
    raise RuntimeError("OpenAI image generation returned no image data.")


def generate_flux_image(prompt: str, config: AppConfig) -> ImageResult:
    path = generate_visual(prompt, section_idx=1, style_name=config.channel.image_generation_style)
    if not path:
        raise RuntimeError("ComfyUI generation returned no image.")
    return ImageResult(url=str(path), extension=Path(path).suffix or ".png")


def generate_image(prompt: str, config: AppConfig) -> ImageResult:
    backend = config.image_backend.type
    logging.info("Image backend: %s", backend)
    if backend == "flux":
        return generate_flux_image(prompt, config)
    if backend == "comfyui":
        path = generate_visual(prompt, section_idx=1, style_name=config.channel.image_generation_style)
        if not path:
            raise RuntimeError("ComfyUI generation returned no image.")
        return ImageResult(url=str(path), extension=Path(path).suffix or ".png")
    if backend == "leonardo":
        path = generate_visual(prompt, section_idx=1, style_name=config.channel.image_generation_style)
        if not path:
            raise RuntimeError("Leonardo generation returned no image.")
        return ImageResult(url=str(path), extension=Path(path).suffix or ".png")
    if backend == "openai":
        return generate_openai_image(prompt, config)
    logging.warning("Unknown IMAGE_BACKEND '%s'; falling back to OpenAI.", backend)
    return generate_openai_image(prompt, config)


def save_image(result: ImageResult, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if result.url:
        if result.url.startswith("file://"):
            src = Path(result.url[7:])
            if src.exists():
                dest.write_bytes(src.read_bytes())
                return dest
        src = Path(result.url)
        if src.exists():
            dest.write_bytes(src.read_bytes())
            return dest
        resp = requests.get(result.url, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(8192):
                handle.write(chunk)
        return dest
    if result.b64:
        with open(dest, "wb") as handle:
            handle.write(base64.b64decode(result.b64))
        return dest
    raise RuntimeError("No image data to save.")
