#!/usr/bin/env python3
from __future__ import annotations

import httpx
from openai import OpenAI


def build_openai_client() -> OpenAI:
    return OpenAI(http_client=httpx.Client())
