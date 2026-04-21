"""
clipping.ai_provider — AI provider abstraction for Gemini or custom gateway.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    return base_url[:-1] if base_url.endswith("/") else base_url


def build_client(cfg):
    """Create a google-genai client for either Gemini or custom gateway."""
    from google import genai
    from google.genai import types

    http_kwargs: dict[str, Any] = {}

    base_url = _normalize_base_url(getattr(cfg, "ai_base_url", ""))
    api_version = (getattr(cfg, "ai_api_version", "") or "").strip()

    if base_url:
        http_kwargs["base_url"] = base_url
        # For gateways that already include /v1 in the URL, let the caller control version.
        if api_version:
            http_kwargs["api_version"] = api_version
        else:
            # Avoid auto-appending /v1beta when a fully-qualified custom gateway is used.
            http_kwargs["api_version"] = ""
        http_kwargs["base_url_resource_scope"] = types.ResourceScope.COLLECTION
    elif api_version:
        http_kwargs["api_version"] = api_version

    return genai.Client(
        api_key=cfg.ai_api_key,
        http_options=types.HttpOptions(**http_kwargs),
    )


def generate_json_with_retry(client, model, fallback_model, contents, config):
    """Generate JSON from any supported AI provider using the google-genai SDK."""
    MAX_ATTEMPTS = 10
    INITIAL_WAIT_SECONDS = 60
    WAIT_INCREMENT_SECONDS = 30
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def _extract_status_code(exc: Exception):
        for attr in ("status_code", "code", "status"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

        match = re.search(r"\b(408|429|500|502|503|504)\b", str(exc))
        return int(match.group(1)) if match else None

    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, json.JSONDecodeError):
            return True

        code = _extract_status_code(exc)
        if code in RETRYABLE_STATUS_CODES:
            return True

        msg = str(exc).lower()
        keywords = (
            "timeout", "temporarily unavailable", "deadline",
            "connection reset", "connection aborted", "service unavailable",
        )
        return any(k in msg for k in keywords)

    last_exc = None
    status_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[AI] Attempt {attempt}/{MAX_ATTEMPTS}...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("AI provider mengembalikan response.text kosong.")
            return json.loads(text)
        except Exception as exc:
            last_exc = exc
            status_code = _extract_status_code(exc)
            retryable = _is_retryable(exc)
            print(
                f"[AI] Attempt {attempt}/{MAX_ATTEMPTS} gagal | status={status_code} | error={exc}"
            )
            if (not retryable) or attempt == MAX_ATTEMPTS:
                break
            wait_seconds = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
            print(f"[AI] Retry lagi dalam {wait_seconds} detik...")

    print(f"[AI] Percobaan dengan model utama ({model}) gagal.")
    if fallback_model:
        print(f"[AI] Mencoba satu kali lagi dengan fallback model ({fallback_model})...")
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("AI fallback mengembalikan response.text kosong.")
            return json.loads(text)
        except Exception as exc_fallback:
            print(f"[AI] Fallback model gagal | error={exc_fallback}")
            raise RuntimeError(
                f"Gagal memanggil AI utama & fallback. Laporan Utama status={status_code}, error={last_exc} | Laporan Fallback error={exc_fallback}"
            ) from exc_fallback

    raise RuntimeError(
        f"Gagal memanggil AI setelah {MAX_ATTEMPTS} percobaan. Error terakhir: {last_exc}"
    ) from last_exc
