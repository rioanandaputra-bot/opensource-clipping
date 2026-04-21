"""
clipping.ai_provider — AI provider abstraction for Gemini or OpenAI-compatible gateway.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from urllib.parse import urlsplit, urlunsplit


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""

    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break

    rebuilt = urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment)).rstrip("/")
    if rebuilt.endswith("/api"):
        rebuilt = rebuilt[:-4]
    return rebuilt


def build_client(cfg):
    """Create a provider descriptor for either Gemini or gateway."""
    provider = getattr(cfg, "ai_provider", "gemini")
    if provider == "gateway":
        return {
            "provider": "gateway",
            "base_url": _normalize_base_url(getattr(cfg, "ai_base_url", "")),
            "api_key": getattr(cfg, "ai_api_key", ""),
        }

    from google import genai
    from google.genai import types

    http_kwargs: dict[str, Any] = {}
    base_url = _normalize_base_url(getattr(cfg, "ai_base_url", ""))
    if base_url:
        http_kwargs["base_url"] = base_url
        http_kwargs["base_url_resource_scope"] = types.ResourceScope.COLLECTION

    return {
        "provider": "gemini",
        "client": genai.Client(
            api_key=cfg.ai_api_key,
            http_options=types.HttpOptions(**http_kwargs),
        ),
    }


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
    if code in {408, 429, 500, 502, 503, 504}:
        return True

    msg = str(exc).lower()
    keywords = (
        "timeout",
        "temporarily unavailable",
        "deadline",
        "connection reset",
        "connection aborted",
        "service unavailable",
    )
    return any(k in msg for k in keywords)


def _extract_json_text(text: str) -> str:
    """Try to recover a JSON array/object from a model response."""
    text = (text or "").strip()
    if not text:
        return text
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    def _balanced_slice(s: str, open_ch: str, close_ch: str) -> str:
        start = s.find(open_ch)
        if start == -1:
            return ""
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(s)):
            ch = s[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
        return ""

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        sliced = _balanced_slice(text, open_ch, close_ch)
        if sliced:
            return sliced

    return text


def _extract_openai_content(message_obj: Any) -> str:
    """Extract assistant text from an OpenAI-compatible message object."""
    if message_obj is None:
        return ""
    if isinstance(message_obj, str):
        return message_obj
    if isinstance(message_obj, dict):
        return (message_obj.get("content") or "").strip()
    return (getattr(message_obj, "content", "") or "").strip()


def _gateway_generate_json(client: dict, contents, config):
    base_url = client["base_url"]
    api_key = client["api_key"]
    model = client["model"]

    if not api_key:
        raise RuntimeError("GATEWAY_API_KEY tidak boleh kosong saat ai_provider=gateway.")
    if not model:
        raise RuntimeError("GATEWAY_MODEL / --ai-model tidak boleh kosong saat ai_provider=gateway.")

    # Prefer OpenAI SDK; it is the standard client for OpenAI-compatible gateways.
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "Package 'openai' belum terpasang. Install/upgrade dengan pip install -U openai."
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    timeout_seconds = float(getattr(config, "ai_timeout_seconds", 1800) or 1800)
    max_tokens = getattr(config, "ai_max_tokens", None)
    if max_tokens is None:
        max_tokens = 8192

    payload_messages = [
        {
            "role": "system",
            "content": "You are an Art Director, Video Editor, and Metadata Strategist for short-form content. Return ONLY a valid JSON array. Do not wrap in markdown or code fences. Every clip object MUST include rank, hook_start_time, hook_end_time, start_time, end_time, typography_plan, broll_list, alasan, bgm_mood, title_indonesia, title_inggris, hastag, description_hook, description_context, keyword_tags, tiktok_title_id, tiktok_caption_id, and tiktok_caption. The time fields are mandatory and must be numbers.",
        },
        {"role": "user", "content": contents},
    ]

    client_openai = OpenAI(**client_kwargs)
    response = client_openai.chat.completions.create(
        model=model,
        messages=payload_messages,
        temperature=0.2,
        stream=False,
        max_tokens=max_tokens,
        top_p=1,
    )

    text = _extract_openai_content(response.choices[0].message)
    if not text:
        raise RuntimeError("Gateway mengembalikan text kosong dari OpenAI SDK.")

    return json.loads(_extract_json_text(text))


def generate_json_with_retry(client, model, contents, config):
    """Generate JSON from either Gemini or gateway with retry logic."""
    MAX_ATTEMPTS = 10
    INITIAL_WAIT_SECONDS = 60
    WAIT_INCREMENT_SECONDS = 30

    provider = client.get("provider", "gemini")
    if provider == "gateway":
        client = {**client, "model": model}
    else:
        client["model"] = model

    last_exc = None
    status_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[AI] Attempt {attempt}/{MAX_ATTEMPTS}...")
            if provider == "gateway":
                return _gateway_generate_json(client, contents, config)

            response = client["client"].models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("AI provider mengembalikan response.text kosong.")
            return json.loads(_extract_json_text(text))
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
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gagal memanggil AI setelah {MAX_ATTEMPTS} percobaan. Error terakhir: {last_exc}"
    ) from last_exc
