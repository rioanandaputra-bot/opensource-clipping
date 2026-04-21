"""
clipping.ai_provider — AI provider abstraction for Gemini or OpenAI-compatible gateway.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
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
    if text.startswith("[") or text.startswith("{"):
        return text

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        return text[first_arr : last_arr + 1]

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        return text[first_obj : last_obj + 1]

    return text


def _gateway_generate_json(client: dict, contents, config):
    base_url = client["base_url"]
    api_key = client["api_key"]
    model = client["model"]

    if not base_url:
        raise RuntimeError("GATEWAY_BASE_URL tidak boleh kosong saat ai_provider=gateway.")
    if not api_key:
        raise RuntimeError("GATEWAY_API_KEY tidak boleh kosong saat ai_provider=gateway.")
    if not model:
        raise RuntimeError("GATEWAY_MODEL / --ai-model tidak boleh kosong saat ai_provider=gateway.")

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON generator. Reply with a valid JSON array only, without markdown or explanation.",
            },
            {"role": "user", "content": contents},
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "OpenSource-Clipping/0.9.4",
    }

    with httpx.Client(timeout=15 * 60, follow_redirects=True) as http:
        resp = http.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Gateway response tidak memiliki choices[0].message.content: {data}") from exc

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
