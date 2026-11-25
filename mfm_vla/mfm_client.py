"""OpenAI-compatible Multimodal Foundation Model (MFM) session client."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Dict, Optional

import httpx
import numpy as np
from PIL import Image


DEFAULT_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    os.environ.get("MFM_BASE_URL", "https://api.openai.com/v1"),
)
DEFAULT_MODEL = os.environ.get(
    "MFM_MODEL",
    os.environ.get("OPENAI_MODEL", "gpt-4o"),
)


class MFMClientError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.S)
    candidate = fenced.group(1) if fenced else stripped
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"MFM returned no JSON object: {stripped[:240]!r}")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("MFM response must be a JSON object")
    return value


def _image_data_url(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class MFMClientSession:
    """Stateless OpenAI-compatible chat client with episode-level token accounting."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        proxy: Optional[str] = None,
        timeout_s: float = 300.0,
        max_tokens: int = 1200,
        retries: int = 12,
        parse_retries: int = 2,
        http_client: Optional[httpx.Client] = None,
    ):
        key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("MFM_API_KEY")
        )
        if not key:
            raise RuntimeError("OPENAI_API_KEY (or MFM_API_KEY) is not set")
        self.model = model
        self.agent_id = f"mfm:{model}"
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.retries = retries
        self.parse_retries = parse_retries
        self._owns_http = http_client is None
        if http_client is None:
            proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get(
                "https_proxy"
            )
            http_client = httpx.Client(
                proxy=proxy,
                timeout=httpx.Timeout(timeout_s, connect=30.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
            )
        self._http = http_client
        self.decisions: list[dict] = []
        self._usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _post(self, payload: dict) -> dict:
        delay = 2.0
        response: Optional[httpx.Response] = None
        for attempt in range(self.retries):
            try:
                response = self._http.post(
                    self.base_url + "/chat/completions",
                    json=payload,
                )
            except httpx.HTTPError as exc:
                if attempt < self.retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise MFMClientError(f"MFM API transport error: {exc}") from exc
            if response.status_code == 429 and attempt < self.retries - 1:
                retry_after = response.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else delay)
                delay = min(60.0, delay * 2.0)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise MFMClientError(
                    f"MFM API HTTP {response.status_code}: {response.text[:1000]}"
                ) from exc
            body = response.json()
            if isinstance(body, dict) and body.get("error"):
                raise MFMClientError(f"MFM API error: {body['error']}")
            return body
        assert response is not None
        raise MFMClientError(f"MFM API remained rate limited: {response.text[:1000]}")

    def decide_json(self, prompt: str, rgb: Dict[str, np.ndarray]) -> dict:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for name, image in sorted(rgb.items()):
            content.extend(
                (
                    {"type": "text", "text": f"Camera frame: {name}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                    },
                )
            )
        max_tokens, parse_attempt = self.max_tokens, 0
        while True:
            body = self._post(
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content}],
                }
            )
            try:
                text = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise MFMClientError(
                    f"MFM API response has no assistant content: {str(body)[:1000]}"
                ) from exc
            try:
                decision = _extract_json(text)
                break
            except ValueError:
                self._usage["parse_failures"] = self._usage.get("parse_failures", 0) + 1
                if not (text or "").strip() and max_tokens < 4 * self.max_tokens:
                    max_tokens *= 2
                    self._usage["empty_retries"] = self._usage.get("empty_retries", 0) + 1
                    continue
                if parse_attempt >= self.parse_retries:
                    raise
                parse_attempt += 1
        usage = body.get("usage") or {}
        self._usage["requests"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._usage[key] += int(usage.get(key, 0) or 0)
        self.decisions.append({"decision": decision, "usage": usage})
        return decision

    def usage(self) -> dict:
        return dict(self._usage)

    def close(self) -> None:
        if self._owns_http:
            self._http.close()
