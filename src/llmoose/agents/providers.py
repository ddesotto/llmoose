"""Small OpenAI-compatible HTTP transport, defaulting to OpenRouter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderError(RuntimeError):
    """An inference request failed; this is not an invalid model action."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str = ""
    finish_reason: str | None = None
    usage: Mapping[str, object] = field(default_factory=dict)
    reasoning: str | None = None
    reasoning_details: tuple[Mapping[str, object], ...] = ()
    request_id: str | None = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class ChatCompletionProvider:
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str = OPENROUTER_BASE_URL
    timeout: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.0
    include_reasoning: bool = False

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("Base URL must be an HTTP(S) API base URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "Base URL must not contain credentials, a query, or a fragment"
            )
        if not self.model.strip():
            raise ValueError("A model ID is required")
        if url.hostname == "openrouter.ai" and not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY before using OpenRouter")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Mapping[str, object] | None = None,
    ) -> Completion:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = self.request_payload(messages, response_format=response_format)
        request = Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with build_opener(_NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                payload = json.load(response)
        except HTTPError as exc:
            hints = {
                401: "check the API key",
                402: "check account credits",
                403: "check model access",
                404: "check the model ID and base URL",
                429: "rate limited; retry later",
            }
            raise ProviderError(
                f"Inference HTTP {exc.code}: {hints.get(exc.code, 'check provider status and request settings')}"
            ) from None
        except (URLError, TimeoutError, OSError):
            raise ProviderError(
                "Inference connection failed or timed out; check the endpoint"
            ) from None
        except (ValueError, UnicodeError):
            raise ProviderError("Inference endpoint returned invalid JSON") from None
        try:
            if payload.get("error"):
                raise ProviderError("Inference endpoint returned an API error")
            choice = payload["choices"][0]
            if choice.get("error") or choice.get("finish_reason") == "error":
                raise ProviderError("Inference provider failed to generate a response")
            content = choice["message"]["content"]
            # A refusal or exhausted reasoning budget may have no text. Let the
            # policy count this as an invalid output, with the finish reason logged.
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise TypeError
            usage = payload.get("usage") or {}
            if not isinstance(usage, dict):
                raise TypeError
            reasoning = choice["message"].get("reasoning")
            details = choice["message"].get("reasoning_details") or []
            if reasoning is not None and not isinstance(reasoning, str):
                raise TypeError
            if not isinstance(details, list) or any(
                not isinstance(d, dict) for d in details
            ):
                raise TypeError
            request_id = payload.get("id")
            if request_id is not None and not isinstance(request_id, str):
                raise TypeError
            return Completion(
                content,
                payload.get("model", self.model),
                choice.get("finish_reason"),
                usage,
                reasoning,
                tuple(details),
                request_id,
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError(
                "Inference response has no valid chat completion"
            ) from None

    def request_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """The exact JSON body sent by this transport, without authorization headers."""
        body: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.include_reasoning:
            body["reasoning"] = {"enabled": True, "exclude": False}
        if response_format is not None:
            body["response_format"] = dict(response_format)
            if urlsplit(self.base_url).hostname == "openrouter.ai":
                body["provider"] = {"require_parameters": True}
        return body
