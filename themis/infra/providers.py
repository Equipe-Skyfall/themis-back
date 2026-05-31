import json
import logging
from typing import Any

from langfuse.openai import OpenAI
from openai import APIStatusError, APITimeoutError, BadRequestError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from themis.infra.clients import gemini_client, openai_client
from themis.infra.settings import (
    GEMINI_JUDGE_MODEL,
    GEMINI_QUERY_MODEL,
    GROQ_API_KEY,
    GROQ_GENERATION_MODEL,
    GROQ_JUDGE_MODEL,
    GROQ_QUERY_MODEL,
    JUDGE_MODEL,
    PROVIDER,
    QUERY_MODEL,
)
from themis.interfaces.providers import ChatProvider

logger = logging.getLogger(__name__)

# ── Retry policies ─────────────────────────────────────────────────────────────

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}


def _make_retry(is_retryable, attempts: int, min_wait: int, max_wait: int):
    return retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        before_sleep=lambda state: logger.warning(
            "LLM call failed (%s), retrying (attempt %d)…",
            state.outcome.exception(), state.attempt_number,
        ),
        reraise=True,
    )


def _is_openai_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APITimeoutError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in _RETRYABLE_STATUS_CODES:
        return True
    return False


def _is_gemini_retryable(exc: BaseException) -> bool:
    from google.genai.errors import ClientError, ServerError
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError) and exc.status == "RESOURCE_EXHAUSTED":
        return True
    return False


# Gemini needs more attempts and longer waits due to stricter rate limits
_openai_retry = _make_retry(_is_openai_retryable, attempts=3, min_wait=1, max_wait=30)
_gemini_retry = _make_retry(_is_gemini_retryable, attempts=6, min_wait=5, max_wait=60)


# ── Provider implementations ───────────────────────────────────────────────────

class OpenAICompatibleChat:
    """Wraps any OpenAI-compatible endpoint (OpenAI, Groq, etc.)."""

    def __init__(self, client: OpenAI, model: str, name: str):
        self.name = name
        self._client = client
        self._model = model

    @_openai_retry
    def complete(self, messages: list[dict], temperature: float = 1.0) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    @_openai_retry
    def complete_json(self, messages: list[dict], temperature: float = 1.0, response_schema: Any = None) -> Any:
        if response_schema is not None:
            try:
                response = self._client.beta.chat.completions.parse(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_schema,
                )
                return json.loads(response.choices[0].message.content)
            except BadRequestError:
                logger.info("Model %s does not support json_schema, falling back to json_object.", self._model)
        patched = list(messages)
        if not any("json" in m.get("content", "").lower() for m in patched):
            patched.insert(0, {"role": "system", "content": "Respond in json format."})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=patched,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


class GeminiChat:
    """Wraps the Google AI Studio SDK for Gemini models."""

    name = "gemini"

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    @_gemini_retry
    def complete(self, messages: list[dict], temperature: float = 1.0) -> str:
        content = "\n".join(m["content"] for m in messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=content,
            config={"temperature": temperature},
        )
        return response.text

    @_gemini_retry
    def complete_json(self, messages: list[dict], temperature: float = 1.0, response_schema: Any = None) -> Any:
        content = "\n".join(m["content"] for m in messages)
        config: dict = {"temperature": temperature, "response_mime_type": "application/json"}
        if response_schema is not None:
            config["response_schema"] = response_schema
        response = self._client.models.generate_content(
            model=self._model,
            contents=content,
            config=config,
        )
        obj, _ = json.JSONDecoder().raw_decode(response.text.strip())
        return obj


# ── Registry ───────────────────────────────────────────────────────────────────
# Maps provider name → (query_provider, judge_provider).
# Each entry is only registered if the required credentials are available.

def _build_registry() -> dict[str, tuple[ChatProvider, ChatProvider]]:
    registry: dict[str, tuple[ChatProvider, ChatProvider]] = {
        "openai": (
            OpenAICompatibleChat(openai_client, QUERY_MODEL, name="openai"),
            OpenAICompatibleChat(openai_client, JUDGE_MODEL, name="openai"),
        ),
    }
    if GROQ_API_KEY:
        groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        registry["groq"] = (
            OpenAICompatibleChat(groq_client, GROQ_QUERY_MODEL, name="groq"),
            OpenAICompatibleChat(groq_client, GROQ_JUDGE_MODEL, name="groq"),
        )
    if gemini_client:
        registry["gemini"] = (
            GeminiChat(gemini_client, GEMINI_QUERY_MODEL),
            GeminiChat(gemini_client, GEMINI_JUDGE_MODEL),
        )
    return registry


_REGISTRY = _build_registry()

if PROVIDER not in _REGISTRY:
    raise RuntimeError(f"PROVIDER='{PROVIDER}' is not available. Choose one of: {', '.join(_REGISTRY)}")


def resolve_providers(provider_name: str) -> tuple[ChatProvider, ChatProvider]:
    if provider_name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {', '.join(_REGISTRY)}")
    return _REGISTRY[provider_name]


def get_summary_provider() -> ChatProvider | None:
    if "groq" not in _REGISTRY:
        return None
    return _REGISTRY["groq"][0]


def get_generation_provider() -> ChatProvider:
    if GROQ_API_KEY:
        groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        return OpenAICompatibleChat(groq_client, GROQ_GENERATION_MODEL, name="groq")
    return _REGISTRY[PROVIDER][0]
