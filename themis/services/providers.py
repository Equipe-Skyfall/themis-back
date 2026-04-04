from typing import Protocol
from langfuse.openai import OpenAI


class ChatProvider(Protocol):
    """Strategy interface for LLM chat completions."""
    name: str

    def complete(self, messages: list[dict], temperature: float = 1.0) -> str: ...


class _OpenAICompatibleChat:
    """
    Base implementation for any OpenAI-compatible chat completions endpoint.
    Groq, OpenAI, and any other compatible provider share this implementation —
    they differ only in the client base_url and the model name passed at construction.
    """

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def complete(self, messages: list[dict], temperature: float = 1.0) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


class OpenAIChat(_OpenAICompatibleChat):
    name = "openai"


class GroqChat(_OpenAICompatibleChat):
    name = "groq"
