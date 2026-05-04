from typing import Any, Protocol


class ChatProvider(Protocol):
    """Strategy interface for LLM chat completions."""
    name: str

    def complete(self, messages: list[dict], temperature: float = 1.0) -> str: ...

    def complete_json(self, messages: list[dict], temperature: float = 1.0, response_schema: Any = None) -> Any: ...
