from typing import Protocol


class PromptProvider(Protocol):
    """Strategy interface for retrieving and compiling prompt templates."""

    def compile(self, name: str, **variables: str) -> str:
        """Return a prompt string with variables substituted."""
        ...
