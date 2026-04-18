from pathlib import Path

from themis.infra.clients import langfuse_client

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class LangfusePromptProvider:
    """Fetches and compiles prompt templates from Langfuse (production use)."""

    def compile(self, name: str, **variables: str) -> str:
        prompt = langfuse_client.get_prompt(name)
        return prompt.compile(**variables)


class LocalPromptProvider:
    """
    Reads and compiles prompt templates from local files (harness / evaluation use).

    Templates use Langfuse-style double-brace placeholders: {{variable_name}}.
    """

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR):
        self._dir = prompts_dir

    def compile(self, name: str, **variables: str) -> str:
        path = self._dir / f"{name}.txt"
        template = path.read_text(encoding="utf-8")
        for key, value in variables.items():
            template = template.replace("{{" + key + "}}", value)
        return template
