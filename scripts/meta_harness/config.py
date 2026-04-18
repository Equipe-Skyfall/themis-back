"""
Harness configuration — all tunable pipeline parameters in one place.

During Meta-Harness optimization, Claude Code edits this file directly
to test different configurations. Production uses env-based defaults.
"""

from dataclasses import dataclass, field


@dataclass
class HarnessConfig:
    # ── Retrieval ──────────────────────────────────────────────────
    candidates: int = 5
    vector_score_threshold: float = 0.70
    query_extraction_temperature: float = 0.0
    results_per_embedding_multiplier: int = 4  # limit = candidates * this

    # ── Judge ──────────────────────────────────────────────────────
    use_json: bool = True
    use_score: bool = True
    judge_temperature: float = 1.0
    petition_text_limit: int = 2000   # chars of petition sent to the judge
    field_limits: dict[str, int] = field(default_factory=lambda: {
        "tese": 400,
        "questao": 200,
        "textoEmenta": 400,
    })

    # ── Prompt names (mapped to files in themis/prompts/) ─────────
    query_prompt: str = "query-extraction"
    judge_prompt: str = "judge-v2"

    # ── Retrieval mode ─────────────────────────────────────────────
    use_queries: bool = True  # False → search with petition text only (no query extraction)

    # ── Rate limiting ──────────────────────────────────────────────
    delay_between_petitions: float = 60  # seconds to wait between petitions
