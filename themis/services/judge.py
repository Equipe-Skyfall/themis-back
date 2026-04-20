import logging
from langfuse import observe
from pydantic import BaseModel

from themis.errors.exceptions import JudgeParseError
from themis.interfaces.prompts import PromptProvider
from themis.interfaces.providers import ChatProvider
from themis.models.domain import USELESS, Judgment, RankedPrecedent, RetrievedPrecedent


class _JudgmentEntry(BaseModel):
    index: int
    score: int
    explanation: str


class _JudgeResponse(BaseModel):
    judgments: list[_JudgmentEntry]


_JUDGE_SCHEMA = _JudgeResponse

DEFAULT_FIELD_LIMITS: dict[str, int] = {
    "tese": 1000,
    "questao": 400,
    "textoEmenta": 1000,
}

logger = logging.getLogger(__name__)


def _format_precedent(
    position: int,
    precedent: RetrievedPrecedent,
    field_limits: dict[str, int] = DEFAULT_FIELD_LIMITS,
) -> str:
    def truncate(text: str, limit: int) -> str:
        return text[:limit] + "…" if len(text) > limit else text

    named_fields = [
        ("tese", precedent.tese),
        ("questao", precedent.questao),
        ("textoEmenta", precedent.textoEmenta),
    ]
    fields = [
        truncate(value, field_limits.get(name, 1000))
        for name, value in named_fields
        if value and value.strip().lower() not in USELESS
    ]
    return (
        f"[{position + 1}] ID: {precedent.id}\n"
        f"Tipo: {precedent.tipo} | Órgão: {precedent.orgao}\n"
        + "\n".join(fields)
    )


def _build_candidates_text(
    precedents: list[RetrievedPrecedent],
    field_limits: dict[str, int] = DEFAULT_FIELD_LIMITS,
) -> str:
    return "\n\n".join(
        _format_precedent(position, p, field_limits)
        for position, p in enumerate(precedents)
    )


def _parse_text_response(
    llm_response: str,
    use_score: bool,
    provider_name: str,
    candidate_count: int,
) -> dict[int, Judgment]:
    """Parse a line-based judge response into judgments keyed by 1-based index."""
    parse_line = Judgment.from_score_line if use_score else Judgment.from_label_line

    judgments: dict[int, Judgment] = {}
    for line in llm_response.strip().splitlines():
        parsed = parse_line(line)
        if parsed is not None:
            index, judgment = parsed
            judgments[index] = judgment

    if not judgments:
        raise JudgeParseError(
            provider=provider_name,
            candidate_count=candidate_count,
            raw_response=llm_response,
        )

    unparsed = candidate_count - len(judgments)
    if unparsed > 0:
        logger.warning(
            "Judge failed to parse %d/%d candidates (provider: %s).",
            unparsed, candidate_count, provider_name,
        )

    return judgments


def _parse_json_response(data: dict, candidate_count: int) -> dict[int, Judgment]:
    """
    Parse a structured JSON judge response into judgments keyed by 1-based index.

    Expected format: {"judgments": [{"index": 1, "score": 8, "explanation": "..."}]}
    """
    judgments: dict[int, Judgment] = {}
    if isinstance(data, list):
        entries = data
    else:
        entries = next((v for v in data.values() if isinstance(v, list)), [])
    for entry in entries:
        if isinstance(entry, str):
            parsed = Judgment.from_score_line(entry)
            if parsed is not None:
                index, judgment = parsed
                judgments[index] = judgment
            continue
        # Handle {"N": "score | explanation"} format returned by Gemini
        if len(entry) == 1:
            key, value = next(iter(entry.items()))
            parsed = Judgment.from_score_line(f"{key}: {value}")
            if parsed is not None:
                index, judgment = parsed
                judgments[index] = judgment
            continue
        index = entry.get("index") or entry.get("id") or entry.get("numero") or entry.get("número")
        score = entry.get("score")
        explanation = entry.get("explanation") or entry.get("explicacao") or entry.get("explicação")
        if index is None or score is None:
            continue
        score = max(0, min(10, int(score)))
        judgments[index] = Judgment(
            label=Judgment._score_to_label(score),
            score=score,
            explanation=explanation,
        )

    unparsed = candidate_count - len(judgments)
    if unparsed > 0:
        logger.warning("Judge JSON missing %d/%d candidates.", unparsed, candidate_count)

    return judgments


def _rank_candidates(
    candidates: list[RetrievedPrecedent],
    judgments: dict[int, Judgment],
) -> list[RankedPrecedent]:
    """Merge judgments with candidates and sort by relevance score descending."""
    default = Judgment(label="nao aplicavel", score=0, explanation=None)
    ranked = [
        RankedPrecedent.from_retrieved(candidate, judgments.get(position, default))
        for position, candidate in enumerate(candidates, start=1)
    ]
    return sorted(ranked, key=lambda p: p.relevance_score, reverse=True)


@observe(name="judge_and_rank")
def judge_and_rank(
    petition_text: str,
    candidates: list[RetrievedPrecedent],
    provider: ChatProvider,
    prompt_provider: PromptProvider,
    *,
    use_score: bool = False,
    use_json: bool = False,
    field_limits: dict[str, int] = DEFAULT_FIELD_LIMITS,
    judge_prompt_name: str | None = None,
    temperature: float = 1.0,
    petition_text_limit: int | None = None,
) -> list[RankedPrecedent]:
    """
    Classify and rank candidate precedents using an LLM judge.

    use_score=False (default): 3-tier label (aplicavel / possivelmente / nao aplicavel).
    use_score=True:            numeric score 0–10 per candidate, finer-grained ranking.
    use_json=True:             use structured JSON output — eliminates parsing failures.
    petition_text_limit:       truncate petition text to this many chars before sending to judge.
    """
    if not candidates:
        return []

    prompt_petition = petition_text[:petition_text_limit] if petition_text_limit else petition_text
    prompt_name = judge_prompt_name or ("judge-v2" if use_score else "judge")
    prompt_content = prompt_provider.compile(
        prompt_name,
        petition_text=prompt_petition,
        candidates_text=_build_candidates_text(candidates, field_limits),
    )
    messages = [{"role": "user", "content": prompt_content}]

    if use_json:
        data = provider.complete_json(messages, temperature=temperature, response_schema=_JUDGE_SCHEMA)
        logger.info("Judge raw response (json): %s", data)
        judgments = _parse_json_response(data, len(candidates))
    else:
        llm_response = provider.complete(messages, temperature=temperature)
        logger.info("Judge raw response (text): %s", llm_response)
        judgments = _parse_text_response(llm_response, use_score, provider.name, len(candidates))

    return _rank_candidates(candidates, judgments)
