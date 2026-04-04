from langfuse import observe
from themis.config import USELESS, langfuse_client
from themis.models.responses import LABEL_TO_RELEVANCE
from themis.services.providers import ChatProvider

# Derives a 3-tier label from a 0–10 numeric score returned by judge-v2.
def _score_to_label(score: int) -> str:
    if score >= 7:
        return "aplicavel"
    if score >= 4:
        return "possivelmente aplicavel"
    return "nao aplicavel"


def _build_candidates_text(docs: list[dict]) -> str:
    FIELD_LIMIT = 400

    def _truncate(v: str) -> str:
        return v[:FIELD_LIMIT] + "…" if len(v) > FIELD_LIMIT else v

    return "\n\n".join(
        f"[{i + 1}] ID: {doc['id']}\n"
        f"Tipo: {doc.get('tipo')} | Órgão: {doc.get('orgao')}\n"
        + "\n".join(
            _truncate(v)
            for v in [doc.get('tese'), doc.get('questao'), doc.get('textoEmenta')]
            if v and v.strip().lower() not in USELESS
        )
        for i, doc in enumerate(docs)
    )


@observe(name="judge_and_rank")
def judge_and_rank(
    petition_text: str,
    docs: list[dict],
    provider: ChatProvider,
    use_score: bool = False,
) -> list[dict]:
    """
    Classify and rank candidate precedents using an LLM judge.

    use_score=False (default): uses prompt "judge" — 3-tier label (aplicavel / possivelmente / nao aplicavel).
    use_score=True:            uses prompt "judge-v2" — numeric score 0–10 per candidate, finer-grained ranking.

    Each returned document gains:
    - 'relevance_label': derived from label or score
    - 'relevance_score': 2 | 1 | 0  (for label mode) or 0–10 (for score mode)
    - 'explanation': one-sentence justification (or None if unparseable)

    Precedents that fail to parse default to "nao aplicavel" / score 0.
    """
    if not docs:
        return []

    candidates_text = _build_candidates_text(docs)
    prompt = langfuse_client.get_prompt("judge-v2" if use_score else "judge")
    raw = provider.complete(
        messages=[{"role": "user", "content": prompt.compile(
            petition_text=petition_text,
            candidates_text=candidates_text,
        )}],
    )

    score_map: dict[int, int] = {}
    label_map: dict[int, str] = {}
    explanation_map: dict[int, str] = {}

    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0].strip())
            rest = parts[1].strip()
            value, explanation = (rest.split("|", 1) + [None])[:2]
            value = value.strip().lower()
            if explanation:
                explanation_map[idx] = explanation.strip()
            if use_score:
                score = int(value)
                if 0 <= score <= 10:
                    score_map[idx] = score
                    label_map[idx] = _score_to_label(score)
            else:
                if value in LABEL_TO_RELEVANCE:
                    label_map[idx] = value
        except ValueError:
            continue

    results = []
    for i, doc in enumerate(docs):
        label = label_map.get(i + 1, "nao aplicavel")
        if use_score:
            relevance_score = score_map.get(i + 1, 0)
        else:
            relevance_score = LABEL_TO_RELEVANCE[label]
        results.append({
            **doc,
            "relevance_label": label,
            "relevance_score": relevance_score,
            "explanation": explanation_map.get(i + 1),
        })

    return sorted(results, key=lambda d: d["relevance_score"], reverse=True)
