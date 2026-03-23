from themis.config import openai_client, USELESS, JUDGE_MODEL
from themis.models.responses import LABEL_TO_RELEVANCE



def judge_and_rank(petition_text: str, docs: list[dict]) -> list[dict]:
    """
    Classify each candidate precedent for relevance to the petition using an LLM judge.

    The model receives the full petition text alongside all candidate precedents and
    classifies each one as directly applicable, possibly applicable, or not applicable.
    Results are sorted by relevance score descending so the most useful precedents appear first.

    Each document in the returned list gains two new fields:
    - 'relevance_label': "aplicavel" | "possivelmente aplicavel" | "nao aplicavel"
    - 'relevance_score': 2 | 1 | 0
    - 'explanation': one-sentence justification from the model (or None if unparseable)

    Precedents that the model fails to classify default to "nao aplicavel" / 0.
    """
    if not docs:
        return []

    # Each field is capped at 400 chars to prevent long ADI ementas from dominating
    # the model's attention (length bias). Short súmulas and long ADIs get equal weight.
    FIELD_LIMIT = 400

    def _truncate(v: str) -> str:
        return v[:FIELD_LIMIT] + "…" if len(v) > FIELD_LIMIT else v

    # Build a numbered list of precedents for the prompt.
    # We send tese, questao, and textoEmenta — whichever are present and not placeholder
    # values (filtered via USELESS).
    candidates_text = "\n\n".join(
        f"[{i + 1}] ID: {doc['id']}\n"
        f"Tipo: {doc.get('tipo')} | Órgão: {doc.get('orgao')}\n"
        + "\n".join(
            _truncate(v)
            for v in [doc.get('tese'), doc.get('questao'), doc.get('textoEmenta')]
            if v and v.strip().lower() not in USELESS
        )
        for i, doc in enumerate(docs)
    )

    # The prompt instructs the model to act as a senior STF justice, enforcing:
    # - Merit-based classification (not surface keyword overlap)
    # - Civil/criminal domain separation
    # - Consistent scoring for identical teses
    # - Length neutrality (súmulas are not penalised vs. long ementas)
    # Output format: "<number>: <label> | <one-sentence explanation>"
    prompt = f"""Você é um Ministro do Supremo Tribunal Federal analisando a aplicabilidade de precedentes a uma petição.

Compare a tese central da petição com a questão/tese de cada precedente.

Regras obrigatórias:
- Classifique como "aplicavel" APENAS se o precedente responder diretamente ao conflito jurídico de mérito da petição
- Precedentes cíveis/administrativos NÃO se aplicam a casos criminais e vice-versa
- Precedentes com tese idêntica ou quase idêntica DEVEM receber obrigatoriamente a mesma classificação
- Documentos longos não têm mais peso do que súmulas curtas — avalie o mérito jurídico, não o tamanho

Para cada precedente, responda APENAS com o número, a classificação e uma explicação curta no formato:
<número>: <classificação> | <explicação curta>

Classificações permitidas (use exatamente estas):
- aplicavel: o precedente trata diretamente da mesma questão jurídica da petição e pode ser citado como fundamento
- possivelmente aplicavel: o precedente aborda tema relacionado mas não é diretamente sobre o caso concreto
- nao aplicavel: o precedente trata de matéria distinta e não tem utilidade para o caso

Petição:
{petition_text}

Precedentes:
{candidates_text}

Classificações:"""

    response = openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse the model's response line by line.
    # Expected format per line: "<int>: <label> | <explanation>"
    # The pipe separator is optional — if absent, only the label is extracted.
    # Any line that doesn't match the expected structure is silently skipped.
    label_map: dict[int, str] = {}
    explanation_map: dict[int, str] = {}
    for line in response.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0].strip())
            rest = parts[1].strip()
            if "|" in rest:
                label_part, explanation = rest.split("|", 1)
                label = label_part.strip().lower()
                explanation_map[idx] = explanation.strip()
            else:
                label = rest.lower()
            if label in LABEL_TO_RELEVANCE:
                label_map[idx] = label
        except ValueError:
            continue

    # Merge classification results back into the original documents.
    # Docs not found in label_map (parse failure or omitted by model) default to "nao aplicavel".
    results = []
    for i, doc in enumerate(docs):
        label = label_map.get(i + 1, "nao aplicavel")
        results.append({
            **doc,
            "relevance_label": label,
            "relevance_score": LABEL_TO_RELEVANCE[label],
            "explanation": explanation_map.get(i + 1),
        })

    return sorted(results, key=lambda d: d["relevance_score"], reverse=True)
