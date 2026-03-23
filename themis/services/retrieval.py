from themis.config import collection, openai_client, EMBEDDING_MODEL, VECTOR_INDEX, QUERY_MODEL, VECTOR_SCORE_THRESHOLD


def _dedup(hits: list) -> list:
    """
    Remove duplicate documents from a flat list of vector search hits, keeping
    the highest-scoring occurrence of each precedent ID.

    Duplicates arise because vector_search embeds multiple inputs (the full petition
    plus each extracted legal query), and the same document can be returned by
    more than one vector search pass.
    """
    seen: dict = {}
    for document in hits:
        precedent_id = document.get("id", "unknown")
        if precedent_id not in seen or document["score"] > seen[precedent_id]["score"]:
            seen[precedent_id] = document
    return sorted(seen.values(), key=lambda d: d["score"], reverse=True)


def extract_legal_queries(petition_text: str) -> list[str]:
    """
    Extract 3–5 specific legal questions from the petition using an LLM.

    These queries serve as additional embedding inputs for the vector search step,
    improving recall for precedents whose surface language differs significantly
    from the petition's wording. The full petition embedding captures overall
    context; the targeted queries zero in on individual legal issues.

    temperature=0 ensures deterministic, focused output — we want precise legal
    terminology, not creative paraphrases.
    """
    response = openai_client.chat.completions.create(
        model=QUERY_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Você é um especialista em direito brasileiro.

Leia a petição abaixo e extraia de 3 a 5 questões jurídicas substantivas e específicas que precisam de precedentes.

Regras:
- Foque no tema central e concreto do caso (ex: direito específico discutido, relação jurídica em questão)
- Evite questões genéricas como "competência do juízo", "legitimidade das partes" ou "responsabilidade civil"
- Inclua questões procedimentais quando forem específicas e centrais ao caso, usando a terminologia técnica exata (ex: "legitimidade ativa de incapaz representado no JEFP", "competência do Juizado Especial da Fazenda Pública para menor", "incapaz assistido CPC art. 71 Lei 12.153/09")
- Cada questão deve ser específica o suficiente para distinguir este caso de outros
- Use linguagem direta, sem "qual é" ou "como se", prefira afirmações ou termos-chave
- Responda APENAS com as questões, uma por linha, sem numeração ou explicação

Petição:
{petition_text}

Questões jurídicas:""",
        }],
        temperature=0,
    )
    queries = [
        q.strip()
        for q in response.choices[0].message.content.strip().split("\n")
        if q.strip()
    ]
    return queries


def vector_search(
    petition_text: str,
    candidates: int,
) -> list[dict]:
    """
    Retrieve the top-N candidate precedents for a petition using vector search.

    Pipeline:
      1. Extract specific legal queries from the petition via LLM (extract_legal_queries).
      2. Embed the full petition text + all queries in a single batched API call.
      3. Run an ANN vector search per embedding against the Atlas vector index,
         collecting a large pool of candidates (candidates * 4 per query).
      4. Deduplicate hits across all query embeddings, keeping the best score per document.
      5. Filter by minimum cosine similarity threshold and return the top-N.

    The full petition embedding captures overall context; the extracted queries improve
    recall for precedents that match specific legal sub-issues rather than the full text.
    """
    # Step 1 — extract targeted legal queries to augment the petition embedding
    queries = extract_legal_queries(petition_text)
    all_inputs = [petition_text] + queries

    # Step 2 — embed all inputs in a single batched API call to minimise latency
    response = openai_client.embeddings.create(input=all_inputs, model=EMBEDDING_MODEL)
    query_vectors = [r.embedding for r in response.data]

    # Step 3 — run a vector search for each embedding and collect all hits.
    # candidates * 4 per query gives a large enough pool for dedup + threshold filtering.
    vector_hits: list = []
    for vector in query_vectors:
        pipeline = [
            {"$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": 500,   # ANN candidates Atlas considers internally
                "limit": candidates * 4,
            }},
            # Project out the raw embedding vector — it's large and not needed downstream
            {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
        ]
        vector_hits.extend(list(collection.aggregate(pipeline)))

    # Step 4 — deduplicate across query embeddings, filter by minimum cosine similarity,
    # and preserve the raw Atlas cosine similarity before anything else overwrites `score`.
    # This is the only point where the absolute similarity value (0–1) is available.
    results = []
    for d in _dedup(vector_hits):
        if d.get("score", 0) >= VECTOR_SCORE_THRESHOLD:
            d["cosine_similarity"] = d["score"]
            results.append(d)

    return results[:candidates]
