import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langfuse import observe

from themis.infra.settings import VECTOR_SCORE_THRESHOLD
from themis.interfaces.embeddings import EmbeddingProvider
from themis.interfaces.prompts import PromptProvider
from themis.interfaces.providers import ChatProvider
from themis.interfaces.repositories import PrecedentRepository
from themis.models.domain import RetrievedPrecedent

logger = logging.getLogger(__name__)


def _dedup(hits: list[RetrievedPrecedent]) -> list[RetrievedPrecedent]:
    """
    Remove duplicates from a flat list of vector search hits, keeping
    the highest-scoring occurrence of each precedent ID.

    Duplicates arise because vector_search embeds multiple inputs (the full petition
    plus each extracted legal query), and the same document can be returned by
    more than one vector search pass.
    """
    best_by_id: dict[str, RetrievedPrecedent] = {}
    for hit in hits:
        if hit.id not in best_by_id or hit.cosine_similarity > best_by_id[hit.id].cosine_similarity:
            best_by_id[hit.id] = hit
    return sorted(best_by_id.values(), key=lambda hit: hit.cosine_similarity, reverse=True)


@observe(name="extract_legal_queries")
def extract_legal_queries(
    petition_text: str,
    provider: ChatProvider,
    prompt_provider: PromptProvider,
    *,
    prompt_name: str = "query-extraction",
    temperature: float = 0.0,
) -> list[str]:
    """
    Extract 3–5 specific legal questions from the petition using an LLM.

    These queries serve as additional embedding inputs for the vector search step,
    improving recall for precedents whose surface language differs significantly
    from the petition's wording. The full petition embedding captures overall
    context; the targeted queries zero in on individual legal issues.

    temperature=0 ensures deterministic, focused output — we want precise legal
    terminology, not creative paraphrases.
    """
    content = provider.complete(
        messages=[{
            "role": "user",
            "content": prompt_provider.compile(prompt_name, petition_text=petition_text),
        }],
        temperature=temperature,
    )
    queries = [q.strip() for q in content.strip().split("\n") if q.strip()]
    logger.info("Extracted %d legal queries (provider: %s).", len(queries), provider.name)
    return queries


@observe(name="vector_search")
def vector_search(
    petition_text: str,
    candidates: int,
    query_provider: ChatProvider,
    repository: PrecedentRepository,
    embedding_provider: EmbeddingProvider,
    prompt_provider: PromptProvider,
    *,
    score_threshold: float = VECTOR_SCORE_THRESHOLD,
    results_multiplier: int = 4,
    query_prompt_name: str = "query-extraction",
    query_temperature: float = 0.0,
) -> list[RetrievedPrecedent]:
    """
    Retrieve the top-N candidate precedents for a petition using vector search.

    Pipeline:
      1. Extract specific legal queries from the petition via LLM (extract_legal_queries).
      2. Embed the full petition text + all queries in a single batched API call.
      3. Run an ANN vector search per embedding against the Atlas vector index,
         collecting a large pool of candidates (candidates * results_multiplier per query).
      4. Deduplicate hits across all query embeddings, keeping the best score per document.
      5. Filter by minimum cosine similarity threshold and return the top-N.

    The full petition embedding captures overall context; the extracted queries improve
    recall for precedents that match specific legal sub-issues rather than the full text.
    """
    queries = extract_legal_queries(
        petition_text, query_provider, prompt_provider,
        prompt_name=query_prompt_name, temperature=query_temperature,
    )
    embeddings = embedding_provider.embed([petition_text] + queries)

    limit_per_query = candidates * results_multiplier
    with ThreadPoolExecutor(max_workers=len(embeddings)) as pool:
        futures = [pool.submit(repository.search, emb, limit_per_query) for emb in embeddings]
        hits: list[RetrievedPrecedent] = []
        for future in as_completed(futures):
            hits.extend(future.result())

    results = [
        hit for hit in _dedup(hits)
        if hit.cosine_similarity >= score_threshold
    ][:candidates]

    if not results:
        logger.warning("Vector search returned no candidates above threshold %.2f.", score_threshold)

    return results
