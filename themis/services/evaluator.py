from langfuse import get_client

from themis.models.responses import EvaluationResponse, PrecedentResult

_langfuse = get_client()


def compute_evaluation(
    retrieved: list[dict],
    results: list[PrecedentResult],
    expected_id: str,
) -> EvaluationResponse:
    """
    Compute retrieval and pipeline metrics for a single labeled petition.

    Compares the full pipeline output against the one known correct precedent
    and logs all scores to Langfuse for aggregate analysis.
    """
    sorted_by_sim = sorted(retrieved, key=lambda d: d.get("cosine_similarity", 0), reverse=True)

    retrieval_rank: int | None = next(
        (i + 1 for i, d in enumerate(sorted_by_sim) if d["id"] == expected_id),
        None,
    )
    is_retrieved = retrieval_rank is not None

    sim_score: int | None = None
    if is_retrieved:
        match = next(d for d in sorted_by_sim if d["id"] == expected_id)
        sim_score = round(match.get("cosine_similarity", 0) * 100)

    pipeline_rank: int | None = next(
        (i + 1 for i, r in enumerate(results) if r.id == expected_id),
        None,
    )
    classification = next(
        (r.relevance_label for r in results if r.id == expected_id),
        None,
    )

    hit_at_k: dict[int, bool] = {
        k: any(d["id"] == expected_id for d in sorted_by_sim[:k])
        for k in [1, 5, 10, 25]
    }

    rr = (1.0 / pipeline_rank) if pipeline_rank is not None else 0.0

    _log_scores(is_retrieved, retrieval_rank, classification, hit_at_k, rr)

    return EvaluationResponse(
        results=results,
        retrieved=is_retrieved,
        retrieval_rank=retrieval_rank,
        similarity_score=sim_score,
        pipeline_rank=pipeline_rank,
        classification=classification,
        hit_at_k=hit_at_k,
        reciprocal_rank=rr,
    )


def _log_scores(
    is_retrieved: bool,
    retrieval_rank: int | None,
    classification: str | None,
    hit_at_k: dict[int, bool],
    reciprocal_rank: float,
) -> None:
    _langfuse.score_current_trace(name="retrieved", value=1.0 if is_retrieved else 0.0)
    _langfuse.score_current_trace(name="reciprocal_rank", value=reciprocal_rank)
    for k, hit in hit_at_k.items():
        _langfuse.score_current_trace(name=f"hit_at_{k}", value=1.0 if hit else 0.0)
    if retrieval_rank is not None:
        _langfuse.score_current_trace(name="retrieval_rank", value=float(retrieval_rank))
    if classification:
        _langfuse.score_current_trace(
            name="classification",
            value=1.0 if classification == "aplicavel" else 0.5 if classification == "possivelmente aplicavel" else 0.0,
            comment=classification,
        )
