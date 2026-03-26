from langfuse import observe, get_client

from themis.config import CANDIDATES, PROVIDER, resolve_providers
from themis.models.responses import EvaluationResponse, PetitionResponse, PrecedentResult
from themis.services.pdf_extractor import extract_text_from_bytes
from themis.services.retrieval import vector_search
from themis.services.judge import judge_and_rank
from themis.services.evaluator import compute_evaluation

_langfuse = get_client()


@observe(name="analyze_petition")
def process_petition(
    pdf_bytes: bytes,
    candidates: int = CANDIDATES,
    provider_name: str = PROVIDER,
) -> PetitionResponse:
    query_provider, judge_provider = resolve_providers(provider_name)
    _langfuse.update_current_span(metadata={"provider": provider_name, "candidates": candidates})

    petition_text = extract_text_from_bytes(pdf_bytes)
    retrieved = vector_search(petition_text, candidates=candidates, query_provider=query_provider)
    ranked = judge_and_rank(petition_text, retrieved, judge_provider)

    return PetitionResponse(results=[PrecedentResult.from_doc(doc) for doc in ranked])


@observe(name="evaluate_petition")
def evaluate_petition(
    pdf_bytes: bytes,
    expected_id: str,
    candidates: int = CANDIDATES,
    provider_name: str = PROVIDER,
) -> EvaluationResponse:
    query_provider, judge_provider = resolve_providers(provider_name)
    _langfuse.update_current_span(metadata={"provider": provider_name, "expected_id": expected_id, "candidates": candidates})

    petition_text = extract_text_from_bytes(pdf_bytes)
    retrieved = vector_search(petition_text, candidates=candidates, query_provider=query_provider)
    ranked = judge_and_rank(petition_text, retrieved, judge_provider)

    return compute_evaluation(retrieved, [PrecedentResult.from_doc(doc) for doc in ranked], expected_id)
