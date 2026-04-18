import logging
from langfuse import get_client, observe

from themis.errors.exceptions import EmptyPetitionError
from themis.infra.settings import CANDIDATES, USE_JSON_JUDGE
from themis.interfaces.embeddings import EmbeddingProvider
from themis.interfaces.prompts import PromptProvider
from themis.interfaces.providers import ChatProvider
from themis.interfaces.repositories import PrecedentRepository
from themis.models.responses import PetitionResponse, PrecedentResult
from themis.services.judge import judge_and_rank
from themis.services.pdf import extract_text
from themis.services.retrieval import vector_search

logger = logging.getLogger(__name__)
_langfuse = get_client()


class PetitionAnalyzer:
    def __init__(
        self,
        query_provider: ChatProvider,
        judge_provider: ChatProvider,
        repository: PrecedentRepository,
        embedding_provider: EmbeddingProvider,
        prompt_provider: PromptProvider,
    ):
        self._query_provider = query_provider
        self._judge_provider = judge_provider
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._prompts = prompt_provider

    @observe(name="analyze_petition")
    def analyze(self, pdf_bytes: bytes, candidates: int = CANDIDATES) -> PetitionResponse:
        _langfuse.update_current_span(metadata={
            "provider": self._query_provider.name,
            "candidates": candidates,
        })

        petition_text = extract_text(pdf_bytes)
        if not petition_text:
            raise EmptyPetitionError()

        logger.info("Extracted petition text (%d chars). Running vector search.", len(petition_text))

        retrieved = vector_search(
            petition_text,
            candidates=candidates,
            query_provider=self._query_provider,
            repository=self._repository,
            embedding_provider=self._embedding_provider,
            prompt_provider=self._prompts,
        )
        logger.info("Vector search returned %d candidates. Running judge.", len(retrieved))

        ranked = judge_and_rank(
            petition_text, retrieved, self._judge_provider,
            prompt_provider=self._prompts, use_score=True, use_json=USE_JSON_JUDGE,
        )
        logger.info("Judge ranked %d precedents.", len(ranked))

        return PetitionResponse(results=[PrecedentResult.from_domain(precedent) for precedent in ranked])
