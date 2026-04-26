import logging
from langfuse import get_client, observe

from themis.errors.exceptions import EmptyPetitionError
from themis.infra.settings import CANDIDATES, JUDGE_PROMPT, QUERY_PROMPT, USE_JSON_JUDGE
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

_SUMMARY_PROMPT = (
    "Você é um assistente jurídico. Resuma a petição a seguir em 3 a 5 frases, "
    "destacando o tipo de ação, as partes envolvidas e o pedido principal. Retorne somente o resumo, sem nenhum comentário.\n\n"
    "PETIÇÃO:\n{text}"
)
_SUMMARY_MAX_CHARS = 8000


class PetitionAnalyzer:
    def __init__(
        self,
        query_provider: ChatProvider,
        judge_provider: ChatProvider,
        repository: PrecedentRepository,
        embedding_provider: EmbeddingProvider,
        prompt_provider: PromptProvider,
        history_repository=None,
        summary_provider: ChatProvider | None = None,
    ):
        self._query_provider = query_provider
        self._judge_provider = judge_provider
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._prompts = prompt_provider
        self._history_repo = history_repository
        self._summary_provider = summary_provider

    @observe(name="analyze_petition")
    def analyze(
        self,
        pdf_bytes: bytes,
        candidates: int = CANDIDATES,
        user_id: str | None = None,
        filename: str | None = None,
    ) -> PetitionResponse:
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
            query_prompt_name=QUERY_PROMPT,
        )
        logger.info("Vector search returned %d candidates. Running judge.", len(retrieved))

        ranked = judge_and_rank(
            petition_text, retrieved, self._judge_provider,
            prompt_provider=self._prompts, use_score=True, use_json=USE_JSON_JUDGE,
            judge_prompt_name=JUDGE_PROMPT,
        )
        logger.info("Judge ranked %d precedents.", len(ranked))

        try:
            summary = self._summarize(petition_text)
        except Exception:
            logger.exception("Failed to generate petition summary.")
            summary = ""
        response = PetitionResponse(
            results=[PrecedentResult.from_domain(p) for p in ranked],
            summary=summary or None,
        )

        if self._history_repo and user_id:
            try:
                self._history_repo.save(
                    user_id=user_id,
                    filename=filename or "unknown.pdf",
                    pdf_bytes=pdf_bytes,
                    summary=summary,
                    results=[r.model_dump() for r in response.results],
                )
                logger.info("Saved analysis history for user %s.", user_id)
            except Exception:
                logger.exception("Failed to save analysis history for user %s.", user_id)

        return response

    def _summarize(self, text: str) -> str:
        if not self._summary_provider:
            return ""
        messages = [{"role": "user", "content": _SUMMARY_PROMPT.format(text=text[:_SUMMARY_MAX_CHARS])}]
        return self._summary_provider.complete(messages, temperature=0.3)
