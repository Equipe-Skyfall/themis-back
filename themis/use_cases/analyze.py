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
    "Você é um analista jurídico sênior especializado em direito brasileiro. "
    "Analise a petição abaixo e produza um resumo estruturado e objetivo seguindo estas diretrizes:\n\n"
    "1. Identifique o TIPO DE AÇÃO (ex.: Ação Declaratória, Mandado de Segurança, Ação Civil Pública, etc.).\n"
    "2. Descreva de forma genérica as PARTES envolvidas usando apenas seus papéis processuais "
    "(ex.: 'o autor', 'a ré', 'a empresa reclamada') — nunca mencione nomes, CPFs, CNPJs ou qualquer dado pessoal.\n"
    "3. Exponha o FUNDAMENTO JURÍDICO central invocado (dispositivos legais, princípios constitucionais ou teses).\n"
    "4. Descreva o PEDIDO PRINCIPAL e eventuais pedidos acessórios relevantes (tutela de urgência, danos morais, etc.).\n"
    "5. Indique o CONTEXTO FÁTICO essencial que motivou a ação, em no máximo duas frases.\n\n"
    "Regras de formato:\n"
    "- Escreva entre 4 e 6 frases corridas, em um único parágrafo coeso.\n"
    "- Use linguagem técnica porém acessível, sem jargões desnecessários.\n"
    "- Não inclua saudações, comentários, títulos, bullet points ou qualquer texto além do resumo.\n"
    "- Omita informações de terceiros, testemunhas ou dados que não sejam essenciais para compreender o objeto da ação.\n\n"
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
