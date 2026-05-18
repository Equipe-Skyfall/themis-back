import asyncio
import logging

from themis.infra.gemini_petition import generate_peticao
from themis.infra.settings import CANDIDATES, JUDGE_PROMPT, QUERY_PROMPT, USE_JSON_JUDGE
from themis.interfaces.embeddings import EmbeddingProvider
from themis.interfaces.prompts import PromptProvider
from themis.interfaces.providers import ChatProvider
from themis.interfaces.repositories import PrecedentRepository
from themis.models.domain import GeneratedPetition, RankedPrecedent
from themis.models.responses import PrecedentResult
from themis.services.judge import judge_and_rank
from themis.services.retrieval import vector_search

logger = logging.getLogger(__name__)


class PetitionGenerator:
    def __init__(
        self,
        query_provider: ChatProvider,
        judge_provider: ChatProvider,
        generation_provider: ChatProvider,
        repository: PrecedentRepository,
        embedding_provider: EmbeddingProvider,
        prompt_provider: PromptProvider,
        petition_repository=None,
    ):
        self._query_provider = query_provider
        self._judge_provider = judge_provider
        self._generation_provider = generation_provider
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._prompt_provider = prompt_provider
        self._petition_repo = petition_repository

    async def generate(
        self,
        case_description: str,
        user_id: str | None = None,
        orgao_filter: str | None = None,
    ) -> GeneratedPetition:
        # 1. Search and judge precedents
        precedents = await asyncio.to_thread(self._search_and_judge, case_description)

        # 2. Filter relevant
        relevant = [
            p for p in precedents
            if p.relevance_label in ("aplicavel", "possivelmente aplicavel")
        ]

        # 3. Apply orgao filter if provided
        if orgao_filter and relevant:
            filtered = [
                p for p in relevant
                if p.orgao and orgao_filter.upper() in p.orgao.upper()
            ]
            if filtered:
                relevant = filtered

        has_strong = any(p.relevance_label == "aplicavel" for p in relevant)
        weak = not has_strong
        if weak:
            logger.warning("No strong precedents found for petition generation.")

        logger.info(
            "Precedent search done: %d found, %d relevant, strong=%s.",
            len(precedents), len(relevant), has_strong,
        )

        # 4. Generate peticao
        petition_text = await asyncio.to_thread(
            self._generate_peticao, case_description, relevant,
        )
        logger.info("Petition generated (%d chars).", len(petition_text))

        result = GeneratedPetition(
            case_description=case_description,
            petition_text=petition_text,
            precedents=relevant,
            weak_precedents=weak,
        )

        # 5. Save
        if self._petition_repo and user_id:
            try:
                self._petition_repo.save(
                    user_id=user_id,
                    case_description=case_description,
                    petition_text=petition_text,
                    precedent_results=[
                        PrecedentResult.from_domain(p).model_dump()
                        for p in relevant
                    ],
                    weak_precedents=weak,
                )
                logger.info("Saved generated petition for user %s.", user_id)
            except Exception:
                logger.exception("Failed to save generated petition.")

        return result

    async def regenerate(
        self,
        case_description: str,
        user_id: str | None = None,
        instructions: str | None = None,
        petition_text: str | None = None,
    ) -> GeneratedPetition:
        if not petition_text:
            raise ValueError("petition_text is required for regeneration.")

        # Search and judge precedents based on the petition text or case description
        search_text = petition_text or case_description
        precedents = await asyncio.to_thread(self._search_and_judge, search_text)

        relevant = [
            p for p in precedents
            if p.relevance_label in ("aplicavel", "possivelmente aplicavel")
        ]

        has_strong = any(p.relevance_label == "aplicavel" for p in relevant)
        weak = not has_strong
        if weak:
            logger.warning("No strong precedents found for petition regeneration.")

        logger.info(
            "Regenerate precedent search: %d found, %d relevant, strong=%s.",
            len(precedents), len(relevant), has_strong,
        )

        # Generate petition with found precedents + instructions
        new_text = await asyncio.to_thread(
            generate_peticao,
            self._generation_provider,
            case_description=case_description,
            precedents=[
                {
                    "tipo": p.tipo, "orgao": p.orgao, "tese": p.tese,
                    "textoEmenta": p.textoEmenta,
                    "relevance_label": p.relevance_label,
                    "explanation": p.explanation,
                }
                for p in relevant
            ],
            instructions=instructions,
        )
        logger.info("Petition regenerated (%d chars).", len(new_text))

        result = GeneratedPetition(
            case_description=case_description,
            petition_text=new_text,
            precedents=relevant,
            weak_precedents=weak,
        )

        # Save
        if self._petition_repo and user_id:
            try:
                self._petition_repo.save(
                    user_id=user_id,
                    case_description=case_description,
                    petition_text=new_text,
                    precedent_results=[
                        PrecedentResult.from_domain(p).model_dump()
                        for p in relevant
                    ],
                    weak_precedents=weak,
                    instructions=instructions,
                )
                logger.info("Saved regenerated petition for user %s.", user_id)
            except Exception:
                logger.exception("Failed to save regenerated petition.")

        return result

    def search_precedents(self, query: str):
        return vector_search(
            query,
            candidates=CANDIDATES,
            query_provider=self._query_provider,
            repository=self._repository,
            embedding_provider=self._embedding_provider,
            prompt_provider=self._prompt_provider,
            query_prompt_name=QUERY_PROMPT,
        )

    def _search_and_judge(self, text: str) -> list[RankedPrecedent]:
        retrieved = vector_search(
            text, candidates=CANDIDATES,
            query_provider=self._query_provider,
            repository=self._repository,
            embedding_provider=self._embedding_provider,
            prompt_provider=self._prompt_provider,
            query_prompt_name=QUERY_PROMPT,
        )
        logger.info("Vector search returned %d candidates.", len(retrieved))

        ranked = judge_and_rank(
            text, retrieved, self._judge_provider,
            prompt_provider=self._prompt_provider,
            use_score=True, use_json=USE_JSON_JUDGE,
            judge_prompt_name=JUDGE_PROMPT,
        )
        logger.info("Judge ranked %d precedents.", len(ranked))
        return ranked

    def _generate_peticao(
        self, case_description: str, precedents: list[RankedPrecedent],
    ) -> str:
        precedent_dicts = [
            {
                "tipo": p.tipo, "orgao": p.orgao, "tese": p.tese,
                "textoEmenta": p.textoEmenta,
                "relevance_label": p.relevance_label,
                "explanation": p.explanation,
            }
            for p in precedents
        ]
        return generate_peticao(
            self._generation_provider,
            case_description=case_description,
            precedents=precedent_dicts,
        )
