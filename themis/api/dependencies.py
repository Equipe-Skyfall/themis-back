from functools import lru_cache

from themis.infra.clients import gemini_client, history_collection, pdf_bucket
from themis.infra.embeddings import embedding_provider, repository
from themis.infra.prompts import LangfusePromptProvider
from themis.infra.providers import resolve_providers
from themis.infra.repositories import HistoryRepository
from themis.infra.settings import GEMINI_QUERY_MODEL, PROVIDER
from themis.use_cases.analyze import PetitionAnalyzer


@lru_cache
def get_history_repo() -> HistoryRepository:
    return HistoryRepository(history_collection, pdf_bucket)


@lru_cache
def get_analyzer() -> PetitionAnalyzer:
    query_provider, judge_provider = resolve_providers(PROVIDER)
    return PetitionAnalyzer(
        query_provider=query_provider,
        judge_provider=judge_provider,
        repository=repository,
        embedding_provider=embedding_provider,
        prompt_provider=LangfusePromptProvider(),
        history_repository=get_history_repo(),
        gemini_client=gemini_client,
        gemini_model=GEMINI_QUERY_MODEL,
    )
