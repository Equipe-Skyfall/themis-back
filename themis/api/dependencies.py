from functools import lru_cache

from themis.infra.clients import (
    case_analysis_collection, gemini_client,
    generated_petitions_collection, history_collection, pdf_bucket,
)
from themis.infra.embeddings import embedding_provider, repository
from themis.infra.prompts import LangfusePromptProvider
from themis.infra.providers import get_generation_provider, get_summary_provider, resolve_providers
from themis.infra.repositories import CaseAnalysisRepository, GeneratedPetitionRepository, HistoryRepository
from themis.infra.settings import GEMINI_CASE_MODEL, PROVIDER
from themis.use_cases.analyze import PetitionAnalyzer
from themis.use_cases.case_analysis import CaseAnalyzer
from themis.use_cases.petition_generation import PetitionGenerator


@lru_cache
def get_history_repo() -> HistoryRepository:
    return HistoryRepository(history_collection, pdf_bucket)


@lru_cache
def get_case_analysis_repo() -> CaseAnalysisRepository:
    return CaseAnalysisRepository(case_analysis_collection)


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
        summary_provider=get_summary_provider(),
    )


@lru_cache
def get_case_analyzer() -> CaseAnalyzer:
    if not gemini_client:
        raise RuntimeError("AI_STUDIO_API_KEY is required for case analysis.")
    query_provider, judge_provider = resolve_providers(PROVIDER)
    return CaseAnalyzer(
        client=gemini_client,
        model=GEMINI_CASE_MODEL,
        query_provider=query_provider,
        judge_provider=judge_provider,
        repository=repository,
        embedding_provider=embedding_provider,
        prompt_provider=LangfusePromptProvider(),
        case_analysis_repository=get_case_analysis_repo(),
    )


@lru_cache
def get_generated_petition_repo() -> GeneratedPetitionRepository:
    return GeneratedPetitionRepository(generated_petitions_collection)


@lru_cache
def get_petition_generator() -> PetitionGenerator:
    query_provider, judge_provider = resolve_providers(PROVIDER)
    return PetitionGenerator(
        query_provider=query_provider,
        judge_provider=judge_provider,
        generation_provider=get_generation_provider(),
        repository=repository,
        embedding_provider=embedding_provider,
        prompt_provider=LangfusePromptProvider(),
        petition_repository=get_generated_petition_repo(),
    )
