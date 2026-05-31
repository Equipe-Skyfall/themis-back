import asyncio
import logging

from google import genai

from themis.infra.gemini_case import (
    MAX_PAGES_PER_REQUEST,
    analyze_pdf,
    generate_minuta,
    merge_batch_results,
    summarize_petition_pdf,
)
from themis.infra.settings import CANDIDATES, JUDGE_PROMPT, QUERY_PROMPT, USE_JSON_JUDGE
from themis.interfaces.embeddings import EmbeddingProvider
from themis.interfaces.prompts import PromptProvider
from themis.interfaces.providers import ChatProvider
from themis.interfaces.repositories import PrecedentRepository
from themis.models.domain import CaseAnalysisResult, DocumentSegment, RankedPrecedent
from themis.models.responses import PrecedentResult
from themis.services.judge import judge_and_rank
from themis.services.pdf import extract_text, page_count, split_pdf
from themis.services.retrieval import vector_search

logger = logging.getLogger(__name__)


class CaseAnalyzer:
    def __init__(
        self,
        client: genai.Client,
        model: str,
        query_provider: ChatProvider,
        judge_provider: ChatProvider,
        repository: PrecedentRepository,
        embedding_provider: EmbeddingProvider,
        prompt_provider: PromptProvider,
        case_analysis_repository=None,
    ):
        self._client = client
        self._model = model
        self._query_provider = query_provider
        self._judge_provider = judge_provider
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._prompt_provider = prompt_provider
        self._case_repo = case_analysis_repository

    async def analyze(
        self,
        pdf_bytes: bytes,
        user_id: str | None = None,
        filename: str | None = None,
    ) -> CaseAnalysisResult:
        # 1. Segment the case
        total_pages = page_count(pdf_bytes)
        logger.info("Case PDF has %d pages (max per request: %d)", total_pages, MAX_PAGES_PER_REQUEST)

        if total_pages <= MAX_PAGES_PER_REQUEST:
            segmentation = await analyze_pdf(self._client, self._model, pdf_bytes)
        else:
            logger.info("PDF exceeds page limit, splitting into batches.")
            batch_results = await self._collect_batches(pdf_bytes, total_pages, page_offset=0)
            logger.info("Got %d batch results, merging...", len(batch_results))
            segmentation = await merge_batch_results(self._client, self._model, batch_results)

        documents = [
            DocumentSegment(
                type=doc["type"],
                title=doc["title"],
                start_page=doc["start_page"],
                end_page=doc["end_page"],
                summary=doc["summary"],
            )
            for doc in segmentation.get("documents", [])
        ]

        result = CaseAnalysisResult(
            case_summary=segmentation.get("case_summary", ""),
            documents=documents,
            total_pages=total_pages,
        )

        # 2. Find and process petição inicial
        petition_segment = next(
            (d for d in documents if d.type == "peticao_inicial"), None,
        )
        if not petition_segment:
            logger.warning("No petição inicial found in segmentation.")
            return result

        petition_pdf = split_pdf(pdf_bytes, petition_segment.start_page, petition_segment.end_page)
        del pdf_bytes
        petition_text = extract_text(petition_pdf)
        logger.info(
            "Extracted petição inicial (pages %d-%d, %d chars).",
            petition_segment.start_page, petition_segment.end_page, len(petition_text),
        )

        # 3. Summarize petition + search precedents in parallel
        summary_result, precedents = await asyncio.gather(
            self._summarize_petition(petition_pdf),
            asyncio.to_thread(self._search_and_judge, petition_text),
        )

        result.petition_summary = summary_result
        result.precedents = [
            p for p in precedents
            if p.relevance_label in ("aplicavel", "possivelmente aplicavel")
        ]

        has_strong = any(p.relevance_label == "aplicavel" for p in result.precedents)
        result.weak_precedents = not has_strong
        if result.weak_precedents:
            logger.warning("No strong precedents found — only weak or none.")

        logger.info(
            "Precedent search done: %d found, %d relevant, strong=%s.",
            len(precedents), len(result.precedents), has_strong,
        )

        # 4. Generate minuta de sentença
        if result.precedents:
            result.minuta = await self._generate_minuta(result)
            logger.info("Minuta generated (%d chars).", len(result.minuta or ""))

        # 5. Save to database
        if self._case_repo and user_id:
            try:
                self._case_repo.save(
                    user_id=user_id,
                    filename=filename or "unknown.pdf",
                    case_summary=result.case_summary,
                    documents=[
                        {
                            "type": d.type, "title": d.title,
                            "start_page": d.start_page, "end_page": d.end_page,
                            "summary": d.summary,
                        }
                        for d in result.documents
                    ],
                    petition_summary=result.petition_summary,
                    precedent_results=[
                        PrecedentResult.from_domain(p).model_dump()
                        for p in result.precedents
                    ],
                    minuta=result.minuta,
                )
                logger.info("Saved case analysis for user %s.", user_id)
            except Exception:
                logger.exception("Failed to save case analysis for user %s.", user_id)

        return result

    async def _generate_minuta(self, result: CaseAnalysisResult) -> str:
        try:
            precedent_dicts = [
                {
                    "tipo": p.tipo,
                    "orgao": p.orgao,
                    "tese": p.tese,
                    "textoEmenta": p.textoEmenta,
                    "relevance_label": p.relevance_label,
                    "explanation": p.explanation,
                }
                for p in result.precedents
            ]
            document_dicts = [
                {
                    "type": d.type, "title": d.title,
                    "start_page": d.start_page, "end_page": d.end_page,
                    "summary": d.summary,
                }
                for d in result.documents
            ]
            return await generate_minuta(
                self._client, self._model,
                case_summary=result.case_summary,
                petition_summary=result.petition_summary,
                documents=document_dicts,
                precedents=precedent_dicts,
            )
        except Exception:
            logger.exception("Failed to generate minuta.")
            return ""

    async def _summarize_petition(self, petition_pdf: bytes) -> str:
        try:
            return await summarize_petition_pdf(self._client, self._model, petition_pdf)
        except Exception:
            logger.exception("Failed to summarize petition.")
            return ""

    def _search_and_judge(self, petition_text: str) -> list[RankedPrecedent]:
        retrieved = vector_search(
            petition_text,
            candidates=CANDIDATES,
            query_provider=self._query_provider,
            repository=self._repository,
            embedding_provider=self._embedding_provider,
            prompt_provider=self._prompt_provider,
            query_prompt_name=QUERY_PROMPT,
        )
        logger.info("Vector search returned %d candidates.", len(retrieved))

        ranked = judge_and_rank(
            petition_text,
            retrieved,
            self._judge_provider,
            prompt_provider=self._prompt_provider,
            use_score=True,
            use_json=USE_JSON_JUDGE,
            judge_prompt_name=JUDGE_PROMPT,
        )
        logger.info("Judge ranked %d precedents.", len(ranked))
        return ranked

    async def _collect_batches(
        self, pdf_bytes: bytes, total_pages: int, page_offset: int,
    ) -> list[dict]:
        if total_pages <= MAX_PAGES_PER_REQUEST:
            result = await analyze_pdf(self._client, self._model, pdf_bytes)
            if page_offset > 0:
                for doc in result.get("documents", []):
                    doc["start_page"] += page_offset
                    doc["end_page"] += page_offset
            return [result]

        mid = total_pages // 2
        first_half = split_pdf(pdf_bytes, 1, mid)
        first_batches = await self._collect_batches(first_half, mid, page_offset)
        del first_half

        second_half = split_pdf(pdf_bytes, mid + 1, total_pages)
        second_batches = await self._collect_batches(second_half, total_pages - mid, page_offset + mid)
        del second_half

        return first_batches + second_batches

    def extract_segment(self, pdf_bytes: bytes, segment: DocumentSegment) -> bytes:
        return split_pdf(pdf_bytes, segment.start_page, segment.end_page)
