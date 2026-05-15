import asyncio
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from themis.api.auth import require_auth
from themis.api.dependencies import get_analyzer, get_case_analyzer, get_history_repo, get_petition_generator
from themis.infra.repositories import HistoryRepository
from themis.models.responses import (
    CaseAnalysisResponse,
    GeneratedPetitionResponse,
    HistoryEntry,
    HistoryListResponse,
    PetitionResponse,
    RegeneratePetitionRequest,
    RetrievedPrecedentResult,
    SearchPrecedentsRequest,
    SearchPrecedentsResponse,
)
from themis.services.pdf import extract_text
from themis.use_cases.analyze import PetitionAnalyzer
from themis.use_cases.case_analysis import CaseAnalyzer
from themis.use_cases.petition_generation import PetitionGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/petition", tags=["petition"])

# In-memory job store for case analysis
_jobs: dict[str, dict] = {}


@router.post("/analyze", response_model=PetitionResponse)
async def analyze_petition_route(
    file: UploadFile = File(...),
    token: dict = Depends(require_auth),
    analyzer: Annotated[PetitionAnalyzer, Depends(get_analyzer)] = None,
):
    """
    Analyse a petition PDF and return ranked precedents with relevance classification.

    Accepts a PDF upload and runs the full pipeline:
    - Text extraction from the PDF
    - Vector search to retrieve candidate precedents
    - LLM judge to classify and rank each precedent by applicability
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    return analyzer.analyze(
        await file.read(),
        user_id=token.get("userId"),
        filename=file.filename,
    )


@router.post("/analyze-case")
async def analyze_case_route(
    file: UploadFile = File(...),
    token: dict = Depends(require_auth),
    case_analyzer: Annotated[CaseAnalyzer, Depends(get_case_analyzer)] = None,
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    job_id = uuid.uuid4().hex
    pdf_bytes = await file.read()
    user_id = token.get("userId")
    filename = file.filename

    _jobs[job_id] = {"status": "processing"}

    async def _run():
        try:
            result = await case_analyzer.analyze(
                pdf_bytes, user_id=user_id, filename=filename,
            )
            _jobs[job_id] = {
                "status": "done",
                "result": CaseAnalysisResponse.from_domain(result).model_dump(),
            }
        except Exception:
            logger.exception("Case analysis job %s failed.", job_id)
            _jobs[job_id] = {"status": "error", "detail": "Analysis failed."}

    asyncio.create_task(_run())
    return {"job_id": job_id}


@router.get("/case-status/{job_id}")
async def case_status_route(
    job_id: str,
    token: dict = Depends(require_auth),
):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.post("/analyze-case-test")
async def analyze_case_test_route(
    file: UploadFile = File(...),
    token: dict = Depends(require_auth),
):
    job_id = f"test-{uuid.uuid4().hex}"
    _jobs[job_id] = {"status": "processing"}

    async def _mock():
        await asyncio.sleep(15)
        from themis.api.dependencies import get_case_analysis_repo
        case_repo = get_case_analysis_repo()
        doc = case_repo.find_random()
        if not doc:
            _jobs[job_id] = {"status": "error", "detail": "No case analyses found."}
            return
        doc.pop("_id", None)
        doc.pop("userId", None)
        doc.pop("timestamp", None)
        _jobs[job_id] = {"status": "done", "result": doc}

    asyncio.create_task(_mock())
    return {"job_id": job_id}


@router.post("/generate")
async def generate_petition_route(
    case_description: str = Form(...),
    orgao_filter: str | None = Form(None),
    file: UploadFile | None = File(None),
    token: dict = Depends(require_auth),
    generator: Annotated[PetitionGenerator, Depends(get_petition_generator)] = None,
):
    # Build full case description from text + optional PDF
    full_description = case_description
    if file:
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=400, detail="File must be a PDF.")
        pdf_bytes = await file.read()
        pdf_text = extract_text(pdf_bytes)
        if pdf_text:
            full_description = f"{case_description}\n\nDOCUMENTO ANEXADO:\n{pdf_text}"

    job_id = uuid.uuid4().hex
    user_id = token.get("userId")
    _jobs[job_id] = {"status": "processing"}

    async def _run():
        try:
            result = await generator.generate(
                case_description=full_description,
                user_id=user_id,
                orgao_filter=orgao_filter,
            )
            _jobs[job_id] = {
                "status": "done",
                "result": GeneratedPetitionResponse.from_domain(result).model_dump(),
            }
        except Exception:
            logger.exception("Petition generation job %s failed.", job_id)
            _jobs[job_id] = {"status": "error", "detail": "Generation failed."}

    asyncio.create_task(_run())
    return {"job_id": job_id}


@router.post("/search-precedents", response_model=SearchPrecedentsResponse)
async def search_precedents_route(
    body: SearchPrecedentsRequest,
    token: dict = Depends(require_auth),
    generator: Annotated[PetitionGenerator, Depends(get_petition_generator)] = None,
):
    retrieved = await asyncio.to_thread(generator.search_precedents, body.query)
    return SearchPrecedentsResponse(
        results=[RetrievedPrecedentResult.from_domain(p) for p in retrieved],
    )


@router.post("/regenerate")
async def regenerate_petition_route(
    body: RegeneratePetitionRequest,
    token: dict = Depends(require_auth),
    generator: Annotated[PetitionGenerator, Depends(get_petition_generator)] = None,
):
    job_id = uuid.uuid4().hex
    user_id = token.get("userId")
    _jobs[job_id] = {"status": "processing"}

    async def _run():
        try:
            result = await generator.regenerate(
                case_description=body.case_description,
                user_id=user_id,
                instructions=body.instructions,
                petition_text=body.petition_text,
            )
            _jobs[job_id] = {
                "status": "done",
                "result": GeneratedPetitionResponse.from_domain(result).model_dump(),
            }
        except Exception:
            logger.exception("Petition regeneration job %s failed.", job_id)
            _jobs[job_id] = {"status": "error", "detail": "Regeneration failed."}

    asyncio.create_task(_run())
    return {"job_id": job_id}


@router.get("/history", response_model=HistoryListResponse)
async def get_history_route(
    token: dict = Depends(require_auth),
    history_repo: HistoryRepository = Depends(get_history_repo),
):
    """Return all past analyses for the authenticated user, newest first."""
    docs = history_repo.find_by_user(token.get("userId"))
    return HistoryListResponse(history=[HistoryEntry.from_document(d) for d in docs])

