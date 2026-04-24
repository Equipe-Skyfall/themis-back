from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from themis.api.auth import require_auth
from themis.api.dependencies import get_analyzer, get_history_repo
from themis.infra.repositories import HistoryRepository
from themis.models.responses import HistoryEntry, HistoryListResponse, PetitionResponse
from themis.use_cases.analyze import PetitionAnalyzer

router = APIRouter(prefix="/petition", tags=["petition"])


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


@router.get("/history", response_model=HistoryListResponse)
async def get_history_route(
    token: dict = Depends(require_auth),
    history_repo: HistoryRepository = Depends(get_history_repo),
):
    """Return all past analyses for the authenticated user, newest first."""
    docs = history_repo.find_by_user(token.get("userId"))
    return HistoryListResponse(history=[HistoryEntry.from_document(d) for d in docs])

