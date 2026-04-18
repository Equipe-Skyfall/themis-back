from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from themis.api.auth import require_auth
from themis.api.dependencies import get_analyzer
from themis.models.responses import PetitionResponse
from themis.use_cases.analyze import PetitionAnalyzer

router = APIRouter(prefix="/petition", tags=["petition"])


@router.post("/analyze", response_model=PetitionResponse)
async def analyze_petition_route(
    file: UploadFile = File(...),
    _: dict = Depends(require_auth),
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

    return analyzer.analyze(await file.read())

