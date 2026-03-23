from fastapi import APIRouter, File, UploadFile, HTTPException
from themis.models.responses import PetitionResponse
from themis.controller import process_petition

router = APIRouter(prefix="/petition", tags=["petition"])


@router.post("/analyze", response_model=PetitionResponse)
async def analyze_petition(
    file: UploadFile = File(...),
):
    """
    Analyse a petition PDF and return ranked precedents with relevance classification.

    Accepts a PDF upload and runs the full pipeline:
    - Text extraction from the PDF
    - Hybrid search (vector + BM25) to retrieve candidate precedents
    - LLM judge to classify and rank each precedent by applicability

    Returns precedents sorted by relevance score (2 = aplicavel, 1 = possivelmente aplicavel,
    0 = nao aplicavel), each with a one-sentence explanation of why it is or isn't relevant.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    return process_petition(await file.read())
