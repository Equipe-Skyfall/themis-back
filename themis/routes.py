from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from themis.auth import require_auth
from themis.config import PROVIDER, PROVIDER_SETS
from themis.models.responses import PetitionResponse
from themis.controller import process_petition

router = APIRouter(prefix="/petition", tags=["petition"])


@router.post("/analyze", response_model=PetitionResponse)
async def analyze_petition(
    file: UploadFile = File(...),
    _: dict = Depends(require_auth),
):
    """
    Analyse a petition PDF and return ranked precedents with relevance classification.

    Accepts a PDF upload and runs the full pipeline:
    - Text extraction from the PDF
    - Vector search to retrieve candidate precedents
    - LLM judge to classify and rank each precedent by applicability

    The LLM provider is configured via the PROVIDER environment variable.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    return process_petition(await file.read(), provider_name=PROVIDER)


# @router.post("/evaluate", response_model=EvaluationResponse)
# async def evaluate_petition_route(
#     file: UploadFile = File(...),
#     expected_id: str = Form(...),
#     strategy: str = Form("label"),
#     _: dict = Depends(require_auth),
# ):
#     """
#     Run the full pipeline and measure how well it surfaces the single known best-match precedent.

#     `expected_id` is the ID of the one precedent known to be the correct answer for this petition.
#     `strategy` controls the judge ranking approach:
#     - "label"  (default) — 3-tier label sorting, uses prompt "judge"
#     - "score"             — numeric 0–10 score sorting, uses prompt "judge-v2"

#     Returns:
#     - **retrieved**: whether vector search found it at all
#     - **retrieval_rank**: its 1-based position among candidates ordered by cosine similarity
#     - **similarity_score**: its cosine similarity as a percentage
#     - **hit_at_k**: `{1, 5, 10, 25}` — whether it appears in each top-k
#     - **pipeline_rank**: position in the judge-ranked results
#     - **classification**: label the judge assigned to it
#     - **reciprocal_rank**: `1 / pipeline_rank` for computing MRR across many petitions
#     """
#     if file.content_type not in ("application/pdf", "application/octet-stream"):
#         raise HTTPException(status_code=400, detail="File must be a PDF.")
#     if strategy not in ("label", "score"):
#         raise HTTPException(status_code=400, detail="strategy must be 'label' or 'score'.")

#     return evaluate_petition(await file.read(), expected_id, provider_name=PROVIDER, strategy=strategy)
