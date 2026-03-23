import tempfile
from pathlib import Path

from themis.config import CANDIDATES
from themis.models.responses import PetitionResponse, PrecedentResult
from themis.services.pdf_extractor import extract_text
from themis.services.retrieval import vector_search
from themis.services.judge import judge_and_rank


def process_petition(
    pdf_bytes: bytes,
    candidates: int = CANDIDATES,
) -> PetitionResponse:
    """
    Orchestrate the full petition analysis pipeline.

    Steps:
      1. Write the PDF bytes to a temporary file and extract its text.
      2. Run hybrid search to retrieve the top-N candidate precedents.
      3. Pass the petition text and candidates to the LLM judge for classification.
      4. Return the ranked results as a serialisable dict.

    The temporary file is always cleaned up regardless of extraction success or failure.
    """
    # Write PDF bytes to a named temp file — pdfplumber requires a file path, not a buffer
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        petition_text = extract_text(tmp_path)
    finally:
        # Always clean up the temp file, even if extraction raises
        Path(tmp_path).unlink(missing_ok=True)

    retrieved = vector_search(petition_text, candidates=candidates)

    # Debug output — prints each candidate's cosine similarity score
    print(f"\n--- {len(retrieved)} candidates sent to judge ---")
    for d in retrieved:
        print(f"  {d['id']:<30} similarity={d.get('cosine_similarity', 0):.4f}")
    print()

    ranked = judge_and_rank(petition_text, retrieved)

    # Pydantic handles field filtering — internal fields (vector_score, text_score,
    # rrf score, relevance_score) are automatically excluded by the model definition.
    return PetitionResponse(
        results=[
            PrecedentResult(
                id=doc["id"],
                tipo=doc.get("tipo"),
                orgao=doc.get("orgao"),
                tese=doc.get("tese"),
                questao=doc.get("questao"),
                textoEmenta=doc.get("textoEmenta"),
                textoDecisao=doc.get("textoDecisao"),
                relevance_label=doc["relevance_label"],
                explanation=doc.get("explanation"),
                similarity_score=round(doc.get("cosine_similarity", 0) * 100),
            )
            for doc in ranked
        ]
    )
