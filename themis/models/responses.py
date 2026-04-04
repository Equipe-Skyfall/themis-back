from typing import Literal
from pydantic import BaseModel

RelevanceLabel = Literal["aplicavel", "possivelmente aplicavel", "nao aplicavel"]

# Maps each label to a numeric score used for sorting — higher is more relevant.
# Keep in sync with RelevanceLabel above.
LABEL_TO_RELEVANCE: dict[str, int] = {
    "aplicavel": 2,
    "possivelmente aplicavel": 1,
    "nao aplicavel": 0,
}


class PrecedentResult(BaseModel):
    id: str
    tipo: str | None
    orgao: str | None
    tese: str | None
    questao: str | None
    textoEmenta: str | None
    textoDecisao: str | None
    relevance_label: RelevanceLabel
    explanation: str | None
    similarity_score: int | None

    @classmethod
    def from_doc(cls, doc: dict) -> "PrecedentResult":
        return cls(
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


class PetitionResponse(BaseModel):
    results: list[PrecedentResult]


class EvaluationResponse(BaseModel):
    results: list[PrecedentResult]  # full ranked pipeline output

    # Was the correct precedent retrieved by vector search at all?
    retrieved: bool
    # Position (1-based) in retrieval results ordered by cosine similarity. None if not retrieved.
    retrieval_rank: int | None
    # Cosine similarity score of the correct precedent. None if not retrieved.
    similarity_score: int | None

    # Position (1-based) in the final pipeline results. None if not retrieved.
    pipeline_rank: int | None
    # Label the judge assigned to the correct precedent. None if not retrieved.
    classification: RelevanceLabel | None
    # Numeric score the judge assigned (0–10 in score mode, 0/1/2 in label mode). None if not retrieved.
    judge_score: int | None

    # Hit@k: was the correct precedent in the top-k retrieval results?
    hit_at_k: dict[int, bool]

    # 1 / pipeline_rank — useful for computing MRR across many evaluations. 0.0 if not retrieved.
    reciprocal_rank: float
