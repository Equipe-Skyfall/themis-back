from typing import Literal
from pydantic import BaseModel, Field, computed_field

_LABEL_CONFIDENCE: dict[str, float] = {
    "aplicavel": 1.0,
    "possivelmente aplicavel": 0.5,
    "nao aplicavel": 0.0,
}

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
    similarity_score: int | None = Field(exclude=True)

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

    @computed_field
    @property
    def confidence_score(self) -> int | None:
        """
        0–100 confidence that this precedent is applicable.
        Weighted combination: judge label (70%) + embedding similarity (30%).
        Both signals agreeing raises confidence; divergence lowers it.
        """
        if self.similarity_score is None:
            return None
        label_score = _LABEL_CONFIDENCE[self.relevance_label]
        return round((label_score * 0.7 + (self.similarity_score / 100) * 0.3) * 100)


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

    # Hit@k: was the correct precedent in the top-k retrieval results?
    hit_at_k: dict[int, bool]

    # 1 / pipeline_rank — useful for computing MRR across many evaluations. 0.0 if not retrieved.
    reciprocal_rank: float
