from pydantic import BaseModel

from themis.models.domain import RankedPrecedent, RelevanceLabel


class PrecedentResult(BaseModel):
    id: str
    tipo: str | None
    orgao: str | None
    situacao: str | None
    tese: str | None
    questao: str | None
    textoEmenta: str | None
    textoDecisao: str | None
    relevance_label: RelevanceLabel
    explanation: str | None
    similarity_score: int

    @classmethod
    def from_domain(cls, precedent: RankedPrecedent) -> "PrecedentResult":
        return cls(
            id=precedent.id,
            tipo=precedent.tipo,
            orgao=precedent.orgao,
            situacao=precedent.situacao,
            tese=precedent.tese,
            questao=precedent.questao,
            textoEmenta=precedent.textoEmenta,
            textoDecisao=precedent.textoDecisao,
            relevance_label=precedent.relevance_label,
            explanation=precedent.explanation,
            similarity_score=round(precedent.cosine_similarity * 100),
        )


class PetitionResponse(BaseModel):
    results: list[PrecedentResult]
