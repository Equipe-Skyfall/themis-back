from datetime import datetime

from pydantic import BaseModel

from themis.models.domain import CaseAnalysisResult, DocumentSegment, RankedPrecedent, RelevanceLabel


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
    summary: str | None = None


class HistoryEntry(BaseModel):
    id: str
    filename: str
    timestamp: datetime
    summary: str | None
    results: list[PrecedentResult]

    @classmethod
    def from_document(cls, doc: dict) -> "HistoryEntry":
        return cls(
            id=str(doc["_id"]),
            filename=doc["filename"],
            timestamp=doc["timestamp"],
            summary=doc.get("summary"),
            results=doc.get("results", []),
        )


class HistoryListResponse(BaseModel):
    history: list[HistoryEntry]


# ── Case analysis ─────────────────────────────────────────────────────────────


class DocumentSegmentResponse(BaseModel):
    type: str
    title: str
    start_page: int
    end_page: int
    summary: str

    @classmethod
    def from_domain(cls, segment: DocumentSegment) -> "DocumentSegmentResponse":
        return cls(
            type=segment.type,
            title=segment.title,
            start_page=segment.start_page,
            end_page=segment.end_page,
            summary=segment.summary,
        )


class CaseAnalysisResponse(BaseModel):
    case_summary: str
    documents: list[DocumentSegmentResponse]
    total_pages: int
    petition_summary: str | None = None
    precedent_results: list[PrecedentResult] = []
    minuta: str | None = None
    weak_precedents: bool = False

    @classmethod
    def from_domain(cls, result: CaseAnalysisResult) -> "CaseAnalysisResponse":
        return cls(
            case_summary=result.case_summary,
            documents=[DocumentSegmentResponse.from_domain(d) for d in result.documents],
            total_pages=result.total_pages,
            petition_summary=result.petition_summary,
            precedent_results=[PrecedentResult.from_domain(p) for p in result.precedents],
            minuta=result.minuta,
            weak_precedents=result.weak_precedents,
        )
