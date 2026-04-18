from dataclasses import dataclass, field
from typing import Literal

RelevanceLabel = Literal["aplicavel", "possivelmente aplicavel", "nao aplicavel"]

LABEL_TO_RELEVANCE: dict[str, int] = {
    "aplicavel": 2,
    "possivelmente aplicavel": 1,
    "nao aplicavel": 0,
}

USELESS = {"não informado", "n/a"}


@dataclass(kw_only=True)
class Precedent:
    id: str
    tipo: str | None = None
    orgao: str | None = None
    situacao: str | None = None
    tese: str | None = None
    questao: str | None = None
    textoEmenta: str | None = None
    textoDecisao: str | None = None


@dataclass(kw_only=True)
class RetrievedPrecedent(Precedent):
    cosine_similarity: float = 0.0


@dataclass(kw_only=True)
class RankedPrecedent(RetrievedPrecedent):
    relevance_label: RelevanceLabel = "nao aplicavel"
    relevance_score: int = 0
    explanation: str | None = None

    @classmethod
    def from_retrieved(cls, precedent: RetrievedPrecedent, judgment: "Judgment") -> "RankedPrecedent":
        return cls(
            id=precedent.id,
            tipo=precedent.tipo,
            orgao=precedent.orgao,
            situacao=precedent.situacao,
            tese=precedent.tese,
            questao=precedent.questao,
            textoEmenta=precedent.textoEmenta,
            textoDecisao=precedent.textoDecisao,
            cosine_similarity=precedent.cosine_similarity,
            relevance_label=judgment.label,
            relevance_score=judgment.score,
            explanation=judgment.explanation,
        )


@dataclass
class Judgment:
    label: RelevanceLabel
    score: int
    explanation: str | None

    @staticmethod
    def _score_to_label(score: int) -> RelevanceLabel:
        if score >= 7:
            return "aplicavel"
        if score >= 4:
            return "possivelmente aplicavel"
        return "nao aplicavel"

    @staticmethod
    def _parse_parts(line: str) -> tuple[int, str, str | None] | None:
        index_and_rest = line.strip().split(":", 1)
        if len(index_and_rest) != 2:
            return None
        try:
            index = int(index_and_rest[0].strip())
        except ValueError:
            return None
        value_and_explanation = index_and_rest[1].strip().split("|", 1)
        value = value_and_explanation[0].strip().lower()
        explanation = value_and_explanation[1].strip() if len(value_and_explanation) > 1 else None
        return index, value, explanation

    @classmethod
    def from_label_line(cls, line: str) -> tuple[int, "Judgment"] | None:
        parts = cls._parse_parts(line)
        if parts is None:
            return None
        index, value, explanation = parts
        if value not in LABEL_TO_RELEVANCE:
            return None
        return index, cls(label=value, score=LABEL_TO_RELEVANCE[value], explanation=explanation)

    @classmethod
    def from_score_line(cls, line: str) -> tuple[int, "Judgment"] | None:
        parts = cls._parse_parts(line)
        if parts is None:
            return None
        index, value, explanation = parts
        try:
            score = int(value)
        except ValueError:
            return None
        if not 0 <= score <= 10:
            return None
        return index, cls(label=cls._score_to_label(score), score=score, explanation=explanation)
