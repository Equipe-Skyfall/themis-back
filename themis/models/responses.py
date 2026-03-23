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


class PetitionResponse(BaseModel):
    results: list[PrecedentResult]
