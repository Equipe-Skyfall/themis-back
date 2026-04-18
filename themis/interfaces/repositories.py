from typing import Protocol

from themis.models.domain import RetrievedPrecedent


class PrecedentRepository(Protocol):
    def search(self, vector: list[float], limit: int) -> list[RetrievedPrecedent]: ...
