from typing import Protocol

from themis.models.domain import RetrievedPrecedent


class PrecedentRepository(Protocol):
    def search(self, vector: list[float], limit: int) -> list[RetrievedPrecedent]: ...
    def find_by_ids(self, ids: list[str]) -> list[RetrievedPrecedent]: ...
