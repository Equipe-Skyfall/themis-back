from themis.infra.settings import ANN_CANDIDATES, VECTOR_INDEX
from themis.models.domain import RetrievedPrecedent


class QdrantPrecedentRepository:
    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection = collection_name

    def search(self, vector: list[float], limit: int) -> list[RetrievedPrecedent]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedPrecedent(
                id=r.payload["id"],
                tipo=r.payload.get("tipo"),
                orgao=r.payload.get("orgao"),
                situacao=r.payload.get("situacao"),
                tese=r.payload.get("tese"),
                questao=r.payload.get("questao"),
                textoEmenta=r.payload.get("textoEmenta"),
                cosine_similarity=r.score,
            )
            for r in response.points
        ]


class MongoAtlasPrecedentRepository:
    def __init__(self, collection, vector_index: str = VECTOR_INDEX, ann_candidates: int = ANN_CANDIDATES):
        self._collection = collection
        self._vector_index = vector_index
        self._ann_candidates = ann_candidates

    def search(self, vector: list[float], limit: int) -> list[RetrievedPrecedent]:
        pipeline = [
            {"$vectorSearch": {
                "index": self._vector_index,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": self._ann_candidates,
                "limit": limit,
            }},
            {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
        ]
        return [
            RetrievedPrecedent(
                id=doc["id"],
                tipo=doc.get("tipo"),
                orgao=doc.get("orgao"),
                situacao=doc.get("situacao"),
                tese=doc.get("tese"),
                questao=doc.get("questao"),
                textoEmenta=doc.get("textoEmenta"),
                textoDecisao=doc.get("textoDecisao"),
                cosine_similarity=doc["score"],
            )
            for doc in self._collection.aggregate(pipeline)
        ]
