import os

from themis.infra.clients import collection, gemini_client, openai_client
from themis.infra.repositories import MongoAtlasPrecedentRepository, QdrantPrecedentRepository
from themis.infra.settings import EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL


class OpenAIEmbeddingProvider:
    def __init__(self, client, model: str = EMBEDDING_MODEL):
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class GeminiEmbeddingProvider:
    def __init__(self, client, model: str = GEMINI_EMBEDDING_MODEL):
        self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            self._client.models.embed_content(
                model=self._model,
                contents=text,
                config={"task_type": "RETRIEVAL_QUERY"},
            ).embeddings[0].values
            for text in texts
        ]


def resolve_embedding_stack(provider: str):
    if provider == "gemini":
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url=os.environ["QDRANT"], api_key=os.environ["QDRANT_API_KEY"])
        collection_name = os.getenv("QDRANT_COLLECTION", "precedentes_gemini")
        return GeminiEmbeddingProvider(gemini_client), QdrantPrecedentRepository(qdrant, collection_name)
    # default: openai
    return OpenAIEmbeddingProvider(openai_client), MongoAtlasPrecedentRepository(collection)


embedding_provider, repository = resolve_embedding_stack(os.getenv("EMBEDDING_PROVIDER", "openai"))
