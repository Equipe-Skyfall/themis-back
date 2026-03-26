import os
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI  # drop-in wrapper — auto-traces all LLM calls
from pymongo import MongoClient

from themis.services.providers import GroqChat, OpenAIChat

load_dotenv()

# --- External service credentials ---
MONGO_URI = os.environ["MONGO_URI"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # optional — only needed when PROVIDER=groq
JWT_SECRET = os.environ["JWT_SECRET"]

# --- Active LLM provider ---
# Must match a key in PROVIDER_SETS ("openai" or "groq")
PROVIDER = os.getenv("PROVIDER", "openai")

# --- Model configuration ---
# Embedding model must match the model used when the Atlas vector index was built.
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# OpenAI models
QUERY_MODEL = os.environ["QUERY_MODEL"]
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

# Groq models (only used when GROQ_API_KEY is set)
GROQ_QUERY_MODEL = os.getenv("GROQ_QUERY_MODEL", "llama-3.1-8b-instant")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

# --- Atlas configuration ---
VECTOR_INDEX = os.environ["VECTOR_INDEX"]
DB_NAME = os.environ["DB_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

# Higher CANDIDATES improves recall at the cost of more LLM judge tokens.
CANDIDATES = int(os.environ["CANDIDATES"])

# Hits below this threshold are too semantically distant to be useful candidates.
VECTOR_SCORE_THRESHOLD = float(os.environ["VECTOR_SCORE_THRESHOLD"])

# --- Clients (module-level singletons) ---
mongo_client = MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# Embeddings always use OpenAI — the Atlas index was built with text-embedding-3-large.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Langfuse singleton — reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST from env
langfuse_client = get_client()

# --- LLM provider registry ---
# Each entry is (query_provider, judge_provider) for that vendor.
PROVIDER_SETS: dict[str, tuple] = {
    "openai": (
        OpenAIChat(openai_client, QUERY_MODEL),
        OpenAIChat(openai_client, JUDGE_MODEL),
    ),
}

if GROQ_API_KEY:
    _groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    PROVIDER_SETS["groq"] = (
        GroqChat(_groq_client, GROQ_QUERY_MODEL),
        GroqChat(_groq_client, GROQ_JUDGE_MODEL),
    )

if PROVIDER not in PROVIDER_SETS:
    available = ", ".join(PROVIDER_SETS)
    raise RuntimeError(f"PROVIDER='{PROVIDER}' is not available. Choose one of: {available}")


def resolve_providers(provider_name: str) -> tuple:
    if provider_name not in PROVIDER_SETS:
        available = ", ".join(PROVIDER_SETS)
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
    return PROVIDER_SETS[provider_name]


# --- Content filtering ---
# Placeholder values found in the database that carry no legal meaning.
USELESS = {
    "não informado",
    "n/a"
}
