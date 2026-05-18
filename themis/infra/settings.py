import os
from dotenv import load_dotenv

load_dotenv()

# ── Validation ─────────────────────────────────────────────────────────────────
# Fail fast at startup if any required variable is missing.

_REQUIRED = [
    "MONGO_URI",
    "OPENAI_API_KEY",
    "JWT_SECRET",
    "EMBEDDING_MODEL",
    "QUERY_MODEL",
    "JUDGE_MODEL",
    "VECTOR_INDEX",
    "DB_NAME",
    "COLLECTION_NAME",
    "CANDIDATES",
    "VECTOR_SCORE_THRESHOLD",
]

_missing = [key for key in _REQUIRED if not os.getenv(key)]
if _missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(_missing)}")

# ── Database ───────────────────────────────────────────────────────────────────

MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ["DB_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

# ── Auth ───────────────────────────────────────────────────────────────────────

JWT_SECRET = os.environ["JWT_SECRET"]

# ── LLM provider ───────────────────────────────────────────────────────────────
# PROVIDER selects which LLM backend is used: "openai", "groq", or "gemini".

PROVIDER = os.getenv("PROVIDER", "openai")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
QUERY_MODEL = os.environ["QUERY_MODEL"]
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_QUERY_MODEL = os.getenv("GROQ_QUERY_MODEL", "llama-3.1-8b-instant")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")

# ── Embedding provider ─────────────────────────────────────────────────────────
# EMBEDDING_MODEL is used by OpenAI; Gemini model is controlled separately.

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# ── Vector search ──────────────────────────────────────────────────────────────
# CANDIDATES: number of precedents returned to the judge after retrieval.
# ANN_CANDIDATES: internal MongoDB ANN pool size — larger pool = better recall.
# VECTOR_SCORE_THRESHOLD: minimum cosine similarity to keep a candidate.

VECTOR_INDEX = os.environ["VECTOR_INDEX"]
CANDIDATES = int(os.environ["CANDIDATES"])
VECTOR_SCORE_THRESHOLD = float(os.environ["VECTOR_SCORE_THRESHOLD"])
ANN_CANDIDATES = CANDIDATES * 4

# ── Judge ──────────────────────────────────────────────────────────────────────

USE_JSON_JUDGE = os.getenv("USE_JSON_JUDGE", "false").lower() == "true"
JUDGE_PROMPT = os.getenv("JUDGE_PROMPT", "judge-v2")
QUERY_PROMPT = os.getenv("QUERY_PROMPT", "query-extraction")

# ── Gemini / AI Studio ─────────────────────────────────────────────────────────
# Optional — only required when PROVIDER=gemini or EMBEDDING_PROVIDER=gemini.

AI_STUDIO_API_KEY = os.getenv("AI_STUDIO_API_KEY")
GEMINI_QUERY_MODEL = os.getenv("GEMINI_QUERY_MODEL", "gemini-2.0-flash")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
GEMINI_CASE_MODEL = os.getenv("GEMINI_CASE_MODEL", "gemini-2.5-flash")

# ── Supabase Storage ──────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
