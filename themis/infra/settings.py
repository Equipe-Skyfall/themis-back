import os
from dotenv import load_dotenv

load_dotenv()

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

MONGO_URI = os.environ["MONGO_URI"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JWT_SECRET = os.environ["JWT_SECRET"]

PROVIDER = os.getenv("PROVIDER", "openai")

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
QUERY_MODEL = os.environ["QUERY_MODEL"]
JUDGE_MODEL = os.environ["JUDGE_MODEL"]
GROQ_QUERY_MODEL = os.getenv("GROQ_QUERY_MODEL", "llama-3.1-8b-instant")
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile")

VECTOR_INDEX = os.environ["VECTOR_INDEX"]
DB_NAME = os.environ["DB_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

CANDIDATES = int(os.environ["CANDIDATES"])
VECTOR_SCORE_THRESHOLD = float(os.environ["VECTOR_SCORE_THRESHOLD"])
ANN_CANDIDATES = CANDIDATES * 4
USE_JSON_JUDGE = os.getenv("USE_JSON_JUDGE", "false").lower() == "true"


AI_STUDIO_API_KEY = os.getenv("AI_STUDIO_API_KEY")
GEMINI_QUERY_MODEL = os.getenv("GEMINI_QUERY_MODEL", "gemini-2.0-flash")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")
