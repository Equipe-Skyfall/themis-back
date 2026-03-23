import os
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- External service credentials (required) ---
MONGO_URI = os.environ["MONGO_URI"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# --- Model configuration ---
# Embedding model used to vectorise petitions and precedents.
# Must match the model used when the Atlas vector index was built.
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# LLM used to extract targeted legal queries from a petition for vector search.
# A fast, cheap model is sufficient here — output is used as embedding input, not shown to users.
QUERY_MODEL = os.environ["QUERY_MODEL"]

# LLM used to classify and rank candidate precedents against the petition.
# A more capable model improves classification accuracy at slightly higher cost.
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

# --- Atlas configuration ---
VECTOR_INDEX = os.environ["VECTOR_INDEX"]
DB_NAME = os.environ["DB_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

# Number of candidate precedents retrieved before the judge step.
# Higher values improve recall at the cost of more LLM judge tokens.
CANDIDATES = int(os.environ["CANDIDATES"])

# Minimum cosine similarity score for a vector search hit to be included.
# Hits below this threshold are too semantically distant to be useful candidates.
VECTOR_SCORE_THRESHOLD = float(os.environ["VECTOR_SCORE_THRESHOLD"])

# --- Database and API clients (module-level singletons) ---
mongo_client = MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Content filtering ---
# Placeholder values found in the database that carry no legal meaning.
# Any field whose stripped, lowercased value matches an entry here is excluded
# before being sent to the LLM judge, preventing the model from reasoning over garbage.
USELESS = {
    "não informado",
    "n/a"
}
