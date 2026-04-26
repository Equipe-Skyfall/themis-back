# Module-level singletons — instantiated once at import time and shared across the app.

import gridfs
from langfuse import get_client
from langfuse.openai import OpenAI
from pymongo import MongoClient
from google import genai

from themis.infra.settings import (
    MONGO_URI, OPENAI_API_KEY, DB_NAME, COLLECTION_NAME,
    AI_STUDIO_API_KEY,
)

mongo_client = MongoClient(MONGO_URI)
_db = mongo_client[DB_NAME]
collection = _db[COLLECTION_NAME]
history_collection = _db["user_search_history"]
pdf_bucket = gridfs.GridFS(_db, collection="petition_files")

# langfuse.openai.OpenAI wraps the standard OpenAI client to auto-trace all calls.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse_client = get_client()

# Gemini client is optional — only initialised when AI_STUDIO_API_KEY is set.
gemini_client = genai.Client(api_key=AI_STUDIO_API_KEY) if AI_STUDIO_API_KEY else None
