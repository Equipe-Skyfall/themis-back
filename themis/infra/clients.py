# Module-level singletons — instantiated once at import time and shared across the app.

import gridfs
from langfuse import get_client
from langfuse.openai import OpenAI
from pymongo import MongoClient
from google import genai

from supabase import create_client

from themis.infra.settings import (
    MONGO_URI, OPENAI_API_KEY, DB_NAME, COLLECTION_NAME,
    AI_STUDIO_API_KEY, SUPABASE_URL, SUPABASE_KEY,
)

mongo_client = MongoClient(MONGO_URI)
_db = mongo_client[DB_NAME]
collection = _db[COLLECTION_NAME]
history_collection = _db["user_search_history"]
case_analysis_collection = _db["case_analysis_history"]
generated_petitions_collection = _db["generated_petitions"]
pdf_bucket = gridfs.GridFS(_db, collection="petition_files")

# langfuse.openai.OpenAI wraps the standard OpenAI client to auto-trace all calls.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse_client = get_client()

# Gemini client is optional — only initialised when AI_STUDIO_API_KEY is set.
gemini_client = genai.Client(api_key=AI_STUDIO_API_KEY) if AI_STUDIO_API_KEY else None

# Supabase client — optional, used for storing case PDFs.
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
