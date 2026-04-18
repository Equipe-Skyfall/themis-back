from langfuse import get_client
from langfuse.openai import OpenAI
from pymongo import MongoClient
from google import genai
from themis.infra.settings import (
    MONGO_URI, OPENAI_API_KEY, DB_NAME, COLLECTION_NAME,
    AI_STUDIO_API_KEY,
)

mongo_client = MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse_client = get_client()

if AI_STUDIO_API_KEY:

    gemini_client = genai.Client(api_key=AI_STUDIO_API_KEY)
else:
    gemini_client = None
