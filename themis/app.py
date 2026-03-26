from fastapi import FastAPI
from themis.routes import router

app = FastAPI(title="Themis", description="Brazilian legal precedent retrieval API")

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
