import logging
from fastapi import FastAPI

from themis.api.health import router as health_router
from themis.api.routes import router as petition_router
from themis.errors import handlers

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    app = FastAPI(title="Themis", description="Brazilian legal precedent retrieval API")
    app.include_router(health_router)
    app.include_router(petition_router)
    handlers.register(app)
    return app


app = create_app()
