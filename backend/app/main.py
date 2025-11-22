from fastapi import FastAPI

from .core.init_db import init_db
from .api.routes_datasets import router as datasets_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RFP Analyst – Data Intelligence Agent",
        version="1.0.0"
    )

    @app.on_event("startup")
    def on_startup():
        init_db()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # Include datasets API routes
    app.include_router(datasets_router)

    return app


app = create_app()
