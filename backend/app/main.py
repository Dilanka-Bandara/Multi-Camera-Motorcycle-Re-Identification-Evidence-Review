"""Application entry point and built frontend serving."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.routes import routes


class HealthResponse(BaseModel):
    """Report availability without loading expensive inference models."""

    status: str = "ok"
    device: str = "not_initialized"
    models_loaded: list[str] = []


def create_app(static_dir: Path | None = None, data_dir: Path | None = None) -> FastAPI:
    """Create an isolated API instance, optionally serving the Vite build."""
    application = FastAPI(title="MotoReID", version="0.1.0")
    application.include_router(routes(data_dir))
    assets = static_dir or Path(os.environ.get("MOTO_STATIC_DIR", "frontend/dist"))

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report API readiness; inference is introduced in subsequent phases."""
        import torch

        return HealthResponse(device="cuda" if torch.cuda.is_available() else "cpu")

    if (assets / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=assets / "assets"), name="assets")

    @application.get("/", response_model=None, include_in_schema=False)
    def index() -> FileResponse | JSONResponse:
        """Serve the compiled frontend or an honest setup error."""
        if (assets / "index.html").is_file():
            return FileResponse(assets / "index.html")
        return JSONResponse(
            status_code=503,
            content={"detail": "Frontend is not built. Run npm ci and npm run build in frontend/."},
        )

    return application


app = create_app()
