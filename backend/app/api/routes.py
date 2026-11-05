"""Run submission, review downloads and restricted Range-capable media routes."""

import json
import os
import re
from pathlib import Path
from shutil import copyfileobj
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.api.jobs import Jobs
from app.config import AppConfig, StrictModel, load_config
from app.schemas import RunResult, RunStatus


class PresetRequest(StrictModel):
    preset: str
    config_overrides: dict[str, Any] = {}


def routes(data: Path | None = None, models: Path | None = None) -> APIRouter:
    router = APIRouter()
    root = data or Path(os.getenv("MOTO_DATA_DIR", "/app/data/user"))
    jobs = Jobs(root, models or Path(os.getenv("MOTO_MODELS_DIR", "/app/models")))
    fixture = root / "fixtures" / "dev"
    if data is None and not fixture.exists():
        fixture = Path(os.environ.get("MOTO_FIXTURE_DIR", "/app/data/fixtures/dev"))

    def directory(run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise HTTPException(404, "Unknown run")
        return root / "runs" / run_id

    def file(path: Path) -> Path:
        if not path.is_file():
            raise HTTPException(404, "File unavailable")
        return path

    def config(preset: str | None, overrides: dict[str, Any]) -> AppConfig:
        try:
            if "runtime" in overrides or {"weights", "fallback_weights"}.intersection(
                overrides.get("detector", {}) if isinstance(overrides.get("detector"), dict) else {}
            ):
                raise ValueError("Storage and model paths are deployment settings")
            return load_config(preset, overrides)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/api/presets")
    def presets() -> list[str]:
        return (
            ["dev_fixture"]
            if all(
                (fixture / f).is_file()
                for f in ("camera_A.mp4", "camera_B.mp4", "gt_A.csv", "gt_B.csv")
            )
            else []
        )

    @router.post("/api/runs/from-preset")
    def preset(body: PresetRequest) -> dict[str, str]:
        if body.preset not in presets():
            raise HTTPException(404, "Preset data unavailable")
        cfg = config(body.preset, body.config_overrides)
        return {
            "run_id": jobs.submit(
                cfg,
                {"A": fixture / "camera_A.mp4", "B": fixture / "camera_B.mp4"},
                {"A": fixture / "gt_A.csv", "B": fixture / "gt_B.csv"},
            )
        }

    @router.post("/api/runs")
    def upload(
        video_a: Annotated[UploadFile, File()],
        video_b: Annotated[UploadFile, File()],
        config_overrides: Annotated[str, Form()] = "{}",
    ) -> dict[str, str]:
        try:
            overrides = json.loads(config_overrides)
            if not isinstance(overrides, dict):
                raise ValueError("Overrides must be an object")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        cfg = config(None, overrides)
        if cfg.detector.mode == "groundtruth":
            raise HTTPException(422, "Uploads require an image-based detector")
        inputs = root / "uploads" / uuid4().hex
        inputs.mkdir(parents=True)
        videos = {"A": inputs / "A.mp4", "B": inputs / "B.mp4"}
        for camera, source in (("A", video_a), ("B", video_b)):
            with videos[camera].open("xb") as destination:
                copyfileobj(source.file, destination)
        return {"run_id": jobs.submit(cfg, videos, None)}

    @router.get("/api/runs")
    def listing() -> list[RunStatus]:
        return [
            jobs.status(p.stem)
            for p in sorted((root / "jobs").glob("*.json"), key=lambda p: -p.stat().st_mtime)
        ]

    @router.get("/api/runs/{run_id}/status")
    def status(run_id: str) -> RunStatus:
        directory(run_id)
        file(root / "jobs" / f"{run_id}.json")
        return jobs.status(run_id)

    def full(run_id: str) -> RunResult:
        return RunResult.model_validate_json(file(directory(run_id) / "results.json").read_bytes())

    @router.get("/api/runs/{run_id}")
    def results(run_id: str) -> dict[str, Any]:
        return full(run_id).model_dump(
            exclude={key: {"__all__": {"boxes"}} for key in ("tracks_A", "tracks_B")}
        )

    @router.get("/api/runs/{run_id}/tracks/{camera}")
    def tracks(run_id: str, camera: Literal["A", "B"]) -> FileResponse:
        return FileResponse(file(directory(run_id) / f"tracks_{camera}.json"))

    @router.get("/api/runs/{run_id}/queries/{track_id}")
    def query(run_id: str, track_id: str) -> dict[str, Any]:
        for q in full(run_id).queries:
            if q.query_track_id == track_id:
                return q.model_dump()
        raise HTTPException(404, "Unknown query")

    @router.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str) -> dict[str, Any] | None:
        return full(run_id).metrics

    @router.get("/api/runs/{run_id}/export.json")
    @router.get("/api/runs/{run_id}/export.csv")
    @router.get("/api/runs/{run_id}/queries.csv")
    def download(run_id: str, request: Request) -> FileResponse:
        name = {
            "export.json": "results.json",
            "export.csv": "candidates.csv",
            "queries.csv": "queries.csv",
        }[request.url.path.rsplit("/", 1)[-1]]
        path = file(directory(run_id) / name)
        return FileResponse(path, filename=name)

    @router.get("/media/runs/{run_id}/{media:path}")
    def media(run_id: str, media: str) -> FileResponse:
        base = directory(run_id).resolve()
        path = (base / media).resolve()
        if not path.is_relative_to(base) or path.suffix.lower() not in {".mp4", ".jpg", ".png"}:
            raise HTTPException(404, "Unknown media")
        return FileResponse(file(path))

    return router
