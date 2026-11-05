"""Serialize pipeline jobs and maintain atomic, restart-readable status records."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.config import AppConfig
from app.pipeline.orchestrator import run_pipeline
from app.schemas import Camera, RunStatus


class Jobs:
    def __init__(self, data: Path, models: Path) -> None:
        self.data = data
        self.models = models
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moto-pipeline")
        self.lock = Lock()
        self.active: dict[str, str] = {}

    def status(self, run_id: str) -> RunStatus:
        path = self.data / "jobs" / f"{run_id}.json"
        status = RunStatus.model_validate_json(path.read_bytes())
        if status.state in ("queued", "running") and run_id not in self.active.values():
            status.state = "failed"
            status.error = "Worker stopped before completing the run"
        return status

    def save(self, status: RunStatus) -> None:
        directory = self.data / "jobs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{status.run_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(status.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def submit(
        self, cfg: AppConfig, videos: dict[Camera, Path], groundtruth: dict[Camera, Path] | None
    ) -> str:
        digest = hashlib.sha256(b"phase6-v1")
        digest.update(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode())
        for path in [videos["A"], videos["B"], *((groundtruth or {}).values())]:
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        key = digest.hexdigest()
        cache = self.data / "cache" / key
        with self.lock:
            if key in self.active:
                return self.active[key]
            if cache.is_file():
                previous = cache.read_text().strip()
                if (self.data / "runs" / previous / "results.json").is_file():
                    return previous
            run_id = uuid4().hex
            status = RunStatus(run_id=run_id, state="queued", stage="queued", progress=0)
            self.save(status)
            self.active[key] = run_id
            self.pool.submit(self.run, key, status, cfg, videos, groundtruth, cache)
            return run_id

    def run(
        self,
        key: str,
        status: RunStatus,
        cfg: AppConfig,
        videos: dict[Camera, Path],
        truth: dict[Camera, Path] | None,
        cache: Path,
    ) -> None:
        def update(stage: str, progress: float) -> None:
            status.state, status.stage, status.progress = "running", stage, progress
            self.save(status)

        try:
            update("starting", 0)
            meta = run_pipeline(
                cfg, videos, self.data / "runs" / status.run_id, truth, self.models, update
            )
            status.timings_s = meta["timings_s"]
            status.fps_processed = meta["fps_processed"]
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(status.run_id)
            status.state, status.stage, status.progress = "done", "done", 1
        except Exception as exc:
            status.state, status.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.save(status)
            with self.lock:
                self.active.pop(key, None)
