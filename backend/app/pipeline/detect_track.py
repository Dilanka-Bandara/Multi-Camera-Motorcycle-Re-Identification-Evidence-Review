"""Offline YOLO tracking adapter. Heavy libraries are imported only on demand."""

import gc
import logging
import os
from pathlib import Path
from typing import Any

from app.config import AppConfig
from app.schemas import BBox, Detection, VideoMeta

LOGGER = logging.getLogger(__name__)
COCO_CLASSES = {"person": 0, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def process_stride(fps: float, target_fps: float) -> int:
    """Choose the guide's processing cadence from the configured target FPS."""
    if fps <= target_fps:
        return 1
    if fps <= 2 * target_fps:
        return 2
    return max(1, round(fps / target_fps))


def result_detections(result: Any, frame_idx: int, meta: VideoMeta) -> list[Detection]:
    """Convert tracked normalized boxes; discard untracked and degenerate boxes."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    rows = zip(
        boxes.xyxyn.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.cls.cpu().tolist(),
        boxes.id.cpu().tolist(),
        strict=True,
    )
    output: list[Detection] = []
    for coordinates, confidence, cls, track_id in rows:
        x1, y1, x2, y2 = [max(0.0, min(1.0, value)) for value in coordinates]
        if x2 <= x1 or y2 <= y1:
            continue
        output.append(
            Detection(
                frame_idx=frame_idx,
                t_s=meta.start_time_s + frame_idx / meta.fps,
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                conf=confidence,
                cls=int(cls),
                track_id=int(track_id),
            )
        )
    return output


def detect_track(
    video: Path,
    meta: VideoMeta,
    config: AppConfig,
    models_dir: Path,
    run_id: str,
) -> tuple[list[Detection], dict[str, Any]]:
    """Track with local weights, retry fallback weights and record model provenance."""
    paths = [models_dir / config.detector.weights, models_dir / config.detector.fallback_weights]
    if not any(path.is_file() for path in paths):
        raise FileNotFoundError("No local YOLO weights found; runtime downloads are disabled")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO

    cfg = config.detector
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    half = device != "cpu" and (cfg.half == "auto" or cfg.half is True)
    stride = (
        process_stride(meta.fps, cfg.target_fps) if cfg.vid_stride == "auto" else cfg.vid_stride
    )
    classes = [COCO_CLASSES[name] for name in config.target.classes] + [COCO_CLASSES["person"]]
    failures: list[str] = []
    for path in paths:
        if not path.is_file():
            failures.append(f"Local weights missing: {path.name}")
            continue
        model: Any = None
        try:
            model = YOLO(str(path))
            results = model.track(
                source=str(video),
                stream=True,
                persist=True,
                tracker=config.tracker.type,
                classes=classes,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=cfg.iou,
                half=half,
                device=device,
                vid_stride=stride,
                verbose=False,
                save=False,
            )
            detections: list[Detection] = []
            for index, result in enumerate(results):
                # Installed Ultralytics starts at frame 0, then skips stride-1 frames.
                frame_idx = index * stride
                detections.extend(result_detections(result, frame_idx, meta))
            return detections, {
                "weights": path.name,
                "device": device,
                "half": half,
                "stride": stride,
                "fallback_notes": failures,
            }
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            LOGGER.warning("run_id=%s YOLO failed: %s", run_id, failures[-1])
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("All local YOLO models failed: " + "; ".join(failures))
