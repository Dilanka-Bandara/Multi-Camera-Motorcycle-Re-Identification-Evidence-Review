"""Run Phase 2 stages and persist normalized tracks, crops and browser videos."""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.config import AppConfig
from app.pipeline.crops import Observation, measure_observations, save_crops, select_crops
from app.pipeline.detect_track import COCO_CLASSES, detect_track
from app.pipeline.gt_detector import detect_groundtruth
from app.pipeline.ingest import ingest_video
from app.pipeline.motion import summarize_motion
from app.pipeline.representation import represent_tracks
from app.pipeline.rider_assoc import associate_riders
from app.pipeline.stitching import Fragment, stitch_fragments
from app.pipeline.track_quality import score_track
from app.schemas import Camera, Detection, FrameBox, Track

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    config: AppConfig,
    videos: dict[Camera, Path],
    run_dir: Path,
    groundtruth: dict[Camera, Path] | None = None,
    models_dir: Path = Path("models"),
) -> dict[str, Any]:
    """Execute sequential camera stages; fail safely if a run directory already exists."""
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = run_dir.name
    started = time.perf_counter()
    timings: dict[str, float] = {}
    rates: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "config_snapshot": config.model_dump(mode="json"),
        "videos": {},
        "detectors": {},
        "dropped_tracks": {},
    }
    target_classes = {COCO_CLASSES[name] for name in config.target.classes}
    try:
        for camera in ("A", "B"):
            camera_config = getattr(config.cameras, camera)
            t0 = time.perf_counter()
            web = run_dir / f"camera_{camera}.mp4"
            meta = ingest_video(
                videos[camera], web, camera_config.start_time_s, config.detector.max_process_height
            )
            metadata["videos"][camera] = meta.model_dump(mode="json") | {"path": web.name}
            timings[f"ingest_{camera}"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            if config.detector.mode == "groundtruth":
                if groundtruth is None or camera not in groundtruth:
                    raise ValueError("Ground-truth mode requires both GT CSV paths")
                detections = detect_groundtruth(
                    groundtruth[camera],
                    fps=meta.fps,
                    start_time_s=meta.start_time_s,
                    options=config.groundtruth,
                    seed=config.runtime.seed,
                )
                provenance: dict[str, Any] = {"mode": "groundtruth"}
            else:
                detections, provenance = detect_track(web, meta, config, models_dir, run_id)
            timings[f"detect_{camera}"] = time.perf_counter() - t0
            rates[f"detect_{camera}"] = len({d.frame_idx for d in detections}) / max(
                timings[f"detect_{camera}"], 1e-12
            )
            metadata["detectors"][camera] = provenance
            t0 = time.perf_counter()
            vehicles = [d for d in detections if d.cls in target_classes]
            riders = [d for d in detections if d.cls == COCO_CLASSES["person"]]
            associated = associate_riders(vehicles, riders, config.rider_association)
            observations = measure_observations(web, associated, config)
            by_id: dict[int, list[Detection]] = defaultdict(list)
            for detection in vehicles:
                by_id[detection.track_id].append(detection)
            obs_by_id: dict[int, list[Observation]] = defaultdict(list)
            for observation in observations:
                obs_by_id[observation.associated.detection.track_id].append(observation)
            fragments: list[Fragment] = []
            for raw_id, items in by_id.items():
                histograms = [o.histogram for o in obs_by_id[raw_id]]
                appearance = [mean(values) for values in zip(*histograms, strict=True)]
                fragments.append(
                    Fragment([raw_id], sorted(items, key=lambda d: d.frame_idx), appearance)
                )
            stitched = stitch_fragments(fragments, config.tracker.stitching)
            selected: dict[str, list[Observation]] = {}
            tracks: list[Track] = []
            dropped = 0
            for index, fragment in enumerate(stitched):
                items = fragment.detections
                obs = [o for raw_id in fragment.raw_ids for o in obs_by_id[raw_id]]
                chosen = select_crops(obs, meta.height, config.crops)
                if len(items) < config.tracker.min_track_frames or not chosen:
                    dropped += 1
                    continue
                track_id = f"{camera}-{index:04d}"
                selected[track_id] = chosen
                heights = [(d.bbox.y2 - d.bbox.y1) * meta.height for d in items]
                quality, breakdown = score_track(
                    n_frames=len(items),
                    mean_conf=mean(d.conf for d in items),
                    mean_crop_quality=mean(o.quality for o in chosen),
                    truncated_fraction=mean(o.truncated for o in obs),
                    occluded_fraction=mean(o.occluded for o in obs),
                    median_height_px=median(heights),
                    config=config.track_quality,
                )
                entry, exit_point, vector = summarize_motion([d.bbox for d in items], config.motion)
                tracks.append(
                    Track(
                        track_id=track_id,
                        camera=camera,
                        raw_track_ids=fragment.raw_ids,
                        first_frame=items[0].frame_idx,
                        last_frame=items[-1].frame_idx,
                        t_start_s=items[0].t_s,
                        t_end_s=items[-1].t_s,
                        boxes=[FrameBox(frame_idx=d.frame_idx, bbox=d.bbox) for d in items],
                        n_frames=len(items),
                        mean_conf=mean(d.conf for d in items),
                        median_box_h_px=median(heights),
                        entry_point=entry,
                        exit_point=exit_point,
                        motion_vector=vector,
                        quality=quality,
                        quality_breakdown=breakdown,
                        crops=[],
                        has_rider=any(o.associated.rider_box is not None for o in obs),
                    )
                )
            saved = save_crops(web, run_dir, selected)
            for track in tracks:
                track.crops = saved[track.track_id]
            (run_dir / f"tracks_{camera}.json").write_text(
                json.dumps([t.model_dump(mode="json") for t in tracks], indent=2), encoding="utf-8"
            )
            metadata["dropped_tracks"][camera] = dropped
            timings[f"tracks_crops_{camera}"] = time.perf_counter() - t0
            rates[f"tracks_crops_{camera}"] = meta.frame_count / timings[f"tracks_crops_{camera}"]
            LOGGER.info(
                "run_id=%s camera=%s tracks=%s dropped=%s", run_id, camera, len(tracks), dropped
            )
            t0 = time.perf_counter()
            metadata.setdefault("representations", {})[camera] = represent_tracks(
                tracks, run_dir, config, models_dir
            )
            timings[f"representations_{camera}"] = time.perf_counter() - t0
        metadata["state"] = "done"
    except Exception as exc:
        metadata["state"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings["total"] = time.perf_counter() - started
        metadata["timings_s"] = timings
        metadata["fps_processed"] = rates
        (run_dir / "pipeline.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
