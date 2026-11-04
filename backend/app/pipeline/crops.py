"""Image measurements and bounded top-K crop persistence."""

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import AppConfig, CropsConfig
from app.pipeline.geometry import iou
from app.pipeline.rider_assoc import AssociatedDetection
from app.schemas import BBox, CropInfo

ImageArray = NDArray[np.uint8]


@dataclass
class Observation:
    """Scalar measurements only; image arrays are not retained between frames."""

    associated: AssociatedDetection
    quality: float
    sharpness: float
    truncated: bool
    occluded: bool
    histogram: list[float]


def crop_image(frame: ImageArray, box: BBox) -> ImageArray:
    """Convert normalized edges into clamped pixel indices at extraction time."""
    height, width = frame.shape[:2]
    return frame[
        max(0, math.floor(box.y1 * height)) : min(height, math.ceil(box.y2 * height)),
        max(0, math.floor(box.x1 * width)) : min(width, math.ceil(box.x2 * width)),
    ]


def measure_observations(
    video: Path,
    associated: list[AssociatedDetection],
    config: AppConfig,
) -> list[Observation]:
    """Score every observed object and compute a histogram for stitching."""
    by_frame: dict[int, list[AssociatedDetection]] = defaultdict(list)
    for item in associated:
        by_frame[item.detection.frame_idx].append(item)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot read {video}")
    observations: list[Observation] = []
    frame_idx = 0
    cfg = config.crops
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for item in by_frame.get(frame_idx, []):
                box = item.union_box
                crop = crop_image(frame, box)
                sharpness = float(
                    cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                )
                truncated = min(box.x1, box.y1, 1 - box.x2, 1 - box.y2) <= cfg.border_margin
                occluded = any(
                    other is not item and iou(box, other.union_box) > cfg.occlusion_iou
                    for other in by_frame[frame_idx]
                )
                aspect = crop.shape[1] / crop.shape[0]
                aspect_score = min(1, aspect / cfg.aspect_min, cfg.aspect_max / aspect)
                quality = (
                    cfg.weights.size * min(1, (box.y2 - box.y1) / cfg.size_reference)
                    + cfg.weights.conf * item.detection.conf
                    + cfg.weights.sharpness * min(1, sharpness / cfg.sharpness_reference)
                    + cfg.weights.aspect * aspect_score
                    - cfg.truncated_penalty * truncated
                    - cfg.occluded_penalty * occluded
                )
                hsv = cv2.cvtColor(crop_image(frame, item.vehicle_box), cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv], [0, 1, 2], None, list(config.color.hsv_bins), [0, 180, 0, 256, 0, 256]
                ).flatten()
                hist /= max(float(hist.sum()), 1)
                observations.append(
                    Observation(
                        item, max(0, min(1, quality)), sharpness, truncated, occluded, hist.tolist()
                    )
                )
            frame_idx += 1
    finally:
        capture.release()
    if len(observations) != len(associated):
        raise ValueError("Video ended before all detection frames were decoded")
    return observations


def select_crops(
    observations: list[Observation], height: int, config: CropsConfig
) -> list[Observation]:
    """Greedily choose quality-ranked crops while enforcing temporal diversity."""
    chosen: list[Observation] = []
    for item in sorted(observations, key=lambda o: (-o.quality, o.associated.detection.frame_idx)):
        box = item.associated.union_box
        pixel_height = math.ceil(box.y2 * height) - math.floor(box.y1 * height)
        if pixel_height < config.min_box_height_px:
            continue
        if any(
            abs(item.associated.detection.frame_idx - other.associated.detection.frame_idx)
            < config.min_frame_gap
            for other in chosen
        ):
            continue
        chosen.append(item)
        if len(chosen) == config.top_k:
            break
    return chosen


def save_crops(
    video: Path,
    run_dir: Path,
    selected: dict[str, list[Observation]],
) -> dict[str, list[CropInfo]]:
    """Decode once and persist full/vehicle/available rider crops as JPEGs."""
    by_frame: dict[int, list[tuple[str, Observation]]] = defaultdict(list)
    result: dict[str, list[CropInfo]] = {key: [] for key in selected}
    for track_id, items in selected.items():
        for item in items:
            by_frame[item.associated.detection.frame_idx].append((track_id, item))
    capture = cv2.VideoCapture(str(video))
    frame_idx = 0
    try:
        while by_frame:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Video ended before selected crop frames")
            for track_id, item in by_frame.pop(frame_idx, []):
                associated = item.associated
                paths: dict[str, str | None] = {"full": None, "vehicle": None, "rider": None}
                for part, box in (
                    ("full", associated.union_box),
                    ("vehicle", associated.vehicle_box),
                    ("rider", associated.rider_box),
                ):
                    if box is None:
                        continue
                    relative = Path("crops") / track_id / f"{frame_idx:06d}_{part}.jpg"
                    destination = run_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise FileExistsError(destination)
                    if not cv2.imwrite(str(destination), crop_image(frame, box)):
                        raise OSError(f"Failed to write {destination}")
                    paths[part] = relative.as_posix()
                result[track_id].append(
                    CropInfo(
                        frame_idx=frame_idx,
                        t_s=associated.detection.t_s,
                        bbox=associated.union_box,
                        quality=item.quality,
                        sharpness=item.sharpness,
                        truncated=item.truncated,
                        occluded=item.occluded,
                        path_full=paths["full"],
                        path_vehicle=paths["vehicle"],
                        path_rider=paths["rider"],
                    )
                )
            frame_idx += 1
    finally:
        capture.release()
    for track_id in result:
        ranks = {o.associated.detection.frame_idx: i for i, o in enumerate(selected[track_id])}
        result[track_id].sort(key=lambda crop: ranks[crop.frame_idx])
    return result
