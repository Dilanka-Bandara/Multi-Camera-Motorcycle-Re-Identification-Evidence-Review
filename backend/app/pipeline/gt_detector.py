"""Deterministic ground-truth tracker simulator; no image libraries required."""

import csv
import random
from collections import defaultdict
from pathlib import Path

from app.config import GroundTruthConfig
from app.schemas import BBox, Detection


def detect_groundtruth(
    path: Path, *, fps: float, start_time_s: float, options: GroundTruthConfig, seed: int,
) -> list[Detection]:
    """Read normalized GT; optionally drop rows and split each track once at a gap.

    Fragment probability is per track, not per frame. New IDs exceed every input
    identity. A missing cls column means motorcycle (COCO 3). Timestamps use only
    the supplied camera clock; fixture truth metadata is never consulted.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    rng = random.Random(seed)
    grouped: dict[int, list[Detection]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame_idx"])
            detection = Detection(
                frame_idx=frame, t_s=start_time_s + frame / fps,
                bbox=BBox(**{key: float(row[key]) for key in ("x1", "y1", "x2", "y2")}),
                conf=options.confidence, cls=int(row.get("cls") or 3),
                track_id=int(row["identity_id"]),
            )
            grouped[detection.track_id].append(detection)
    next_id = max(grouped, default=-1) + 1
    output: list[Detection] = []
    for identity in sorted(grouped):
        detections = sorted(grouped[identity], key=lambda d: d.frame_idx)
        if len({d.frame_idx for d in detections}) != len(detections):
            raise ValueError(f"Duplicate GT frame for identity {identity}")
        split_frame: int | None = None
        fragment_id = identity
        gap = options.fragment_gap_frames
        possible = [d.frame_idx for d in detections[1:] if d.frame_idx + gap
                    < detections[-1].frame_idx]
        if possible and rng.random() < options.fragment_prob:
            split_frame = rng.choice(possible)
            fragment_id = next_id
            next_id += 1
        for detection in detections:
            if split_frame is not None:
                if split_frame <= detection.frame_idx < split_frame + gap:
                    continue
                if detection.frame_idx >= split_frame + gap:
                    detection = detection.model_copy(update={"track_id": fragment_id})
            if rng.random() >= options.drop_rate:
                output.append(detection)
    return sorted(output, key=lambda d: (d.frame_idx, d.track_id))
