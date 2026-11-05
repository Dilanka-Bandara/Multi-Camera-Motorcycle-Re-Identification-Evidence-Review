"""Map track observations to annotated identities by per-frame IoU votes."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from app.config import EvaluationConfig
from app.pipeline.geometry import iou
from app.schemas import BBox, Track


def map_truth(
    tracks: list[Track], path: Path, cfg: EvaluationConfig
) -> tuple[dict[str, str], set[str]]:
    frames: dict[int, list[tuple[str, BBox]]] = defaultdict(list)
    identities = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            ident = row["identity_id"]
            identities.add(ident)
            frames[int(row["frame_idx"])].append(
                (ident, BBox.model_validate({k: row[k] for k in ("x1", "y1", "x2", "y2")}))
            )
    labels = {}
    for t in tracks:
        votes: Counter[str] = Counter()
        for obs in t.boxes:
            best = max(
                ((iou(obs.bbox, box), ident) for ident, box in frames.get(obs.frame_idx, [])),
                default=None,
            )
            if best and best[0] >= cfg.min_iou:
                votes[best[1]] += 1
        if votes:
            ident, count = votes.most_common(1)[0]
            if count >= len(t.boxes) * cfg.min_vote_fraction:
                labels[t.track_id] = ident
    return labels, identities
