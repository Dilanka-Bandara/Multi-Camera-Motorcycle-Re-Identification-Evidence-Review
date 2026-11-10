"""Connect independent evidence checks to the sequential pipeline."""

import logging
import os
from pathlib import Path

from app.config import AppConfig
from app.features.camera_color import corrected_lab
from app.features.keypoints import Keypoints
from app.schemas import Candidate, ClockOffset, QueryResult, Track, TrackRepresentation
from app.verify.calibrator import Calibrator
from app.verify.checks import review
from app.verify.policy import resolve


def review_queries(
    rows: dict[str, list[Candidate]],
    cameras: dict[str, list[Track]],
    features: dict[str, TrackRepresentation],
    clock: ClockOffset,
    cfg: AppConfig,
    root: Path,
) -> list[QueryResult]:
    directory = Path(
        os.environ.get("MOTO_CONFIG_DIR", Path(__file__).resolve().parents[3] / "configs")
    )
    calibrator = Calibrator.load(directory / cfg.verifier.calibrator)
    tracks = {t.track_id: t for camera in cameras.values() for t in camera}
    labs = {
        k: v for group in cameras.values() for k, v in corrected_lab(group, root, cfg.color).items()
    }
    matcher = Keypoints(cfg.verifier.keypoints, cfg.detector.device)
    claims: dict[str, float] = {}
    for row in rows.values():
        if row:
            lead = row[0]
            claims[lead.candidate_track_id] = max(
                claims.get(lead.candidate_track_id, 0), lead.scores.combined
            )
    try:
        results = []
        for index, (query_id, row) in enumerate(rows.items(), start=1):
            logging.getLogger(__name__).info(
                "run_id=%s reviewing query=%s %s/%s", root.name, query_id, index, len(rows)
            )
            left = tracks[query_id]
            for candidate in row[: cfg.verifier.review_top_n]:
                right = tracks[candidate.candidate_track_id]
                geometry = matcher.review(left, right, root, candidate.pair_id)
                rival = max((c.scores.combined for c in row if c is not candidate), default=0)
                candidate.verification = review(
                    candidate,
                    left,
                    right,
                    features[query_id],
                    features[right.track_id],
                    labs[query_id],
                    labs[right.track_id],
                    clock,
                    cfg,
                    calibrator,
                    geometry.inliers,
                    geometry.extracted,
                    geometry.backend,
                    geometry.note,
                    geometry.path,
                    candidate.scores.combined - rival,
                    claims.get(right.track_id, 0) > candidate.scores.combined,
                )
            results.append(resolve(query_id, row))
        return results
    finally:
        matcher.close()
