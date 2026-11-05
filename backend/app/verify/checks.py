"""Independent evidence computed from timestamps, tracks and per-crop representations."""

from math import dist, hypot

from app.config import AppConfig
from app.matching.similarity import color_similarity, set_similarity
from app.matching.temporal import direction_score, transit_score
from app.schemas import (
    Candidate,
    ClockOffset,
    EvidenceItem,
    Track,
    TrackRepresentation,
    Verification,
)
from app.verify.calibrator import Calibrator
from app.verify.policy import decide, summarize


def review(
    candidate: Candidate,
    left: Track,
    right: Track,
    a: TrackRepresentation,
    b: TrackRepresentation,
    lab_a: list[float],
    lab_b: list[float],
    clock: ClockOffset,
    cfg: AppConfig,
    calibrator: Calibrator,
    inliers: int,
    extracted: bool,
    backend: str,
    note: str | None,
    image: str | None,
    margin: float,
    conflict: bool,
) -> Verification:
    opt = cfg.verifier
    items: list[EvidenceItem] = []

    def record(
        check: str,
        value: float | str | None,
        good: bool,
        bad: bool,
        threshold: float | str | None,
        message: str,
        severity: str = "medium",
        missing: bool = False,
    ) -> None:
        polarity = (
            "missing" if missing else "contradicts" if bad else "supports" if good else "neutral"
        )
        items.append(
            EvidenceItem(
                check=check,
                polarity=polarity,
                severity=severity if bad else "low",
                value=value,
                threshold=threshold,
                message=message,
            )
        )

    top_m = cfg.matching.set_sim_top_m
    visual = set_similarity(a.emb_full, b.emb_full, top_m)
    vehicle = set_similarity(a.emb_vehicle, b.emb_vehicle, top_m)
    rider = set_similarity(a.emb_rider, b.emb_rider, top_m) if a.emb_rider and b.emb_rider else None
    record(
        "vehicle_part_similarity",
        vehicle,
        vehicle >= opt.vehicle_support,
        vehicle < opt.vehicle_fail,
        f"{opt.vehicle_fail} / {opt.vehicle_support}",
        f"vehicle similarity is {vehicle:.2f}",
        "high",
    )
    if rider is None:
        record(
            "rider_part_similarity",
            None,
            False,
            False,
            None,
            "rider crops are unavailable",
            missing=True,
        )
    else:
        record(
            "rider_part_similarity",
            rider,
            rider >= opt.rider_support,
            rider < opt.rider_fail,
            f"{opt.rider_fail} / {opt.rider_support}",
            f"rider similarity is {rider:.2f}",
        )
        gap = abs(rider - vehicle)
        record(
            "part_disagreement",
            gap,
            False,
            gap > opt.part_disagreement_gap,
            opt.part_disagreement_gap,
            f"rider and vehicle scores differ by {gap:.2f}",
        )
    color = color_similarity(a.color_hist, b.color_hist)
    de = dist(lab_a, lab_b)
    record(
        "color_consistency",
        de,
        de < opt.color_support_de,
        de > opt.color_fail_de,
        f"{opt.color_support_de} / {opt.color_fail_de}",
        f"adjusted colour distance is {de:.1f}",
        "high",
    )
    adjusted = (
        right.t_start_s - left.t_end_s - (1 if left.camera == "A" else -1) * clock.estimated_s
    )
    temporal, impossible = transit_score(adjusted, clock, cfg.topology, cfg.matching)
    inside = cfg.topology.min_transit_s <= adjusted <= cfg.topology.max_transit_s
    record(
        "timing_plausibility",
        adjusted,
        inside,
        impossible or not inside,
        f"{cfg.topology.min_transit_s}..{cfg.topology.max_transit_s} s",
        f"adjusted transit is {adjusted:.1f} s",
        "critical" if impossible else "medium",
    )
    record(
        "clock_offset_reliability",
        clock.estimated_s,
        clock.reliable,
        False,
        None,
        "clock alignment is available"
        if clock.reliable
        else "timing uncertain: clock offset unknown",
        missing=not clock.reliable,
    )
    directions = []
    for track in (left, right):
        expected = (
            cfg.topology.expected_exit_dir_A
            if track.camera == "A"
            else cfg.topology.expected_entry_dir_B
        )
        known = bool(expected and hypot(*expected) and hypot(*track.motion_vector))
        score = direction_score(track.motion_vector, expected)
        directions.append(score)
        cosine = 2 * score - 1
        record(
            "direction_consistency",
            cosine if known else None,
            known and cosine >= opt.direction_support,
            known and cosine < opt.direction_fail,
            f"{opt.direction_fail} / {opt.direction_support}",
            f"camera {track.camera} motion cosine is {cosine:.2f}"
            if known
            else "motion is unknown",
            "high",
            missing=not known,
        )
        record(
            f"track_quality_{track.camera}",
            track.quality,
            track.quality >= opt.quality_support,
            track.quality < opt.min_track_quality,
            opt.min_track_quality,
            f"camera {track.camera} track quality is {track.quality:.2f}",
        )
        if len(track.crops) < opt.min_crops:
            record(
                "missing_info",
                len(track.crops),
                False,
                False,
                opt.min_crops,
                f"camera {track.camera} has few usable crops",
                missing=True,
            )
    kp = opt.keypoints
    record(
        "keypoint_inliers",
        inliers,
        inliers >= kp.min_inliers_pass,
        extracted and inliers <= kp.max_inliers_fail,
        f"{kp.max_inliers_fail} / {kp.min_inliers_pass}",
        f"{backend} found {inliers} geometric inliers"
        if extracted
        else "keypoints are unavailable",
        missing=not extracted,
    )
    if note:
        record("keypoint_backend", backend, False, False, None, note)
    record(
        "ambiguity",
        margin,
        margin >= 2 * cfg.matching.ambiguity_margin,
        candidate.ambiguous,
        cfg.matching.ambiguity_margin,
        "another pairing has similar support"
        if candidate.ambiguous
        else f"ranking margin is {margin:.2f}",
        "high",
    )
    if conflict:
        record(
            "assignment_conflict",
            None,
            False,
            True,
            None,
            "another query has a stronger claim",
            "high",
        )
    features = dict(
        visual=visual,
        vehicle_part=vehicle,
        rider_part=rider if rider is not None else 0.5,
        color_sim=color,
        keypoint_inliers_norm=min(1, inliers / kp.min_inliers_pass),
        temporal=temporal,
        direction=sum(directions) / 2,
        track_quality_min=min(left.quality, right.quality),
        margin=margin,
        missing_count=float(sum(e.polarity == "missing" for e in items)),
    )
    probability = calibrator.probability(features)
    status = decide(probability, items, candidate.ambiguous, features["track_quality_min"], opt)
    return Verification(
        status=status,
        confidence=probability,
        evidence=items,
        features=features,
        summary=summarize(status, items),
        keypoint_viz_path=image,
    )
