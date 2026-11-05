"""Clock alignment and transit evidence in seconds, independent of video backends."""

import math
from statistics import median

import numpy as np

from app.config import ClockConfig, MatchingConfig, TopologyConfig
from app.schemas import ClockOffset


def estimate_clock(deltas: list[float], cfg: ClockConfig, sign: int = 1) -> ClockOffset:
    """Report the canonical B-minus-A offset from confident, mutual-best observations."""
    if cfg.mode != "estimate":
        return ClockOffset(
            mode=cfg.mode,
            estimated_s=cfg.fixed_s if cfg.mode == "fixed" else 0,
            reliable=True,
            n_pairs_used=0,
        )
    mode = None
    if deltas:
        # Evaluate the Gaussian density over the configured search interval.
        count = math.ceil(2 * cfg.search_range_s / cfg.kde_bandwidth_s) * 4 + 1
        grid = np.linspace(-cfg.search_range_s, cfg.search_range_s, count)
        density = sum(np.exp(-0.5 * ((grid - d) / cfg.kde_bandwidth_s) ** 2) for d in deltas)
        mode = float(grid[np.argmax(density)])
    reliable = len(deltas) >= cfg.min_confident_pairs and cfg.expected_transit_s is not None
    offset = 0.0
    if reliable and cfg.expected_transit_s is not None:
        offset = float(
            np.clip(
                sign * (median(deltas) - cfg.expected_transit_s),
                -cfg.search_range_s,
                cfg.search_range_s,
            )
        )
    return ClockOffset(
        mode=cfg.mode,
        estimated_s=offset,
        n_pairs_used=len(deltas),
        reliable=reliable,
        raw_delta_mode_s=mode,
    )


def transit_score(
    adjusted: float, clock: ClockOffset, topology: TopologyConfig, matching: MatchingConfig
) -> tuple[float, bool]:
    if adjusted < matching.impossible_delta_s:
        return 0.0, True
    widen = 0 if clock.reliable else topology.clock_offset.search_range_s
    distance = max(
        0.0, topology.min_transit_s - widen - adjusted, adjusted - topology.max_transit_s - widen
    )
    return math.exp(-((distance / matching.temporal_soft_tau_s) ** 2)), False


def direction_score(motion: tuple[float, float], expected: tuple[float, float] | None) -> float:
    if expected is None or not (length := math.hypot(*expected) * math.hypot(*motion)):
        return 0.5
    cosine = sum(a * b for a, b in zip(motion, expected, strict=True)) / length
    return (max(-1.0, min(1.0, cosine)) + 1) / 2
