"""Pure track-level quality scoring."""

from app.config import TrackQualityConfig


def score_track(
    *,
    n_frames: int,
    mean_conf: float,
    mean_crop_quality: float,
    truncated_fraction: float,
    occluded_fraction: float,
    median_height_px: float,
    config: TrackQualityConfig,
) -> tuple[float, dict[str, float]]:
    """Combine interpretable quality components using configured weights."""
    breakdown = {
        "length": min(1, n_frames / config.reference_frames),
        "confidence": mean_conf,
        "crop_quality": mean_crop_quality,
        "untruncated": 1 - truncated_fraction,
        "unoccluded": 1 - occluded_fraction,
        "size": min(1, median_height_px / config.reference_height_px),
    }
    if set(config.weights) != set(breakdown) or abs(sum(config.weights.values()) - 1) > 1e-9:
        raise ValueError("Track-quality weights must cover all components and sum to one")
    score = sum(breakdown[key] * weight for key, weight in config.weights.items())
    return max(0, min(1, score)), breakdown
