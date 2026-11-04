"""Pure motion summaries in normalized coordinates."""

import math

from app.config import MotionConfig
from app.pipeline.geometry import center
from app.schemas import BBox


def summarize_motion(
    boxes: list[BBox],
    config: MotionConfig,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return smoothed entry, exit and unit motion (zero for negligible motion)."""
    if not boxes:
        raise ValueError("Motion requires at least one box")

    def mean(items: list[BBox]) -> tuple[float, float]:
        points = [center(box) for box in items]
        return tuple(sum(p[i] for p in points) / len(points) for i in range(2))

    entry = mean(boxes[: config.endpoint_boxes])
    exit_point = mean(boxes[-config.endpoint_boxes :])
    dx, dy = exit_point[0] - entry[0], exit_point[1] - entry[1]
    length = math.hypot(dx, dy)
    vector = (dx / length, dy / length) if length > config.min_displacement else (0.0, 0.0)
    return entry, exit_point, vector
