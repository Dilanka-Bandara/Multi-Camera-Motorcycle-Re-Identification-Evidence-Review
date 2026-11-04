"""Pure greedy track stitching; appearance vectors are supplied by image code."""

import math
from dataclasses import dataclass

from app.config import StitchingConfig
from app.pipeline.geometry import center
from app.schemas import Detection


@dataclass
class Fragment:
    """Raw IDs, time-ordered observations and mean appearance histogram."""

    raw_ids: list[int]
    detections: list[Detection]
    appearance: list[float]


def cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity, or zero when either vector has no information."""
    if len(a) != len(b):
        raise ValueError("Appearance dimensions differ")
    norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def stitch_fragments(fragments: list[Fragment], config: StitchingConfig) -> list[Fragment]:
    """Merge nonoverlapping compatible fragments in earliest-first order."""
    merged: list[Fragment] = []
    for source in sorted(fragments, key=lambda f: (f.detections[0].t_s, f.raw_ids)):
        current = Fragment(list(source.raw_ids), list(source.detections), list(source.appearance))
        if config.enabled:
            for previous in merged:
                tail, head = previous.detections[-1], current.detections[0]
                gap = head.t_s - tail.t_s
                if tail.cls != head.cls or not 0 < gap <= config.max_gap_s:
                    continue
                a, b = center(tail.bbox), center(head.bbox)
                if math.dist(a, b) > config.max_center_jump:
                    continue
                if cosine(previous.appearance, current.appearance) < config.min_appearance_sim:
                    continue
                n, m = len(previous.detections), len(current.detections)
                previous.appearance = [
                    (x * n + y * m) / (n + m)
                    for x, y in zip(previous.appearance, current.appearance, strict=True)
                ]
                previous.raw_ids.extend(current.raw_ids)
                previous.detections.extend(current.detections)
                break
            else:
                merged.append(current)
        else:
            merged.append(current)
    return merged
