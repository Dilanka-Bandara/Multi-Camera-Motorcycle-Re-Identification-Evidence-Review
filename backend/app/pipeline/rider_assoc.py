"""Associate rider rectangles without importing image or model libraries."""

from dataclasses import dataclass

from app.config import AssociationConfig
from app.pipeline.geometry import area, center, intersection, union
from app.schemas import BBox, Detection


@dataclass(frozen=True)
class AssociatedDetection:
    """Vehicle observation and optional rider observation in the same frame."""

    detection: Detection
    vehicle_box: BBox
    rider_box: BBox | None
    union_box: BBox


def associate_riders(
    vehicles: list[Detection],
    riders: list[Detection],
    config: AssociationConfig,
) -> list[AssociatedDetection]:
    """Pick the highest-overlap rider for each motorcycle; cars have no rider."""
    result: list[AssociatedDetection] = []
    for vehicle in vehicles:
        best: BBox | None = None
        best_overlap = -1.0
        for rider in riders if vehicle.cls == 3 else []:
            if rider.frame_idx != vehicle.frame_idx:
                continue
            box = rider.bbox
            if not vehicle.bbox.x1 <= center(box)[0] <= vehicle.bbox.x2:
                continue
            lower = BBox(x1=box.x1, x2=box.x2, y1=(box.y1 + box.y2) / 2, y2=box.y2)
            overlap = intersection(lower, vehicle.bbox) / area(box)
            if overlap >= config.min_overlap and overlap > best_overlap:
                best, best_overlap = box, overlap
        result.append(
            AssociatedDetection(
                vehicle, vehicle.bbox, best, union(vehicle.bbox, best) if best else vehicle.bbox
            )
        )
    return result
