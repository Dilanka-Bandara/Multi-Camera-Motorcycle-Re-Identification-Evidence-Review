"""Pure normalized-coordinate geometry shared by pipeline stages."""

from app.schemas import BBox


def area(box: BBox) -> float:
    """Return normalized rectangle area."""
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def intersection(a: BBox, b: BBox) -> float:
    """Return intersection area, zero for disjoint rectangles."""
    return max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))


def iou(a: BBox, b: BBox) -> float:
    """Return intersection over union."""
    overlap = intersection(a, b)
    return overlap / (area(a) + area(b) - overlap)


def union(a: BBox, b: BBox) -> BBox:
    """Return the smallest enclosing rectangle."""
    return BBox(x1=min(a.x1, b.x1), y1=min(a.y1, b.y1), x2=max(a.x2, b.x2), y2=max(a.y2, b.y2))


def center(box: BBox) -> tuple[float, float]:
    """Return a normalized midpoint."""
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
