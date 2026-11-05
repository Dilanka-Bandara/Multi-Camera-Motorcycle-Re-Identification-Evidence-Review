"""Compare crop colours under a per-camera grey-world illuminant estimate."""

from pathlib import Path

import cv2
import numpy as np

from app.config import ColorConfig
from app.features.color import extract_color
from app.schemas import Track


def corrected_lab(tracks: list[Track], root: Path, cfg: ColorConfig) -> dict[str, list[float]]:
    totals = np.zeros(3)
    pixels = 0
    for t in tracks:
        for c in t.crops:
            image = cv2.imread(str(root / c.path_vehicle))
            if image is None:
                raise ValueError(f"Cannot read {c.path_vehicle}")
            totals += image.sum(axis=(0, 1))
            pixels += image.shape[0] * image.shape[1]
    if not pixels:
        return {}
    means = totals / pixels
    gain = np.divide(means.mean(), means, out=np.ones(3), where=means > 0)
    result = {}
    for t in tracks:
        labs = []
        for crop in t.crops:
            image = cv2.imread(str(root / crop.path_vehicle))
            balanced = np.clip(image * gain, 0, 255).astype(np.uint8)
            labs.append(extract_color(balanced, cfg)[1])
        result[t.track_id] = np.mean(labs, axis=0).tolist()
    return result
