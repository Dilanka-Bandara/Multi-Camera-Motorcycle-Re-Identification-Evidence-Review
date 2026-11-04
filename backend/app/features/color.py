"""HSV and perceptual Lab features extracted from the inner crop region."""

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import ColorConfig


def extract_color(image: NDArray[np.uint8], config: ColorConfig) -> tuple[list[float], list[float]]:
    """Return an L1-normalized HSV histogram and Lab mean in physical Lab units."""
    height, width = image.shape[:2]
    margin = (1 - config.center_fraction) / 2
    x, y = int(width * margin), int(height * margin)
    center = image[y : height - y, x : width - x]
    if center.size == 0:
        raise ValueError("Color features require a nonempty crop")
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(config.hsv_bins), [0, 180, 0, 256, 0, 256]
    ).flatten()
    hist /= hist.sum()
    lab = cv2.cvtColor(center.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return hist.tolist(), lab.mean(axis=(0, 1)).tolist()
