"""Cache reuse must preserve feature matches and release bounded state."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from app.config import load_config
from app.features.keypoints import Image, Keypoints


def test_crop_cache_eviction_and_disable(tmp_path: Path) -> None:
    cfg = load_config().verifier.keypoints.model_copy(update={"backend": "sift", "cache_crops": 2})
    matcher = Keypoints(cfg)
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for path in paths:
        assert cv2.imwrite(str(path), np.zeros((16, 16), dtype=np.uint8))
    first = matcher._crop(paths[0])
    matcher._crop(paths[1])
    assert matcher._crop(paths[0]) is first
    matcher._crop(paths[2])
    assert list(matcher.cache) == [paths[0], paths[2]]
    matcher.close()
    assert not matcher.cache
    disabled = Keypoints(cfg.model_copy(update={"cache_crops": 0}))
    assert disabled._crop(paths[0]) is not disabled._crop(paths[0])
    assert not disabled.cache


def test_sift_cache_reuses_extraction_without_changing_matches(tmp_path: Path) -> None:
    cfg = load_config().verifier.keypoints.model_copy(update={"backend": "sift"})
    path = tmp_path / "texture.png"
    pixels = np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8)
    assert cv2.imwrite(str(path), pixels)
    matcher = Keypoints(cfg)
    original = matcher.sift
    calls = 0

    def detect(image: Image, mask: None) -> tuple[Any, Any]:
        nonlocal calls
        calls += 1
        return original.detectAndCompute(image, mask)

    matcher.sift = SimpleNamespace(detectAndCompute=detect)
    crop = matcher._crop(path)
    p, q, available = matcher._points(crop, crop)
    repeated = matcher._points(matcher._crop(path), matcher._crop(path))
    assert calls == 1
    assert available and len(p) > 0
    uncached = Keypoints(cfg.model_copy(update={"cache_crops": 0}))
    expected = uncached._points(uncached._crop(path), uncached._crop(path))
    np.testing.assert_array_equal(p, expected[0])
    np.testing.assert_array_equal(q, expected[1])
    np.testing.assert_array_equal(p, repeated[0])
    matcher.close()
    uncached.close()
