"""Offline local feature matching and RANSAC geometry on vehicle crops."""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import KeypointConfig
from app.schemas import Track

Image = NDArray[np.uint8]
Points = NDArray[np.float32]


@dataclass
class Geometry:
    inliers: int
    extracted: bool
    backend: str
    note: str | None
    path: str | None


@dataclass
class CachedCrop:
    """Immutable crop pixels and reusable features held in CPU memory."""

    image: Image
    learned: dict[str, Any] | None = None
    sift: tuple[Any, Any] | None = None


class Keypoints:
    def __init__(self, cfg: KeypointConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.cache: OrderedDict[Path, CachedCrop] = OrderedDict()
        self.device = "cpu"
        self.sift = cv2.SIFT_create(nfeatures=cfg.max_keypoints)
        self.extractor: Any = None
        self.matcher: Any = None
        self.note: str | None = None
        if cfg.backend == "lightglue":
            try:
                import torch
                from lightglue import ALIKED, LightGlue

                cache = Path(torch.hub.get_dir()) / "checkpoints"
                names = (
                    "aliked-n16.pth",
                    f"aliked_lightglue_{LightGlue.version.replace('.', '-')}.pth",
                )
                if not all((cache / name).is_file() for name in names):
                    raise FileNotFoundError("local keypoint weights are unavailable")
                self.device = (
                    ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
                )
                self.extractor = ALIKED(max_num_keypoints=cfg.max_keypoints).eval().to(self.device)
                self.matcher = LightGlue(features="aliked").eval().to(self.device)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self.close()
                self.note = f"SIFT fallback ({type(exc).__name__})"

    def close(self) -> None:
        """Release learned models before the next pipeline job."""
        self.extractor = None
        self.matcher = None
        self.cache.clear()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def _crop(self, path: Path) -> CachedCrop:
        """Read each immutable run crop once while it remains in the bounded cache."""
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError("Cannot read selected vehicle crop")
        crop = CachedCrop(pixels)
        if self.cfg.cache_crops:
            self.cache[path] = crop
            if len(self.cache) > self.cfg.cache_crops:
                self.cache.popitem(last=False)
        return crop

    def _learned(self, crop: CachedCrop) -> dict[str, Any]:
        """Reuse ALIKED output without retaining a growing CUDA cache."""
        import torch

        if crop.learned is not None:
            return {k: v.to(self.device) for k, v in crop.learned.items()}
        features = self.extractor.extract(
            torch.from_numpy(crop.image).float().to(self.device)[None] / 255
        )
        if self.cfg.cache_crops:
            crop.learned = {k: v.detach().cpu() for k, v in features.items()}
        return features

    def _sift(self, crop: CachedCrop) -> tuple[Any, Any]:
        if crop.sift is None:
            features = self.sift.detectAndCompute(crop.image, None)
            if self.cfg.cache_crops:
                crop.sift = features
            return features
        return crop.sift

    def _points(self, a: CachedCrop, b: CachedCrop) -> tuple[Points, Points, bool]:
        if self.matcher is not None:
            import torch

            try:
                with torch.inference_mode():
                    x, y = (self._learned(v) for v in (a, b))
                    pairs = self.matcher({"image0": x, "image1": y})["matches"][0]
                    return (
                        x["keypoints"][0, pairs[:, 0]].cpu().numpy(),
                        y["keypoints"][0, pairs[:, 1]].cpu().numpy(),
                        bool(x["keypoints"].shape[1] and y["keypoints"].shape[1]),
                    )
            except (RuntimeError, ValueError, IndexError) as exc:
                self.close()
                self.note = f"SIFT fallback during inference ({type(exc).__name__})"
        ka, da = self._sift(a)
        kb, db = self._sift(b)
        empty = np.empty((0, 2), dtype=np.float32)
        if da is None or db is None:
            return empty, empty, False
        pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < self.cfg.ratio * pair[1].distance
        ]
        return (
            np.float32([ka[m.queryIdx].pt for m in good]),
            np.float32([kb[m.trainIdx].pt for m in good]),
            True,
        )

    def review(self, left: Track, right: Track, root: Path, pair_id: str) -> Geometry:
        pairs = sorted(
            product(left.crops, right.crops), key=lambda pair: -pair[0].quality * pair[1].quality
        )
        best = -1
        extracted = False
        output = Path("keypoints") / f"{pair_id}.png"
        backend = "sift"
        for x, y in pairs[: self.cfg.pairs_to_check]:
            crop_a, crop_b = (self._crop(root / c.path_vehicle) for c in (x, y))
            p, q, available = self._points(crop_a, crop_b)
            a, b = crop_a.image, crop_b.image
            extracted |= available
            mask = np.zeros(len(p), dtype=bool)
            if len(p) >= 8:  # Eight-point fundamental matrix.
                matrix, inliers = cv2.findFundamentalMat(
                    p, q, cv2.FM_RANSAC, self.cfg.ransac_px, self.cfg.ransac_confidence
                )
                if matrix is not None and inliers is not None:
                    mask = inliers.ravel().astype(bool)
            count = int(mask.sum())
            if count <= best:
                continue
            best = count
            backend = "lightglue" if self.matcher is not None else "sift"
            canvas = np.zeros((max(a.shape[0], b.shape[0]), a.shape[1] + b.shape[1]), np.uint8)
            canvas[: a.shape[0], : a.shape[1]] = a
            canvas[: b.shape[0], a.shape[1] :] = b
            rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            for u, v in zip(p[mask], q[mask], strict=True):
                cv2.line(
                    rgb, tuple(u.astype(int)), (int(v[0]) + a.shape[1], int(v[1])), (0, 255, 0)
                )
            (root / output).parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(root / output), rgb):
                raise OSError("Could not save geometry preview")
        return Geometry(
            max(best, 0), extracted, backend, self.note, output.as_posix() if best >= 0 else None
        )
