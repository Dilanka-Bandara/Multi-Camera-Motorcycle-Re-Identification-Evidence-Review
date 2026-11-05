"""Offline local feature matching and RANSAC geometry on vehicle crops."""

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


class Keypoints:
    def __init__(self, cfg: KeypointConfig) -> None:
        self.cfg = cfg
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
                self.extractor = ALIKED(max_num_keypoints=cfg.max_keypoints).eval()
                self.matcher = LightGlue(features="aliked").eval()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self.note = f"SIFT fallback ({type(exc).__name__})"

    def _points(self, a: Image, b: Image) -> tuple[Points, Points, bool]:
        if self.matcher is not None:
            import torch

            try:
                with torch.inference_mode():
                    x, y = (
                        self.extractor.extract(torch.from_numpy(v).float()[None] / 255)
                        for v in (a, b)
                    )
                    pairs = self.matcher({"image0": x, "image1": y})["matches"][0]
                    return (
                        x["keypoints"][0, pairs[:, 0]].numpy(),
                        y["keypoints"][0, pairs[:, 1]].numpy(),
                        bool(x["keypoints"].shape[1] and y["keypoints"].shape[1]),
                    )
            except (RuntimeError, ValueError, IndexError) as exc:
                self.matcher = None
                self.note = f"SIFT fallback during inference ({type(exc).__name__})"
        ka, da = self.sift.detectAndCompute(a, None)
        kb, db = self.sift.detectAndCompute(b, None)
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
            a, b = (cv2.imread(str(root / c.path_vehicle), cv2.IMREAD_GRAYSCALE) for c in (x, y))
            if a is None or b is None:
                raise ValueError("Cannot read selected vehicle crop")
            p, q, available = self._points(a, b)
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
