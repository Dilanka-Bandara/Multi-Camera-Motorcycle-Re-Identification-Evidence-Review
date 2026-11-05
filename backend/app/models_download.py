"""Provision and load-test all inference assets during the Docker build."""

import logging
import os
from pathlib import Path

from app.config import load_config


def main() -> None:
    from lightglue import ALIKED, LightGlue
    from transformers import AutoImageProcessor, AutoModel
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    logging.basicConfig(level=logging.INFO)
    root = Path(os.environ.get("MOTO_MODELS_DIR", "/app/models")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    for name in (cfg.detector.weights, cfg.detector.fallback_weights):
        path = root / name
        downloaded = Path(attempt_download_asset(path))
        if not downloaded.is_file():
            raise FileNotFoundError(f"Build could not provision {name}")
        if downloaded.resolve() != path:
            raise RuntimeError(f"Unexpected detector weight location: {downloaded}")
        YOLO(str(path))
    checkpoint = "facebook/dinov2-small"
    target = root / "dinov2-small"
    AutoImageProcessor.from_pretrained(checkpoint, use_fast=False).save_pretrained(target)
    AutoModel.from_pretrained(checkpoint).save_pretrained(target)
    ALIKED(max_num_keypoints=cfg.verifier.keypoints.max_keypoints).eval()
    LightGlue(features="aliked").eval()
    logging.info("Detector, appearance and geometry assets provisioned for offline runtime")


if __name__ == "__main__":
    main()
