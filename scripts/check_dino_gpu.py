"""Measure real fixture DINOv2 peak allocated GPU memory under Python 3.11."""

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import load_config
from app.features.embedder import DinoEmbedder
from app.schemas import Track


def main() -> None:
    """Embed every selected fixture crop and record the Phase 3 memory check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/dinov2-small"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run acceptance on Python 3.11")
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Phase 3 GPU acceptance cannot pass")
    cfg = load_config("dev_fixture")
    tracks = [
        Track.model_validate(track)
        for camera in ("A", "B")
        for track in json.loads((args.run / f"tracks_{camera}.json").read_text())
    ]
    paths = [
        args.run / path
        for track in tracks
        for crop in track.crops
        for path in (crop.path_full, crop.path_vehicle, crop.path_rider)
        if path is not None
    ]
    torch.manual_seed(cfg.runtime.seed)
    embedder = DinoEmbedder(args.model, cfg.embedding.batch_size, "cuda")
    try:
        vectors = embedder.embed(paths)
        peak = embedder.peak_memory_bytes()
        result = {
            "python": sys.version,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "tracks": len(tracks),
            "crops": len(paths),
            "dimension": vectors.shape[1],
            "batch_size": cfg.embedding.batch_size,
            "peak_allocated_bytes": peak,
            "below_3_gib": peak < 3 * 1024**3,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        logging.basicConfig(level=logging.INFO)
        logging.info("run_id=%s GPU acceptance: %s", args.run.name, json.dumps(result))
        if not result["below_3_gib"]:
            raise RuntimeError("DINO peak allocated memory exceeds the Phase 3 limit")
    finally:
        embedder.close()


if __name__ == "__main__":
    main()
