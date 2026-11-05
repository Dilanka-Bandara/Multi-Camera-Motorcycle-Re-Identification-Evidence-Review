"""Compare GPU verifier geometry with and without crop-feature reuse."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import torch

from app.config import AppConfig
from app.features.keypoints import Keypoints
from app.schemas import RunResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    run = RunResult.model_validate_json((args.run / "results.json").read_bytes())
    cfg = AppConfig.model_validate(run.config_snapshot)
    tracks = {t.track_id: t.model_copy(deep=True) for t in run.tracks_A + run.tracks_B}
    for track in tracks.values():
        for crop in track.crops:
            crop.path_vehicle = str((args.run / crop.path_vehicle).resolve())
    candidates = [
        c for q in run.queries[: args.queries] for c in q.candidates[: cfg.verifier.review_top_n]
    ]
    if not candidates:
        raise ValueError("No candidates to benchmark")
    args.out.mkdir(parents=True)
    summaries: dict[str, Any] = {}
    geometry: dict[str, list[dict[str, Any]]] = {}
    for label, capacity in (("uncached", 0), ("cached", cfg.verifier.keypoints.cache_crops)):
        root = args.out / label
        root.mkdir()
        options = cfg.verifier.keypoints.model_copy(
            update={"cache_crops": capacity, "backend": "lightglue"}
        )
        torch.manual_seed(cfg.runtime.seed)
        matcher = Keypoints(options, "cuda")
        try:
            if matcher.matcher is None:
                raise RuntimeError(f"Learned backend unavailable: {matcher.note}")
            first = candidates[0]
            matcher.review(
                tracks[first.query_track_id], tracks[first.candidate_track_id], root, "warmup"
            )
            matcher.cache.clear()
            cv2.setRNGSeed(cfg.runtime.seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rows = []
            for c in candidates:
                value = asdict(
                    matcher.review(
                        tracks[c.query_track_id], tracks[c.candidate_track_id], root, c.pair_id
                    )
                )
                value.pop("path")
                rows.append(value)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            summaries[label] = {
                "seconds": seconds,
                "pairs": len(rows),
                "cache_capacity": capacity,
                "cached_crops": len(matcher.cache),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            geometry[label] = rows
            print(json.dumps({label: summaries[label]}), flush=True)
        finally:
            matcher.close()
    previews_match = all(
        (args.out / "uncached" / "keypoints" / f"{c.pair_id}.png").read_bytes()
        == (args.out / "cached" / "keypoints" / f"{c.pair_id}.png").read_bytes()
        for c in candidates
    )
    report = {
        "source_run": str(args.run),
        "device": torch.cuda.get_device_name(),
        "modes": summaries,
        "identical_geometry": geometry["cached"] == geometry["uncached"],
        "speedup": summaries["uncached"]["seconds"] / summaries["cached"]["seconds"],
        "identical_previews": previews_match,
        "geometry": geometry,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["identical_geometry"] or not previews_match:
        raise RuntimeError("Geometry changed; do not deploy this optimization")


if __name__ == "__main__":
    main()
