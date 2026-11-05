"""Run the current pipeline stages without starting the web API."""

import argparse
import logging
from pathlib import Path
from uuid import uuid4

from app.config import load_config
from app.pipeline.orchestrator import run_pipeline


def main() -> None:
    """Load a preset and run fixture or explicitly supplied video paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="dev_fixture")
    parser.add_argument("--fixture-dir", type=Path, default=Path("data/fixtures/dev"))
    parser.add_argument("--video-a", type=Path)
    parser.add_argument("--video-b", type=Path)
    parser.add_argument("--gt-a", type=Path)
    parser.add_argument("--gt-b", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fragment-prob", type=float)
    parser.add_argument(
        "--config-dir", type=Path, default=Path(__file__).resolve().parents[1] / "configs"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    overrides = (
        {"groundtruth": {"fragment_prob": args.fragment_prob}}
        if (args.fragment_prob is not None)
        else {}
    )
    config = load_config(args.preset, overrides, args.config_dir)
    inputs = args.data_dir / config.inputs.directory if config.inputs else args.fixture_dir
    videos = {
        "A": args.video_a or inputs / "camera_A.mp4",
        "B": args.video_b or inputs / "camera_B.mp4",
    }
    truth = {camera: inputs / f"gt_{camera}.csv" for camera in ("A", "B")}
    if args.video_a or args.video_b:
        truth = None
    if args.gt_a or args.gt_b:
        if not (args.gt_a and args.gt_b):
            parser.error("Provide both --gt-a and --gt-b")
        truth = {"A": args.gt_a, "B": args.gt_b}
    if truth and not all(path.is_file() for path in truth.values()):
        truth = None
    out = args.out or Path("data/user/runs") / uuid4().hex
    run_pipeline(config, videos, out, truth, args.models_dir)
    logging.info("run_id=%s completed: %s", out.name, out)


if __name__ == "__main__":
    main()
